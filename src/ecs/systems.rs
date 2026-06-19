//! Per-frame systems: simulate (fly camera, orbit light), then upload GPU
//! buffers. The render system lives in `crate::render::draw`.

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::camera::{self, CameraUniform};
use crate::ecs::components::{Camera, CameraGpu, FlyController, LightGpu, PointLight, Transform};
use crate::ecs::resources::{Input, SkyboxRes, ViewProj, VoxelGpu, VoxelSettingsRes};
use crate::gltf_model::GltfModel;
use crate::light::LightUniform;
use crate::render::context::RenderContext;
use crate::utils::uniform_bytes;

// --- Sim ---

/// Apply held input to the camera entity: arrow keys rotate yaw/pitch, WASD +
/// Space/Shift move along the look/strafe axes (strafe stays horizontal).
pub fn fly_camera(
    input: Res<Input>,
    mut q: Query<(&mut Transform, &mut FlyController), With<Camera>>,
) {
    let Ok((mut transform, mut fly)) = q.single_mut() else {
        return;
    };

    if input.look_left {
        fly.yaw -= fly.rotate_speed;
    }
    if input.look_right {
        fly.yaw += fly.rotate_speed;
    }
    if input.look_up {
        fly.pitch += fly.rotate_speed;
    }
    if input.look_down {
        fly.pitch -= fly.rotate_speed;
    }
    fly.pitch = fly.pitch.clamp(-camera::MAX_PITCH, camera::MAX_PITCH);

    let forward = camera::forward(fly.yaw, fly.pitch);
    let right = forward.cross(Vec3::Y).normalize_or_zero();
    let speed = fly.speed;

    if input.forward {
        transform.translation += forward * speed;
    }
    if input.backward {
        transform.translation -= forward * speed;
    }
    if input.right {
        transform.translation += right * speed;
    }
    if input.left {
        transform.translation -= right * speed;
    }
    if input.up {
        transform.translation += Vec3::Y * speed;
    }
    if input.down {
        transform.translation -= Vec3::Y * speed;
    }
}

/// Orbit every point light 1° around the world +Y axis each frame.
pub fn orbit_light(mut q: Query<&mut Transform, With<PointLight>>) {
    let rotation = Quat::from_axis_angle(Vec3::Y, 1f32.to_radians());
    for mut transform in &mut q {
        transform.translation = rotation * transform.translation;
    }
}

// --- Upload ---

/// Recompute the camera's view-projection from its transform + controller.
pub fn update_view_proj(
    mut view_proj: ResMut<ViewProj>,
    q: Query<(&Transform, &FlyController, &Camera)>,
) {
    let Ok((transform, fly, cam)) = q.single() else {
        return;
    };
    view_proj.0 = camera::view_projection(
        transform.translation,
        fly.yaw,
        fly.pitch,
        cam.fovy,
        cam.aspect,
        cam.znear,
        cam.zfar,
    );
}

/// Write the camera uniform (view position + view-projection).
pub fn upload_camera(
    ctx: NonSend<RenderContext>,
    view_proj: Res<ViewProj>,
    q: Query<(&Transform, &CameraGpu)>,
) {
    let Ok((transform, gpu)) = q.single() else {
        return;
    };
    let uniform = CameraUniform {
        view_position: transform.translation.extend(1.0),
        view_proj: view_proj.0,
    };
    ctx.queue
        .write_buffer(&gpu.buffer, 0, &uniform_bytes(&uniform));
}

/// Write each light's uniform (position from its transform + color).
pub fn upload_light(ctx: NonSend<RenderContext>, q: Query<(&Transform, &PointLight, &LightGpu)>) {
    for (transform, light, gpu) in &q {
        let uniform = LightUniform {
            position: transform.translation,
            color: light.color,
        };
        ctx.queue
            .write_buffer(&gpu.buffer, 0, &uniform_bytes(&uniform));
    }
}

/// Keep the sky oriented with the camera (it samples by view direction).
pub fn upload_skybox(
    ctx: NonSend<RenderContext>,
    view_proj: Res<ViewProj>,
    skybox: Res<SkyboxRes>,
) {
    skybox.0.update(&ctx.queue, view_proj.0);
}

/// Write each glTF model's transform matrix to its uniform, only when the
/// entity's `Transform` changes (newly-spawned entities count as changed).
pub fn upload_model_transforms(
    ctx: NonSend<RenderContext>,
    q: Query<(&Transform, &GltfModel), Changed<Transform>>,
) {
    for (transform, model) in &q {
        ctx.queue
            .write_buffer(&model.model_buffer, 0, &uniform_bytes(&transform.matrix()));
    }
}

/// Re-upload the voxel settings (AO flag) only when they change.
pub fn upload_voxel_settings(
    ctx: NonSend<RenderContext>,
    settings: Res<VoxelSettingsRes>,
    gpu: Res<VoxelGpu>,
) {
    if settings.is_changed() {
        ctx.queue
            .write_buffer(&gpu.settings_buffer, 0, &uniform_bytes(&settings.0));
    }
}
