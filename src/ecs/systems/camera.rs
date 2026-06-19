use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::ecs::components::{Camera, CameraGpu, FlyController, Transform};
use crate::ecs::resources::{Input, ViewProj};
use crate::render::context::RenderContext;
use crate::scene::camera::{self, CameraUniform};
use crate::util::uniform_bytes;

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
