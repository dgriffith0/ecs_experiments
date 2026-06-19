use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::ecs::components::{LightGpu, PointLight, Transform};
use crate::render::context::RenderContext;
use crate::scene::light::LightUniform;
use crate::util::uniform_bytes;

/// Orbit every point light 1° around the world +Y axis each frame.
pub fn orbit_light(mut q: Query<&mut Transform, With<PointLight>>) {
    let rotation = Quat::from_axis_angle(Vec3::Y, 1f32.to_radians());
    for mut transform in &mut q {
        transform.translation = rotation * transform.translation;
    }
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
