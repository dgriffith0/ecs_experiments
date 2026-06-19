//! ECS components — per-entity data.
//!
//! Spatial entities (the camera, the light, every chunk, every glTF model) carry
//! a [`Transform`]; the camera and light additionally carry their GPU mirror
//! (`CameraGpu` / `LightGpu`) so the render system can fetch their bind groups by
//! query. The mesh components ([`crate::voxel::VoxelChunk`],
//! [`crate::gltf_model::GltfModel`]) derive `Component` in their own modules.

use bevy_ecs::prelude::Component;
use glam::{Mat4, Quat, Vec3};

/// Position / orientation / scale of an entity in world space.
#[derive(Component, Clone, Copy)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            translation,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// Marks the camera entity and holds its projection parameters. The eye position
/// lives in the entity's [`Transform`]; the look direction in [`FlyController`].
#[derive(Component)]
pub struct Camera {
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
    pub aspect: f32,
}

/// Free-fly look/move state for the camera entity (yaw/pitch + tuning).
#[derive(Component)]
pub struct FlyController {
    pub yaw: f32,
    pub pitch: f32,
    pub speed: f32,
    pub rotate_speed: f32,
}

/// Marks a point-light entity. Its position is the entity's [`Transform`].
#[derive(Component)]
pub struct PointLight {
    pub color: Vec3,
}

/// GPU mirror for the single camera entity: its uniform buffer + bind group.
#[derive(Component)]
pub struct CameraGpu {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

/// GPU mirror for a light entity: its uniform buffer + bind group.
#[derive(Component)]
pub struct LightGpu {
    pub buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}
