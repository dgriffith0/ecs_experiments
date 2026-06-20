//! ECS resources — singletons shared across systems.
//!
//! The window/surface-bearing [`crate::render::context::RenderContext`] is stored
//! separately as a *non-send* resource (see `setup`); everything here is plain
//! `Send + Sync` GPU state or small per-frame data.

use bevy_ecs::prelude::Resource;
use glam::Mat4;
use winit::keyboard::KeyCode;

use crate::render::texture;
use crate::scene::{model, skybox, terrain};

/// The three render pipelines, built once at startup and shared every frame.
#[derive(Resource)]
pub struct Pipelines {
    pub voxel: wgpu::RenderPipeline,
    pub light: wgpu::RenderPipeline,
    pub gltf: wgpu::RenderPipeline,
}

/// Depth buffer; recreated on resize to match the surface size.
#[derive(Resource)]
pub struct DepthTexture(pub texture::Texture);

/// Shared GPU state for the voxel pass: the texture-array bind group plus the
/// runtime settings (AO toggle) buffer + bind group.
#[derive(Resource)]
pub struct VoxelGpu {
    pub texture_bind_group: wgpu::BindGroup,
    pub settings_buffer: wgpu::Buffer,
    pub settings_bind_group: wgpu::BindGroup,
}

/// CPU-side voxel render settings (AO on/off). Uploaded to `VoxelGpu` on change.
#[derive(Resource)]
pub struct VoxelSettingsRes(pub terrain::VoxelSettings);

/// The cubemap sky (owns its own pipeline, uniform buffer, and bind groups).
#[derive(Resource)]
pub struct SkyboxRes(pub skybox::Skybox);

/// The cube mesh drawn as the orbiting light's marker.
#[derive(Resource)]
pub struct LightMarker(pub model::Model);

/// Clear color, driven by the cursor position.
#[derive(Resource)]
pub struct BackgroundColor(pub wgpu::Color);

/// The camera's combined view-projection for the current frame, computed once
/// and read by both the camera and skybox uploads.
#[derive(Resource)]
pub struct ViewProj(pub Mat4);

/// Per-frame timing. `delta` is the wall-clock seconds since the last frame,
/// used to advance animations at real speed.
#[derive(Resource, Default)]
pub struct Time {
    pub delta: f32,
    pub last: Option<std::time::Instant>,
}

/// The currently selected pawns (set by picking / box-select).
#[derive(Resource, Default)]
pub struct Selection(pub Vec<bevy_ecs::entity::Entity>);

/// Last physical cursor position, used by picking.
#[derive(Resource, Default, Clone, Copy)]
pub struct CursorPos(pub f32, pub f32);

/// A reusable world-space line overlay drawn on top of the scene: a `COPY_DST`
/// vertex buffer (capacity `capacity` line vertices) re-filled each frame, plus a
/// `SelUniform` (MVP + colour). Used for selection boxes and destination markers.
pub struct LineOverlay {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub lines: wgpu::Buffer,
    pub capacity: u32,
    pub num_vertices: u32,
    pub visible: bool,
}

/// Yellow boxes around the selected pawns.
#[derive(Resource)]
pub struct SelectionOverlay(pub LineOverlay);

/// Green boxes on each moving pawn's destination cell.
#[derive(Resource)]
pub struct DestinationOverlay(pub LineOverlay);

/// Debug overlay for the navigation mesh: a line list of walkable cell-to-cell
/// links, drawn on top of the scene (toggle with `N`). Rebuilt with the terrain.
#[derive(Resource)]
pub struct NavOverlay {
    pub pipeline: wgpu::RenderPipeline,
    pub uniform: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub lines: wgpu::Buffer,
    pub num_vertices: u32,
    pub visible: bool,
}

/// Held-key state for the free-fly camera, fed by winit keyboard events.
#[derive(Resource, Default)]
pub struct Input {
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub up: bool,
    pub down: bool,
    pub look_left: bool,
    pub look_right: bool,
    pub look_up: bool,
    pub look_down: bool,
}

impl Input {
    /// Map a key to its movement/look flag. Returns `true` if the key was a
    /// camera control (so the caller knows it was consumed).
    pub fn set(&mut self, code: KeyCode, pressed: bool) -> bool {
        match code {
            KeyCode::KeyW => self.forward = pressed,
            KeyCode::KeyS => self.backward = pressed,
            KeyCode::KeyA => self.left = pressed,
            KeyCode::KeyD => self.right = pressed,
            KeyCode::Space => self.up = pressed,
            KeyCode::ShiftLeft | KeyCode::ShiftRight => self.down = pressed,
            KeyCode::ArrowLeft => self.look_left = pressed,
            KeyCode::ArrowRight => self.look_right = pressed,
            KeyCode::ArrowUp => self.look_up = pressed,
            KeyCode::ArrowDown => self.look_down = pressed,
            _ => return false,
        }
        true
    }
}
