use encase::ShaderType;
use glam::Vec3;

/// Mirrors the `Light` uniform in the shaders. encase derives the std140 layout,
/// so each `vec3` is aligned/padded to 16 bytes for us — no manual padding fields.
#[derive(Debug, Copy, Clone, ShaderType)]
pub struct LightUniform {
    pub position: Vec3,
    pub color: Vec3,
}
