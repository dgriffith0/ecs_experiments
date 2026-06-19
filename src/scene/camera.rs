//! Camera math: free-fly view/projection from an eye position + yaw/pitch, and
//! the `CameraUniform` GPU mirror. The camera itself is an ECS entity
//! (`Camera` + `Transform` + `FlyController` + `CameraGpu`); this module just
//! holds the pure functions and the uniform layout.

use encase::ShaderType;
use glam::{Mat4, Vec3, Vec4};

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
]);

/// Pitch is clamped to just under 90° so looking up/down never flips the view.
pub const MAX_PITCH: f32 = 1.54; // ~88°

/// Unit look direction from yaw (around +Y, 0 looks toward +X) and pitch.
pub fn forward(yaw: f32, pitch: f32) -> Vec3 {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize()
}

/// Combined view-projection matrix (GL depth corrected to wgpu's [0, 1]).
pub fn view_projection(
    eye: Vec3,
    yaw: f32,
    pitch: f32,
    fovy: f32,
    aspect: f32,
    znear: f32,
    zfar: f32,
) -> Mat4 {
    let view = Mat4::look_to_rh(eye, forward(yaw, pitch), Vec3::Y);
    let proj = Mat4::perspective_rh_gl(fovy.to_radians(), aspect, znear, zfar);
    OPENGL_TO_WGPU_MATRIX * proj * view
}

/// Mirrors the `Camera` uniform in the shaders (std140 via encase).
#[derive(Copy, Clone, ShaderType)]
pub struct CameraUniform {
    pub view_position: Vec4,
    pub view_proj: Mat4,
}
