use encase::ShaderType;
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;
use winit::keyboard::KeyCode;

use crate::camera_controller::CameraController;
use crate::utils::uniform_bytes;

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: Mat4 = Mat4::from_cols_array(&[
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.0, 0.0, 0.5, 1.0,
]);

/// A free-fly camera defined by a position and an orientation (yaw + pitch),
/// rather than being locked to a focal point. This lets it move and look
/// independently.
pub struct Camera {
    pub position: Vec3,
    /// Rotation around the world +Y axis, in radians. 0 looks toward +X.
    pub yaw: f32,
    /// Up/down rotation, in radians. Positive looks up.
    pub pitch: f32,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    /// Unit vector the camera is looking along, derived from yaw and pitch.
    pub fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(cos_pitch * cos_yaw, sin_pitch, cos_pitch * sin_yaw).normalize()
    }

    pub fn build_view_projection_matrix(&self) -> Mat4 {
        // Look *along* a direction instead of *at* a target point.
        let view = Mat4::look_to_rh(self.position, self.forward(), Vec3::Y);
        // `fovy` is kept in degrees for the public API; glam wants radians.
        // We use the GL-style projection and correct its [-1, 1] depth range to
        // wgpu's [0, 1] with `OPENGL_TO_WGPU_MATRIX`.
        let proj =
            Mat4::perspective_rh_gl(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

/// Mirrors the `Camera` uniform in the shaders. encase derives the std140
/// layout from the glam types, so the buffer matches WGSL's `vec4`/`mat4x4`.
#[derive(Copy, Clone, ShaderType)]
pub struct CameraUniform {
    view_position: Vec4,
    view_proj: Mat4,
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            view_position: Vec4::ZERO,
            view_proj: Mat4::IDENTITY,
        }
    }

    pub fn update_view_proj(&mut self, camera: &Camera) {
        // We're using a vec4 because of the uniform's 16 byte spacing requirement
        self.view_position = camera.position.extend(1.0);
        // `build_view_projection_matrix` already folds in OPENGL_TO_WGPU_MATRIX.
        self.view_proj = camera.build_view_projection_matrix();
    }
}

/// Bundles the camera, its input controller, and the GPU resources that mirror
/// the camera into a uniform buffer the shaders read each frame.
pub struct CameraSystem {
    camera: Camera,
    controller: CameraController,
    uniform: CameraUniform,
    buffer: wgpu::Buffer,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl CameraSystem {
    pub fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration, speed: f32) -> Self {
        let camera = Camera {
            // 1 unit up and 2 units back from the origin...
            position: Vec3::new(0.0, 1.0, 2.0),
            // ...looking toward -Z (yaw = -90°) and level with the horizon.
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut uniform = CameraUniform::new();
        uniform.update_view_proj(&camera);

        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: &uniform_bytes(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
            label: Some("camera_bind_group_layout"),
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        Self {
            camera,
            controller: CameraController::new(speed),
            uniform,
            buffer,
            bind_group_layout,
            bind_group,
        }
    }

    /// Advance the camera from input and push the new view/projection to the GPU.
    pub fn update(&mut self, queue: &wgpu::Queue) {
        self.controller.update_camera(&mut self.camera);
        self.uniform.update_view_proj(&self.camera);
        queue.write_buffer(&self.buffer, 0, &uniform_bytes(&self.uniform));
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        // Keep the projection in sync so the scene doesn't stretch.
        self.camera.aspect = width as f32 / height as f32;
    }

    pub fn process_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        self.controller.handle_key(code, is_pressed)
    }

    pub fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    /// The current combined view-projection matrix (already includes the
    /// GL→wgpu depth correction and the camera translation).
    pub fn view_proj(&self) -> Mat4 {
        self.uniform.view_proj
    }

    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }
}
