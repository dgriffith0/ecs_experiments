//! Minimal glTF (.glb) model: a single textured triangle mesh rendered in its
//! static bind pose. We read only positions and UVs (the Fox sample carries no
//! normals); the shader reconstructs flat normals from screen-space derivatives.
//! Skinning and animation are ignored.

/// A vertex for a loaded glTF mesh: position plus base-color UVs.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GltfVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
}

impl GltfVertex {
    const ATTRS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x2, // tex_coords
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

/// One loaded glTF model: its geometry, base-color texture, and world transform.
/// Many of these share a single pipeline; each carries its own bind groups so
/// they can be drawn in a loop.
pub struct GltfModel {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
    /// Texture + sampler bound at `@group(0)` in `gltf.wgsl`.
    pub texture_bind_group: wgpu::BindGroup,
    /// Per-model transform matrix uniform bound at `@group(3)`.
    pub model_bind_group: wgpu::BindGroup,
}
