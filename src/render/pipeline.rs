//! Render-pipeline construction shared by the voxel, light, and glTF passes.

use encase::ShaderType;
use glam::{Mat4, Vec4};
use wgpu::util::DeviceExt;

use crate::ecs::resources::SelectionBox;
use crate::render::texture;

/// Uniform for the selection-box shader: clip-space transform + line color.
#[derive(Copy, Clone, ShaderType)]
pub struct SelUniform {
    pub mvp: Mat4,
    pub color: Vec4,
}

/// The 24 endpoints (12 edges) of the unit cube `[0,1]³`, as a line list.
#[rustfmt::skip]
const CUBE_EDGES: [[f32; 3]; 24] = [
    // bottom square
    [0.,0.,0.],[1.,0.,0.], [1.,0.,0.],[1.,0.,1.], [1.,0.,1.],[0.,0.,1.], [0.,0.,1.],[0.,0.,0.],
    // top square
    [0.,1.,0.],[1.,1.,0.], [1.,1.,0.],[1.,1.,1.], [1.,1.,1.],[0.,1.,1.], [0.,1.,1.],[0.,1.,0.],
    // verticals
    [0.,0.,0.],[0.,1.,0.], [1.,0.,0.],[1.,1.,0.], [1.,0.,1.],[1.,1.,1.], [0.,0.,1.],[0.,1.,1.],
];

/// Build the wireframe selection-box resources (pipeline, geometry, uniform).
pub fn create_selection_box(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
) -> SelectionBox {
    let edges = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("selection_edges"),
        contents: bytemuck::cast_slice(&CUBE_EDGES),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let initial = SelUniform {
        mvp: Mat4::IDENTITY,
        color: Vec4::new(1.0, 0.9, 0.2, 1.0),
    };
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("selection_uniform"),
        contents: &crate::util::uniform_bytes(&initial),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("selection_layout"),
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
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("selection_bind_group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("selection_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("selection_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("selection_pipeline"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: 12,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![0 => Float32x3],
            }],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            ..Default::default()
        },
        // Draw on top of the scene: a depth attachment is present, but never test/write.
        depth_stencil: Some(wgpu::DepthStencilState {
            format: texture::Texture::DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    });

    SelectionBox {
        pipeline,
        edges,
        uniform,
        bind_group,
        visible: false,
    }
}

/// Build a standard opaque render pipeline: triangle list, CCW front faces with
/// back-face culling, depth test + write (`Less`), and `REPLACE` blending.
pub fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Render Pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: vertex_layouts,
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            polygon_mode: wgpu::PolygonMode::Fill,
            unclipped_depth: false,
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview_mask: None,
        cache: None,
    })
}
