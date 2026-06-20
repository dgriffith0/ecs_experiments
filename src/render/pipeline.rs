//! Render-pipeline construction shared by the voxel, light, and glTF passes.

use encase::ShaderType;
use glam::{Mat4, Vec4};
use wgpu::util::DeviceExt;

use glam::Vec3;

use crate::ecs::resources::{LineOverlay, NavOverlay};
use crate::render::texture;
use crate::scene::nav::NavMesh;

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

/// The 24 line-list endpoints of an AABB's wireframe, in world space.
pub fn box_edges(min: Vec3, max: Vec3) -> [[f32; 3]; 24] {
    let size = max - min;
    CUBE_EDGES.map(|v| {
        [
            min.x + v[0] * size.x,
            min.y + v[1] * size.y,
            min.z + v[2] * size.z,
        ]
    })
}

/// Build a reusable world-space line overlay: a `LineList` pipeline (drawn on top,
/// ignoring depth) reusing the selection shader + uniform, with a `COPY_DST` vertex
/// buffer sized for `capacity_boxes` wireframe boxes, re-filled each frame.
pub fn create_line_overlay(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    capacity_boxes: u32,
) -> LineOverlay {
    let capacity = capacity_boxes * CUBE_EDGES.len() as u32;
    let lines = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("line_overlay_lines"),
        size: (capacity as u64) * 12,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let initial = SelUniform {
        mvp: Mat4::IDENTITY,
        color: Vec4::ONE,
    };
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("line_overlay_uniform"),
        contents: &crate::util::uniform_bytes(&initial),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("line_overlay_layout"),
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

    LineOverlay {
        pipeline,
        uniform,
        bind_group,
        lines,
        capacity,
        num_vertices: 0,
        visible: false,
    }
}

/// GPU vertex buffer of the nav-mesh overlay's link line endpoints. Returns the
/// buffer and its vertex count (0 → nothing to draw). Falls back to a 1-vertex
/// dummy buffer when there are no links so `create_buffer_init` never sees 0 bytes.
pub fn nav_lines_buffer(device: &wgpu::Device, nav: &NavMesh) -> (wgpu::Buffer, u32) {
    let verts = nav.segments();
    if verts.is_empty() {
        let lines = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("nav_lines"),
            contents: bytemuck::cast_slice(&[[0.0f32; 3]]),
            usage: wgpu::BufferUsages::VERTEX,
        });
        return (lines, 0);
    }
    let lines = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("nav_lines"),
        contents: bytemuck::cast_slice(verts),
        usage: wgpu::BufferUsages::VERTEX,
    });
    (lines, verts.len() as u32)
}

/// Build the nav-mesh debug overlay: a line-list pipeline (drawn on top, ignoring
/// depth) reusing the selection shader + uniform, fed by the link line buffer.
pub fn create_nav_overlay(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    nav: &NavMesh,
) -> NavOverlay {
    let (lines, num_vertices) = nav_lines_buffer(device, nav);

    let initial = SelUniform {
        mvp: Mat4::IDENTITY,
        color: Vec4::new(0.2, 1.0, 0.4, 1.0),
    };
    let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("nav_uniform"),
        contents: &crate::util::uniform_bytes(&initial),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("nav_layout"),
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
        label: Some("nav_bind_group"),
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nav_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("shaders/selection.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("nav_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("nav_pipeline"),
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
        // Drawn on top of the scene: depth attachment present, never tested/written.
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

    NavOverlay {
        pipeline,
        uniform,
        bind_group,
        lines,
        num_vertices,
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
