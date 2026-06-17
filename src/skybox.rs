//! Cubemap skybox.
//!
//! Renders a single fullscreen triangle at the far plane and samples a cubemap
//! by the per-pixel view direction. Drawn *after* the scene with depth-test
//! `LessEqual` and no depth writes, so it only fills pixels the terrain didn't
//! cover (the cleared far-plane depth of 1.0). It never touches the shared
//! camera uniform — instead it keeps its own inverse view-projection matrix.

use encase::ShaderType;
use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::{resources, texture, utils};

/// Uniform mirrored by `skybox.wgsl`: the inverse of the camera's full
/// view-projection, used to unproject clip-space pixels back into world rays.
#[derive(Copy, Clone, ShaderType)]
struct SkyUniform {
    inv_view_proj: Mat4,
}

pub struct Skybox {
    cube_bind_group: wgpu::BindGroup,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
}

impl Skybox {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
    ) -> anyhow::Result<Self> {
        // Faces in wgpu cube-layer order: +X, -X, +Y, -Y, +Z, -Z.
        let cubemap = resources::load_cubemap(
            [
                "skybox/right.png",
                "skybox/left.png",
                "skybox/top.png",
                "skybox/bottom.png",
                "skybox/front.png",
                "skybox/back.png",
            ],
            device,
            queue,
        )
        .await?;

        let cube_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skybox_cube_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            view_dimension: wgpu::TextureViewDimension::Cube,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let cube_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skybox_cube_bind_group"),
            layout: &cube_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&cubemap.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&cubemap.sampler),
                },
            ],
        });

        let uniform = SkyUniform {
            inv_view_proj: Mat4::IDENTITY,
        };
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("skybox_uniform_buffer"),
            contents: &utils::uniform_bytes(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("skybox_uniform_bind_group_layout"),
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

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("skybox_uniform_bind_group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Skybox Pipeline Layout"),
            bind_group_layouts: &[
                Some(&cube_bind_group_layout),
                Some(&uniform_bind_group_layout),
            ],
            immediate_size: 0,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Skybox Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("skybox.wgsl").into()),
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Skybox Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                // The fullscreen triangle has no meaningful winding to cull.
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: texture::Texture::DEPTH_FORMAT,
                // Don't write depth; only draw where nothing closer was drawn.
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
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

        Ok(Self {
            cube_bind_group,
            uniform_buffer,
            uniform_bind_group,
            pipeline,
        })
    }

    /// Push the current inverse view-projection so the sky tracks the camera.
    pub fn update(&self, queue: &wgpu::Queue, view_proj: Mat4) {
        let uniform = SkyUniform {
            inv_view_proj: view_proj.inverse(),
        };
        queue.write_buffer(&self.uniform_buffer, 0, &utils::uniform_bytes(&uniform));
    }

    /// Draw the sky. Call after the scene, inside the same render pass.
    pub fn draw(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.cube_bind_group, &[]);
        render_pass.set_bind_group(1, &self.uniform_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}
