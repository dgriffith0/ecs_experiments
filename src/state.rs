use std::sync::Arc;

use glam::{Quat, Vec3};
use wgpu::util::DeviceExt;
use winit::{event_loop::ActiveEventLoop, keyboard::KeyCode, window::Window};

use crate::camera::CameraSystem;
use crate::gpu::GpuContext;
use crate::light::LightUniform;
use crate::model::Vertex;
use crate::voxel::{self, VoxelChunk, VoxelSettings};
use crate::{model, resources, texture, utils};

/// Side length (in chunks) of the flat voxel chunk grid.
const CHUNK_GRID_SIZE: u32 = 4;

// This will store the state of our game
pub struct State {
    ctx: GpuContext,
    window_background_color: wgpu::Color,
    camera: CameraSystem,
    depth_texture: texture::Texture,
    obj_model: model::Model,
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
    voxel_render_pipeline: wgpu::RenderPipeline,
    voxel_texture_bind_group: wgpu::BindGroup,
    voxel_chunks: Vec<VoxelChunk>,
    voxel_settings: VoxelSettings,
    voxel_settings_buffer: wgpu::Buffer,
    voxel_settings_bind_group: wgpu::BindGroup,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let ctx = GpuContext::new(window).await?;

        let texture_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            // This should match the filterable field of the
                            // corresponding Texture entry above.
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                    label: Some("texture_bind_group_layout"),
                });

        let obj_model = resources::load_model(
            "cube.obj",
            &ctx.device,
            &ctx.queue,
            &texture_bind_group_layout,
        )
        .await
        .unwrap();

        let window_background_color = wgpu::Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 1.0,
        };

        let camera = CameraSystem::new(&ctx.device, &ctx.config, 0.2);

        let light_uniform = LightUniform {
            position: glam::vec3(2.0, 2.0, 2.0),
            color: glam::vec3(1.0, 1.0, 1.0),
        };

        // We'll want to update our lights position, so we use COPY_DST
        let light_buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Light VB"),
                contents: &utils::uniform_bytes(&light_uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let light_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
                    label: None,
                });

        let light_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&ctx.device, &ctx.config, "depth_texture");

        let voxel_texture = resources::load_texture_array(
            "array_texture.png",
            voxel::NUM_TEXTURE_LAYERS,
            &ctx.device,
            &ctx.queue,
        )
        .await?;

        let voxel_texture_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("voxel_texture_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2Array,
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

        let voxel_texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &voxel_texture_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&voxel_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&voxel_texture.sampler),
                },
            ],
            label: Some("voxel_texture_bind_group"),
        });

        // Ambient occlusion starts enabled; toggle at runtime with the O key.
        let voxel_settings = VoxelSettings::new(true);
        let voxel_settings_buffer =
            ctx.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Voxel Settings Buffer"),
                    contents: &utils::uniform_bytes(&voxel_settings),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let voxel_settings_bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("voxel_settings_bind_group_layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });

        let voxel_settings_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &voxel_settings_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: voxel_settings_buffer.as_entire_binding(),
            }],
            label: Some("voxel_settings_bind_group"),
        });

        let voxel_render_pipeline = {
            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Voxel Pipeline Layout"),
                    bind_group_layouts: &[
                        Some(&voxel_texture_bind_group_layout),
                        Some(camera.bind_group_layout()),
                        Some(&light_bind_group_layout),
                        Some(&voxel_settings_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Voxel Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("voxel.wgsl").into()),
            };
            create_render_pipeline(
                &ctx.device,
                &layout,
                ctx.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[voxel::VoxelVertex::desc()],
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = ctx
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Light Pipeline Layout"),
                    bind_group_layouts: &[
                        Some(camera.bind_group_layout()),
                        Some(&light_bind_group_layout),
                    ],
                    immediate_size: 0,
                });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &ctx.device,
                &layout,
                ctx.config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
            )
        };

        let voxel_chunks = voxel::generate_chunk_grid(&ctx.device, CHUNK_GRID_SIZE);

        Ok(Self {
            ctx,
            window_background_color,
            camera,
            depth_texture,
            obj_model,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            voxel_render_pipeline,
            voxel_texture_bind_group,
            voxel_chunks,
            voxel_settings,
            voxel_settings_buffer,
            voxel_settings_bind_group,
        })
    }

    pub fn window(&self) -> &Window {
        &self.ctx.window
    }

    pub fn set_background_color(&mut self, color: wgpu::Color) {
        self.window_background_color = color;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.ctx.resize(width, height);
            // The depth texture must match the surface size, so recreate it
            // whenever we reconfigure — otherwise the render pass attachments
            // mismatch and wgpu panics.
            self.depth_texture = texture::Texture::create_depth_texture(
                &self.ctx.device,
                &self.ctx.config,
                "depth_texture",
            );
            self.camera.resize(width, height);
        }
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        if code == KeyCode::Escape && is_pressed {
            event_loop.exit();
        } else if code == KeyCode::KeyO && is_pressed {
            // Toggle ambient occlusion and push the new flag to the GPU.
            self.voxel_settings.toggle();
            self.ctx.queue.write_buffer(
                &self.voxel_settings_buffer,
                0,
                &utils::uniform_bytes(&self.voxel_settings),
            );
        } else {
            self.camera.process_key(code, is_pressed);
        }
    }

    pub fn update(&mut self) {
        self.camera.update(&self.ctx.queue);

        // Update the light: orbit it 1° around the +y axis each frame.
        let rotation = Quat::from_axis_angle(Vec3::Y, 1f32.to_radians());
        self.light_uniform.position = rotation * self.light_uniform.position;
        self.ctx.queue.write_buffer(
            &self.light_buffer,
            0,
            &utils::uniform_bytes(&self.light_uniform),
        );
    }

    pub fn render(&mut self) -> anyhow::Result<()> {
        self.ctx.window.request_redraw();

        // We can't render unless the surface is configured
        if !self.ctx.is_surface_configured {
            return Ok(());
        }

        let output = match self.ctx.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(surface_texture) => surface_texture,
            wgpu::CurrentSurfaceTexture::Suboptimal(surface_texture) => {
                self.ctx.reconfigure();
                surface_texture
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => {
                // Skip this frame
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.ctx.reconfigure();
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                // You could recreate the devices and all resources
                // created with it here, but we'll just bail
                anyhow::bail!("Lost device");
            }
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.window_background_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),

                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            // Draw the orbiting light marker.
            use crate::model::DrawLight;
            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(
                &self.obj_model,
                self.camera.bind_group(),
                &self.light_bind_group,
            );

            // Draw the voxel chunks. The texture array (group 0), camera
            // (group 1), and light (group 2) are shared across every chunk;
            // each chunk only swaps its own vertex/index buffers.
            render_pass.set_pipeline(&self.voxel_render_pipeline);
            render_pass.set_bind_group(0, &self.voxel_texture_bind_group, &[]);
            render_pass.set_bind_group(1, self.camera.bind_group(), &[]);
            render_pass.set_bind_group(2, &self.light_bind_group, &[]);
            render_pass.set_bind_group(3, &self.voxel_settings_bind_group, &[]);
            for chunk in &self.voxel_chunks {
                render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
                render_pass
                    .set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
            }
        }

        // submit will accept anything that implements IntoIter
        self.ctx.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

fn create_render_pipeline(
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
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLIP_CONTROL
            unclipped_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
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
