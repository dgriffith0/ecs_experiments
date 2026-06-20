//! Slint UI embedded as a guest: rendered to a CPU buffer by Slint's software
//! renderer, uploaded to a texture, and composited as a premultiplied-alpha
//! overlay at the end of our wgpu render pass. No wgpu types cross the Slint
//! boundary, so it stays decoupled from our wgpu version.

use std::rc::Rc;
use std::time::Instant;

use bevy_ecs::prelude::*;
use slint::platform::software_renderer::{
    MinimalSoftwareWindow, PremultipliedRgbaColor, RepaintBufferType,
};
use slint::platform::{Platform, PointerEventButton, WindowAdapter, WindowEvent};
use slint::{ComponentHandle, LogicalPosition, PhysicalSize};

use crate::ecs::components::{AnimationPlayer, Camera, Transform};
use crate::ecs::resources::{Selection, Time, VoxelSettingsRes};
use crate::render::context::RenderContext;
use crate::render::texture;
use crate::scene::terrain::VoxelSettings;

slint::include_modules!(); // generates `AppWindow` from ui/app.slint

/// Debug toggle: when `true`, skip the title screen and start in-game.
pub const SKIP_TITLE_SCREEN: bool = false;

/// Minimal Slint platform: hands back our software window, no event loop (we
/// drive rendering manually each frame).
struct UiPlatform {
    window: Rc<MinimalSoftwareWindow>,
    start: Instant,
}

impl Platform for UiPlatform {
    fn create_window_adapter(&self) -> Result<Rc<dyn WindowAdapter>, slint::PlatformError> {
        Ok(self.window.clone())
    }
    fn duration_since_start(&self) -> core::time::Duration {
        self.start.elapsed()
    }
}

/// The live UI (non-send: Slint is `Rc`-based and stays on the main thread).
pub struct Ui {
    pub adapter: Rc<MinimalSoftwareWindow>,
    pub component: AppWindow,
    pub scale: f32,
    pub last_cursor: LogicalPosition,
}

/// GPU + CPU resources for compositing the UI texture over the scene.
#[derive(Resource)]
pub struct UiOverlay {
    pub texture: wgpu::Texture,
    pub bind_group: wgpu::BindGroup,
    pub pipeline: wgpu::RenderPipeline,
    pub buffer: Vec<PremultipliedRgbaColor>,
    pub width: u32,
    pub height: u32,
}

/// Create the Slint platform + UI component for a surface of the given size.
pub fn create_ui(width: u32, height: u32, scale: f32) -> Ui {
    let adapter = MinimalSoftwareWindow::new(RepaintBufferType::ReusedBuffer);
    slint::platform::set_platform(Box::new(UiPlatform {
        window: adapter.clone(),
        start: Instant::now(),
    }))
    .expect("slint platform set once");

    let component = AppWindow::new().expect("create AppWindow");
    component.show().expect("show AppWindow");

    adapter.dispatch_event(WindowEvent::ScaleFactorChanged {
        scale_factor: scale,
    });
    adapter.set_size(PhysicalSize::new(width, height));

    Ui {
        adapter,
        component,
        scale,
        last_cursor: LogicalPosition::new(0.0, 0.0),
    }
}

/// Build the overlay texture/pipeline/CPU buffer at the given surface size.
pub fn create_overlay(
    device: &wgpu::Device,
    color_format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) -> UiOverlay {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("ui_overlay_texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        ..Default::default()
    });

    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ui_overlay_layout"),
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
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("ui_overlay_bind_group"),
        layout: &layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ui_overlay_shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("../render/shaders/ui.wgsl").into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ui_overlay_pipeline_layout"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ui_overlay_pipeline"),
        layout: Some(&pipeline_layout),
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
                format: color_format,
                // Premultiplied "over": result = src + dst * (1 - src.a).
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha: wgpu::BlendComponent {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        // Pass has a depth attachment, so declare one but never write/test it.
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

    UiOverlay {
        texture,
        bind_group,
        pipeline,
        buffer: vec![PremultipliedRgbaColor::default(); (width * height) as usize],
        width,
        height,
    }
}

/// Render the UI into the CPU buffer (only if it changed) and upload it.
pub fn render_ui(ctx: NonSend<RenderContext>, ui: NonSend<Ui>, mut overlay: ResMut<UiOverlay>) {
    slint::platform::update_timers_and_animations();
    let stride = overlay.width as usize;
    let drawn = ui.adapter.draw_if_needed(|renderer| {
        renderer.render(&mut overlay.buffer, stride);
    });
    if drawn {
        let (w, h) = (overlay.width, overlay.height);
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &overlay.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&overlay.buffer),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * w),
                rows_per_image: Some(h),
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Sync UI ↔ ECS each frame: push control values into the world and pull display
/// values back into the UI.
#[allow(clippy::too_many_arguments)]
pub fn sync_ui(
    ui: NonSend<Ui>,
    time: Res<Time>,
    selection: Res<Selection>,
    mut settings: ResMut<VoxelSettingsRes>,
    mut players: Query<&mut AnimationPlayer>,
    camera: Query<&Transform, With<Camera>>,
) {
    let c = &ui.component;

    // UI → ECS
    let ao = c.get_ao_enabled();
    if (settings.0.ao_enabled != 0) != ao {
        settings.0 = VoxelSettings::new(ao);
    }
    let clip = c.get_animation_clip().max(0) as usize;
    for mut player in &mut players {
        if player.clip != clip {
            player.clip = clip;
            player.time = 0.0;
        }
    }

    // ECS → UI
    c.set_fps(if time.delta > 0.0 {
        1.0 / time.delta
    } else {
        0.0
    });
    if let Ok(t) = camera.single() {
        c.set_camera_pos(
            format!(
                "{:.0}, {:.0}, {:.0}",
                t.translation.x, t.translation.y, t.translation.z
            )
            .into(),
        );
    }

    let label = match selection.0.len() {
        0 => "none".to_string(),
        1 => "1 pawn".to_string(),
        n => format!("{n} pawns"),
    };
    c.set_selected(label.into());
}

/// Forward a winit pointer event to Slint (in logical coordinates). Keyboard is
/// intentionally not forwarded — the UI has no text fields, and the camera owns
/// the keyboard. Returns nothing; UI consumes what it's over.
pub fn forward_event(ui: &mut Ui, event: &winit::event::WindowEvent) {
    use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent as WE};

    let scale = ui.scale;
    let to_logical = |x: f64, y: f64| LogicalPosition::new(x as f32 / scale, y as f32 / scale);
    let map_button = |b: &MouseButton| match b {
        MouseButton::Left => PointerEventButton::Left,
        MouseButton::Right => PointerEventButton::Right,
        MouseButton::Middle => PointerEventButton::Middle,
        _ => PointerEventButton::Other,
    };

    let slint_event = match event {
        WE::CursorMoved { position, .. } => {
            ui.last_cursor = to_logical(position.x, position.y);
            WindowEvent::PointerMoved {
                position: ui.last_cursor,
            }
        }
        WE::MouseInput { state, button, .. } => {
            let position = ui.last_cursor;
            let button = map_button(button);
            match state {
                ElementState::Pressed => WindowEvent::PointerPressed { position, button },
                ElementState::Released => WindowEvent::PointerReleased { position, button },
            }
        }
        WE::MouseWheel { delta, .. } => {
            let (dx, dy) = match delta {
                MouseScrollDelta::LineDelta(x, y) => (x * 20.0, y * 20.0),
                MouseScrollDelta::PixelDelta(p) => (p.x as f32, p.y as f32),
            };
            WindowEvent::PointerScrolled {
                position: ui.last_cursor,
                delta_x: dx,
                delta_y: dy,
            }
        }
        WE::CursorLeft { .. } => WindowEvent::PointerExited,
        _ => return,
    };
    ui.adapter.window().dispatch_event(slint_event);
}
