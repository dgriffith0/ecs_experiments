use std::sync::Arc;

use bevy_ecs::prelude::{Schedule, World};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::ecs::resources::{BackgroundColor, Input, VoxelSettingsRes};
use crate::ecs::setup::{build_schedule, build_world};
use crate::render::context::RenderContext;
use crate::render::draw::resize;
use crate::utils::ColorFromXY;

/// The winit runner. Owns the ECS `World` and the per-frame `Schedule`, and
/// translates window events into resource/world mutations.
pub struct App {
    world: Option<World>,
    schedule: Schedule,
}

impl App {
    pub fn new() -> Self {
        Self {
            world: None,
            schedule: build_schedule(),
        }
    }
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_maximized(true);
        let window = Arc::new(event_loop.create_window(attrs).unwrap());

        let world = pollster::block_on(build_world(window)).unwrap();
        // Kick off the first frame; the render system re-arms the redraw each
        // frame. Auto-maximized macOS windows get no initial RedrawRequested.
        world
            .non_send_resource::<RenderContext>()
            .window
            .request_redraw();
        self.world = Some(world);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let Some(world) = self.world.as_mut() else {
            return;
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => resize(world, size.width, size.height),
            WindowEvent::RedrawRequested => self.schedule.run(world),
            WindowEvent::CursorMoved { position, .. } => {
                let size = world
                    .non_send_resource::<RenderContext>()
                    .window
                    .inner_size();
                world.resource_mut::<BackgroundColor>().0 = wgpu::Color::from_xy(
                    position.x,
                    (0.0, size.width as f64),
                    position.y,
                    (0.0, size.height as f64),
                );
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state,
                        ..
                    },
                ..
            } => handle_key(world, event_loop, code, state.is_pressed()),
            _ => {}
        }
    }
}

fn handle_key(world: &mut World, event_loop: &ActiveEventLoop, code: KeyCode, pressed: bool) {
    if code == KeyCode::Escape && pressed {
        event_loop.exit();
    } else if code == KeyCode::KeyO && pressed {
        // Toggle ambient occlusion; `upload_voxel_settings` re-uploads on change.
        world.resource_mut::<VoxelSettingsRes>().0.toggle();
    } else {
        world.resource_mut::<Input>().set(code, pressed);
    }
}

pub fn run() -> anyhow::Result<()> {
    env_logger::init();

    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;

    Ok(())
}
