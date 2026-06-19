use std::sync::Arc;

use bevy_ecs::prelude::{Schedule, World};
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use crate::ecs::components::{AnimationPlayer, SkinnedMesh};
use crate::ecs::resources::{CursorPos, Input, VoxelSettingsRes};
use crate::ecs::systems::regenerate_terrain;
use crate::ecs::world::{build_schedule, build_world};
use crate::picking;
use crate::render::context::RenderContext;
use crate::render::draw::resize;
use crate::ui::{self, Ui};

/// Logical width of the Slint side panel; clicks left of this go to the UI.
const UI_PANEL_WIDTH: f32 = 260.0;

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

        // Forward pointer events to the Slint UI (it ignores non-pointer events).
        {
            let mut slint_ui = world.non_send_resource_mut::<Ui>();
            ui::forward_event(&mut slint_ui, &event);
        }

        // The title-screen Exit button quits the app.
        if world
            .non_send_resource::<Ui>()
            .component
            .get_exit_requested()
        {
            event_loop.exit();
            return;
        }

        // The terrain generator's Regenerate button (and entering the screen)
        // rebuilds the world from the slider values.
        if world
            .non_send_resource::<Ui>()
            .component
            .get_regenerate_requested()
        {
            world
                .non_send_resource::<Ui>()
                .component
                .set_regenerate_requested(false);
            regenerate_terrain(world);
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => resize(world, size.width, size.height),
            WindowEvent::RedrawRequested => self.schedule.run(world),
            WindowEvent::CursorMoved { position, .. } => {
                let mut cursor = world.resource_mut::<CursorPos>();
                cursor.0 = position.x as f32;
                cursor.1 = position.y as f32;
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // Pick only in-game, and only when the click is in the viewport
                // (not over the HUD panel).
                let (scale, in_game) = {
                    let ui = world.non_send_resource::<Ui>();
                    (ui.scale, ui.component.get_in_game())
                };
                let logical_x = world.resource::<CursorPos>().0 / scale;
                if in_game && logical_x > UI_PANEL_WIDTH {
                    picking::pick_at(world);
                }
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
    } else if pressed && matches!(code, KeyCode::Digit1 | KeyCode::Digit2 | KeyCode::Digit3) {
        // Switch the animation clip (Survey / Walk / Run) on every player.
        let clip = match code {
            KeyCode::Digit1 => 0,
            KeyCode::Digit2 => 1,
            _ => 2,
        };
        let mut q = world.query::<(&mut AnimationPlayer, &SkinnedMesh)>();
        for (mut player, skin) in q.iter_mut(world) {
            if let Some(c) = skin.clips.get(clip) {
                player.clip = clip;
                player.time = 0.0;
                println!("playing animation: {}", c.name);
            }
        }
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
