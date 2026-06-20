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
use crate::ecs::resources::{CursorPos, Input, NavOverlay, VoxelSettingsRes};
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
    /// Whether Shift is held (additive selection).
    shift: bool,
    /// Physical cursor position where a left-drag began (for box-select).
    drag_start: Option<(f32, f32)>,
}

impl App {
    pub fn new() -> Self {
        Self {
            world: None,
            schedule: build_schedule(),
            shift: false,
            drag_start: None,
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
            WindowEvent::ModifiersChanged(mods) => {
                self.shift = mods.state().shift_key();
            }
            WindowEvent::CursorMoved { position, .. } => {
                {
                    let mut cursor = world.resource_mut::<CursorPos>();
                    cursor.0 = position.x as f32;
                    cursor.1 = position.y as f32;
                }
                // While dragging, draw the rubber-band rectangle (logical px).
                if let Some(start) = self.drag_start {
                    let scale = world.non_send_resource::<Ui>().scale;
                    let (x0, x1) = (
                        start.0.min(position.x as f32),
                        start.0.max(position.x as f32),
                    );
                    let (y0, y1) = (
                        start.1.min(position.y as f32),
                        start.1.max(position.y as f32),
                    );
                    let c = &world.non_send_resource::<Ui>().component;
                    c.set_drag_x(x0 / scale);
                    c.set_drag_y(y0 / scale);
                    c.set_drag_width((x1 - x0) / scale);
                    c.set_drag_height((y1 - y0) / scale);
                    c.set_drag_active(true);
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => {
                // Begin a left-drag in the viewport (resolved to click vs box on release).
                let (scale, in_game) = {
                    let ui = world.non_send_resource::<Ui>();
                    (ui.scale, ui.component.get_in_game())
                };
                let cursor = *world.resource::<CursorPos>();
                if in_game && cursor.0 / scale > UI_PANEL_WIDTH {
                    self.drag_start = Some((cursor.0, cursor.1));
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => {
                if let Some(start) = self.drag_start.take() {
                    world
                        .non_send_resource::<Ui>()
                        .component
                        .set_drag_active(false);
                    let cursor = *world.resource::<CursorPos>();
                    let moved = (cursor.0 - start.0).hypot(cursor.1 - start.1);
                    if moved < 6.0 {
                        picking::pick_at(world, self.shift); // a click
                    } else {
                        let min = (start.0.min(cursor.0), start.1.min(cursor.1));
                        let max = (start.0.max(cursor.0), start.1.max(cursor.1));
                        picking::box_select(world, min, max, self.shift);
                    }
                }
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Right,
                ..
            } => {
                // Right-click: order the selected pawns to walk to the clicked voxel.
                let (scale, in_game) = {
                    let ui = world.non_send_resource::<Ui>();
                    (ui.scale, ui.component.get_in_game())
                };
                if in_game && world.resource::<CursorPos>().0 / scale > UI_PANEL_WIDTH {
                    picking::command_pawns(world);
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
    } else if code == KeyCode::KeyN && pressed {
        // Toggle the navigation-mesh debug overlay.
        let mut nav = world.resource_mut::<NavOverlay>();
        nav.visible = !nav.visible;
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
