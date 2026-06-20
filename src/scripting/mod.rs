//! Luau scripting layer (via `mlua`). Scripts run against a small `game` API and
//! drive the ECS indirectly: API calls record [`ScriptCmd`]s into a buffer, which
//! the [`run_scripts`] exclusive system drains and applies to the world after the
//! script returns. This keeps Lua callbacks free of any `&mut World` borrow.
//!
//! The host calls the script's global `on_update(dt)` each frame. The VM is a
//! non-send resource (Lua is `Rc`-based, main-thread only), like the renderer.

use std::cell::RefCell;
use std::rc::Rc;

use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use mlua::Lua;

use crate::assets::AssetRegistry;
use crate::ecs::components::{AnimationPlayer, NavAgent, Pawn, Pickable, Placed, Transform};
use crate::ecs::resources::Time;
use crate::render::context::RenderContext;
use crate::scene::terrain::Heightmap;

/// An action a script requested, applied to the world after the script runs.
enum ScriptCmd {
    Log(String),
    Spawn { asset: String, x: f32, z: f32 },
}

type CmdBuffer = Rc<RefCell<Vec<ScriptCmd>>>;

/// The Luau virtual machine plus the command buffer its `game` API writes into.
/// Stored as a non-send resource.
pub struct ScriptVm {
    lua: Lua,
    commands: CmdBuffer,
}

/// Load the Luau scripts and wire up the `game` API. Returns the ready VM.
pub async fn load_scripts() -> anyhow::Result<ScriptVm> {
    let source = crate::assets::load_string("scripts/main.luau").await?;
    // `mlua::Error` is `!Send`/`!Sync` (non-send Lua), so it can't flow through
    // `?` into `anyhow` — stringify it at the boundary instead.
    build_vm(&source).map_err(|e| anyhow::anyhow!("scripting: {e}"))
}

/// Build the VM, register the `game` API, and run the top-level script.
fn build_vm(source: &str) -> mlua::Result<ScriptVm> {
    let lua = Lua::new();
    let commands: CmdBuffer = Rc::new(RefCell::new(Vec::new()));

    let game = lua.create_table()?;
    // game.log(msg)
    let buf = commands.clone();
    game.set(
        "log",
        lua.create_function(move |_, msg: String| {
            buf.borrow_mut().push(ScriptCmd::Log(msg));
            Ok(())
        })?,
    )?;
    // game.spawn(asset_name, x, z) — places a registered asset on the surface.
    let buf = commands.clone();
    game.set(
        "spawn",
        lua.create_function(move |_, (asset, x, z): (String, f32, f32)| {
            buf.borrow_mut().push(ScriptCmd::Spawn { asset, x, z });
            Ok(())
        })?,
    )?;
    lua.globals().set("game", game)?;

    lua.load(source).set_name("main.luau").exec()?;
    Ok(ScriptVm { lua, commands })
}

/// Per-frame: call each script's `on_update(dt)`, then apply the commands it
/// queued. Exclusive so it can mutate the world when applying `spawn`.
pub fn run_scripts(world: &mut World) {
    let dt = world.resource::<Time>().delta;

    // Run the script and take ownership of whatever it queued, releasing the VM
    // borrow before we touch the world mutably.
    let commands: Vec<ScriptCmd> = {
        let vm = world.non_send_resource::<ScriptVm>();
        if let Ok(on_update) = vm.lua.globals().get::<mlua::Function>("on_update")
            && let Err(e) = on_update.call::<()>(dt)
        {
            log::error!("on_update: {e}");
        }
        vm.commands.borrow_mut().drain(..).collect()
    };

    for cmd in commands {
        match cmd {
            ScriptCmd::Log(msg) => log::info!("[luau] {msg}"),
            ScriptCmd::Spawn { asset, x, z } => spawn_asset_at(world, &asset, x, z),
        }
    }
}

/// Spawn a registered asset on the terrain surface at `(x, z)`. Builds the bundle
/// while borrowing the registry/heightmap immutably, then spawns — so we never
/// hold a resource borrow across the mutable `world.spawn`.
fn spawn_asset_at(world: &mut World, name: &str, x: f32, z: f32) {
    let device = world.non_send_resource::<RenderContext>().device.clone();
    let built = {
        let registry = world.resource::<AssetRegistry>();
        let heightmap = world.resource::<Heightmap>();
        registry.get(name).map(|asset| {
            let pos = Vec3::new(x, heightmap.surface_y(x, z), z);
            let transform = Transform {
                translation: pos,
                rotation: Quat::IDENTITY,
                scale: Vec3::splat(asset.scale),
            };
            let skin = asset
                .template
                .skin
                .clone()
                .map(|s| (s, asset.clip.unwrap_or(0)));
            (
                asset.template.instantiate(&device),
                transform,
                Pickable {
                    local_aabb: asset.template.local_aabb,
                },
                skin,
                asset.is_pawn,
            )
        })
    };

    if let Some((model, transform, pickable, skin, is_pawn)) = built {
        let mut entity = world.spawn((model, transform, pickable, Placed));
        if let Some((skin, clip)) = skin {
            entity.insert((
                skin,
                AnimationPlayer {
                    clip,
                    time: 0.0,
                    speed: 1.0,
                },
            ));
        }
        if is_pawn {
            entity.insert((
                Pawn,
                NavAgent {
                    path: Vec::new(),
                    speed: 2.0,
                },
            ));
        }
    }
}
