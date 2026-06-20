//! CPU mouse control of pawns: unproject the cursor to a world ray to select pawns
//! (click / shift-click / drag-box) and order the selection to walk to a voxel.

use std::collections::{HashSet, VecDeque};

use bevy_ecs::prelude::*;
use glam::{Vec3, Vec4, Vec4Swizzles};

use crate::ecs::components::{Camera, NavAgent, Pawn, Pickable, Transform};
use crate::ecs::resources::{CursorPos, Selection, ViewProj};
use crate::render::context::RenderContext;
use crate::scene::nav::NavMesh;
use crate::scene::terrain::{self, Heightmap};

/// Unproject the cursor into a world-space `(origin, direction)` ray, or `None` if
/// there's no surface / camera yet.
fn cursor_ray(world: &mut World) -> Option<(Vec3, Vec3)> {
    let cursor = *world.resource::<CursorPos>();
    let (sw, sh) = {
        let ctx = world.non_send_resource::<RenderContext>();
        (ctx.config.width as f32, ctx.config.height as f32)
    };
    if sw <= 0.0 || sh <= 0.0 {
        return None;
    }
    let view_proj = world.resource::<ViewProj>().0;
    let eye = {
        let mut q = world.query_filtered::<&Transform, With<Camera>>();
        q.iter(world).next()?.translation
    };
    // Same near/far convention as skybox.wgsl.
    let ndc_x = 2.0 * (cursor.0 / sw) - 1.0;
    let ndc_y = 1.0 - 2.0 * (cursor.1 / sh);
    let inv = view_proj.inverse();
    let unproject = |z: f32| {
        let p = inv * Vec4::new(ndc_x, ndc_y, z, 1.0);
        p.xyz() / p.w
    };
    Some((eye, (unproject(1.0) - unproject(0.0)).normalize()))
}

/// Left-click: select the pawn under the cursor. `additive` (shift) toggles it in
/// the selection; otherwise it replaces the selection. Clicking empty ground (no
/// pawn) clears the selection unless `additive`.
pub fn pick_at(world: &mut World, additive: bool) {
    let Some((origin, dir)) = cursor_ray(world) else {
        return;
    };

    // Nearest pawn under the cursor (pawns only — foxes etc. aren't selectable).
    let mut best_t = f32::INFINITY;
    let mut hit: Option<Entity> = None;
    let mut q = world.query_filtered::<(Entity, &Transform, &Pickable), With<Pawn>>();
    for (entity, transform, pickable) in q.iter(world) {
        if let Some(t) = pickable
            .local_aabb
            .transformed(&transform.matrix())
            .ray_intersect(origin, dir)
            && t < best_t
        {
            best_t = t;
            hit = Some(entity);
        }
    }

    let mut selection = world.resource_mut::<Selection>();
    match hit {
        Some(e) if additive => match selection.0.iter().position(|&x| x == e) {
            Some(i) => {
                selection.0.remove(i);
            }
            None => selection.0.push(e),
        },
        Some(e) => {
            selection.0.clear();
            selection.0.push(e);
        }
        None if !additive => selection.0.clear(),
        None => {}
    }
}

/// Drag-box select: every pawn whose screen position falls inside the rectangle
/// (physical pixels). `additive` (shift) adds to the selection instead of replacing.
pub fn box_select(world: &mut World, min: (f32, f32), max: (f32, f32), additive: bool) {
    let view_proj = world.resource::<ViewProj>().0;
    let (sw, sh) = {
        let ctx = world.non_send_resource::<RenderContext>();
        (ctx.config.width as f32, ctx.config.height as f32)
    };
    if sw <= 0.0 || sh <= 0.0 {
        return;
    }

    let mut inside: Vec<Entity> = Vec::new();
    let mut q = world.query_filtered::<(Entity, &Transform), With<Pawn>>();
    for (entity, transform) in q.iter(world) {
        let clip = view_proj * transform.translation.extend(1.0);
        if clip.w <= 0.0 {
            continue; // behind the camera
        }
        let ndc = clip.xyz() / clip.w;
        let sx = (ndc.x * 0.5 + 0.5) * sw;
        let sy = (1.0 - (ndc.y * 0.5 + 0.5)) * sh;
        if sx >= min.0 && sx <= max.0 && sy >= min.1 && sy <= max.1 {
            inside.push(entity);
        }
    }

    let mut selection = world.resource_mut::<Selection>();
    if !additive {
        selection.0.clear();
    }
    for e in inside {
        if !selection.0.contains(&e) {
            selection.0.push(e);
        }
    }
}

/// Right-click: order every selected pawn to walk to the clicked voxel. Each pawn
/// gets its **own** distinct destination cell in a compact formation around the
/// target (no stacking), then A*-paths there. No-op without a selection / terrain hit.
pub fn command_pawns(world: &mut World) {
    // Live selected pawns (clone the list so we don't hold the resource borrow).
    let pawns: Vec<Entity> = world
        .resource::<Selection>()
        .0
        .clone()
        .into_iter()
        .filter(|&e| world.get::<Pawn>(e).is_some())
        .collect();
    if pawns.is_empty() {
        return;
    }

    let Some((origin, dir)) = cursor_ray(world) else {
        return;
    };
    let goal = {
        let heightmap = world.resource::<Heightmap>();
        let Some((_, hit)) = raymarch_terrain(origin, dir, heightmap) else {
            return;
        };
        heightmap.cell_coords(hit.x, hit.z)
    };

    // One walkable destination cell per pawn, closest-to-target first.
    let cells = {
        let nav = world.resource::<NavMesh>();
        formation_cells(nav, goal, pawns.len())
    };

    for (i, &pawn) in pawns.iter().enumerate() {
        let Some(&dest) = cells.get(i) else { break };
        let Some(pos) = world.get::<Transform>(pawn).map(|t| t.translation) else {
            continue;
        };
        let start = world.resource::<Heightmap>().cell_coords(pos.x, pos.z);
        let path = {
            let nav = world.resource::<NavMesh>();
            let heightmap = world.resource::<Heightmap>();
            nav.find_path(heightmap, start, dest)
        };
        if let Some(mut path) = path
            && let Some(mut agent) = world.get_mut::<NavAgent>(pawn)
        {
            path.reverse(); // pop() yields the next waypoint
            path.pop(); // drop the start cell
            agent.path = path;
        }
    }
}

/// Collect up to `n` distinct walkable cells, breadth-first outward from `goal`
/// (so closest cells fill first) — a compact formation around the target.
fn formation_cells(nav: &NavMesh, goal: (usize, usize), n: usize) -> Vec<(usize, usize)> {
    let extent = nav.extent() as i32;
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();
    seen.insert(goal);
    queue.push_back(goal);
    while let Some((x, z)) = queue.pop_front() {
        if nav.is_walkable(x as i64, z as i64) {
            out.push((x, z));
            if out.len() >= n {
                break;
            }
        }
        for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
            let (nx, nz) = (x as i32 + dx, z as i32 + dz);
            if nx < 0 || nz < 0 || nx >= extent || nz >= extent {
                continue;
            }
            let cell = (nx as usize, nz as usize);
            if seen.insert(cell) {
                queue.push_back(cell);
            }
        }
    }
    out
}

/// March the ray against the terrain heightmap (only where chunks exist). Returns
/// `(distance, hit_point)` of the first surface crossing. The march is clamped to
/// the t-interval where the ray is inside the terrain's vertical slab, so a
/// sky-bound ray (which never re-enters the slab) costs almost nothing.
fn raymarch_terrain(origin: Vec3, dir: Vec3, heightmap: &Heightmap) -> Option<(f32, Vec3)> {
    const MAX_T: f32 = 500.0;
    const STEP: f32 = 0.25;

    let (y_min, y_max) = terrain::terrain_y_bounds();
    let (t_enter, t_exit) = if dir.y.abs() < 1e-6 {
        // Nearly horizontal: only relevant if the ray is already in the slab.
        if origin.y < y_min || origin.y > y_max {
            return None;
        }
        (0.0, MAX_T)
    } else {
        let (ta, tb) = ((y_min - origin.y) / dir.y, (y_max - origin.y) / dir.y);
        (ta.min(tb).max(0.0), ta.max(tb).min(MAX_T))
    };
    if t_enter > t_exit {
        return None;
    }

    let surface_at = |x: f32, z: f32| -> f32 {
        match terrain::chunk_coord_at(x, z, heightmap.grid()) {
            Some(_) => heightmap.surface_y(x, z),
            None => f32::NEG_INFINITY, // no terrain outside the rendered grid
        }
    };

    let mut t = t_enter;
    while t < t_exit {
        let p = origin + dir * t;
        if p.y <= surface_at(p.x, p.z) {
            // Crossed between t-STEP and t; binary-refine the surface point.
            let (mut lo, mut hi) = ((t - STEP).max(t_enter), t);
            for _ in 0..16 {
                let mid = 0.5 * (lo + hi);
                let pm = origin + dir * mid;
                if pm.y > surface_at(pm.x, pm.z) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some((hi, origin + dir * hi));
        }
        t += STEP;
    }
    None
}
