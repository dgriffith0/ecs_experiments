//! CPU mouse picking: unproject the cursor to a world ray, test object AABBs and
//! ray-march the terrain heightmap, and store the nearest hit in `Selected`.

use bevy_ecs::prelude::*;
use glam::{Vec3, Vec4, Vec4Swizzles};

use crate::ecs::components::{Camera, NavAgent, Pawn, Pickable, Transform};
use crate::ecs::resources::{CursorPos, Selected, ViewProj};
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

/// Cast a ray from the cursor and set `Selected` to the nearest entity (or clear it).
pub fn pick_at(world: &mut World) {
    let Some((origin, dir)) = cursor_ray(world) else {
        return;
    };

    // Nearest hit across pickable objects (fox, light) and the terrain voxel.
    let mut best_t = f32::INFINITY;
    let mut best = Selected::None;

    {
        let mut q = world.query::<(Entity, &Transform, &Pickable)>();
        for (entity, transform, pickable) in q.iter(world) {
            if let Some(t) = pickable
                .local_aabb
                .transformed(&transform.matrix())
                .ray_intersect(origin, dir)
                && t < best_t
            {
                best_t = t;
                best = Selected::Object(entity);
            }
        }
    }

    // Terrain: ray-march the heightmap; nudge inside the surface to land in the
    // solid voxel, then snap to that voxel cell.
    {
        let heightmap = world.resource::<Heightmap>();
        if let Some((t, hit)) = raymarch_terrain(origin, dir, heightmap)
            && t < best_t
        {
            let (coord, min, max) = terrain::voxel_cell_at(hit + dir * 0.02, heightmap.grid());
            best = Selected::Voxel { coord, min, max };
        }
    }

    *world.resource_mut::<Selected>() = best;
}

/// If a [`Pawn`] is selected, ray-cast the cursor onto the terrain and order that
/// pawn to walk there: A* a path from its current cell to the clicked cell and
/// store it on its [`NavAgent`]. No-op if nothing/​non-pawn is selected or the
/// cursor isn't over the terrain.
pub fn command_pawn(world: &mut World) {
    let Selected::Object(pawn) = *world.resource::<Selected>() else {
        return;
    };
    if world.get::<Pawn>(pawn).is_none() {
        return;
    }
    let Some((origin, dir)) = cursor_ray(world) else {
        return;
    };

    // Target cell under the cursor.
    let goal = {
        let heightmap = world.resource::<Heightmap>();
        let Some((_, hit)) = raymarch_terrain(origin, dir, heightmap) else {
            return;
        };
        heightmap.cell_coords(hit.x, hit.z)
    };
    // Pawn's current cell.
    let Some(pos) = world.get::<Transform>(pawn).map(|t| t.translation) else {
        return;
    };
    let start = world.resource::<Heightmap>().cell_coords(pos.x, pos.z);

    // Plan the route and hand it to the pawn.
    let path = {
        let nav = world.resource::<NavMesh>();
        let heightmap = world.resource::<Heightmap>();
        nav.find_path(heightmap, start, goal)
    };
    if let Some(mut path) = path
        && let Some(mut agent) = world.get_mut::<NavAgent>(pawn)
    {
        path.reverse(); // pop() yields the next waypoint
        path.pop(); // drop the start cell
        agent.path = path;
    }
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
