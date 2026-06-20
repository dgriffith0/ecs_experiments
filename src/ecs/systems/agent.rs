use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use rand::Rng;

use crate::ecs::components::{AnimationPlayer, NavAgent, Pawn, SkinnedMesh, Transform, Wander};
use crate::ecs::resources::Time;
use crate::scene::nav::NavMesh;
use crate::scene::terrain::Heightmap;

/// Give idle wanderers (the foxes) a fresh route: when a `Wander` agent's path is
/// empty, A* to a random walkable cell. Pawns lack `Wander`, so they sit still
/// until a player command fills their path.
pub fn wander(
    nav: Res<NavMesh>,
    heightmap: Res<Heightmap>,
    mut agents: Query<(&Transform, &mut NavAgent), With<Wander>>,
) {
    let cells = nav.cells();
    if cells.is_empty() {
        return;
    }
    let mut rng = rand::rng();
    for (transform, mut agent) in &mut agents {
        if !agent.path.is_empty() {
            continue;
        }
        let start = heightmap.cell_coords(transform.translation.x, transform.translation.z);
        let goal = cells[rng.random_range(0..cells.len())];
        if let Some(mut path) = nav.find_path(&heightmap, start, goal) {
            path.reverse(); // pop() now yields the next waypoint
            path.pop(); // drop the start cell
            agent.path = path;
        }
    }
}

/// Advance every nav agent (foxes and pawns) along its queued path: move toward the
/// next waypoint, ramping elevation diagonally, facing the direction of travel, and
/// hugging the surface.
pub fn move_agents(
    time: Res<Time>,
    heightmap: Res<Heightmap>,
    mut agents: Query<(&mut Transform, &mut NavAgent)>,
) {
    let dt = time.delta;
    if dt <= 0.0 {
        return;
    }
    for (mut transform, mut agent) in &mut agents {
        let Some(&(cx, cz)) = agent.path.last() else {
            continue;
        };
        let target = heightmap.cell_center(cx as i64, cz as i64);
        let pos = transform.translation;
        let to = Vec3::new(target.x - pos.x, 0.0, target.z - pos.z);
        let dist = to.length();
        let step = (agent.speed * dt).max(0.05);

        if dist <= step {
            // Reached this waypoint; snap to its surface and advance.
            transform.translation = target;
            agent.path.pop();
        } else {
            let dir = to / dist;
            transform.translation.x += dir.x * step;
            transform.translation.z += dir.z * step;
            // Ramp the elevation toward the waypoint's height in proportion to the
            // horizontal progress, so an elevation change is a smooth diagonal
            // rather than a vertical step. (Flat ground keeps `target.y == y`.)
            let frac = step / dist;
            transform.translation.y += (target.y - transform.translation.y) * frac;
            // Face the direction of travel (the model's forward is +Z).
            transform.rotation = Quat::from_rotation_y(dir.x.atan2(dir.z));
        }
    }
}

/// Play each pawn's "Walk" clip while it's following a path and "Idle" otherwise.
/// Runs after `move_agents` so the path state is current.
pub fn update_pawn_animation(
    mut pawns: Query<(&NavAgent, &SkinnedMesh, &mut AnimationPlayer), With<Pawn>>,
) {
    for (agent, skin, mut player) in &mut pawns {
        let wanted = if agent.path.is_empty() { "Idle" } else { "Walk" };
        if let Some(clip) = skin.clips.iter().position(|c| c.name == wanted)
            && player.clip != clip
        {
            player.clip = clip;
            player.time = 0.0;
        }
    }
}
