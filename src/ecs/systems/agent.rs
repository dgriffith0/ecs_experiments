use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};
use rand::Rng;

use crate::ecs::components::{NavAgent, Transform};
use crate::ecs::resources::Time;
use crate::scene::nav::NavMesh;
use crate::scene::terrain::Heightmap;

/// Wander every nav agent (the foxes) over the walkable surface: follow the queued
/// path, and when it runs out A* to a fresh random walkable cell. Movement hugs the
/// terrain surface and turns to face the direction of travel.
pub fn wander_foxes(
    time: Res<Time>,
    nav: Res<NavMesh>,
    heightmap: Res<Heightmap>,
    mut agents: Query<(&mut Transform, &mut NavAgent)>,
) {
    let dt = time.delta;
    let cells = nav.cells();
    if dt <= 0.0 || cells.is_empty() {
        return;
    }
    let mut rng = rand::rng();

    for (mut transform, mut agent) in &mut agents {
        if agent.path.is_empty() {
            // Plan a new route from the current cell to a random walkable one.
            let start = heightmap.cell_coords(transform.translation.x, transform.translation.z);
            let goal = cells[rng.random_range(0..cells.len())];
            if let Some(mut path) = nav.find_path(&heightmap, start, goal) {
                path.reverse(); // pop() now yields the next waypoint
                path.pop(); // drop the start cell
                agent.path = path;
            }
            continue;
        }

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
            // Face the direction of travel (the Fox model's forward is +Z).
            transform.rotation = Quat::from_rotation_y(dir.x.atan2(dir.z));
        }
    }
}
