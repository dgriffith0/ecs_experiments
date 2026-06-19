use bevy_ecs::prelude::*;

use crate::ecs::resources::Time;

/// Measure wall-clock delta time at the start of each frame.
pub fn update_time(mut time: ResMut<Time>) {
    let now = std::time::Instant::now();
    time.delta = match time.last {
        Some(last) => now.duration_since(last).as_secs_f32(),
        None => 0.0,
    };
    time.last = Some(now);
}
