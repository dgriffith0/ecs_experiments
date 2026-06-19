use bevy_ecs::prelude::*;

use crate::render::context::RenderContext;
use crate::scene::terrain::{self, Heightmap};

/// Mesh the voxel chunks from the precomputed `Heightmap` and spawn them. Run once.
pub fn generate_terrain(
    ctx: NonSend<RenderContext>,
    heightmap: Res<Heightmap>,
    mut commands: Commands,
) {
    for chunk in terrain::generate_chunk_grid(&ctx.device, &heightmap) {
        commands.spawn(chunk);
    }
}
