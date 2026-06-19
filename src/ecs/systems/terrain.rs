use bevy_ecs::prelude::*;
use glam::Vec3;

use crate::ecs::components::{FlyController, SkinnedMesh, Transform};
use crate::render::context::RenderContext;
use crate::scene::terrain::{self, Heightmap, TerrainParams, VoxelChunk};
use crate::ui::Ui;

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

/// Rebuild the world from the terrain-generator sliders: re-sample the heightmap,
/// replace the chunk meshes, and re-frame the preview camera + fox. Triggered by
/// the generator's "Regenerate" button (and on entering that screen).
pub fn regenerate_terrain(world: &mut World) {
    // Read the slider values off the Slint component.
    let params = {
        let c = &world.non_send_resource::<Ui>().component;
        TerrainParams {
            seed: c.get_gen_seed().round() as u32,
            frequency: c.get_gen_frequency() as f64,
            octaves: c.get_gen_octaves().round() as usize,
            lacunarity: c.get_gen_lacunarity() as f64,
            persistence: c.get_gen_persistence() as f64,
            max_height: c.get_gen_max_height().round() as u32,
            grid_size: c.get_gen_world_size().round() as u32,
        }
    };
    let heightmap = Heightmap::generate(&params);

    // Swap the chunk meshes: despawn the old grid, mesh + spawn the new one.
    let old: Vec<Entity> = world
        .query_filtered::<Entity, With<VoxelChunk>>()
        .iter(world)
        .collect();
    for e in old {
        world.despawn(e);
    }
    let chunks = {
        let device = &world.non_send_resource::<RenderContext>().device;
        terrain::generate_chunk_grid(device, &heightmap)
    };
    for chunk in chunks {
        world.spawn(chunk);
    }

    // Re-frame the camera to an overview looking at the world centre.
    let grid = heightmap.grid();
    let (cx, cz) = terrain::world_center_xz(grid);
    let span = terrain::world_span(grid);
    let center = Vec3::new(
        cx,
        terrain::terrain_y_bounds().0 + params.max_height as f32 * 0.5,
        cz,
    );
    let cam = Vec3::new(cx, center.y + span * 0.5, cz + span * 0.6);
    let to = (center - cam).normalize();
    let (yaw, pitch) = (to.z.atan2(to.x), to.y.asin());
    if let Some((mut t, mut fly)) = world
        .query::<(&mut Transform, &mut FlyController)>()
        .iter_mut(world)
        .next()
    {
        t.translation = cam;
        fly.yaw = yaw;
        fly.pitch = pitch;
    }

    // Sit the fox on the new surface at the world centre.
    let fox_pos = Vec3::new(cx, heightmap.surface_y(cx, cz), cz);
    if let Some(mut t) = world
        .query_filtered::<&mut Transform, With<SkinnedMesh>>()
        .iter_mut(world)
        .next()
    {
        t.translation = fox_pos;
    }

    world.insert_resource(heightmap);
    world.insert_resource(params);
}
