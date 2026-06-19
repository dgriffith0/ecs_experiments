use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::assets::GltfTemplate;
use crate::ecs::components::{AnimationPlayer, FlyController, Pickable, SkinnedMesh, Transform};
use crate::render::context::RenderContext;
use crate::scene::gltf_model::GltfModel;
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

type FoxBundle = (GltfModel, Transform, SkinnedMesh, AnimationPlayer, Pickable);

/// Build `heightmap.params().fox_count` fox instances scattered across valid
/// surface points, each with a deterministic (seeded) random facing + animation
/// phase so they don't move in lockstep. Empty if the template carries no skin.
pub fn fox_bundles(
    device: &wgpu::Device,
    template: &GltfTemplate,
    heightmap: &Heightmap,
) -> Vec<FoxBundle> {
    let Some(skin) = &template.skin else {
        return Vec::new();
    };
    let p = heightmap.params();
    heightmap
        .scatter_surface(p.fox_count)
        .into_iter()
        .enumerate()
        .map(|(i, pos)| {
            let yaw = terrain::hash01(i as i64, 7, 7, p.seed) * std::f32::consts::TAU;
            let phase = terrain::hash01(i as i64, 9, 9, p.seed) * 2.0;
            let transform = Transform {
                translation: pos,
                rotation: Quat::from_rotation_y(yaw),
                scale: Vec3::splat(0.01),
            };
            (
                template.instantiate(device),
                transform,
                skin.clone(),
                AnimationPlayer {
                    clip: 1, // Walk
                    time: phase,
                    speed: 1.0,
                },
                Pickable {
                    local_aabb: template.local_aabb,
                },
            )
        })
        .collect()
}

/// Rebuild the world from the terrain-generator sliders: re-sample the heightmap,
/// replace the chunk meshes, re-scatter the foxes, and re-frame the preview
/// camera. Triggered by the generator's "Regenerate" button (and on entering it).
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
            flatness: c.get_gen_flatness() as f64,
            peakiness: c.get_gen_peakiness() as f64,
            layer_blend: c.get_gen_layer_blend() as f64,
            fox_count: c.get_gen_fox_count().round() as u32,
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

    // Replace the foxes: despawn the old ones, scatter a fresh set on the surface.
    let old_foxes: Vec<Entity> = world
        .query_filtered::<Entity, With<SkinnedMesh>>()
        .iter(world)
        .collect();
    for e in old_foxes {
        world.despawn(e);
    }
    let foxes = {
        let device = &world.non_send_resource::<RenderContext>().device;
        let template = world.resource::<GltfTemplate>();
        fox_bundles(device, template, &heightmap)
    };
    for fox in foxes {
        world.spawn(fox);
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

    world.insert_resource(heightmap);
    world.insert_resource(params);
}
