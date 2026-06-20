use bevy_ecs::prelude::*;
use glam::{Quat, Vec3};

use crate::assets::{AssetRegistry, LoadedAsset};
use crate::ecs::components::{
    AnimationPlayer, FlyController, NavAgent, Pickable, Placed, SkinnedMesh, Transform, Tree,
};
use crate::ecs::resources::NavOverlay;
use crate::render::context::RenderContext;
use crate::render::pipeline::nav_lines_buffer;
use crate::scene::gltf_model::GltfModel;
use crate::scene::nav::NavMesh;
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

type FoxBundle = (
    GltfModel,
    Transform,
    SkinnedMesh,
    AnimationPlayer,
    Pickable,
    NavAgent,
);

/// Build `heightmap.params().fox_count` fox instances scattered across valid
/// surface points, each with a deterministic (seeded) random facing + animation
/// phase so they don't move in lockstep. Empty if the template carries no skin.
pub fn fox_bundles(
    device: &wgpu::Device,
    fox: &LoadedAsset,
    heightmap: &Heightmap,
) -> Vec<FoxBundle> {
    let Some(skin) = &fox.template.skin else {
        return Vec::new();
    };
    let p = heightmap.params();
    let clip = fox.clip.unwrap_or(0);
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
                scale: Vec3::splat(fox.scale),
            };
            (
                fox.template.instantiate(device),
                transform,
                skin.clone(),
                AnimationPlayer {
                    clip,
                    time: phase,
                    speed: 1.0,
                },
                Pickable {
                    local_aabb: fox.template.local_aabb,
                },
                NavAgent {
                    path: Vec::new(),
                    speed: 2.5,
                },
            )
        })
        .collect()
}

type TreeBundle = (GltfModel, Transform, Pickable, Tree);

/// Build `heightmap.params().tree_count` tree instances at Poisson-distributed
/// surface points, each with a deterministic random facing.
pub fn tree_bundles(
    device: &wgpu::Device,
    tree: &LoadedAsset,
    heightmap: &Heightmap,
) -> Vec<TreeBundle> {
    let p = heightmap.params();
    // Keep trees at least their footprint apart so the models never overlap.
    let aabb = tree.template.local_aabb;
    let footprint = (aabb.max.x - aabb.min.x).max(aabb.max.z - aabb.min.z) * tree.scale;
    heightmap
        .poisson_surface(p.tree_count, footprint)
        .into_iter()
        .enumerate()
        .map(|(i, pos)| {
            let yaw = terrain::hash01(i as i64, 3, 3, p.seed) * std::f32::consts::TAU;
            (
                tree.template.instantiate(device),
                Transform {
                    translation: pos,
                    rotation: Quat::from_rotation_y(yaw),
                    scale: Vec3::splat(tree.scale),
                },
                Pickable {
                    local_aabb: tree.template.local_aabb,
                },
                Tree,
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
            tree_count: c.get_gen_tree_count().round() as u32,
            forest_density: c.get_gen_forest_density() as f64,
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

    // Replace the foxes: despawn the old ones (foxes carry a `NavAgent`, so this
    // leaves other skinned characters alone), scatter a fresh set on the surface.
    let old_foxes: Vec<Entity> = world
        .query_filtered::<Entity, With<NavAgent>>()
        .iter(world)
        .collect();
    for e in old_foxes {
        world.despawn(e);
    }
    let foxes = {
        let device = &world.non_send_resource::<RenderContext>().device;
        match world.resource::<AssetRegistry>().get("fox") {
            Some(fox) => fox_bundles(device, fox, &heightmap),
            None => Vec::new(),
        }
    };
    for fox in foxes {
        world.spawn(fox);
    }

    // Replace the trees: despawn the old ones, re-scatter (Poisson) on the surface.
    let old_trees: Vec<Entity> = world
        .query_filtered::<Entity, With<Tree>>()
        .iter(world)
        .collect();
    for e in old_trees {
        world.despawn(e);
    }
    let trees = {
        let device = &world.non_send_resource::<RenderContext>().device;
        match world.resource::<AssetRegistry>().get("tree") {
            Some(tree) => tree_bundles(device, tree, &heightmap),
            None => Vec::new(),
        }
    };
    for tree in trees {
        world.spawn(tree);
    }

    // Rebuild the nav mesh and refresh its overlay's line buffer.
    let nav_mesh = NavMesh::build(&heightmap);
    let (lines, num_vertices) = {
        let device = &world.non_send_resource::<RenderContext>().device;
        nav_lines_buffer(device, &nav_mesh)
    };
    {
        let mut overlay = world.resource_mut::<NavOverlay>();
        overlay.lines = lines;
        overlay.num_vertices = num_vertices;
    }
    world.insert_resource(nav_mesh);

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

    // Keep declaratively-placed models (`assets.ron`) planted on the new surface.
    let mut figures = world.query_filtered::<&mut Transform, With<Placed>>();
    for mut t in figures.iter_mut(world) {
        let (x, z) = (t.translation.x, t.translation.z);
        t.translation.y = heightmap.surface_y(x, z);
    }

    world.insert_resource(heightmap);
    world.insert_resource(params);
}
