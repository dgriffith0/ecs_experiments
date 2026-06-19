use bevy_ecs::prelude::*;

use crate::ecs::components::{AnimationPlayer, SkinnedMesh};
use crate::ecs::resources::Time;
use crate::render::context::RenderContext;
use crate::scene::animation;
use crate::scene::gltf_model::{GltfModel, GltfVertex};

/// Advance each skinned mesh's animation and re-upload its CPU-skinned vertices.
/// Samples the active clip → joint matrices → linear-blend skin → vertex buffer.
pub fn animate(
    ctx: NonSend<RenderContext>,
    time: Res<Time>,
    mut q: Query<(&mut AnimationPlayer, &SkinnedMesh, &GltfModel)>,
) {
    for (mut player, skin, model) in &mut q {
        let Some(clip) = skin.clips.get(player.clip) else {
            continue;
        };
        if clip.duration > 0.0 {
            player.time = (player.time + time.delta * player.speed).rem_euclid(clip.duration);
        }
        let locals = clip.sample(&skin.skeleton, player.time);
        let mats = animation::joint_matrices(&skin.skeleton, &locals);
        let positions =
            animation::skin_positions(&skin.base_positions, &skin.joints, &skin.weights, &mats);
        let vertices: Vec<GltfVertex> = positions
            .iter()
            .zip(&skin.tex_coords)
            .map(|(p, &tex_coords)| GltfVertex {
                position: p.to_array(),
                tex_coords,
            })
            .collect();
        ctx.queue
            .write_buffer(&model.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
    }
}
