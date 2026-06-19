use bevy_ecs::prelude::*;

use crate::ecs::components::Transform;
use crate::ecs::resources::{SkyboxRes, ViewProj, VoxelGpu, VoxelSettingsRes};
use crate::render::context::RenderContext;
use crate::scene::gltf_model::GltfModel;
use crate::util::uniform_bytes;

/// Keep the sky oriented with the camera (it samples by view direction).
pub fn upload_skybox(
    ctx: NonSend<RenderContext>,
    view_proj: Res<ViewProj>,
    skybox: Res<SkyboxRes>,
) {
    skybox.0.update(&ctx.queue, view_proj.0);
}

/// Write each glTF model's transform matrix to its uniform, only when the
/// entity's `Transform` changes (newly-spawned entities count as changed).
pub fn upload_model_transforms(
    ctx: NonSend<RenderContext>,
    q: Query<(&Transform, &GltfModel), Changed<Transform>>,
) {
    for (transform, model) in &q {
        ctx.queue
            .write_buffer(&model.model_buffer, 0, &uniform_bytes(&transform.matrix()));
    }
}

/// Re-upload the voxel settings (AO flag) only when they change.
pub fn upload_voxel_settings(
    ctx: NonSend<RenderContext>,
    settings: Res<VoxelSettingsRes>,
    gpu: Res<VoxelGpu>,
) {
    if settings.is_changed() {
        ctx.queue
            .write_buffer(&gpu.settings_buffer, 0, &uniform_bytes(&settings.0));
    }
}
