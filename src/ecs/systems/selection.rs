use bevy_ecs::prelude::*;
use glam::{Mat4, Vec4};

use crate::ecs::components::{Pickable, Transform};
use crate::ecs::resources::{NavOverlay, Selected, SelectionBox, ViewProj};
use crate::render::context::RenderContext;
use crate::render::pipeline::SelUniform;
use crate::util::uniform_bytes;

/// Position the wireframe selection box on the selection's world AABB (the object's
/// transformed bounds, or the picked voxel's 1 m cube).
pub fn update_selection_box(
    ctx: NonSend<RenderContext>,
    selected: Res<Selected>,
    view_proj: Res<ViewProj>,
    mut sel_box: ResMut<SelectionBox>,
    objects: Query<(&Transform, &Pickable)>,
) {
    let (min, max) = match *selected {
        Selected::None => {
            sel_box.visible = false;
            return;
        }
        Selected::Object(entity) => match objects.get(entity) {
            Ok((transform, pickable)) => {
                let aabb = pickable.local_aabb.transformed(&transform.matrix());
                (aabb.min, aabb.max)
            }
            Err(_) => {
                sel_box.visible = false;
                return;
            }
        },
        Selected::Voxel { min, max, .. } => (min, max),
    };
    // Map the unit cube → the world AABB, then to clip space.
    let box_model = Mat4::from_translation(min) * Mat4::from_scale(max - min);
    let uniform = SelUniform {
        mvp: view_proj.0 * box_model,
        color: Vec4::new(1.0, 0.9, 0.2, 1.0),
    };
    ctx.queue
        .write_buffer(&sel_box.uniform, 0, &uniform_bytes(&uniform));
    sel_box.visible = true;
}

/// Keep the nav-mesh overlay's transform uniform current (its lines are in world
/// space, so the MVP is just the view-projection). Only runs when visible.
pub fn upload_nav_overlay(
    ctx: NonSend<RenderContext>,
    view_proj: Res<ViewProj>,
    nav: Res<NavOverlay>,
) {
    if !nav.visible {
        return;
    }
    let uniform = SelUniform {
        mvp: view_proj.0,
        color: Vec4::new(0.2, 1.0, 0.4, 1.0),
    };
    ctx.queue
        .write_buffer(&nav.uniform, 0, &uniform_bytes(&uniform));
}
