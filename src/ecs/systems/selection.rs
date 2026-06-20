use bevy_ecs::prelude::*;
use glam::{Mat4, Vec3, Vec4};

use crate::ecs::components::{NavAgent, Pawn, Pickable, Transform};
use crate::ecs::resources::{
    DestinationOverlay, LineOverlay, NavOverlay, Selection, SelectionOverlay, ViewProj,
};
use crate::render::context::RenderContext;
use crate::render::pipeline::{box_edges, SelUniform};
use crate::scene::terrain::Heightmap;
use crate::util::uniform_bytes;

/// Re-fill a line overlay's vertex buffer + uniform for this frame. Lines are in
/// world space, so the MVP is just the view-projection.
fn upload_line_overlay(
    queue: &wgpu::Queue,
    overlay: &mut LineOverlay,
    mvp: Mat4,
    color: Vec4,
    verts: &[[f32; 3]],
) {
    let n = (verts.len() as u32).min(overlay.capacity);
    overlay.num_vertices = n;
    overlay.visible = n > 0;
    if n == 0 {
        return;
    }
    queue.write_buffer(
        &overlay.lines,
        0,
        bytemuck::cast_slice(&verts[..n as usize]),
    );
    queue.write_buffer(
        &overlay.uniform,
        0,
        &uniform_bytes(&SelUniform { mvp, color }),
    );
}

/// Draw a yellow wireframe box around every selected pawn (its world-space AABB).
pub fn update_selection_overlay(
    ctx: NonSend<RenderContext>,
    view_proj: Res<ViewProj>,
    selection: Res<Selection>,
    mut overlay: ResMut<SelectionOverlay>,
    pickables: Query<(&Transform, &Pickable)>,
) {
    let mut verts: Vec<[f32; 3]> = Vec::new();
    for &entity in &selection.0 {
        if let Ok((transform, pickable)) = pickables.get(entity) {
            let aabb = pickable.local_aabb.transformed(&transform.matrix());
            verts.extend_from_slice(&box_edges(aabb.min, aabb.max));
        }
    }
    upload_line_overlay(
        &ctx.queue,
        &mut overlay.0,
        view_proj.0,
        Vec4::new(1.0, 0.9, 0.2, 1.0),
        &verts,
    );
}

/// Draw a green wireframe box on each moving pawn's destination cell (the goal of
/// its current path), so you can see where the group is headed until it arrives.
pub fn update_destination_overlay(
    ctx: NonSend<RenderContext>,
    view_proj: Res<ViewProj>,
    heightmap: Res<Heightmap>,
    mut overlay: ResMut<DestinationOverlay>,
    pawns: Query<&NavAgent, With<Pawn>>,
) {
    let mut verts: Vec<[f32; 3]> = Vec::new();
    for agent in &pawns {
        // Path is stored reversed, so `first()` is the goal (assigned) cell.
        if let Some(&(cx, cz)) = agent.path.first() {
            let c = heightmap.cell_center(cx as i64, cz as i64);
            let min = Vec3::new(c.x - 0.5, c.y, c.z - 0.5);
            let max = Vec3::new(c.x + 0.5, c.y + 1.0, c.z + 0.5);
            verts.extend_from_slice(&box_edges(min, max));
        }
    }
    upload_line_overlay(
        &ctx.queue,
        &mut overlay.0,
        view_proj.0,
        Vec4::new(0.2, 1.0, 0.4, 1.0),
        &verts,
    );
}

/// Keep the nav-mesh overlay's transform uniform current (lines are world-space,
/// so the MVP is just the view-projection). Only runs when visible.
pub fn upload_nav_overlay(
    ctx: NonSend<RenderContext>,
    view_proj: Res<ViewProj>,
    nav: Res<NavOverlay>,
) {
    if !nav.visible {
        return;
    }
    queue_uniform(
        &ctx.queue,
        &nav.uniform,
        view_proj.0,
        Vec4::new(0.2, 1.0, 0.4, 1.0),
    );
}

/// Write a `SelUniform { mvp, color }` into an overlay's uniform buffer.
fn queue_uniform(queue: &wgpu::Queue, buffer: &wgpu::Buffer, mvp: Mat4, color: Vec4) {
    queue.write_buffer(buffer, 0, &uniform_bytes(&SelUniform { mvp, color }));
}
