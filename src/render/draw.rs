//! The render exclusive system (`render`) and the shared `resize` helper.
//!
//! `render` takes `&mut World` and uses a [`SystemState`] to borrow all of its
//! read-only resources and queries together for the duration of one render pass.

use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemState;

use crate::ecs::components::{Camera, CameraGpu, LightGpu};
use crate::ecs::resources::{
    BackgroundColor, DepthTexture, LightMarker, Pipelines, SkyboxRes, VoxelGpu,
};
use crate::gltf_model::GltfModel;
use crate::model::DrawLight;
use crate::render::context::RenderContext;
use crate::texture;
use crate::voxel::VoxelChunk;

/// Reconfigure the surface + depth texture and update the camera aspect ratio.
/// Shared by the `Resized` event and the render system's lazy first-configure.
pub fn resize(world: &mut World, width: u32, height: u32) {
    if width == 0 || height == 0 {
        return;
    }
    world
        .non_send_resource_mut::<RenderContext>()
        .resize(width, height);

    // The depth texture must match the surface size or the pass attachments mismatch.
    let depth = {
        let ctx = world.non_send_resource::<RenderContext>();
        texture::Texture::create_depth_texture(&ctx.device, &ctx.config, "depth_texture")
    };
    world.resource_mut::<DepthTexture>().0 = depth;

    let aspect = width as f32 / height as f32;
    let mut q = world.query::<&mut Camera>();
    for mut cam in q.iter_mut(world) {
        cam.aspect = aspect;
    }
}

// The render system genuinely needs many resources + queries at once; the big
// `SystemState` tuple is the idiomatic way to borrow them together.
#[allow(clippy::type_complexity)]
pub fn render(world: &mut World) {
    // Lazy first-configure: some platforms (macOS `with_maximized(true)`) report a
    // 0×0 size at creation and send no initial `Resized`. Do this mutating step
    // before the read-only draw borrow below.
    if !world
        .non_send_resource::<RenderContext>()
        .is_surface_configured
    {
        let size = world
            .non_send_resource::<RenderContext>()
            .window
            .inner_size();
        resize(world, size.width, size.height);
    }
    {
        let ctx = world.non_send_resource::<RenderContext>();
        ctx.window.request_redraw();
        if !ctx.is_surface_configured {
            return;
        }
    }

    let mut state: SystemState<(
        NonSend<RenderContext>,
        Res<Pipelines>,
        Res<DepthTexture>,
        Res<VoxelGpu>,
        Res<BackgroundColor>,
        Res<SkyboxRes>,
        Res<LightMarker>,
        Query<&CameraGpu>,
        Query<&LightGpu>,
        Query<&VoxelChunk>,
        Query<&GltfModel>,
    )> = SystemState::new(world);
    let (ctx, pipelines, depth, voxel_gpu, bg, skybox, marker, cam_q, light_q, chunks, gltfs) =
        state.get(world);

    let output = match ctx.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(t) => t,
        wgpu::CurrentSurfaceTexture::Suboptimal(t) => {
            ctx.reconfigure();
            t
        }
        wgpu::CurrentSurfaceTexture::Timeout
        | wgpu::CurrentSurfaceTexture::Occluded
        | wgpu::CurrentSurfaceTexture::Validation => return,
        wgpu::CurrentSurfaceTexture::Outdated => {
            ctx.reconfigure();
            return;
        }
        wgpu::CurrentSurfaceTexture::Lost => {
            log::error!("surface lost");
            return;
        }
    };

    let (Ok(cam_gpu), Ok(light_gpu)) = (cam_q.single(), light_q.single()) else {
        return;
    };
    let camera_bg = &cam_gpu.bind_group;
    let light_bg = &light_gpu.bind_group;

    let view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

    {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(bg.0),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth.0.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        // Light marker.
        render_pass.set_pipeline(&pipelines.light);
        render_pass.draw_light_model(&marker.0, camera_bg, light_bg);

        // Voxel chunks: texture (0), camera (1), light (2), settings (3) are shared;
        // each chunk entity swaps its own vertex/index buffers.
        render_pass.set_pipeline(&pipelines.voxel);
        render_pass.set_bind_group(0, &voxel_gpu.texture_bind_group, &[]);
        render_pass.set_bind_group(1, camera_bg, &[]);
        render_pass.set_bind_group(2, light_bg, &[]);
        render_pass.set_bind_group(3, &voxel_gpu.settings_bind_group, &[]);
        for chunk in chunks.iter() {
            render_pass.set_vertex_buffer(0, chunk.vertex_buffer.slice(..));
            render_pass.set_index_buffer(chunk.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..chunk.num_indices, 0, 0..1);
        }

        // glTF models: camera (1) + light (2) shared; each entity swaps texture (0),
        // transform (3), and buffers.
        render_pass.set_pipeline(&pipelines.gltf);
        render_pass.set_bind_group(1, camera_bg, &[]);
        render_pass.set_bind_group(2, light_bg, &[]);
        for m in gltfs.iter() {
            render_pass.set_bind_group(0, &m.texture_bind_group, &[]);
            render_pass.set_bind_group(3, &m.model_bind_group, &[]);
            render_pass.set_vertex_buffer(0, m.vertex_buffer.slice(..));
            render_pass.set_index_buffer(m.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..m.num_indices, 0, 0..1);
        }

        // Sky last so it only fills pixels the scene didn't cover.
        skybox.0.draw(&mut render_pass);
    }

    ctx.queue.submit(std::iter::once(encoder.finish()));
    output.present();
}
