//! World + schedule construction. `build_world` ports the old `State::new`:
//! create the GPU context, build pipelines/bind groups, load assets, and spawn
//! the camera / light / chunk / glTF entities. `build_schedule` wires the
//! per-frame systems as `Sim → Upload → Render`.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ExecutorKind;
use glam::Vec3;
use wgpu::util::DeviceExt;
use winit::window::Window;

use crate::scene::camera::{self, CameraUniform};
use crate::ecs::components::{
    Camera, CameraGpu, FlyController, LightGpu, Pickable, PointLight, Transform,
};
use crate::ecs::resources::{
    BackgroundColor, CursorPos, DepthTexture, Input, LightMarker, Pipelines, Selected, SkyboxRes,
    Time, ViewProj, VoxelGpu, VoxelSettingsRes,
};
use crate::ecs::systems::{
    animate, fly_camera, fox_bundles, generate_terrain, orbit_light, update_selection_box,
    update_time, update_view_proj, upload_camera, upload_light, upload_model_transforms,
    upload_skybox, upload_voxel_settings,
};
use crate::assets;
use crate::render::context::RenderContext;
use crate::render::draw::render;
use crate::render::pipeline::{create_render_pipeline, create_selection_box};
use crate::render::texture;
use crate::scene::gltf_model;
use crate::scene::light::LightUniform;
use crate::scene::model::{self, Vertex};
use crate::scene::skybox;
use crate::scene::terrain::{self as voxel, VoxelSettings};
use crate::ui::{self, render_ui, sync_ui};
use crate::util as utils;

/// Build the per-frame schedule: simulate, upload, then render (chained, run
/// single-threaded since the render context and uploads are non-send).
pub fn build_schedule() -> Schedule {
    let mut schedule = Schedule::default();
    schedule.set_executor_kind(ExecutorKind::SingleThreaded);
    schedule.add_systems(
        (
            update_time,
            (fly_camera, orbit_light),
            sync_ui,
            update_view_proj,
            (
                upload_camera,
                upload_light,
                upload_skybox,
                upload_voxel_settings,
                upload_model_transforms,
                animate,
                update_selection_box,
                render_ui,
            ),
            render,
        )
            .chain(),
    );
    schedule
}

/// A simple uniform-buffer bind group layout (one uniform at binding 0).
fn uniform_layout(device: &wgpu::Device, label: &str, vis: wgpu::ShaderStages) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: vis,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    })
}

/// A texture + sampler bind group layout for the given view dimension.
fn texture_layout(
    device: &wgpu::Device,
    label: &str,
    dim: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayout {
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    multisampled: false,
                    view_dimension: dim,
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    })
}

pub async fn build_world(window: Arc<Window>) -> anyhow::Result<World> {
    let ctx = RenderContext::new(window).await?;
    let device = &ctx.device;
    let mut world = World::new();

    // Precompute the terrain heightmap once; chunk meshing and picking read it.
    let terrain_params = voxel::TerrainParams::default();
    let heightmap = voxel::Heightmap::generate(&terrain_params);

    // --- Shared bind group layouts ---
    let camera_layout = uniform_layout(
        device,
        "camera_bind_group_layout",
        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
    );
    let light_layout = uniform_layout(
        device,
        "light_bind_group_layout",
        wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
    );
    let voxel_settings_layout =
        uniform_layout(device, "voxel_settings_layout", wgpu::ShaderStages::FRAGMENT);
    let gltf_model_layout =
        uniform_layout(device, "gltf_model_layout", wgpu::ShaderStages::VERTEX);
    let voxel_texture_layout =
        texture_layout(device, "voxel_texture_layout", wgpu::TextureViewDimension::D2Array);
    let gltf_texture_layout =
        texture_layout(device, "gltf_texture_layout", wgpu::TextureViewDimension::D2);

    // --- Camera entity ---
    let (cam_x, cam_z) = (16.0, 64.0);
    let eye = Vec3::new(cam_x, heightmap.surface_y(cam_x, cam_z) + 1.7, cam_z);
    let yaw = -std::f32::consts::FRAC_PI_2;
    let aspect = if ctx.config.height > 0 {
        ctx.config.width as f32 / ctx.config.height as f32
    } else {
        1.0
    };
    let cam_uniform = CameraUniform {
        view_position: eye.extend(1.0),
        view_proj: camera::view_projection(eye, yaw, 0.0, 45.0, aspect, 0.1, 500.0),
    };
    let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Camera Buffer"),
        contents: &utils::uniform_bytes(&cam_uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &camera_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: camera_buffer.as_entire_binding(),
        }],
        label: Some("camera_bind_group"),
    });
    world.spawn((
        Camera {
            fovy: 45.0,
            znear: 0.1,
            zfar: 500.0,
            aspect,
        },
        Transform::from_translation(eye),
        FlyController {
            yaw,
            pitch: 0.0,
            speed: 0.5,
            rotate_speed: 0.03,
        },
        CameraGpu {
            buffer: camera_buffer,
            bind_group: camera_bind_group,
        },
    ));

    // --- Light entity --- (load the marker mesh now so we can bound it for picking)
    let (obj_model, light_aabb) = assets::load_model("cube.obj", &ctx.device).await.unwrap();
    let light_uniform = LightUniform {
        position: glam::vec3(30.0, 40.0, 30.0),
        color: glam::vec3(1.0, 1.0, 1.0),
    };
    let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Light Buffer"),
        contents: &utils::uniform_bytes(&light_uniform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &light_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: light_buffer.as_entire_binding(),
        }],
        label: Some("light_bind_group"),
    });
    world.spawn((
        PointLight {
            color: light_uniform.color,
        },
        Transform::from_translation(light_uniform.position),
        LightGpu {
            buffer: light_buffer,
            bind_group: light_bind_group,
        },
        Pickable {
            local_aabb: light_aabb,
        },
    ));

    // --- Voxel texture + settings ---
    let voxel_texture = assets::load_texture_array(
        "array_texture.png",
        voxel::NUM_TEXTURE_LAYERS,
        &ctx.device,
        &ctx.queue,
    )
    .await?;
    let voxel_texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &voxel_texture_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&voxel_texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&voxel_texture.sampler),
            },
        ],
        label: Some("voxel_texture_bind_group"),
    });

    let voxel_settings = VoxelSettings::new(true);
    let voxel_settings_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Voxel Settings Buffer"),
        contents: &utils::uniform_bytes(&voxel_settings),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let voxel_settings_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &voxel_settings_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: voxel_settings_buffer.as_entire_binding(),
        }],
        label: Some("voxel_settings_bind_group"),
    });

    // --- Pipelines ---
    let depth_format = Some(texture::Texture::DEPTH_FORMAT);
    let voxel_pipeline = {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Voxel Pipeline Layout"),
            bind_group_layouts: &[
                Some(&voxel_texture_layout),
                Some(&camera_layout),
                Some(&light_layout),
                Some(&voxel_settings_layout),
            ],
            immediate_size: 0,
        });
        create_render_pipeline(
            device,
            &layout,
            ctx.config.format,
            depth_format,
            &[voxel::VoxelVertex::desc()],
            wgpu::ShaderModuleDescriptor {
                label: Some("Voxel Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    concat!(
                        include_str!("../render/shaders/common.wgsl"),
                        include_str!("../render/shaders/voxel.wgsl")
                    )
                    .into(),
                ),
            },
        )
    };
    let light_pipeline = {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Light Pipeline Layout"),
            bind_group_layouts: &[Some(&camera_layout), Some(&light_layout)],
            immediate_size: 0,
        });
        create_render_pipeline(
            device,
            &layout,
            ctx.config.format,
            depth_format,
            &[model::ModelVertex::desc()],
            wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    concat!(
                        include_str!("../render/shaders/common.wgsl"),
                        include_str!("../render/shaders/light.wgsl")
                    )
                    .into(),
                ),
            },
        )
    };
    let gltf_pipeline = {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("glTF Pipeline Layout"),
            bind_group_layouts: &[
                Some(&gltf_texture_layout),
                Some(&camera_layout),
                Some(&light_layout),
                Some(&gltf_model_layout),
            ],
            immediate_size: 0,
        });
        create_render_pipeline(
            device,
            &layout,
            ctx.config.format,
            depth_format,
            &[gltf_model::GltfVertex::desc()],
            wgpu::ShaderModuleDescriptor {
                label: Some("glTF Shader"),
                source: wgpu::ShaderSource::Wgsl(
                    concat!(
                        include_str!("../render/shaders/common.wgsl"),
                        include_str!("../render/shaders/gltf.wgsl")
                    )
                    .into(),
                ),
            },
        )
    };

    // --- Foxes: scatter `fox_count` instances across the surface, built from a
    // shared template kept as a resource so regeneration can re-spawn them. ---
    let fox_template = assets::load_gltf_template(
        "fox.glb",
        &ctx.device,
        &ctx.queue,
        &gltf_texture_layout,
        &gltf_model_layout,
    )
    .await?;
    for fox in fox_bundles(&ctx.device, &fox_template, &heightmap) {
        world.spawn(fox);
    }
    world.insert_resource(fox_template);

    // Chunk entities are spawned by the `generate_terrain` startup system (below).

    // --- Remaining singletons ---
    let depth_texture =
        texture::Texture::create_depth_texture(&ctx.device, &ctx.config, "depth_texture");
    let skybox = skybox::Skybox::new(&ctx.device, &ctx.queue, &ctx.config).await?;
    let selection_box = create_selection_box(&ctx.device, ctx.config.format);

    // --- Slint UI overlay (software-rendered). Sized to the surface; recreated
    // on resize. The real size is set on the first resize when the surface is 0×0.
    let ui_w = ctx.config.width.max(1);
    let ui_h = ctx.config.height.max(1);
    let slint_ui = ui::create_ui(ui_w, ui_h, ctx.window.scale_factor() as f32);
    slint_ui
        .component
        .set_ao_enabled(voxel_settings.ao_enabled != 0);
    slint_ui.component.set_animation_clip(1); // Walk
    slint_ui.component.set_in_game(ui::SKIP_TITLE_SCREEN);
    // Seed the terrain-generator sliders from the default parameters.
    let c = &slint_ui.component;
    c.set_gen_seed(terrain_params.seed as f32);
    c.set_gen_frequency(terrain_params.frequency as f32);
    c.set_gen_octaves(terrain_params.octaves as f32);
    c.set_gen_lacunarity(terrain_params.lacunarity as f32);
    c.set_gen_persistence(terrain_params.persistence as f32);
    c.set_gen_max_height(terrain_params.max_height as f32);
    c.set_gen_world_size(terrain_params.grid_size as f32);
    c.set_gen_flatness(terrain_params.flatness as f32);
    c.set_gen_peakiness(terrain_params.peakiness as f32);
    c.set_gen_layer_blend(terrain_params.layer_blend as f32);
    c.set_gen_fox_count(terrain_params.fox_count as f32);
    let ui_overlay = ui::create_overlay(&ctx.device, ctx.config.format, ui_w, ui_h);

    world.insert_resource(Pipelines {
        voxel: voxel_pipeline,
        light: light_pipeline,
        gltf: gltf_pipeline,
    });
    world.insert_resource(DepthTexture(depth_texture));
    world.insert_resource(VoxelGpu {
        texture_bind_group: voxel_texture_bind_group,
        settings_buffer: voxel_settings_buffer,
        settings_bind_group: voxel_settings_bind_group,
    });
    world.insert_resource(VoxelSettingsRes(voxel_settings));
    world.insert_resource(SkyboxRes(skybox));
    world.insert_resource(LightMarker(obj_model));
    world.insert_resource(BackgroundColor(wgpu::Color {
        r: 0.0,
        g: 0.0,
        b: 0.0,
        a: 1.0,
    }));
    world.insert_resource(ViewProj(cam_uniform.view_proj));
    world.insert_resource(Time::default());
    world.insert_resource(Input::default());
    world.insert_resource(Selected::default());
    world.insert_resource(CursorPos::default());
    world.insert_resource(selection_box);
    world.insert_resource(ui_overlay);
    world.insert_resource(heightmap);
    world.insert_resource(terrain_params);
    world.insert_non_send_resource(slint_ui);
    world.insert_non_send_resource(ctx);

    // Run terrain generation exactly once (it reads the heightmap + render context
    // and spawns the chunk entities).
    let mut startup = Schedule::default();
    startup.set_executor_kind(ExecutorKind::SingleThreaded);
    startup.add_systems(generate_terrain);
    startup.run(&mut world);

    Ok(world)
}
