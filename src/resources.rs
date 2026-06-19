use std::io::Cursor;

use wgpu::util::DeviceExt;

use crate::gltf_model::{GltfModel, GltfVertex};
use crate::{model, texture, utils};

pub async fn load_string(file_name: &str) -> anyhow::Result<String> {
    let txt = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read_to_string(path)?
    };

    Ok(txt)
}

pub async fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    let data = {
        let path = std::path::Path::new(env!("OUT_DIR"))
            .join("res")
            .join(file_name);
        std::fs::read(path)?
    };

    Ok(data)
}

/// Load a vertically-stacked image file as a 2D texture array of `layers`
/// square tiles (e.g. `array_texture.png` is 250×1000 = four 250×250 layers).
pub async fn load_texture_array(
    file_name: &str,
    layers: u32,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name).await?;
    let img = image::load_from_memory(&data)?;
    texture::Texture::from_image_array(device, queue, &img, layers, Some(file_name))
}

/// Load six skybox face images into a cubemap texture. `faces` are filenames in
/// wgpu cube-layer order: `[+X, -X, +Y, -Y, +Z, -Z]`.
pub async fn load_cubemap(
    faces: [&str; 6],
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let mut images = Vec::with_capacity(6);
    for name in faces {
        let data = load_binary(name).await?;
        images.push(image::load_from_memory(&data)?);
    }
    let images: [image::DynamicImage; 6] = images
        .try_into()
        .map_err(|_| anyhow::anyhow!("expected 6 cubemap faces"))?;
    texture::Texture::from_cube_images(device, queue, &images, Some("skybox"))
}

/// Load an OBJ as geometry only. The model is drawn solely as the unlit light
/// marker, which reads vertex positions, so we ignore materials/uvs/normals.
pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
) -> anyhow::Result<model::Model> {
    let obj_text = load_string(file_name).await?;
    let obj_cursor = Cursor::new(obj_text);

    let (models, _materials) = tobj::tokio::load_obj_buf(
        obj_cursor,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
        |p| async move {
            let mat_text = load_string(&p.to_string_lossy()).await.unwrap();
            tobj::tokio::load_mtl_buf(Cursor::new(mat_text)).await
        },
    )
    .await?;

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
                    position: [
                        m.mesh.positions[i * 3],
                        m.mesh.positions[i * 3 + 1],
                        m.mesh.positions[i * 3 + 2],
                    ],
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Vertex Buffer", file_name)),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{:?} Index Buffer", file_name)),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            model::Mesh {
                vertex_buffer,
                index_buffer,
                num_elements: m.mesh.indices.len() as u32,
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes })
}

/// Load a binary glTF (`.glb`) as a single static, textured mesh. We read the
/// first mesh's first primitive (positions + UVs + indices) in its bind pose and
/// decode the material's base-color image to an RGBA texture. Normals, skinning,
/// and animation are ignored — `fox.wgsl` derives flat normals in the shader.
pub async fn load_glb(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    transform: glam::Mat4,
    texture_layout: &wgpu::BindGroupLayout,
    model_layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<GltfModel> {
    let bytes = load_binary(file_name).await?;
    let (document, buffers, images) = gltf::import_slice(&bytes)?;

    let mesh = document
        .meshes()
        .next()
        .ok_or_else(|| anyhow::anyhow!("{file_name}: no meshes"))?;
    let primitive = mesh
        .primitives()
        .next()
        .ok_or_else(|| anyhow::anyhow!("{file_name}: mesh has no primitives"))?;

    let reader = primitive.reader(|b| Some(buffers[b.index()].0.as_slice()));
    let positions = reader
        .read_positions()
        .ok_or_else(|| anyhow::anyhow!("{file_name}: primitive has no positions"))?;
    let mut tex_coords = reader
        .read_tex_coords(0)
        .ok_or_else(|| anyhow::anyhow!("{file_name}: primitive has no tex coords"))?
        .into_f32();
    let vertices: Vec<GltfVertex> = positions
        .map(|position| GltfVertex {
            position,
            tex_coords: tex_coords.next().unwrap_or([0.0, 0.0]),
        })
        .collect();
    // Some primitives (the Fox sample included) are non-indexed flat triangle
    // lists; synthesize a sequential index buffer so the draw path is uniform.
    let indices: Vec<u32> = match reader.read_indices() {
        Some(read) => read.into_u32().collect(),
        None => (0..vertices.len() as u32).collect(),
    };

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{file_name} Vertex Buffer")),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{file_name} Index Buffer")),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    // Decode the base-color texture to RGBA8 (the Fox's image is RGB PNG).
    let base_color = primitive
        .material()
        .pbr_metallic_roughness()
        .base_color_texture()
        .ok_or_else(|| anyhow::anyhow!("{file_name}: material has no base color texture"))?;
    let image = &images[base_color.texture().source().index()];
    let rgba: Vec<u8> = match image.format {
        gltf::image::Format::R8G8B8A8 => image.pixels.clone(),
        gltf::image::Format::R8G8B8 => image
            .pixels
            .chunks_exact(3)
            .flat_map(|c| [c[0], c[1], c[2], 255])
            .collect(),
        other => anyhow::bail!("{file_name}: unsupported base-color image format {other:?}"),
    };
    let texture =
        texture::Texture::from_rgba8(device, queue, image.width, image.height, &rgba, Some(file_name))?;

    let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: texture_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture.view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&texture.sampler),
            },
        ],
        label: Some("gltf_texture_bind_group"),
    });

    // Per-model transform uniform (group 3).
    let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(&format!("{file_name} Model Buffer")),
        contents: &utils::uniform_bytes(&transform),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: model_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: model_buffer.as_entire_binding(),
        }],
        label: Some("gltf_model_bind_group"),
    });

    Ok(GltfModel {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
        texture_bind_group,
        model_bind_group,
    })
}
