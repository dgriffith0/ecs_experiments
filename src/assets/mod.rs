use std::collections::HashMap;
use std::io::Cursor;

use glam::{Mat4, Quat, Vec3};
use wgpu::util::DeviceExt;

use crate::ecs::components::SkinnedMesh;
use crate::picking::Aabb;
use crate::render::texture;
use crate::scene::animation::{AnimationClip, Channel, ChannelData, Skeleton, Trs};
use crate::scene::gltf_model::{GltfModel, GltfVertex};
use crate::scene::model;
use crate::util as utils;

/// A parsed glTF kept on the CPU so many GPU instances can be spawned from it
/// without re-reading the file: geometry + decoded texture + the bind-group
/// layouts needed to build per-instance bind groups, plus optional skin data.
#[derive(bevy_ecs::prelude::Resource)]
pub struct GltfTemplate {
    vertices: Vec<GltfVertex>,
    indices: Vec<u32>,
    texture: texture::Texture,
    texture_layout: wgpu::BindGroupLayout,
    model_layout: wgpu::BindGroupLayout,
    pub skin: Option<SkinnedMesh>,
    pub local_aabb: Aabb,
}

impl GltfTemplate {
    /// Build a fresh GPU instance: its own `COPY_DST` vertex buffer (so the
    /// `animate` system can re-skin it each frame) and transform uniform; the
    /// index buffer and texture bind group are rebuilt cheaply from shared data.
    pub fn instantiate(&self, device: &wgpu::Device) -> GltfModel {
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gltf instance vertex buffer"),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gltf instance index buffer"),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        let texture_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.texture.sampler),
                },
            ],
            label: Some("gltf_texture_bind_group"),
        });
        let model_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gltf instance model buffer"),
            contents: &utils::uniform_bytes(&glam::Mat4::IDENTITY),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let model_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &self.model_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model_buffer.as_entire_binding(),
            }],
            label: Some("gltf_model_bind_group"),
        });
        GltfModel {
            vertex_buffer,
            index_buffer,
            num_indices: self.indices.len() as u32,
            texture_bind_group,
            model_bind_group,
            model_buffer,
        }
    }
}

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

/// Load an OBJ as geometry only, returning the model and its local bounding box.
/// The model is drawn solely as the unlit light marker, which reads vertex
/// positions, so we ignore materials/uvs/normals.
pub async fn load_model(
    file_name: &str,
    device: &wgpu::Device,
) -> anyhow::Result<(model::Model, Aabb)> {
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

    // Local AABB across all meshes (for picking the light marker).
    let aabb = Aabb::from_points(models.iter().flat_map(|m| {
        (0..m.mesh.positions.len() / 3).map(move |i| {
            Vec3::new(
                m.mesh.positions[i * 3],
                m.mesh.positions[i * 3 + 1],
                m.mesh.positions[i * 3 + 2],
            )
        })
    }));

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

    Ok((model::Model { meshes }, aabb))
}

/// Load a binary glTF (`.glb`) as a single textured mesh. Reads the first mesh's
/// first primitive (positions + UVs + indices) and decodes the base-color image.
/// If the file is skinned, also parses the skeleton + animation clips into a
/// [`SkinnedMesh`] (the vertex buffer is then `COPY_DST` so it can be re-skinned
/// each frame). Normals are ignored — `gltf.wgsl` derives flat normals.
pub async fn load_gltf_template(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture_layout: &wgpu::BindGroupLayout,
    model_layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<GltfTemplate> {
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
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or_else(|| anyhow::anyhow!("{file_name}: primitive has no positions"))?
        .collect();
    let tex_coords: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
        Some(tc) => tc.into_f32().collect(),
        None => vec![[0.0, 0.0]; positions.len()],
    };
    let vertices: Vec<GltfVertex> = positions
        .iter()
        .zip(&tex_coords)
        .map(|(&position, &tex_coords)| GltfVertex {
            position,
            tex_coords,
        })
        .collect();
    // Some primitives (the Fox sample included) are non-indexed flat triangle
    // lists; synthesize a sequential index buffer so the draw path is uniform.
    let indices: Vec<u32> = match reader.read_indices() {
        Some(read) => read.into_u32().collect(),
        None => (0..vertices.len() as u32).collect(),
    };

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

    // If the file is skinned, parse the skeleton + clips and keep the CPU data
    // needed to re-skin the mesh each frame.
    let skin = if let Some(gltf_skin) = document.skins().next() {
        let joints: Vec<[u16; 4]> = reader
            .read_joints(0)
            .ok_or_else(|| anyhow::anyhow!("{file_name}: skinned mesh has no JOINTS_0"))?
            .into_u16()
            .collect();
        let weights: Vec<[f32; 4]> = reader
            .read_weights(0)
            .ok_or_else(|| anyhow::anyhow!("{file_name}: skinned mesh has no WEIGHTS_0"))?
            .into_f32()
            .collect();
        let (skeleton, node_to_joint) = build_skeleton(&gltf_skin, &buffers);
        let clips = build_clips(&document, &buffers, &node_to_joint);
        Some(SkinnedMesh {
            base_positions: positions.iter().map(|&p| Vec3::from_array(p)).collect(),
            tex_coords,
            joints,
            weights,
            skeleton,
            clips,
        })
    } else {
        None
    };

    let local_aabb = Aabb::from_points(positions.iter().map(|p| Vec3::from_array(*p)));

    Ok(GltfTemplate {
        vertices,
        indices,
        texture,
        texture_layout: texture_layout.clone(),
        model_layout: model_layout.clone(),
        skin,
        local_aabb,
    })
}

/// Build a [`Skeleton`] from a glTF skin, plus a node-index → joint-index map
/// (used to resolve animation channel targets).
fn build_skeleton(
    skin: &gltf::Skin,
    buffers: &[gltf::buffer::Data],
) -> (Skeleton, HashMap<usize, usize>) {
    let joint_nodes: Vec<gltf::Node> = skin.joints().collect();
    let node_to_joint: HashMap<usize, usize> = joint_nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.index(), i))
        .collect();

    let locals: Vec<Trs> = joint_nodes
        .iter()
        .map(|n| {
            let (t, r, s) = n.transform().decomposed();
            Trs {
                translation: Vec3::from_array(t),
                rotation: Quat::from_array(r),
                scale: Vec3::from_array(s),
            }
        })
        .collect();

    // A joint's parent is whichever joint lists it as a child.
    let mut parents: Vec<Option<usize>> = vec![None; joint_nodes.len()];
    for (i, n) in joint_nodes.iter().enumerate() {
        for child in n.children() {
            if let Some(&cj) = node_to_joint.get(&child.index()) {
                parents[cj] = Some(i);
            }
        }
    }

    let reader = skin.reader(|b| Some(buffers[b.index()].0.as_slice()));
    let inverse_bind: Vec<Mat4> = match reader.read_inverse_bind_matrices() {
        Some(it) => it.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
        None => vec![Mat4::IDENTITY; joint_nodes.len()],
    };

    (
        Skeleton {
            parents,
            locals,
            inverse_bind,
        },
        node_to_joint,
    )
}

/// Parse all animation clips, keeping only channels that target a skin joint.
fn build_clips(
    document: &gltf::Document,
    buffers: &[gltf::buffer::Data],
    node_to_joint: &HashMap<usize, usize>,
) -> Vec<AnimationClip> {
    use gltf::animation::util::ReadOutputs;
    use gltf::animation::Property;

    document
        .animations()
        .map(|anim| {
            let mut duration = 0.0f32;
            let mut channels = Vec::new();
            for ch in anim.channels() {
                let Some(&joint) = node_to_joint.get(&ch.target().node().index()) else {
                    continue;
                };
                let reader = ch.reader(|b| Some(buffers[b.index()].0.as_slice()));
                let Some(times) = reader.read_inputs().map(|it| it.collect::<Vec<f32>>()) else {
                    continue;
                };
                if let Some(&last) = times.last() {
                    duration = duration.max(last);
                }
                let data = match (ch.target().property(), reader.read_outputs()) {
                    (Property::Translation, Some(ReadOutputs::Translations(it))) => {
                        ChannelData::Translation {
                            times,
                            values: it.map(Vec3::from_array).collect(),
                        }
                    }
                    (Property::Rotation, Some(ReadOutputs::Rotations(it))) => ChannelData::Rotation {
                        times,
                        values: it.into_f32().map(Quat::from_array).collect(),
                    },
                    (Property::Scale, Some(ReadOutputs::Scales(it))) => ChannelData::Scale {
                        times,
                        values: it.map(Vec3::from_array).collect(),
                    },
                    _ => continue,
                };
                channels.push(Channel { joint, data });
            }
            AnimationClip {
                name: anim.name().unwrap_or("").to_string(),
                duration,
                channels,
            }
        })
        .collect()
}
