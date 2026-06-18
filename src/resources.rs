use std::io::Cursor;

use wgpu::util::DeviceExt;

use crate::{model, texture};

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
