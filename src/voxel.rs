//! Voxel chunk generation and meshing.
//!
//! Each chunk is a 32³ slice of a continuous landscape. A single global 2D
//! height map (Perlin fractal noise sampled by world-voxel coordinate) decides
//! how tall each (x,z) column is, and every column is filled solid from the
//! chunk floor up to that height. We mesh the result with `block-mesh`'s
//! `visible_block_faces`, which emits a quad only for faces that border an empty
//! voxel — so interior/occluded faces are culled. Each voxel's texture-array
//! layer is chosen by its elevation (water → grass → dirt → stone, low → high),
//! giving clean horizontal strata across the whole terrain. Because the height
//! map is global and the chunks' padding borders sample it too, neighbouring
//! chunks line up seamlessly with no doubled geometry at the seams.

use block_mesh::ndshape::{ConstShape, ConstShape3u32};
use block_mesh::{
    visible_block_faces, UnitQuadBuffer, Voxel, VoxelVisibility, RIGHT_HANDED_Y_UP_CONFIG,
};
use encase::ShaderType;
use glam::{IVec3, UVec2, Vec3};
use noise::{Fbm, NoiseFn, Perlin};
use wgpu::util::DeviceExt;

/// Number of texture layers in the array texture.
pub const NUM_TEXTURE_LAYERS: u32 = 4;

/// Edge length of a chunk in voxels.
const CHUNK: u32 = 32;
/// World-space size of a single voxel. Voxels are small so a 4×4 grid of 32³
/// chunks fits comfortably within the camera's view (znear 0.1, zfar 100).
const VOXEL_SIZE: f32 = 0.1;

/// Seed for the terrain height map. Fixed so the world is deterministic.
const TERRAIN_SEED: u32 = 42;
/// Horizontal frequency of the height map, in world-voxel units. Smaller values
/// give broader, gentler hills; larger values give choppier terrain.
const NOISE_SCALE: f64 = 0.03;
/// Tallest possible column. Must stay `<= CHUNK` so columns fit the interior.
const MAX_TERRAIN_HEIGHT: u32 = 24;
/// Shortest possible column, so valleys keep a solid floor instead of holes.
const MIN_TERRAIN_HEIGHT: u32 = 2;

// The meshing kernel needs a 1-voxel border of padding so it can test the
// neighbours of boundary voxels without reading out of bounds, hence 32 + 2.
type ChunkShape = ConstShape3u32<{ CHUNK + 2 }, { CHUNK + 2 }, { CHUNK + 2 }>;

/// A voxel that is either empty or solid. Solid voxels are opaque so any face
/// between two solid voxels is culled.
#[derive(Clone, Copy, Eq, PartialEq)]
struct BoolVoxel(bool);

const EMPTY: BoolVoxel = BoolVoxel(false);
const FILLED: BoolVoxel = BoolVoxel(true);

impl Voxel for BoolVoxel {
    fn get_visibility(&self) -> VoxelVisibility {
        if *self == EMPTY {
            VoxelVisibility::Empty
        } else {
            VoxelVisibility::Opaque
        }
    }
}

/// A single mesh vertex for a voxel face. `layer` selects the texture-array
/// layer and is constant across a chunk. `ao` is a baked ambient-occlusion
/// brightness multiplier (1.0 = fully lit, lower = occluded), interpolated
/// across the face.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VoxelVertex {
    position: [f32; 3],
    normal: [f32; 3],
    tex_coords: [f32; 2],
    layer: u32,
    ao: f32,
}

impl VoxelVertex {
    const ATTRS: [wgpu::VertexAttribute; 5] = wgpu::vertex_attr_array![
        0 => Float32x3, // position
        1 => Float32x3, // normal
        2 => Float32x2, // tex_coords
        3 => Uint32,    // texture-array layer
        4 => Float32,   // baked ambient-occlusion brightness
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRS,
        }
    }
}

/// GPU buffers for one meshed chunk.
pub struct VoxelChunk {
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub num_indices: u32,
}

/// Runtime render settings for the voxel pass, mirrored as a uniform in
/// `voxel.wgsl`. The baked ambient occlusion is always present in the mesh;
/// this flag just decides whether the shader applies it.
#[derive(Debug, Copy, Clone, ShaderType)]
pub struct VoxelSettings {
    /// 1 = apply baked ambient occlusion, 0 = ignore it (fully lit).
    pub ao_enabled: u32,
    // Padding so the uniform buffer is a full 16 bytes. WGSL rounds a
    // uniform-address struct up to 16, and Metal silently mis-binds a smaller
    // buffer (the flag then reads as 0, so AO never shows and the toggle looks
    // dead). Scalar u32s pack tightly to 4+4+4+4 = 16 (an array would get a
    // 16-byte std140 stride and blow the size up).
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl VoxelSettings {
    pub fn new(ao_enabled: bool) -> Self {
        Self {
            ao_enabled: ao_enabled as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }

    /// Flip AO on/off.
    pub fn toggle(&mut self) {
        self.ao_enabled ^= 1;
    }
}

/// Per-level brightness for vertex ambient occlusion. Index = number of
/// unoccluded contributions (0 = deepest corner, 3 = fully open). Tune to taste.
const AO_BRIGHTNESS: [f32; 4] = [0.2, 0.45, 0.7, 1.0];

/// Classic voxel AO: given the two edge neighbours and the corner neighbour
/// (all on the air side of a face), return an occlusion level 0..3.
fn ao_level(side1: bool, side2: bool, corner: bool) -> usize {
    if side1 && side2 {
        0 // two solid sides fully box in the corner
    } else {
        3 - (side1 as usize + side2 as usize + corner as usize)
    }
}

/// Two triangles for a quad, choosing the diagonal via `flip` and preserving the
/// face winding via `ccw`. The non-flip arms reproduce block-mesh's own index
/// patterns; the flip arms are the mirrored triangulation used to keep AO
/// gradients from leaking light across the wrong diagonal.
fn quad_indices(start: u32, ccw: bool, flip: bool) -> [u32; 6] {
    match (ccw, flip) {
        (true, false) => [start, start + 1, start + 2, start + 1, start + 3, start + 2],
        (true, true) => [start, start + 1, start + 3, start, start + 3, start + 2],
        (false, false) => [start, start + 2, start + 1, start + 1, start + 2, start + 3],
        (false, true) => [start, start + 3, start + 1, start, start + 2, start + 3],
    }
}

/// Pick a texture-array layer from a voxel's absolute height. Banding the layers
/// by elevation gives the terrain clean horizontal strata.
fn layer_for_height(y: u32) -> u32 {
    let f = y as f32 / MAX_TERRAIN_HEIGHT as f32;
    if f < 0.25 {
        2 // water
    } else if f < 0.50 {
        0 // grass
    } else if f < 0.75 {
        1 // dirt
    } else {
        3 // stone
    }
}

/// Generate one chunk of the global height-mapped terrain at `world_origin`,
/// mesh its visible faces, and upload the result to the GPU. `chunk_voxel_offset`
/// is this chunk's position in the global voxel grid (so the shared `noise`
/// field is sampled in world coordinates and tiles seamlessly across chunks).
pub fn generate_chunk(
    device: &wgpu::Device,
    world_origin: Vec3,
    chunk_voxel_offset: UVec2,
    grid_voxel_extent: u32,
    noise: &Fbm<Perlin>,
) -> VoxelChunk {
    // Fill the padded volume from the global height map. We sample every column
    // including the 1-voxel border: that border mirrors the neighbouring chunk's
    // edge columns, so block-mesh culls the faces at chunk seams. The top
    // (y == CHUNK+1) and bottom (y == 0) stay empty so surface/floor faces emit.
    // Each solid voxel records its texture layer, indexed the same way as `voxels`.
    let mut voxels = [EMPTY; ChunkShape::SIZE as usize];
    let mut layers = vec![0u32; ChunkShape::SIZE as usize];
    for z in 0..CHUNK + 2 {
        for x in 0..CHUNK + 2 {
            // Global voxel coords of this column; `-1` removes the padding offset.
            let gx = chunk_voxel_offset.x as i64 + x as i64 - 1;
            let gz = chunk_voxel_offset.y as i64 + z as i64 - 1;
            // Padding columns beyond the rendered grid have no real neighbour
            // chunk. Leaving them empty makes block-mesh emit the terrain's
            // outer boundary walls instead of culling them against the infinite
            // height map. Interior seams still mirror their neighbour (their
            // padding maps to an in-range column) and stay culled.
            let extent = grid_voxel_extent as i64;
            if gx < 0 || gz < 0 || gx >= extent || gz >= extent {
                continue;
            }
            let n = noise.get([gx as f64 * NOISE_SCALE, gz as f64 * NOISE_SCALE]); // -1..1
            let t = ((n + 1.0) * 0.5).clamp(0.0, 1.0); // 0..1
            let height =
                MIN_TERRAIN_HEIGHT + (t * (MAX_TERRAIN_HEIGHT - MIN_TERRAIN_HEIGHT) as f64) as u32;
            for y in 1..=height {
                let i = ChunkShape::linearize([x, y, z]) as usize;
                voxels[i] = FILLED;
                layers[i] = layer_for_height(y);
            }
        }
    }

    let mut buffer = UnitQuadBuffer::new();
    visible_block_faces(
        &voxels,
        &ChunkShape {},
        [0; 3],
        [CHUNK + 1; 3],
        &RIGHT_HANDED_Y_UP_CONFIG.faces,
        &mut buffer,
    );

    let mut vertices: Vec<VoxelVertex> = Vec::with_capacity(buffer.num_quads() * 4);
    let mut indices: Vec<u32> = Vec::with_capacity(buffer.num_quads() * 6);

    // Solidity sampler for ambient occlusion. Samples outside the padded volume
    // count as empty so corner lookups near the border never index out of range.
    let is_solid = |c: IVec3| -> bool {
        let max = (CHUNK + 1) as i32;
        if c.cmplt(IVec3::ZERO).any() || c.cmpgt(IVec3::splat(max)).any() {
            return false;
        }
        voxels[ChunkShape::linearize([c.x as u32, c.y as u32, c.z as u32]) as usize] == FILLED
    };

    for (group, face) in buffer
        .groups
        .into_iter()
        .zip(RIGHT_HANDED_Y_UP_CONFIG.faces)
    {
        for unit_quad in group {
            // Look up the texture layer of the voxel this face belongs to.
            let layer = layers[ChunkShape::linearize(unit_quad.minimum) as usize];

            let quad = unit_quad.into();
            let positions = face.quad_mesh_positions(&quad, VOXEL_SIZE);
            let normals = face.quad_mesh_normals();
            let uvs = face.tex_coords(RIGHT_HANDED_Y_UP_CONFIG.u_flip_face, true, &quad);

            // Compute per-vertex ambient occlusion. The corners are returned in
            // u/v order (0,0),(1,0),(0,1),(1,1) — matching positions/normals/uvs.
            // For each corner we sample the two edge neighbours and the corner
            // neighbour on the air side (air = solid voxel + face normal).
            // block-mesh returns its own (older) glam types, so convert via arrays.
            let iv = |a: [u32; 3]| IVec3::new(a[0] as i32, a[1] as i32, a[2] as i32);
            let corners = face.quad_corners(&quad);
            let c0 = iv(corners[0].to_array());
            let p = iv(unit_quad.minimum);
            let n = face.signed_normal().to_array();
            let air = p + IVec3::from_array(n);
            let u = iv(corners[1].to_array()) - c0;
            let v = iv(corners[2].to_array()) - c0;
            const SIGNS: [(i32, i32); 4] = [(-1, -1), (1, -1), (-1, 1), (1, 1)];
            let mut ao = [0.0f32; 4];
            for (i, (su, sv)) in SIGNS.iter().enumerate() {
                let s1 = air + u * *su;
                let s2 = air + v * *sv;
                let cor = air + u * *su + v * *sv;
                ao[i] = AO_BRIGHTNESS[ao_level(is_solid(s1), is_solid(s2), is_solid(cor))];
            }

            // Pick the diagonal so the AO gradient interpolates without leaking
            // light, keeping block-mesh's winding (ccw pattern starts s, s+1, …).
            let start = vertices.len() as u32;
            let base = face.quad_mesh_indices(start);
            let ccw = base[1] == start + 1;
            let flip = ao[0] + ao[3] < ao[1] + ao[2];
            indices.extend_from_slice(&quad_indices(start, ccw, flip));

            for (((position, normal), uv), ao) in positions.iter().zip(&normals).zip(&uvs).zip(&ao)
            {
                vertices.push(VoxelVertex {
                    // Shift out the 1-voxel padding offset, then place in world.
                    position: [
                        position[0] - VOXEL_SIZE + world_origin.x,
                        position[1] - VOXEL_SIZE + world_origin.y,
                        position[2] - VOXEL_SIZE + world_origin.z,
                    ],
                    normal: *normal,
                    tex_coords: *uv,
                    layer,
                    ao: *ao,
                });
            }
        }
    }

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Voxel Vertex Buffer"),
        contents: bytemuck::cast_slice(&vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });
    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Voxel Index Buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    VoxelChunk {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32,
    }
}

/// Build a flat `grid × grid` plane of chunks forming one seamless landscape
/// driven by a single global height map.
pub fn generate_chunk_grid(device: &wgpu::Device, grid: u32) -> Vec<VoxelChunk> {
    let noise = Fbm::<Perlin>::new(TERRAIN_SEED);
    // Chunks touch exactly so the terrain is continuous across borders.
    let spacing = CHUNK as f32 * VOXEL_SIZE;
    let half = (grid as f32 - 1.0) * spacing / 2.0;
    let chunk_extent = CHUNK as f32 * VOXEL_SIZE;

    let grid_voxel_extent = grid * CHUNK;
    let mut chunks = Vec::with_capacity((grid * grid) as usize);
    for cz in 0..grid {
        for cx in 0..grid {
            let origin = Vec3::new(
                cx as f32 * spacing - half,
                // Drop the grid below the camera's eye line and centre it on y=0.
                -chunk_extent / 2.0,
                // Push the grid in front of the camera (which looks toward -Z).
                cz as f32 * spacing - half - 6.0,
            );
            let chunk_voxel_offset = UVec2::new(cx * CHUNK, cz * CHUNK);
            chunks.push(generate_chunk(
                device,
                origin,
                chunk_voxel_offset,
                grid_voxel_extent,
                &noise,
            ));
        }
    }
    chunks
}
