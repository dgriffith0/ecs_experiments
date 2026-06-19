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
use noise::{Fbm, MultiFractal, NoiseFn, Perlin};
use wgpu::util::DeviceExt;

/// Number of texture layers in the array texture.
pub const NUM_TEXTURE_LAYERS: u32 = 4;

/// Edge length of a chunk in voxels.
const CHUNK: u32 = 32;
/// World-space size of a single voxel: 1 unit = 1 metre, so one block is 1 m
/// (Minecraft-style). A 4×4 grid of 32³ chunks is then a 128 m × 128 m world.
const VOXEL_SIZE: f32 = 1.0;

/// Shortest possible column, so valleys keep a solid floor instead of holes.
const MIN_TERRAIN_HEIGHT: u32 = 2;
/// Largest world (chunks per side) the generator allows.
pub const MAX_GRID: u32 = 8;

/// Tunable terrain-generation parameters: the Fbm noise knobs plus world size.
/// Stored as a resource; the [`Heightmap`] is (re)built from these.
#[derive(bevy_ecs::prelude::Resource, Clone, Copy)]
pub struct TerrainParams {
    pub seed: u32,
    /// Horizontal frequency of the noise, in world-voxel units. Smaller is
    /// broader/gentler hills; larger is choppier.
    pub frequency: f64,
    pub octaves: usize,
    pub lacunarity: f64,
    pub persistence: f64,
    /// Tallest possible column; clamped to `(MIN_TERRAIN_HEIGHT, CHUNK]`.
    pub max_height: u32,
    /// Chunks per side; clamped to `[1, MAX_GRID]`.
    pub grid_size: u32,
}

impl Default for TerrainParams {
    /// Reproduces the original world: `Fbm::new(42)` defaults + a `×0.03` scale.
    fn default() -> Self {
        Self {
            seed: 42,
            frequency: 0.03,
            octaves: 6,
            lacunarity: std::f64::consts::PI * 2.0 / 3.0,
            persistence: 0.5,
            max_height: 24,
            grid_size: 4,
        }
    }
}

impl TerrainParams {
    /// Clamp every field to a valid range so a stray slider value can't produce
    /// columns that overflow a chunk or a zero-size world.
    fn sanitized(&self) -> Self {
        Self {
            octaves: self.octaves.clamp(1, 32),
            max_height: self.max_height.clamp(MIN_TERRAIN_HEIGHT + 1, CHUNK),
            grid_size: self.grid_size.clamp(1, MAX_GRID),
            ..*self
        }
    }
}

/// World-space Y of every chunk's floor. Chunks are centred on the origin
/// vertically, so the terrain surface sits between this and `+MAX_TERRAIN_HEIGHT`.
const TERRAIN_BASE_Y: f32 = -(CHUNK as f32 * VOXEL_SIZE) / 2.0;
/// Distance the whole grid is pushed along -Z. 0 keeps it centred on the origin.
const GRID_Z_PUSH: f32 = 0.0;

/// Half the world-space span between the first and last chunk origins; used to
/// centre the grid on the origin in X/Z. Also converts world↔global-voxel coords.
fn grid_half(grid: u32) -> f32 {
    (grid as f32 - 1.0) * CHUNK as f32 * VOXEL_SIZE / 2.0
}

/// Solid-voxel height of the global terrain column at voxel coords `(gx, gz)`.
/// Shared by chunk generation and surface queries so they always agree.
fn column_height(noise: &Fbm<Perlin>, gx: i64, gz: i64, params: &TerrainParams) -> u32 {
    let n = noise.get([gx as f64 * params.frequency, gz as f64 * params.frequency]); // -1..1
    let t = ((n + 1.0) * 0.5).clamp(0.0, 1.0); // 0..1
    MIN_TERRAIN_HEIGHT + (t * (params.max_height - MIN_TERRAIN_HEIGHT) as f64) as u32
}

/// Precomputed terrain: the solid-voxel height of every column in a `grid`×`grid`
/// world. Built once from the noise (see `Heightmap::generate`) and stored as a
/// resource, so chunk meshing and picking are plain array lookups — no per-query
/// noise sampling.
#[derive(bevy_ecs::prelude::Resource)]
pub struct Heightmap {
    grid: u32,
    /// Columns per side (`grid * CHUNK`).
    extent: u32,
    /// Tallest column these heights were built with (for texture banding).
    max_height: u32,
    /// Column heights, row-major: `heights[gz * extent + gx]`.
    heights: Vec<u32>,
}

impl Heightmap {
    /// Sample the noise once for every column, per the given parameters.
    pub fn generate(params: &TerrainParams) -> Self {
        let p = params.sanitized();
        let noise = Fbm::<Perlin>::new(p.seed)
            .set_octaves(p.octaves)
            .set_lacunarity(p.lacunarity)
            .set_persistence(p.persistence);
        let grid = p.grid_size;
        let extent = grid * CHUNK;
        let heights = (0..extent * extent)
            .map(|i| column_height(&noise, (i % extent) as i64, (i / extent) as i64, &p))
            .collect();
        Self {
            grid,
            extent,
            max_height: p.max_height,
            heights,
        }
    }

    pub fn grid(&self) -> u32 {
        self.grid
    }

    /// Solid-voxel height of the column at global voxel coords `(gx, gz)`.
    /// Columns outside the grid report `MIN_TERRAIN_HEIGHT`.
    pub fn height(&self, gx: i64, gz: i64) -> u32 {
        let e = self.extent as i64;
        if gx < 0 || gz < 0 || gx >= e || gz >= e {
            MIN_TERRAIN_HEIGHT
        } else {
            self.heights[(gz * e + gx) as usize]
        }
    }

    /// World-space Y of the terrain surface (top of the tallest solid voxel) at
    /// world `(x, z)`.
    pub fn surface_y(&self, world_x: f32, world_z: f32) -> f32 {
        let half = grid_half(self.grid);
        let gx = ((world_x + half) / VOXEL_SIZE).round() as i64;
        let gz = ((world_z + half + GRID_Z_PUSH) / VOXEL_SIZE).round() as i64;
        self.height(gx, gz) as f32 * VOXEL_SIZE + TERRAIN_BASE_Y
    }
}

/// World-space vertical extent any terrain can occupy, for bounding ray casts.
/// Conservative (a full chunk tall) so it holds for any `max_height`.
pub fn terrain_y_bounds() -> (f32, f32) {
    (TERRAIN_BASE_Y, TERRAIN_BASE_Y + CHUNK as f32 * VOXEL_SIZE)
}

/// Total world-space span (metres) of a `grid`×`grid` world.
pub fn world_span(grid: u32) -> f32 {
    grid as f32 * CHUNK as f32 * VOXEL_SIZE
}

/// World-space X/Z centre of the terrain grid (invariant, ~`(16, 16)`).
pub fn world_center_xz(grid: u32) -> (f32, f32) {
    let half = grid_half(grid);
    let mid = (grid * CHUNK / 2) as f32 * VOXEL_SIZE;
    (mid - half, mid - half - GRID_Z_PUSH)
}

/// The grid coordinate `(gx, gy, gz)` and world-space cube `(min, max)` of the
/// voxel cell containing `point`, snapped to the same grid the meshes use.
pub fn voxel_cell_at(point: Vec3, grid: u32) -> (glam::IVec3, Vec3, Vec3) {
    let half = grid_half(grid);
    let gx = ((point.x + half) / VOXEL_SIZE).floor();
    let gy = ((point.y - TERRAIN_BASE_Y) / VOXEL_SIZE).floor();
    let gz = ((point.z + half + GRID_Z_PUSH) / VOXEL_SIZE).floor();
    let min = Vec3::new(
        gx * VOXEL_SIZE - half,
        gy * VOXEL_SIZE + TERRAIN_BASE_Y,
        gz * VOXEL_SIZE - half - GRID_Z_PUSH,
    );
    (
        glam::IVec3::new(gx as i32, gy as i32, gz as i32),
        min,
        min + Vec3::splat(VOXEL_SIZE),
    )
}

/// Map a world `(x, z)` to its chunk grid coordinate, or `None` if outside the grid.
pub fn chunk_coord_at(world_x: f32, world_z: f32, grid: u32) -> Option<(u32, u32)> {
    let half = grid_half(grid);
    let gx = ((world_x + half) / VOXEL_SIZE).floor() as i64;
    let gz = ((world_z + half + GRID_Z_PUSH) / VOXEL_SIZE).floor() as i64;
    let extent = (grid * CHUNK) as i64;
    if gx < 0 || gz < 0 || gx >= extent || gz >= extent {
        return None;
    }
    Some(((gx / CHUNK as i64) as u32, (gz / CHUNK as i64) as u32))
}

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

/// GPU buffers for one meshed chunk. One ECS entity per chunk.
#[derive(bevy_ecs::prelude::Component)]
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
fn layer_for_height(y: u32, max_height: u32) -> u32 {
    let f = y as f32 / max_height as f32;
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
    heightmap: &Heightmap,
) -> VoxelChunk {
    let grid_voxel_extent = heightmap.extent;
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
            let height = heightmap.height(gx, gz);
            for y in 1..=height {
                let i = ChunkShape::linearize([x, y, z]) as usize;
                voxels[i] = FILLED;
                layers[i] = layer_for_height(y, heightmap.max_height);
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

/// Build a flat plane of chunks forming one seamless landscape, meshed from the
/// precomputed `Heightmap`.
pub fn generate_chunk_grid(device: &wgpu::Device, heightmap: &Heightmap) -> Vec<VoxelChunk> {
    let grid = heightmap.grid;
    // Chunks touch exactly so the terrain is continuous across borders.
    let spacing = CHUNK as f32 * VOXEL_SIZE;
    let half = grid_half(grid);

    let mut chunks = Vec::with_capacity((grid * grid) as usize);
    for cz in 0..grid {
        for cx in 0..grid {
            let origin = Vec3::new(
                cx as f32 * spacing - half,
                TERRAIN_BASE_Y,
                cz as f32 * spacing - half - GRID_Z_PUSH,
            );
            let chunk_voxel_offset = UVec2::new(cx * CHUNK, cz * CHUNK);
            chunks.push(generate_chunk(
                device,
                origin,
                chunk_voxel_offset,
                heightmap,
            ));
        }
    }
    chunks
}
