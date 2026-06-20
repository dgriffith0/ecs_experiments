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
use fast_poisson::Poisson2D;
use glam::{IVec3, UVec2, Vec3};
use noise::{Curve, Fbm, MultiFractal, NoiseFn, Perlin};
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
    /// Height-curve shaping (`0..1`). Higher widens the low, flat plains.
    pub flatness: f64,
    /// Height-curve shaping (`0..1`). Higher makes the high end steeper/peakier.
    pub peakiness: f64,
    /// Probabilistic block-layer dithering width (`0` = hard strata lines).
    pub layer_blend: f64,
    /// How many foxes to scatter across the surface; clamped to `[0, MAX_FOXES]`.
    pub fox_count: u32,
    /// Approximate number of Poisson-distributed trees; clamped to `[0, MAX_TREES]`.
    pub tree_count: u32,
    /// Fraction of the map that is forest (`0..1`). A low-frequency noise mask
    /// keeps trees in the densest regions, leaving the rest as clearings.
    pub forest_density: f64,
}

/// Most foxes the generator allows.
pub const MAX_FOXES: u32 = 64;
/// Most trees the generator allows.
pub const MAX_TREES: u32 = 400;

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
            flatness: 0.6,
            peakiness: 0.6,
            layer_blend: 0.12,
            fox_count: 5,
            tree_count: 60,
            forest_density: 0.5,
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
            flatness: self.flatness.clamp(0.0, 1.0),
            peakiness: self.peakiness.clamp(0.0, 1.0),
            layer_blend: self.layer_blend.clamp(0.0, 0.35),
            fox_count: self.fox_count.min(MAX_FOXES),
            tree_count: self.tree_count.min(MAX_TREES),
            forest_density: self.forest_density.clamp(0.0, 1.0),
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

fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Smooth 0→1 ramp between `e0` and `e1` (Hermite), clamped outside.
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Build the shaped height field: a base Fbm run through a `Curve` whose control
/// points form a flat-plains → hills → peaks spline, driven by `flatness`
/// (how far the flat low plateau extends) and `peakiness` (how steep the top
/// ramp into peaks is). Sampling it yields `t ∈ [0, 1]` mapped to column height.
fn shaped_noise(params: &TerrainParams) -> Curve<f64, Fbm<Perlin>, 2> {
    let fbm = Fbm::<Perlin>::new(params.seed)
        .set_octaves(params.octaves)
        .set_lacunarity(params.lacunarity)
        .set_persistence(params.persistence);

    // Control points live in the Fbm's *output* domain, which clusters near 0
    // (rarely past ±0.6), so the knee sits low. Higher flatness pushes the flat
    // plateau wider/lower; higher peakiness steepens the climb into the peaks.
    let flat_knee = lerp(-0.30, 0.15, params.flatness);
    let flat_out = lerp(0.18, 0.04, params.flatness);
    let hill_in = lerp(flat_knee, 0.6, 0.5);
    let hill_out = lerp(0.52, 0.34, params.peakiness);
    let peak_out = lerp(0.80, 1.0, params.peakiness);
    // Output is a [0, 1] height factor.
    Curve::new(fbm)
        .add_control_point(-1.0, 0.0)
        .add_control_point(flat_knee, flat_out)
        .add_control_point(hill_in, hill_out)
        .add_control_point(1.0, peak_out)
}

/// Solid-voxel height of the global terrain column at voxel coords `(gx, gz)`.
fn column_height(
    noise: &Curve<f64, Fbm<Perlin>, 2>,
    gx: i64,
    gz: i64,
    params: &TerrainParams,
) -> u32 {
    let t = noise
        .get([gx as f64 * params.frequency, gz as f64 * params.frequency])
        .clamp(0.0, 1.0);
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
    /// The sanitized parameters these heights were built with (max height for
    /// texture banding, seed + blend for the probabilistic layers).
    params: TerrainParams,
    /// Column heights, row-major: `heights[gz * extent + gx]`.
    heights: Vec<u32>,
}

impl Heightmap {
    /// Sample the noise once for every column, per the given parameters.
    pub fn generate(params: &TerrainParams) -> Self {
        let p = params.sanitized();
        let noise = shaped_noise(&p);
        let grid = p.grid_size;
        let extent = grid * CHUNK;
        let heights = (0..extent * extent)
            .map(|i| column_height(&noise, (i % extent) as i64, (i / extent) as i64, &p))
            .collect();
        Self {
            grid,
            extent,
            params: p,
            heights,
        }
    }

    pub fn grid(&self) -> u32 {
        self.grid
    }

    /// Columns per side (`grid * CHUNK`).
    pub fn extent(&self) -> u32 {
        self.extent
    }

    /// World-space centre of the top face of the column at global voxel coords
    /// `(gx, gz)`. The voxel cube spans `[gx·VS − half, gx·VS − half + VS]`, so the
    /// centre is half a voxel past the grid line — standing an entity here keeps it
    /// centred on the block instead of overhanging the edge.
    pub fn cell_center(&self, gx: i64, gz: i64) -> Vec3 {
        let half = grid_half(self.grid);
        let mid = VOXEL_SIZE * 0.5;
        let x = gx as f32 * VOXEL_SIZE - half + mid;
        let z = gz as f32 * VOXEL_SIZE - half - GRID_Z_PUSH + mid;
        Vec3::new(
            x,
            self.height(gx, gz) as f32 * VOXEL_SIZE + TERRAIN_BASE_Y,
            z,
        )
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

    /// The parameters this world was built with.
    pub fn params(&self) -> &TerrainParams {
        &self.params
    }

    /// Grid cell `(gx, gz)` whose voxel cube contains world `(x, z)`, clamped into
    /// the grid. Uses `floor` to match the cube spans in [`cell_center`].
    pub fn cell_coords(&self, world_x: f32, world_z: f32) -> (usize, usize) {
        let half = grid_half(self.grid);
        let max = self.extent as i64 - 1;
        let gx = (((world_x + half) / VOXEL_SIZE).floor() as i64).clamp(0, max);
        let gz = (((world_z + half + GRID_Z_PUSH) / VOXEL_SIZE).floor() as i64).clamp(0, max);
        (gx as usize, gz as usize)
    }

    /// `count` deterministic surface points scattered across the world (seeded by
    /// the world seed, so the same world always places them identically). Used to
    /// spawn entities like foxes on valid ground within the grid.
    pub fn scatter_surface(&self, count: u32) -> Vec<Vec3> {
        const MARGIN: f32 = 3.0;
        let (cx, cz) = world_center_xz(self.grid);
        let half = world_span(self.grid) / 2.0;
        let (x0, x1) = (cx - half + MARGIN, cx + half - MARGIN);
        let (z0, z1) = (cz - half + MARGIN, cz + half - MARGIN);
        (0..count)
            .map(|i| {
                let x = x0 + (x1 - x0) * hash01(i as i64, 0, 1, self.params.seed);
                let z = z0 + (z1 - z0) * hash01(i as i64, 0, 2, self.params.seed);
                Vec3::new(x, self.surface_y(x, z), z)
            })
            .collect()
    }

    /// Voxel-centre points for trees: an even, no-overlap **Poisson-disk** base
    /// (min spacing = `max(count radius, min_spacing)`) masked by a low-frequency
    /// noise field so points survive only in the densest regions — giving clumped
    /// **forests with clearings** rather than uniform coverage. `forest_density`
    /// (`0..1`) sets how much of the map is forest. Each point snaps to its voxel.
    pub fn poisson_surface(&self, count: u32, min_spacing: f32) -> Vec<Vec3> {
        if count == 0 {
            return Vec::new();
        }
        const MARGIN: f32 = 3.0;
        /// Spatial frequency of the forest mask — sets grove size (~1/FREQ metres).
        const FOREST_FREQ: f64 = 0.05;
        /// Soft width of the grove edge (in mask units) for a density falloff.
        const EDGE: f32 = 0.18;
        let (cx, cz) = world_center_xz(self.grid);
        let half = world_span(self.grid) / 2.0;
        let (x0, z0) = (cx - half + MARGIN, cz - half + MARGIN);
        let extent = (world_span(self.grid) - 2.0 * MARGIN).max(1.0);

        // A *dense*, no-overlap candidate base (footprint spacing). Forests are then
        // carved out of it, so grove cores are packed tight rather than uniform.
        let radius = min_spacing.max(1.0);
        // `coverage` (Forest slider) → how much of the map is grove; `amount` (Trees
        // slider) → how densely those groves fill, thinning the base probabilistically.
        let coverage = self.params.forest_density as f32;
        let amount = (count as f32 / 200.0).clamp(0.0, 1.0);
        // Grove mask: smooth low-frequency field, seeded apart from the terrain.
        let mask = Fbm::<Perlin>::new(self.params.seed.wrapping_add(0x5EED)).set_octaves(3);
        let threshold = 1.0 - coverage; // high-noise regions are groves

        let mut points: Vec<Vec3> = Poisson2D::new()
            .with_dimensions([extent as f64, extent as f64], radius as f64)
            .with_seed(self.params.seed as u64)
            .generate()
            .into_iter()
            .filter_map(|[px, pz]| {
                let (wx, wz) = (x0 + px as f32, z0 + pz as f32);
                let n = ((mask.get([wx as f64 * FOREST_FREQ, wz as f64 * FOREST_FREQ]) + 1.0) * 0.5)
                    as f32;
                // Density: full inside a grove, fading across its edge, zero in
                // clearings; `amount` thins the whole field.
                let keep = smoothstep(threshold - EDGE, threshold + EDGE, n) * amount;
                let (gx, gz) = self.cell_coords(wx, wz);
                if hash01(gx as i64, 7, gz as i64, self.params.seed) <= keep {
                    // Snap to the voxel centre so the placed model sits on the block.
                    Some(self.cell_center(gx as i64, gz as i64))
                } else {
                    None
                }
            })
            .collect();
        points.truncate(MAX_TREES as usize);
        points
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

/// Texture-array layers in ascending-elevation order, with the normalized height
/// each one is centred on: water → grass → dirt → stone.
const LAYER_BANDS: [(u32, f32); 4] = [(2, 0.12), (0, 0.38), (1, 0.63), (3, 0.88)];

/// Deterministic per-voxel value in `[0, 1)` from its global coords + seed. A hash
/// (not RNG) so a given world is reproducible and dithers differently per seed.
pub fn hash01(gx: i64, gy: i64, gz: i64, seed: u32) -> f32 {
    let mut h = seed as u64 ^ 0x9E3779B97F4A7C15;
    for v in [gx as u64, gy as u64, gz as u64] {
        h ^= v.wrapping_mul(0xD1B54A32D192ED03);
        h = h.wrapping_mul(0xCA4BCAA75EC3F625);
        h ^= h >> 29;
    }
    ((h >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Probabilistically pick a texture layer for a voxel from its elevation. Each
/// layer's weight is a triangular falloff (width `layer_blend`) around its band
/// centre, so near a boundary the two layers interleave into a dithered blend
/// instead of a hard line. `layer_blend == 0` collapses to nearest-band strata.
fn layer_at(gx: i64, gy: i64, gz: i64, params: &TerrainParams) -> u32 {
    let f = (gy as f32 / params.max_height as f32).clamp(0.0, 1.0);
    let blend = params.layer_blend as f32;

    let mut weights = [0.0f32; 4];
    let mut total = 0.0;
    for (k, (_, centre)) in LAYER_BANDS.iter().enumerate() {
        // Distance from the band centre, but the bottom/top bands extend outward
        // (clamp distance below water's / above stone's centre) so there are no gaps.
        let d = match k {
            0 => (f - centre).max(0.0),
            3 => (centre - f).max(0.0),
            _ => (f - centre).abs(),
        };
        let w = (1.0 - d / blend.max(1e-4)).max(0.0);
        weights[k] = w;
        total += w;
    }
    if total <= 0.0 {
        // `f` fell between bands with blend too small to reach: snap to nearest.
        return LAYER_BANDS
            .iter()
            .min_by(|a, b| (f - a.1).abs().total_cmp(&(f - b.1).abs()))
            .unwrap()
            .0;
    }

    let mut r = hash01(gx, gy, gz, params.seed) * total;
    for (k, (layer, _)) in LAYER_BANDS.iter().enumerate() {
        r -= weights[k];
        if r < 0.0 {
            return *layer;
        }
    }
    LAYER_BANDS[3].0
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
                layers[i] = layer_at(gx, y as i64, gz, &heightmap.params);
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
