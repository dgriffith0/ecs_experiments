// Voxel shader: samples a texture array (one layer per chunk) and applies
// world-space Blinn-Phong lighting using the per-face geometric normal.

// `Camera` and `Light` are defined in common.wgsl (prepended at build time).
@group(1) @binding(0)
var<uniform> camera: Camera;

@group(2) @binding(0)
var<uniform> light: Light;

struct VoxelSettings {
    ao_enabled: u32,
}
@group(3) @binding(0)
var<uniform> settings: VoxelSettings;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    @location(3) layer: u32,
    @location(4) ao: f32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) tex_coords: vec2<f32>,
    // The layer is constant per chunk, so don't interpolate it.
    @location(3) @interpolate(flat) layer: u32,
    // Baked ambient occlusion, smoothly interpolated across the face.
    @location(4) ao: f32,
}

@vertex
fn vs_main(model: VertexInput) -> VertexOutput {
    // Vertex positions are already in world space (the chunk offset is baked in).
    var out: VertexOutput;
    out.clip_position = camera.view_proj * vec4<f32>(model.position, 1.0);
    out.world_position = model.position;
    out.world_normal = model.normal;
    out.tex_coords = model.tex_coords;
    out.layer = model.layer;
    out.ao = model.ao;
    return out;
}

// Fragment shader

@group(0) @binding(0)
var t_array: texture_2d_array<f32>;
@group(0) @binding(1)
var s_array: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color: vec4<f32> =
        textureSample(t_array, s_array, in.tex_coords, i32(in.layer));

    let normal = normalize(in.world_normal);
    let light_dir = normalize(light.position - in.world_position);
    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    let half_dir = normalize(view_dir + light_dir);

    // Higher ambient gives the baked AO headroom to read on faces the moving
    // point light leaves in shadow (otherwise those faces are already ~black).
    let ambient_strength = 0.35;
    let ambient_color = light.color * ambient_strength;

    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    // Ambient occlusion darkens the whole lit result so it reads clearly.
    // The toggle decides whether the baked AO is applied or ignored (fully lit).
    let ao = select(1.0, in.ao, settings.ao_enabled != 0u);
    let result =
        (ambient_color + diffuse_color + specular_color) * object_color.xyz * ao;

    return vec4<f32>(result, object_color.a);
}
