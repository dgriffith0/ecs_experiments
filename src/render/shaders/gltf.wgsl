// Static glTF model shader: samples a base-color texture and applies Blinn-Phong
// lighting. The mesh has no normals, so the fragment shader reconstructs a flat
// per-triangle normal from screen-space derivatives of the world position.
//
// `Camera` and `Light` are defined in common.wgsl (prepended at build time).

@group(0) @binding(0)
var t_base: texture_2d<f32>;
@group(0) @binding(1)
var s_base: sampler;

@group(1) @binding(0)
var<uniform> camera: Camera;

@group(2) @binding(0)
var<uniform> light: Light;

struct Model {
    matrix: mat4x4<f32>,
}
@group(3) @binding(0)
var<uniform> model: Model;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world = model.matrix * vec4<f32>(in.position, 1.0);
    out.clip_position = camera.view_proj * world;
    out.world_position = world.xyz;
    out.tex_coords = in.tex_coords;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let object_color = textureSample(t_base, s_base, in.tex_coords);

    // No vertex normals: derive a flat per-triangle normal from how the world
    // position changes across the fragment quad. Gives clean faceted shading.
    let normal = normalize(cross(dpdx(in.world_position), dpdy(in.world_position)));

    let light_dir = normalize(light.position - in.world_position);
    let view_dir = normalize(camera.view_pos.xyz - in.world_position);
    let half_dir = normalize(view_dir + light_dir);

    let ambient_strength = 0.35;
    let ambient_color = light.color * ambient_strength;

    let diffuse_strength = max(dot(normal, light_dir), 0.0);
    let diffuse_color = light.color * diffuse_strength;

    let specular_strength = pow(max(dot(normal, half_dir), 0.0), 32.0);
    let specular_color = specular_strength * light.color;

    let result = (ambient_color + diffuse_color + specular_color) * object_color.xyz;
    return vec4<f32>(result, object_color.a);
}
