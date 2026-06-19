// Skybox: draws a fullscreen triangle at the far plane and samples a cubemap by
// the world-space view ray reconstructed from the inverse view-projection.

struct Sky {
    // Inverse of the camera's full view-projection (with translation), so a clip
    // position unprojects back to an actual world-space point along the ray.
    inv_view_proj: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> sky: Sky;

@group(0) @binding(0)
var t_sky: texture_cube<f32>;
@group(0) @binding(1)
var s_sky: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    // Clip-space xy of this corner, interpolated for the fragment ray.
    @location(0) clip_xy: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    // Oversized fullscreen triangle: corners at (-1,-1), (3,-1), (-1,3).
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    let clip = uv * 2.0 - 1.0;
    var out: VertexOutput;
    // z = 1.0 keeps the sky at the far plane (wgpu clip depth range is [0, 1]).
    out.clip_position = vec4<f32>(clip, 1.0, 1.0);
    out.clip_xy = clip;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Unproject the near and far points of this pixel's ray, then take their
    // difference as the world-space direction to sample the cubemap with.
    let near = sky.inv_view_proj * vec4<f32>(in.clip_xy, 0.0, 1.0);
    let far = sky.inv_view_proj * vec4<f32>(in.clip_xy, 1.0, 1.0);
    let dir = normalize(far.xyz / far.w - near.xyz / near.w);
    return textureSample(t_sky, s_sky, dir);
}
