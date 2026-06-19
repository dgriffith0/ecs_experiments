// Composite the Slint UI (a premultiplied-alpha texture) over the scene with a
// fullscreen triangle. The texture is sampled as sRGB → linear; the pipeline
// uses premultiplied "over" blending (src=One, dst=OneMinusSrcAlpha).

@group(0) @binding(0) var t_ui: texture_2d<f32>;
@group(0) @binding(1) var s_ui: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    let xy = corners[i];
    var out: VsOut;
    out.pos = vec4<f32>(xy, 0.0, 1.0);
    // Clip XY → UV, flipping Y (texture origin is top-left).
    out.uv = vec2<f32>((xy.x + 1.0) * 0.5, (1.0 - xy.y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    return textureSample(t_ui, s_ui, in.uv);
}
