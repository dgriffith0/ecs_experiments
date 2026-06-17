// Shared shader definitions, prepended to other shaders at module-creation time
// (WGSL has no `#include`, so we concatenate via `include_str!` in Rust). Keep
// this to type definitions only — the `@group`/`@binding` of each uniform lives
// in the shader that owns it, since the same struct binds to different slots in
// different pipelines.

struct Camera {
    view_pos: vec4<f32>,
    view_proj: mat4x4<f32>,
}

struct Light {
    position: vec3<f32>,
    color: vec3<f32>,
}
