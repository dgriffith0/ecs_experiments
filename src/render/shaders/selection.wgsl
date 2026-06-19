// Wireframe selection box: draws unit-cube edges transformed to a world AABB.

struct Sel {
    mvp: mat4x4<f32>,
    color: vec4<f32>,
}
@group(0) @binding(0) var<uniform> sel: Sel;

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return sel.mvp * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return sel.color;
}
