//! Hand-rolled glTF skeletal-animation runtime: sample animation clips into
//! joint transforms, build skinning matrices, and CPU linear-blend skin a mesh.
//! Renderer-agnostic — pure glam math, no wgpu. Assumes LINEAR interpolation
//! (lerp/slerp), which is all the Fox sample uses.

use glam::{Mat4, Quat, Vec3};

/// Local transform of a joint (translation / rotation / scale).
#[derive(Clone, Copy)]
pub struct Trs {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Trs {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

/// A skeleton: joints in skin order, each with a parent joint index (within the
/// joint set, or `None` for a root), a base local transform, and an inverse bind
/// matrix.
#[derive(Clone)]
pub struct Skeleton {
    pub parents: Vec<Option<usize>>,
    pub locals: Vec<Trs>,
    pub inverse_bind: Vec<Mat4>,
}

/// Keyframe data for one animated channel (one joint, one property). LINEAR.
#[derive(Clone)]
pub enum ChannelData {
    Translation { times: Vec<f32>, values: Vec<Vec3> },
    Rotation { times: Vec<f32>, values: Vec<Quat> },
    Scale { times: Vec<f32>, values: Vec<Vec3> },
}

#[derive(Clone)]
pub struct Channel {
    pub joint: usize,
    pub data: ChannelData,
}

#[derive(Clone)]
pub struct AnimationClip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

impl AnimationClip {
    /// Sample every channel at `time`, starting from the skeleton's base locals
    /// (so joints without a channel for a given property keep their bind value).
    pub fn sample(&self, skeleton: &Skeleton, time: f32) -> Vec<Trs> {
        let mut locals = skeleton.locals.clone();
        for ch in &self.channels {
            match &ch.data {
                ChannelData::Translation { times, values } => {
                    locals[ch.joint].translation = sample_vec3(times, values, time);
                }
                ChannelData::Rotation { times, values } => {
                    locals[ch.joint].rotation = sample_quat(times, values, time);
                }
                ChannelData::Scale { times, values } => {
                    locals[ch.joint].scale = sample_vec3(times, values, time);
                }
            }
        }
        locals
    }
}

/// Skinning matrix per joint: `global_joint_transform * inverse_bind`.
pub fn joint_matrices(skeleton: &Skeleton, locals: &[Trs]) -> Vec<Mat4> {
    let n = skeleton.parents.len();
    let mut global: Vec<Option<Mat4>> = vec![None; n];
    (0..n)
        .map(|i| {
            resolve_global(i, &skeleton.parents, locals, &mut global) * skeleton.inverse_bind[i]
        })
        .collect()
}

/// Resolve a joint's global transform via its parent chain, memoizing results so
/// the order of joints in the skin list doesn't matter.
fn resolve_global(
    i: usize,
    parents: &[Option<usize>],
    locals: &[Trs],
    global: &mut Vec<Option<Mat4>>,
) -> Mat4 {
    if let Some(g) = global[i] {
        return g;
    }
    let local = locals[i].matrix();
    let g = match parents[i] {
        Some(p) => resolve_global(p, parents, locals, global) * local,
        None => local,
    };
    global[i] = Some(g);
    g
}

/// Linear-blend skin `base` positions by per-vertex joint indices + weights.
pub fn skin_positions(
    base: &[Vec3],
    joints: &[[u16; 4]],
    weights: &[[f32; 4]],
    joint_mats: &[Mat4],
) -> Vec<Vec3> {
    base.iter()
        .enumerate()
        .map(|(i, &pos)| {
            let (j, w) = (joints[i], weights[i]);
            let mut skinned = Vec3::ZERO;
            for k in 0..4 {
                if w[k] > 0.0 {
                    skinned += w[k] * joint_mats[j[k] as usize].transform_point3(pos);
                }
            }
            skinned
        })
        .collect()
}

/// Find the keyframe segment `(i0, i1, alpha)` for `t`, clamping at the ends.
fn segment(times: &[f32], t: f32) -> (usize, usize, f32) {
    let last = times.len().saturating_sub(1);
    if times.is_empty() || t <= times[0] {
        return (0, 0, 0.0);
    }
    if t >= times[last] {
        return (last, last, 0.0);
    }
    let i1 = times.partition_point(|&x| x <= t); // first index with x > t
    let i0 = i1 - 1;
    let span = times[i1] - times[i0];
    let alpha = if span > 0.0 {
        (t - times[i0]) / span
    } else {
        0.0
    };
    (i0, i1, alpha)
}

fn sample_vec3(times: &[f32], values: &[Vec3], t: f32) -> Vec3 {
    let (i0, i1, a) = segment(times, t);
    values[i0].lerp(values[i1], a)
}

fn sample_quat(times: &[f32], values: &[Quat], t: f32) -> Quat {
    let (i0, i1, a) = segment(times, t);
    values[i0].slerp(values[i1], a)
}
