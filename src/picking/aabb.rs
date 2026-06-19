use glam::{Mat4, Vec3};

/// Axis-aligned bounding box.
#[derive(Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Aabb {
    pub fn from_points(points: impl IntoIterator<Item = Vec3>) -> Self {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for p in points {
            min = min.min(p);
            max = max.max(p);
        }
        Self { min, max }
    }

    /// World-space AABB after applying an affine transform (re-bounds the 8 corners).
    pub fn transformed(&self, m: &Mat4) -> Self {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        for i in 0..8u32 {
            let corner = Vec3::new(
                if i & 1 == 0 { self.min.x } else { self.max.x },
                if i & 2 == 0 { self.min.y } else { self.max.y },
                if i & 4 == 0 { self.min.z } else { self.max.z },
            );
            let w = m.transform_point3(corner);
            min = min.min(w);
            max = max.max(w);
        }
        Self { min, max }
    }

    /// Nearest ray hit distance (slab method), if the ray crosses the box ahead.
    pub fn ray_intersect(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        let inv = dir.recip();
        let t1 = (self.min - origin) * inv;
        let t2 = (self.max - origin) * inv;
        let t_near = t1.min(t2).max_element();
        let t_far = t1.max(t2).min_element();
        (t_near <= t_far && t_far >= 0.0).then(|| t_near.max(0.0))
    }
}
