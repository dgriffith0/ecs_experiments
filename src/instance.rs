use glam::{Mat3, Mat4, Quat, Vec3};

pub struct Instance {
    pub position: Vec3,
    pub rotation: Quat,
}

impl Instance {
    pub fn to_raw(&self) -> InstanceRaw {
        let model = Mat4::from_translation(self.position) * Mat4::from_quat(self.rotation);
        InstanceRaw {
            model: model.to_cols_array_2d(),
            // NEW!
            normal: Mat3::from_quat(self.rotation).to_cols_array_2d(),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl InstanceRaw {
    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We'll have to reassemble the mat4 in the shader.
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials, we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5, not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_instance_produces_identity_matrices() {
        let instance = Instance {
            position: Vec3::ZERO,
            rotation: Quat::IDENTITY,
        };
        let raw = instance.to_raw();

        let identity4 = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        (0..4).for_each(|r| {
            (0..4).for_each(|c| {
                assert!((raw.model[r][c] - identity4[r][c]).abs() < 1e-6);
            });
        });

        let identity3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        (0..3).for_each(|r| {
            (0..3).for_each(|c| {
                assert!((raw.normal[r][c] - identity3[r][c]).abs() < 1e-6);
            });
        });
    }

    #[test]
    fn translation_lands_in_model_matrix() {
        let instance = Instance {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::IDENTITY,
        };
        let raw = instance.to_raw();

        // glam matrices are column-major; the translation occupies the last column.
        assert!((raw.model[3][0] - 1.0).abs() < 1e-6);
        assert!((raw.model[3][1] - 2.0).abs() < 1e-6);
        assert!((raw.model[3][2] - 3.0).abs() < 1e-6);
        assert!((raw.model[3][3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rotation_only_leaves_translation_at_origin() {
        let instance = Instance {
            position: Vec3::ZERO,
            rotation: Quat::from_axis_angle(Vec3::Z, 90f32.to_radians()),
        };
        let raw = instance.to_raw();

        // No translation component.
        assert!((raw.model[3][0]).abs() < 1e-6);
        assert!((raw.model[3][1]).abs() < 1e-6);
        assert!((raw.model[3][2]).abs() < 1e-6);

        // A 90° rotation about +z maps the model's x axis onto +y, so the
        // normal matrix's first column should be ~(0, 1, 0).
        assert!((raw.normal[0][0]).abs() < 1e-6);
        assert!((raw.normal[0][1] - 1.0).abs() < 1e-6);
    }
}
