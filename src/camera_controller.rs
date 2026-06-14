use winit::keyboard::KeyCode;

use cgmath::InnerSpace;

use crate::camera::Camera;

pub struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    pub fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyW | KeyCode::ArrowUp => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA | KeyCode::ArrowLeft => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyS | KeyCode::ArrowDown => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyD | KeyCode::ArrowRight => {
                self.is_right_pressed = is_pressed;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when the camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and the eye so
            // that it doesn't change. The eye, therefore, still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgmath::{Point3, Vector3};

    fn test_camera() -> Camera {
        Camera {
            eye: Point3::new(0.0, 0.0, 2.0),
            target: Point3::new(0.0, 0.0, 0.0),
            up: Vector3::unit_y(),
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }

    #[test]
    fn new_starts_with_no_keys_pressed() {
        let c = CameraController::new(0.2);
        assert!(!c.is_forward_pressed);
        assert!(!c.is_backward_pressed);
        assert!(!c.is_left_pressed);
        assert!(!c.is_right_pressed);
    }

    #[test]
    fn handle_key_maps_wasd_and_arrows() {
        let mut c = CameraController::new(0.2);

        assert!(c.handle_key(KeyCode::KeyW, true));
        assert!(c.is_forward_pressed);
        assert!(c.handle_key(KeyCode::ArrowUp, false));
        assert!(!c.is_forward_pressed);

        assert!(c.handle_key(KeyCode::KeyA, true));
        assert!(c.is_left_pressed);
        assert!(c.handle_key(KeyCode::KeyS, true));
        assert!(c.is_backward_pressed);
        assert!(c.handle_key(KeyCode::KeyD, true));
        assert!(c.is_right_pressed);
    }

    #[test]
    fn handle_key_ignores_unmapped_keys() {
        let mut c = CameraController::new(0.2);
        assert!(!c.handle_key(KeyCode::Space, true));
        assert!(!c.is_forward_pressed);
        assert!(!c.is_backward_pressed);
        assert!(!c.is_left_pressed);
        assert!(!c.is_right_pressed);
    }

    #[test]
    fn forward_moves_eye_toward_target() {
        let controller = CameraController {
            speed: 0.2,
            is_forward_pressed: true,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        };
        let mut camera = test_camera();
        controller.update_camera(&mut camera);
        // eye started at z=2.0, target at origin, so it should move closer.
        assert!((camera.eye.z - 1.8).abs() < 1e-6);
    }

    #[test]
    fn backward_moves_eye_away_from_target() {
        let controller = CameraController {
            speed: 0.2,
            is_forward_pressed: false,
            is_backward_pressed: true,
            is_left_pressed: false,
            is_right_pressed: false,
        };
        let mut camera = test_camera();
        controller.update_camera(&mut camera);
        assert!((camera.eye.z - 2.2).abs() < 1e-6);
    }

    #[test]
    fn forward_does_not_overshoot_target() {
        // When the eye is closer than `speed`, the forward guard prevents movement.
        let controller = CameraController {
            speed: 0.2,
            is_forward_pressed: true,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        };
        let mut camera = test_camera();
        camera.eye = Point3::new(0.0, 0.0, 0.1); // magnitude 0.1 < speed 0.2
        controller.update_camera(&mut camera);
        assert!((camera.eye.z - 0.1).abs() < 1e-6);
    }

    #[test]
    fn strafing_preserves_distance_to_target() {
        let controller = CameraController {
            speed: 0.2,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: true,
        };
        let mut camera = test_camera();
        let before = (camera.target - camera.eye).magnitude();
        controller.update_camera(&mut camera);
        let after = (camera.target - camera.eye).magnitude();
        assert!((before - after).abs() < 1e-6);
        // Right strafe should pull the eye off the pure -z axis.
        assert!(camera.eye.x.abs() > 1e-3);
    }
}
