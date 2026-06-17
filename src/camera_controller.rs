use glam::Vec3;
use winit::keyboard::KeyCode;

use crate::camera::Camera;

/// Pitch is clamped to just under 90° so looking up/down never flips the view.
const MAX_PITCH: f32 = 1.54; // ~88°

/// A free-fly camera controller: WASD to move on the look/strafe axes, Space and
/// Shift to rise/descend, and the arrow keys to look around (yaw/pitch).
pub struct CameraController {
    speed: f32,
    rotate_speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
    is_up_pressed: bool,
    is_down_pressed: bool,
    look_left: bool,
    look_right: bool,
    look_up: bool,
    look_down: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            // Radians per frame for the look keys. Tuned to feel similar to the
            // movement speed at typical scene scale.
            rotate_speed: 0.03,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
            is_up_pressed: false,
            is_down_pressed: false,
            look_left: false,
            look_right: false,
            look_up: false,
            look_down: false,
        }
    }

    pub fn handle_key(&mut self, code: KeyCode, is_pressed: bool) -> bool {
        match code {
            KeyCode::KeyW => {
                self.is_forward_pressed = is_pressed;
                true
            }
            KeyCode::KeyS => {
                self.is_backward_pressed = is_pressed;
                true
            }
            KeyCode::KeyA => {
                self.is_left_pressed = is_pressed;
                true
            }
            KeyCode::KeyD => {
                self.is_right_pressed = is_pressed;
                true
            }
            KeyCode::Space => {
                self.is_up_pressed = is_pressed;
                true
            }
            KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                self.is_down_pressed = is_pressed;
                true
            }
            KeyCode::ArrowLeft => {
                self.look_left = is_pressed;
                true
            }
            KeyCode::ArrowRight => {
                self.look_right = is_pressed;
                true
            }
            KeyCode::ArrowUp => {
                self.look_up = is_pressed;
                true
            }
            KeyCode::ArrowDown => {
                self.look_down = is_pressed;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        // --- Look: rotate yaw/pitch from the arrow keys ---
        if self.look_left {
            camera.yaw -= self.rotate_speed;
        }
        if self.look_right {
            camera.yaw += self.rotate_speed;
        }
        if self.look_up {
            camera.pitch += self.rotate_speed;
        }
        if self.look_down {
            camera.pitch -= self.rotate_speed;
        }
        camera.pitch = camera.pitch.clamp(-MAX_PITCH, MAX_PITCH);

        // --- Move: forward follows where we look; up/down stays world-vertical ---
        let forward = camera.forward();
        // Right is horizontal so strafing doesn't drift vertically.
        let right = forward.cross(Vec3::Y).normalize_or_zero();

        if self.is_forward_pressed {
            camera.position += forward * self.speed;
        }
        if self.is_backward_pressed {
            camera.position -= forward * self.speed;
        }
        if self.is_right_pressed {
            camera.position += right * self.speed;
        }
        if self.is_left_pressed {
            camera.position -= right * self.speed;
        }
        if self.is_up_pressed {
            camera.position += Vec3::Y * self.speed;
        }
        if self.is_down_pressed {
            camera.position -= Vec3::Y * self.speed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_camera() -> Camera {
        Camera {
            position: Vec3::new(0.0, 0.0, 0.0),
            // Looking toward -Z, level with the horizon.
            yaw: -std::f32::consts::FRAC_PI_2,
            pitch: 0.0,
            aspect: 1.0,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        }
    }

    fn pressed(set: impl FnOnce(&mut CameraController)) -> CameraController {
        let mut c = CameraController::new(0.2);
        set(&mut c);
        c
    }

    #[test]
    fn new_starts_with_no_input() {
        let c = CameraController::new(0.2);
        assert!(!c.is_forward_pressed);
        assert!(!c.is_backward_pressed);
        assert!(!c.is_left_pressed);
        assert!(!c.is_right_pressed);
        assert!(!c.is_up_pressed);
        assert!(!c.is_down_pressed);
        assert!(!c.look_left && !c.look_right && !c.look_up && !c.look_down);
    }

    #[test]
    fn handle_key_maps_movement_and_look() {
        let mut c = CameraController::new(0.2);

        assert!(c.handle_key(KeyCode::KeyW, true));
        assert!(c.is_forward_pressed);
        assert!(c.handle_key(KeyCode::KeyS, true));
        assert!(c.is_backward_pressed);
        assert!(c.handle_key(KeyCode::KeyA, true));
        assert!(c.is_left_pressed);
        assert!(c.handle_key(KeyCode::KeyD, true));
        assert!(c.is_right_pressed);
        assert!(c.handle_key(KeyCode::Space, true));
        assert!(c.is_up_pressed);
        assert!(c.handle_key(KeyCode::ShiftLeft, true));
        assert!(c.is_down_pressed);
        assert!(c.handle_key(KeyCode::ArrowLeft, true));
        assert!(c.look_left);
        assert!(c.handle_key(KeyCode::ArrowRight, true));
        assert!(c.look_right);
        assert!(c.handle_key(KeyCode::ArrowUp, true));
        assert!(c.look_up);
        assert!(c.handle_key(KeyCode::ArrowDown, true));
        assert!(c.look_down);
    }

    #[test]
    fn handle_key_ignores_unmapped_keys() {
        let mut c = CameraController::new(0.2);
        assert!(!c.handle_key(KeyCode::Enter, true));
        assert!(!c.is_forward_pressed);
    }

    #[test]
    fn forward_moves_along_the_look_direction() {
        let controller = pressed(|c| c.is_forward_pressed = true);
        let mut camera = test_camera();
        controller.update_camera(&mut camera);
        // Facing -Z, so forward movement decreases z and leaves x/y untouched.
        assert!((camera.position.z - -0.2).abs() < 1e-6);
        assert!(camera.position.x.abs() < 1e-6);
        assert!(camera.position.y.abs() < 1e-6);
    }

    #[test]
    fn strafe_is_horizontal_even_when_looking_up() {
        let controller = pressed(|c| c.is_right_pressed = true);
        let mut camera = test_camera();
        camera.pitch = 1.0; // look upward
        controller.update_camera(&mut camera);
        // Strafing right while facing -Z moves +X, and never changes height.
        assert!((camera.position.x - 0.2).abs() < 1e-6);
        assert!(camera.position.y.abs() < 1e-6);
    }

    #[test]
    fn up_down_move_world_vertical() {
        let mut camera = test_camera();
        pressed(|c| c.is_up_pressed = true).update_camera(&mut camera);
        assert!((camera.position.y - 0.2).abs() < 1e-6);

        let mut camera = test_camera();
        pressed(|c| c.is_down_pressed = true).update_camera(&mut camera);
        assert!((camera.position.y - -0.2).abs() < 1e-6);
    }

    #[test]
    fn look_changes_orientation_without_moving() {
        let mut camera = test_camera();
        let pos_before = camera.position;
        let yaw_before = camera.yaw;
        pressed(|c| c.look_right = true).update_camera(&mut camera);
        assert!(camera.yaw > yaw_before);
        assert!((camera.position - pos_before).length() < 1e-9);
    }

    #[test]
    fn pitch_is_clamped_to_avoid_flipping() {
        let controller = pressed(|c| c.look_up = true);
        let mut camera = test_camera();
        for _ in 0..1000 {
            controller.update_camera(&mut camera);
        }
        assert!(camera.pitch <= MAX_PITCH + 1e-6);
        assert!(camera.pitch >= MAX_PITCH - 1e-6);
    }
}
