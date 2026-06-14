use wgpu::Color;

pub fn normalize(value: f64, start: f64, stop: f64) -> f64 {
    assert!(start != stop, "cannot normalize over a zero-width range");
    (value - start) / (stop - start)
}

pub trait ColorFromXY {
    /// Build a `Color` from an (x, y) coordinate pair.
    ///
    /// `x` is normalized into the red channel, `y` into green,
    /// and `z_from` computes blue from the already-normalized r and g.
    /// Alpha is set to 1.0.
    fn from_xy(x: f64, x_range: (f64, f64), y: f64, y_range: (f64, f64)) -> Self;
}

impl ColorFromXY for Color {
    fn from_xy(x: f64, x_range: (f64, f64), y: f64, y_range: (f64, f64)) -> Self {
        let r = normalize(x, x_range.0, x_range.1);
        let g = normalize(y, y_range.0, y_range.1);
        let b = (r + g) / 2.0;
        Color { r, g, b, a: 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn normalize_maps_endpoints_and_midpoint() {
        assert!((normalize(0.0, 0.0, 10.0) - 0.0).abs() < EPS);
        assert!((normalize(5.0, 0.0, 10.0) - 0.5).abs() < EPS);
        assert!((normalize(10.0, 0.0, 10.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn normalize_extrapolates_outside_range() {
        assert!((normalize(15.0, 0.0, 10.0) - 1.5).abs() < EPS);
        assert!((normalize(-5.0, 0.0, 10.0) - -0.5).abs() < EPS);
    }

    #[test]
    fn normalize_handles_negative_range() {
        assert!((normalize(0.0, -10.0, 10.0) - 0.5).abs() < EPS);
    }

    #[test]
    #[should_panic(expected = "zero-width range")]
    fn normalize_panics_on_zero_width_range() {
        normalize(1.0, 5.0, 5.0);
    }

    #[test]
    fn from_xy_normalizes_each_channel() {
        let c = Color::from_xy(5.0, (0.0, 10.0), 2.0, (0.0, 10.0));
        assert!((c.r - 0.5).abs() < EPS);
        assert!((c.g - 0.2).abs() < EPS);
        // blue is the average of the normalized r and g
        assert!((c.b - 0.35).abs() < EPS);
        assert!((c.a - 1.0).abs() < EPS);
    }

    #[test]
    fn from_xy_uses_independent_ranges_per_axis() {
        let c = Color::from_xy(50.0, (0.0, 100.0), 1.0, (0.0, 4.0));
        assert!((c.r - 0.5).abs() < EPS);
        assert!((c.g - 0.25).abs() < EPS);
        assert!((c.b - 0.375).abs() < EPS);
    }
}

