//! Frame Interpolation for Video Generation
//!
//! Methods for generating intermediate frames between keyframes,
//! useful for temporal upsampling and smoother video output.
//!
//! # Methods
//!
//! - **Linear**: Simple weighted average between frames
//! - **SLERP**: Spherical interpolation for normalized latent vectors
//! - **Flow-based**: Motion estimation and warping (requires optical flow model)

use burn::prelude::*;

/// Frame interpolation configuration
#[derive(Debug, Clone)]
pub struct InterpolationConfig {
    /// Number of intermediate frames to generate
    pub num_intermediate: usize,
    /// Interpolation method
    pub method: InterpolationMethod,
    /// Whether to include original frames in output
    pub include_endpoints: bool,
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            num_intermediate: 1,
            method: InterpolationMethod::Linear,
            include_endpoints: true,
        }
    }
}

impl InterpolationConfig {
    pub fn new(num_intermediate: usize) -> Self {
        Self {
            num_intermediate,
            ..Default::default()
        }
    }

    pub fn with_method(mut self, method: InterpolationMethod) -> Self {
        self.method = method;
        self
    }

    pub fn with_endpoints(mut self, include: bool) -> Self {
        self.include_endpoints = include;
        self
    }

    /// Total output frames for each input pair
    pub fn output_frames(&self) -> usize {
        if self.include_endpoints {
            self.num_intermediate + 2
        } else {
            self.num_intermediate
        }
    }
}

/// Interpolation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolationMethod {
    /// Linear blend: (1-t) * a + t * b
    Linear,
    /// Spherical linear interpolation (for normalized vectors)
    Slerp,
    /// Cosine interpolation (smoother than linear)
    Cosine,
    /// Cubic Hermite interpolation (requires 4 frames)
    CubicHermite,
}

/// Linear interpolation between two tensors
///
/// `t` should be in [0, 1] where t=0 returns `a` and t=1 returns `b`
pub fn lerp<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>, t: f32) -> Tensor<B, D> {
    a.clone() * (1.0 - t) + b * t
}

/// Batch linear interpolation at multiple t values
///
/// Returns a tensor with an extra dimension for the interpolated frames
pub fn lerp_batch<B: Backend>(
    a: Tensor<B, 4>, // [batch, channels, height, width]
    b: Tensor<B, 4>,
    t_values: &[f32],
) -> Tensor<B, 5> {
    // [batch, time, channels, height, width]
    let frames: Vec<_> = t_values
        .iter()
        .map(|&t| lerp(a.clone(), b.clone(), t))
        .collect();

    // Stack along new time dimension
    Tensor::stack(frames, 1)
}

/// Spherical linear interpolation for normalized latent vectors
///
/// Better than lerp for interpolating on the unit hypersphere.
/// Maintains constant speed and stays on the geodesic.
pub fn slerp<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>, t: f32) -> Tensor<B, D> {
    // Normalize inputs
    let a_norm = normalize_tensor(a.clone());
    let b_norm = normalize_tensor(b.clone());

    // Compute angle between vectors via dot product
    let dot = tensor_dot(&a_norm, &b_norm);
    let dot_val: f32 = dot.into_scalar().elem();

    // Clamp to avoid numerical issues with acos
    let dot_clamped = dot_val.clamp(-1.0, 1.0);

    // If vectors are nearly parallel, fall back to lerp
    if dot_clamped.abs() > 0.9995 {
        return lerp(a, b, t);
    }

    let omega = dot_clamped.acos();
    let sin_omega = omega.sin();

    let s0 = ((1.0 - t) * omega).sin() / sin_omega;
    let s1 = (t * omega).sin() / sin_omega;

    a * s0 + b * s1
}

/// Batch spherical interpolation
pub fn slerp_batch<B: Backend>(a: Tensor<B, 4>, b: Tensor<B, 4>, t_values: &[f32]) -> Tensor<B, 5> {
    let frames: Vec<_> = t_values
        .iter()
        .map(|&t| slerp(a.clone(), b.clone(), t))
        .collect();

    Tensor::stack(frames, 1)
}

/// Cosine interpolation - smoother transitions at endpoints
///
/// Uses cosine curve instead of linear: (1 - cos(t*π)) / 2
pub fn cosine_interp<B: Backend, const D: usize>(
    a: Tensor<B, D>,
    b: Tensor<B, D>,
    t: f32,
) -> Tensor<B, D> {
    let t_smooth = (1.0 - (t * std::f32::consts::PI).cos()) * 0.5;
    lerp(a, b, t_smooth)
}

/// Cubic Hermite interpolation for 4 control points
///
/// Provides smooth interpolation using two intermediate points and their tangents.
/// Good for generating multiple frames with smooth motion.
pub fn cubic_hermite<B: Backend, const D: usize>(
    p0: Tensor<B, D>, // Frame before start
    p1: Tensor<B, D>, // Start frame
    p2: Tensor<B, D>, // End frame
    p3: Tensor<B, D>, // Frame after end
    t: f32,
) -> Tensor<B, D> {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom spline coefficients
    let a0 = -0.5 * t3 + t2 - 0.5 * t;
    let a1 = 1.5 * t3 - 2.5 * t2 + 1.0;
    let a2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t;
    let a3 = 0.5 * t3 - 0.5 * t2;

    p0 * a0 + p1 * a1 + p2 * a2 + p3 * a3
}

/// Interpolate between two frames using specified method
pub fn interpolate_frames<B: Backend>(
    a: Tensor<B, 4>,
    b: Tensor<B, 4>,
    config: &InterpolationConfig,
) -> Tensor<B, 5> {
    let num_interp = config.num_intermediate;
    let total = if config.include_endpoints {
        num_interp + 2
    } else {
        num_interp
    };

    // Generate t values
    let t_values: Vec<f32> = if config.include_endpoints {
        (0..total).map(|i| i as f32 / (total - 1) as f32).collect()
    } else {
        (1..=num_interp)
            .map(|i| i as f32 / (num_interp + 1) as f32)
            .collect()
    };

    match config.method {
        InterpolationMethod::Linear => lerp_batch(a, b, &t_values),
        InterpolationMethod::Slerp => slerp_batch(a, b, &t_values),
        InterpolationMethod::Cosine => {
            let frames: Vec<_> = t_values
                .iter()
                .map(|&t| cosine_interp(a.clone(), b.clone(), t))
                .collect();
            Tensor::stack(frames, 1)
        }
        InterpolationMethod::CubicHermite => {
            // For cubic hermite with only 2 points, use them as both control and endpoints
            let frames: Vec<_> = t_values
                .iter()
                .map(|&t| cubic_hermite(a.clone(), a.clone(), b.clone(), b.clone(), t))
                .collect();
            Tensor::stack(frames, 1)
        }
    }
}

/// Interpolate an entire video sequence
///
/// Input: [batch, channels, time, height, width]
/// Output: upsampled video with (time-1) * num_intermediate + time frames
pub fn interpolate_video<B: Backend>(
    video: Tensor<B, 5>,
    config: &InterpolationConfig,
) -> Tensor<B, 5> {
    let [batch, channels, time, height, width] = video.dims();

    if time < 2 {
        return video;
    }

    let mut all_frames = Vec::new();

    for t in 0..time - 1 {
        let frame_a = video
            .clone()
            .slice([0..batch, 0..channels, t..t + 1, 0..height, 0..width]);
        let frame_b =
            video
                .clone()
                .slice([0..batch, 0..channels, t + 1..t + 2, 0..height, 0..width]);

        // Squeeze time dimension for interpolation
        let frame_a = frame_a.reshape([batch, channels, height, width]);
        let frame_b = frame_b.reshape([batch, channels, height, width]);

        let interpolated = interpolate_frames(frame_a, frame_b, config);
        let [_b, interp_t, _c, _h, _w] = interpolated.dims();

        // Don't include last frame except for final segment (avoid duplicates)
        let take_frames = if t < time - 2 && config.include_endpoints {
            interp_t - 1
        } else {
            interp_t
        };

        for i in 0..take_frames {
            let frame =
                interpolated
                    .clone()
                    .slice([0..batch, i..i + 1, 0..channels, 0..height, 0..width]);
            let frame = frame.reshape([batch, channels, 1, height, width]);
            all_frames.push(frame);
        }
    }

    Tensor::cat(all_frames, 2)
}

/// Optical flow-based frame warping
///
/// Warps source frame according to flow field.
/// Flow shape: [batch, 2, height, width] where 2 = (dx, dy)
pub fn warp_frame<B: Backend>(
    frame: Tensor<B, 4>, // [batch, channels, height, width]
    flow: Tensor<B, 4>,  // [batch, 2, height, width]
) -> Tensor<B, 4> {
    let [batch, _channels, height, width] = frame.dims();
    let device = frame.device();

    // Create base grid
    let (grid_x, grid_y) = create_grid::<B>(height, width, &device);

    // Extract flow components and normalize to [-1, 1]
    let flow_x = flow.clone().slice([0..batch, 0..1, 0..height, 0..width]);
    let flow_y = flow.slice([0..batch, 1..2, 0..height, 0..width]);

    // Normalize flow to [-1, 1] range
    let flow_x = flow_x / (width as f32 / 2.0);
    let flow_y = flow_y / (height as f32 / 2.0);

    // Add flow to grid
    let sample_x = grid_x + flow_x.reshape([batch, height, width]);
    let sample_y = grid_y + flow_y.reshape([batch, height, width]);

    // Bilinear sample (simplified - in practice would use grid_sample)
    bilinear_sample(frame, sample_x, sample_y)
}

/// Create normalized coordinate grid
fn create_grid<B: Backend>(
    height: usize,
    width: usize,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 3>) {
    // X coordinates: -1 to 1
    let x_coords: Vec<f32> = (0..width)
        .map(|i| (2.0 * i as f32 / (width - 1) as f32) - 1.0)
        .collect();
    let x = Tensor::<B, 1>::from_floats(x_coords.as_slice(), device);
    let x = x.reshape([1, 1, width]).repeat_dim(1, height);

    // Y coordinates: -1 to 1
    let y_coords: Vec<f32> = (0..height)
        .map(|i| (2.0 * i as f32 / (height - 1) as f32) - 1.0)
        .collect();
    let y = Tensor::<B, 1>::from_floats(y_coords.as_slice(), device);
    let y = y.reshape([1, height, 1]).repeat_dim(2, width);

    (x, y)
}

/// Bilinear sampling from image given continuous coordinates
fn bilinear_sample<B: Backend>(
    img: Tensor<B, 4>, // [batch, channels, height, width]
    x: Tensor<B, 3>,   // [batch, out_h, out_w] normalized coords [-1, 1]
    y: Tensor<B, 3>,   // [batch, out_h, out_w] normalized coords [-1, 1]
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = img.dims();
    let [_, out_h, out_w] = x.dims();

    // Convert from [-1, 1] to pixel coordinates [0, width-1] and [0, height-1]
    let x_pixel = (x.clone() + 1.0) * ((width - 1) as f32 / 2.0);
    let y_pixel = (y.clone() + 1.0) * ((height - 1) as f32 / 2.0);

    // Get integer coordinates for the four corners
    let x0 = x_pixel.clone().floor();
    let y0 = y_pixel.clone().floor();
    let x1 = x0.clone() + 1.0;
    let y1 = y0.clone() + 1.0;

    // Compute interpolation weights
    let wa = (x1.clone() - x_pixel.clone()) * (y1.clone() - y_pixel.clone());
    let wb = (x_pixel.clone() - x0.clone()) * (y1.clone() - y_pixel.clone());
    let wc = (x1.clone() - x_pixel.clone()) * (y_pixel.clone() - y0.clone());
    let wd = (x_pixel.clone() - x0.clone()) * (y_pixel.clone() - y0.clone());

    // Clamp coordinates to valid range
    let x0 = x0.clamp(0.0, (width - 1) as f32);
    let y0 = y0.clamp(0.0, (height - 1) as f32);
    let x1 = x1.clamp(0.0, (width - 1) as f32);
    let y1 = y1.clamp(0.0, (height - 1) as f32);

    // Flatten image spatial dimensions: [batch, channels, height * width]
    let img_flat = img.reshape([batch, channels, height * width]);

    // Compute linear indices for each corner: idx = y * width + x
    // Indices shape: [batch, out_h, out_w]
    let x0_i = x0.int();
    let y0_i = y0.int();
    let x1_i = x1.int();
    let y1_i = y1.int();

    let width_i = width as i64;
    let idx_a = (y0_i.clone() * width_i + x0_i.clone()).reshape([batch, 1, out_h * out_w]);
    let idx_b = (y0_i.clone() * width_i + x1_i.clone()).reshape([batch, 1, out_h * out_w]);
    let idx_c = (y1_i.clone() * width_i + x0_i).reshape([batch, 1, out_h * out_w]);
    let idx_d = (y1_i * width_i + x1_i).reshape([batch, 1, out_h * out_w]);

    // Expand indices for all channels: [batch, channels, out_h * out_w]
    let idx_a = idx_a.repeat_dim(1, channels);
    let idx_b = idx_b.repeat_dim(1, channels);
    let idx_c = idx_c.repeat_dim(1, channels);
    let idx_d = idx_d.repeat_dim(1, channels);

    // Gather values from each corner
    let va = img_flat.clone().gather(2, idx_a);
    let vb = img_flat.clone().gather(2, idx_b);
    let vc = img_flat.clone().gather(2, idx_c);
    let vd = img_flat.gather(2, idx_d);

    // Reshape gathered values: [batch, channels, out_h, out_w]
    let va = va.reshape([batch, channels, out_h, out_w]);
    let vb = vb.reshape([batch, channels, out_h, out_w]);
    let vc = vc.reshape([batch, channels, out_h, out_w]);
    let vd = vd.reshape([batch, channels, out_h, out_w]);

    // Expand weights to match: [batch, 1, out_h, out_w] -> broadcast with channels
    let wa = wa.unsqueeze_dim::<4>(1);
    let wb = wb.unsqueeze_dim::<4>(1);
    let wc = wc.unsqueeze_dim::<4>(1);
    let wd = wd.unsqueeze_dim::<4>(1);

    // Bilinear blend
    va * wa + vb * wb + vc * wc + vd * wd
}

/// Normalize tensor to unit length
fn normalize_tensor<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let sum_sq = x.clone().powf_scalar(2.0).sum();
    let norm: f32 = sum_sq.into_scalar().elem();
    let norm = norm.sqrt().max(1e-8);
    x / norm
}

/// Compute dot product between two tensors (flattened)
fn tensor_dot<B: Backend, const D: usize>(a: &Tensor<B, D>, b: &Tensor<B, D>) -> Tensor<B, 1> {
    let a_flat = a.clone().flatten::<1>(0, D - 1);
    let b_flat = b.clone().flatten::<1>(0, D - 1);
    (a_flat * b_flat).sum()
}

/// Flow estimation result
pub struct FlowResult<B: Backend> {
    /// Forward flow: frame0 -> frame1
    pub forward: Tensor<B, 4>,
    /// Backward flow: frame1 -> frame0
    pub backward: Tensor<B, 4>,
    /// Occlusion mask (optional)
    pub occlusion: Option<Tensor<B, 4>>,
}

/// Bidirectional interpolation using forward and backward flow
pub fn flow_interpolate<B: Backend>(
    frame0: Tensor<B, 4>,
    frame1: Tensor<B, 4>,
    flow: &FlowResult<B>,
    t: f32,
) -> Tensor<B, 4> {
    // Warp frame0 forward by t * forward_flow
    let forward_scaled = flow.forward.clone() * t;
    let warped0 = warp_frame(frame0.clone(), forward_scaled);

    // Warp frame1 backward by (1-t) * backward_flow
    let backward_scaled = flow.backward.clone() * (1.0 - t);
    let warped1 = warp_frame(frame1.clone(), backward_scaled);

    // Blend based on occlusion or simple average
    match &flow.occlusion {
        Some(occ) => {
            let w0 = occ.clone() * (1.0 - t);
            let w1 = (occ.clone() * -1.0 + 1.0) * t;
            let total = w0.clone() + w1.clone();
            (warped0 * w0 + warped1 * w1) / total.clamp_min(1e-8)
        }
        None => lerp(warped0, warped1, t),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_lerp() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 2>::from_floats([[0.0, 0.0]], &device);
        let b = Tensor::<TestBackend, 2>::from_floats([[1.0, 2.0]], &device);

        let mid = lerp(a.clone(), b.clone(), 0.5);
        let data: Vec<f32> = mid.to_data().to_vec().unwrap();
        assert!((data[0] - 0.5).abs() < 1e-5);
        assert!((data[1] - 1.0).abs() < 1e-5);

        let start = lerp(a.clone(), b.clone(), 0.0);
        let data: Vec<f32> = start.to_data().to_vec().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-5);

        let end = lerp(a, b, 1.0);
        let data: Vec<f32> = end.to_data().to_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_lerp_batch() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 4>::zeros([1, 3, 4, 4], &device);
        let b = Tensor::<TestBackend, 4>::ones([1, 3, 4, 4], &device);

        let result = lerp_batch(a, b, &[0.0, 0.5, 1.0]);
        assert_eq!(result.dims(), [1, 3, 3, 4, 4]);
    }

    #[test]
    fn test_slerp_parallel_vectors() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0, 0.0], &device);
        let b = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0, 0.0], &device);

        // Should fall back to lerp for parallel vectors
        let mid = slerp(a, b, 0.5);
        let data: Vec<f32> = mid.to_data().to_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_slerp_orthogonal() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 1>::from_floats([1.0, 0.0], &device);
        let b = Tensor::<TestBackend, 1>::from_floats([0.0, 1.0], &device);

        let mid = slerp(a, b, 0.5);
        let data: Vec<f32> = mid.to_data().to_vec().unwrap();
        // At t=0.5 between orthogonal unit vectors, should be on the arc
        let norm = (data[0] * data[0] + data[1] * data[1]).sqrt();
        assert!((norm - 1.0).abs() < 0.1); // Close to unit length
    }

    #[test]
    fn test_cosine_interp() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let b = Tensor::<TestBackend, 1>::from_floats([1.0], &device);

        let start = cosine_interp(a.clone(), b.clone(), 0.0);
        let mid = cosine_interp(a.clone(), b.clone(), 0.5);
        let end = cosine_interp(a, b, 1.0);

        let s: f32 = start.into_scalar().elem();
        let m: f32 = mid.into_scalar().elem();
        let e: f32 = end.into_scalar().elem();

        assert!((s - 0.0).abs() < 1e-5);
        assert!((m - 0.5).abs() < 1e-5);
        assert!((e - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_cubic_hermite() {
        let device = Default::default();
        let p0 = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let p1 = Tensor::<TestBackend, 1>::from_floats([1.0], &device);
        let p2 = Tensor::<TestBackend, 1>::from_floats([2.0], &device);
        let p3 = Tensor::<TestBackend, 1>::from_floats([3.0], &device);

        let mid = cubic_hermite(p0, p1, p2, p3, 0.5);
        let val: f32 = mid.into_scalar().elem();
        // Should be close to linear interpolation between p1 and p2
        assert!((val - 1.5).abs() < 0.2);
    }

    #[test]
    fn test_interpolation_config() {
        let config = InterpolationConfig::default();
        assert_eq!(config.num_intermediate, 1);
        assert_eq!(config.output_frames(), 3); // 2 endpoints + 1 intermediate

        let config = InterpolationConfig::new(3).with_endpoints(false);
        assert_eq!(config.output_frames(), 3);
    }

    #[test]
    fn test_interpolate_frames() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 4>::zeros([1, 3, 8, 8], &device);
        let b = Tensor::<TestBackend, 4>::ones([1, 3, 8, 8], &device);

        let config = InterpolationConfig::new(2);
        let result = interpolate_frames(a, b, &config);
        // 2 intermediate + 2 endpoints = 4
        assert_eq!(result.dims(), [1, 4, 3, 8, 8]);
    }

    #[test]
    fn test_interpolate_video() {
        let device = Default::default();
        // 4 frame video
        let video = Tensor::<TestBackend, 5>::zeros([1, 3, 4, 8, 8], &device);

        let config = InterpolationConfig::new(1); // 1 intermediate between each pair
        let result = interpolate_video(video, &config);

        // Original: 4 frames
        // Between frame 0-1: 3 frames (including endpoints)
        // Between frame 1-2: 2 frames (excluding first endpoint to avoid duplicate)
        // Between frame 2-3: 3 frames (including last)
        // Total: 3 + 2 + 2 = 7 frames (avoiding duplicates)
        assert_eq!(result.dims()[2], 7);
    }

    #[test]
    fn test_create_grid() {
        let device = Default::default();
        let (x, y) = create_grid::<TestBackend>(4, 4, &device);

        assert_eq!(x.dims(), [1, 4, 4]);
        assert_eq!(y.dims(), [1, 4, 4]);
    }

    #[test]
    fn test_warp_frame_identity() {
        let device = Default::default();
        let frame = Tensor::<TestBackend, 4>::ones([1, 3, 8, 8], &device);
        let flow = Tensor::<TestBackend, 4>::zeros([1, 2, 8, 8], &device); // Zero flow

        let warped = warp_frame(frame.clone(), flow);
        assert_eq!(warped.dims(), frame.dims());
    }

    #[test]
    fn test_interpolation_methods() {
        let device = Default::default();
        let a = Tensor::<TestBackend, 4>::zeros([1, 1, 4, 4], &device);
        let b = Tensor::<TestBackend, 4>::ones([1, 1, 4, 4], &device);

        for method in [
            InterpolationMethod::Linear,
            InterpolationMethod::Slerp,
            InterpolationMethod::Cosine,
            InterpolationMethod::CubicHermite,
        ] {
            let config = InterpolationConfig::new(1).with_method(method);
            let result = interpolate_frames(a.clone(), b.clone(), &config);
            assert_eq!(result.dims(), [1, 3, 1, 4, 4]);
        }
    }
}
