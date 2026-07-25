//! 3D VAE for Video Generation
//!
//! A 3D Variational Autoencoder that compresses videos spatially and temporally.
//! Used in video generation models like CogVideoX, Wan, Mochi, etc.
//!
//! # Architecture
//!
//! ```text
//! Video [B, C, T, H, W]
//!     │
//!     ▼ Encoder (3D conv + temporal downsample)
//! Latent [B, Z, T/4, H/8, W/8]
//!     │
//!     ▼ Decoder (3D conv + temporal upsample)
//! Video [B, C, T, H, W]
//! ```
//!
//! # Compression
//!
//! - Temporal: 4x (16 frames → 4 latent frames)
//! - Spatial: 8x (512x512 → 64x64)
//! - Total: 256x compression

use burn::prelude::*;

/// Configuration for 3D VAE
#[derive(Debug, Clone)]
pub struct Vae3dConfig {
    /// Input channels (usually 3 for RGB)
    pub in_channels: usize,
    /// Latent channels
    pub latent_channels: usize,
    /// Base channel multiplier
    pub base_channels: usize,
    /// Channel multipliers for each level
    pub channel_mults: Vec<usize>,
    /// Temporal compression factor
    pub temporal_compression: usize,
    /// Spatial compression factor
    pub spatial_compression: usize,
    /// Number of residual blocks per level
    pub num_res_blocks: usize,
}

impl Default for Vae3dConfig {
    fn default() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 4,
            base_channels: 128,
            channel_mults: vec![1, 2, 4, 4],
            temporal_compression: 4,
            spatial_compression: 8,
            num_res_blocks: 2,
        }
    }
}

impl Vae3dConfig {
    /// CogVideoX-style VAE
    pub fn cogvideox() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 16,
            base_channels: 128,
            channel_mults: vec![1, 2, 4, 4],
            temporal_compression: 4,
            spatial_compression: 8,
            num_res_blocks: 2,
        }
    }

    /// Compute output shape for encoder
    pub fn encoded_shape(&self, time: usize, height: usize, width: usize) -> (usize, usize, usize) {
        (
            time / self.temporal_compression,
            height / self.spatial_compression,
            width / self.spatial_compression,
        )
    }
}

/// Temporal downsampling (average pool in time)
pub fn temporal_downsample<B: Backend>(x: Tensor<B, 5>, factor: usize) -> Tensor<B, 5> {
    let [batch, channels, time, height, width] = x.dims();
    let new_time = time / factor;
    let device = x.device();

    // Reshape to [batch, channels, new_time, factor, height, width]
    // Then average over factor dimension
    // Simplified: just slice every factor'th frame

    let mut frames = Vec::new();
    for t in 0..new_time {
        let frame = x.clone().slice([
            0..batch,
            0..channels,
            t * factor..t * factor + 1,
            0..height,
            0..width,
        ]);
        frames.push(frame);
    }

    if frames.is_empty() {
        Tensor::zeros([batch, channels, 0, height, width], &device)
    } else {
        Tensor::cat(frames, 2)
    }
}

/// Temporal upsampling (repeat frames)
pub fn temporal_upsample<B: Backend>(x: Tensor<B, 5>, factor: usize) -> Tensor<B, 5> {
    // Repeat each frame 'factor' times
    // [B, C, T, H, W] -> [B, C, T*factor, H, W]
    x.repeat_dim(2, factor)
}

/// Spatial downsampling for 3D tensors
pub fn spatial_downsample_3d<B: Backend>(x: Tensor<B, 5>, factor: usize) -> Tensor<B, 5> {
    let [batch, channels, time, height, width] = x.dims();
    let new_height = height / factor;
    let new_width = width / factor;

    // Simplified: slice to downsample
    // Real implementation would use strided conv or avg pool
    let mut result_frames = Vec::new();
    for t in 0..time {
        let frame = x
            .clone()
            .slice([0..batch, 0..channels, t..t + 1, 0..height, 0..width]);
        // Simple subsample (every factor'th pixel)
        let frame_3d = frame.reshape([batch, channels, height, width]);
        let subsampled = subsample_2d(frame_3d, factor);
        let frame_5d = subsampled.reshape([batch, channels, 1, new_height, new_width]);
        result_frames.push(frame_5d);
    }

    if result_frames.is_empty() {
        let device = x.device();
        Tensor::zeros([batch, channels, 0, new_height, new_width], &device)
    } else {
        Tensor::cat(result_frames, 2)
    }
}

fn subsample_2d<B: Backend>(x: Tensor<B, 4>, factor: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.dims();
    let new_height = height / factor;
    let new_width = width / factor;

    // Simple subsample - take every factor'th element
    // Real implementation would do proper resampling
    let mut rows = Vec::new();
    for h in 0..new_height {
        let row = x
            .clone()
            .slice([0..batch, 0..channels, h * factor..h * factor + 1, 0..width]);
        // Subsample width
        let mut cols = Vec::new();
        for w in 0..new_width {
            let col = row
                .clone()
                .slice([0..batch, 0..channels, 0..1, w * factor..w * factor + 1]);
            cols.push(col);
        }
        let subsampled_row = Tensor::cat(cols, 3);
        rows.push(subsampled_row);
    }

    if rows.is_empty() {
        let device = x.device();
        Tensor::zeros([batch, channels, new_height, new_width], &device)
    } else {
        Tensor::cat(rows, 2).reshape([batch, channels, new_height, new_width])
    }
}

/// Spatial upsampling for 3D tensors (nearest neighbor)
pub fn spatial_upsample_3d<B: Backend>(x: Tensor<B, 5>, factor: usize) -> Tensor<B, 5> {
    let [batch, channels, time, height, width] = x.dims();
    let new_height = height * factor;
    let new_width = width * factor;

    let mut result_frames = Vec::new();
    for t in 0..time {
        let frame = x
            .clone()
            .slice([0..batch, 0..channels, t..t + 1, 0..height, 0..width]);
        let frame_4d = frame.reshape([batch, channels, height, width]);

        // Upsample height then width using repeat
        let upsampled_h = frame_4d
            .unsqueeze_dim::<5>(3)
            .repeat_dim(3, factor)
            .reshape([batch, channels, new_height, width]);

        let upsampled = upsampled_h
            .unsqueeze_dim::<5>(4)
            .repeat_dim(4, factor)
            .reshape([batch, channels, new_height, new_width]);

        let frame_5d = upsampled.reshape([batch, channels, 1, new_height, new_width]);
        result_frames.push(frame_5d);
    }

    if result_frames.is_empty() {
        let device = x.device();
        Tensor::zeros([batch, channels, 0, new_height, new_width], &device)
    } else {
        Tensor::cat(result_frames, 2)
    }
}

/// 3D VAE Encoder output
pub struct Vae3dEncoderOutput<B: Backend> {
    /// Latent mean
    pub mean: Tensor<B, 5>,
    /// Latent log variance
    pub logvar: Tensor<B, 5>,
}

impl<B: Backend> Vae3dEncoderOutput<B> {
    /// Sample from the latent distribution
    pub fn sample(&self) -> Tensor<B, 5> {
        let std = (self.logvar.clone() * 0.5).exp();
        let noise = Tensor::random_like(&self.mean, burn::tensor::Distribution::Normal(0.0, 1.0));
        self.mean.clone() + std * noise
    }

    /// Get deterministic latent (just the mean)
    pub fn deterministic(&self) -> Tensor<B, 5> {
        self.mean.clone()
    }
}

/// Statistics for video VAE operations
#[derive(Debug, Clone)]
pub struct Vae3dStats {
    pub input_shape: [usize; 5],
    pub latent_shape: [usize; 5],
    pub compression_ratio: f32,
    pub temporal_compression: usize,
    pub spatial_compression: usize,
}

impl Vae3dStats {
    pub fn new(
        config: &Vae3dConfig,
        batch: usize,
        time: usize,
        height: usize,
        width: usize,
    ) -> Self {
        let (lat_t, lat_h, lat_w) = config.encoded_shape(time, height, width);

        let input_elements = batch * config.in_channels * time * height * width;
        let latent_elements = batch * config.latent_channels * lat_t * lat_h * lat_w;

        Self {
            input_shape: [batch, config.in_channels, time, height, width],
            latent_shape: [batch, config.latent_channels, lat_t, lat_h, lat_w],
            compression_ratio: input_elements as f32 / latent_elements as f32,
            temporal_compression: config.temporal_compression,
            spatial_compression: config.spatial_compression,
        }
    }
}

// CubeCL-accelerated 3D VAE implementation
#[cfg(feature = "cubecl")]
pub mod cubecl {
    //! CubeCL-accelerated 3D VAE encoder/decoder
    //!
    //! Uses fused Conv3d, GroupNorm+SiLU kernels for efficient video encoding/decoding.

    use super::*;
    use burn_cubecl::{CubeBackend, CubeRuntime, tensor::CubeTensor};
    use sketchpad_cubecl::{
        Conv3dLayer, Conv3dOptions, GroupNormSiLuOptions, cube_to_tensor, groupnorm_silu,
        tensor_to_cube,
    };

    type B<R> = CubeBackend<R, f32, i32, u32>;

    /// 3D ResNet block with CubeCL fused operations
    #[derive(Debug)]
    pub struct ResBlock3dCubeCL<R: CubeRuntime> {
        norm1_weight: CubeTensor<R>,
        norm1_bias: CubeTensor<R>,
        conv1: Conv3dLayer<R>,
        norm2_weight: CubeTensor<R>,
        norm2_bias: CubeTensor<R>,
        conv2: Conv3dLayer<R>,
        skip_conv: Option<Conv3dLayer<R>>,
        num_groups: usize,
    }

    impl<R: CubeRuntime> ResBlock3dCubeCL<R> {
        /// Creates a new 3D residual block
        pub fn new(in_channels: usize, out_channels: usize, device: &R::Device) -> Self {
            let num_groups = 32.min(in_channels);

            // GroupNorm weights
            let norm1_weight = burn::tensor::Tensor::<B<R>, 1>::ones([in_channels], device);
            let norm1_bias = burn::tensor::Tensor::<B<R>, 1>::zeros([in_channels], device);
            let norm2_weight = burn::tensor::Tensor::<B<R>, 1>::ones([out_channels], device);
            let norm2_bias = burn::tensor::Tensor::<B<R>, 1>::zeros([out_channels], device);

            // Conv3d weights (Xavier init)
            let conv1_weight = burn::tensor::Tensor::<B<R>, 5>::random(
                [out_channels, in_channels, 3, 3, 3],
                burn::tensor::Distribution::Uniform(
                    -((6.0 / (in_channels + out_channels) as f64).sqrt()),
                    (6.0 / (in_channels + out_channels) as f64).sqrt(),
                ),
                device,
            );
            let conv1_bias = burn::tensor::Tensor::<B<R>, 1>::zeros([out_channels], device);

            let conv2_weight = burn::tensor::Tensor::<B<R>, 5>::random(
                [out_channels, out_channels, 3, 3, 3],
                burn::tensor::Distribution::Uniform(
                    -((6.0 / (out_channels * 2) as f64).sqrt()),
                    (6.0 / (out_channels * 2) as f64).sqrt(),
                ),
                device,
            );
            let conv2_bias = burn::tensor::Tensor::<B<R>, 1>::zeros([out_channels], device);

            let conv1 = Conv3dLayer::new(
                tensor_to_cube(conv1_weight),
                Some(tensor_to_cube(conv1_bias)),
                Conv3dOptions {
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            );

            let conv2 = Conv3dLayer::new(
                tensor_to_cube(conv2_weight),
                Some(tensor_to_cube(conv2_bias)),
                Conv3dOptions {
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            );

            let skip_conv = if in_channels != out_channels {
                let skip_weight = burn::tensor::Tensor::<B<R>, 5>::random(
                    [out_channels, in_channels, 1, 1, 1],
                    burn::tensor::Distribution::Uniform(
                        -((6.0 / (in_channels + out_channels) as f64).sqrt()),
                        (6.0 / (in_channels + out_channels) as f64).sqrt(),
                    ),
                    device,
                );
                Some(Conv3dLayer::new(
                    tensor_to_cube(skip_weight),
                    None,
                    Conv3dOptions::default(),
                ))
            } else {
                None
            };

            Self {
                norm1_weight: tensor_to_cube(norm1_weight),
                norm1_bias: tensor_to_cube(norm1_bias),
                conv1,
                norm2_weight: tensor_to_cube(norm2_weight),
                norm2_bias: tensor_to_cube(norm2_bias),
                conv2,
                skip_conv,
                num_groups,
            }
        }

        /// Forward pass with fused GroupNorm+SiLU
        pub fn forward(&self, x: CubeTensor<R>) -> CubeTensor<R> {
            let residual = match &self.skip_conv {
                Some(conv) => conv.forward(x.clone()).expect("skip conv failed"),
                None => x.clone(),
            };

            // First block: GroupNorm+SiLU -> Conv3d
            let h = groupnorm_silu(
                x,
                self.norm1_weight.clone(),
                self.norm1_bias.clone(),
                GroupNormSiLuOptions::with_groups(self.num_groups),
            );
            let h = self.conv1.forward(h).expect("conv1 failed");

            // Second block: GroupNorm+SiLU -> Conv3d
            let h = groupnorm_silu(
                h,
                self.norm2_weight.clone(),
                self.norm2_bias.clone(),
                GroupNormSiLuOptions::with_groups(self.num_groups),
            );
            let h = self.conv2.forward(h).expect("conv2 failed");

            // Add residual - need to convert for element-wise ops
            let h_tensor: burn::tensor::Tensor<B<R>, 5> = cube_to_tensor(h);
            let r_tensor: burn::tensor::Tensor<B<R>, 5> = cube_to_tensor(residual);
            tensor_to_cube(h_tensor + r_tensor)
        }
    }

    /// Downsample block using strided Conv3d
    #[derive(Debug)]
    pub struct Downsample3dCubeCL<R: CubeRuntime> {
        conv: Conv3dLayer<R>,
    }

    impl<R: CubeRuntime> Downsample3dCubeCL<R> {
        /// Create a new downsample block
        ///
        /// * `temporal_stride` - Temporal downsampling factor (usually 2 or 1)
        /// * `spatial_stride` - Spatial downsampling factor (usually 2)
        pub fn new(
            channels: usize,
            temporal_stride: usize,
            spatial_stride: usize,
            device: &R::Device,
        ) -> Self {
            let weight = burn::tensor::Tensor::<B<R>, 5>::random(
                [channels, channels, 3, 3, 3],
                burn::tensor::Distribution::Uniform(
                    -((6.0 / (channels * 2) as f64).sqrt()),
                    (6.0 / (channels * 2) as f64).sqrt(),
                ),
                device,
            );
            let bias = burn::tensor::Tensor::<B<R>, 1>::zeros([channels], device);

            let conv = Conv3dLayer::new(
                tensor_to_cube(weight),
                Some(tensor_to_cube(bias)),
                Conv3dOptions {
                    stride: [temporal_stride, spatial_stride, spatial_stride],
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            );

            Self { conv }
        }

        pub fn forward(&self, x: CubeTensor<R>) -> CubeTensor<R> {
            self.conv.forward(x).expect("downsample conv failed")
        }
    }

    /// Upsample block using repeat + Conv3d
    #[derive(Debug)]
    pub struct Upsample3dCubeCL<R: CubeRuntime> {
        conv: Conv3dLayer<R>,
        temporal_factor: usize,
        spatial_factor: usize,
    }

    impl<R: CubeRuntime> Upsample3dCubeCL<R> {
        pub fn new(
            channels: usize,
            temporal_factor: usize,
            spatial_factor: usize,
            device: &R::Device,
        ) -> Self {
            let weight = burn::tensor::Tensor::<B<R>, 5>::random(
                [channels, channels, 3, 3, 3],
                burn::tensor::Distribution::Uniform(
                    -((6.0 / (channels * 2) as f64).sqrt()),
                    (6.0 / (channels * 2) as f64).sqrt(),
                ),
                device,
            );
            let bias = burn::tensor::Tensor::<B<R>, 1>::zeros([channels], device);

            let conv = Conv3dLayer::new(
                tensor_to_cube(weight),
                Some(tensor_to_cube(bias)),
                Conv3dOptions {
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            );

            Self {
                conv,
                temporal_factor,
                spatial_factor,
            }
        }

        pub fn forward(&self, x: CubeTensor<R>) -> CubeTensor<R> {
            // Upsample using tensor ops, then conv
            let tensor: burn::tensor::Tensor<B<R>, 5> = cube_to_tensor(x);
            let [b, c, t, h, w] = tensor.dims();

            // Temporal upsample
            let tensor = if self.temporal_factor > 1 {
                tensor.repeat_dim(2, self.temporal_factor)
            } else {
                tensor
            };

            // Spatial upsample (reshape trick for nearest neighbor)
            let tensor = if self.spatial_factor > 1 {
                let new_h = h * self.spatial_factor;
                let new_w = w * self.spatial_factor;
                let new_t = t * self.temporal_factor;

                tensor
                    .reshape([b, c, new_t, h, 1, w, 1])
                    .repeat_dim(4, self.spatial_factor)
                    .repeat_dim(6, self.spatial_factor)
                    .reshape([b, c, new_t, new_h, new_w])
            } else {
                tensor
            };

            let upsampled = tensor_to_cube(tensor);
            self.conv.forward(upsampled).expect("upsample conv failed")
        }
    }

    /// CubeCL-accelerated 3D VAE Encoder
    #[derive(Debug)]
    #[allow(clippy::type_complexity)]
    pub struct Vae3dEncoderCubeCL<R: CubeRuntime> {
        conv_in: Conv3dLayer<R>,
        down_blocks: Vec<(Vec<ResBlock3dCubeCL<R>>, Option<Downsample3dCubeCL<R>>)>,
        mid_block1: ResBlock3dCubeCL<R>,
        mid_block2: ResBlock3dCubeCL<R>,
        norm_out_weight: CubeTensor<R>,
        norm_out_bias: CubeTensor<R>,
        conv_out: Conv3dLayer<R>,
        /// VAE configuration
        pub config: Vae3dConfig,
    }

    impl<R: CubeRuntime> Vae3dEncoderCubeCL<R> {
        pub fn new(config: Vae3dConfig, device: &R::Device) -> Self {
            let mut channels = config.base_channels;

            // Input conv
            let conv_in_weight = burn::tensor::Tensor::<B<R>, 5>::random(
                [channels, config.in_channels, 3, 3, 3],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            );
            let conv_in_bias = burn::tensor::Tensor::<B<R>, 1>::zeros([channels], device);
            let conv_in = Conv3dLayer::new(
                tensor_to_cube(conv_in_weight),
                Some(tensor_to_cube(conv_in_bias)),
                Conv3dOptions {
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            );

            // Down blocks
            let mut down_blocks = Vec::new();
            for (i, &mult) in config.channel_mults.iter().enumerate() {
                let out_ch = config.base_channels * mult;
                let is_last = i == config.channel_mults.len() - 1;

                let mut res_blocks = Vec::new();
                for _ in 0..config.num_res_blocks {
                    res_blocks.push(ResBlock3dCubeCL::new(channels, out_ch, device));
                    channels = out_ch;
                }

                let downsample = if !is_last {
                    // Downsample spatially by 2, temporally every other level
                    let temporal_stride = if i % 2 == 0 { 2 } else { 1 };
                    Some(Downsample3dCubeCL::new(
                        channels,
                        temporal_stride,
                        2,
                        device,
                    ))
                } else {
                    None
                };

                down_blocks.push((res_blocks, downsample));
            }

            // Mid blocks
            let mid_block1 = ResBlock3dCubeCL::new(channels, channels, device);
            let mid_block2 = ResBlock3dCubeCL::new(channels, channels, device);

            // Output (mean + logvar)
            let norm_out_weight = burn::tensor::Tensor::<B<R>, 1>::ones([channels], device);
            let norm_out_bias = burn::tensor::Tensor::<B<R>, 1>::zeros([channels], device);

            let conv_out_weight = burn::tensor::Tensor::<B<R>, 5>::random(
                [config.latent_channels * 2, channels, 3, 3, 3],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            );
            let conv_out_bias =
                burn::tensor::Tensor::<B<R>, 1>::zeros([config.latent_channels * 2], device);
            let conv_out = Conv3dLayer::new(
                tensor_to_cube(conv_out_weight),
                Some(tensor_to_cube(conv_out_bias)),
                Conv3dOptions {
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            );

            Self {
                conv_in,
                down_blocks,
                mid_block1,
                mid_block2,
                norm_out_weight: tensor_to_cube(norm_out_weight),
                norm_out_bias: tensor_to_cube(norm_out_bias),
                conv_out,
                config,
            }
        }

        /// Encode video to latent distribution
        pub fn forward(&self, x: burn::tensor::Tensor<B<R>, 5>) -> Vae3dEncoderOutput<B<R>> {
            let mut h = tensor_to_cube(x);

            // Input conv
            h = self.conv_in.forward(h).expect("conv_in failed");

            // Down blocks
            for (res_blocks, downsample) in &self.down_blocks {
                for block in res_blocks {
                    h = block.forward(h);
                }
                if let Some(down) = downsample {
                    h = down.forward(h);
                }
            }

            // Mid blocks
            h = self.mid_block1.forward(h);
            h = self.mid_block2.forward(h);

            // Output
            h = groupnorm_silu(
                h,
                self.norm_out_weight.clone(),
                self.norm_out_bias.clone(),
                GroupNormSiLuOptions::with_groups(32),
            );
            h = self.conv_out.forward(h).expect("conv_out failed");

            // Split into mean and logvar
            let tensor: burn::tensor::Tensor<B<R>, 5> = cube_to_tensor(h);
            let [b, c, t, height, w] = tensor.dims();
            let half_c = c / 2;

            let mean = tensor
                .clone()
                .slice([0..b, 0..half_c, 0..t, 0..height, 0..w]);
            let logvar = tensor.slice([0..b, half_c..c, 0..t, 0..height, 0..w]);

            Vae3dEncoderOutput { mean, logvar }
        }
    }

    /// CubeCL-accelerated 3D VAE Decoder
    #[derive(Debug)]
    #[allow(clippy::type_complexity)]
    pub struct Vae3dDecoderCubeCL<R: CubeRuntime> {
        conv_in: Conv3dLayer<R>,
        mid_block1: ResBlock3dCubeCL<R>,
        mid_block2: ResBlock3dCubeCL<R>,
        up_blocks: Vec<(Vec<ResBlock3dCubeCL<R>>, Option<Upsample3dCubeCL<R>>)>,
        norm_out_weight: CubeTensor<R>,
        norm_out_bias: CubeTensor<R>,
        conv_out: Conv3dLayer<R>,
    }

    impl<R: CubeRuntime> Vae3dDecoderCubeCL<R> {
        pub fn new(config: Vae3dConfig, device: &R::Device) -> Self {
            let final_mult = *config.channel_mults.last().unwrap_or(&1);
            let mut channels = config.base_channels * final_mult;

            // Input conv (from latent)
            let conv_in_weight = burn::tensor::Tensor::<B<R>, 5>::random(
                [channels, config.latent_channels, 3, 3, 3],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            );
            let conv_in_bias = burn::tensor::Tensor::<B<R>, 1>::zeros([channels], device);
            let conv_in = Conv3dLayer::new(
                tensor_to_cube(conv_in_weight),
                Some(tensor_to_cube(conv_in_bias)),
                Conv3dOptions {
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            );

            // Mid blocks
            let mid_block1 = ResBlock3dCubeCL::new(channels, channels, device);
            let mid_block2 = ResBlock3dCubeCL::new(channels, channels, device);

            // Up blocks (reversed)
            let mut up_blocks = Vec::new();
            for (i, &mult) in config.channel_mults.iter().rev().enumerate() {
                let out_ch = config.base_channels * mult;
                let is_last = i == config.channel_mults.len() - 1;

                let mut res_blocks = Vec::new();
                for _ in 0..config.num_res_blocks + 1 {
                    res_blocks.push(ResBlock3dCubeCL::new(channels, out_ch, device));
                    channels = out_ch;
                }

                let upsample = if !is_last {
                    let temporal_factor = if i % 2 == 0 { 2 } else { 1 };
                    Some(Upsample3dCubeCL::new(channels, temporal_factor, 2, device))
                } else {
                    None
                };

                up_blocks.push((res_blocks, upsample));
            }

            // Output
            let norm_out_weight = burn::tensor::Tensor::<B<R>, 1>::ones([channels], device);
            let norm_out_bias = burn::tensor::Tensor::<B<R>, 1>::zeros([channels], device);

            let conv_out_weight = burn::tensor::Tensor::<B<R>, 5>::random(
                [config.in_channels, channels, 3, 3, 3],
                burn::tensor::Distribution::Uniform(-0.1, 0.1),
                device,
            );
            let conv_out_bias =
                burn::tensor::Tensor::<B<R>, 1>::zeros([config.in_channels], device);
            let conv_out = Conv3dLayer::new(
                tensor_to_cube(conv_out_weight),
                Some(tensor_to_cube(conv_out_bias)),
                Conv3dOptions {
                    padding: [1, 1, 1],
                    ..Default::default()
                },
            );

            Self {
                conv_in,
                mid_block1,
                mid_block2,
                up_blocks,
                norm_out_weight: tensor_to_cube(norm_out_weight),
                norm_out_bias: tensor_to_cube(norm_out_bias),
                conv_out,
            }
        }

        /// Decode latent to video
        pub fn forward(&self, z: burn::tensor::Tensor<B<R>, 5>) -> burn::tensor::Tensor<B<R>, 5> {
            let mut h = tensor_to_cube(z);

            // Input conv
            h = self.conv_in.forward(h).expect("conv_in failed");

            // Mid blocks
            h = self.mid_block1.forward(h);
            h = self.mid_block2.forward(h);

            // Up blocks
            for (res_blocks, upsample) in &self.up_blocks {
                for block in res_blocks {
                    h = block.forward(h);
                }
                if let Some(up) = upsample {
                    h = up.forward(h);
                }
            }

            // Output
            h = groupnorm_silu(
                h,
                self.norm_out_weight.clone(),
                self.norm_out_bias.clone(),
                GroupNormSiLuOptions::with_groups(32),
            );
            h = self.conv_out.forward(h).expect("conv_out failed");

            cube_to_tensor(h)
        }
    }

    /// Complete 3D VAE (encoder + decoder)
    #[derive(Debug)]
    pub struct Vae3dCubeCL<R: CubeRuntime> {
        pub encoder: Vae3dEncoderCubeCL<R>,
        pub decoder: Vae3dDecoderCubeCL<R>,
        pub config: Vae3dConfig,
    }

    impl<R: CubeRuntime> Vae3dCubeCL<R> {
        pub fn new(config: Vae3dConfig, device: &R::Device) -> Self {
            Self {
                encoder: Vae3dEncoderCubeCL::new(config.clone(), device),
                decoder: Vae3dDecoderCubeCL::new(config.clone(), device),
                config,
            }
        }

        /// Encode video to latent
        pub fn encode(&self, x: burn::tensor::Tensor<B<R>, 5>) -> Vae3dEncoderOutput<B<R>> {
            self.encoder.forward(x)
        }

        /// Decode latent to video
        pub fn decode(&self, z: burn::tensor::Tensor<B<R>, 5>) -> burn::tensor::Tensor<B<R>, 5> {
            self.decoder.forward(z)
        }
    }

    /// Weight key mapping for 3D VAE models
    ///
    /// Different video generation models use different key naming conventions.
    /// This enum provides mappings for common models.
    #[derive(Debug, Clone, Default)]
    pub enum Vae3dKeyMapping {
        /// CogVideoX naming: `encoder.down_blocks.0.resnets.0.conv1.weight`
        #[default]
        CogVideoX,
        /// Mochi naming (similar to CogVideoX)
        Mochi,
        /// Custom mapping function
        Custom(fn(&str) -> String),
    }

    impl Vae3dKeyMapping {
        /// Map a canonical key to the model's key format
        pub fn map_key(&self, key: &str) -> String {
            match self {
                Vae3dKeyMapping::CogVideoX => self.map_cogvideox(key),
                Vae3dKeyMapping::Mochi => self.map_mochi(key),
                Vae3dKeyMapping::Custom(f) => f(key),
            }
        }

        fn map_cogvideox(&self, key: &str) -> String {
            // CogVideoX uses a similar naming to Stable Diffusion VAE
            // Example mappings:
            // encoder.conv_in.weight -> encoder.conv_in.weight
            // encoder.down.0.block.0.norm1.weight -> encoder.down_blocks.0.resnets.0.norm1.weight
            // decoder.up.0.block.0.conv1.weight -> decoder.up_blocks.0.resnets.0.conv1.weight
            key.replace("down.", "down_blocks.")
                .replace("up.", "up_blocks.")
                .replace("block.", "resnets.")
        }

        fn map_mochi(&self, key: &str) -> String {
            // Mochi has its own naming convention
            // Similar to CogVideoX for now
            self.map_cogvideox(key)
        }
    }

    /// Information about expected VAE weights
    #[derive(Debug)]
    pub struct Vae3dWeightInfo {
        /// Canonical key name
        pub key: String,
        /// Expected shape
        pub shape: Vec<usize>,
        /// Description
        pub description: &'static str,
    }

    /// Get expected weights for a VAE configuration
    pub fn expected_vae3d_weights(config: &Vae3dConfig) -> Vec<Vae3dWeightInfo> {
        let mut weights = Vec::new();
        let base = config.base_channels;

        // Encoder conv_in
        weights.push(Vae3dWeightInfo {
            key: "encoder.conv_in.weight".into(),
            shape: vec![base, config.in_channels, 3, 3, 3],
            description: "Encoder input conv weight",
        });
        weights.push(Vae3dWeightInfo {
            key: "encoder.conv_in.bias".into(),
            shape: vec![base],
            description: "Encoder input conv bias",
        });

        // Down blocks
        let mut channels = base;
        for (level, &mult) in config.channel_mults.iter().enumerate() {
            let out_ch = base * mult;

            for block in 0..config.num_res_blocks {
                let in_ch = if block == 0 { channels } else { out_ch };

                // ResBlock weights
                for suffix in ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"] {
                    weights.push(Vae3dWeightInfo {
                        key: format!("encoder.down.{level}.block.{block}.{suffix}"),
                        shape: vec![if suffix.starts_with("norm1") {
                            in_ch
                        } else {
                            out_ch
                        }],
                        description: "GroupNorm affine params",
                    });
                }
            }
            channels = out_ch;
        }

        // Decoder weights follow similar pattern...
        // (Abbreviated for brevity - full implementation would include all weights)

        weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn test_config_default() {
        let config = Vae3dConfig::default();
        assert_eq!(config.in_channels, 3);
        assert_eq!(config.temporal_compression, 4);
        assert_eq!(config.spatial_compression, 8);
    }

    #[test]
    fn test_encoded_shape() {
        let config = Vae3dConfig::default();
        let (t, h, w) = config.encoded_shape(16, 512, 512);
        assert_eq!(t, 4); // 16 / 4
        assert_eq!(h, 64); // 512 / 8
        assert_eq!(w, 64); // 512 / 8
    }

    #[test]
    fn test_temporal_downsample() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 5>::zeros([1, 3, 16, 64, 64], &device);

        let down = temporal_downsample(x, 4);
        assert_eq!(down.dims(), [1, 3, 4, 64, 64]);
    }

    #[test]
    fn test_temporal_upsample() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 64, 64], &device);

        let up = temporal_upsample(x, 4);
        assert_eq!(up.dims(), [1, 4, 16, 64, 64]);
    }

    #[test]
    fn test_spatial_downsample_3d() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 5>::zeros([1, 3, 4, 64, 64], &device);

        let down = spatial_downsample_3d(x, 2);
        assert_eq!(down.dims(), [1, 3, 4, 32, 32]);
    }

    #[test]
    fn test_spatial_upsample_3d() {
        let device = Default::default();
        let x = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 32, 32], &device);

        let up = spatial_upsample_3d(x, 2);
        assert_eq!(up.dims(), [1, 4, 4, 64, 64]);
    }

    #[test]
    fn test_vae3d_stats() {
        let config = Vae3dConfig::default();
        let stats = Vae3dStats::new(&config, 1, 16, 512, 512);

        assert_eq!(stats.input_shape, [1, 3, 16, 512, 512]);
        assert_eq!(stats.latent_shape, [1, 4, 4, 64, 64]);
        // 3*16*512*512 / (4*4*64*64) = 12,582,912 / 65,536 = 192
        assert!(stats.compression_ratio > 100.0);
    }

    #[test]
    fn test_encoder_output_sample() {
        let device = Default::default();
        let mean = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 64, 64], &device);
        let logvar = Tensor::<TestBackend, 5>::zeros([1, 4, 4, 64, 64], &device);

        let output = Vae3dEncoderOutput { mean, logvar };

        let sampled = output.sample();
        assert_eq!(sampled.dims(), [1, 4, 4, 64, 64]);

        let det = output.deterministic();
        assert_eq!(det.dims(), [1, 4, 4, 64, 64]);
    }
}
