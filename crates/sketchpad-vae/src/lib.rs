//! Variational Autoencoder (VAE) for Latent Diffusion
//!
//! This crate provides the VAE encoder and decoder used in Stable Diffusion
//! to convert between pixel space and latent space.
//!
//! # Components
//!
//! - [`Encoder`] - Compresses images to latent representations
//! - [`Decoder`] - Reconstructs images from latents
//!
//! # Scaling Factors
//!
//! Different models use different latent scaling:
//! - SD 1.x: `0.18215`
//! - SDXL: `0.13025`
//!
//! Use [`scaling`] utilities for correct scaling.
//!
//! # Example
//!
//! ```ignore
//! use sketchpad_vae::{Decoder, DecoderConfig, scaling};
//!
//! let config = DecoderConfig::sd1x();
//! let decoder = config.init::<Backend>(&device);
//!
//! // Decode latents to image
//! let latents = latents / scaling::SD1X_SCALE_FACTOR;
//! let image = decoder.forward(latents);
//! ```

pub mod autoencoder;
pub mod decoder;
pub mod encoder;

pub use decoder::{
    Decoder, DecoderBlock, DecoderConfig, ResnetBlock, SelfAttention, Upsample, scaling,
};
pub use encoder::{Downsample as EncoderDownsample, Encoder, EncoderBlock, EncoderConfig};
