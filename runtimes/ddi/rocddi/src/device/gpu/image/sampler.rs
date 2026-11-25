// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU sampler descriptors from frontend-independent, representable settings.

use super::Generation;

/// Coordinate handling outside an image dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerAddressMode {
    /// Repeat the image periodically.
    Repeat,
    /// Clamp coordinates to the nearest edge texel.
    ClampEdge,
    /// Use the border color outside the image.
    ClampBorder,
    /// Mirror every other repetition of the image.
    MirroredRepeat,
}

impl SamplerAddressMode {
    const fn encoding(self) -> u32 {
        match self {
            Self::Repeat => 0,
            Self::ClampEdge => 2,
            Self::ClampBorder => 6,
            Self::MirroredRepeat => 1,
        }
    }
}

/// Filtering within a mip level.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerFilter {
    /// Select the nearest texel.
    Point,
    /// Interpolate neighboring texels.
    Linear,
}

/// Filtering between mip levels.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerMipFilter {
    /// Access only the base mip level.
    Disabled,
    /// Select the nearest mip level.
    Point,
    /// Interpolate neighboring mip levels.
    Linear,
}

/// Interpretation of image coordinates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerCoordinates {
    /// Coordinates are expressed in texels.
    Unnormalized,
    /// Coordinates are relative to image dimensions.
    Normalized,
}

/// A representable maximum anisotropy ratio.
///
/// The frontend owns normalization of requests that are not powers of two.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SamplerAnisotropy {
    /// Isotropic filtering.
    One,
    /// Maximum anisotropy ratio of 2:1.
    Two,
    /// Maximum anisotropy ratio of 4:1.
    Four,
    /// Maximum anisotropy ratio of 8:1.
    Eight,
    /// Maximum anisotropy ratio of 16:1.
    Sixteen,
}

impl SamplerAnisotropy {
    const fn log2(self) -> u32 {
        match self {
            Self::One => 0,
            Self::Two => 1,
            Self::Four => 2,
            Self::Eight => 3,
            Self::Sixteen => 4,
        }
    }
}

/// Validated sampling settings without public API enums or handles.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SamplerDescription {
    /// Addressing for the x, y, and z dimensions.
    pub address: [SamplerAddressMode; 3],
    /// Coordinate normalization.
    pub coordinates: SamplerCoordinates,
    /// Minification and magnification filter.
    pub filter: SamplerFilter,
    /// Mip-level filter.
    pub mip_filter: SamplerMipFilter,
    /// Representable maximum anisotropy selected by the frontend.
    pub anisotropy: SamplerAnisotropy,
}

/// Encodes a sampler for the specified resource-descriptor generation.
///
/// Every input value is representable. The descriptor uses the full mip range
/// and zero LOD bias; image capability and API policy checks belong to callers.
#[must_use]
pub fn sampler_descriptor(generation: Generation, description: SamplerDescription) -> [u32; 8] {
    let anisotropy = description.anisotropy.log2();
    let word0 = description.address[0].encoding()
        | description.address[1].encoding() << 3
        | description.address[2].encoding() << 6
        | anisotropy << 9
        | u32::from(description.coordinates == SamplerCoordinates::Unnormalized) << 15
        | (anisotropy >> 1) << 16
        | anisotropy << 21;
    let word1 = match generation {
        Generation::Gfx12 => 4095 << 13,
        Generation::Gfx9 | Generation::Gfx10 | Generation::Gfx103 | Generation::Gfx11 => 4095 << 12,
    };
    let filter =
        u32::from(description.filter == SamplerFilter::Linear) | (u32::from(anisotropy != 0) << 1);
    let mip_filter = match description.mip_filter {
        SamplerMipFilter::Disabled => 0,
        SamplerMipFilter::Point => 1,
        SamplerMipFilter::Linear => 2,
    };
    let word2 = (filter << 20) | (filter << 22) | (mip_filter << 26);
    [word0, word1, word2, 0, 0, 0, 0, 0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_words_preserve_each_register_generation() {
        let nearest = SamplerDescription {
            address: [SamplerAddressMode::Repeat; 3],
            coordinates: SamplerCoordinates::Normalized,
            filter: SamplerFilter::Point,
            mip_filter: SamplerMipFilter::Disabled,
            anisotropy: SamplerAnisotropy::One,
        };
        let anisotropic = SamplerDescription {
            address: [
                SamplerAddressMode::ClampEdge,
                SamplerAddressMode::ClampBorder,
                SamplerAddressMode::MirroredRepeat,
            ],
            coordinates: SamplerCoordinates::Unnormalized,
            filter: SamplerFilter::Linear,
            mip_filter: SamplerMipFilter::Linear,
            anisotropy: SamplerAnisotropy::Sixteen,
        };
        for (generation, lod_word) in [
            (Generation::Gfx9, 0x00ff_f000),
            (Generation::Gfx10, 0x00ff_f000),
            (Generation::Gfx103, 0x00ff_f000),
            (Generation::Gfx11, 0x00ff_f000),
            (Generation::Gfx12, 0x01ff_e000),
        ] {
            assert_eq!(
                sampler_descriptor(generation, nearest),
                [0, lod_word, 0, 0, 0, 0, 0, 0]
            );
            assert_eq!(
                sampler_descriptor(generation, anisotropic),
                [0x0082_8872, lod_word, 0x08f0_0000, 0, 0, 0, 0, 0]
            );
            assert_eq!(
                sampler_descriptor(
                    generation,
                    SamplerDescription {
                        mip_filter: SamplerMipFilter::Point,
                        filter: SamplerFilter::Linear,
                        ..nearest
                    }
                ),
                [0, lod_word, 0x0450_0000, 0, 0, 0, 0, 0]
            );
        }
    }
}
