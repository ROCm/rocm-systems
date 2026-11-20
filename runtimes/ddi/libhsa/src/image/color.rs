// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA clear values and stored color conversion.

use std::ffi::c_void;

use crate::ffi::*;

/// Converts an access value to standard RGB while keeping alpha linear.
fn linear_to_srgb(value: f32) -> f32 {
    let value = value.clamp(0.0, 1.0);
    if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

/// Converts a stored standard-RGB component into a linear access value.
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.040_45 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Converts the three color components of tightly packed RGBA8 texels.
pub(super) fn convert_srgb_pixels(pixels: &mut [u8], to_srgb: bool) {
    for pixel in pixels.chunks_exact_mut(4) {
        for component in &mut pixel[..3] {
            let value = f32::from(*component) / 255.0;
            let converted = if to_srgb {
                linear_to_srgb(value)
            } else {
                srgb_to_linear(value)
            };
            *component = (converted * 255.0).round_ties_even() as u8;
        }
    }
}

/// Converts a 32-bit float to its 16-bit image channel encoding.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = i32::try_from((bits >> 23) & 0xff).unwrap_or(0);
    let mantissa = bits & 0x007f_ffff;
    if exponent == 255 {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let truncated = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = truncated
            + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0));
        return sign | rounded as u16;
    }
    let truncated = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    let rounded =
        truncated + u32::from(remainder > 0x1000 || (remainder == 0x1000 && truncated & 1 != 0));
    let mut exponent = half_exponent as u32;
    let mantissa = if rounded == 0x400 {
        exponent += 1;
        0
    } else {
        rounded
    };
    if exponent >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exponent as u16) << 10) | mantissa as u16
    }
}

#[allow(unsafe_code)]
/// Encodes a clear color in the image channel format.
pub(super) unsafe fn clear_pixel(
    image_format: HsaExtImageFormat,
    bytes: usize,
    data: *const c_void,
) -> Result<[u8; 16], Status> {
    let order = image_format.channel_order;
    // Clear colors use image access order (RGBA), independently of the
    // storage channel order. Depth, luminance and intensity use the first value.
    let channels: &[usize] = match order {
        0 => &[3],
        1 | 16..=18 => &[0],
        3 => &[0, 1],
        5 => &[0, 3],
        6 => &[0, 1, 2],
        8 | 14 => &[0, 1, 2, 3],
        9 => &[2, 1, 0, 3],
        10 => &[3, 0, 1, 2],
        _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
    };
    let mut pixel = [0_u8; 16];
    let channel_type = image_format.channel_type;
    if matches!(channel_type, 5..=7) {
        let fields: &[(usize, u32, u16)] = match (order, channel_type) {
            (6, 5) => &[(0, 10, 31), (1, 5, 31), (2, 0, 31)],
            (6, 6) => &[(0, 11, 31), (1, 5, 63), (2, 0, 31)],
            (6, 7) => &[(0, 20, 1023), (1, 10, 1023), (2, 0, 1023)],
            (8, 7) => &[(0, 0, 1023), (1, 10, 1023), (2, 20, 1023), (3, 30, 3)],
            _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
        };
        let mut packed = 0;
        for &(component, shift, maximum) in fields {
            // SAFETY: Packed normalized formats receive floating access values.
            let value = unsafe { data.cast::<f32>().add(component).read_unaligned() };
            packed |=
                ((value.clamp(0.0, 1.0) * f32::from(maximum)).round_ties_even() as u32) << shift;
        }
        pixel[..bytes].copy_from_slice(&packed.to_le_bytes()[..bytes]);
        return Ok(pixel);
    }
    let component_bytes = bytes / channels.len();
    for (index, &component) in channels.iter().enumerate() {
        let encoded = if matches!(channel_type, 0..=3 | 14 | 15) {
            // SAFETY: The caller provides the image format's access components.
            let mut value = unsafe { data.cast::<f32>().add(component).read_unaligned() };
            if order == 14 && component != 3 {
                value = linear_to_srgb(value);
            }
            match channel_type {
                0 => u32::from(((value.clamp(-1.0, 1.0) * 127.0).round_ties_even() as i8) as u8),
                1 => {
                    u32::from(((value.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16) as u16)
                }
                2 => (value.clamp(0.0, 1.0) * 255.0).round_ties_even() as u32,
                3 => (value.clamp(0.0, 1.0) * 65535.0).round_ties_even() as u32,
                14 => u32::from(f32_to_f16(value)),
                15 => value.to_bits(),
                _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
            }
        } else if matches!(channel_type, 8..=10) {
            // SAFETY: Signed image access values are 32-bit integers.
            let value = unsafe { data.cast::<i32>().add(component).read_unaligned() };
            match channel_type {
                8 => u32::from(value.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8 as u8),
                9 => u32::from(value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16 as u16),
                10 => value as u32,
                _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
            }
        } else {
            // SAFETY: Unsigned image access values are 32-bit integers.
            let value = unsafe { data.cast::<u32>().add(component).read_unaligned() };
            match channel_type {
                11 => value.min(u32::from(u8::MAX)),
                12 => value.min(u32::from(u16::MAX)),
                13 => value,
                _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
            }
        };
        let start = index * component_bytes;
        pixel[start..start + component_bytes]
            .copy_from_slice(&encoded.to_le_bytes()[..component_bytes]);
    }
    Ok(pixel)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn clear_colors_follow_storage_order() {
        let color = [0x11_u32, 0x22, 0x33, 0x44];
        for (order, expected) in [
            (0, &[0x44][..]),
            (1, &[0x11][..]),
            (3, &[0x11, 0x22][..]),
            (5, &[0x11, 0x44][..]),
            (8, &[0x11, 0x22, 0x33, 0x44][..]),
            (9, &[0x33, 0x22, 0x11, 0x44][..]),
            (10, &[0x44, 0x11, 0x22, 0x33][..]),
        ] {
            let format = HsaExtImageFormat {
                channel_order: order,
                channel_type: 11,
            };
            let encoded =
                unsafe { clear_pixel(format, expected.len(), color.as_ptr().cast()) }.unwrap();
            assert_eq!(&encoded[..expected.len()], expected);
        }
    }

    #[test]
    fn clear_packed_formats_place_each_channel() {
        let color = [1.0_f32, 0.0, 0.5, 1.0];
        for (order, channel_type, bytes, expected) in [
            (6, 5, 2, 0x7c10_u32),
            (6, 6, 2, 0xf810),
            (6, 7, 4, (1023 << 20) | 512),
            (8, 7, 4, (3 << 30) | (512 << 20) | 1023),
        ] {
            let format = HsaExtImageFormat {
                channel_order: order,
                channel_type,
            };
            let encoded = unsafe { clear_pixel(format, bytes, color.as_ptr().cast()) }.unwrap();
            assert_eq!(&encoded[..bytes], &expected.to_le_bytes()[..bytes]);
        }
    }

    #[test]
    fn srgb_clear_and_copy_convert_rgb_and_preserve_alpha() {
        let color = [0.0_f32, 0.5, 1.0, 0.5];
        let format = HsaExtImageFormat {
            channel_order: 14,
            channel_type: 2,
        };
        let encoded = unsafe { clear_pixel(format, 4, color.as_ptr().cast()) }.unwrap();
        assert_eq!(&encoded[..4], &[0, 188, 255, 128]);
        let mut pixels = [0, 128, 255, 37, 64, 32, 16, 99];
        let original = pixels;
        convert_srgb_pixels(&mut pixels, true);
        assert_eq!(&pixels[..4], &[0, 188, 255, 37]);
        assert_eq!(pixels[7], 99);
        convert_srgb_pixels(&mut pixels, false);
        for (actual, expected) in pixels.into_iter().zip(original) {
            assert!(actual.abs_diff(expected) <= 1);
        }
    }
}
