// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native surface layout contracts for `AddrLib2` and `AddrLib3` families.

#![allow(clippy::unwrap_used, clippy::panic)]

use rocaddr3::{ResourceType, SurfaceMapper, SurfaceRequest, compute, compute_linear};

fn request(family: u32, resource_type: ResourceType) -> SurfaceRequest {
    SurfaceRequest {
        family,
        chip_revision: 1,
        gb_addr_config: 0,
        bits_per_pixel: 32,
        width: 64,
        height: 32,
        slices: 4,
        mip_levels: 5,
        resource_type,
    }
}

#[test]
fn each_supported_address_family_produces_complete_mip_storage() {
    for family in [
        0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x94, 0x95, 0x96, 0x97, 0x98, 0x9a,
    ] {
        for resource in [ResourceType::Texture2D, ResourceType::Texture3D] {
            let request = request(family, resource);
            let layout = compute(request)
                .unwrap_or_else(|error| panic!("family {family:x}, {resource:?}: {error:?}"));
            assert_eq!(layout.mips.len(), 5);
            assert!(layout.alignment.is_power_of_two());
            assert!(layout.size >= 64 * 32 * 4 * 4);
            assert!(layout.pitch >= request.width);
            assert!(layout.swizzle_mode != 0);
            for mip in &layout.mips {
                assert!(mip.offset < layout.size);
            }
        }
    }
}

#[test]
fn family_linear_layout_matches_its_row_and_slice_pitches() {
    for family in [0x8d, 0x8f, 0x91, 0x98] {
        let request = SurfaceRequest {
            mip_levels: 1,
            ..request(family, ResourceType::Texture2D)
        };
        let layout = compute_linear(request, 0).unwrap();
        assert_eq!(layout.swizzle_mode, 0);
        assert_eq!(layout.pitch, 64);
        assert_eq!(layout.slice_size, 64 * 32 * 4);
        assert_eq!(layout.size, 64 * 32 * 4 * 4);
        let single = SurfaceRequest {
            slices: 1,
            ..request
        };
        let explicit = compute_linear(single, 128).unwrap();
        assert_eq!(explicit.pitch, 128);
        assert!(explicit.size >= 128 * 32 * 4);
    }
}

#[test]
fn linear_layout_rejects_unsupported_explicit_pitches() {
    for family in [
        0x8d, 0x8e, 0x8f, 0x90, 0x91, 0x92, 0x94, 0x95, 0x96, 0x97, 0x98, 0x9a,
    ] {
        for bits_per_pixel in [8, 32, 128] {
            let block_elements = 256 / (bits_per_pixel / 8);
            let request = SurfaceRequest {
                bits_per_pixel,
                width: 17,
                height: 1,
                slices: 2,
                mip_levels: 1,
                ..request(family, ResourceType::Texture1D)
            };
            // At 32 bpp the 96-element case is the reported 384-byte pitch.
            for pitch in [1, block_elements * 3 / 2] {
                if family == 0x98 && pitch != 1 {
                    continue;
                }
                assert_eq!(
                    compute_linear(request, pitch),
                    Err(rocaddr3::Error::NoLayout),
                    "family {family:x}, bpp {bits_per_pixel}, pitch {pitch}"
                );
            }
            let pitch = block_elements * 2;
            let valid = compute_linear(request, pitch).unwrap();
            assert_eq!(valid.pitch, pitch);
            assert_eq!(valid.slice_size, 512);
            assert_eq!(valid.size, 1024);
            assert!(compute_linear(request, 0).is_ok());
            // An aligned pitch can still be smaller than the image width.
            let wider = SurfaceRequest {
                width: pitch + 1,
                ..request
            };
            assert_eq!(compute_linear(wider, pitch), Err(rocaddr3::Error::NoLayout));
        }
    }
}

#[test]
fn gfx12_linear_pitches_respect_row_and_slice_alignment() {
    for (resource_type, height, slices, small_slice, large_slice) in [
        (ResourceType::Texture1D, 1, 1, 256, 512),
        (ResourceType::Texture1D, 1, 2, 256, 512),
        (ResourceType::Texture2D, 1, 1, 256, 512),
        (ResourceType::Texture2D, 1, 2, 256, 512),
        (ResourceType::Texture2D, 2, 2, 512, 1024),
        (ResourceType::Texture2D, 3, 1, 512, 1280),
        (ResourceType::Texture2D, 3, 2, 768, 1536),
        (ResourceType::Texture3D, 1, 1, 256, 512),
        (ResourceType::Texture3D, 3, 2, 768, 1536),
    ] {
        // HIP's float-width-30 and short-width-179 allocations use 128-byte
        // and 384-byte rows. Also retain an oversized custom-pitch case.
        for (bits_per_pixel, width, pitch, expected_slice) in [
            (32, 30, 32, small_slice),
            (16, 179, 192, large_slice),
            (32, 17, 96, large_slice),
        ] {
            let request = SurfaceRequest {
                bits_per_pixel,
                width,
                height,
                slices,
                mip_levels: 1,
                ..request(0x98, resource_type)
            };
            let layout = compute_linear(request, pitch).unwrap();
            assert_eq!(layout.pitch, pitch);
            assert_eq!(layout.mips[0].pitch, pitch);
            assert_eq!(layout.slice_size, expected_slice);
            assert_eq!(layout.size, expected_slice * u64::from(slices));

            let mut mapper = SurfaceMapper::new(request, &layout).unwrap();
            let mut offsets = [0; 2];
            mapper
                .map_row(0, width - 2, height - 1, slices - 1, &mut offsets)
                .unwrap();
            let element_bytes = u64::from(bits_per_pixel / 8);
            let expected = u64::from(slices - 1) * expected_slice
                + (u64::from(height - 1) * u64::from(pitch) + u64::from(width - 2)) * element_bytes;
            assert_eq!(offsets, [expected, expected + element_bytes]);
            assert!(offsets[1] + element_bytes <= layout.size);

            assert_eq!(
                compute_linear(request, pitch - 1),
                Err(rocaddr3::Error::NoLayout)
            );
            assert_eq!(
                compute_linear(
                    SurfaceRequest {
                        width: pitch + 1,
                        ..request
                    },
                    pitch
                ),
                Err(rocaddr3::Error::NoLayout)
            );
        }
    }
}

#[test]
fn linear_mapper_preserves_row_and_slice_pitches() {
    for (width, slices) in [(17, 1), (17, 2), (64, 1), (64, 2)] {
        for requested_pitch in [0, 64, 96, 128] {
            let request = SurfaceRequest {
                width,
                slices,
                mip_levels: 1,
                ..request(0x98, ResourceType::Texture2D)
            };
            let layout = compute_linear(request, requested_pitch).unwrap();
            let mut mapper = SurfaceMapper::new(request, &layout).unwrap();
            let pitch = if requested_pitch == 0 {
                layout.mips[0].pitch
            } else {
                requested_pitch
            };
            for slice in 0..slices {
                for row in [0, 1, request.height - 1] {
                    let mut offsets = [0; 2];
                    mapper.map_row(0, 3, row, slice, &mut offsets).unwrap();
                    let expected = u64::from(slice) * layout.slice_size
                        + (u64::from(row) * u64::from(pitch) + 3) * 4;
                    assert_eq!(
                        offsets,
                        [expected, expected + 4],
                        "width {width}, pitch {requested_pitch}, row {row}, slice {slice}"
                    );
                }
            }
            for (x, y, slice) in [(width - 1, 0, 0), (0, request.height, 0), (0, 0, slices)] {
                assert_eq!(
                    mapper.map_row(0, x, y, slice, &mut [0; 2]),
                    Err(rocaddr3::Error::NoLayout)
                );
            }
        }
    }
}

fn check_gfx12_large_surface_sizes(linear: bool) {
    // These aligned extents need no padding. The two-level chain adds a
    // quarter-sized mip to the base image, taking a 4-GiB slice to 5 GiB.
    for (bits_per_pixel, mip_levels, expected_slice) in [
        (64, 1, 2_u64 << 30),
        (128, 1, 4_u64 << 30),
        (128, 2, 5_u64 << 30),
    ] {
        for slices in [1, 2] {
            let request = SurfaceRequest {
                bits_per_pixel,
                width: 16384,
                height: 16384,
                slices,
                mip_levels,
                ..request(0x98, ResourceType::Texture2D)
            };
            let layout = if linear {
                compute_linear(request, 0)
            } else {
                compute(request)
            }
            .unwrap_or_else(|error| panic!("{request:?}, linear {linear}: {error:?}"));

            assert_eq!(layout.slice_size, expected_slice, "{request:?}");
            assert_eq!(layout.size, expected_slice * u64::from(slices));
            assert_eq!(layout.pitch, request.width);
            assert_eq!(layout.height, request.height);
            assert_eq!(layout.mips.len(), mip_levels as usize);
        }
    }
}

#[test]
fn gfx12_large_linear_surface_sizes_do_not_wrap() {
    check_gfx12_large_surface_sizes(true);
}

#[test]
fn gfx12_large_tiled_surface_sizes_do_not_wrap() {
    check_gfx12_large_surface_sizes(false);
}
