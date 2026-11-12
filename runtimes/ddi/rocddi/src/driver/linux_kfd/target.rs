// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! KFD discovery-code decoding and Linux transport qualification.

use crate::topology::GpuTargetProfile;

pub(super) trait KfdTargetProfile {
    fn from_kfd_target(target: u32) -> Self;
    fn supports_aql(self) -> bool;
    fn supports_kernel_submission(self) -> bool;
}

impl KfdTargetProfile for GpuTargetProfile {
    /// Resolves a KFD target code once when the endpoint is discovered.
    fn from_kfd_target(target: u32) -> Self {
        match target {
            90008 | 90010 | 90402 | 90500 => Self::Gfx9,
            100_100..=109_999 => Self::Gfx10,
            110_000..=119_999 => Self::Gfx11,
            120_000 => Self::Gfx1200,
            120_001 => Self::Gfx1201,
            120_500 => Self::Gfx1250,
            _ => Self::Other,
        }
    }

    /// Whether the current backend qualifies an AQL queue transport.
    fn supports_aql(self) -> bool {
        match self {
            Self::Other => false,
            Self::Gfx9
            | Self::Gfx10
            | Self::Gfx11
            | Self::Gfx1200
            | Self::Gfx1201
            | Self::Gfx1250 => true,
        }
    }

    /// Whether the qualified DRM kernel command transport is available.
    fn supports_kernel_submission(self) -> bool {
        match self {
            Self::Gfx1200 | Self::Gfx1201 | Self::Gfx1250 => true,
            Self::Other | Self::Gfx9 | Self::Gfx10 | Self::Gfx11 => false,
        }
    }
}

#[cfg(test)]
mod target_profile_tests {
    use super::KfdTargetProfile;
    use crate::topology::{GpuScratchEncoding, GpuSdmaDialect, GpuTargetProfile as Profile};

    #[test]
    fn qualified_targets_keep_packet_and_scratch_dialects_together() {
        for target in [90008, 90010, 90402, 90500] {
            let profile = Profile::from_kfd_target(target);
            assert_eq!(profile, Profile::Gfx9);
            assert_eq!(profile.sdma_dialect(), Some(GpuSdmaDialect::Gfx9));
            assert_eq!(profile.scratch_encoding(64), Some(GpuScratchEncoding::Gfx9));
            assert_eq!(profile.scratch_encoding(32), None);
        }
        for (target, profile, dialect) in [
            (120_000, Profile::Gfx1200, GpuSdmaDialect::Gfx1200),
            (120_001, Profile::Gfx1201, GpuSdmaDialect::Gfx1200),
            (120_500, Profile::Gfx1250, GpuSdmaDialect::Gfx1250),
        ] {
            assert_eq!(Profile::from_kfd_target(target), profile);
            assert_eq!(profile.sdma_dialect(), Some(dialect));
            assert_eq!(
                profile.scratch_encoding(32),
                Some(GpuScratchEncoding::Gfx12)
            );
            assert_eq!(profile.scratch_encoding(64), None);
            assert!(profile.supports_kernel_submission());
        }
        for target in [90009, 120_002, 120_501] {
            let profile = Profile::from_kfd_target(target);
            assert_eq!(profile, Profile::Other);
            assert!(!profile.supports_aql());
            assert_eq!(profile.sdma_dialect(), None);
            assert_eq!(profile.scratch_encoding(32), None);
        }
    }
}
