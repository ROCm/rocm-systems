// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA's native trace-ID codec, separate from ordinary sampling creation.

use rocddi::device::gpu::{
    GpuDevice,
    profiling::{PcSampling, PcSamplingRequest},
};
use std::num::NonZeroU32;

/// Transfers a stopped native trace only through its qualified provider codec.
///
/// # Safety
/// The HSA caller transfers sole control of the stopped trace on this GPU,
/// with its matching configuration and retained trap/sample backing.
#[allow(unsafe_code)]
pub(crate) unsafe fn adopt(
    gpu: &GpuDevice<'_>,
    request: PcSamplingRequest,
    id: NonZeroU32,
) -> Result<PcSampling, rocddi::Error> {
    #[cfg(target_os = "linux")]
    {
        use rocddi::device::gpu::profiling::linux::{self, KfdPcSamplingTraceId};
        // SAFETY: The HSA create-from-ID caller supplies this ownership and
        // configuration contract; errors leave the native trace with it.
        unsafe { linux::adopt(gpu, request, KfdPcSamplingTraceId::new(id)) }
    }
    #[cfg(windows)]
    {
        let _ = (gpu, request, id);
        Err(rocddi::Error::Operation {
            kind: rocddi::ErrorKind::Unsupported,
            detail: "HSA trace-ID adoption is not qualified for this provider",
        })
    }
}
