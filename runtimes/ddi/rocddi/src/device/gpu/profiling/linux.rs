// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD sampling-registration adoption, with its native context contract.

use super::{DriverPcSampling, PcSampling, PcSamplingRequest};
use crate::Error;
use crate::device::gpu::GpuDevice;
use crate::driver::{GpuDriverRef, PcSamplingState};
use std::num::NonZeroU32;

/// A stopped KFD trace registration in the primary process context.
/// Equal numeric IDs from another driver or context are not interchangeable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KfdPcSamplingTraceId(NonZeroU32);

impl KfdPcSamplingTraceId {
    /// Wraps the native numeric namespace; this does not establish ownership.
    #[must_use]
    pub const fn new(id: NonZeroU32) -> Self {
        Self(id)
    }
}

/// Adopts a stopped registration associated with this exact GPU and primary
/// KFD context. Ownership transfers only on success. The caller retains the
/// compatible trap code and sample buffers through use and checked retirement.
///
/// # Errors
/// Rejects another provider, secondary contexts, unsupported requests or loss.
///
/// # Safety
/// The caller solely owns a live, stopped trace on this exact GPU in the
/// primary KFD context, and `request` describes its actual configuration.
/// No other owner may control, adopt, destroy or reuse its ID. Trap code and
/// sample buffers remain live through use and any uncertain native cleanup.
#[allow(unsafe_code)]
pub unsafe fn adopt(
    gpu: &GpuDevice<'_>,
    request: PcSamplingRequest,
    trace_id: KfdPcSamplingTraceId,
) -> Result<PcSampling, Error> {
    match gpu.driver {
        GpuDriverRef::LinuxKfd { driver, state } => {
            let native = driver.adopt_pc_sampling(state, request, trace_id.0)?;
            Ok(PcSampling {
                inner: PcSamplingState::LinuxKfd(DriverPcSampling::new(driver.clone(), native)),
            })
        }
        #[cfg(test)]
        GpuDriverRef::Test { .. } => Err(Error::Operation {
            kind: crate::ErrorKind::Unsupported,
            detail: "KFD trace adoption requires a Linux KFD GPU",
        }),
    }
}
