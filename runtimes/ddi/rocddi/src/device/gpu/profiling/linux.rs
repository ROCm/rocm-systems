// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD sampling-registration adoption, with its native context contract.

use super::{DriverPcSampling, PcSampling, PcSamplingRequest};
use crate::Error;
use crate::device::gpu::GpuDevice;
use crate::driver::{GpuDriverRef, PcSamplingState};
use std::num::NonZeroU32;

/// A normal trap retained by the primary process context across session shutdown.
/// Clones retain only native backing and its exact VM, never a session driver.
#[derive(Clone)]
pub struct ProcessTrap {
    inner: crate::memory::storage::Shared<crate::driver::KfdProcessTrap>,
}

impl ProcessTrap {
    /// Returns the saved address only for the same live native GPU VM.
    ///
    /// # Errors
    /// Rejects another VM/provider, inherited state or device loss.
    pub fn address(&self, gpu: &GpuDevice<'_>) -> Result<u64, Error> {
        match gpu.driver {
            GpuDriverRef::LinuxKfd { state, .. } => self.inner.address(state),
            #[cfg(test)]
            GpuDriverRef::Test { .. } => Err(Error::Operation {
                kind: crate::ErrorKind::Unsupported,
                detail: "process trap requires a Linux KFD GPU",
            }),
        }
    }
}

impl GpuDevice<'_> {
    /// Installs immutable normal trap code in the primary process context.
    /// Backing is shared across sessions and retained until process exit, even
    /// on a failed publication or bind. A later call reuses the same code and VA.
    /// Dropping the returned token neither unbinds nor frees the trap.
    ///
    /// # Errors
    /// Rejects a session context, mismatched debugger registration, changed code,
    /// uncertain publication, or a native allocation/publication/binding failure.
    ///
    /// # Safety
    /// `code` must implement the normal second-stage trap ABI for this GPU with
    /// no trap-memory argument. The caller must coordinate changes to the native
    /// trap binding with other clients of this shared process context.
    #[allow(unsafe_code)]
    pub unsafe fn install_process_trap(
        &self,
        registration_address: usize,
        code: &[u8],
        cancel: &std::sync::atomic::AtomicBool,
    ) -> Result<ProcessTrap, Error> {
        match self.driver {
            GpuDriverRef::LinuxKfd { driver, state } => {
                let inner = driver.install_process_trap(
                    state,
                    registration_address,
                    code,
                    |address, length| self.invalidate_code_cache(address, length, cancel),
                )?;
                Ok(ProcessTrap { inner })
            }
            #[cfg(test)]
            GpuDriverRef::Test { .. } => Err(Error::Operation {
                kind: crate::ErrorKind::Unsupported,
                detail: "process trap requires a Linux KFD GPU",
            }),
        }
    }

    /// Queries whether the active native runtime uses this debugger rendezvous.
    /// A preferred registration can lose to another client between passive
    /// session configuration and GPU activation; check before installing traps.
    ///
    /// # Errors
    /// Rejects another provider or an inherited/invalid native connection.
    pub fn debugger_registration_matches(
        &self,
        address: usize,
        setup_ttmp: bool,
    ) -> Result<bool, Error> {
        match self.driver {
            GpuDriverRef::LinuxKfd { state, .. } => {
                state.debugger_registration_matches(address, setup_ttmp)
            }
            #[cfg(test)]
            GpuDriverRef::Test { .. } => Err(Error::Operation {
                kind: crate::ErrorKind::Unsupported,
                detail: "runtime debugger registration requires a Linux KFD GPU",
            }),
        }
    }
}

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
