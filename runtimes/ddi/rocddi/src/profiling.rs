// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Device timing and performance-monitoring contracts.
//!
//! These types describe implementation-neutral profiling operations. Native
//! registration state remains privately owned by the selected platform driver.

use crate::Error;
use crate::driver::GpuProfilingDriver;
use crate::gpu::GpuDevice;

/// Correlated device and system clocks returned by the native driver.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ClockCounters {
    /// GPU timestamp counter.
    pub gpu: u64,
    /// Host timestamp counter sampled by the native driver.
    pub host: u64,
    /// System timestamp counter sampled with the GPU counter.
    pub system: u64,
    /// System timestamp frequency in hertz.
    pub system_frequency: u64,
    /// GPU timestamp frequency in hertz reported by the native device.
    pub gpu_frequency: u64,
}

impl GpuDevice<'_> {
    /// Returns one driver-correlated device, host, and system clock sample.
    #[doc(hidden)]
    pub fn clock_counters(&self) -> Result<ClockCounters, Error> {
        self.driver.clock_counters(self.state)
    }

    /// Replaces this device's process-level second-stage trap handler.
    /// Supplying zero for both addresses removes the current handler.
    ///
    /// # Safety
    /// A nonzero handler address must refer to executable device memory with
    /// valid trap instructions, and a nonzero memory address must refer to the
    /// writable argument record required by those instructions. The caller
    /// must keep both allocations live and unchanged while the handler can
    /// run, including after an ambiguous native result, until a successful
    /// removal or conclusive device teardown.
    #[doc(hidden)]
    #[allow(unsafe_code)]
    pub unsafe fn set_trap_handler(
        &self,
        handler_address: u64,
        memory_address: u64,
    ) -> Result<(), Error> {
        // SAFETY: The public caller retains the handler code and argument
        // storage until successful removal or conclusive device teardown.
        unsafe {
            self.driver
                .set_trap_handler(self.state, handler_address, memory_address)
        }
    }

    /// Acquires this device's stream performance monitor.
    #[doc(hidden)]
    pub fn spm_acquire(&self) -> Result<(), Error> {
        self.driver.spm_acquire(self.state)
    }

    /// Releases this device's stream performance monitor.
    #[doc(hidden)]
    pub fn spm_release(&self) -> Result<(), Error> {
        self.driver.spm_release(self.state)
    }

    /// Replaces the stream performance monitor destination buffer.
    /// Passing `None` stops copying to the previous destination.
    ///
    /// # Safety
    /// When `destination` is `Some`, it must address at least `size` writable
    /// bytes that KFD can access. The caller must retain that backing and
    /// exclude conflicting CPU or device access until a later replacement or
    /// unset has completed successfully, or until the SPM and device are
    /// conclusively torn down. A failed call may leave either the previous or
    /// the new destination reachable by KFD, so both must be retained until
    /// that uncertainty is resolved. The address must not be derived from a
    /// short lived borrow because KFD writes after this call returns.
    #[doc(hidden)]
    #[allow(unsafe_code)]
    pub unsafe fn spm_set_destination(
        &self,
        size: u32,
        timeout: &mut u32,
        bytes_copied: &mut u32,
        destination: Option<usize>,
        data_loss: &mut bool,
    ) -> Result<(), Error> {
        // SAFETY: The public caller retains both potentially reachable
        // destinations until a successful unset or conclusive teardown.
        unsafe {
            self.driver.spm_set_destination(
                self.state,
                size,
                timeout,
                bytes_copied,
                destination,
                data_loss,
            )
        }
    }
}
