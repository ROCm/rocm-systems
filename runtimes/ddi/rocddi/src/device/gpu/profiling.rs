// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Device timing and performance-monitoring contracts.
//!
//! These types describe implementation-neutral profiling operations. Native
//! registration state remains privately owned by the selected platform driver.

use crate::Error;
use crate::device::gpu::GpuDevice;
use crate::driver::{self, Driver, PcSamplingResource};
use crate::memory::storage::{Owned, Shared};
use std::num::NonZeroU32;

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

/// KFD PC sampling mechanism available on an activated GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcSamplingMethod {
    /// Wave trap handler records program counters.
    HostTrap,
    /// Hardware periodically snapshots program counters.
    Stochastic,
}

/// Unit used for a PC sampling interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcSamplingIntervalUnit {
    /// Wall-clock microseconds.
    Microseconds,
    /// GPU clock cycles.
    ClockCycles,
    /// Issued instructions.
    Instructions,
}

/// Native PC sampling capability reported for one method and interval unit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PcSamplingConfiguration {
    /// Sampling mechanism.
    pub method: PcSamplingMethod,
    /// Unit in which the interval bounds are expressed.
    pub unit: PcSamplingIntervalUnit,
    /// Smallest interval accepted by the driver.
    pub minimum_interval: u64,
    /// Largest interval accepted by the driver.
    pub maximum_interval: u64,
    /// Native restrictions, including the power-of-two interval flag.
    pub flags: u64,
}

/// Selects immutable driver support or configurations available right now.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcSamplingConfigurationSet {
    /// All configurations supported by the driver and device.
    Supported,
    /// Configurations allowed alongside existing native registrations.
    Available,
}

/// A requested native PC sampling method and interval.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PcSamplingRequest {
    /// Sampling mechanism.
    pub method: PcSamplingMethod,
    /// Unit in which `interval` is expressed.
    pub unit: PcSamplingIntervalUnit,
    /// Nonzero sampling interval.
    pub interval: u64,
}

/// Retains the exact driver instance alongside a native PC sampling owner.
pub(crate) struct DriverPcSampling<D: Driver, P: PcSamplingResource> {
    /// Native registration dropped before its creating driver.
    inner: Owned<P>,
    /// Exact driver instance retained through cleanup.
    _driver: Shared<D>,
}

impl<D: Driver, P: PcSamplingResource> DriverPcSampling<D, P> {
    pub(crate) fn new(driver: Shared<D>, inner: Owned<P>) -> Self {
        Self {
            inner,
            _driver: driver,
        }
    }

    pub(crate) fn start(&mut self) -> Result<(), Error> {
        self.inner.start()
    }

    pub(crate) fn stop(&mut self) -> Result<(), Error> {
        self.inner.stop()
    }

    pub(crate) fn destroy(&mut self) -> Result<(), Error> {
        self.inner.destroy()
    }

    pub(crate) fn relinquish_adopted(&mut self) -> Result<(), Error> {
        self.inner.relinquish_adopted()
    }
}

/// Owned PC sampling registration with explicit, retryable native teardown.
pub struct PcSampling {
    /// Concrete registration and retryable native cleanup state.
    pub(crate) inner: driver::PcSamplingState,
}

impl PcSampling {
    /// Starts sampling after the caller has installed the required trap and
    /// sample buffer resources for this method.
    ///
    /// # Errors
    /// Reports native setup failure or device loss. An ambiguous result keeps
    /// this owner and its native context alive for explicit recovery.
    pub fn start(&mut self) -> Result<(), Error> {
        self.inner.start()
    }

    /// Stops a started sampling registration.
    ///
    /// # Errors
    /// Reports a native stop failure while retaining the registration.
    pub fn stop(&mut self) -> Result<(), Error> {
        self.inner.stop()
    }

    /// Unregisters the native session. A failed operation retains its owner
    /// for a later retry or process teardown.
    ///
    /// # Errors
    /// Reports native stop or destroy failure.
    pub fn destroy(&mut self) -> Result<(), Error> {
        self.inner.destroy()
    }

    /// Gives an unstarted adopted trace back to its original owner when an
    /// adapter cannot complete setup after adoption.
    #[doc(hidden)]
    pub fn relinquish_adopted(&mut self) -> Result<(), Error> {
        self.inner.relinquish_adopted()
    }
}

impl GpuDevice<'_> {
    /// Returns one driver-correlated device, host, and system clock sample.
    #[doc(hidden)]
    pub fn clock_counters(&self) -> Result<ClockCounters, Error> {
        self.driver.clock_counters()
    }

    /// Queries PC sampling configurations supported by the GPU driver or
    /// currently available alongside native registrations. An empty result
    /// means the driver offered no usable method in the selected set.
    ///
    /// # Errors
    /// Returns `Unsupported` on an older KFD, or the native query failure.
    #[doc(hidden)]
    pub fn pc_sampling_configurations(
        &self,
        set: PcSamplingConfigurationSet,
    ) -> Result<Vec<PcSamplingConfiguration>, Error> {
        self.driver.pc_sampling_configurations(set)
    }

    /// Registers a native PC sampler. The caller must install and retain the
    /// matching trap handler and GPU-visible sample buffers before `start`.
    ///
    /// # Errors
    /// Rejects an invalid interval or unavailable method and reports native
    /// registration failure.
    #[doc(hidden)]
    pub fn create_pc_sampling(&self, request: PcSamplingRequest) -> Result<PcSampling, Error> {
        Ok(PcSampling {
            inner: self.driver.create_pc_sampling(request, None)?,
        })
    }

    /// Takes ownership of an existing stopped native registration on this
    /// device. The creator retains ownership if this call returns an error.
    ///
    /// # Errors
    /// Rejects an unavailable method or interval, or a lost device context.
    #[doc(hidden)]
    pub fn adopt_pc_sampling(
        &self,
        request: PcSamplingRequest,
        trace_id: NonZeroU32,
    ) -> Result<PcSampling, Error> {
        Ok(PcSampling {
            inner: self.driver.create_pc_sampling(request, Some(trace_id))?,
        })
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
                .set_trap_handler(handler_address, memory_address)
        }
    }

    /// Acquires this device's stream performance monitor.
    #[doc(hidden)]
    pub fn spm_acquire(&self) -> Result<(), Error> {
        self.driver.spm_acquire()
    }

    /// Releases this device's stream performance monitor.
    #[doc(hidden)]
    pub fn spm_release(&self) -> Result<(), Error> {
        self.driver.spm_release()
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
            self.driver
                .spm_set_destination(size, timeout, bytes_copied, destination, data_loss)
        }
    }
}
