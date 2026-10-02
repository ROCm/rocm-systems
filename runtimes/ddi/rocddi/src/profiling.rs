//! Device timing, performance-monitoring, and PC-sampling contracts.
//!
//! These types describe implementation-neutral profiling operations. Native
//! registration state remains privately owned by the selected platform driver.

use crate::Error;
use crate::driver::{self, DeviceDriver, PcSamplingDriver};
use crate::gpu::GpuDevice;
use crate::host_storage::{Buffer, Owned};

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
}

/// PC-sampling mechanism selected by the native driver.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcSamplingMethod {
    /// Periodic shader trap sampling.
    HostTrapV1,
    /// Hardware stochastic snapshot sampling.
    StochasticV1,
}

/// Unit in which a native PC-sampling interval is expressed.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PcSamplingUnits {
    /// Microseconds.
    Microseconds,
    /// Graphics-clock cycles.
    ClockCycles,
    /// Issued instructions.
    Instructions,
}

/// One currently available native PC-sampling configuration.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PcSamplingConfiguration {
    /// Sampling mechanism.
    pub method: PcSamplingMethod,
    /// Interval units.
    pub units: PcSamplingUnits,
    /// Inclusive minimum interval.
    pub minimum_interval: u64,
    /// Inclusive maximum interval.
    pub maximum_interval: u64,
    /// Native restriction flags; bit zero requires a power-of-two interval.
    pub flags: u64,
}

/// Requires a power-of-two PC-sampling interval.
#[doc(hidden)]
pub const PC_SAMPLING_INTERVAL_POWER_OF_TWO: u64 = 1;

/// Owns one native PC-sampling registration.
#[doc(hidden)]
pub struct PcSampling {
    pub(crate) inner: Owned<driver::NativePcSampling>,
}

impl PcSampling {
    /// Returns the driver trace identifier while the registration is live.
    ///
    /// # Errors
    /// Returns `InvalidArgument` after successful destruction.
    pub fn trace_id(&self) -> Result<u32, Error> {
        driver::PlatformDriver::pc_sampling_trace_id(&self.inner)
    }

    /// Starts sampling, or succeeds without native work when already active.
    ///
    /// # Errors
    /// Returns a native failure or `DriverContract` after an ambiguous prior transition.
    pub fn start(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::start_pc_sampling(&mut self.inner)
    }

    /// Stops sampling, or succeeds without native work when already inactive.
    ///
    /// # Errors
    /// Returns a native failure while retaining the registration for retry or destruction.
    pub fn stop(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::stop_pc_sampling(&mut self.inner)
    }

    /// Stops and releases the native registration. Successful repeats are harmless.
    ///
    /// # Errors
    /// Returns a native failure while preserving a trace that may remain live.
    pub fn destroy(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::destroy_pc_sampling(&mut self.inner)
    }
}

impl GpuDevice<'_> {
    /// Returns one driver-correlated device, host, and system clock sample.
    #[doc(hidden)]
    pub fn clock_counters(&self) -> Result<ClockCounters, Error> {
        self.device.driver.clock_counters(&self.device.state)
    }

    /// Replaces this device's process-level second-stage trap handler.
    /// Supplying zero for both addresses removes the current handler.
    #[doc(hidden)]
    pub fn set_trap_handler(&self, handler_address: u64, memory_address: u64) -> Result<(), Error> {
        self.device
            .driver
            .set_trap_handler(&self.device.state, handler_address, memory_address)
    }

    /// Acquires this device's stream performance monitor.
    #[doc(hidden)]
    pub fn spm_acquire(&self) -> Result<(), Error> {
        self.device.driver.spm_acquire(&self.device.state)
    }

    /// Releases this device's stream performance monitor.
    #[doc(hidden)]
    pub fn spm_release(&self) -> Result<(), Error> {
        self.device.driver.spm_release(&self.device.state)
    }

    /// Replaces the stream performance monitor destination buffer.
    #[doc(hidden)]
    pub fn spm_set_destination(
        &self,
        size: u32,
        timeout: &mut u32,
        bytes_copied: &mut u32,
        destination: Option<usize>,
        data_loss: &mut bool,
    ) -> Result<(), Error> {
        self.device.driver.spm_set_destination(
            &self.device.state,
            size,
            timeout,
            bytes_copied,
            destination,
            data_loss,
        )
    }

    /// Returns the PC-sampling configurations currently available on this device.
    #[doc(hidden)]
    pub fn pc_sampling_configurations(&self) -> Result<Buffer<PcSamplingConfiguration>, Error> {
        self.device
            .driver
            .pc_sampling_configurations(&self.device.state)
    }

    /// Reserves a native PC-sampling trace in the stopped state.
    #[doc(hidden)]
    pub fn create_pc_sampling(
        &self,
        configuration: PcSamplingConfiguration,
        interval: u64,
    ) -> Result<PcSampling, Error> {
        Ok(PcSampling {
            inner: self.device.driver.create_pc_sampling(
                &self.device.state,
                configuration,
                interval,
            )?,
        })
    }

    /// Takes ownership of a stopped PC-sampling trace reserved by the caller.
    #[doc(hidden)]
    pub fn adopt_pc_sampling(&self, trace_id: u32) -> Result<PcSampling, Error> {
        Ok(PcSampling {
            inner: self
                .device
                .driver
                .adopt_pc_sampling(&self.device.state, trace_id)?,
        })
    }
}
