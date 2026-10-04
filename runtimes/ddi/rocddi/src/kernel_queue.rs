//! Bounded, kernel-mediated GPU command submission.
//!
//! Command bytes remain in caller-owned device memory. This owner carries only
//! the native submission context and its bounded progress state; packet
//! encoding, public handles, and submission policy belong to the frontend.

use crate::driver::{self, KernelQueueDriver};
use crate::gpu::GpuDevice;
use crate::host_storage::{Owned, Shared};
use crate::{Error, ErrorKind};

/// Native command representation selected for a kernel-mediated queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelQueueFormat {
    /// AMD GPU PM4 command stream submitted to a compute engine.
    Pm4,
    /// AMD GPU SDMA command stream submitted to a copy engine.
    Sdma,
}

/// One already-materialized, executable device-memory command range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelCommand {
    /// Stable address in the queue device's address domain.
    pub device_address: u64,
    /// Nonzero dword-aligned command length in bytes.
    pub byte_length: u64,
}

/// Cached retirement and observed native failure of one queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct KernelQueueStatus {
    /// Greatest accepted submission whose native command storage is reusable.
    pub retired_submission: u64,
    /// Sticky native failure. Failure alone does not prove retirement.
    pub terminal: Option<ErrorKind>,
}

/// Result of one explicit bounded native wait.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelQueueWait {
    /// The requested submission's native command storage is reusable.
    Retired,
    /// The deadline expired without proving retirement.
    TimedOut,
}

/// Owns one native submission context and its retryable teardown state.
pub struct KernelQueue {
    inner: Owned<driver::NativeKernelQueue>,
    _driver: Shared<driver::PlatformDriver>,
    format: KernelQueueFormat,
}

impl KernelQueue {
    /// Returns the command representation selected at creation.
    #[must_use]
    pub const fn format(&self) -> KernelQueueFormat {
        self.format
    }

    /// Submits one opaque command range without a library allocation or lock.
    ///
    /// A native outcome that cannot distinguish rejection from acceptance is
    /// conservatively published as an accepted, failed submission. The caller
    /// retains command storage until status reports retirement.
    ///
    /// # Errors
    /// Reports a proved rejection, unavailable slot, or lost device.
    pub fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        driver::PlatformDriver::submit_kernel_queue(&self.inner, command)
    }

    /// Reads cached retirement and terminal state without entering the driver.
    #[must_use]
    pub fn status(&self) -> KernelQueueStatus {
        driver::PlatformDriver::kernel_queue_status(&self.inner)
    }

    /// Waits through the native context under one caller-supplied deadline.
    ///
    /// # Errors
    /// Reports an invalid submission, native wait failure, or device loss.
    pub fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        driver::PlatformDriver::wait_kernel_queue(
            &self.inner,
            submission,
            timeout_nanoseconds,
            poll_duration_nanoseconds,
        )
    }

    /// Releases the native context after all submissions retire.
    ///
    /// A failed release retains this owner for a later destruction attempt.
    ///
    /// # Errors
    /// Returns `Busy` while command storage may still be in use, or a native
    /// teardown error while retaining every unreleased dependency.
    pub fn destroy(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::destroy_kernel_queue(&mut self.inner)
    }
}

impl GpuDevice<'_> {
    /// Creates a kernel-mediated queue with all bounded resources ready.
    ///
    /// # Errors
    /// Rejects an unqualified format, native context failure, or exhaustion.
    pub fn create_kernel_queue(&self, format: KernelQueueFormat) -> Result<KernelQueue, Error> {
        let inner = self
            .device
            .driver
            .create_kernel_queue(&self.device.state, format)?;
        Ok(KernelQueue {
            inner,
            _driver: self.device.driver.clone(),
            format,
        })
    }
}
