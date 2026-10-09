// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Bounded, kernel-mediated GPU command submission.
//!
//! Command bytes remain in caller-owned device memory. This owner carries only
//! the native submission context and its bounded progress state; packet
//! encoding, public handles, and submission policy belong to the frontend.

use crate::driver::{self, Driver, GpuDriver, KernelQueueResource};
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
    /// AMD GPU SDMA command stream submitted to one DRM DMA ring.
    SdmaOnRing(u32),
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

/// Retains a concrete driver and its kernel queue until native retirement.
struct DriverKernelQueue<D: Driver, Q: KernelQueueResource> {
    inner: Owned<Q>,
    _driver: Shared<D>,
}

impl<D: Driver, Q: KernelQueueResource> DriverKernelQueue<D, Q> {
    fn new(driver: Shared<D>, inner: Owned<Q>) -> Self {
        Self {
            inner,
            _driver: driver,
        }
    }

    /// # Safety
    /// Command storage remains reachable until native retirement is proved.
    #[allow(unsafe_code)]
    unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        // SAFETY: The caller retains the command through retirement.
        unsafe { self.inner.submit(command) }
    }

    fn status(&self) -> KernelQueueStatus {
        self.inner.status()
    }

    fn refresh_status(&self) -> Result<KernelQueueStatus, Error> {
        self.inner.refresh_status()
    }

    fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        self.inner
            .wait(submission, timeout_nanoseconds, poll_duration_nanoseconds)
    }

    fn destroy(&mut self) -> Result<(), Error> {
        self.inner.destroy()
    }
}

/// Selects the concrete kernel queue owner behind the public resource.
enum KernelQueueState {
    LinuxKfd(DriverKernelQueue<driver::KfdDriver, driver::KfdKernelQueue>),
}

impl KernelQueueState {
    /// # Safety
    /// Command storage remains reachable until native retirement is proved.
    #[allow(unsafe_code)]
    unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        match self {
            // SAFETY: The caller preserves the command backing.
            Self::LinuxKfd(queue) => unsafe { queue.submit(command) },
        }
    }

    fn status(&self) -> KernelQueueStatus {
        match self {
            Self::LinuxKfd(queue) => queue.status(),
        }
    }

    fn refresh_status(&self) -> Result<KernelQueueStatus, Error> {
        match self {
            Self::LinuxKfd(queue) => queue.refresh_status(),
        }
    }

    fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        match self {
            Self::LinuxKfd(queue) => {
                queue.wait(submission, timeout_nanoseconds, poll_duration_nanoseconds)
            }
        }
    }

    fn destroy(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(queue) => queue.destroy(),
        }
    }
}

/// Owns one native submission context and its retryable teardown state.
pub struct KernelQueue {
    inner: KernelQueueState,
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
    /// conservatively published as an accepted, failed submission with an
    /// identity. An error proves rejection. The caller retains command storage
    /// until status reports retirement.
    ///
    /// # Errors
    /// Reports a proved rejection, unavailable slot, or lost device.
    ///
    /// # Safety
    /// The command range must remain device accessible, executable, and
    /// unchanged until [`Self::status`] or [`Self::wait`] proves its submission
    /// retired. It must contain valid packets for this queue's format, and the
    /// caller must synchronize writes before submission. An ambiguous native
    /// outcome is returned as an accepted submission, whose storage must be
    /// retained until retirement or conclusive teardown.
    #[allow(unsafe_code)]
    pub unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        // SAFETY: The public caller retains and synchronizes the command range
        // until the native retirement frontier advances.
        unsafe { self.inner.submit(command) }
    }

    /// Reads cached retirement and terminal state without entering the driver.
    #[must_use]
    pub fn status(&self) -> KernelQueueStatus {
        self.inner.status()
    }

    /// Checks native completion once without waiting and returns the checked
    /// retirement frontier and sticky terminal state.
    ///
    /// # Errors
    /// Reports a native observation failure. Earlier checked retirement remains
    /// available through [`Self::status`].
    pub fn refresh_status(&self) -> Result<KernelQueueStatus, Error> {
        self.inner.refresh_status()
    }

    /// Waits through the queue's private submission context under one
    /// caller-supplied deadline.
    ///
    /// # Errors
    /// Reports an invalid submission, native wait failure, or device loss.
    pub fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        self.inner
            .wait(submission, timeout_nanoseconds, poll_duration_nanoseconds)
    }

    /// Releases the queue's private submission context after all submissions
    /// retire.
    ///
    /// A failed release retains this owner for a later destruction attempt.
    ///
    /// # Errors
    /// Returns `Busy` while command storage may still be in use, or a native
    /// teardown error while retaining every unreleased dependency.
    pub fn destroy(&mut self) -> Result<(), Error> {
        self.inner.destroy()
    }
}

impl GpuDevice<'_> {
    /// Returns the DRM DMA ring bitmask available for kernel-mediated SDMA
    /// submissions on this activated device.
    ///
    /// # Errors
    /// Reports an unqualified target or a native ring-query failure.
    pub fn available_sdma_rings(&self) -> Result<u32, Error> {
        self.device.driver_state.available_sdma_rings()
    }

    /// Creates a kernel-mediated queue with all bounded resources ready.
    ///
    /// # Errors
    /// Rejects an unqualified format, failure to create a submission context,
    /// or resource exhaustion.
    pub fn create_kernel_queue(&self, format: KernelQueueFormat) -> Result<KernelQueue, Error> {
        match &self.device.driver_state {
            driver::DeviceDriverState::LinuxKfd { driver, state } => {
                let inner = driver.create_kernel_queue(state, format)?;
                Ok(KernelQueue {
                    inner: KernelQueueState::LinuxKfd(DriverKernelQueue::new(
                        driver.clone(),
                        inner,
                    )),
                    format,
                })
            }
            #[cfg(test)]
            driver::DeviceDriverState::Test { .. } => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "activated driver has no GPU kernel queue capability",
            }),
        }
    }
}
