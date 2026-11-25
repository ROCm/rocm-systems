// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Bounded, kernel-mediated GPU command submission.
//!
//! Command bytes remain in caller-owned device memory. This owner carries only
//! the native submission context and its bounded progress state; packet
//! encoding, public handles, and submission policy belong to the frontend.
//! Retirement closes admission for new submissions; completion queries,
//! waits, and native destruction remain available for accepted work.

use crate::device::gpu::GpuDevice;
use crate::driver::{self, Driver, KernelQueueResource};
use crate::memory::storage::{Owned, Shared};
use crate::{Error, ErrorKind};

/// Native command representation selected for a kernel-mediated queue.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KernelQueueFormat {
    /// AMD GPU PM4 command stream submitted to a compute engine.
    Pm4,
    /// AMD GPU SDMA command stream submitted to a copy engine.
    Sdma,
    /// AMD GPU SDMA command stream submitted to one qualified logical engine.
    /// The index is a bit position returned by [`GpuDevice::available_sdma_engines`].
    /// Drivers map it to their own native submission target.
    SdmaOnEngine(u32),
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
pub(crate) struct DriverKernelQueue<D: Driver, Q: KernelQueueResource> {
    /// Native submission owner retained until retirement and destruction.
    inner: Owned<Q>,
    /// Keeps the exact creating driver alive through queue cleanup.
    driver: Shared<D>,
}

impl<D: Driver, Q: KernelQueueResource> DriverKernelQueue<D, Q> {
    pub(crate) fn new(driver: Shared<D>, inner: Owned<Q>) -> Self {
        Self { inner, driver }
    }

    /// # Safety
    /// Command storage remains reachable until native retirement is proved.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        let _operation = self.driver.admission().enter()?;
        // SAFETY: The caller retains the command through retirement.
        unsafe { self.inner.submit(command) }
    }

    pub(crate) fn status(&self) -> KernelQueueStatus {
        self.inner.status()
    }

    pub(crate) fn refresh_status(&self) -> Result<KernelQueueStatus, Error> {
        self.inner.refresh_status()
    }

    pub(crate) fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        self.inner
            .wait(submission, timeout_nanoseconds, poll_duration_nanoseconds)
    }

    pub(crate) fn destroy(&mut self) -> Result<(), Error> {
        self.inner.destroy()
    }
}

/// Owns one native submission context and its retryable teardown state.
pub struct KernelQueue {
    /// Concrete driver-mediated queue owner.
    pub(crate) inner: driver::KernelQueueState,
    /// Native command format selected at creation.
    pub(crate) format: KernelQueueFormat,
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
    /// Returns a bitmask of logical SDMA engine indices available for
    /// kernel-mediated submissions on this activated device. A set bit at
    /// position `n` permits [`KernelQueueFormat::SdmaOnEngine`]. The indices
    /// are scoped to this device and do not expose a driver ring number.
    ///
    /// # Errors
    /// Reports an unqualified target or a native engine-query failure.
    pub fn available_sdma_engines(&self) -> Result<u32, Error> {
        self.driver.available_sdma_engines()
    }

    /// Creates a kernel-mediated queue with all bounded resources ready.
    ///
    /// # Errors
    /// Rejects an unqualified format, failure to create a submission context,
    /// or resource exhaustion.
    pub fn create_kernel_queue(&self, format: KernelQueueFormat) -> Result<KernelQueue, Error> {
        Ok(KernelQueue {
            inner: self.driver.create_kernel_queue(format)?,
            format,
        })
    }
}

#[cfg(test)]
#[allow(unsafe_code, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::driver::test_driver::TestDriver;
    use crate::session::DriverContextLifetime;
    use crate::topology::EndpointKind;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    struct NativeQueue {
        submissions: Arc<AtomicU64>,
        destroyed: Arc<AtomicBool>,
    }

    impl KernelQueueResource for NativeQueue {
        unsafe fn submit(&self, _: KernelCommand) -> Result<u64, Error> {
            Ok(self.submissions.fetch_add(1, Ordering::Relaxed) + 1)
        }

        fn status(&self) -> KernelQueueStatus {
            KernelQueueStatus {
                retired_submission: self.submissions.load(Ordering::Relaxed),
                terminal: None,
            }
        }

        fn refresh_status(&self) -> Result<KernelQueueStatus, Error> {
            Ok(self.status())
        }

        fn wait(&self, _: u64, _: u64, _: u64) -> Result<KernelQueueWait, Error> {
            Ok(KernelQueueWait::Retired)
        }

        fn destroy(&mut self) -> Result<(), Error> {
            self.destroyed.store(true, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn retirement_blocks_submission_but_preserves_completion_and_destroy() {
        let driver = Shared::new(TestDriver::new(
            [1; 16],
            EndpointKind::Npu,
            DriverContextLifetime::Session,
        ))
        .unwrap();
        let submissions = Arc::new(AtomicU64::new(0));
        let destroyed = Arc::new(AtomicBool::new(false));
        let mut queue = DriverKernelQueue::new(
            driver.clone(),
            Owned::new(NativeQueue {
                submissions: submissions.clone(),
                destroyed: destroyed.clone(),
            })
            .unwrap(),
        );
        let command = KernelCommand {
            device_address: 0x1000,
            byte_length: 4,
        };
        // SAFETY: The fake records submissions without accessing command bytes.
        assert_eq!(unsafe { queue.submit(command) }.unwrap(), 1);
        driver.admission().begin_retirement().unwrap();
        // SAFETY: The fake has no native device-memory consumer.
        assert_eq!(
            unsafe { queue.submit(command) }.unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(submissions.load(Ordering::Relaxed), 1);
        assert_eq!(queue.refresh_status().unwrap().retired_submission, 1);
        assert_eq!(queue.wait(1, 1, 1).unwrap(), KernelQueueWait::Retired);
        queue.destroy().unwrap();
        assert!(destroyed.load(Ordering::Relaxed));
    }
}
