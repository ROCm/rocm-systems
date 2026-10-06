// SPDX-License-Identifier: MIT

//! Linux KFD GPU notification events shared with API frontends.
//!
//! This implementation module is re-exported only through
//! `gpu::event::linux`. Its event identifiers and mailbox slots are Linux KFD
//! transport details, not requirements of the platform-neutral device model.

use crate::driver;
use crate::driver::linux_interop::LinuxGpuEventDriver;
use crate::gpu::GpuDevice;
use crate::host_storage::Owned;
use crate::memory::Allocation;
use crate::queue::QueueErrorEvent;
use crate::{Error, ErrorKind};

/// One process-level GPU virtual-memory fault reported by the native driver.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent KFD memory-fault cause bits mirror the native event payload"
)]
pub struct GpuMemoryFault {
    /// KFD's per-boot identifier for the faulting GPU.
    pub kfd_gpu_id: u32,
    /// Virtual address reported by KFD.
    pub virtual_address: u64,
    /// The address was not present or required supervisor privilege.
    pub page_not_present: bool,
    /// The access attempted to write a read-only page.
    pub read_only: bool,
    /// The access attempted to execute a non-executable page.
    pub no_execute: bool,
    /// The reported virtual address may be imprecise.
    pub imprecise: bool,
    /// Native memory-exception error classification.
    pub error_type: u32,
}

/// KFD identity and mailbox slot assigned to one interrupt-capable signal.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignalEventInfo {
    /// Process-local KFD event identifier written by the GPU on notification.
    pub kfd_event_id: u32,
    /// Eight-byte slot within the process signal event page.
    pub event_page_slot_index: u32,
}

/// Owns one process-local KFD signal event.
#[doc(hidden)]
pub struct SignalEvent {
    pub(crate) inner: Owned<driver::NativeSignalEvent>,
    pub(crate) info: SignalEventInfo,
}

impl SignalEvent {
    /// Returns the immutable event identity and mailbox slot.
    #[must_use]
    pub fn info(&self) -> SignalEventInfo {
        self.info
    }

    /// Creates the opaque notification descriptor used by a KFD-backed AQL
    /// queue to report an exception through this event.
    #[must_use]
    pub fn queue_error_event(&self, payload_address: u64) -> QueueErrorEvent {
        QueueErrorEvent {
            payload_address,
            native_event_token: u64::from(self.info.kfd_event_id),
        }
    }

    /// Releases the native signal event while preserving retry state on failure.
    ///
    /// # Errors
    /// Reports the native destruction failure and retains the event identity
    /// when retrying is safe.
    pub fn destroy(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::destroy_kfd_signal_event(&mut self.inner)
    }
}

/// Owns the KFD signal-event page across an attempted event creation.
///
/// KFD can install this page before its event ioctl fails and keeps the page
/// mapped until process teardown. Dropping an attempted page conservatively
/// retains the complete allocation if explicit transfer has not succeeded.
#[doc(hidden)]
pub struct SignalEventPage {
    allocation: Option<Allocation>,
    attempted: bool,
}

impl SignalEventPage {
    /// Wraps a page allocation before it is offered to KFD.
    #[must_use]
    pub fn new(allocation: Allocation) -> Self {
        Self {
            allocation: Some(allocation),
            attempted: false,
        }
    }

    /// Returns the existing host mapping of the page, if any.
    #[must_use]
    pub fn host_address(&self) -> Option<usize> {
        self.allocation.as_ref()?.info().host_address
    }

    /// Whether this page was offered to KFD, even if event creation failed.
    #[must_use]
    pub fn was_offered(&self) -> bool {
        self.attempted
    }

    /// Transfers a page used in an event creation attempt to KFD process
    /// lifetime. On error, this owner still retains the complete allocation.
    ///
    /// # Errors
    /// Rejects an allocation that is not a live, host-visible KFD event page.
    pub fn retain_for_process(&mut self) -> Result<(), Error> {
        if !self.attempted {
            return Ok(());
        }
        let allocation = self.allocation.as_mut().ok_or(Error::Operation {
            kind: ErrorKind::Internal,
            detail: "attempted signal event page lost its allocation",
        })?;
        driver::PlatformDriver::retain_kfd_signal_event_page(allocation.inner.native_mut())?;
        self.attempted = false;
        self.allocation = None;
        Ok(())
    }
}

impl Drop for SignalEventPage {
    fn drop(&mut self) {
        if self.attempted {
            if let Some(allocation) = self.allocation.take() {
                std::mem::forget(allocation);
            }
        }
    }
}

/// Claims and polls this session's process-level KFD GPU memory-fault event.
/// Once claimed, ordinary device checks leave memory-fault delivery to the
/// caller while continuing to observe terminal hardware loss.
///
/// # Errors
/// Returns a native KFD error, or `DeviceLost` when terminal loss was already
/// observed.
pub fn poll_memory_fault(device: GpuDevice<'_>) -> Result<Option<GpuMemoryFault>, Error> {
    device
        .device
        .driver
        .poll_kfd_memory_fault(&device.device.state)
}

/// Creates an auto-reset KFD signal event. The first event in a process supplies
/// an owned signal-event page; later events reuse that process page.
///
/// # Errors
/// Rejects an invalid page allocation and reports native event creation
/// failures without publishing a partial owner.
pub fn create_signal_event(
    device: GpuDevice<'_>,
    event_page: Option<&mut SignalEventPage>,
) -> Result<SignalEvent, Error> {
    let native_page = match event_page.as_deref() {
        Some(page) => {
            let allocation = page.allocation.as_ref().ok_or(Error::Operation {
                kind: ErrorKind::Internal,
                detail: "signal event page lost its allocation",
            })?;
            Some(allocation.inner.native())
        }
        None => None,
    };
    let mut page_offered = false;
    let result = device.device.driver.create_kfd_signal_event(
        &device.device.state,
        native_page,
        &mut page_offered,
    );
    if let Some(page) = event_page {
        // KFD can install a page even when CREATE_EVENT reports an error.
        // Validation and metadata allocation before dispatch do not offer it.
        page.attempted |= page_offered;
    }
    let inner = result?;
    let info = inner.info();
    Ok(SignalEvent { inner, info })
}
