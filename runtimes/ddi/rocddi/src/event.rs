// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Device notifications shared with API frontends.
//!
//! System events describe observed device failures. Each subscription has its
//! own delivery cursor; the driver observes each native notification once and
//! retains unread records plus the latest details of each exception kind for
//! subscribers that register after an operation observed it.
//! Frontends own callback dispatch, error policy, and any worker threads.

use crate::device::Device;
use crate::driver;
use crate::gpu::GpuDevice;
use crate::host_storage::Owned;
use crate::memory::Allocation;
use crate::queue::QueueErrorEvent;
use crate::{Error, ErrorKind};

/// One process-level GPU virtual-memory fault reported by a driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent KFD memory-fault cause bits mirror the native event payload"
)]
pub struct GpuMemoryFault {
    /// Activated endpoint that reported the fault, when known to this session.
    pub endpoint_id: Option<[u8; 16]>,
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

/// One GPU hardware exception and its reset information.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuHardwareException {
    /// Activated endpoint that reported the exception, when known.
    pub endpoint_id: Option<[u8; 16]>,
    /// Driver reset type. Zero denotes a whole-GPU reset on Linux KFD.
    pub reset_type: u32,
    /// Whether the driver reported loss of device memory.
    pub memory_lost: bool,
    /// Driver reset cause. Zero denotes a GPU hang and one denotes ECC on KFD.
    pub reset_cause: u32,
}

/// A device event retained by the DDI for independent frontend observers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceEvent {
    /// A process-level GPU virtual-memory fault.
    GpuMemoryFault(GpuMemoryFault),
    /// A GPU hardware exception or reset.
    GpuHardwareException(GpuHardwareException),
}

/// One observer of a driver's system event stream.
///
/// A subscription retains the native event owner until it is dropped. Polling
/// never invokes frontend code, and an event is returned at most once to this
/// subscription. Retained events are replayed to late subscribers. If a slow
/// subscriber fills the finite event buffer, native polling waits until that
/// subscriber reads or drops its backlog.
pub struct DeviceEventSubscription {
    inner: driver::EventSubscription,
}

impl DeviceEventSubscription {
    /// Returns the next observed event, or `None` when no new event is ready.
    ///
    /// # Errors
    /// Reports native polling or driver-lifetime failures. `Busy` means a
    /// subscriber has not drained the finite event buffer; retry later. A
    /// later poll can still retrieve an event recorded before a polling failure.
    pub fn poll(&mut self) -> Result<Option<DeviceEvent>, Error> {
        self.inner.poll()
    }
}

/// Subscribes to events on the activated device's driver connection.
///
/// Multiple devices on one connection share native observation, while every
/// returned subscription has an independent cursor. A driver without event
/// support returns `Unsupported`.
///
/// # Errors
/// Reports unsupported drivers and unavailable native event state.
pub fn subscribe(device: &Device) -> Result<DeviceEventSubscription, Error> {
    Ok(DeviceEventSubscription {
        inner: device.activation.subscribe_events()?,
    })
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
    pub(crate) inner: Owned<driver::KfdSignalEvent>,
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
        driver::KfdDriver::destroy_kfd_signal_event(&mut self.inner)
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
        driver::KfdDriver::retain_kfd_signal_event_page(allocation.inner.driver_state_mut())?;
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
            Some(allocation.inner.driver_state())
        }
        None => None,
    };
    let mut page_offered = false;
    let result =
        device
            .driver
            .create_kfd_signal_event(device.state, native_page, &mut page_offered);
    if let Some(page) = event_page {
        // KFD can install a page even when CREATE_EVENT reports an error.
        // Validation and metadata allocation before dispatch do not offer it.
        page.attempted |= page_offered;
    }
    let inner = result?;
    let info = inner.info();
    Ok(SignalEvent { inner, info })
}
