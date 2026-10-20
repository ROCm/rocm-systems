// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD signal-event interoperability.

use crate::device::Device;
use crate::device::gpu::GpuDevice;
use crate::device::gpu::queue::QueueErrorEvent;
use crate::driver;
use crate::memory::Allocation;
use crate::memory::storage::Owned;
use crate::{Error, ErrorKind};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Mutex, MutexGuard};

/// Process-wide KFD page identity shared by the AMDF and HSA adapters.
/// The page allocation itself stays with the adapter that offered it and is
/// transferred to process lifetime during that adapter's shutdown.
struct ProcessPage {
    host_address: Option<usize>,
    mappings: Vec<([u8; 16], u64)>,
    confirmed: bool,
}

struct ProcessPageSlot {
    process: u32,
    page: Mutex<ProcessPage>,
}

// Fork children must not lock an inherited mutex or use a parent KFD page.
// Published slots intentionally live until their process exits.
static PROCESS_PAGE_SLOT: AtomicPtr<ProcessPageSlot> = AtomicPtr::new(std::ptr::null_mut());

#[allow(unsafe_code)]
fn process_page_slot() -> &'static ProcessPageSlot {
    let process = std::process::id();
    loop {
        let current = PROCESS_PAGE_SLOT.load(Ordering::Acquire);
        if !current.is_null() {
            // SAFETY: A published slot is never freed or moved.
            if unsafe { (*current).process == process } {
                return unsafe { &*current };
            }
        }
        let candidate = Box::into_raw(Box::new(ProcessPageSlot {
            process,
            page: Mutex::new(ProcessPage {
                host_address: None,
                mappings: Vec::new(),
                confirmed: false,
            }),
        }));
        match PROCESS_PAGE_SLOT.compare_exchange(
            current,
            candidate,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return unsafe { &*candidate },
            Err(_) => unsafe { drop(Box::from_raw(candidate)) },
        }
    }
}

/// Serializes KFD page installation and records the GPU mappings that existed
/// when the page was offered. Hold this through native event creation.
#[doc(hidden)]
pub struct ProcessSignalEventPageGuard(MutexGuard<'static, ProcessPage>);

impl ProcessSignalEventPageGuard {
    /// Address of the page KFD may have installed, including uncertain offers.
    #[must_use]
    pub fn host_address(&self) -> Option<usize> {
        self.0.host_address
    }

    /// GPU mailbox base recorded for this endpoint when the page was offered.
    #[must_use]
    pub fn device_address(&self, endpoint_id: [u8; 16]) -> Option<u64> {
        self.0
            .mappings
            .iter()
            .find_map(|(id, address)| (*id == endpoint_id).then_some(*address))
    }

    /// Whether a process-lifetime device could currently install or use a page.
    #[must_use]
    pub fn can_create_for(&self, endpoint_id: [u8; 16]) -> bool {
        self.0.host_address.is_none()
            || (self.0.confirmed && self.device_address(endpoint_id).is_some())
    }

    /// Records a dispatched page offer, even if KFD returned an error after
    /// installing it. The caller retains that page until process teardown.
    ///
    /// # Errors
    /// Rejects a conflicting page identity or empty mapping set.
    pub fn record_offer(
        &mut self,
        host_address: usize,
        mappings: Vec<([u8; 16], u64)>,
    ) -> Result<(), Error> {
        if host_address == 0 || mappings.is_empty() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "KFD signal event page has no valid mapping",
            });
        }
        if let Some(existing) = self.0.host_address {
            if existing != host_address || self.0.mappings != mappings {
                return Err(Error::Operation {
                    kind: ErrorKind::DriverContract,
                    detail: "KFD signal event page identity changed",
                });
            }
        } else {
            self.0.host_address = Some(host_address);
            self.0.mappings = mappings;
        }
        Ok(())
    }

    /// Marks the process page as installed after a valid event was created.
    pub fn confirm(&mut self) {
        self.0.confirmed = true;
    }

    /// Whether KFD has returned a valid event for the process page.
    #[must_use]
    pub fn is_confirmed(&self) -> bool {
        self.0.confirmed
    }
}

/// Locks the process KFD page identity across an event creation transaction.
///
/// # Errors
/// Reports poisoned process state.
pub fn lock_process_signal_event_page() -> Result<ProcessSignalEventPageGuard, Error> {
    process_page_slot()
        .page
        .lock()
        .map(ProcessSignalEventPageGuard)
        .map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "KFD process signal page lock poisoned",
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
    /// Native KFD event retained until destruction succeeds.
    pub(crate) inner: Owned<driver::KfdSignalEvent>,
    /// Stable event ID and mailbox slot returned to the caller.
    pub(crate) info: SignalEventInfo,
}

impl SignalEvent {
    /// Returns the immutable event identity and mailbox slot.
    #[must_use]
    pub fn info(&self) -> SignalEventInfo {
        self.info
    }

    /// Signals this native event for a host observer.
    ///
    /// # Errors
    /// Reports a native event or context failure.
    pub fn signal(&self) -> Result<(), Error> {
        self.inner.signal()
    }

    /// Waits for a native notification. A successful wake is a hint to
    /// recheck the caller's authoritative completion or stop predicate.
    ///
    /// # Errors
    /// Reports a native event or context failure.
    pub fn wait(&self, timeout_nanoseconds: u64) -> Result<bool, Error> {
        self.inner.wait(timeout_nanoseconds)
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
    /// Allocated page retained until explicit transfer or conclusive cleanup.
    allocation: Option<Allocation>,
    /// KFD may have installed the page even if event creation failed.
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

    /// Returns this page's established GPU address on a mapped device.
    ///
    /// # Errors
    /// Rejects a device outside the allocation's live access set.
    pub fn device_address(&self, device: &Device) -> Result<u64, Error> {
        self.allocation
            .as_ref()
            .ok_or(Error::Operation {
                kind: ErrorKind::Internal,
                detail: "signal event page lost its allocation",
            })?
            .device_address(device)
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
        driver::KfdDriver::retain_kfd_signal_event_page(
            allocation.inner.linux_kfd_mut()?.driver_state_mut(),
        )?;
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
            Some(allocation.inner.linux_kfd()?.driver_state())
        }
        None => None,
    };
    let (driver, state) = device.device.driver_state.linux_kfd()?;
    let mut page_offered = false;
    let result = driver.create_kfd_signal_event(state, native_page, &mut page_offered);
    if let Some(page) = event_page {
        // KFD can install a page even when CREATE_EVENT reports an error.
        // Validation and metadata allocation before dispatch do not offer it.
        page.attempted |= page_offered;
    }
    let inner = result?;
    let info = inner.info();
    Ok(SignalEvent { inner, info })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    static TEST_PAGE: Mutex<ProcessPage> = Mutex::new(ProcessPage {
        host_address: None,
        mappings: Vec::new(),
        confirmed: false,
    });

    #[test]
    fn offered_process_page_limits_capability_to_its_mapped_endpoints() {
        let first = [1; 16];
        let second = [2; 16];
        let mut page = ProcessSignalEventPageGuard(TEST_PAGE.lock().unwrap());
        assert!(page.can_create_for(first));
        assert!(page.can_create_for(second));
        page.record_offer(0x1000, vec![(first, 0x2000)]).unwrap();
        assert!(!page.can_create_for(first));
        page.confirm();
        assert!(page.can_create_for(first));
        assert!(!page.can_create_for(second));
        assert_eq!(page.device_address(first), Some(0x2000));
        assert!(page.record_offer(0x3000, vec![(second, 0x3000)]).is_err());
    }
}
