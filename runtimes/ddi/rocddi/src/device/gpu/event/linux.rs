// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Linux KFD signal-event interoperability.

use crate::device::gpu::GpuDevice;
use crate::device::gpu::queue::QueueErrorEvent;
use crate::device::{Device, DeviceResources};
use crate::driver::{self, DeviceDriverState, Driver};
use crate::memory::Allocation;
use crate::memory::device::{DeviceAccess, MemoryKind};
use crate::memory::host::host_page_size;
use crate::memory::storage::{Owned, TryPush};
use crate::session::DriverContextLifetime;
use crate::{Error, ErrorKind};
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{Mutex, MutexGuard};

type EventPageMappings = Vec<([u8; 16], u64)>;

/// Process-wide KFD page identity shared by the AMDF and HSA adapters.
/// The service that first offered the page retains its allocation until it
/// transfers that allocation to process lifetime during shutdown.
struct ProcessPage {
    host_address: Option<usize>,
    mappings: EventPageMappings,
    confirmed: bool,
    /// The offering service retired without proving that KFD installed it.
    retired_unconfirmed: bool,
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
                retired_unconfirmed: false,
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
struct ProcessSignalEventPageGuard(MutexGuard<'static, ProcessPage>);

impl ProcessSignalEventPageGuard {
    /// GPU mailbox base recorded for this endpoint when the page was offered.
    #[must_use]
    fn device_address(&self, endpoint_id: [u8; 16]) -> Option<u64> {
        self.0
            .mappings
            .iter()
            .find_map(|(id, address)| (*id == endpoint_id).then_some(*address))
    }

    /// Whether a process-lifetime device could currently install or use a page.
    #[must_use]
    fn can_create_for(&self, endpoint_id: [u8; 16]) -> bool {
        !self.0.retired_unconfirmed
            && (self.0.host_address.is_none()
                || (self.0.confirmed && self.device_address(endpoint_id).is_some()))
    }

    fn compatible_mappings(
        &self,
        endpoints: impl IntoIterator<Item = [u8; 16]>,
        require_identity: bool,
    ) -> Result<Option<(usize, u64)>, Error> {
        if self.0.retired_unconfirmed {
            return Err(Error::Operation {
                kind: ErrorKind::DriverContract,
                detail: "KFD signal page installation was never confirmed",
            });
        }
        let Some(host) = self.0.host_address else {
            return Ok(None);
        };
        let mut selected = None;
        for endpoint in endpoints {
            let address = self.device_address(endpoint).ok_or(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "KFD signal page was not mapped to this GPU",
            })?;
            if require_identity && address != host as u64 {
                return Err(Error::Operation {
                    kind: ErrorKind::Unsupported,
                    detail: "KFD signal page requires identical CPU and GPU addresses",
                });
            }
            selected.get_or_insert(address);
        }
        Ok(selected.map(|address| (host, address)))
    }

    /// Records a dispatched page offer, even if KFD returned an error after
    /// installing it. The caller retains that page until process teardown.
    ///
    /// # Errors
    /// Rejects a conflicting page identity or empty mapping set.
    fn record_offer(
        &mut self,
        host_address: usize,
        mappings: EventPageMappings,
    ) -> Result<(), Error> {
        if self.0.retired_unconfirmed {
            return Err(Error::Operation {
                kind: ErrorKind::DriverContract,
                detail: "retired KFD signal page cannot be offered again",
            });
        }
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
    fn confirm(&mut self) {
        self.0.confirmed = true;
    }

    /// Prevents another frontend from assuming that an offered page was
    /// installed after its last owner has retired.
    fn retire_unconfirmed(&mut self) {
        if !self.0.confirmed {
            self.0.retired_unconfirmed = true;
        }
    }
}

/// Locks the process KFD page identity across an event creation transaction.
///
/// # Errors
/// Reports poisoned process state.
fn lock_process_signal_event_page() -> Result<ProcessSignalEventPageGuard, Error> {
    process_page_slot()
        .page
        .lock()
        .map(ProcessSignalEventPageGuard)
        .map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "KFD process signal page lock poisoned",
        })
}

/// Reports whether the process KFD signal page can serve this activated device.
/// An offered but unconfirmed page cannot yet advertise native events.
///
/// # Errors
/// Reports poisoned process state.
pub fn process_signal_event_available(device: &Device) -> Result<bool, Error> {
    Ok(lock_process_signal_event_page()?.can_create_for(device.endpoint().id))
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
struct SignalEventPage {
    /// Allocated page retained until explicit transfer or conclusive cleanup.
    allocation: Option<Allocation>,
    /// KFD may have installed the page even if event creation failed.
    attempted: bool,
}

impl SignalEventPage {
    /// Wraps a page allocation before it is offered to KFD.
    #[must_use]
    fn new(allocation: Allocation) -> Self {
        Self {
            allocation: Some(allocation),
            attempted: false,
        }
    }

    /// Returns the existing host mapping of the page, if any.
    #[must_use]
    fn host_address(&self) -> Option<usize> {
        self.allocation.as_ref()?.info().host_address
    }

    /// Returns this page's established GPU address on a mapped device.
    ///
    /// # Errors
    /// Rejects a device outside the allocation's live access set.
    fn device_address(&self, device: &Device) -> Result<u64, Error> {
        self.allocation
            .as_ref()
            .ok_or(Error::Operation {
                kind: ErrorKind::Internal,
                detail: "signal event page lost its allocation",
            })?
            .device_address(device)
    }

    /// Transfers a page used in an event creation attempt to KFD process
    /// lifetime. On error, this owner still retains the complete allocation.
    ///
    /// # Errors
    /// Rejects an allocation that is not a live, host-visible KFD event page.
    fn retain_for_process(&mut self) -> Result<(), Error> {
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

const SIGNAL_EVENT_PAGE_BYTES: u64 = 4096 * 8;

/// Address relationship required by the frontend's signal representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SignalEventAddressing {
    /// The signal publishes one address usable by both CPU and GPU.
    Identity,
    /// The signal publishes the GPU mailbox address separately.
    Mapped,
}

impl SignalEventAddressing {
    fn requires_identity(self) -> bool {
        self == Self::Identity
    }
}

/// Addresses of one KFD event mailbox in the process page.
#[derive(Clone, Copy, Debug)]
pub struct SignalEventMailbox {
    /// CPU address used by a host signal implementation.
    pub host: usize,
    /// GPU address published to a device-side producer.
    pub device: u64,
}

/// Owns first-event page preparation, native installation, and uncertain-page
/// retention for one frontend's KFD context. Process services coordinate via
/// the shared process-page identity; session services keep a private page.
pub struct SignalEventService {
    lifetime: DriverContextLifetime,
    process_id: u32,
    page: Option<SignalEventPage>,
    confirmed: bool,
}

impl SignalEventService {
    /// Creates an inert service without allocating or installing a page.
    #[must_use]
    pub fn new(lifetime: DriverContextLifetime) -> Self {
        Self {
            lifetime,
            process_id: std::process::id(),
            page: None,
            confirmed: false,
        }
    }

    /// Checks whether a session has external driver owners before this
    /// service releases its internally retained signal page. A Busy result
    /// leaves the page and event creation state unchanged.
    ///
    /// # Errors
    /// Reports a foreign process, a page from another session, or a live
    /// external owner of the session or one of its drivers.
    pub fn preflight_session_destroy(
        &self,
        session: &mut crate::session::Session,
    ) -> Result<(), Error> {
        self.ensure_process()?;
        session.preflight_destroy_allowing(
            self.page.as_ref().and_then(|page| page.allocation.as_ref()),
        )
    }

    fn ensure_process(&self) -> Result<(), Error> {
        if self.process_id == std::process::id() {
            Ok(())
        } else {
            Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "KFD signal event service belongs to another process",
            })
        }
    }

    /// Install the one KFD page on every GPU this connection can discover.
    /// A later frontend may activate a different GPU after the first page
    /// offer, when KFD no longer permits its mapping set to grow.
    fn event_page_peers(device: &Device) -> Result<Vec<Device>, Error> {
        let (driver, _) = device.driver_state.linux_kfd()?;
        let mut peers = Vec::new();
        driver.enumerate(&mut |endpoint| {
            if endpoint.gpu().is_none() || endpoint.id == device.endpoint().id {
                return Ok(());
            }
            // A peer unavailable to this frontend must not prevent creation
            // on the requested GPU. Its later activation will report the
            // missing page mapping through compatible_mappings.
            let Ok(state) = driver.activate(&endpoint) else {
                return Ok(());
            };
            let driver_state = DeviceDriverState::LinuxKfd {
                driver: driver.clone(),
                state,
            };
            let Ok(resources) = DeviceResources::for_activated(&endpoint, &driver_state) else {
                return Ok(());
            };
            peers.try_push(Device {
                driver_state,
                endpoint,
                resources,
            })?;
            Ok(())
        })?;
        Ok(peers)
    }

    fn prepare_page(&mut self, device: &Device, peers: &[&Device]) -> Result<(), Error> {
        let page_size = host_page_size()?;
        let size = SIGNAL_EVENT_PAGE_BYTES
            .div_ceil(page_size)
            .checked_mul(page_size)
            .ok_or(Error::Capacity {
                resource: "KFD signal page size",
            })?;
        let allocation = device.allocate_with_peers(
            peers,
            MemoryKind::System,
            size,
            page_size,
            DeviceAccess::READ | DeviceAccess::WRITE,
        )?;
        let info = allocation.info();
        if info.host_address.is_none() || info.size < SIGNAL_EVENT_PAGE_BYTES {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "KFD signal page requires host-visible system memory",
            });
        }
        self.page = Some(SignalEventPage::new(allocation));
        Ok(())
    }

    fn page_mappings(
        page: &SignalEventPage,
        device: &Device,
        peers: &[&Device],
        require_identity: bool,
    ) -> Result<(usize, EventPageMappings), Error> {
        let host = page.host_address().ok_or(Error::Operation {
            kind: ErrorKind::Internal,
            detail: "KFD signal page lost its host mapping",
        })?;
        let mut mappings = Vec::new();
        mappings
            .try_reserve_exact(peers.len() + 1)
            .map_err(|_| Error::Capacity {
                resource: "KFD signal page mappings",
            })?;
        for mapped in std::iter::once(device).chain(peers.iter().copied()) {
            let address = page.device_address(mapped)?;
            if require_identity && address != host as u64 {
                return Err(Error::Operation {
                    kind: ErrorKind::Unsupported,
                    detail: "KFD signal page requires identical CPU and GPU addresses",
                });
            }
            mappings.push((mapped.endpoint().id, address));
        }
        Ok((host, mappings))
    }

    /// Creates an event for `device`. rocddi discovers every GPU for the
    /// process KFD connection before the first page is offered, because KFD
    /// cannot add mappings to an installed signal page.
    /// `addressing` specifies whether the frontend requires equal CPU and GPU
    /// addresses or can publish the distinct GPU mailbox address returned here.
    ///
    /// # Errors
    /// Reports incompatible existing process mappings, allocation or driver
    /// failures. An offered page remains retained even when CREATE fails.
    pub fn create(
        &mut self,
        device: &Device,
        addressing: SignalEventAddressing,
    ) -> Result<(SignalEvent, SignalEventMailbox), Error> {
        self.ensure_process()?;
        let require_identity = addressing.requires_identity();
        let mut process = if self.lifetime == DriverContextLifetime::Process {
            Some(lock_process_signal_event_page()?)
        } else {
            None
        };
        let existing = process
            .as_ref()
            .map(|guard| {
                guard.compatible_mappings(std::iter::once(device.endpoint().id), require_identity)
            })
            .transpose()?
            .flatten();
        let peer_devices = if existing.is_some() {
            Vec::new()
        } else {
            Self::event_page_peers(device)?
        };
        let peers = peer_devices.iter().collect::<Vec<_>>();
        let (host, address, install, mappings) = if let Some((host, address)) = existing {
            (host, address, false, Vec::new())
        } else {
            if self.page.is_some() && !self.confirmed {
                let usable = self.page.as_ref().is_some_and(|page| {
                    Self::page_mappings(page, device, &peers, require_identity).is_ok()
                });
                if !usable && self.page.as_ref().is_some_and(|page| page.attempted) {
                    return Err(Error::Operation {
                        kind: ErrorKind::Unsupported,
                        detail: "offered KFD signal page cannot change GPU mappings",
                    });
                }
                if !usable {
                    self.page = None;
                }
            }
            if self.page.is_none() {
                self.prepare_page(device, &peers)?;
            }
            let page = self.page.as_ref().ok_or(Error::Operation {
                kind: ErrorKind::Internal,
                detail: "KFD signal page preparation lost its owner",
            })?;
            let (host, mappings) = Self::page_mappings(page, device, &peers, require_identity)?;
            let address = mappings[0].1;
            (host, address, !self.confirmed, mappings)
        };
        let gpu = device.gpu()?;
        let first = if install {
            create_signal_event(gpu, self.page.as_mut())
        } else {
            create_signal_event(gpu, None)
        };
        let offered = install && self.page.as_ref().is_some_and(|page| page.attempted);
        if offered {
            if let Some(process) = process.as_mut() {
                process.record_offer(host, mappings)?;
            }
        }
        let event = retry_unconfirmed_event_page(
            first,
            offered,
            process
                .as_ref()
                .map_or(self.confirmed, |process| process.0.confirmed),
            || create_signal_event(gpu, None),
        )?;
        if let Some(process) = process.as_mut() {
            process.confirm();
        }
        self.confirmed = true;
        let mailbox = mailbox(host, address, event.info().event_page_slot_index)?;
        Ok((event, mailbox))
    }

    /// Resolves a retained event's mailbox after frontend event pooling.
    ///
    /// # Errors
    /// Reports a lost page identity or invalid native mailbox slot.
    pub fn mailbox(
        &self,
        device: &Device,
        event: &SignalEvent,
    ) -> Result<SignalEventMailbox, Error> {
        self.ensure_process()?;
        let (host, address) = if self.lifetime == DriverContextLifetime::Process {
            let process = lock_process_signal_event_page()?;
            process
                .compatible_mappings(std::iter::once(device.endpoint().id), false)?
                .ok_or(Error::Operation {
                    kind: ErrorKind::Internal,
                    detail: "KFD process signal page is missing",
                })?
        } else {
            let page = self.page.as_ref().ok_or(Error::Operation {
                kind: ErrorKind::Internal,
                detail: "KFD session signal page is missing",
            })?;
            let host = page.host_address().ok_or(Error::Operation {
                kind: ErrorKind::Internal,
                detail: "KFD session signal page lost its host mapping",
            })?;
            (host, page.device_address(device)?)
        };
        mailbox(host, address, event.info().event_page_slot_index)
    }

    /// Transfers an offered page to KFD process lifetime before its session
    /// can close. Returns whether no event ever confirmed an offered page.
    ///
    /// # Errors
    /// Retains the page for a retry if native transfer fails.
    pub fn release(&mut self) -> Result<bool, Error> {
        self.ensure_process()?;
        let mut process = if self.lifetime == DriverContextLifetime::Process {
            Some(lock_process_signal_event_page()?)
        } else {
            None
        };
        let unresolved = self.page.as_ref().is_some_and(|page| page.attempted)
            && !process
                .as_ref()
                .map_or(self.confirmed, |page| page.0.confirmed);
        if let Some(page) = self.page.as_mut() {
            page.retain_for_process()?;
        }
        if unresolved && let Some(process) = process.as_mut() {
            process.retire_unconfirmed();
        }
        self.page = None;
        Ok(unresolved)
    }
}

fn mailbox(host: usize, address: u64, slot: u32) -> Result<SignalEventMailbox, Error> {
    let offset = u64::from(slot).checked_mul(8).ok_or(Error::Operation {
        kind: ErrorKind::InvalidData,
        detail: "KFD signal mailbox offset overflowed",
    })?;
    let host_offset = usize::try_from(offset).map_err(|_| Error::Operation {
        kind: ErrorKind::InvalidData,
        detail: "KFD signal mailbox offset is not host-addressable",
    })?;
    Ok(SignalEventMailbox {
        host: host.checked_add(host_offset).ok_or(Error::Operation {
            kind: ErrorKind::InvalidData,
            detail: "KFD signal host mailbox overflowed",
        })?,
        device: address.checked_add(offset).ok_or(Error::Operation {
            kind: ErrorKind::InvalidData,
            detail: "KFD signal device mailbox overflowed",
        })?,
    })
}

fn retry_unconfirmed_event_page<T, E>(
    first: Result<T, E>,
    offered: bool,
    confirmed: bool,
    retry: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    match first {
        Ok(value) => Ok(value),
        Err(_) if offered && !confirmed => retry(),
        Err(error) => Err(error),
    }
}

/// Creates an auto-reset KFD signal event. The first event in a process supplies
/// an owned signal-event page; later events reuse that process page.
///
/// # Errors
/// Rejects an invalid page allocation and reports native event creation
/// failures without publishing a partial owner.
fn create_signal_event(
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
        retired_unconfirmed: false,
    });
    static TEST_COMPATIBLE_PAGE: Mutex<ProcessPage> = Mutex::new(ProcessPage {
        host_address: None,
        mappings: Vec::new(),
        confirmed: false,
        retired_unconfirmed: false,
    });
    static TEST_RETIRED_PAGE: Mutex<ProcessPage> = Mutex::new(ProcessPage {
        host_address: None,
        mappings: Vec::new(),
        confirmed: false,
        retired_unconfirmed: false,
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

    #[test]
    fn process_page_checks_each_frontends_mapping_requirements() {
        let mut page = ProcessSignalEventPageGuard(TEST_COMPATIBLE_PAGE.lock().unwrap());
        let first = [1; 16];
        let second = [2; 16];
        let late = [3; 16];
        page.record_offer(0x1000, vec![(first, 0x1000), (second, 0x2000)])
            .unwrap();
        page.confirm();
        // A second frontend may confirm an earlier owner's offered page.
        page.retire_unconfirmed();
        assert!(!page.0.retired_unconfirmed);
        assert_eq!(
            page.compatible_mappings([first], true).unwrap(),
            Some((0x1000, 0x1000))
        );
        assert_eq!(
            page.compatible_mappings([second], false).unwrap(),
            Some((0x1000, 0x2000))
        );
        assert!(page.compatible_mappings([second], true).is_err());
        assert!(page.compatible_mappings([first, late], true).is_err());
    }

    #[test]
    fn unconfirmed_page_cannot_be_reused_after_its_owner_retires() {
        let endpoint = [4; 16];
        let mut page = ProcessSignalEventPageGuard(TEST_RETIRED_PAGE.lock().unwrap());
        page.record_offer(0x1000, vec![(endpoint, 0x2000)]).unwrap();
        page.retire_unconfirmed();
        assert!(!page.can_create_for(endpoint));
        assert!(page.compatible_mappings([endpoint], false).is_err());
        assert!(page.record_offer(0x1000, vec![(endpoint, 0x2000)]).is_err());
    }

    #[test]
    fn signal_mailbox_checks_both_address_spaces() {
        let addresses = mailbox(0x1000, 0x2000, 3).unwrap();
        assert_eq!(addresses.host, 0x1018);
        assert_eq!(addresses.device, 0x2018);
        assert!(mailbox(usize::MAX - 3, 0x2000, 1).is_err());
        assert!(mailbox(0x1000, u64::MAX - 3, 1).is_err());
    }

    #[test]
    fn first_event_retries_only_after_the_page_was_offered() {
        let mut retries = 0;
        let result = retry_unconfirmed_event_page(Err::<u32, ()>(()), true, false, || {
            retries += 1;
            Ok(7)
        });
        assert_eq!(result, Ok(7));
        assert_eq!(retries, 1);

        let result = retry_unconfirmed_event_page(Err::<u32, ()>(()), false, false, || {
            retries += 1;
            Ok(8)
        });
        assert_eq!(result, Err(()));
        assert_eq!(retries, 1);

        let result = retry_unconfirmed_event_page(Err::<u32, ()>(()), true, true, || {
            retries += 1;
            Ok(9)
        });
        assert_eq!(result, Err(()));
        assert_eq!(retries, 1);
    }
}
