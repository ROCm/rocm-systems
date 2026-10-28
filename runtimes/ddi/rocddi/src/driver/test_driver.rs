// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Minimal non-GPU driver used to exercise heterogeneous session routing.

use super::{AddressSpaceInfo, AllocationOperations, CachedInfo, Driver, EndpointSelector};
use crate::device::event::DeviceEvent;
use crate::memory::host::HostBackingLease;
use crate::memory::storage::Owned;
use crate::memory::{AllocationInfo, DeviceAccess, HostRegistration, MemoryKind};
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, EndpointKind};
use crate::{Error, ErrorKind};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};

/// Test-only event stream shared by activated endpoints of one driver.
struct TestEvents {
    /// Events and waiters protected by the same lock as publication.
    state: Mutex<TestEventState>,
    /// Wakes subscribers after a new record is published.
    changed: Condvar,
    /// Lets a test publisher wait until a subscriber has entered its wait.
    wait_started: Condvar,
}

/// Mutable event records and waiter count for one test driver.
struct TestEventState {
    /// Records retained until delivery or as the latest of their kind.
    records: Vec<(u64, DeviceEvent)>,
    /// Last sequence assigned to a published event.
    sequence: u64,
    /// Independent positions of active subscribers.
    cursors: Vec<Arc<AtomicU64>>,
    /// Subscribers currently waiting for a new record.
    waiters: usize,
}

impl TestEventState {
    /// Mirrors KFD retention: undelivered records and the latest of each kind.
    fn prune(&mut self) {
        let delivered = self
            .cursors
            .iter()
            .map(|cursor| cursor.load(Ordering::Acquire))
            .min()
            .unwrap_or(self.sequence);
        let latest_memory = self.records.iter().rev().find_map(|(sequence, event)| {
            matches!(event, DeviceEvent::GpuMemoryFault(_)).then_some(*sequence)
        });
        let latest_hardware = self.records.iter().rev().find_map(|(sequence, event)| {
            matches!(event, DeviceEvent::GpuHardwareException(_)).then_some(*sequence)
        });
        self.records.retain(|(sequence, _)| {
            *sequence > delivered
                || Some(*sequence) == latest_memory
                || Some(*sequence) == latest_hardware
        });
    }

    fn has_next(&self, cursor: u64) -> bool {
        self.records.iter().any(|(sequence, _)| *sequence > cursor)
    }
}

/// One independent cursor over a test driver's event stream.
pub(crate) struct TestEventSubscription {
    /// Keeps the test event stream live after its device handle is dropped.
    events: Arc<TestEvents>,
    /// Last delivered sequence, shared with pruning.
    cursor: Arc<AtomicU64>,
}

impl TestEventSubscription {
    /// Reads one retained event for this subscriber.
    pub(crate) fn poll(&mut self) -> Result<Option<DeviceEvent>, Error> {
        let mut state = self.events.state.lock().map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "test event stream lock poisoned",
        })?;
        let event = state
            .records
            .iter()
            .find(|(sequence, _)| *sequence > self.cursor.load(Ordering::Acquire))
            .copied();
        if let Some((sequence, _)) = event {
            self.cursor.store(sequence, Ordering::Release);
        }
        state.prune();
        Ok(event.map(|(_, event)| event))
    }

    /// Waits until a new test record is ready or the timeout expires.
    pub(crate) fn wait_native(&self, timeout: std::time::Duration) -> Result<(), Error> {
        let mut state = self.events.state.lock().map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "test event stream lock poisoned",
        })?;
        if state.has_next(self.cursor.load(Ordering::Acquire)) {
            return Ok(());
        }
        state.waiters += 1;
        self.events.wait_started.notify_all();
        let (mut state, _) = self
            .events
            .changed
            .wait_timeout_while(state, timeout, |state| {
                !state.has_next(self.cursor.load(Ordering::Acquire))
            })
            .map_err(|_| Error::Operation {
                kind: ErrorKind::Internal,
                detail: "test event wait lock poisoned",
            })?;
        state.waiters -= 1;
        Ok(())
    }
}

impl Drop for TestEventSubscription {
    fn drop(&mut self) {
        let mut state = self
            .events
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state
            .cursors
            .retain(|cursor| !Arc::ptr_eq(cursor, &self.cursor));
        state.prune();
    }
}

/// Test driver instance with only the common driver contract.
pub(crate) struct TestDriver {
    instance: super::DriverInstanceId,
    id: [u8; 16],
    kind: EndpointKind,
    lifetime: DriverContextLifetime,
    host_registration: bool,
    pub(crate) shutdowns: Arc<AtomicUsize>,
    pub(crate) fail_shutdown: Arc<AtomicBool>,
    pub(crate) enumerations: Arc<AtomicUsize>,
    pub(crate) opens: Arc<AtomicUsize>,
    /// Event stream available from every activated test endpoint.
    events: Arc<TestEvents>,
}

/// Activated test endpoint. It intentionally has no GPU resource capabilities.
#[derive(Clone)]
pub(crate) struct TestDevice {
    id: [u8; 16],
    events: Arc<TestEvents>,
}

impl TestDevice {
    /// Subscribes to events emitted by the owning test driver.
    pub(crate) fn subscribe_events(&self) -> TestEventSubscription {
        let mut state = self
            .events
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let first = state
            .records
            .first()
            .map_or(state.sequence, |(sequence, _)| sequence - 1);
        let cursor = Arc::new(AtomicU64::new(first));
        state.cursors.push(cursor.clone());
        TestEventSubscription {
            events: self.events.clone(),
            cursor,
        }
    }
}

/// Numeric-only allocation used to test backing lifetimes across drivers.
pub(crate) struct TestAllocation {
    id: [u8; 16],
    host_address: usize,
    device_address: u64,
    size: u64,
    released: bool,
    backing: Option<HostBackingLease>,
}

impl CachedInfo for TestAllocation {
    type Info = AllocationInfo;

    fn cached_info(&self) -> AllocationInfo {
        AllocationInfo {
            device_address: self.device_address,
            host_address: Some(self.host_address),
            size: self.size,
            native_size: self.size,
            physical_backing_id: [0; 2],
        }
    }
}

impl TestDriver {
    pub(crate) fn new(id: [u8; 16], kind: EndpointKind, lifetime: DriverContextLifetime) -> Self {
        Self {
            instance: super::new_driver_instance(),
            id,
            kind,
            lifetime,
            host_registration: false,
            shutdowns: Arc::new(AtomicUsize::new(0)),
            fail_shutdown: Arc::new(AtomicBool::new(false)),
            enumerations: Arc::new(AtomicUsize::new(0)),
            opens: Arc::new(AtomicUsize::new(0)),
            events: Arc::new(TestEvents {
                state: Mutex::new(TestEventState {
                    records: Vec::new(),
                    sequence: 0,
                    cursors: Vec::new(),
                    waiters: 0,
                }),
                changed: Condvar::new(),
                wait_started: Condvar::new(),
            }),
        }
    }

    /// Emits a device event to every active test subscription.
    pub(crate) fn publish_event(&self, event: DeviceEvent) -> Result<(), Error> {
        let mut state = self.events.state.lock().map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "test event stream lock poisoned",
        })?;
        state.records.try_reserve(1).map_err(|_| Error::Capacity {
            resource: "test event stream",
        })?;
        state.sequence = state.sequence.checked_add(1).ok_or(Error::Capacity {
            resource: "test event sequence",
        })?;
        let sequence = state.sequence;
        state.records.push((sequence, event));
        state.prune();
        drop(state);
        self.events.changed.notify_all();
        Ok(())
    }

    /// Waits until a subscriber is blocked in the test event wait path.
    pub(crate) fn wait_for_subscriber(&self, timeout: std::time::Duration) -> Result<bool, Error> {
        let state = self.events.state.lock().map_err(|_| Error::Operation {
            kind: ErrorKind::Internal,
            detail: "test event stream lock poisoned",
        })?;
        let (state, _) = self
            .events
            .wait_started
            .wait_timeout_while(state, timeout, |state| state.waiters == 0)
            .map_err(|_| Error::Operation {
                kind: ErrorKind::Internal,
                detail: "test event wait-start lock poisoned",
            })?;
        Ok(state.waiters != 0)
    }

    pub(crate) fn with_host_registration(mut self) -> Self {
        self.host_registration = true;
        self
    }

    fn endpoint(&self) -> Endpoint {
        let mut name = [0; 128];
        name[..8].copy_from_slice(b"test-npu");
        Endpoint {
            id: self.id,
            name,
            product_name: None,
            pci: None,
            kind: self.kind,
            local_memory_bytes: 0,
            host_visible_local_memory_bytes: 0,
            host_local_cacheability: None,
            allocation_granularity: 4096,
            address_bit_count: 48,
            minimum_address: 0x1000,
            maximum_address: 0xffff,
            supported_permissions: if self.host_registration {
                DeviceAccess::READ | DeviceAccess::WRITE
            } else {
                DeviceAccess::NONE
            },
            driver_instance: self.instance,
            selector: EndpointSelector::Opaque(u64::from_le_bytes([
                self.id[0], self.id[1], self.id[2], self.id[3], self.id[4], self.id[5], self.id[6],
                self.id[7],
            ])),
        }
    }
}

impl AddressSpaceInfo for TestDevice {
    fn address_range(&self) -> (u64, u64) {
        (0x1000, 0xffff)
    }

    fn shares_address_domain(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Driver for TestDriver {
    type DeviceState = TestDevice;

    fn driver_instance(&self) -> super::DriverInstanceId {
        self.instance
    }

    fn context_lifetime(&self) -> DriverContextLifetime {
        self.lifetime
    }

    fn shutdown(&mut self) -> Result<(), Error> {
        self.shutdowns.fetch_add(1, Ordering::Relaxed);
        if self.fail_shutdown.swap(false, Ordering::Relaxed) {
            return Err(Error::Operation {
                kind: ErrorKind::Driver,
                detail: "injected shutdown failure",
            });
        }
        Ok(())
    }

    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.enumerations.fetch_add(1, Ordering::Relaxed);
        visitor(self.endpoint())
    }

    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        self.opens.fetch_add(1, Ordering::Relaxed);
        if id != self.id {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "unknown test endpoint",
            });
        }
        Ok(self.endpoint())
    }

    fn probe_endpoint(&self, id: [u8; 16]) -> Result<Option<Endpoint>, Error> {
        self.opens.fetch_add(1, Ordering::Relaxed);
        Ok((id == self.id).then(|| self.endpoint()))
    }

    fn populate_endpoint(&self, endpoint: &mut Endpoint) -> Result<(), Error> {
        endpoint.product_name = Some("Test endpoint".to_owned());
        if let EndpointKind::Gpu { info } = &mut endpoint.kind {
            info.drm_asic_family_id = Some(42);
            info.counter_frequency_hz = Some(1_000_000);
        }
        Ok(())
    }

    fn activate(&self, endpoint: &Endpoint) -> Result<TestDevice, Error> {
        if !endpoint.matches_discovery(&self.endpoint()) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "foreign test endpoint",
            });
        }
        Ok(TestDevice {
            id: self.id,
            events: self.events.clone(),
        })
    }
}

impl AllocationOperations for TestDriver {
    type Allocation = TestAllocation;

    fn supports_host_registration(&self, _: &Endpoint) -> bool {
        self.host_registration
    }

    fn check_allocation(allocation: &Self::Allocation) -> Result<(), Error> {
        if allocation.released {
            Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "test registration has been freed",
            })
        } else {
            Ok(())
        }
    }

    fn allocation_is_device_local(_: &Self::Allocation) -> bool {
        false
    }

    fn allocation_device_address(
        allocation: &Self::Allocation,
        device: &Self::DeviceState,
    ) -> Result<u64, Error> {
        Self::check_allocation(allocation)?;
        if allocation.id != device.id {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "test device has no registration",
            });
        }
        Ok(allocation.device_address)
    }

    fn allocation_is_owned_by(allocation: &Self::Allocation, device: &Self::DeviceState) -> bool {
        allocation.id == device.id && !allocation.released
    }

    fn free_allocation(allocation: &mut Self::Allocation) -> Result<(), Error> {
        allocation.released = true;
        allocation.backing = None;
        Ok(())
    }

    fn allocate_owned(
        &self,
        _: &Self::DeviceState,
        _: &[&Self::DeviceState],
        _: MemoryKind,
        _: u64,
        _: u64,
        _: DeviceAccess,
    ) -> Result<Owned<Self::Allocation>, Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "test driver has no owned allocation",
        })
    }

    /// # Safety
    /// Borrowed callers keep the supplied host pages mapped until registration
    /// ends; owned requests transfer a backing lease into the allocation.
    #[allow(unsafe_code)]
    unsafe fn register_host(
        &self,
        device: &Self::DeviceState,
        peers: &[&Self::DeviceState],
        request: HostRegistration,
    ) -> Result<Owned<Self::Allocation>, Error> {
        if !self.host_registration
            || request.address == 0
            || request.size == 0
            || !request.alignment.is_power_of_two()
            || !request.permissions.contains(DeviceAccess::READ)
            || peers.iter().any(|peer| peer.id != device.id)
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "invalid test host registration",
            });
        }
        let device_address = u64::from(self.id[0]) * 0x1_0000 + 0x1_0000;
        Ok(Owned::new(TestAllocation {
            id: device.id,
            host_address: request.address,
            device_address,
            size: request.size,
            released: false,
            backing: request.backing,
        })?)
    }
}
