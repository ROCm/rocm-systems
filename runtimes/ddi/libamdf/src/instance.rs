// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Passive endpoint descriptions and explicit native device activation.
//!
//! Handles borrow their parent exactly as the AMDF contract requires. A cold
//! instance-local list recognizes registration of a provider-owned host view;
//! metadata queries never search or lock that list.

use crate::generated::amdf::*;
use crate::memory::Scope;
use crate::platform;
use crate::support::*;
use rocddi::device::event::{
    DeviceEvent, DeviceEventOriginId, DeviceEventSourceId, DeviceEventSubscription,
    GpuHardwareException, subscribe,
};
use rocddi::device::gpu::event::{SignalEventService, signal_event_available};
use rocddi::host::cache::{CacheMaintenance, CachePolicy};
use rocddi::memory::storage::TryPush;
use rocddi::memory::storage::{Allocator, Owned, OwnerVec, Shared, UninitOwned};
use rocddi::{device as native_device, memory as native_memory, session, topology};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

const INITIAL_RESET_EPOCH: u64 = 1;

/// Root AMDF owner containing native session state and embedded host scope.
///
/// `children` prevents destruction while endpoint handles remain public.
/// Cached host facts stay valid until destruction succeeds.
pub(crate) struct Instance {
    /// Native session that discovers and activates devices.
    pub native: session::Session,
    /// Callback policy for public frontend owner records.
    pub host_allocator: Allocator,
    /// Number of public children that still borrow this instance.
    pub children: AtomicU64,
    /// Prevents new children during destruction.
    pub closing: bool,
    /// Source of public device IDs within this instance.
    pub ids: AtomicU64,
    /// Head of the linked list of provider-owned host backings.
    pub host_backing_head: Mutex<usize>,
    /// Native signal page service for the process KFD connection.
    pub event_service: Mutex<SignalEventService>,
    /// One polling hub per installed driver event source.
    event_hubs: Mutex<OwnerVec<EventHubRecord>>,
    /// Host page size cached for memory profile queries.
    pub host_page_size: u64,
    /// Qualified write-back recipe retained for mapping capability queries.
    pub host_write_back_cache: Option<CacheMaintenance>,
    /// Embedded host memory profile namespace.
    pub scope: Scope,
}

impl Instance {
    /// Rejects an inherited handle before entering frontend mutexes or
    /// releasing native owners that belong to its creating process.
    pub(crate) fn require_current_process(&self) -> Result<(), u64> {
        if self.native.is_current_process() {
            Ok(())
        } else {
            Err(PRECONDITION)
        }
    }

    /// Reports the immutable native lifetime selected for this instance.
    pub(crate) fn process_lifetime(&self) -> bool {
        self.native.driver_context_lifetime() == session::DriverContextLifetime::Process
    }

    /// Reserves frontend owner storage under this instance's callback policy.
    pub(crate) fn owner_slot<T>(&self) -> Result<UninitOwned<T>, u64> {
        self.require_current_process()?;
        self.host_allocator
            .with_scope(|| Owned::<T>::try_new_uninit().map_err(|_| EXHAUSTED))
    }

    /// Creates a shared frontend owner retaining the same callback policy.
    fn shared<T>(&self, value: T) -> Result<Shared<T>, u64> {
        self.require_current_process()?;
        self.host_allocator
            .with_scope(|| Shared::new(value).map_err(|_| EXHAUSTED))
    }
}

/// Passive endpoint handle borrowing its instance and owning no device state.
pub(crate) struct Endpoint {
    /// Parent instance borrowed by this public endpoint handle.
    pub instance: *mut Instance,
    /// Passive topology record used for activation and metadata queries.
    pub native: topology::Endpoint,
    /// Host NUMA locality captured during passive open.
    pub host_numa_node: u32,
    /// Number of public devices that borrow this endpoint.
    pub children: AtomicU64,
    /// Embedded endpoint memory profile namespace.
    pub scope: Scope,
}

impl Endpoint {
    /// Rejects an inherited endpoint before its instance can allocate or
    /// activate native state.
    pub(crate) fn require_current_process(&self) -> Result<(), u64> {
        // SAFETY: A live endpoint borrows its live instance.
        unsafe { &*self.instance }.require_current_process()
    }
}

/// Explicitly activated device handle borrowing its endpoint.
///
/// Queue ownership is counted for BUSY destruction. The endpoint observer
/// advances this handle's reset epoch on native exception notifications;
/// operation errors also advance it when they observe terminal loss first.
pub(crate) struct Device {
    /// Parent endpoint borrowed by this public device handle.
    pub endpoint: *mut Endpoint,
    /// Activated native device and its retained driver state.
    pub native: native_device::Device,
    /// Public identity assigned at activation.
    pub id: amdf_device_id_t,
    /// Number of public queues that borrow this device.
    pub queues: AtomicU64,
    /// Public native events borrowing this device.
    pub events: AtomicU64,
    /// Native signal-page capability sampled at device activation.
    pub native_events_available: bool,
    /// Generation shared with the endpoint's exception observer.
    pub reset_epoch: Shared<AtomicU64>,
    /// Keeps exception polling alive while this handle is public.
    event_observer: Shared<EventObserver>,
}

/// Polling hub and worker for one driver event source in this instance.
struct EventHubRecord {
    source: DeviceEventSourceId,
    hub: Shared<EventHub>,
    worker: Option<JoinHandle<()>>,
}

/// Instance hub distributing one subscription to its endpoint observers.
struct EventHub {
    state: Mutex<EventHubState>,
    stop: AtomicBool,
    polling_failed: Shared<AtomicBool>,
}

struct EventHubState {
    observers: OwnerVec<ObserverRecord>,
    /// Preserve one reset per native GPU for endpoints activated after polling.
    latest_hardware_events: OwnerVec<RetainedHardwareEvent>,
    /// Loss of device memory invalidates every endpoint on this connection.
    latest_memory_loss: Option<DeviceEvent>,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum HardwareEventKey {
    Origin(DeviceEventOriginId),
    Endpoint([u8; 16]),
}

struct RetainedHardwareEvent {
    key: HardwareEventKey,
    event: DeviceEvent,
}

impl HardwareEventKey {
    fn from_exception(exception: GpuHardwareException) -> Option<Self> {
        exception
            .origin_id
            .map(Self::Origin)
            .or_else(|| exception.endpoint_id.map(Self::Endpoint))
    }
}

/// One endpoint's reset state and registered public device handles.
struct ObserverRecord {
    /// Endpoint whose device handles receive these events.
    endpoint_id: [u8; 16],
    /// Observer shared by those handles.
    observer: Shared<EventObserver>,
}

/// Fans reset evidence out to every public handle for one endpoint.
struct EventObserver {
    /// Filters process-level native events for this endpoint.
    endpoint_id: [u8; 16],
    /// Identifies events observed before public endpoint activation.
    origin_id: Option<DeviceEventOriginId>,
    /// Serialized reset generation and registered device epochs.
    state: Mutex<ObserverState>,
    /// A terminal polling failure prevents reliable exception observation.
    polling_failed: Shared<AtomicBool>,
}

/// Reset generations and deduplication state serialized with handle registration.
struct ObserverState {
    /// Latest reset generation published by this observer.
    generation: u64,
    /// Epoch cells for the currently registered public device handles.
    epochs: OwnerVec<Shared<AtomicU64>>,
    /// A connection-wide terminal loss cannot become valid again.
    terminal_loss_observed: bool,
    /// A queue reported loss before its corresponding exception was polled.
    operation_reset_pending: bool,
}

impl ObserverState {
    fn new(allocator: Allocator) -> Self {
        Self {
            generation: INITIAL_RESET_EPOCH,
            epochs: OwnerVec::with_allocator(allocator),
            terminal_loss_observed: false,
            operation_reset_pending: false,
        }
    }

    fn latest_epoch(&self) -> u64 {
        self.epochs
            .iter()
            .map(|epoch| epoch.load(Ordering::Acquire))
            .max()
            .unwrap_or(self.generation)
            .max(self.generation)
    }

    fn publish_epoch(&self, generation: u64) {
        for epoch in &self.epochs {
            epoch.fetch_max(generation, Ordering::AcqRel);
        }
    }
}

impl EventObserver {
    fn register_epoch(&self, epoch: Shared<AtomicU64>) -> Result<(), u64> {
        let mut state = self.state.lock().map_err(|_| INTERNAL)?;
        if self.polling_failed.load(Ordering::Acquire) {
            return Err(INTERNAL);
        }
        // Holding the epoch list lock pairs registration with event delivery:
        // a newly activated handle observes delivered events and any terminal
        // operation report that reached another handle first.
        epoch.fetch_max(state.latest_epoch(), Ordering::AcqRel);
        state.epochs.try_push(epoch).map_err(|_| EXHAUSTED)
    }

    fn unregister_epoch(&self, epoch: &Shared<AtomicU64>) -> Result<bool, u64> {
        let mut state = self.state.lock().map_err(|_| INTERNAL)?;
        let index = state
            .epochs
            .iter()
            .position(|candidate| Shared::ptr_eq(candidate, epoch))
            .ok_or(INTERNAL)?;
        let last = state.epochs.len() - 1;
        state.epochs.as_mut_slice().swap(index, last);
        let _ = state.epochs.pop();
        Ok(state.epochs.is_empty())
    }

    fn record_event(&self, event: DeviceEvent) -> Result<(), u64> {
        if event_resets_device(event, self.endpoint_id, self.origin_id) {
            let mut state = self.state.lock().map_err(|_| INTERNAL)?;
            // An operation can report loss before this notification is polled.
            // Adopt its advanced epoch instead of counting the same reset twice.
            let generation = state.latest_epoch().max(state.generation.saturating_add(1));
            state.generation = generation;
            state.publish_epoch(generation);
            state.operation_reset_pending = false;
            if matches!(event, DeviceEvent::GpuHardwareException(exception) if exception.memory_lost)
            {
                state.terminal_loss_observed = true;
            }
        }
        Ok(())
    }

    /// Counts an operation's loss once. A queue supplies its creation epoch;
    /// device-level terminal loss reads the current epoch under this lock.
    fn record_operation_loss(&self, epoch: &AtomicU64, resource_epoch: Option<u64>) {
        // Only epoch atomics and scalar state change while this lock is held.
        // Recovery from poison still invalidates resources on terminal loss.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.terminal_loss_observed {
            return;
        }
        let terminal = resource_epoch.is_none();
        if state.operation_reset_pending {
            if terminal {
                state.publish_epoch(state.latest_epoch());
            }
            state.terminal_loss_observed = terminal;
            return;
        }
        let previous = epoch.load(Ordering::Acquire);
        advance_reset_epoch(epoch, resource_epoch.unwrap_or(previous));
        if epoch.load(Ordering::Acquire) != previous {
            if terminal {
                state.publish_epoch(epoch.load(Ordering::Acquire));
            }
            state.operation_reset_pending = !terminal;
            state.terminal_loss_observed = terminal;
        }
    }
}

impl EventHub {
    fn register_endpoint(
        &self,
        endpoint_id: [u8; 16],
        origin_id: Option<DeviceEventOriginId>,
        epoch: Shared<AtomicU64>,
        allocator: Allocator,
        create: impl FnOnce(EventObserver) -> Result<Shared<EventObserver>, u64>,
    ) -> Result<Shared<EventObserver>, u64> {
        let mut state = self.state.lock().map_err(|_| INTERNAL)?;
        if let Some(record) = state
            .observers
            .iter()
            .find(|record| record.endpoint_id == endpoint_id)
        {
            record.observer.register_epoch(epoch)?;
            return Ok(record.observer.clone());
        }
        state.observers.try_reserve(1).map_err(|_| EXHAUSTED)?;
        let observer = create(EventObserver {
            endpoint_id,
            origin_id,
            state: Mutex::new(ObserverState::new(allocator)),
            polling_failed: self.polling_failed.clone(),
        })?;
        let key = origin_id.map_or(
            HardwareEventKey::Endpoint(endpoint_id),
            HardwareEventKey::Origin,
        );
        let replay = state.latest_memory_loss.or_else(|| {
            state
                .latest_hardware_events
                .iter()
                .find(|retained| retained.key == key)
                .map(|retained| retained.event)
        });
        if let Some(event) = replay {
            observer.record_event(event)?;
        }
        observer.register_epoch(epoch)?;
        state
            .observers
            .try_push(ObserverRecord {
                endpoint_id,
                observer: observer.clone(),
            })
            .map_err(|_| EXHAUSTED)?;
        Ok(observer)
    }

    fn record_event(&self, event: DeviceEvent) -> Result<(), u64> {
        let mut state = self.state.lock().map_err(|_| INTERNAL)?;
        if let DeviceEvent::GpuHardwareException(exception) = event {
            if exception.memory_lost {
                state.latest_memory_loss = Some(event);
            } else if let Some(key) = HardwareEventKey::from_exception(exception) {
                if let Some(retained) = state
                    .latest_hardware_events
                    .iter_mut()
                    .find(|retained| retained.key == key)
                {
                    retained.event = event;
                } else {
                    state
                        .latest_hardware_events
                        .try_push(RetainedHardwareEvent { key, event })
                        .map_err(|_| EXHAUSTED)?;
                }
            }
        }
        for record in &state.observers {
            record.observer.record_event(event)?;
        }
        Ok(())
    }

    fn record_polling_failure(&self) {
        let _state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        self.polling_failed.store(true, Ordering::Release);
    }
}

pub(crate) fn advance_reset_epoch(epoch: &AtomicU64, resource_epoch: u64) {
    epoch.fetch_max(resource_epoch.saturating_add(1), Ordering::AcqRel);
}

impl Device {
    /// Rejects an inherited device before entering frontend or native state.
    pub(crate) fn require_current_process(&self) -> Result<(), u64> {
        // SAFETY: A live device borrows its live endpoint and instance.
        unsafe { device_instance(self) }.require_current_process()
    }

    /// Reports loss of the native exception observer without inventing reset.
    pub(crate) fn event_observation_failed(&self) -> bool {
        self.event_observer.polling_failed.load(Ordering::Acquire)
    }

    pub(crate) fn observe_loss(&self, resource_epoch: u64) {
        self.event_observer
            .record_operation_loss(&self.reset_epoch, Some(resource_epoch));
    }

    pub(crate) fn current_reset_epoch(&self) -> u64 {
        self.reset_epoch.load(Ordering::Acquire)
    }

    /// Records terminal device loss reported by a native operation before
    /// converting its error to an AMDF status.
    pub(crate) fn native_error_status(&self, error: &rocddi::Error) -> u64 {
        if error.kind() == rocddi::ErrorKind::DeviceLost {
            self.event_observer
                .record_operation_loss(&self.reset_epoch, None);
        }
        native(error)
    }
}

/// Applies AMDF reset-epoch policy to events from one DDI driver source.
///
/// The worker owns one subscription and a shared hub. Destruction joins it
/// before the final public handle releases the native connection.
fn device_event_worker(mut subscription: DeviceEventSubscription, hub: &EventHub) {
    device_event_worker_with(|timeout| subscription.wait(timeout), thread::sleep, hub);
}

/// Runs the observer policy with an injectable event source and retry pause.
fn device_event_worker_with(
    mut wait: impl FnMut(Duration) -> Result<Option<DeviceEvent>, rocddi::Error>,
    mut pause: impl FnMut(Duration),
    hub: &EventHub,
) {
    let mut retry_delay = Duration::from_millis(1);
    let mut driver_failures = 0u8;
    while !hub.stop.load(Ordering::Acquire) {
        match wait(Duration::from_millis(1)) {
            Ok(Some(event)) => {
                retry_delay = Duration::from_millis(1);
                driver_failures = 0;
                if hub.record_event(event).is_err() {
                    hub.record_polling_failure();
                    return;
                }
            }
            Ok(None) => {
                retry_delay = Duration::from_millis(1);
                driver_failures = 0;
            }
            Err(error) if error.kind() == rocddi::ErrorKind::Busy => {
                driver_failures = 0;
                pause(retry_delay);
                retry_delay = (retry_delay * 2).min(Duration::from_millis(20));
            }
            Err(error) if error.kind() == rocddi::ErrorKind::Driver && driver_failures < 16 => {
                driver_failures += 1;
                pause(retry_delay);
                retry_delay = (retry_delay * 2).min(Duration::from_millis(20));
            }
            Err(_) => {
                hub.record_polling_failure();
                return;
            }
        }
    }
}

/// Registers a public handle with its endpoint observer. One subscription
/// and worker serve every endpoint on the same installed driver source.
fn register_event_observer(
    instance: &Instance,
    native: &native_device::Device,
    endpoint_id: [u8; 16],
    epoch: Shared<AtomicU64>,
) -> Result<Shared<EventObserver>, u64> {
    let source = native.event_source_id();
    let origin_id = native.event_origin_id();
    let mut hubs = instance.event_hubs.lock().map_err(|_| INTERNAL)?;
    if let Some(record) = hubs.iter().find(|record| record.source == source) {
        return record.hub.register_endpoint(
            endpoint_id,
            origin_id,
            epoch,
            instance.host_allocator,
            |observer| instance.shared(observer),
        );
    }

    hubs.try_reserve(1).map_err(|_| EXHAUSTED)?;
    let subscription = subscribe(native).map_err(|error| crate::support::native(&error))?;
    let polling_failed = instance.shared(AtomicBool::new(false))?;
    let observer = instance.shared(EventObserver {
        endpoint_id,
        origin_id,
        state: Mutex::new(ObserverState::new(instance.host_allocator)),
        polling_failed: polling_failed.clone(),
    })?;
    observer.register_epoch(epoch)?;
    let mut observers = OwnerVec::with_allocator(instance.host_allocator);
    observers
        .try_push(ObserverRecord {
            endpoint_id,
            observer: observer.clone(),
        })
        .map_err(|_| EXHAUSTED)?;
    let hub = instance.shared(EventHub {
        state: Mutex::new(EventHubState {
            observers,
            latest_hardware_events: OwnerVec::with_allocator(instance.host_allocator),
            latest_memory_loss: None,
        }),
        stop: AtomicBool::new(false),
        polling_failed,
    })?;
    hubs.try_push(EventHubRecord {
        source,
        hub: hub.clone(),
        worker: None,
    })
    .map_err(|_| EXHAUSTED)?;
    let worker_hub = hub.clone();
    let worker = thread::Builder::new()
        .name("amdf-driver-events".into())
        .spawn(move || device_event_worker(subscription, &worker_hub));
    let Ok(worker) = worker else {
        let _ = hubs.pop();
        return Err(EXHAUSTED);
    };
    if let Some(record) = hubs.last_mut() {
        record.worker = Some(worker);
    } else {
        hub.stop.store(true, Ordering::Release);
        let _ = worker.join();
        return Err(INTERNAL);
    }
    Ok(observer)
}

/// Removes a handle's epoch. The last handle on a driver source joins its
/// worker before the native device connection can be released.
fn unregister_event_observer(device: &Device) -> Result<(), u64> {
    // SAFETY: A live device borrows its live endpoint and instance.
    let instance = unsafe { &*(*device.endpoint).instance };
    let mut hubs = instance.event_hubs.lock().map_err(|_| INTERNAL)?;
    for hub_index in 0..hubs.len() {
        let mut state = hubs[hub_index].hub.state.lock().map_err(|_| INTERNAL)?;
        let Some(index) = state
            .observers
            .iter()
            .position(|record| Shared::ptr_eq(&record.observer, &device.event_observer))
        else {
            continue;
        };
        if !device
            .event_observer
            .unregister_epoch(&device.reset_epoch)?
        {
            return Ok(());
        }
        let last = state.observers.len() - 1;
        state.observers.as_mut_slice().swap(index, last);
        let _ = state.observers.pop();
        let final_handle = state.observers.is_empty();
        if final_handle {
            hubs[hub_index].hub.stop.store(true, Ordering::Release);
        }
        drop(state);
        let removed = if final_handle {
            let last = hubs.len() - 1;
            hubs.as_mut_slice().swap(hub_index, last);
            hubs.pop()
        } else {
            None
        };
        // The worker needs only the hub state. Join and release the final
        // callback-allocated hub after dropping the instance registry lock.
        drop(hubs);
        if let Some(record) = removed {
            if let Some(worker) = record.worker {
                let _ = worker.join();
            }
        }
        return Ok(());
    }
    Err(INTERNAL)
}

/// A VM fault reports failed work; only reset evidence invalidates this epoch.
pub(crate) fn event_resets_device(
    event: DeviceEvent,
    endpoint_id: [u8; 16],
    origin_id: Option<DeviceEventOriginId>,
) -> bool {
    match event {
        DeviceEvent::GpuMemoryFault(_) => false,
        DeviceEvent::GpuHardwareException(exception) => {
            exception.memory_lost
                || exception.endpoint_id == Some(endpoint_id)
                || (exception.endpoint_id.is_none()
                    && origin_id.is_some()
                    && exception.origin_id == origin_id)
        }
    }
}

pub(crate) fn endpoint_id(id: [u8; 16]) -> amdf_endpoint_id_t {
    let mut lo = [0; 8];
    let mut hi = [0; 8];
    lo.copy_from_slice(&id[..8]);
    hi.copy_from_slice(&id[8..]);
    amdf_endpoint_id_t {
        words: [u64::from_ne_bytes(lo), u64::from_ne_bytes(hi)],
    }
}

fn native_id(id: amdf_endpoint_id_t) -> [u8; 16] {
    let mut bytes = [0; 16];
    bytes[..8].copy_from_slice(&id.words[0].to_ne_bytes());
    bytes[8..].copy_from_slice(&id.words[1].to_ne_bytes());
    bytes
}

fn name(endpoint: &topology::Endpoint) -> [std::ffi::c_char; 128] {
    endpoint
        .name
        .map(|byte| std::ffi::c_char::from_ne_bytes([byte]))
}

fn summary(endpoint: &topology::Endpoint) -> amdf_endpoint_summary_t {
    let engine_kind = match endpoint.kind {
        topology::EndpointKind::Gpu { .. } => AMDF_ENGINE_KIND_GPU,
        topology::EndpointKind::Npu => AMDF_ENGINE_KIND_XDNA,
        _ => AMDF_ENGINE_KIND_UNKNOWN,
    };
    amdf_endpoint_summary_t {
        id: endpoint_id(endpoint.id),
        engine_kind,
        type_flags: if platform::render_supported(endpoint) {
            AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED
        } else {
            0
        },
        name: name(endpoint),
    }
}

pub(crate) fn family_count(endpoint: &topology::Endpoint) -> u32 {
    endpoint
        .gpu()
        .map_or(0, |gpu| queue_families(gpu).map(|_| 1_u32).sum())
}

fn pm4_atomic_capabilities() -> amdf_atomic_capabilities_t {
    let operations = AMDF_ATOMIC_OPERATION_WAIT
        | AMDF_ATOMIC_OPERATION_STORE
        | AMDF_ATOMIC_OPERATION_ADD
        | AMDF_ATOMIC_OPERATION_SUBTRACT
        | AMDF_ATOMIC_OPERATION_AND
        | AMDF_ATOMIC_OPERATION_OR
        | AMDF_ATOMIC_OPERATION_XOR;
    let waits = AMDF_ATOMIC_WAIT_CONDITION_EQUAL
        | AMDF_ATOMIC_WAIT_CONDITION_NOT_EQUAL
        | AMDF_ATOMIC_WAIT_CONDITION_UNSIGNED_GREATER_EQUAL;
    amdf_atomic_capabilities_t {
        operations_32: operations,
        operations_64: operations,
        wait_conditions_32: waits,
        wait_conditions_64: waits,
        operations_without_dispatch_32: operations,
        operations_without_dispatch_64: operations,
    }
}

pub(crate) fn sdma_format_features(
    target_profile: topology::GpuTargetProfile,
    supports_gcr: bool,
) -> amdf_queue_format_features_t {
    let mut features = if supports_gcr {
        AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR
    } else {
        0
    };
    // GFX12 SDMA uses the explicit system bit in FENCE packets. GFX12.5
    // additionally exposes COPY_LINEAR and FENCE memory-scope fields. These
    // are packet-layout facts, independent of the cache operations qualified
    // for a family.
    if target_profile.sdma_fence_system() {
        features |= AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM;
    }
    if target_profile.sdma_memory_scope() {
        features |= AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE;
    }
    features
}

#[allow(
    clippy::too_many_lines,
    reason = "the queue-family record is clearest when each advertised capability is populated together"
)]
pub(crate) fn family(
    endpoint: &topology::Endpoint,
    ordinal: u32,
) -> Result<amdf_queue_family_info_t, u64> {
    gpu_family(endpoint.gpu().ok_or(UNSUPPORTED)?, ordinal)
}

fn queue_families(gpu: &topology::GpuInfo) -> impl Iterator<Item = (u32, bool, bool)> {
    let queues = gpu.queues;
    [
        (AMDF_QUEUE_COMMAND_TYPE_GPU_AQL, queues.aql, false),
        (
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            queues.sdma,
            queues.kernel_sdma,
        ),
        (
            AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
            queues.pm4,
            queues.kernel_pm4,
        ),
    ]
    .into_iter()
    .filter(|(_, direct, kernel)| *direct || *kernel)
}

#[allow(
    clippy::too_many_lines,
    reason = "one complete queue-family ABI projection"
)]
fn gpu_family(gpu: &topology::GpuInfo, ordinal: u32) -> Result<amdf_queue_family_info_t, u64> {
    let queues = gpu.queues;
    let (command_type, direct, kernel) = queue_families(gpu).nth(ordinal as usize).ok_or(RANGE)?;
    let aql = command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
    let sdma = command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA;
    let pm4 = command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
    let cache_control = if aql {
        queues.aql_system_cache_control
    } else if sdma {
        queues.sdma_system_cache_control
    } else {
        queues.pm4_system_cache_control
    };
    Ok(amdf_queue_family_info_t {
        ordinal,
        command_type,
        publication_modes: if direct {
            AMDF_QUEUE_PUBLICATION_MODE_USER
        } else {
            0
        } | if kernel {
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL
        } else {
            0
        },
        format_version: if pm4 {
            AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1
        } else if sdma {
            AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1
        } else {
            1
        },
        format_features: if aql {
            0
        } else if pm4 {
            AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR
                | if gpu.target_profile.supports_pm4_dispatch_scratch()
                    && gpu.xcc_count == 1
                    && gpu.shader_engine_count_per_xcc != 0
                    && gpu.maximum_scratch_wave_count_per_compute_unit != 0
                {
                    AMDF_GPU_PM4_FORMAT_FEATURE_DISPATCH_SCRATCH
                } else {
                    0
                }
        } else {
            sdma_format_features(gpu.target_profile, queues.sdma_system_cache_control)
        },
        roles: if aql {
            AMDF_QUEUE_ROLE_COMPUTE
        } else if pm4 {
            AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_ATOMIC
        } else {
            AMDF_QUEUE_ROLE_TRANSFER
        } | if cache_control {
            AMDF_QUEUE_ROLE_CACHE_CONTROL
        } else {
            0
        },
        cache_operations: if cache_control {
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM | AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM
        } else {
            0
        },
        cache_transition_kinds: if cache_control {
            AMDF_CACHE_TRANSITION_KINDS_GLOBAL
        } else {
            0
        },
        atomic_capabilities: if pm4 {
            pm4_atomic_capabilities()
        } else {
            amdf_atomic_capabilities_t::default()
        },
        user_queue_capabilities: if direct {
            AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER
                | if pm4 {
                    0
                } else {
                    AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER
                }
        } else {
            0
        },
        producer_modes: if direct {
            AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE
                | if aql {
                    AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI
                } else {
                    0
                }
        } else {
            0
        },
        priority_capabilities: AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL
            | if aql {
                AMDF_QUEUE_PRIORITY_CAPABILITY_LOW | AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH
            } else {
                0
            },
        minimum_ring_byte_length: if !direct {
            0
        } else if pm4 {
            4096
        } else {
            1024
        },
        maximum_ring_byte_length: if !direct {
            0
        } else if pm4 {
            4096
        } else {
            1 << 31
        },
        ring_byte_length_alignment: if !direct {
            0
        } else if pm4 {
            4096
        } else {
            1024
        },
        ..Default::default()
    })
}

pub(crate) fn features(endpoint: &topology::Endpoint, host_registration: bool) -> u64 {
    let mut features = AMDF_GPU_DEVICE_FEATURE_DEVICE_RECREATION;
    if host_registration {
        features |= AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION;
    }
    if endpoint.local_memory_bytes > 0 {
        features |= AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
    }
    if endpoint.host_visible_local_memory_bytes > 0 && endpoint.host_local_cacheability.is_some() {
        features |= AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY;
    }
    features
}

pub(crate) unsafe fn device_instance(device: &Device) -> &Instance {
    unsafe { &*(*device.endpoint).instance }
}

pub(crate) fn supported_endpoint(endpoint: &Endpoint) -> Result<(), u64> {
    if endpoint.native.gpu().is_none_or(|gpu| gpu.gfx_major == 0) {
        return Err(UNSUPPORTED);
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn create(
    info: *const amdf_instance_create_info_t,
    out: *mut *mut amdf_instance_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        output_pointer(out)?;
        let info = input(info, AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO)?;
        if info.reserved != 0 || info.native_lifetime > AMDF_NATIVE_LIFETIME_INSTANCE {
            return Err(INVALID);
        }
        if info.host_allocator.allocate.is_none() && info.host_allocator.resize.is_some() {
            return Err(INVALID);
        }
        let host_allocator = Allocator::from_callbacks_with_resize(
            info.host_allocator.user_data,
            info.host_allocator.allocate,
            info.host_allocator.resize,
            info.host_allocator.free,
        )
        .map_err(|_| INVALID)?;
        let slot = host_allocator
            .with_scope(|| Owned::<Instance>::try_new_uninit().map_err(|_| EXHAUSTED))?;
        // The C ABI calls this native_lifetime; rocddi names the policy after
        // the driver context and its retained bindings.
        let lifetime = if info.native_lifetime == AMDF_NATIVE_LIFETIME_PROCESS {
            session::DriverContextLifetime::Process
        } else {
            session::DriverContextLifetime::Session
        };
        let native = session::Session::new(lifetime).map_err(|e| crate::support::native(&e))?;
        let event_service =
            SignalEventService::new(&native).map_err(|error| crate::support::native(&error))?;
        let host_page_size =
            native_memory::host::host_page_size().map_err(|e| crate::support::native(&e))?;
        let host_write_back_cache = CacheMaintenance::qualify(CachePolicy::WriteBack).ok();
        let owner = slot.write(Instance {
            native,
            host_allocator,
            host_page_size,
            host_write_back_cache,
            children: AtomicU64::new(0),
            closing: false,
            ids: AtomicU64::new(1),
            host_backing_head: Mutex::new(0),
            event_service: Mutex::new(event_service),
            event_hubs: Mutex::new(OwnerVec::with_allocator(host_allocator)),
            scope: Scope {
                instance: std::ptr::null_mut(),
                endpoint: std::ptr::null_mut(),
            },
        });
        let pointer = owner.into_raw();
        (*pointer).scope.instance = pointer;
        out.write(pointer.cast());
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn destroy(pointer: *mut amdf_instance_t) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            // Destruction is externally serialized. Keep this owner alive
            // until native shutdown releases every resource it retains.
            let instance = object(pointer.cast::<Instance>())?;
            instance.require_current_process()?;
            if instance.children.load(Ordering::Acquire) != 0 {
                return Err(BUSY);
            }
            if *instance.host_backing_head.lock().map_err(|_| INTERNAL)? != 0 {
                return Err(BUSY);
            }
            // No child can now retain a shared borrow, and the caller
            // serializes direct use of the instance during destruction.
            let instance = exclusive(pointer.cast::<Instance>())?;
            instance
                .event_service
                .lock()
                .map_err(|_| INTERNAL)?
                .preflight_session_destroy(&mut instance.native)
                .map_err(|error| native(&error))?;
            // Native cleanup can make partial progress. Retain this handle on
            // failure for destruction only; no new child may borrow it then.
            instance.closing = true;
            instance
                .event_service
                .lock()
                .map_err(|_| INTERNAL)?
                .release()
                .map_err(|error| native(&error))?;
            instance.native.destroy().map_err(|e| native(&e))?;
            drop(Owned::from_raw(pointer.cast::<Instance>()));
            Ok(())
        }
    })
}

pub(crate) unsafe extern "C" fn enumerate(
    pointer: *mut amdf_instance_t,
    capacity: u32,
    summaries: *mut amdf_endpoint_summary_t,
    count: *mut u32,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let instance = object(pointer.cast::<Instance>())?;
        instance.require_current_process()?;
        if instance.closing {
            return Err(PRECONDITION);
        }
        output_pointer(count)?;
        output_array(summaries, capacity)?;
        let mut staged = OwnerVec::with_allocator(instance.host_allocator);
        staged
            .try_reserve_exact(capacity as usize)
            .map_err(|_| EXHAUSTED)?;
        let mut total = 0u32;
        instance
            .native
            .enumerate(&mut |endpoint| {
                if staged.len() < capacity as usize {
                    staged.try_push(summary(&endpoint))?;
                }
                total = total.checked_add(1).ok_or(rocddi::Error::Capacity {
                    resource: "endpoint count",
                })?;
                Ok(())
            })
            .map_err(|e| native(&e))?;
        if !staged.is_empty() {
            std::ptr::copy_nonoverlapping(staged.as_slice().as_ptr(), summaries, staged.len());
        }
        count.write(total);
        if capacity != 0 && capacity < total {
            Err(SMALL)
        } else {
            Ok(())
        }
    })
}

pub(crate) unsafe extern "C" fn open(
    pointer: *mut amdf_instance_t,
    id: *const amdf_endpoint_id_t,
    out: *mut *mut amdf_endpoint_t,
) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            let instance = object(pointer.cast::<Instance>())?;
            instance.require_current_process()?;
            if instance.closing {
                return Err(PRECONDITION);
            }
            output_pointer(out)?;
            let id = *object(id)?;
            let slot = instance.owner_slot::<Endpoint>()?;
            let native = instance
                .native
                .open_endpoint(native_id(id))
                .map_err(|error| {
                    // Opening a passive identity cannot lose an activated device. A
                    // stale or vanished identity is simply absent from discovery.
                    if matches!(
                        error.kind(),
                        rocddi::ErrorKind::InvalidArgument
                            | rocddi::ErrorKind::DeviceLost
                            | rocddi::ErrorKind::NotFound
                    ) {
                        NOT_FOUND
                    } else {
                        crate::support::native(&error)
                    }
                })?;
            register(&instance.children)?;
            let host_numa_node = platform::host_numa_node(&native);
            let owner = slot.write(Endpoint {
                instance: pointer.cast(),
                native,
                host_numa_node,
                children: AtomicU64::new(0),
                scope: Scope {
                    instance: pointer.cast(),
                    endpoint: std::ptr::null_mut(),
                },
            });
            let endpoint = owner.into_raw();
            (*endpoint).scope.endpoint = endpoint;
            out.write(endpoint.cast());
            Ok(())
        }
    })
}

pub(crate) unsafe extern "C" fn close(pointer: *mut amdf_endpoint_t) -> u64 {
    crate::support::boundary(|| unsafe {
        let endpoint = object(pointer.cast::<Endpoint>())?;
        (*endpoint.instance).require_current_process()?;
        if endpoint.children.load(Ordering::Acquire) != 0 {
            return Err(BUSY);
        }
        unregister(&(*endpoint.instance).children);
        drop(Owned::from_raw(pointer.cast::<Endpoint>()));
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn endpoint_info(
    pointer: *mut amdf_endpoint_t,
    out: *mut amdf_endpoint_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let endpoint = object(pointer.cast::<Endpoint>())?;
        let out = output_with_next(out, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO)?;
        let locality = out.next().cast::<amdf_endpoint_host_locality_t>();
        let locality = if locality.is_null() {
            None
        } else {
            Some(output(
                locality,
                AMDF_STRUCTURE_TYPE_ENDPOINT_HOST_LOCALITY,
            )?)
        };
        let p = endpoint.native.pci.unwrap_or_default();
        let summary = summary(&endpoint.native);
        out.publish(amdf_endpoint_info_t {
            id: endpoint_id(endpoint.native.id),
            engine_kind: summary.engine_kind,
            type_flags: summary.type_flags,
            pci: amdf_pci_info_t {
                vendor_id: p.vendor_id,
                device_id: p.device_id,
                subsystem_vendor_id: p.subsystem_vendor_id,
                subsystem_device_id: p.subsystem_device_id,
                revision_id: p.revision_id,
            },
            name: name(&endpoint.native),
            queue_family_count: family_count(&endpoint.native),
            native_identity: platform::native_identity(&endpoint.native),
            ..Default::default()
        });
        if let Some(locality) = locality {
            locality.publish(amdf_endpoint_host_locality_t {
                numa_node: endpoint.host_numa_node,
                ..Default::default()
            });
        }
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn queue_family_info(
    pointer: *mut amdf_endpoint_t,
    ordinal: u32,
    out: *mut amdf_queue_family_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let endpoint = object(pointer.cast::<Endpoint>())?;
        let out = output(out, AMDF_STRUCTURE_TYPE_QUEUE_FAMILY_INFO)?;
        out.publish(family(&endpoint.native, ordinal)?);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn gpu_endpoint_info(
    pointer: *mut amdf_endpoint_t,
    out: *mut amdf_gpu_endpoint_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let endpoint = object(pointer.cast::<Endpoint>())?;
        let out = output(out, AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO)?;
        let g = endpoint.native.gpu().copied().ok_or(UNSUPPORTED)?;
        if g.gfx_major == 0 {
            return Err(UNSUPPORTED);
        }
        out.publish(amdf_gpu_endpoint_info_t {
            gfx_ip: amdf_gpu_endpoint_info_t__gfx_ip {
                major: g.gfx_major,
                minor: g.gfx_minor,
                stepping: g.gfx_stepping,
            },
            asic_revision: g.asic_revision,
            compute: amdf_gpu_endpoint_info_t__compute {
                wavefront_size: g.wavefront_size,
                compute_unit_count: g.compute_unit_count,
                maximum_wave_count_per_compute_unit: g.maximum_wave_count_per_compute_unit,
                maximum_scratch_wave_count_per_compute_unit: g
                    .maximum_scratch_wave_count_per_compute_unit,
                local_data_share_byte_length: g.local_data_share_byte_length,
            },
            topology: amdf_gpu_endpoint_info_t__topology {
                xcc_count: g.xcc_count,
                shader_engine_count_per_xcc: g.shader_engine_count_per_xcc,
            },
            ..Default::default()
        });
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn device_create(
    pointer: *mut amdf_endpoint_t,
    info: *const amdf_gpu_device_create_info_t,
    out: *mut *mut amdf_device_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let endpoint = object(pointer.cast::<Endpoint>())?;
        endpoint.require_current_process()?;
        output_pointer(out)?;
        let info = input(info, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO)?;
        if info.reserved != 0 {
            return Err(INVALID);
        }
        supported_endpoint(endpoint)?;
        let instance = &*endpoint.instance;
        let slot = instance.owner_slot::<Device>()?;
        let reset_epoch = instance.shared(AtomicU64::new(INITIAL_RESET_EPOCH))?;
        let id = next_id(&instance.ids)?;
        register(&endpoint.children)?;
        let result = instance.native.activate(&endpoint.native);
        let native = match result {
            Ok(device) => device,
            Err(error) => {
                unregister(&endpoint.children);
                return Err(crate::support::native(&error));
            }
        };
        // Feature queries are metadata-only. Sample the process signal-page
        // state here so they never contend on its native-resource mutex.
        let native_events_available = if instance.process_lifetime() {
            signal_event_available(&native).map_err(|error| {
                unregister(&endpoint.children);
                crate::support::native(&error)
            })?
        } else {
            false
        };
        let event_observer = match register_event_observer(
            instance,
            &native,
            endpoint.native.id,
            reset_epoch.clone(),
        ) {
            Ok(observer) => observer,
            Err(status) => {
                unregister(&endpoint.children);
                return Err(status);
            }
        };
        let owner = slot.write(Device {
            endpoint: pointer.cast(),
            native,
            id: amdf_device_id_t {
                words: [endpoint.instance as u64, id],
            },
            queues: AtomicU64::new(0),
            events: AtomicU64::new(0),
            native_events_available,
            reset_epoch,
            event_observer,
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn device_info(
    pointer: *mut amdf_device_t,
    out: *mut amdf_gpu_device_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let device = object(pointer.cast::<Device>())?;
        let out = output(out, AMDF_STRUCTURE_TYPE_GPU_DEVICE_INFO)?;
        out.publish(amdf_gpu_device_info_t {
            id: device.id,
            reset_epoch: device.current_reset_epoch(),
            features: features(
                device.native.endpoint(),
                device_instance(device)
                    .native
                    .supports_host_registration(device.native.endpoint()),
            ) | if device.native_events_available {
                AMDF_GPU_DEVICE_FEATURE_NATIVE_EVENTS
            } else {
                0
            },
            ..Default::default()
        });
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn device_destroy(pointer: *mut amdf_device_t) -> u64 {
    crate::support::boundary(|| unsafe {
        let device = object(pointer.cast::<Device>())?;
        device_instance(device).require_current_process()?;
        if device.queues.load(Ordering::Acquire) != 0 || device.events.load(Ordering::Acquire) != 0
        {
            return Err(BUSY);
        }
        unregister_event_observer(device)?;
        unregister(&(*device.endpoint).children);
        drop(Owned::from_raw(pointer.cast::<Device>()));
        Ok(())
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod observer_tests {
    use super::*;
    use rocddi::device::event::{
        DeviceEventOriginId, GpuHardwareException, GpuResetCause, GpuResetScope,
    };

    fn observer(endpoint_id: [u8; 16]) -> EventObserver {
        EventObserver {
            endpoint_id,
            origin_id: None,
            state: Mutex::new(ObserverState::new(Allocator::system())),
            polling_failed: Shared::new(AtomicBool::new(false)).unwrap(),
        }
    }

    fn hub(endpoint_id: [u8; 16], observer: Shared<EventObserver>) -> EventHub {
        let polling_failed = observer.polling_failed.clone();
        let mut observers = OwnerVec::with_allocator(Allocator::system());
        observers
            .try_push(ObserverRecord {
                endpoint_id,
                observer,
            })
            .unwrap();
        EventHub {
            state: Mutex::new(EventHubState {
                observers,
                latest_hardware_events: OwnerVec::with_allocator(Allocator::system()),
                latest_memory_loss: None,
            }),
            stop: AtomicBool::new(false),
            polling_failed,
        }
    }

    fn reset(endpoint_id: [u8; 16], memory_lost: bool) -> DeviceEvent {
        DeviceEvent::GpuHardwareException(GpuHardwareException {
            endpoint_id: Some(endpoint_id),
            origin_id: None,
            scope: GpuResetScope::WholeGpu,
            memory_lost,
            cause: GpuResetCause::Hang,
        })
    }

    #[test]
    fn driver_hub_fans_out_and_replays_the_latest_reset() {
        let first_id = [7; 16];
        let second_id = [8; 16];
        let first_observer = Shared::new(observer(first_id)).unwrap();
        let hub = hub(first_id, first_observer.clone());
        let first_epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        first_observer.register_epoch(first_epoch.clone()).unwrap();

        hub.record_event(reset(first_id, false)).unwrap();
        assert_eq!(first_epoch.load(Ordering::Acquire), 2);
        let second_epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        let second_observer = hub
            .register_endpoint(
                second_id,
                None,
                second_epoch.clone(),
                Allocator::system(),
                |observer| Shared::new(observer).map_err(|_| EXHAUSTED),
            )
            .unwrap();
        assert_eq!(second_epoch.load(Ordering::Acquire), INITIAL_RESET_EPOCH);

        hub.record_event(reset(second_id, false)).unwrap();
        assert_eq!(first_epoch.load(Ordering::Acquire), 2);
        assert_eq!(second_epoch.load(Ordering::Acquire), 2);
        second_observer.record_operation_loss(&second_epoch, Some(2));
        hub.record_event(reset(second_id, false)).unwrap();
        assert_eq!(second_epoch.load(Ordering::Acquire), 3);

        hub.record_event(reset(first_id, true)).unwrap();
        assert_eq!(first_epoch.load(Ordering::Acquire), 3);
        assert_eq!(second_epoch.load(Ordering::Acquire), 4);
        let late_epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        hub.register_endpoint(
            [9; 16],
            None,
            late_epoch.clone(),
            Allocator::system(),
            |observer| Shared::new(observer).map_err(|_| EXHAUSTED),
        )
        .unwrap();
        assert_eq!(late_epoch.load(Ordering::Acquire), 2);
    }

    #[test]
    fn driver_hub_replays_resets_observed_before_endpoint_activation() {
        let first_id = [7; 16];
        let later_id = [8; 16];
        let origin = DeviceEventOriginId::from_driver_key(77);
        let other_origin = DeviceEventOriginId::from_driver_key(78);
        let first_observer = Shared::new(observer(first_id)).unwrap();
        let hub = hub(first_id, first_observer.clone());
        let first_epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        first_observer.register_epoch(first_epoch.clone()).unwrap();
        let first_reset = GpuHardwareException {
            endpoint_id: None,
            origin_id: Some(origin),
            scope: GpuResetScope::Other,
            memory_lost: false,
            cause: GpuResetCause::Ecc,
        };
        hub.record_event(DeviceEvent::GpuHardwareException(first_reset))
            .unwrap();
        hub.record_event(DeviceEvent::GpuHardwareException(GpuHardwareException {
            origin_id: Some(other_origin),
            ..first_reset
        }))
        .unwrap();
        assert_eq!(first_epoch.load(Ordering::Acquire), INITIAL_RESET_EPOCH);

        let later_epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        hub.register_endpoint(
            later_id,
            Some(origin),
            later_epoch.clone(),
            Allocator::system(),
            |observer| Shared::new(observer).map_err(|_| EXHAUSTED),
        )
        .unwrap();
        assert_eq!(later_epoch.load(Ordering::Acquire), 2);

        let other_epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        hub.register_endpoint(
            [9; 16],
            Some(other_origin),
            other_epoch.clone(),
            Allocator::system(),
            |observer| Shared::new(observer).map_err(|_| EXHAUSTED),
        )
        .unwrap();
        assert_eq!(other_epoch.load(Ordering::Acquire), 2);

        let unrelated_epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        hub.register_endpoint(
            [10; 16],
            Some(DeviceEventOriginId::from_driver_key(79)),
            unrelated_epoch.clone(),
            Allocator::system(),
            |observer| Shared::new(observer).map_err(|_| EXHAUSTED),
        )
        .unwrap();
        assert_eq!(unrelated_epoch.load(Ordering::Acquire), INITIAL_RESET_EPOCH);
    }

    #[test]
    fn shared_observer_fans_out_each_reset_and_initializes_late_handles() {
        let endpoint_id = [7; 16];
        let observer = observer(endpoint_id);
        let first = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        let second = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        observer.register_epoch(first.clone()).unwrap();
        observer.register_epoch(second.clone()).unwrap();
        let reset = reset(endpoint_id, false);

        observer.record_event(reset).unwrap();
        assert_eq!(first.load(Ordering::Acquire), 2);
        assert_eq!(second.load(Ordering::Acquire), 2);
        assert!(!observer.unregister_epoch(&first).unwrap());
        observer.record_event(reset).unwrap();
        assert_eq!(first.load(Ordering::Acquire), 2);
        assert_eq!(second.load(Ordering::Acquire), 3);

        // The operation error and its later event describe one reset.
        observer.record_operation_loss(&second, Some(3));
        observer.record_event(reset).unwrap();
        assert_eq!(second.load(Ordering::Acquire), 4);

        let late = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        observer.register_epoch(late.clone()).unwrap();
        assert_eq!(late.load(Ordering::Acquire), 4);
        observer.record_event(reset).unwrap();
        assert_eq!(second.load(Ordering::Acquire), 5);
        assert_eq!(late.load(Ordering::Acquire), 5);
    }

    #[test]
    fn terminal_operation_loss_advances_after_prior_reset_and_deduplicates_its_event() {
        let endpoint_id = [7; 16];
        let observer = observer(endpoint_id);
        let epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        let peer = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        observer.register_epoch(epoch.clone()).unwrap();
        observer.register_epoch(peer.clone()).unwrap();
        observer.record_event(reset(endpoint_id, false)).unwrap();
        observer.record_event(reset(endpoint_id, false)).unwrap();
        assert_eq!(epoch.load(Ordering::Acquire), 3);

        observer.record_operation_loss(&epoch, None);
        observer.record_operation_loss(&epoch, None);
        assert_eq!(epoch.load(Ordering::Acquire), 4);
        assert_eq!(peer.load(Ordering::Acquire), 4);
        let late = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        observer.register_epoch(late.clone()).unwrap();
        assert_eq!(late.load(Ordering::Acquire), 4);
        observer.record_event(reset(endpoint_id, true)).unwrap();
        assert_eq!(epoch.load(Ordering::Acquire), 4);
        assert_eq!(late.load(Ordering::Acquire), 4);
    }

    #[test]
    fn terminal_event_before_operation_does_not_count_loss_twice() {
        let endpoint_id = [7; 16];
        let observer = observer(endpoint_id);
        let epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        observer.register_epoch(epoch.clone()).unwrap();
        observer.record_event(reset(endpoint_id, true)).unwrap();
        observer.record_operation_loss(&epoch, None);
        assert_eq!(epoch.load(Ordering::Acquire), 2);
    }

    #[test]
    fn poll_failure_does_not_invent_reset_or_hide_later_native_loss() {
        let observer = Shared::new(observer([7; 16])).unwrap();
        let hub = hub([7; 16], observer.clone());
        let first = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        observer.register_epoch(first.clone()).unwrap();
        hub.record_polling_failure();
        hub.record_polling_failure();
        assert_eq!(first.load(Ordering::Acquire), INITIAL_RESET_EPOCH);
        let late = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        assert_eq!(observer.register_epoch(late).err(), Some(INTERNAL));
        observer.record_operation_loss(&first, None);
        assert_eq!(first.load(Ordering::Acquire), 2);
    }

    #[test]
    fn repeated_driver_errors_stop_the_observer_after_bounded_retries() {
        let observer = Shared::new(observer([7; 16])).unwrap();
        let hub = hub([7; 16], observer.clone());
        let mut calls = 0;
        let mut pauses = Vec::new();
        device_event_worker_with(
            |_| {
                calls += 1;
                Err(rocddi::Error::Operation {
                    kind: rocddi::ErrorKind::Driver,
                    detail: "injected event poll failure",
                })
            },
            |delay| pauses.push(delay),
            &hub,
        );
        assert_eq!(calls, 17);
        assert_eq!(pauses.len(), 16);
        assert!(observer.polling_failed.load(Ordering::Acquire));
        assert!(!hub.stop.load(Ordering::Acquire));
    }

    #[test]
    fn queue_loss_and_later_terminal_report_share_one_generation() {
        let endpoint_id = [7; 16];
        let observer = observer(endpoint_id);
        let epoch = Shared::new(AtomicU64::new(INITIAL_RESET_EPOCH)).unwrap();
        observer.register_epoch(epoch.clone()).unwrap();

        observer.record_operation_loss(&epoch, Some(1));
        observer.record_operation_loss(&epoch, None);
        assert_eq!(epoch.load(Ordering::Acquire), 2);
        observer.record_event(reset(endpoint_id, true)).unwrap();
        observer.record_operation_loss(&epoch, None);
        assert_eq!(epoch.load(Ordering::Acquire), 2);
        // The terminal latch remains set even if KFD later reports another
        // hardware notification without a lost-memory bit.
        observer.record_event(reset(endpoint_id, false)).unwrap();
        observer.record_operation_loss(&epoch, None);
        assert_eq!(epoch.load(Ordering::Acquire), 3);
    }
}

#[cfg(test)]
mod family_tests {
    use super::*;

    #[test]
    #[allow(clippy::unwrap_used)]
    fn kernel_only_families_have_dense_ordinals_without_direct_mapping_promises() {
        let mut gpu = topology::GpuInfo::default();
        gpu.queues.kernel_sdma = true;
        gpu.queues.kernel_pm4 = true;
        for (ordinal, command) in [
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA,
            AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
        ]
        .into_iter()
        .enumerate()
        {
            let family = gpu_family(&gpu, u32::try_from(ordinal).unwrap()).unwrap();
            assert_eq!(family.ordinal, u32::try_from(ordinal).unwrap());
            assert_eq!(family.command_type, command);
            assert_eq!(family.publication_modes, AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
            assert_eq!(family.user_queue_capabilities, 0);
            assert_eq!(family.producer_modes, 0);
            assert_eq!(family.minimum_ring_byte_length, 0);
            assert_eq!(family.maximum_ring_byte_length, 0);
            assert_eq!(family.ring_byte_length_alignment, 0);
        }
        assert_eq!(gpu_family(&gpu, 2).err(), Some(RANGE));
        gpu.queues.sdma = true;
        let family = gpu_family(&gpu, 0).unwrap();
        assert_eq!(
            family.publication_modes,
            AMDF_QUEUE_PUBLICATION_MODE_USER | AMDF_QUEUE_PUBLICATION_MODE_KERNEL
        );
        assert_ne!(family.minimum_ring_byte_length, 0);
        gpu.queues.aql = true;
        assert_eq!(
            gpu_family(&gpu, 0).unwrap().command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
        );
        assert_eq!(
            gpu_family(&gpu, 1).unwrap().command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA
        );
    }
}
