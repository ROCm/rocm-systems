//! Direct native queues with cached mappings and explicit producer borrows.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rocddi::gpu::queue::{
    self, QueueAccessWidth, QueueParameters, QueuePriority, QueueProducerMode, QueueRequest,
    QueueScratch,
};
use rocddi::host_storage::{Allocator, Owned};

use crate::generated::amdf::*;
use crate::instance::{self, Device};
use crate::memory;
use crate::support::*;

// Native errors originate in API/errno domains, so the high bit is available
// internally to publish the first terminal status and its loss classification
// together. It is always removed before returning a public AMDF status.
const LOST_BIT: u64 = 1 << 63;
const DEFAULT_RING_BYTES: u64 = 64 * 1024;

/// Native queue implementation or deterministic unit-test fixture.
enum NativeQueue {
    Provider(queue::Queue),
    #[cfg(test)]
    Fixture(std::sync::Arc<tests::Fixture>),
}

impl NativeQueue {
    fn progress(&self) -> Result<(u64, u64), rocddi::Error> {
        match self {
            Self::Provider(queue) => queue.progress(),
            #[cfg(test)]
            Self::Fixture(queue) => queue.progress(),
        }
    }

    fn destroy(&mut self) -> Result<(), rocddi::Error> {
        match self {
            Self::Provider(queue) => queue.destroy(),
            #[cfg(test)]
            Self::Fixture(queue) => queue.destroy(),
        }
    }

    fn map_device(&self, device: &Device) -> Result<queue::QueueTransport, rocddi::Error> {
        match self {
            Self::Provider(queue) => queue.map_device(device.native.gpu()?),
            #[cfg(test)]
            Self::Fixture(_) => Err(rocddi::Error::Operation {
                kind: rocddi::ErrorKind::Unsupported,
                detail: "fixture has no peer queue transport",
            }),
        }
    }
}

/// Public AMDF user queue and every borrow required by its transport.
///
/// Terminal native errors are published once, mappings are counted for BUSY
/// destruction, and teardown remains separate from progress observation so a
/// partial native cleanup cannot restore a revoked public transport.
struct Queue {
    native: NativeQueue,
    allocator: Allocator,
    device: *mut Device,
    // These counters borrow the public device, which cannot be destroyed while
    // this queue is registered. No public device ownership is retained.
    device_queues: *const AtomicU64,
    device_reset_epoch: *const AtomicU64,
    _scratch_borrow: Option<memory::QueueScratchBorrow>,
    info: amdf_user_queue_info_t,
    host_mapping: amdf_user_queue_mapping_info_t,
    device_mapping: Option<amdf_user_queue_mapping_info_t>,
    mappings: AtomicU64,
    destroying: AtomicBool,
    terminal: AtomicU64,
    consumed: AtomicU64,
    producer: AtomicU64,
}

/// Public producer mapping borrowing one queue and its cached transport view.
struct Mapping {
    queue: *mut Queue,
    info: amdf_user_queue_mapping_info_t,
}

impl Queue {
    fn require_usable(&self) -> Result<(), u64> {
        if self.destroying.load(Ordering::Acquire) {
            Err(PRECONDITION)
        } else {
            Ok(())
        }
    }

    fn observe_error(&self, error: &rocddi::Error) -> u64 {
        let status = native(error);
        let lost = error.kind() == rocddi::ErrorKind::DeviceLost;
        let terminal = lost
            || matches!(
                error.kind(),
                rocddi::ErrorKind::DriverContract
                    | rocddi::ErrorKind::InvalidData
                    | rocddi::ErrorKind::Internal
            );
        if terminal {
            let encoded = status | if lost { LOST_BIT } else { 0 };
            let _ = self
                .terminal
                .compare_exchange(0, encoded, Ordering::AcqRel, Ordering::Acquire);
            if lost {
                // SAFETY: The registered queue borrows this still-live device.
                instance::advance_reset_epoch(
                    unsafe { &*self.device_reset_epoch },
                    self.info.reset_epoch,
                );
            }
        }
        status
    }

    fn sample(&self) -> Result<amdf_user_queue_status_t, u64> {
        self.require_usable()?;
        if self.terminal.load(Ordering::Acquire) == 0 {
            let (consumed, producer) = self
                .native
                .progress()
                .map_err(|error| self.observe_error(&error))?;
            // Concurrent readers must not regress the cached frontiers when an
            // older native sample finishes after a newer sample.
            self.consumed.fetch_max(consumed, Ordering::Relaxed);
            self.producer.fetch_max(producer, Ordering::Relaxed);
        }
        let terminal = self.terminal.load(Ordering::Acquire);
        Ok(amdf_user_queue_status_t {
            state: if terminal == 0 {
                AMDF_QUEUE_STATE_ACTIVE
            } else if terminal & LOST_BIT != 0 {
                AMDF_QUEUE_STATE_DEVICE_LOST
            } else {
                AMDF_QUEUE_STATE_FAILED
            },
            reset_epoch: self.info.reset_epoch,
            producer_index: self.producer.load(Ordering::Relaxed),
            consumed_index: self.consumed.load(Ordering::Relaxed),
            terminal_status: terminal & !LOST_BIT,
            ..Default::default()
        })
    }
}

fn descriptor(
    info: &amdf_gpu_user_queue_create_info_t,
    family: &amdf_queue_family_info_t,
    scratch: Option<QueueScratch>,
) -> Result<QueueRequest, u64> {
    if info.reserved != 0 {
        return Err(INVALID);
    }
    if info.required_capabilities & !family.user_queue_capabilities != 0 {
        return Err(UNSUPPORTED);
    }
    let priority = match info.priority {
        AMDF_QUEUE_PRIORITY_LOW => QueuePriority::Low,
        AMDF_QUEUE_PRIORITY_NORMAL => QueuePriority::Normal,
        AMDF_QUEUE_PRIORITY_HIGH => QueuePriority::High,
        _ => return Err(INVALID),
    };
    let producer = match info.producer_mode {
        AMDF_QUEUE_PRODUCER_MODE_SINGLE => QueueProducerMode::Single,
        AMDF_QUEUE_PRODUCER_MODE_MULTI => QueueProducerMode::Multiple,
        _ => return Err(INVALID),
    };
    if family.priority_capabilities & (1 << info.priority) == 0
        || family.producer_modes & (1 << info.producer_mode) == 0
        || family.publication_modes & AMDF_QUEUE_PUBLICATION_MODE_USER == 0
    {
        return Err(UNSUPPORTED);
    }
    let ring = if info.ring_byte_length == 0 {
        if family.minimum_ring_byte_length == family.maximum_ring_byte_length {
            family.minimum_ring_byte_length
        } else {
            DEFAULT_RING_BYTES
        }
    } else {
        info.ring_byte_length
    };
    if !ring.is_power_of_two() {
        return Err(INVALID);
    }
    if ring < family.minimum_ring_byte_length
        || ring > family.maximum_ring_byte_length
        || ring % family.ring_byte_length_alignment != 0
    {
        return Err(RANGE);
    }
    let parameters = match family.command_type {
        AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 => QueueParameters::Pm4,
        AMDF_QUEUE_COMMAND_TYPE_GPU_AQL => QueueParameters::Aql {
            producer_mode: producer,
            inactive_signal: None,
            error_event: None,
            scratch,
        },
        AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA => QueueParameters::Sdma,
        _ => return Err(UNSUPPORTED),
    };
    Ok(QueueRequest {
        ring_size_bytes: ring,
        parameters,
        priority,
        device_producer: info.required_capabilities & AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER
            != 0,
    })
}

unsafe fn scratch_descriptor(
    info: &amdf_gpu_user_queue_create_info_t,
    family: &amdf_queue_family_info_t,
    device: *mut Device,
) -> Result<(Option<QueueScratch>, *mut memory::Memory), u64> {
    let scratch = &info.scratch;
    let disabled = scratch.memory.is_null()
        && scratch.access_ordinal == 0
        && scratch.reserved == 0
        && scratch.byte_offset == 0
        && scratch.byte_length == 0
        && scratch.maximum_private_segment_byte_length == 0
        && scratch.maximum_wave_count == 0;
    if disabled {
        return Ok((None, std::ptr::null_mut()));
    }
    if scratch.memory.is_null()
        || scratch.reserved != 0
        || scratch.byte_length == 0
        || scratch.maximum_private_segment_byte_length == 0
        || scratch.maximum_wave_count == 0
    {
        return Err(INVALID);
    }
    if family.command_type != AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
        || family.roles & AMDF_QUEUE_ROLE_COMPUTE == 0
    {
        return Err(UNSUPPORTED);
    }
    let reset_epoch = unsafe { (*device).current_reset_epoch() };
    let resolved = unsafe {
        memory::queue_scratch(
            scratch.memory,
            scratch.access_ordinal,
            device,
            reset_epoch,
            scratch.byte_offset,
            scratch.byte_length,
        )?
    };
    Ok((
        Some(QueueScratch {
            device_address: resolved.device_address,
            byte_length: scratch.byte_length,
            maximum_private_segment_byte_length: scratch.maximum_private_segment_byte_length,
            maximum_wave_count: scratch.maximum_wave_count,
        }),
        resolved.memory,
    ))
}

fn transport(
    info: queue::QueueTransport,
    command_type: u32,
    format_version: u32,
    format_features: amdf_queue_format_features_t,
    device: bool,
) -> Result<amdf_user_queue_mapping_info_t, u64> {
    let units = match command_type {
        AMDF_QUEUE_COMMAND_TYPE_GPU_PM4 => 4,
        AMDF_QUEUE_COMMAND_TYPE_GPU_AQL => 64,
        AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA => 1,
        _ => return Err(UNSUPPORTED),
    };
    let (ring, read, write, doorbell) = if device {
        (
            info.ring_device_address,
            info.read_index_device_address,
            info.write_index_device_address,
            info.doorbell_device_address.ok_or(UNSUPPORTED)?,
        )
    } else {
        (
            info.ring_host_address as u64,
            info.read_index_host_address as u64,
            info.write_index_host_address as u64,
            info.doorbell_host_address as u64,
        )
    };
    if info.read_index_width != QueueAccessWidth::Bits64
        || info.write_index_width != QueueAccessWidth::Bits64
        || info.doorbell_width != QueueAccessWidth::Bits64
        || info.read_index_wraps != (command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4)
        || info.index_unit_bytes != units
        || ring == 0
        || read == 0
        || write == 0
        || doorbell == 0
        || read & 7 != 0
        || write & 7 != 0
        || doorbell & 7 != 0
    {
        return Err(INTERNAL);
    }
    Ok(amdf_user_queue_mapping_info_t {
        command_type,
        format_version,
        format_features,
        ring_address: ring,
        ring_byte_length: info.ring_size_bytes,
        read_index_address: read,
        write_index_address: write,
        doorbell_address: doorbell,
        index_bits: 64,
        doorbell_bits: 64,
        ..Default::default()
    })
}

fn creation_transports(
    info: queue::QueueTransport,
    family: &amdf_queue_family_info_t,
) -> Result<
    (
        amdf_user_queue_mapping_info_t,
        Option<amdf_user_queue_mapping_info_t>,
    ),
    u64,
> {
    let host = transport(
        info,
        family.command_type,
        family.format_version,
        family.format_features,
        false,
    )?;
    let device = info
        .doorbell_device_address
        .map(|_| {
            transport(
                info,
                family.command_type,
                family.format_version,
                family.format_features,
                true,
            )
        })
        .transpose()?;
    Ok((host, device))
}

fn rollback_creation(queue: &mut queue::Queue, device: &Device, status: u64) -> u64 {
    let status = queue.destroy().err().map_or(status, |error| native(&error));
    unregister(&device.queues);
    status
}

fn wait_consumed(
    queue: &Queue,
    target: u64,
    timeout: u64,
    poll: u64,
    start: Instant,
) -> Result<(), u64> {
    let timeout = (timeout != AMDF_TIMEOUT_INFINITE).then(|| Duration::from_nanos(timeout));
    let poll = timeout.map_or(Duration::from_nanos(poll), |limit| {
        Duration::from_nanos(poll).min(limit)
    });
    loop {
        let sample = queue.sample()?;
        if sample.terminal_status != 0 {
            return Err(sample.terminal_status);
        }
        if target > sample.producer_index {
            return Err(INVALID);
        }
        if sample.consumed_index >= target {
            return Ok(());
        }
        let elapsed = start.elapsed();
        if timeout.is_some_and(|limit| elapsed >= limit) {
            return Err(DEADLINE);
        }
        if elapsed < poll {
            std::hint::spin_loop();
        } else {
            let interval = Duration::from_micros(50);
            let interval = timeout.map_or(interval, |limit| {
                interval.min(limit.saturating_sub(elapsed))
            });
            std::thread::sleep(interval);
        }
    }
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn create(
    pointer: *mut amdf_device_t,
    create_info: *const amdf_gpu_user_queue_create_info_t,
    out: *mut *mut amdf_user_queue_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        output_pointer(out)?;
        let create_info = input(create_info, AMDF_STRUCTURE_TYPE_GPU_USER_QUEUE_CREATE_INFO)?;
        let device = object(pointer.cast::<Device>())?;
        let family = instance::family(device.native.endpoint(), create_info.queue_family_ordinal)?;
        let (scratch, scratch_memory) = scratch_descriptor(&create_info, &family, pointer.cast())?;
        let desc = descriptor(&create_info, &family, scratch)?;
        let instance = instance::device_instance(device);
        let slot = Owned::<Queue>::try_new_uninit(instance.allocator).map_err(|_| EXHAUSTED)?;
        let id = next_id(&instance.ids)?;
        register(&device.queues)?;
        let scratch_borrow = if scratch_memory.is_null() {
            None
        } else {
            match memory::borrow_for_queue(scratch_memory) {
                Ok(borrow) => Some(borrow),
                Err(status) => {
                    unregister(&device.queues);
                    return Err(status);
                }
            }
        };
        let Ok(gpu) = device.native.gpu() else {
            unregister(&device.queues);
            return Err(UNSUPPORTED);
        };
        let mut native_queue = match gpu.create_queue(desc) {
            Ok(queue) => queue,
            Err(error) => {
                unregister(&device.queues);
                return Err(native(&error));
            }
        };
        let native_info = native_queue.info();
        let capabilities = AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER
            | if native_info.doorbell_device_address.is_some() {
                AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER
            } else {
                0
            };
        let (mut host_mapping, mut device_mapping) = match creation_transports(native_info, &family)
        {
            Ok(mappings) => mappings,
            Err(status) => {
                return Err(rollback_creation(&mut native_queue, device, status));
            }
        };
        if create_info.required_capabilities & !capabilities != 0 {
            return Err(rollback_creation(&mut native_queue, device, UNSUPPORTED));
        }
        let epoch = device.current_reset_epoch();
        let id = amdf_queue_id_t {
            words: [device.id.words[0], id],
        };
        host_mapping.queue_id = id;
        host_mapping.queue_reset_epoch = epoch;
        if let Some(mapping) = &mut device_mapping {
            mapping.producer_device_id = device.id;
            mapping.queue_id = id;
            mapping.queue_reset_epoch = epoch;
            mapping.producer_reset_epoch = epoch;
        }
        let queue_info = amdf_user_queue_info_t {
            device_id: device.id,
            queue_id: id,
            reset_epoch: epoch,
            queue_family_ordinal: create_info.queue_family_ordinal,
            command_type: family.command_type,
            format_version: family.format_version,
            format_features: family.format_features,
            producer_mode: create_info.producer_mode,
            priority: create_info.priority,
            capabilities,
            roles: family.roles,
            ring_byte_length: host_mapping.ring_byte_length,
            ..Default::default()
        };
        let owner = slot.write(Queue {
            native: NativeQueue::Provider(native_queue),
            allocator: instance.allocator,
            device: pointer.cast(),
            device_queues: &raw const device.queues,
            device_reset_epoch: &raw const device.reset_epoch,
            _scratch_borrow: scratch_borrow,
            info: queue_info,
            host_mapping,
            device_mapping,
            mappings: AtomicU64::new(0),
            destroying: AtomicBool::new(false),
            terminal: AtomicU64::new(0),
            consumed: AtomicU64::new(0),
            producer: AtomicU64::new(0),
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn info(
    pointer: *mut amdf_user_queue_t,
    out: *mut amdf_user_queue_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let out = output(out, AMDF_STRUCTURE_TYPE_USER_QUEUE_INFO)?;
        let queue = object(pointer.cast::<Queue>())?;
        queue.require_usable()?;
        out.publish(queue.info);
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn map(
    pointer: *mut amdf_user_queue_t,
    producer: *mut amdf_device_t,
    out: *mut *mut amdf_user_queue_mapping_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        output_pointer(out)?;
        let queue = object(pointer.cast::<Queue>())?;
        queue.require_usable()?;
        let host = producer.is_null();
        let peer = if host || producer.cast::<Device>() == queue.device {
            None
        } else {
            let producer = object(producer.cast::<Device>())?;
            let owner = &*queue.device;
            if (*producer.endpoint).instance != (*owner.endpoint).instance {
                return Err(INVALID);
            }
            if queue.info.capabilities & AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER == 0 {
                return Err(UNSUPPORTED);
            }
            Some(producer)
        };
        let slot = Owned::<Mapping>::try_new_uninit(queue.allocator).map_err(|_| EXHAUSTED)?;
        let info = if host {
            queue.host_mapping
        } else if peer.is_none() {
            queue.device_mapping.ok_or(UNSUPPORTED)?
        } else {
            let producer = peer.ok_or(INTERNAL)?;
            let mut info = transport(
                queue
                    .native
                    .map_device(producer)
                    .map_err(|error| native(&error))?,
                queue.info.command_type,
                queue.info.format_version,
                queue.info.format_features,
                true,
            )?;
            info.producer_device_id = producer.id;
            info.queue_id = queue.info.queue_id;
            info.queue_reset_epoch = queue.info.reset_epoch;
            info.producer_reset_epoch = producer.current_reset_epoch();
            info
        };
        register(&queue.mappings)?;
        let owner = slot.write(Mapping {
            queue: pointer.cast(),
            info,
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn mapping_info(
    pointer: *mut amdf_user_queue_mapping_t,
    out: *mut amdf_user_queue_mapping_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let out = output(out, AMDF_STRUCTURE_TYPE_USER_QUEUE_MAPPING_INFO)?;
        let mapping = object(pointer.cast::<Mapping>())?;
        (*mapping.queue).require_usable()?;
        out.publish(mapping.info);
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn unmap(pointer: *mut amdf_user_queue_mapping_t) -> u64 {
    crate::support::boundary(|| unsafe {
        let mapping = object(pointer.cast::<Mapping>())?;
        unregister(&(*mapping.queue).mappings);
        drop(Owned::from_raw(pointer.cast::<Mapping>()));
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn status(
    pointer: *mut amdf_user_queue_t,
    out: *mut amdf_user_queue_status_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let out = output(out, AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS)?;
        let queue = object(pointer.cast::<Queue>())?;
        out.publish(queue.sample()?);
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn wait(
    pointer: *mut amdf_user_queue_t,
    target: u64,
    timeout: u64,
    poll: u64,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let start = Instant::now();
        let queue = object(pointer.cast::<Queue>())?;
        wait_consumed(queue, target, timeout, poll, start)
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn destroy(pointer: *mut amdf_user_queue_t) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            let queue = exclusive(pointer.cast::<Queue>())?;
            if queue.mappings.load(Ordering::Acquire) != 0 {
                return Err(BUSY);
            }
            if !queue.destroying.load(Ordering::Acquire) {
                // A work-only BUSY rejection also leaves all public queue uses
                // available. No producer remains after mapping preflight, so the
                // published frontier cannot advance before native destruction.
                match queue.native.progress() {
                    Ok((consumed, published)) if consumed != published => return Err(BUSY),
                    Ok(_) => (),
                    Err(error) if error.kind() == rocddi::ErrorKind::DeviceLost => {
                        queue.observe_error(&error);
                    }
                    Err(error) => return Err(queue.observe_error(&error)),
                }
            }
            // Native destruction can release some backing before a later step
            // fails. Keep the owner for cleanup retries, but never republish cached
            // addresses or permit further queue use after that attempt begins.
            queue.destroying.store(true, Ordering::Release);
            if let Err(error) = queue.native.destroy() {
                return Err(queue.observe_error(&error));
            }
            unregister(&*queue.device_queues);
            drop(Owned::from_raw(pointer.cast::<Queue>()));
            Ok(())
        }
    })
}

#[cfg(test)]
#[path = "queue_tests.rs"]
mod tests;
