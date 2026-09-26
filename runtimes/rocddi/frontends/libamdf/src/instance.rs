//! Passive endpoint descriptions and explicit native device activation.
//!
//! Handles borrow their parent exactly as the AMDF contract requires. A cold
//! instance-local list recognizes registration of a provider-owned host view;
//! metadata queries never search or lock that list.

use crate::generated::amdf::*;
use crate::memory::Scope;
use crate::support::*;
use rocddi::host_storage::{Allocator, Buffer, Owned};
use rocddi::{device as native_device, memory as native_memory, session, topology};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

const INITIAL_RESET_EPOCH: u64 = 1;

/// Root AMDF owner containing native session state and embedded host scope.
///
/// `children` prevents destruction while endpoint handles remain public.
/// Allocator callbacks and cached host facts stay valid until destruction
/// succeeds.
pub(crate) struct Instance {
    pub allocator: Allocator,
    pub native: session::Session,
    pub children: AtomicU64,
    pub closing: bool,
    pub ids: AtomicU64,
    pub host_backing_head: Mutex<usize>,
    // Cached host facts keep profile queries independent of native calls.
    pub host_page_size: u64,
    pub host_cache_line: Option<u32>,
    pub scope: Scope,
}

/// Passive endpoint handle borrowing its instance and owning no device state.
pub(crate) struct Endpoint {
    pub instance: *mut Instance,
    pub native: topology::Endpoint,
    pub children: AtomicU64,
    pub scope: Scope,
}

/// Explicitly activated device handle borrowing its endpoint.
///
/// Queue ownership is counted for BUSY destruction, while `reset_epoch`
/// monotonically records native loss observed by this device or its children.
pub(crate) struct Device {
    pub endpoint: *mut Endpoint,
    pub native: native_device::Device,
    pub id: amdf_device_id_t,
    pub queues: AtomicU64,
    pub reset_epoch: AtomicU64,
}

pub(crate) fn advance_reset_epoch(epoch: &AtomicU64, resource_epoch: u64) {
    epoch.fetch_max(resource_epoch.saturating_add(1), Ordering::AcqRel);
}

impl Device {
    pub(crate) fn observe_loss(&self, resource_epoch: u64) {
        advance_reset_epoch(&self.reset_epoch, resource_epoch);
    }

    pub(crate) fn current_reset_epoch(&self) -> u64 {
        if self.native.has_observed_loss() {
            self.observe_loss(INITIAL_RESET_EPOCH);
        }
        self.reset_epoch.load(Ordering::Acquire)
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
    let engine_kind = match endpoint.kind() {
        topology::EndpointKind::Gpu { .. } => AMDF_ENGINE_KIND_GPU,
        topology::EndpointKind::Npu => AMDF_ENGINE_KIND_XDNA,
        _ => AMDF_ENGINE_KIND_UNKNOWN,
    };
    amdf_endpoint_summary_t {
        id: endpoint_id(endpoint.id),
        engine_kind,
        type_flags: if endpoint.linux_kfd_drm_info().render_minor.is_some() {
            AMDF_ENDPOINT_TYPE_FLAG_RENDER_SUPPORTED
        } else {
            0
        },
        name: name(endpoint),
    }
}

fn native_identity(endpoint: &topology::Endpoint) -> amdf_endpoint_native_identity_t {
    endpoint.linux_kfd_drm_info().render_minor.map_or_else(
        amdf_endpoint_native_identity_t::default,
        |minor| {
            amdf_endpoint_native_identity_t {
                r#type: AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_LINUX_DEVICE,
                value: amdf_endpoint_native_identity_t__value {
                    linux_device: amdf_endpoint_native_identity_t__value__linux_device {
                        // Linux reserves character-device major 226 for DRM.
                        major: 226,
                        minor,
                    },
                },
            }
        },
    )
}

pub(crate) fn family_count(endpoint: &topology::Endpoint) -> u32 {
    endpoint.gpu().map_or(0, |gpu| {
        u32::from(gpu.queues.aql) + u32::from(gpu.queues.sdma) + u32::from(gpu.queues.pm4)
    })
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
    gfx_major: u32,
    gfx_minor: u32,
    supports_gcr: bool,
) -> amdf_queue_format_features_t {
    let mut features = if supports_gcr {
        AMDF_GPU_SDMA_FORMAT_FEATURE_GCR
    } else {
        0
    };
    // GFX12 SDMA uses the explicit system bit in FENCE packets. GFX12.5
    // additionally exposes COPY_LINEAR and FENCE memory-scope fields. These
    // are packet-layout facts, independent of the cache operations qualified
    // for a family.
    if gfx_major >= 12 {
        features |= AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM;
    }
    if gfx_major == 12 && gfx_minor >= 5 {
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
    let gpu = endpoint.gpu().ok_or(UNSUPPORTED)?;
    let queues = gpu.queues;
    if ordinal >= family_count(endpoint) {
        return Err(RANGE);
    }
    let mut next = 0;
    let command_type = if queues.aql && ordinal == next {
        AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
    } else {
        next += u32::from(queues.aql);
        if queues.sdma && ordinal == next {
            AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA
        } else {
            next += u32::from(queues.sdma);
            if queues.pm4 && ordinal == next {
                AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
            } else {
                return Err(RANGE);
            }
        }
    };
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
        publication_modes: AMDF_QUEUE_PUBLICATION_MODE_USER
            | if (pm4 && queues.kernel_pm4) || (sdma && queues.kernel_sdma) {
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
        } else {
            sdma_format_features(
                gpu.gfx_major,
                gpu.gfx_minor,
                queues.sdma_system_cache_control,
            )
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
        user_queue_capabilities: AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER
            | if pm4 {
                0
            } else {
                AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER
            },
        producer_modes: AMDF_QUEUE_PRODUCER_MODE_BIT_SINGLE
            | if aql {
                AMDF_QUEUE_PRODUCER_MODE_BIT_MULTI
            } else {
                0
            },
        priority_capabilities: AMDF_QUEUE_PRIORITY_CAPABILITY_NORMAL
            | if aql {
                AMDF_QUEUE_PRIORITY_CAPABILITY_LOW | AMDF_QUEUE_PRIORITY_CAPABILITY_HIGH
            } else {
                0
            },
        minimum_ring_byte_length: if pm4 { 4096 } else { 1024 },
        maximum_ring_byte_length: if pm4 { 4096 } else { 1 << 31 },
        ring_byte_length_alignment: if pm4 { 4096 } else { 1024 },
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

#[allow(unused_unsafe)]
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
        let allocator = Allocator::from_callbacks(
            info.host_allocator.user_data,
            info.host_allocator.allocate,
            info.host_allocator.resize,
            info.host_allocator.free,
        )
        .map_err(|_| INVALID)?;
        let slot = Owned::<Instance>::try_new_uninit(allocator).map_err(|_| EXHAUSTED)?;
        let lifetime = if info.native_lifetime == AMDF_NATIVE_LIFETIME_PROCESS {
            session::SessionLifetime::Process
        } else {
            session::SessionLifetime::Session
        };
        let native = session::Session::with_allocator(lifetime, allocator)
            .map_err(|e| crate::support::native(&e))?;
        let host_page_size =
            native_memory::host_page_size().map_err(|e| crate::support::native(&e))?;
        let host_cache_line = native_memory::host_cache_line_size().ok();
        let owner = slot.write(Instance {
            allocator,
            native,
            host_page_size,
            host_cache_line,
            children: AtomicU64::new(0),
            closing: false,
            ids: AtomicU64::new(1),
            host_backing_head: Mutex::new(0),
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

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn destroy(pointer: *mut amdf_instance_t) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            // Destruction is externally serialized. Keep the callback allocator
            // alive until native shutdown releases every owner that can use it.
            let instance = exclusive(pointer.cast::<Instance>())?;
            if instance.children.load(Ordering::Acquire) != 0 {
                return Err(BUSY);
            }
            if *instance.host_backing_head.lock().map_err(|_| INTERNAL)? != 0 {
                return Err(BUSY);
            }
            // Native cleanup can make partial progress. Retain this handle on
            // failure for destruction only; no new child may borrow it then.
            instance.closing = true;
            instance.native.destroy().map_err(|e| native(&e))?;
            drop(Owned::from_raw(pointer.cast::<Instance>()));
            Ok(())
        }
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn enumerate(
    pointer: *mut amdf_instance_t,
    capacity: u32,
    summaries: *mut amdf_endpoint_summary_t,
    count: *mut u32,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let instance = object(pointer.cast::<Instance>())?;
        if instance.closing {
            return Err(PRECONDITION);
        }
        output_pointer(count)?;
        output_array(summaries, capacity)?;
        let mut staged = Buffer::new(instance.allocator);
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

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn open(
    pointer: *mut amdf_instance_t,
    id: *const amdf_endpoint_id_t,
    out: *mut *mut amdf_endpoint_t,
) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            let instance = object(pointer.cast::<Instance>())?;
            if instance.closing {
                return Err(PRECONDITION);
            }
            output_pointer(out)?;
            let id = *object(id)?;
            let slot =
                Owned::<Endpoint>::try_new_uninit(instance.allocator).map_err(|_| EXHAUSTED)?;
            let native = instance
                .native
                .open_endpoint(native_id(id))
                .map_err(|error| {
                    // Opening a passive identity cannot lose an activated device. A
                    // stale or vanished identity is simply absent from discovery.
                    if matches!(
                        error.kind(),
                        rocddi::ErrorKind::InvalidArgument | rocddi::ErrorKind::DeviceLost
                    ) || error.native_error_code() == Some(2)
                    {
                        NOT_FOUND
                    } else {
                        crate::support::native(&error)
                    }
                })?;
            register(&instance.children)?;
            let owner = slot.write(Endpoint {
                instance: pointer.cast(),
                native,
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

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn close(pointer: *mut amdf_endpoint_t) -> u64 {
    crate::support::boundary(|| unsafe {
        let endpoint = object(pointer.cast::<Endpoint>())?;
        if endpoint.children.load(Ordering::Acquire) != 0 {
            return Err(BUSY);
        }
        unregister(&(*endpoint.instance).children);
        drop(Owned::from_raw(pointer.cast::<Endpoint>()));
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn endpoint_info(
    pointer: *mut amdf_endpoint_t,
    out: *mut amdf_endpoint_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let endpoint = object(pointer.cast::<Endpoint>())?;
        let out = output(out, AMDF_STRUCTURE_TYPE_ENDPOINT_INFO)?;
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
            native_identity: native_identity(&endpoint.native),
            ..Default::default()
        });
        Ok(())
    })
}

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn device_create(
    pointer: *mut amdf_endpoint_t,
    info: *const amdf_gpu_device_create_info_t,
    out: *mut *mut amdf_device_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let endpoint = object(pointer.cast::<Endpoint>())?;
        output_pointer(out)?;
        let info = input(info, AMDF_STRUCTURE_TYPE_GPU_DEVICE_CREATE_INFO)?;
        if info.reserved != 0 {
            return Err(INVALID);
        }
        supported_endpoint(endpoint)?;
        let instance = &*endpoint.instance;
        let slot = Owned::<Device>::try_new_uninit(instance.allocator).map_err(|_| EXHAUSTED)?;
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
        let owner = slot.write(Device {
            endpoint: pointer.cast(),
            native,
            id: amdf_device_id_t {
                words: [endpoint.instance as u64, id],
            },
            queues: AtomicU64::new(0),
            reset_epoch: AtomicU64::new(INITIAL_RESET_EPOCH),
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

#[allow(unused_unsafe)]
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
            ),
            ..Default::default()
        });
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn device_destroy(pointer: *mut amdf_device_t) -> u64 {
    crate::support::boundary(|| unsafe {
        let device = object(pointer.cast::<Device>())?;
        if device.queues.load(Ordering::Acquire) != 0 {
            return Err(BUSY);
        }
        unregister(&(*device.endpoint).children);
        drop(Owned::from_raw(pointer.cast::<Device>()));
        Ok(())
    })
}
