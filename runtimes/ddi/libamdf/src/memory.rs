// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMDF storage scopes, explicit access and host-view lifetimes.
//!
//! One resource owns its native backing; public parents are borrowed. This
//! provider establishes every requested SYSTEM-memory GPU access before
//! publication. Qualified direct peers may share LOCAL backing. External
//! imports require native permission and cache evidence before advertisement.

use crate::generated::amdf::*;
use crate::instance::{Device, Endpoint, Instance, endpoint_id, supported_endpoint};
use crate::platform::external_memory as external_transport;
use crate::reservation::Reservation;
use crate::support::*;
use rocddi::host::cache::CacheMaintenance;
use rocddi::memory;
use rocddi::memory::storage::TryPush;
use rocddi::memory::storage::{Allocator, Owned, OwnerVec};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const BACKING_FLAGS: u64 = AMDF_MEMORY_FLAG_HOST_VISIBLE
    | AMDF_MEMORY_FLAG_DEVICE_LOCAL
    | AMDF_MEMORY_FLAG_SHAREABLE
    | AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE
    | AMDF_MEMORY_FLAG_HOST_FILE_IO;
const ACCESS_FLAGS: u64 = AMDF_MEMORY_FLAG_QUEUE_STORAGE
    | AMDF_MEMORY_FLAG_HOST_COHERENT
    | AMDF_MEMORY_FLAG_DEVICE_ADDRESS
    | AMDF_MEMORY_FLAG_MAPPING_SOURCE;
const MAX_ALIGNMENT: u64 = 1 << 30;
const MAX_LENGTH: u64 = (isize::MAX as u64) - MAX_ALIGNMENT;

/// Memory-profile namespace embedded in either an instance or an endpoint.
pub(crate) struct Scope {
    /// Instance that owns this profile namespace.
    pub instance: *mut Instance,
    /// Endpoint for device-scoped profiles, or null for the host scope.
    pub endpoint: *mut Endpoint,
}

/// Native or caller-owned storage underlying one public memory object.
enum Backing {
    /// Host-only allocation managed by rocddi.
    Host(memory::host::HostAllocation),
    /// Caller-owned pages registered with a device.
    Registered,
    /// Driver-owned allocation with qualified device access.
    Gpu(memory::device::Allocation),
    /// View borrowing an existing provider-owned allocation.
    Alias { _borrow: SourceBorrow },
}

/// A registration of a provider-owned host view borrows its original backing.
struct SourceBorrow {
    /// Original public owner whose child count protects this view.
    source: *mut Memory,
}

impl Drop for SourceBorrow {
    fn drop(&mut self) {
        // SAFETY: The borrow registered a child before leaving the instance
        // list lock, so the source remains live through this release.
        unsafe { unregister(&(*self.source).children) };
    }
}

/// One device's established access and translated address for an allocation.
struct Access {
    /// Borrowed public device whose VM was granted access.
    device: *mut Device,
    /// Access requirements and epoch returned to the caller.
    info: amdf_memory_access_info_t,
    /// Address usable by this device.
    address: u64,
}

/// Public AMDF memory owner and all access mappings derived at construction.
///
/// Host mappings and queue-scratch borrows increment `children`, preventing
/// backing destruction while any public child can still reach it. `freeing`
/// prevents a failed or reentrant destruction from restarting the transition.
pub(crate) struct Memory {
    /// Host or endpoint namespace that created this object.
    scope: *mut Scope,
    /// Policy retained for public mapping metadata.
    host_allocator: Allocator,
    /// Native allocation, host storage, registration, or borrowed view.
    backing: Backing,
    /// Whether this attachment owns payload capacity rather than foreign pages.
    allocated_backing: bool,
    /// Public allocation facts fixed at creation.
    info: amdf_memory_info_t,
    /// Device access established before publication.
    access: OwnerVec<Access>,
    /// Mapped CPU address when this memory is host visible.
    host: Option<usize>,
    /// Public host cacheability classification.
    cacheability: u32,
    /// Public mappings and queue scratch borrows still alive.
    children: AtomicU64,
    /// Prevents destruction from being retried after a transition begins.
    freeing: bool,
    /// Next provider-owned host backing in the instance's cold alias list.
    host_backing_next: AtomicUsize,
    /// Whether this object currently appears in that alias list.
    listed_host_backing: bool,
}

/// Publishes cleanup disposition without taking ownership of a public handle.
unsafe fn publish_release(out: *mut amdf_memory_release_info_t, value: amdf_memory_release_info_t) {
    if !out.is_null() {
        // SAFETY: The caller supplies a writable release record when non-null.
        unsafe { out.write(value) };
    }
}

#[cfg_attr(
    windows,
    allow(
        clippy::forget_non_drop,
        reason = "no Windows provider instantiates the native cleanup owner yet"
    )
)]
fn release_gpu(
    mut allocation: memory::device::Allocation,
    device: *mut Device,
    allocated_backing: bool,
) -> amdf_memory_release_info_t {
    match allocation.free() {
        Ok(()) => amdf_memory_release_info_t::default(),
        Err(error) => {
            let status = if device.is_null() {
                native(&error)
            } else {
                // SAFETY: The allocation borrows this device throughout release.
                unsafe { (*device).native_error_status(&error) }
            };
            let abandoned_byte_length = if allocated_backing {
                allocation.remaining_backing_byte_length()
            } else {
                0
            };
            // Explicit AMDF cleanup consumes the attachment. A second native
            // release could replay an ambiguous detach, so retain its owner.
            std::mem::forget(allocation);
            amdf_memory_release_info_t {
                status,
                abandoned_byte_length,
            }
        }
    }
}

impl Memory {
    /// A native BUSY without errno means the allocation and public handle are
    /// still live, so the caller may end its borrows and retry destruction.
    fn release_native_for_destroy(&mut self) -> Result<amdf_memory_release_info_t, u64> {
        let Backing::Gpu(allocation) = &mut self.backing else {
            return Ok(self.release_native());
        };
        match allocation.free() {
            Ok(()) => {
                self.backing = Backing::Registered;
                Ok(amdf_memory_release_info_t::default())
            }
            Err(error)
                if error.kind() == rocddi::ErrorKind::Busy && error.native_status().is_none() =>
            {
                Err(BUSY)
            }
            Err(error) => {
                let device = self
                    .access
                    .first()
                    .map_or(std::ptr::null_mut(), |access| access.device);
                let status = if device.is_null() {
                    native(&error)
                } else {
                    // SAFETY: The allocation borrows this admitted device.
                    unsafe { (*device).native_error_status(&error) }
                };
                let abandoned_byte_length = if self.allocated_backing {
                    allocation.remaining_backing_byte_length()
                } else {
                    0
                };
                let owner = std::mem::replace(&mut self.backing, Backing::Registered);
                std::mem::forget(owner);
                Ok(amdf_memory_release_info_t {
                    status,
                    abandoned_byte_length,
                })
            }
        }
    }

    fn release_native(&mut self) -> amdf_memory_release_info_t {
        let device = self
            .access
            .first()
            .map_or(std::ptr::null_mut(), |a| a.device);
        let backing = std::mem::replace(&mut self.backing, Backing::Registered);
        match backing {
            Backing::Gpu(allocation) => release_gpu(allocation, device, self.allocated_backing),
            Backing::Host(mut allocation) => match allocation.free() {
                Ok(()) => amdf_memory_release_info_t::default(),
                Err(error) => {
                    let status = native(&error);
                    let abandoned_byte_length = self.info.native_allocation_byte_length;
                    std::mem::forget(allocation);
                    amdf_memory_release_info_t {
                        status,
                        abandoned_byte_length,
                    }
                }
            },
            Backing::Registered | Backing::Alias { .. } => amdf_memory_release_info_t::default(),
        }
    }
}

/// Native allocation held only during construction, before a public Memory
/// owner exists. Dropping it reports failed rollback independently.
struct PendingGpu {
    allocation: Option<memory::device::Allocation>,
    device: *mut Device,
    allocated_backing: bool,
    out_release_info: *mut amdf_memory_release_info_t,
}

impl PendingGpu {
    fn new(
        allocation: memory::device::Allocation,
        device: *mut Device,
        allocated_backing: bool,
        out_release_info: *mut amdf_memory_release_info_t,
    ) -> Self {
        Self {
            allocation: Some(allocation),
            device,
            allocated_backing,
            out_release_info,
        }
    }

    fn allocation(&self) -> Result<&memory::device::Allocation, u64> {
        self.allocation.as_ref().ok_or(INTERNAL)
    }

    fn allocation_mut(&mut self) -> Result<&mut memory::device::Allocation, u64> {
        self.allocation.as_mut().ok_or(INTERNAL)
    }

    fn into_inner(mut self) -> Result<memory::device::Allocation, u64> {
        self.allocation.take().ok_or(INTERNAL)
    }
}

impl Drop for PendingGpu {
    fn drop(&mut self) {
        if let Some(allocation) = self.allocation.take() {
            let result = release_gpu(allocation, self.device, self.allocated_backing);
            // SAFETY: The call's optional output remains writable until return.
            unsafe { publish_release(self.out_release_info, result) };
        }
    }
}

/// Resolves a same-instance provider view to its public physical identity and
/// full native extent. Its child borrow stabilizes those facts through the
/// view's lifetime. rocddi separately reuses BO mappings across frontends.
unsafe fn find_host_alias(
    instance: &Instance,
    address: usize,
    byte_length: u64,
    accesses: &[amdf_memory_device_access_t],
) -> Result<Option<(SourceBorrow, u64)>, u64> {
    let head = instance.host_backing_head.lock().map_err(|_| INTERNAL)?;
    let mut cursor = *head;
    while cursor != 0 {
        let source = unsafe { &*(cursor as *mut Memory) };
        cursor = source.host_backing_next.load(Ordering::Relaxed);
        let Some(base) = source.host else { continue };
        let Some(offset) = address.checked_sub(base) else {
            continue;
        };
        let Ok(offset) = u64::try_from(offset) else {
            continue;
        };
        if !source.listed_host_backing
            || source.freeing
            || source.info.memory_class != AMDF_MEMORY_CLASS_SYSTEM
            || !matches!(source.backing, Backing::Gpu(_))
            || offset > source.info.byte_length
            || byte_length > source.info.byte_length - offset
        {
            continue;
        }
        let exact_access = accesses.iter().all(|requested| {
            // SAFETY: The caller keeps every requested device live, and source
            // accesses borrow devices for the lifetime of the listed source.
            let device = unsafe { &*requested.device.cast::<Device>() };
            source.access.iter().any(|existing| {
                let owner = unsafe { &*existing.device };
                device.native.shares_address_domain(&owner.native)
                    && existing.info.access == requested.requirements.access
                    && existing.info.flags & requested.requirements.flags
                        == requested.requirements.flags
                    && existing.info.reset_epoch == device.current_reset_epoch()
            })
        });
        if exact_access {
            register(&source.children)?;
            return Ok(Some((
                SourceBorrow {
                    source: std::ptr::from_ref(source).cast_mut(),
                },
                offset,
            )));
        }
    }
    Ok(None)
}

fn list_host_backing(instance: &Instance, memory: &mut Owned<Memory>) -> Result<(), u64> {
    let mut head = instance.host_backing_head.lock().map_err(|_| INTERNAL)?;
    memory.host_backing_next.store(*head, Ordering::Relaxed);
    *head = memory.as_mut_ptr() as usize;
    Ok(())
}

fn unlink_host_backing(instance: &Instance, pointer: *mut Memory) -> Result<(), u64> {
    let mut head = instance.host_backing_head.lock().map_err(|_| INTERNAL)?;
    // SAFETY: the caller owns this memory exclusively, and the list lock
    // prevents a new alias from acquiring a child between the check and unlink.
    if unsafe { (*pointer).children.load(Ordering::Acquire) } != 0 {
        return Err(BUSY);
    }
    let target = pointer as usize;
    let mut cursor = *head;
    let mut previous = 0usize;
    while cursor != 0 && cursor != target {
        // SAFETY: The list lock protects links and every linked owner remains
        // live until removal under this same lock.
        let node = unsafe { &*(cursor as *mut Memory) };
        previous = cursor;
        cursor = node.host_backing_next.load(Ordering::Relaxed);
    }
    if cursor != target {
        return Err(INTERNAL);
    }
    // SAFETY: target is still linked and live while its exclusive destructor
    // holds this lock.
    let next = unsafe { (*pointer).host_backing_next.load(Ordering::Relaxed) };
    if previous == 0 {
        *head = next;
    } else {
        unsafe {
            (*(previous as *mut Memory))
                .host_backing_next
                .store(next, Ordering::Relaxed);
        };
    }
    Ok(())
}

/// Public host-view handle borrowing one memory owner.
#[derive(Clone, Copy)]
enum MappingOwner {
    Memory(*mut Memory),
    Reservation(*mut Reservation),
}

struct Mapping {
    /// Parent kept live through a child count until this view is destroyed.
    owner: MappingOwner,
    /// Host address, extent, and cache facts returned to callers.
    info: amdf_host_mapping_info_t,
    /// Recipe selected from this mapping's cache policy and retained with it.
    maintenance: Option<CacheMaintenance>,
}

impl Mapping {
    /// A live mapping borrows its parent and instance under the C contract.
    ///
    /// # Safety
    /// The caller holds a live public mapping handle.
    unsafe fn instance(&self) -> &Instance {
        match self.owner {
            MappingOwner::Memory(memory) => unsafe { &*(*(*memory).scope).instance },
            MappingOwner::Reservation(reservation) => unsafe { &*(*reservation).instance },
        }
    }
}

fn rounded(value: u64, granularity: u64) -> Result<u64, u64> {
    value
        .checked_add(granularity - 1)
        .map(|n| n & !(granularity - 1))
        .ok_or(RANGE)
}

fn common_multiple(a: u64, b: u64) -> Result<u64, u64> {
    if a == 0 || b == 0 {
        return if a == b { Ok(0) } else { Err(UNSUPPORTED) };
    }
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    (a / x).checked_mul(b).ok_or(UNSUPPORTED)
}

fn logical_maximum(
    native: u64,
    unit: u64,
    logical_unit: u64,
    pointer_alignment: Option<u64>,
) -> Result<u64, u64> {
    if !unit.is_power_of_two() || logical_unit == 0 {
        return Err(UNSUPPORTED);
    }
    let offset = if let Some(alignment) = pointer_alignment {
        if !alignment.is_power_of_two() {
            return Err(UNSUPPORTED);
        }
        unit - unit.min(alignment)
    } else {
        0
    };
    let maximum = (native - native % unit).saturating_sub(offset);
    let maximum = maximum - maximum % logical_unit;
    if maximum == 0 {
        Err(UNSUPPORTED)
    } else {
        Ok(maximum)
    }
}

fn native_extent(
    size: u64,
    address: Option<u64>,
    limits: amdf_memory_construction_capabilities_t,
) -> Result<(u64, u64), u64> {
    let granularity = limits.native_byte_length_granularity;
    if limits.byte_length_granularity == 0
        || size % limits.byte_length_granularity != 0
        || !granularity.is_power_of_two()
        || limits.native_byte_length_prefix != 0
    {
        return Err(UNSUPPORTED);
    }
    let offset = address.map_or(0, |address| address % granularity);
    Ok((
        offset,
        rounded(offset.checked_add(size).ok_or(RANGE)?, granularity)?,
    ))
}

fn no_transition() -> amdf_cache_transition_t {
    amdf_cache_transition_t {
        kind: AMDF_CACHE_TRANSITION_KIND_NONE,
        ..Default::default()
    }
}

unsafe fn scope_count(scope: &Scope) -> u32 {
    if scope.endpoint.is_null() {
        2
    } else {
        let endpoint = unsafe { &*scope.endpoint };
        u32::from(endpoint.native.local_memory_bytes != 0)
            + u32::from(endpoint.native.host_visible_local_memory_bytes != 0)
    }
}

fn requirements(value: amdf_memory_access_requirements_t) -> Result<(), u64> {
    if value.reserved != 0
        || value.access & !7 != 0
        || value.flags & !ACCESS_FLAGS != 0
        || value.address_kinds & !7 != 0
    {
        Err(INVALID)
    } else {
        Ok(())
    }
}

fn has_identity(words: [u64; 2]) -> bool {
    words != [0; 2]
}

unsafe fn validate_external(value: &amdf_external_memory_t) -> Result<(), u64> {
    if value.reserved != 0
        || value.byte_length == 0
        || value
            .source_byte_offset
            .checked_add(value.byte_length)
            .is_none()
        || !(AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD..=AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS)
            .contains(&value.r#type)
    {
        return Err(INVALID);
    }
    let provenance = has_identity(value.provenance.words);
    match value.r#type {
        AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD => {
            let descriptor = unsafe { value.payload.file_descriptor };
            if descriptor < 0 || provenance {
                Err(INVALID)
            } else {
                Ok(())
            }
        }
        AMDF_EXTERNAL_MEMORY_TYPE_OPAQUE_FD => {
            let descriptor = unsafe { value.payload.file_descriptor };
            if descriptor < 0 || !provenance {
                Err(INVALID)
            } else {
                Ok(())
            }
        }
        AMDF_EXTERNAL_MEMORY_TYPE_D3D12_RESOURCE | AMDF_EXTERNAL_MEMORY_TYPE_HOST_POINTER => {
            let pointer = unsafe { value.payload.native_handle };
            if pointer.is_null() || provenance {
                Err(INVALID)
            } else {
                Ok(())
            }
        }
        AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS => {
            let address = unsafe { value.payload.device_address };
            if address == 0 || !provenance {
                Err(INVALID)
            } else {
                Ok(())
            }
        }
        _ => Err(INVALID),
    }
}

fn external_support(
    profile: &amdf_memory_profile_t,
    external_type: u32,
    required_flag: u32,
    source_offset: u64,
    byte_length: u64,
) -> Result<amdf_external_memory_support_t, u64> {
    let count = usize::try_from(profile.external_memory_support_count).map_err(|_| INTERNAL)?;
    let supports = profile
        .external_memory_support
        .get(..count)
        .ok_or(INTERNAL)?;
    let support = supports
        .iter()
        .find(|support| support.r#type == external_type)
        .ok_or(UNSUPPORTED)?;
    if support.flags & required_flag == 0
        || (source_offset != 0
            && (support.flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET == 0
                || support.source_offset_alignment == 0
                || source_offset % support.source_offset_alignment != 0))
        || support.byte_length_alignment == 0
        || byte_length % support.byte_length_alignment != 0
        || (support.maximum_byte_length != 0 && byte_length > support.maximum_byte_length)
    {
        return Err(UNSUPPORTED);
    }
    Ok(*support)
}

fn supports_dma_buf_export(local: bool, registered: bool, has_endpoint: bool) -> bool {
    !local && !registered && has_endpoint
}

#[allow(
    clippy::too_many_lines,
    reason = "placement and access facts must describe the same complete construction combination"
)]
unsafe fn profile(
    scope: &Scope,
    ordinal: u32,
    endpoint: Option<&Endpoint>,
    request: amdf_memory_access_requirements_t,
    import_supported: bool,
) -> Result<(amdf_memory_profile_t, amdf_memory_access_capabilities_t), u64> {
    if ordinal >= unsafe { scope_count(scope) } {
        return Err(RANGE);
    }
    requirements(request)?;
    let local = !scope.endpoint.is_null();
    let registered = !local && ordinal == 1;
    if local && endpoint.is_none() {
        return Err(UNSUPPORTED);
    }
    let instance = unsafe { &*scope.instance };
    if let Some(endpoint) = endpoint {
        if endpoint.instance != scope.instance {
            return Err(INVALID);
        }
        supported_endpoint(endpoint)?;
    }
    // The cached driver policy query rejects a closing native session. Keep
    // AMDF's PRECONDITION status ahead of that query for valid handles.
    if instance.closing {
        return Err(PRECONDITION);
    }
    if local {
        let endpoint = endpoint.ok_or(UNSUPPORTED)?;
        let owner = unsafe { &*scope.endpoint };
        if !instance
            .native
            .can_access_local_memory(&endpoint.native, &owner.native)
        {
            return Err(UNSUPPORTED);
        }
    }
    // Public VRAM has a different capacity from private VRAM on partial-BAR
    // devices. Separate profiles keep both placement and host-map limits exact.
    let host_visible = !local || ordinal == 1;
    if registered
        && endpoint
            .is_some_and(|endpoint| !instance.native.supports_host_registration(&endpoint.native))
    {
        return Err(UNSUPPORTED);
    }
    let provider = endpoint
        .map(|endpoint| {
            use memory::device::{HostMappingPolicy, MemoryCapabilityRequest, MemoryKind};
            let request = if registered {
                MemoryCapabilityRequest::Registration(HostMappingPolicy::Fine)
            } else {
                MemoryCapabilityRequest::Allocation(if local {
                    MemoryKind::DeviceLocal {
                        host_visible,
                        coherent: false,
                        extended_coherent: false,
                        uncached: false,
                        contiguous: false,
                    }
                } else {
                    MemoryKind::System
                })
            };
            let owner = if local {
                unsafe { &*scope.endpoint }
            } else {
                endpoint
            };
            instance
                .native
                .memory_capabilities(&owner.native, request)
                .map_err(|error| native(&error))
        })
        .transpose()?;
    let access_provider = if local {
        let owner = unsafe { &*scope.endpoint };
        let consumer = endpoint.ok_or(UNSUPPORTED)?;
        Some(
            instance
                .native
                .memory_capabilities(
                    &consumer.native,
                    memory::device::MemoryCapabilityRequest::Access {
                        owner: &owner.native,
                        kind: memory::device::MemoryKind::DeviceLocal {
                            host_visible,
                            coherent: false,
                            extended_coherent: false,
                            uncached: false,
                            contiguous: false,
                        },
                    },
                )
                .map_err(|error| native(&error))?,
        )
    } else {
        provider
    };
    if host_visible
        && provider.is_some_and(|cap| !cap.host_visible || cap.host_cacheability.is_none())
    {
        return Err(UNSUPPORTED);
    }
    let page = provider.map_or(instance.host_page_size, |cap| cap.native.granularity);
    let logical_alignment = provider.map_or(if registered { 1 } else { page }, |cap| {
        cap.logical_alignment
    });
    // Provider-owned host-visible storage uses a qualified cache recipe.
    // Registration can still expose borrowed pages without knowing their cache
    // mode; AMDF represents unavailable maintenance separately from mapping.
    if host_visible
        && (!registered || endpoint.is_some())
        && (instance.host_write_back_cache.is_none()
            || (local && unsafe { (*scope.endpoint).native.host_local_cacheability }.is_none()))
    {
        return Err(UNSUPPORTED);
    }
    let host_only = !local && endpoint.is_none() && !registered;
    let backing_flags = if local {
        AMDF_MEMORY_FLAG_DEVICE_LOCAL
    } else {
        0
    } | if host_visible {
        AMDF_MEMORY_FLAG_HOST_VISIBLE
    } else {
        0
    } | if host_only {
        AMDF_MEMORY_FLAG_HOST_FILE_IO
    } else {
        0
    };
    let maximum = if local {
        let endpoint = unsafe { &*scope.endpoint };
        if host_visible {
            endpoint.native.host_visible_local_memory_bytes
        } else {
            endpoint.native.local_memory_bytes
        }
        .min(MAX_LENGTH)
    } else {
        MAX_LENGTH
    };
    let maximum = provider.map_or(maximum, |cap| maximum.min(cap.native.maximum_size));
    let maximum = logical_maximum(
        maximum,
        page,
        provider.map_or(1, |cap| cap.logical_size_granularity),
        registered.then_some(logical_alignment),
    )?;
    let construction = amdf_memory_construction_capabilities_t {
        maximum_byte_length: maximum,
        byte_length_granularity: provider.map_or(1, |cap| cap.logical_size_granularity),
        registered_host_pointer_alignment: if registered { logical_alignment } else { 0 },
        registered_host_cacheability: if registered {
            match provider.and_then(|cap| cap.host_cacheability) {
                Some(memory::device::HostCacheability::WriteCombined) => {
                    AMDF_HOST_CACHEABILITY_WRITE_COMBINED
                }
                _ => AMDF_HOST_CACHEABILITY_WRITE_BACK,
            }
        } else {
            AMDF_HOST_CACHEABILITY_UNKNOWN
        },
        placement_kinds: 0,
        minimum_alignment: logical_alignment,
        maximum_alignment: provider.map_or(MAX_ALIGNMENT, |cap| {
            MAX_ALIGNMENT.min(cap.maximum_alignment)
        }),
        native_byte_length_granularity: page,
        native_byte_length_prefix: 0,
    };
    let mut p = amdf_memory_profile_t {
        ordinal,
        memory_class: if local {
            AMDF_MEMORY_CLASS_LOCAL
        } else {
            AMDF_MEMORY_CLASS_SYSTEM
        },
        roles: if registered {
            AMDF_MEMORY_PROFILE_ROLE_REGISTER
        } else {
            AMDF_MEMORY_PROFILE_ROLE_CREATE
        } | if host_visible {
            AMDF_MEMORY_PROFILE_ROLE_HOST_MAP
        } else {
            0
        },
        guaranteed_flags: backing_flags,
        supported_flags: backing_flags
            | if host_visible {
                AMDF_MEMORY_FLAG_HOST_VISIBLE
            } else {
                0
            }
            | if host_only
                && memory::host::capabilities()
                    .map_err(|error| native(&error))?
                    .remappable_backing
            {
                AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE
            } else {
                0
            },
        host_mapping: if host_visible {
            amdf_host_mapping_capabilities_t {
                maximum_byte_length: maximum,
                byte_offset_granularity: 1,
                byte_length_granularity: 1,
                supported_access: AMDF_MEMORY_MAP_FLAG_READ
                    | if provider.is_none_or(|cap| cap.host_writable) {
                        AMDF_MEMORY_MAP_FLAG_WRITE
                    } else {
                        0
                    },
                features: if host_only {
                    AMDF_HOST_MAPPING_FEATURE_FILE_IO
                } else {
                    0
                },
            }
        } else {
            amdf_host_mapping_capabilities_t::default()
        },
        ..Default::default()
    };
    if registered {
        p.registration = construction;
    } else {
        p.allocation = construction;
    }
    let mut capabilities = amdf_memory_access_capabilities_t::default();
    if let Some(endpoint) = endpoint {
        let requested_access =
            memory::device::DeviceAccess::from_bits(request.access).ok_or(UNSUPPORTED)?;
        let flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS
            | if access_provider.is_some_and(|cap| cap.host_coherent) {
                AMDF_MEMORY_FLAG_HOST_COHERENT
            } else {
                0
            };
        let cap = access_provider.ok_or(INTERNAL)?;
        let supported_flags = flags
            | if cap.queue_storage {
                AMDF_MEMORY_FLAG_QUEUE_STORAGE
            } else {
                0
            };
        if !requested_access.contains(cap.required_access)
            || !cap.supported_access.contains(requested_access)
            || request.flags & !supported_flags != 0
            || request.address_kinds & !1 != 0
        {
            return Err(UNSUPPORTED);
        }
        capabilities = amdf_memory_access_capabilities_t {
            guaranteed_access: cap.required_access.bits(),
            supported_access: cap.supported_access.bits(),
            guaranteed_flags: flags,
            supported_flags,
            address_kinds: 1,
            device_address: amdf_memory_address_capabilities_t {
                address_domain_ordinal: 0,
                address_bit_count: endpoint.native.address_bit_count,
                minimum_address: endpoint.native.minimum_address,
                maximum_address: endpoint.native.maximum_address,
                minimum_alignment: logical_alignment,
            },
            ..Default::default()
        };
        if supports_dma_buf_export(
            local,
            registered,
            external_transport::supports_device_export(&endpoint.native),
        ) {
            p.roles |= AMDF_MEMORY_PROFILE_ROLE_EXPORT;
            if import_supported {
                p.roles |= AMDF_MEMORY_PROFILE_ROLE_IMPORT;
                p.import = construction;
            }
            p.guaranteed_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
            p.supported_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
            p.external_memory_support_count = 1;
            p.external_memory_support[0] =
                external_transport::support(page, maximum, import_supported);
        }
    }
    Ok((p, capabilities))
}

/// Validated per-device requirements used while composing a joint profile.
#[derive(Clone, Copy)]
struct ProfileAccess<'a> {
    /// Passive endpoint supplying device profile facts.
    endpoint: &'a Endpoint,
    /// Requested permission and address properties.
    request: amdf_memory_access_requirements_t,
    /// Inclusive native address bounds for the endpoint.
    address_range: (u64, u64),
    /// Whether this endpoint supports the requested external import.
    import_supported: bool,
}

fn same_requirements(
    left: amdf_memory_access_requirements_t,
    right: amdf_memory_access_requirements_t,
) -> bool {
    left.access == right.access
        && left.reserved == right.reserved
        && left.flags == right.flags
        && left.address_kinds == right.address_kinds
}

fn uniform_requirements(
    requests: impl IntoIterator<Item = amdf_memory_access_requirements_t>,
) -> bool {
    let mut requests = requests.into_iter();
    let Some(first) = requests.next() else {
        return true;
    };
    requests.all(|request| same_requirements(first, request))
}

fn unique_devices(accesses: &[amdf_memory_device_access_t]) -> bool {
    accesses.iter().enumerate().all(|(index, access)| {
        !accesses[..index]
            .iter()
            .any(|previous| previous.device == access.device)
    })
}

fn physical_owner_index(
    local: bool,
    owner: Option<[u8; 16]>,
    access_endpoints: impl IntoIterator<Item = [u8; 16]>,
) -> Result<usize, u64> {
    if !local {
        return Ok(0);
    }
    let owner = owner.ok_or(INTERNAL)?;
    access_endpoints
        .into_iter()
        .position(|endpoint| endpoint == owner)
        .ok_or(INTERNAL)
}

fn constrain_maximum(value: &mut u64, maximum: u64) {
    if *value != 0 {
        *value = (*value).min(maximum);
    }
}

fn constrain_profile(profile: &mut amdf_memory_profile_t, maximum: u64) {
    constrain_maximum(&mut profile.allocation.maximum_byte_length, maximum);
    constrain_maximum(&mut profile.registration.maximum_byte_length, maximum);
    constrain_maximum(&mut profile.import.maximum_byte_length, maximum);
    constrain_maximum(&mut profile.host_mapping.maximum_byte_length, maximum);
    let count = profile.external_memory_support_count as usize;
    for support in profile.external_memory_support.iter_mut().take(count) {
        constrain_maximum(&mut support.maximum_byte_length, maximum);
    }
}

fn intersect_construction(
    selected: &mut amdf_memory_construction_capabilities_t,
    other: amdf_memory_construction_capabilities_t,
) -> Result<(), u64> {
    if selected.maximum_byte_length == 0 || other.maximum_byte_length == 0 {
        *selected = amdf_memory_construction_capabilities_t::default();
        return Ok(());
    }
    // One AMDF backing has one native extent recipe. Mixed recipes need a
    // distinct shared mapping design and must not silently select the first.
    if selected.native_byte_length_granularity != other.native_byte_length_granularity
        || selected.native_byte_length_prefix != other.native_byte_length_prefix
        || selected.registered_host_cacheability != other.registered_host_cacheability
    {
        return Err(UNSUPPORTED);
    }
    selected.byte_length_granularity = common_multiple(
        selected.byte_length_granularity,
        other.byte_length_granularity,
    )?;
    selected.registered_host_pointer_alignment = common_multiple(
        selected.registered_host_pointer_alignment,
        other.registered_host_pointer_alignment,
    )?;
    selected.minimum_alignment = selected.minimum_alignment.max(other.minimum_alignment);
    selected.maximum_alignment = selected.maximum_alignment.min(other.maximum_alignment);
    selected.placement_kinds &= other.placement_kinds;
    let maximum = selected.maximum_byte_length.min(other.maximum_byte_length);
    selected.maximum_byte_length = maximum - maximum % selected.byte_length_granularity;
    if selected.minimum_alignment > selected.maximum_alignment || selected.maximum_byte_length == 0
    {
        return Err(UNSUPPORTED);
    }
    Ok(())
}

fn intersect_profile(
    selected: &mut amdf_memory_profile_t,
    other: &amdf_memory_profile_t,
) -> Result<(), u64> {
    if selected.memory_class != other.memory_class
        || selected.external_memory_support_count != other.external_memory_support_count
    {
        return Err(UNSUPPORTED);
    }
    intersect_construction(&mut selected.allocation, other.allocation)?;
    intersect_construction(&mut selected.registration, other.registration)?;
    intersect_construction(&mut selected.import, other.import)?;
    selected.roles &= other.roles;
    selected.guaranteed_flags &= other.guaranteed_flags;
    selected.supported_flags &= other.supported_flags;
    selected.host_mapping.maximum_byte_length = selected
        .host_mapping
        .maximum_byte_length
        .min(other.host_mapping.maximum_byte_length);
    selected.host_mapping.byte_offset_granularity = common_multiple(
        selected.host_mapping.byte_offset_granularity,
        other.host_mapping.byte_offset_granularity,
    )?;
    selected.host_mapping.byte_length_granularity = common_multiple(
        selected.host_mapping.byte_length_granularity,
        other.host_mapping.byte_length_granularity,
    )?;
    selected.host_mapping.supported_access &= other.host_mapping.supported_access;
    selected.host_mapping.features &= other.host_mapping.features;
    for (selected, other) in selected
        .external_memory_support
        .iter_mut()
        .zip(&other.external_memory_support)
        .take(other.external_memory_support_count as usize)
    {
        if selected.r#type != other.r#type || selected.provenance.words != other.provenance.words {
            return Err(UNSUPPORTED);
        }
        selected.flags &= other.flags;
        selected.source_offset_alignment = common_multiple(
            selected.source_offset_alignment,
            other.source_offset_alignment,
        )?;
        selected.byte_length_alignment =
            common_multiple(selected.byte_length_alignment, other.byte_length_alignment)?;
        selected.maximum_byte_length = selected.maximum_byte_length.min(other.maximum_byte_length);
    }
    Ok(())
}

#[allow(
    clippy::too_many_lines,
    reason = "one transaction validates and intersects all admitted endpoint profiles"
)]
unsafe fn joint_profile(
    scope: &Scope,
    ordinal: u32,
    accesses: &[ProfileAccess<'_>],
) -> Result<
    (
        amdf_memory_profile_t,
        OwnerVec<amdf_memory_access_capabilities_t>,
    ),
    u64,
> {
    if ordinal >= unsafe { scope_count(scope) } {
        return Err(RANGE);
    }
    if accesses.is_empty() {
        let (profile, _) = unsafe {
            profile(
                scope,
                ordinal,
                None,
                amdf_memory_access_requirements_t::default(),
                false,
            )?
        };
        return Ok((
            profile,
            OwnerVec::with_allocator(unsafe { (*scope.instance).host_allocator }),
        ));
    }
    if !scope.endpoint.is_null() {
        let owner = unsafe { &*scope.endpoint };
        if !accesses
            .iter()
            .any(|access| access.endpoint.native.id == owner.native.id)
        {
            return Err(UNSUPPORTED);
        }
    }
    if !uniform_requirements(accesses.iter().map(|access| access.request)) {
        return Err(UNSUPPORTED);
    }
    let mut capabilities =
        OwnerVec::try_with_allocator(unsafe { (*scope.instance).host_allocator }, accesses.len())
            .map_err(|_| EXHAUSTED)?;
    let mut selected = None;
    let mut common = (0, u64::MAX);
    for access in accesses {
        let (candidate, mut capability) = unsafe {
            profile(
                scope,
                ordinal,
                Some(access.endpoint),
                access.request,
                access.import_supported,
            )?
        };
        if access.address_range.0 > access.address_range.1 {
            return Err(UNSUPPORTED);
        }
        capability.device_address.minimum_address = access.address_range.0;
        capability.device_address.maximum_address = access.address_range.1;
        common.0 = common.0.max(access.address_range.0);
        common.1 = common.1.min(access.address_range.1);
        if let Some(selected) = &mut selected {
            intersect_profile(selected, &candidate)?;
        } else {
            selected = Some(candidate);
        }
        capabilities.try_push(capability).map_err(|_| EXHAUSTED)?;
    }
    if common.0 > common.1 {
        return Err(UNSUPPORTED);
    }
    let mut selected = selected.ok_or(INTERNAL)?;
    let registered = selected.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER != 0;
    let limits = if registered {
        selected.registration
    } else {
        selected.allocation
    };
    let unit = limits.native_byte_length_granularity;
    if !unit.is_power_of_two() {
        return Err(UNSUPPORTED);
    }
    let first = common
        .0
        .checked_add(unit - 1)
        .map(|value| value & !(unit - 1))
        .ok_or(UNSUPPORTED)?;
    let envelope = common
        .1
        .checked_sub(first)
        .ok_or(UNSUPPORTED)?
        .saturating_add(1);
    if accesses.iter().any(|access| !access.import_supported)
        || accesses
            .iter()
            .any(|access| access.endpoint.native.id != accesses[0].endpoint.native.id)
    {
        selected.roles &= !AMDF_MEMORY_PROFILE_ROLE_IMPORT;
        selected.import = amdf_memory_construction_capabilities_t::default();
        for support in selected
            .external_memory_support
            .iter_mut()
            .take(selected.external_memory_support_count as usize)
        {
            support.flags &= !AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT;
        }
    }
    let maximum = logical_maximum(
        envelope,
        unit,
        limits.byte_length_granularity,
        registered.then_some(limits.registered_host_pointer_alignment),
    )?;
    constrain_profile(&mut selected, maximum);
    let address_bit_count = u64::BITS - common.1.leading_zeros();
    for capability in &mut capabilities {
        capability.device_address.address_bit_count = address_bit_count;
        capability.device_address.minimum_address = common.0;
        capability.device_address.maximum_address = common.1;
    }
    Ok((selected, capabilities))
}

unsafe fn scopes(
    scope: *const Scope,
    available: u32,
    capacity: u32,
    out: *mut *mut amdf_memory_scope_t,
    count: *mut u32,
) -> Result<(), u64> {
    output_pointer(count)?;
    output_array(out, capacity)?;
    if capacity != 0 && available != 0 {
        unsafe { out.write(scope.cast_mut().cast()) };
    }
    unsafe { count.write(available) };
    if capacity < available {
        Err(SMALL)
    } else {
        Ok(())
    }
}

pub(crate) unsafe extern "C" fn instance_scopes(
    pointer: *mut amdf_instance_t,
    capacity: u32,
    out: *mut *mut amdf_memory_scope_t,
    count: *mut u32,
) -> u64 {
    crate::support::boundary(|| unsafe {
        object(pointer.cast::<Instance>())?;
        scopes(
            &raw const (*pointer.cast::<Instance>()).scope,
            1,
            capacity,
            out,
            count,
        )
    })
}

pub(crate) unsafe extern "C" fn device_scopes(
    pointer: *mut amdf_device_t,
    capacity: u32,
    out: *mut *mut amdf_memory_scope_t,
    count: *mut u32,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let device = object(pointer.cast::<Device>())?;
        let endpoint = &*device.endpoint;
        scopes(
            &raw const endpoint.scope,
            u32::from(endpoint.native.local_memory_bytes != 0),
            capacity,
            out,
            count,
        )
    })
}

pub(crate) unsafe extern "C" fn scope_info(
    pointer: *mut amdf_memory_scope_t,
    out: *mut amdf_memory_scope_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let scope = object(pointer.cast::<Scope>())?;
        let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO)?;
        out.publish(amdf_memory_scope_info_t {
            kind: if scope.endpoint.is_null() {
                AMDF_MEMORY_SCOPE_KIND_SYSTEM
            } else {
                AMDF_MEMORY_SCOPE_KIND_LOCAL
            },
            memory_profile_count: scope_count(scope),
            physical_endpoint_id: if scope.endpoint.is_null() {
                amdf_endpoint_id_t::default()
            } else {
                endpoint_id((*scope.endpoint).native.id)
            },
            ..Default::default()
        });
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn device_profile(
    pointer: *mut amdf_memory_scope_t,
    ordinal: u32,
    count: u32,
    accesses: *const amdf_memory_device_access_t,
    out: *mut amdf_memory_profile_t,
    caps: *mut amdf_memory_access_capabilities_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let scope = object(pointer.cast::<Scope>())?;
        (*scope.instance).require_current_process()?;
        let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE)?;
        let accesses = array(accesses, count)?;
        output_array(caps, count)?;
        if !unique_devices(accesses) {
            return Err(INVALID);
        }
        for (index, access) in accesses.iter().enumerate() {
            requirements(access.requirements)?;
            let device = object(access.device.cast::<Device>())?;
            if (*device.endpoint).instance != scope.instance {
                return Err(INVALID);
            }
            output(
                caps.add(index),
                AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
            )?;
        }
        if !uniform_requirements(accesses.iter().map(|access| access.requirements)) {
            return Err(UNSUPPORTED);
        }
        let mut staged =
            OwnerVec::try_with_allocator((*scope.instance).host_allocator, count as usize)
                .map_err(|_| EXHAUSTED)?;
        for access in accesses {
            let device = object(access.device.cast::<Device>())?;
            staged
                .try_push(ProfileAccess {
                    endpoint: &*device.endpoint,
                    request: access.requirements,
                    address_range: device.native.address_range(),
                    import_supported: external_transport::supports_system_import(&device.native),
                })
                .map_err(|_| EXHAUSTED)?;
        }
        let (profile, capabilities) = joint_profile(scope, ordinal, &staged)?;
        for (index, capability) in capabilities.iter().copied().enumerate() {
            output(
                caps.add(index),
                AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
            )?
            .publish(capability);
        }
        out.publish(profile);
        Ok(())
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "admission, acquisition, and publication share cleanup ownership"
)]
pub(crate) unsafe extern "C" fn create(
    pointer: *mut amdf_memory_scope_t,
    info: *const amdf_memory_create_info_t,
    out: *mut *mut amdf_memory_t,
    out_release_info: *mut amdf_memory_release_info_t,
) -> u64 {
    crate::support::boundary(|| {
        if !out_release_info.is_null() {
            output_pointer(out_release_info)?;
            // SAFETY: A valid optional output is written on every call.
            unsafe { publish_release(out_release_info, amdf_memory_release_info_t::default()) };
        }
        // SAFETY: The C caller retains this provider-owned scope for the call.
        let scope = unsafe { object(pointer.cast::<Scope>()) }?;
        // SAFETY: A live scope borrows its creating instance.
        unsafe { &*scope.instance }.require_current_process()?;
        output_pointer(out)?;
        // SAFETY: The caller supplies a readable descriptor; input validates
        // its prefix before copying the complete record.
        let create = unsafe { input(info, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO) }?;
        if create.required_flags & !BACKING_FLAGS != 0
            || create.byte_length == 0
            || create.reserved != 0
            || create.placement.kind > AMDF_MEMORY_PLACEMENT_KIND_HOST_NUMA
            || (create.placement.kind == AMDF_MEMORY_PLACEMENT_KIND_NONE
                && create.placement.node != 0)
            || (create.minimum_alignment != 0 && !create.minimum_alignment.is_power_of_two())
        {
            return Err(INVALID);
        }
        // SAFETY: The admitted descriptor supplies a live access array.
        let accesses = unsafe { array(create.accesses, create.access_count) }?;
        // SAFETY: The live scope borrows its parent instance.
        let instance = unsafe { object(scope.instance) }?;
        if instance.closing || !instance.native.is_current_process() {
            return Err(PRECONDITION);
        }
        if !unique_devices(accesses) {
            return Err(INVALID);
        }
        for access in accesses {
            requirements(access.requirements)?;
            // SAFETY: Each requested handle remains live for the call.
            let device = unsafe { object(access.device.cast::<Device>()) }?;
            // SAFETY: A live device borrows its endpoint.
            if unsafe { object(device.endpoint) }?.instance != scope.instance {
                return Err(INVALID);
            }
        }
        if !uniform_requirements(accesses.iter().map(|access| access.requirements)) {
            return Err(UNSUPPORTED);
        }
        let mut staged = OwnerVec::try_with_allocator(instance.host_allocator, accesses.len())
            .map_err(|_| EXHAUSTED)?;
        for access in accesses {
            // SAFETY: These handles were admitted above and remain live.
            let device = unsafe { object(access.device.cast::<Device>()) }?;
            // SAFETY: A live device borrows its endpoint for its entire lifetime.
            let endpoint = unsafe { object(device.endpoint) }?;
            staged
                .try_push(ProfileAccess {
                    endpoint,
                    request: access.requirements,
                    address_range: device.native.address_range(),
                    import_supported: external_transport::supports_system_import(&device.native),
                })
                .map_err(|_| EXHAUSTED)?;
        }
        // SAFETY: Scope and staged endpoint borrows remain live through
        // profile selection.
        let (profile, capabilities) =
            unsafe { joint_profile(scope, create.memory_profile_ordinal, &staged) }?;
        let local = profile.memory_class == AMDF_MEMORY_CLASS_LOCAL;
        let owner = if scope.endpoint.is_null() {
            None
        } else {
            // SAFETY: A non-null scope endpoint is owned by the live instance.
            Some(unsafe { object(scope.endpoint) }?.native.id)
        };
        let owner_index = physical_owner_index(
            local,
            owner,
            staged.iter().map(|access| access.endpoint.native.id),
        )?;
        let device = accesses
            .get(owner_index)
            .map(|access| {
                // SAFETY: The requested device was admitted above.
                unsafe { object(access.device.cast::<Device>()) }
            })
            .transpose()?;
        let request = accesses
            .first()
            .map_or(amdf_memory_access_requirements_t::default(), |a| {
                a.requirements
            });
        let registered = profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER != 0;
        if create.placement.kind != AMDF_MEMORY_PLACEMENT_KIND_NONE {
            return Err(UNSUPPORTED);
        }
        if registered == create.registered_host_pointer.is_null() {
            return Err(INVALID);
        }
        if registered {
            if create.registered_host_cacheability
                != profile.registration.registered_host_cacheability
            {
                return Err(UNSUPPORTED);
            }
        } else if create.registered_host_cacheability != AMDF_HOST_CACHEABILITY_UNKNOWN {
            return Err(INVALID);
        }
        let limits = if registered {
            profile.registration
        } else {
            profile.allocation
        };
        if create.required_flags & !profile.supported_flags != 0
            || create.byte_length > limits.maximum_byte_length
            || create.minimum_alignment > limits.maximum_alignment
        {
            return Err(UNSUPPORTED);
        }
        let alignment = create.minimum_alignment.max(limits.minimum_alignment);
        let page = limits.native_byte_length_granularity;
        let (registered_address, mut source_byte_offset, mut native_length) = if registered {
            let address = create.registered_host_pointer as usize;
            let requested_alignment = usize::try_from(alignment).map_err(|_| RANGE)?;
            if address % requested_alignment != 0
                || address
                    .checked_add(usize::try_from(create.byte_length).map_err(|_| RANGE)?)
                    .is_none()
            {
                return Err(INVALID);
            }
            let (source_byte_offset, native_length) =
                native_extent(create.byte_length, Some(address as u64), limits)?;
            (Some(address), source_byte_offset, native_length)
        } else {
            let (_, length) = native_extent(create.byte_length, None, limits)?;
            (None, 0, length)
        };
        let slot = instance.owner_slot::<Memory>()?;
        let mut access_records =
            OwnerVec::try_with_allocator(instance.host_allocator, accesses.len())
                .map_err(|_| EXHAUSTED)?;
        let mut peer_devices =
            OwnerVec::try_with_allocator(instance.host_allocator, accesses.len().saturating_sub(1))
                .map_err(|_| EXHAUSTED)?;
        for (index, access) in accesses.iter().enumerate() {
            if index == owner_index {
                continue;
            }
            // SAFETY: The peer handle was admitted above and remains live.
            let peer = unsafe { object(access.device.cast::<Device>()) }?;
            peer_devices.try_push(&peer.native).map_err(|_| EXHAUSTED)?;
        }
        // Arbitrary caller storage has no identity. A provider-owned host
        // view can reuse the known BO and its already-established VM access.
        let mut physical_backing_id = if registered {
            amdf_physical_memory_id_t::default()
        } else {
            amdf_physical_memory_id_t {
                words: [scope.instance as u64, next_id(&instance.ids)?],
            }
        };
        let visible = profile.guaranteed_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE != 0;
        let alias = if registered && device.is_some() {
            // SAFETY: The instance owns the locked host-backing list; admitted
            // device handles remain live while the borrow is acquired.
            unsafe {
                find_host_alias(
                    instance,
                    registered_address.ok_or(INTERNAL)?,
                    create.byte_length,
                    accesses,
                )
            }?
        } else {
            None
        };
        let (backing, host, cacheability) = if let Some((borrow, offset)) = alias {
            // SAFETY: the registered child keeps the listed source live.
            let source = unsafe { &*borrow.source };
            source_byte_offset = source
                .info
                .source_byte_offset
                .checked_add(offset)
                .ok_or(RANGE)?;
            native_length = source.info.native_allocation_byte_length;
            physical_backing_id = source.info.physical_backing_id;
            for (index, (requested, capability)) in accesses.iter().zip(&capabilities).enumerate() {
                // SAFETY: The requested device remains live throughout creation.
                let access_device = unsafe { &*requested.device.cast::<Device>() };
                let existing = source
                    .access
                    .iter()
                    .find(|existing| {
                        // SAFETY: source access devices outlive source memory.
                        let owner = unsafe { &*existing.device };
                        access_device.native.shares_address_domain(&owner.native)
                            && existing.info.access == requested.requirements.access
                            && existing.info.reset_epoch == access_device.current_reset_epoch()
                    })
                    .ok_or(INTERNAL)?;
                let address = existing.address.checked_add(offset).ok_or(RANGE)?;
                access_records
                    .try_push(Access {
                        device: requested.device.cast(),
                        address,
                        info: amdf_memory_access_info_t {
                            ordinal: u32::try_from(index).map_err(|_| INTERNAL)?,
                            access: requested.requirements.access,
                            device_id: access_device.id,
                            flags: capability.guaranteed_flags | requested.requirements.flags,
                            atomic_operations_32: capability.atomic_operations_32,
                            atomic_operations_64: capability.atomic_operations_64,
                            address_domain_ordinal: capability
                                .device_address
                                .address_domain_ordinal,
                            address_kinds: capability.address_kinds,
                            reset_epoch: access_device.current_reset_epoch(),
                            ..Default::default()
                        },
                    })
                    .map_err(|_| EXHAUSTED)?;
            }
            (
                Backing::Alias { _borrow: borrow },
                registered_address,
                source.cacheability,
            )
        } else if let Some(device) = device {
            let kind = if local {
                memory::device::MemoryKind::DeviceLocal {
                    host_visible: visible,
                    coherent: false,
                    extended_coherent: false,
                    uncached: false,
                    contiguous: false,
                }
            } else {
                memory::device::MemoryKind::System
            };
            let permissions =
                memory::device::DeviceAccess::from_bits(request.access).ok_or(UNSUPPORTED)?;
            let (native_allocation, rollback) =
                memory::device::with_rollback_report(!registered, || {
                    if let Some(address) = registered_address {
                        // SAFETY: The AMDF REGISTER owner borrows the caller's
                        // host pages until native destruction succeeds. An
                        // uncertain native result retains that owner.
                        unsafe {
                            device.native.register_host_range_with_peers(
                                peer_devices.as_slice(),
                                address,
                                memory::device::HostMappingPolicy::Fine,
                                create.byte_length,
                                alignment.max(page),
                                permissions,
                            )
                        }
                    } else {
                        device
                            .native
                            .allocate_with_peers(
                                peer_devices.as_slice(),
                                kind,
                                native_length,
                                alignment.max(page),
                                permissions,
                            )
                            .map(|allocation| (allocation, 0))
                    }
                });
            if native_allocation.is_err() {
                if let Some(error) = rollback.status {
                    // SAFETY: The optional record remains writable until return.
                    unsafe {
                        publish_release(
                            out_release_info,
                            amdf_memory_release_info_t {
                                status: device.native_error_status(&error),
                                abandoned_byte_length: rollback.abandoned_byte_length,
                            },
                        );
                    }
                }
            }
            let (native_allocation, registration_offset) =
                native_allocation.map_err(|e| device.native_error_status(&e))?;
            if registered {
                source_byte_offset = native_allocation
                    .backing_offset()
                    .unwrap_or(0)
                    .checked_add(registration_offset)
                    .ok_or(RANGE)?;
                native_length = native_allocation.info().native_size;
            }
            let mut native_allocation = PendingGpu::new(
                native_allocation,
                std::ptr::from_ref(device).cast_mut(),
                !registered,
                out_release_info,
            );
            if !registered && !local {
                native_allocation
                    .allocation_mut()?
                    .bind_local_backing_identity(memory::device::LocalBackingIdentity {
                        namespace: physical_backing_id.words[0],
                        object: physical_backing_id.words[1],
                    })
                    .map_err(|error| device.native_error_status(&error))?;
            }
            let addresses = native_allocation.allocation()?.info();
            if registered && addresses.physical_backing_id != [0; 2] {
                physical_backing_id.words = addresses.physical_backing_id;
            }
            if profile.guaranteed_flags & AMDF_MEMORY_FLAG_SHAREABLE != 0 {
                let dma_buf = external_transport::export(native_allocation.allocation()?)
                    .map_err(|error| device.native_error_status(&error))?;
                physical_backing_id.words = dma_buf.info().physical_backing_id;
            }
            for (index, (requested, capability)) in accesses.iter().zip(&capabilities).enumerate() {
                // SAFETY: The requested device remains live throughout creation.
                let access_device = unsafe { &*requested.device.cast::<Device>() };
                let address = native_allocation
                    .allocation()?
                    .device_address(&access_device.native)
                    .map_err(|error| access_device.native_error_status(&error))?
                    .checked_add(registration_offset)
                    .ok_or(RANGE)?;
                access_records
                    .try_push(Access {
                        device: requested.device.cast(),
                        address,
                        info: amdf_memory_access_info_t {
                            ordinal: u32::try_from(index).map_err(|_| INTERNAL)?,
                            access: requested.requirements.access,
                            device_id: access_device.id,
                            flags: capability.guaranteed_flags | requested.requirements.flags,
                            atomic_operations_32: capability.atomic_operations_32,
                            atomic_operations_64: capability.atomic_operations_64,
                            address_domain_ordinal: capability
                                .device_address
                                .address_domain_ordinal,
                            address_kinds: capability.address_kinds,
                            reset_epoch: access_device.current_reset_epoch(),
                            ..Default::default()
                        },
                    })
                    .map_err(|_| EXHAUSTED)?;
            }
            (
                Backing::Gpu(native_allocation.into_inner()?),
                registered_address.or(addresses.host_address),
                if local && visible {
                    match device.native.endpoint().host_local_cacheability {
                        Some(memory::device::HostCacheability::WriteBack) => {
                            AMDF_HOST_CACHEABILITY_WRITE_BACK
                        }
                        Some(memory::device::HostCacheability::WriteCombined) => {
                            AMDF_HOST_CACHEABILITY_WRITE_COMBINED
                        }
                        // Profile admission required a qualified mapping recipe.
                        None => return Err(INTERNAL),
                    }
                } else {
                    AMDF_HOST_CACHEABILITY_WRITE_BACK
                },
            )
        } else if registered {
            (
                Backing::Registered,
                registered_address,
                create.registered_host_cacheability,
            )
        } else {
            let (native_allocation, rollback) = memory::device::with_rollback_report(true, || {
                if create.required_flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE != 0 {
                    memory::host::HostAllocation::allocate_remappable(native_length, alignment)
                } else {
                    memory::host::HostAllocation::allocate(native_length, alignment)
                }
            });
            if native_allocation.is_err() {
                if let Some(error) = rollback.status {
                    // SAFETY: The optional record remains writable until return.
                    unsafe {
                        publish_release(
                            out_release_info,
                            amdf_memory_release_info_t {
                                status: native(&error),
                                abandoned_byte_length: rollback.abandoned_byte_length,
                            },
                        );
                    }
                }
            }
            let native_allocation = native_allocation.map_err(|e| native(&e))?;
            let address = native_allocation.info().host_address;
            (
                Backing::Host(native_allocation),
                Some(address),
                AMDF_HOST_CACHEABILITY_WRITE_BACK,
            )
        };
        let actual_alignment = if registered {
            let host_alignment = 1u64 << (host.ok_or(INTERNAL)? as u64).trailing_zeros();
            access_records
                .iter()
                .fold(host_alignment, |alignment, access| {
                    alignment.min(1u64 << access.address.trailing_zeros())
                })
        } else {
            alignment
        };
        let listed_host_backing = !registered
            && !local
            && create.access_count != 0
            && host.is_some()
            && matches!(backing, Backing::Gpu(_));
        let mut owner = slot.write(Memory {
            scope: pointer.cast(),
            host_allocator: instance.host_allocator,
            backing,
            allocated_backing: !registered,
            host,
            access: access_records,
            cacheability,
            info: amdf_memory_info_t {
                memory_profile_ordinal: profile.ordinal,
                memory_class: profile.memory_class,
                access_count: create.access_count,
                flags: profile.guaranteed_flags | create.required_flags,
                source_byte_offset,
                byte_length: create.byte_length,
                alignment: actual_alignment,
                native_allocation_byte_length: native_length,
                native_allocation_granularity: page,
                physical_backing_id,
                ..Default::default()
            },
            children: AtomicU64::new(0),
            freeing: false,
            host_backing_next: AtomicUsize::new(0),
            listed_host_backing,
        });
        if listed_host_backing {
            if let Err(status) = list_host_backing(instance, &mut owner) {
                let result = owner.release_native();
                // SAFETY: The optional output belongs to this call.
                unsafe { publish_release(out_release_info, result) };
                return Err(status);
            }
        }
        // SAFETY: The output slot was validated before native acquisition;
        // ownership transfers only after every creation step succeeds.
        unsafe { out.write(owner.into_raw().cast()) };
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn info(
    pointer: *mut amdf_memory_t,
    out: *mut amdf_memory_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let memory = object(pointer.cast::<Memory>())?;
        output(out, AMDF_STRUCTURE_TYPE_MEMORY_INFO)?.publish(memory.info);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn access_info(
    pointer: *mut amdf_memory_t,
    ordinal: u32,
    out: *mut amdf_memory_access_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let memory = object(pointer.cast::<Memory>())?;
        let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_INFO)?;
        if ordinal >= memory.info.access_count {
            return Err(RANGE);
        }
        out.publish(memory.access.get(ordinal as usize).ok_or(INTERNAL)?.info);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn address(
    pointer: *mut amdf_memory_t,
    ordinal: u32,
    kind: u32,
    out: *mut u64,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let memory = object(pointer.cast::<Memory>())?;
        output_pointer(out)?;
        if ordinal >= memory.info.access_count {
            return Err(RANGE);
        }
        if kind > AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE {
            return Err(INVALID);
        }
        if memory.freeing {
            return Err(PRECONDITION);
        }
        if kind != AMDF_MEMORY_ADDRESS_GPU {
            return Err(UNSUPPORTED);
        }
        out.write(memory.access.get(ordinal as usize).ok_or(INTERNAL)?.address);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn map(
    pointer: *mut amdf_memory_t,
    info: *const amdf_memory_map_info_t,
    out: *mut *mut amdf_host_mapping_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let memory = object(pointer.cast::<Memory>())?;
        (*(*memory.scope).instance).require_current_process()?;
        output_pointer(out)?;
        let map = input(info, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO)?;
        if map.byte_length == 0 || map.flags == 0 || map.flags & !3 != 0 {
            return Err(INVALID);
        }
        if memory.freeing {
            return Err(PRECONDITION);
        }
        range(map.byte_offset, map.byte_length, memory.info.byte_length).map_err(|_| INVALID)?;
        let host = memory.host.ok_or(UNSUPPORTED)?;
        let slot = memory
            .host_allocator
            .with_scope(|| Owned::<Mapping>::try_new_uninit().map_err(|_| EXHAUSTED))?;
        let (maintenance, line, flush, invalidate) =
            host_transitions(&*(*memory.scope).instance, memory.cacheability)?;
        let address = host
            .checked_add(usize::try_from(map.byte_offset).map_err(|_| RANGE)?)
            .ok_or(RANGE)?;
        register(&memory.children)?;
        let owner = slot.write(Mapping {
            owner: MappingOwner::Memory(pointer.cast()),
            info: amdf_host_mapping_info_t {
                flags: 3,
                cacheability: memory.cacheability,
                pointer: address as *mut c_void,
                memory_byte_offset: map.byte_offset,
                byte_length: map.byte_length,
                byte_offset_granularity: 1,
                byte_length_granularity: 1,
                cache_line_size: line,
                features: if memory.info.flags & AMDF_MEMORY_FLAG_HOST_FILE_IO != 0 {
                    AMDF_HOST_MAPPING_FEATURE_FILE_IO
                } else {
                    0
                },
                flush,
                invalidate,
                ..Default::default()
            },
            maintenance,
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn mapping_info(
    pointer: *mut amdf_host_mapping_t,
    out: *mut amdf_host_mapping_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let mapping = object(pointer.cast::<Mapping>())?;
        output(out, AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO)?.publish_from(&raw const mapping.info);
        Ok(())
    })
}

pub(crate) unsafe fn map_reservation(
    pointer: *mut Reservation,
    reservation: &Reservation,
    map: amdf_memory_map_info_t,
    out: *mut *mut amdf_host_mapping_t,
) -> Result<(), u64> {
    if map.byte_length == 0
        || map.flags == 0
        || map.flags & !3 != 0
        || map.byte_offset > reservation.info.byte_length
        || map.byte_length > reservation.info.byte_length - map.byte_offset
    {
        return Err(INVALID);
    }
    let allowed = u32::from(reservation.info.access & AMDF_MEMORY_ACCESS_READ != 0)
        | (u32::from(reservation.info.access & AMDF_MEMORY_ACCESS_WRITE != 0) << 1);
    if map.flags & !allowed != 0 {
        return Err(UNSUPPORTED);
    }
    let instance = unsafe { object(reservation.instance)? };
    let (maintenance, line, flush, invalidate) =
        host_transitions(instance, AMDF_HOST_CACHEABILITY_WRITE_BACK)?;
    let slot = instance.owner_slot::<Mapping>()?;
    let address = reservation
        .info
        .address
        .checked_add(map.byte_offset)
        .ok_or(RANGE)?;
    let address = usize::try_from(address).map_err(|_| RANGE)?;
    register(&reservation.children)?;
    let owner = slot.write(Mapping {
        owner: MappingOwner::Reservation(pointer),
        info: amdf_host_mapping_info_t {
            flags: map.flags,
            cacheability: AMDF_HOST_CACHEABILITY_WRITE_BACK,
            pointer: address as *mut c_void,
            memory_byte_offset: map.byte_offset,
            byte_length: map.byte_length,
            byte_offset_granularity: 1,
            byte_length_granularity: 1,
            cache_line_size: line,
            features: AMDF_HOST_MAPPING_FEATURE_FILE_IO,
            flush,
            invalidate,
            ..Default::default()
        },
        maintenance,
    });
    // SAFETY: The ABI output slot was validated by reservation::map.
    unsafe { out.write(owner.into_raw().cast()) };
    Ok(())
}

pub(crate) unsafe fn host_mapping_source<'a>(
    pointer: *mut amdf_memory_t,
    instance: *mut Instance,
) -> Result<(&'a memory::host::HostAllocation, u64), u64> {
    let source = unsafe { object(pointer.cast::<Memory>())? };
    if unsafe { (*source.scope).instance } != instance {
        return Err(INVALID);
    }
    if source.info.flags & AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE == 0 {
        return Err(UNSUPPORTED);
    }
    let Backing::Host(allocation) = &source.backing else {
        return Err(UNSUPPORTED);
    };
    Ok((allocation, source.info.byte_length))
}

pub(crate) unsafe extern "C" fn cache_control(
    pointer: *mut amdf_host_mapping_t,
    operation: u32,
    offset: u64,
    length: u64,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let mapping = object(pointer.cast::<Mapping>())?;
        mapping.instance().require_current_process()?;
        if operation != AMDF_HOST_CACHE_OPERATION_FLUSH
            && operation != AMDF_HOST_CACHE_OPERATION_INVALIDATE
        {
            return Err(INVALID);
        }
        range(offset, length, mapping.info.byte_length).map_err(|_| INVALID)?;
        if length == 0 {
            return Ok(());
        }
        let required = if operation == AMDF_HOST_CACHE_OPERATION_FLUSH {
            AMDF_MEMORY_MAP_FLAG_WRITE
        } else {
            AMDF_MEMORY_MAP_FLAG_READ
        };
        if mapping.info.flags & required == 0 {
            return Err(PRECONDITION);
        }
        let transition = if operation == AMDF_HOST_CACHE_OPERATION_FLUSH {
            mapping.info.flush
        } else {
            mapping.info.invalidate
        };
        match transition.kind {
            AMDF_CACHE_TRANSITION_KIND_UNKNOWN => return Err(UNSUPPORTED),
            AMDF_CACHE_TRANSITION_KIND_NONE => return Ok(()),
            _ => {}
        }
        let maintenance = mapping.maintenance.ok_or(UNSUPPORTED)?;
        let address = (mapping.info.pointer as usize)
            .checked_add(usize::try_from(offset).map_err(|_| RANGE)?)
            .ok_or(RANGE)?;
        maintenance.apply(address, length).map_err(|e| native(&e))?;
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn unmap(pointer: *mut amdf_host_mapping_t) -> u64 {
    crate::support::boundary(|| unsafe {
        let mapping = object(pointer.cast::<Mapping>())?;
        mapping.instance().require_current_process()?;
        match mapping.owner {
            MappingOwner::Memory(memory) => unregister(&(*memory).children),
            MappingOwner::Reservation(reservation) => unregister(&(*reservation).children),
        }
        drop(Owned::from_raw(pointer.cast::<Mapping>()));
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn destroy(
    pointer: *mut amdf_memory_t,
    out_release_info: *mut amdf_memory_release_info_t,
) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            if !out_release_info.is_null() {
                output_pointer(out_release_info)?;
                publish_release(out_release_info, amdf_memory_release_info_t::default());
            }
            // The caller has ended every device use before requesting free.
            // Child counts cover explicit host mappings and queue scratch.
            // They cannot find memory references embedded in packets or kernel
            // arguments, which remain the caller's synchronization duty.
            let pointer = pointer.cast::<Memory>();
            output_pointer(pointer)?;
            let instance = &*(*(*pointer).scope).instance;
            instance.require_current_process()?;
            // Alias registration can discover this backing through the
            // instance list without holding its public handle. Remove it
            // under the list lock before creating an exclusive Rust borrow.
            // A registered child keeps the source linked and returns BUSY.
            if (*pointer).listed_host_backing {
                unlink_host_backing(instance, pointer)?;
            }
            let memory = object(pointer)?;
            if memory.children.load(Ordering::Acquire) != 0 {
                return Err(BUSY);
            }
            // Child mappings and scratch users have retired; direct access to
            // this memory handle is serialized by the caller.
            let memory = exclusive(pointer)?;
            memory.freeing = true;
            let release = match memory.release_native_for_destroy() {
                Ok(release) => release,
                Err(status) => {
                    memory.freeing = false;
                    publish_release(
                        out_release_info,
                        amdf_memory_release_info_t {
                            status,
                            ..Default::default()
                        },
                    );
                    return Err(status);
                }
            };
            publish_release(out_release_info, release);
            // The ABI consumes the public memory handle even when a native
            // release fails. Dropping the internal owner finishes safe cleanup
            // or lets the native owner retain uncertain state for process exit.
            drop(Owned::from_raw(pointer.cast::<Memory>()));
            if release.status == 0 {
                Ok(())
            } else {
                Err(release.status)
            }
        }
    })
}

/// Resolved queue scratch range and the public memory owner it borrows.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct QueueScratch {
    /// Public allocation borrowed by the queue.
    pub memory: *mut Memory,
    /// Device address of the scratch range.
    pub device_address: u64,
}

/// One queue-lifetime borrow of a public memory resource.
pub(crate) struct QueueScratchBorrow {
    /// Public owner whose child count is decremented on drop.
    memory: *mut Memory,
}

impl Drop for QueueScratchBorrow {
    fn drop(&mut self) {
        // SAFETY: The public borrowing contract keeps memory live while this
        // guard exists, and construction registered exactly one child.
        unsafe { unregister(&(*self.memory).children) };
    }
}

pub(crate) unsafe fn queue_scratch(
    pointer: *mut amdf_memory_t,
    access_ordinal: u32,
    device: *mut Device,
    reset_epoch: u64,
    byte_offset: u64,
    byte_length: u64,
) -> Result<QueueScratch, u64> {
    let memory = unsafe { object(pointer.cast::<Memory>())? };
    if memory.freeing {
        return Err(PRECONDITION);
    }
    if access_ordinal >= memory.info.access_count {
        return Err(RANGE);
    }
    let access = memory.access.get(access_ordinal as usize).ok_or(INTERNAL)?;
    if access.device != device {
        return Err(INVALID);
    }
    if access.info.address_kinds & (1 << AMDF_MEMORY_ADDRESS_GPU) == 0
        || access.info.flags & AMDF_MEMORY_FLAG_DEVICE_ADDRESS == 0
        || access.info.access & (AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE)
            != (AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE)
    {
        return Err(UNSUPPORTED);
    }
    if access.info.reset_epoch != reset_epoch {
        return Err(PRECONDITION);
    }
    range(byte_offset, byte_length, memory.info.byte_length)?;
    let device_address = access.address.checked_add(byte_offset).ok_or(RANGE)?;
    Ok(QueueScratch {
        memory: pointer.cast(),
        device_address,
    })
}

/// Resolves one immutable command descriptor without reading its bytes.
///
/// The caller owns this memory and every indirect dependency until the native
/// submission retires. This lookup only checks the already-established access
/// record and logical range; it does not retain a public memory handle.
pub(crate) unsafe fn kernel_command(
    pointer: *mut amdf_memory_t,
    access_ordinal: u32,
    device: *mut Device,
    reset_epoch: u64,
    byte_offset: u64,
    byte_length: u64,
) -> Result<rocddi::device::gpu::queue::KernelCommand, u64> {
    if byte_length == 0 || byte_offset % 4 != 0 || byte_length % 4 != 0 {
        return Err(INVALID);
    }
    if byte_length > u64::from(u32::MAX) {
        return Err(RANGE);
    }
    let memory = unsafe { object(pointer.cast::<Memory>())? };
    if memory.freeing {
        return Err(PRECONDITION);
    }
    if access_ordinal >= memory.info.access_count {
        return Err(RANGE);
    }
    let access = memory.access.get(access_ordinal as usize).ok_or(INTERNAL)?;
    if access.device != device {
        return Err(INVALID);
    }
    if access.info.reset_epoch != reset_epoch {
        return Err(PRECONDITION);
    }
    if access.info.access & AMDF_MEMORY_ACCESS_EXECUTE == 0
        || access.info.flags & AMDF_MEMORY_FLAG_DEVICE_ADDRESS == 0
        || access.info.address_kinds & (1 << AMDF_MEMORY_ADDRESS_GPU) == 0
    {
        return Err(UNSUPPORTED);
    }
    range(byte_offset, byte_length, memory.info.byte_length)?;
    let device_address = access.address.checked_add(byte_offset).ok_or(RANGE)?;
    if device_address == 0 || device_address % 4 != 0 {
        return Err(INVALID);
    }
    Ok(rocddi::device::gpu::queue::KernelCommand {
        device_address,
        byte_length,
    })
}

pub(crate) unsafe fn borrow_for_queue(pointer: *mut Memory) -> Result<QueueScratchBorrow, u64> {
    let memory = unsafe { object(pointer)? };
    if memory.freeing {
        return Err(PRECONDITION);
    }
    register(&memory.children)?;
    Ok(QueueScratchBorrow { memory: pointer })
}

#[allow(
    clippy::too_many_lines,
    reason = "external admission, native attachment, and move-on-success publication form one transaction"
)]
pub(crate) unsafe extern "C" fn import(
    scope_pointer: *mut amdf_memory_scope_t,
    info: *const amdf_memory_import_info_t,
    external_pointer: *mut amdf_external_memory_t,
    out: *mut *mut amdf_memory_t,
    out_release_info: *mut amdf_memory_release_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        if !out_release_info.is_null() {
            output_pointer(out_release_info)?;
            publish_release(out_release_info, amdf_memory_release_info_t::default());
        }
        let scope = object(scope_pointer.cast::<Scope>())?;
        (*scope.instance).require_current_process()?;
        let info = input(info, AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO)?;
        output_pointer(out)?;
        let external = *object(external_pointer)?;
        if info.required_flags & !BACKING_FLAGS != 0
            || (info.minimum_alignment != 0 && !info.minimum_alignment.is_power_of_two())
        {
            return Err(INVALID);
        }
        validate_external(&external)?;
        let accesses = array(info.accesses, info.access_count)?;
        for access in accesses {
            requirements(access.requirements)?;
            let device = object(access.device.cast::<Device>())?;
            if (*device.endpoint).instance != scope.instance {
                return Err(INVALID);
            }
        }
        if !unique_devices(accesses) {
            return Err(INVALID);
        }
        let instance = &*scope.instance;
        instance.require_current_process()?;
        if instance.closing {
            return Err(PRECONDITION);
        }
        if !uniform_requirements(accesses.iter().map(|access| access.requirements)) {
            return Err(UNSUPPORTED);
        }
        let mut staged = OwnerVec::try_with_allocator(instance.host_allocator, accesses.len())
            .map_err(|_| EXHAUSTED)?;
        let mut devices = OwnerVec::try_with_allocator(instance.host_allocator, accesses.len())
            .map_err(|_| EXHAUSTED)?;
        for access in accesses {
            let device = object(access.device.cast::<Device>())?;
            staged
                .try_push(ProfileAccess {
                    endpoint: &*device.endpoint,
                    request: access.requirements,
                    address_range: device.native.address_range(),
                    import_supported: external_transport::supports_system_import(&device.native),
                })
                .map_err(|_| EXHAUSTED)?;
            devices.try_push(&device.native).map_err(|_| EXHAUSTED)?;
        }
        let (profile, capabilities) = joint_profile(scope, info.memory_profile_ordinal, &staged)?;
        if profile.roles & AMDF_MEMORY_PROFILE_ROLE_IMPORT == 0
            || profile.memory_class != AMDF_MEMORY_CLASS_SYSTEM
            || info.required_flags & !profile.supported_flags != 0
            || external.byte_length > profile.import.maximum_byte_length
            || info.minimum_alignment > profile.import.maximum_alignment
        {
            return Err(UNSUPPORTED);
        }
        let alignment = info.minimum_alignment.max(profile.import.minimum_alignment);
        if external.source_byte_offset % alignment != 0 {
            return Err(UNSUPPORTED);
        }
        let support = external_support(
            &profile,
            external.r#type,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT,
            external.source_byte_offset,
            external.byte_length,
        )?;
        if external.provenance.words != support.provenance.words {
            return Err(UNSUPPORTED);
        }
        let slot = instance.owner_slot::<Memory>()?;
        let mut access_records =
            OwnerVec::try_with_allocator(instance.host_allocator, accesses.len())
                .map_err(|_| EXHAUSTED)?;
        let permissions = memory::device::DeviceAccess::from_bits(accesses[0].requirements.access)
            .ok_or(UNSUPPORTED)?;
        let (native_allocation, rollback) = memory::device::with_rollback_report(false, || {
            external_transport::import_system(
                &instance.native,
                devices.as_slice(),
                &external,
                alignment,
                permissions,
            )
        });
        if native_allocation.is_err() {
            if let Some(error) = rollback.status {
                publish_release(
                    out_release_info,
                    amdf_memory_release_info_t {
                        status: native(&error),
                        abandoned_byte_length: 0,
                    },
                );
            }
        }
        let native_allocation = native_allocation.map_err(|error| {
            let device = &*accesses[0].device.cast::<Device>();
            device.native_error_status(&error)
        })?;
        let native_allocation = PendingGpu::new(
            native_allocation,
            accesses[0].device.cast(),
            false,
            out_release_info,
        );
        let native_info = native_allocation.allocation()?.info();
        if has_identity(external.physical_backing_id.words)
            && external.physical_backing_id.words != native_info.physical_backing_id
        {
            return Err(PRECONDITION);
        }
        let host = native_info.host_address.ok_or(INTERNAL)?;
        for (index, (requested, capability)) in accesses.iter().zip(&capabilities).enumerate() {
            let device = &*requested.device.cast::<Device>();
            let address = native_allocation
                .allocation()?
                .device_address(&device.native)
                .map_err(|error| device.native_error_status(&error))?;
            access_records
                .try_push(Access {
                    device: requested.device.cast(),
                    address,
                    info: amdf_memory_access_info_t {
                        ordinal: u32::try_from(index).map_err(|_| INTERNAL)?,
                        access: requested.requirements.access,
                        device_id: device.id,
                        flags: capability.guaranteed_flags | requested.requirements.flags,
                        atomic_operations_32: capability.atomic_operations_32,
                        atomic_operations_64: capability.atomic_operations_64,
                        address_domain_ordinal: capability.device_address.address_domain_ordinal,
                        address_kinds: capability.address_kinds,
                        reset_epoch: device.current_reset_epoch(),
                        ..Default::default()
                    },
                })
                .map_err(|_| EXHAUSTED)?;
        }
        let mut owner = slot.write(Memory {
            scope: scope_pointer.cast(),
            host_allocator: instance.host_allocator,
            backing: Backing::Gpu(native_allocation.into_inner()?),
            allocated_backing: false,
            info: amdf_memory_info_t {
                memory_profile_ordinal: profile.ordinal,
                memory_class: profile.memory_class,
                access_count: info.access_count,
                flags: profile.guaranteed_flags | info.required_flags,
                source_byte_offset: external.source_byte_offset,
                byte_length: external.byte_length,
                alignment,
                native_allocation_byte_length: native_info.native_size,
                native_allocation_granularity: instance.host_page_size,
                physical_backing_id: amdf_physical_memory_id_t {
                    words: native_info.physical_backing_id,
                },
                ..Default::default()
            },
            access: access_records,
            host: Some(host),
            cacheability: AMDF_HOST_CACHEABILITY_WRITE_BACK,
            children: AtomicU64::new(0),
            freeing: false,
            host_backing_next: AtomicUsize::new(0),
            listed_host_backing: true,
        });
        if let Err(status) = list_host_backing(instance, &mut owner) {
            let result = owner.release_native();
            publish_release(out_release_info, result);
            return Err(status);
        }
        // The independently retained native descriptor permits the exported
        // payload to be released before the public memory handle is published.
        let consumed = external_pointer.replace(amdf_external_memory_t::default());
        if let Some(release) = consumed.release {
            release(
                consumed.release_user_data,
                consumed.r#type,
                consumed.payload,
            );
        }
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn export(
    pointer: *mut amdf_memory_t,
    info: *const amdf_memory_export_info_t,
    out: *mut amdf_external_memory_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let memory = object(pointer.cast::<Memory>())?;
        (*(*memory.scope).instance).require_current_process()?;
        let export = input(info, AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO)?;
        output_pointer(out)?;
        if export.reserved != 0
            || export.external_memory_type == 0
            || export.external_memory_type > AMDF_EXTERNAL_MEMORY_TYPE_DEVICE_ADDRESS
            || export.byte_length == 0
        {
            return Err(INVALID);
        }
        range(
            export.byte_offset,
            export.byte_length,
            memory.info.byte_length,
        )
        .map_err(|_| INVALID)?;
        if memory.freeing {
            return Err(PRECONDITION);
        }
        if memory.info.flags & AMDF_MEMORY_FLAG_SHAREABLE == 0
            || memory.info.memory_profile_ordinal == AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN
        {
            return Err(UNSUPPORTED);
        }
        let access = memory.access.first().ok_or(UNSUPPORTED)?;
        let device = &*access.device;
        let request = amdf_memory_access_requirements_t {
            access: access.info.access,
            flags: access.info.flags,
            address_kinds: access.info.address_kinds,
            ..Default::default()
        };
        let (profile, _) = profile(
            &*memory.scope.cast::<Scope>(),
            memory.info.memory_profile_ordinal,
            Some(&*device.endpoint),
            request,
            external_transport::supports_system_import(&device.native),
        )?;
        if profile.roles & AMDF_MEMORY_PROFILE_ROLE_EXPORT == 0 {
            return Err(UNSUPPORTED);
        }
        let source_offset = memory
            .info
            .source_byte_offset
            .checked_add(export.byte_offset)
            .ok_or(INVALID)?;
        let support = external_support(
            &profile,
            export.external_memory_type,
            AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT,
            source_offset,
            export.byte_length,
        )?;
        let Backing::Gpu(allocation) = &memory.backing else {
            return Err(UNSUPPORTED);
        };
        let dma_buf = external_transport::export(allocation)
            .map_err(|error| device.native_error_status(&error))?;
        let dma_buf_info = dma_buf.info();
        if dma_buf_info.byte_length != memory.info.native_allocation_byte_length
            || dma_buf_info.physical_backing_id != memory.info.physical_backing_id.words
        {
            return Err(INTERNAL);
        }
        out.write(amdf_external_memory_t {
            r#type: export.external_memory_type,
            provenance: support.provenance,
            source_byte_offset: source_offset,
            byte_length: export.byte_length,
            physical_backing_id: memory.info.physical_backing_id,
            ..dma_buf.into_abi()
        });
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn external_release(pointer: *mut amdf_external_memory_t) {
    if pointer.is_null() {
        return;
    }
    let _ = boundary(|| {
        output_pointer(pointer)?;
        // SAFETY: The caller exclusively owns this move-only value. Clear it
        // before invoking its callback so reentrant release cannot repeat it.
        let value = unsafe { pointer.replace(amdf_external_memory_t::default()) };
        if value.r#type != AMDF_EXTERNAL_MEMORY_TYPE_NONE {
            if let Some(release) = value.release {
                unsafe { release(value.release_user_data, value.r#type, value.payload) };
            }
        }
        Ok(())
    });
}

mod cache;
use cache::host_transitions;
#[cfg(all(test, target_arch = "x86_64"))]
use cache::{Site, describe_pair, device_site};
pub(crate) use cache::{pair_info, scope_pair_info};

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "memory_tests.rs"]
mod tests;

#[cfg(test)]
mod capability_geometry_tests {
    use super::*;
    #[test]
    #[allow(clippy::unwrap_used)]
    fn logical_maxima_fit_native_rounding_and_all_permitted_pointer_offsets() {
        assert_eq!(logical_maximum(131_071, 65536, 1, None), Ok(65536));
        assert_eq!(logical_maximum(131_072, 65536, 1, Some(1)), Ok(65537));
        assert_eq!(logical_maximum(131_072, 65536, 64, Some(1)), Ok(65536));
        let base = amdf_memory_construction_capabilities_t {
            maximum_byte_length: 131_072,
            byte_length_granularity: 1,
            registered_host_pointer_alignment: 1,
            minimum_alignment: 1,
            maximum_alignment: 65536,
            native_byte_length_granularity: 65536,
            ..Default::default()
        };
        let strict = amdf_memory_construction_capabilities_t {
            byte_length_granularity: 65536,
            registered_host_pointer_alignment: 65536,
            ..base
        };
        for (mut a, b) in [(base, strict), (strict, base)] {
            intersect_construction(&mut a, b).unwrap();
            assert_eq!(a.byte_length_granularity, 65536);
            assert_eq!(a.registered_host_pointer_alignment, 65536);
            assert_eq!(native_extent(1, Some(65536), a), Err(UNSUPPORTED));
        }
        let incompatible = amdf_memory_construction_capabilities_t {
            native_byte_length_granularity: 4096,
            ..base
        };
        assert_eq!(
            intersect_construction(&mut { base }, incompatible),
            Err(UNSUPPORTED)
        );
    }

    #[test]
    fn construction_uses_provider_units_and_checks_unrounded_logical_lengths() {
        let mut limits = amdf_memory_construction_capabilities_t {
            native_byte_length_granularity: 65536,
            byte_length_granularity: 1,
            ..Default::default()
        };
        assert_eq!(native_extent(1, None, limits), Ok((0, 65536)));
        assert_eq!(native_extent(2, Some(65535), limits), Ok((65535, 131_072)));
        limits.byte_length_granularity = 64;
        assert_eq!(native_extent(1, None, limits), Err(UNSUPPORTED));
        assert_eq!(native_extent(64, None, limits), Ok((0, 65536)));
    }
}
