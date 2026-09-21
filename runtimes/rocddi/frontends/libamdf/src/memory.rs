//! AMDF storage scopes, explicit access and host-view lifetimes.
//!
//! One resource owns its native backing; public parents are borrowed. This
//! provider establishes every requested SYSTEM-memory GPU access before
//! publication. Qualified direct peers may share LOCAL backing. External
//! imports require native permission and cache evidence before advertisement.

use crate::generated::amdf::*;
use crate::instance::{Device, Endpoint, Instance, endpoint_id, supported_endpoint};
use crate::support::*;
use rocddi::host_storage::{Buffer, Owned};
use rocddi::memory;
use rocddi::memory::interop::linux as linux_interop;
use std::ffi::{c_int, c_void};
use std::os::fd::IntoRawFd;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

unsafe extern "C" {
    fn close(descriptor: c_int) -> c_int;
}

const BACKING_FLAGS: u64 =
    AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_SHAREABLE;
const ACCESS_FLAGS: u64 = AMDF_MEMORY_FLAG_QUEUE_STORAGE
    | AMDF_MEMORY_FLAG_HOST_COHERENT
    | AMDF_MEMORY_FLAG_DEVICE_ADDRESS;
const MAX_ALIGNMENT: u64 = 1 << 30;
const MAX_LENGTH: u64 = (isize::MAX as u64) - MAX_ALIGNMENT;

/// Memory-profile namespace embedded in either an instance or an endpoint.
pub(crate) struct Scope {
    pub instance: *mut Instance,
    pub endpoint: *mut Endpoint,
}

/// Native or caller-owned storage underlying one public memory object.
enum Backing {
    Host(memory::HostAllocation),
    Registered,
    Gpu(memory::Allocation),
    Alias { _borrow: SourceBorrow },
}

/// A registration of a provider-owned host view borrows its original backing.
struct SourceBorrow {
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
    device: *mut Device,
    info: amdf_memory_access_info_t,
    address: u64,
}

/// Public AMDF memory owner and all access mappings derived at construction.
///
/// Host mappings and queue-scratch borrows increment `children`, preventing
/// backing destruction while any public child can still reach it. `freeing`
/// prevents a failed or reentrant destruction from restarting the transition.
pub(crate) struct Memory {
    scope: *mut Scope,
    backing: Backing,
    info: amdf_memory_info_t,
    access: Buffer<Access>,
    host: Option<usize>,
    cacheability: u32,
    children: AtomicU64,
    freeing: bool,
    host_backing_next: AtomicUsize,
    listed_host_backing: bool,
}

/// Recognizes the exact provider-owned host range before asking KFD to create
/// a USERPTR BO. KFD cannot register every GPU BO host VMA as a second USERPTR
/// allocation. This cold list is scoped to one instance and retains no backing.
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
struct Mapping {
    memory: *mut Memory,
    info: amdf_host_mapping_info_t,
}

fn rounded(value: u64, granularity: u64) -> Result<u64, u64> {
    value
        .checked_add(granularity - 1)
        .map(|n| n & !(granularity - 1))
        .ok_or(RANGE)
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

unsafe extern "C" fn close_dma_buf(
    _user_data: *mut c_void,
    external_type: u32,
    payload: amdf_external_memory_payload_t,
) {
    if external_type != AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD {
        return;
    }
    let descriptor = unsafe { payload.file_descriptor };
    if let Ok(descriptor) = i32::try_from(descriptor) {
        // SAFETY: AMDF export transferred ownership of this descriptor to the
        // external-memory value, whose release callback runs at most once.
        let _ = unsafe { close(descriptor) };
    }
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
    if let Some(endpoint) = endpoint {
        if endpoint.instance != scope.instance {
            return Err(INVALID);
        }
        supported_endpoint(endpoint)?;
        if local {
            let owner = unsafe { &*scope.endpoint };
            if !endpoint.native.can_access_local_memory(&owner.native) {
                return Err(UNSUPPORTED);
            }
        }
    }
    // Public VRAM has a different capacity from private VRAM on partial-BAR
    // devices. Separate profiles keep both placement and host-map limits exact.
    let host_visible = !local || ordinal == 1;
    let instance = unsafe { &*scope.instance };
    if instance.closing {
        return Err(PRECONDITION);
    }
    if registered
        && endpoint
            .is_some_and(|endpoint| !instance.native.supports_host_registration(&endpoint.native))
    {
        return Err(UNSUPPORTED);
    }
    let page = instance.host_page_size;
    // Provider-owned host-visible storage uses a qualified cache recipe.
    // Registration can still expose borrowed pages without knowing their cache
    // mode; AMDF represents unavailable maintenance separately from mapping.
    if host_visible
        && (!registered || endpoint.is_some())
        && (instance.host_cache_line.is_none()
            || (local && unsafe { (*scope.endpoint).native.host_local_cacheability }.is_none()))
    {
        return Err(UNSUPPORTED);
    }
    let backing_flags = if local {
        AMDF_MEMORY_FLAG_DEVICE_LOCAL
    } else {
        0
    } | if host_visible {
        AMDF_MEMORY_FLAG_HOST_VISIBLE
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
    let construction = amdf_memory_construction_capabilities_t {
        maximum_byte_length: maximum,
        byte_length_granularity: 1,
        registered_host_pointer_alignment: u64::from(registered),
        registered_host_cacheability: if registered {
            AMDF_HOST_CACHEABILITY_WRITE_BACK
        } else {
            AMDF_HOST_CACHEABILITY_UNKNOWN
        },
        reserved: 0,
        minimum_alignment: if registered { 1 } else { page },
        maximum_alignment: MAX_ALIGNMENT,
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
            },
        host_mapping: if host_visible {
            amdf_host_mapping_capabilities_t {
                maximum_byte_length: maximum,
                byte_offset_granularity: 1,
                byte_length_granularity: 1,
                supported_access: AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
                reserved: 0,
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
            memory::DeviceAccess::from_bits(request.access).ok_or(UNSUPPORTED)?;
        let flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS
            | if local {
                0
            } else {
                AMDF_MEMORY_FLAG_HOST_COHERENT
            };
        // KFD always installs readable GPU PTEs; WRITE and EXECUTE are
        // independent additions. Queue storage uses these same native BOs.
        let supported_flags = flags | AMDF_MEMORY_FLAG_QUEUE_STORAGE;
        if !requested_access.contains(memory::DeviceAccess::READ)
            || !endpoint
                .native
                .supported_permissions
                .contains(requested_access)
            || request.flags & !supported_flags != 0
            || request.address_kinds & !1 != 0
        {
            return Err(UNSUPPORTED);
        }
        capabilities = amdf_memory_access_capabilities_t {
            guaranteed_access: AMDF_MEMORY_ACCESS_READ,
            supported_access: endpoint.native.supported_permissions.bits(),
            guaranteed_flags: flags,
            supported_flags,
            address_kinds: 1,
            device_address: amdf_memory_address_capabilities_t {
                address_domain_ordinal: 0,
                address_bit_count: endpoint.native.address_bit_count,
                minimum_address: endpoint.native.minimum_address,
                maximum_address: endpoint.native.maximum_address,
                minimum_alignment: if registered { 1 } else { page },
            },
            ..Default::default()
        };
        if supports_dma_buf_export(local, registered, true) {
            p.roles |= AMDF_MEMORY_PROFILE_ROLE_EXPORT;
            if import_supported {
                p.roles |= AMDF_MEMORY_PROFILE_ROLE_IMPORT;
                p.import = construction;
            }
            p.guaranteed_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
            p.supported_flags |= AMDF_MEMORY_FLAG_SHAREABLE;
            p.external_memory_support_count = 1;
            p.external_memory_support[0] = amdf_external_memory_support_t {
                r#type: AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
                flags: AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_EXPORT
                    | if import_supported {
                        AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT
                    } else {
                        0
                    }
                    | AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_SOURCE_OFFSET
                    | AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_CROSS_PROCESS,
                source_offset_alignment: page,
                byte_length_alignment: 1,
                maximum_byte_length: maximum,
                ..Default::default()
            };
        }
    }
    Ok((p, capabilities))
}

/// Validated per-device requirements used while composing a joint profile.
#[derive(Clone, Copy)]
struct ProfileAccess<'a> {
    endpoint: &'a Endpoint,
    request: amdf_memory_access_requirements_t,
    address_range: (u64, u64),
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

unsafe fn joint_profile(
    scope: &Scope,
    ordinal: u32,
    accesses: &[ProfileAccess<'_>],
    allocator: rocddi::host_storage::Allocator,
) -> Result<
    (
        amdf_memory_profile_t,
        Buffer<amdf_memory_access_capabilities_t>,
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
        return Ok((profile, Buffer::new(allocator)));
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
        Buffer::try_with_capacity(accesses.len(), allocator).map_err(|_| EXHAUSTED)?;
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
        selected.get_or_insert(candidate);
        capabilities.try_push(capability).map_err(|_| EXHAUSTED)?;
    }
    if common.0 > common.1 {
        return Err(UNSUPPORTED);
    }
    let page = unsafe { (*scope.instance).host_page_size };
    let envelope = common
        .1
        .checked_sub(common.0)
        .and_then(|extent| extent.checked_add(1))
        .unwrap_or(u64::MAX);
    let native_maximum = envelope - envelope % page;
    let mut selected = selected.ok_or(INTERNAL)?;
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
    let maximum = if selected.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER != 0 {
        native_maximum.saturating_sub(page - 1)
    } else {
        native_maximum
    };
    if maximum == 0 {
        return Err(UNSUPPORTED);
    }
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

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
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
        let allocator = (*scope.instance).allocator;
        let mut staged =
            Buffer::try_with_capacity(count as usize, allocator).map_err(|_| EXHAUSTED)?;
        for access in accesses {
            let device = object(access.device.cast::<Device>())?;
            staged
                .try_push(ProfileAccess {
                    endpoint: &*device.endpoint,
                    request: access.requirements,
                    address_range: device.native.address_range(),
                    import_supported: linux_interop::supports_system_dma_buf_import(&device.native),
                })
                .map_err(|_| EXHAUSTED)?;
        }
        let (profile, capabilities) = joint_profile(scope, ordinal, &staged, allocator)?;
        for (index, capability) in capabilities.into_iter().enumerate() {
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

#[allow(unused_unsafe)]
#[allow(
    clippy::too_many_lines,
    reason = "admission, acquisition, and publication share cleanup ownership"
)]
pub(crate) unsafe extern "C" fn create(
    pointer: *mut amdf_memory_scope_t,
    info: *const amdf_memory_create_info_t,
    out: *mut *mut amdf_memory_t,
) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            let scope = object(pointer.cast::<Scope>())?;
            output_pointer(out)?;
            let create = input(info, AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO)?;
            if create.required_flags & !BACKING_FLAGS != 0
                || create.byte_length == 0
                || create.reserved != 0
                || (create.minimum_alignment != 0 && !create.minimum_alignment.is_power_of_two())
            {
                return Err(INVALID);
            }
            let accesses = array(create.accesses, create.access_count)?;
            let instance = &*scope.instance;
            if !unique_devices(accesses) {
                return Err(INVALID);
            }
            for access in accesses {
                requirements(access.requirements)?;
                let device = object(access.device.cast::<Device>())?;
                if (*device.endpoint).instance != scope.instance {
                    return Err(INVALID);
                }
            }
            if !uniform_requirements(accesses.iter().map(|access| access.requirements)) {
                return Err(UNSUPPORTED);
            }
            let mut staged = Buffer::try_with_capacity(accesses.len(), instance.allocator)
                .map_err(|_| EXHAUSTED)?;
            for access in accesses {
                let device = object(access.device.cast::<Device>())?;
                staged
                    .try_push(ProfileAccess {
                        endpoint: &*device.endpoint,
                        request: access.requirements,
                        address_range: device.native.address_range(),
                        import_supported: linux_interop::supports_system_dma_buf_import(
                            &device.native,
                        ),
                    })
                    .map_err(|_| EXHAUSTED)?;
            }
            let (profile, capabilities) = joint_profile(
                scope,
                create.memory_profile_ordinal,
                &staged,
                instance.allocator,
            )?;
            let local = profile.memory_class == AMDF_MEMORY_CLASS_LOCAL;
            let owner = (!scope.endpoint.is_null()).then(|| (*scope.endpoint).native.id);
            let owner_index = physical_owner_index(
                local,
                owner,
                staged.iter().map(|access| access.endpoint.native.id),
            )?;
            let device = accesses
                .get(owner_index)
                .map(|access| &*access.device.cast::<Device>());
            let request = accesses
                .first()
                .map_or(amdf_memory_access_requirements_t::default(), |a| {
                    a.requirements
                });
            let registered = profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER != 0;
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
            let page = instance.host_page_size;
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
                let source_byte_offset = address as u64 & (page - 1);
                let native_length = rounded(
                    source_byte_offset
                        .checked_add(create.byte_length)
                        .ok_or(RANGE)?,
                    page,
                )?;
                (Some(address), source_byte_offset, native_length)
            } else {
                (None, 0, rounded(create.byte_length, page)?)
            };
            let slot =
                Owned::<Memory>::try_new_uninit(instance.allocator).map_err(|_| EXHAUSTED)?;
            let mut access_records = Buffer::try_with_capacity(accesses.len(), instance.allocator)
                .map_err(|_| EXHAUSTED)?;
            let mut peer_devices =
                Buffer::try_with_capacity(accesses.len().saturating_sub(1), instance.allocator)
                    .map_err(|_| EXHAUSTED)?;
            for (index, access) in accesses.iter().enumerate() {
                if index == owner_index {
                    continue;
                }
                peer_devices
                    .try_push(&(*access.device.cast::<Device>()).native)
                    .map_err(|_| EXHAUSTED)?;
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
                unsafe {
                    find_host_alias(
                        instance,
                        registered_address.ok_or(INTERNAL)?,
                        create.byte_length,
                        accesses,
                    )?
                }
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
                for (index, (requested, capability)) in
                    accesses.iter().zip(&capabilities).enumerate()
                {
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
                let kind = if let Some(address) = registered_address {
                    memory::MemoryKind::RegisteredHost {
                        address,
                        uncached: false,
                    }
                } else if local {
                    memory::MemoryKind::DeviceLocal {
                        host_visible: visible,
                        coherent: false,
                        uncached: false,
                        contiguous: false,
                    }
                } else {
                    memory::MemoryKind::System
                };
                let permissions =
                    memory::DeviceAccess::from_bits(request.access).ok_or(UNSUPPORTED)?;
                let native_allocation = device
                    .native
                    .allocate_with_peers(
                        peer_devices.as_slice(),
                        kind,
                        native_length,
                        alignment.max(page),
                        permissions,
                    )
                    .map_err(|e| native(&e))?;
                let addresses = native_allocation.info();
                if profile.guaranteed_flags & AMDF_MEMORY_FLAG_SHAREABLE != 0 {
                    let dma_buf = linux_interop::export_dma_buf(&native_allocation)
                        .map_err(|error| native(&error))?;
                    physical_backing_id.words = dma_buf.info().physical_backing_id;
                }
                for (index, (requested, capability)) in
                    accesses.iter().zip(&capabilities).enumerate()
                {
                    let access_device = &*requested.device.cast::<Device>();
                    let address = native_allocation
                        .device_address(&access_device.native)
                        .map_err(|error| native(&error))?;
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
                    Backing::Gpu(native_allocation),
                    addresses.host_address,
                    if local && visible {
                        match device.native.endpoint().host_local_cacheability {
                            Some(memory::HostCacheability::WriteBack) => {
                                AMDF_HOST_CACHEABILITY_WRITE_BACK
                            }
                            Some(memory::HostCacheability::WriteCombined) => {
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
                let native_allocation = instance
                    .native
                    .allocate_host(native_length, alignment)
                    .map_err(|e| native(&e))?;
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
                backing,
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
                list_host_backing(instance, &mut owner)?;
            }
            out.write(owner.into_raw().cast());
            Ok(())
        }
    })
}

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn map(
    pointer: *mut amdf_memory_t,
    info: *const amdf_memory_map_info_t,
    out: *mut *mut amdf_host_mapping_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let memory = object(pointer.cast::<Memory>())?;
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
        let slot = Owned::<Mapping>::try_new_uninit((*(*memory.scope).instance).allocator)
            .map_err(|_| EXHAUSTED)?;
        let (line, flush, invalidate) =
            host_transitions(&*(*memory.scope).instance, memory.cacheability)?;
        register(&memory.children)?;
        let owner = slot.write(Mapping {
            memory: pointer.cast(),
            info: amdf_host_mapping_info_t {
                flags: 3,
                cacheability: memory.cacheability,
                pointer: (host + usize::try_from(map.byte_offset).map_err(|_| RANGE)?)
                    as *mut c_void,
                memory_byte_offset: map.byte_offset,
                byte_length: map.byte_length,
                byte_offset_granularity: 1,
                byte_length_granularity: 1,
                cache_line_size: line,
                flush,
                invalidate,
                ..Default::default()
            },
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

#[allow(unused_unsafe)]
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

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn cache_control(
    pointer: *mut amdf_host_mapping_t,
    operation: u32,
    offset: u64,
    length: u64,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let mapping = object(pointer.cast::<Mapping>())?;
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
        memory::host_cache_control(
            mapping.info.pointer as usize + usize::try_from(offset).map_err(|_| RANGE)?,
            length,
            mapping.info.cache_line_size,
        )
        .map_err(|e| native(&e))?;
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn unmap(pointer: *mut amdf_host_mapping_t) -> u64 {
    crate::support::boundary(|| unsafe {
        let mapping = object(pointer.cast::<Mapping>())?;
        unregister(&(*mapping.memory).children);
        drop(Owned::from_raw(pointer.cast::<Mapping>()));
        Ok(())
    })
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn destroy(pointer: *mut amdf_memory_t) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            // The caller has ended every device use before requesting free.
            // Child counts cover explicit host mappings and queue scratch.
            // They cannot find memory references embedded in packets or kernel
            // arguments, which remain the caller's synchronization duty.
            let pointer = pointer.cast::<Memory>();
            output_pointer(pointer)?;
            // Alias registration can discover this backing through the
            // instance list without holding its public handle. Remove it
            // under the list lock before creating an exclusive Rust borrow.
            // A registered child keeps the source linked and returns BUSY.
            if (*pointer).listed_host_backing {
                let instance = &*(*(*pointer).scope).instance;
                unlink_host_backing(instance, pointer)?;
            }
            let memory = exclusive(pointer)?;
            if memory.children.load(Ordering::Acquire) != 0 {
                return Err(BUSY);
            }
            memory.freeing = true;
            let result = match &mut memory.backing {
                Backing::Host(allocation) => allocation.free(),
                Backing::Gpu(allocation) => allocation.free(),
                Backing::Registered | Backing::Alias { .. } => Ok(()),
            };
            // The ABI consumes the public memory handle even when a native
            // release fails. Dropping the internal owner finishes safe cleanup
            // or lets the native owner retain uncertain state for process exit.
            drop(Owned::from_raw(pointer.cast::<Memory>()));
            result.map_err(|error| native(&error))
        }
    })
}

/// Resolved queue scratch range and the public memory owner it borrows.
#[derive(Debug, Eq, PartialEq)]
pub(crate) struct QueueScratch {
    pub memory: *mut Memory,
    pub device_address: u64,
}

/// One queue-lifetime borrow of a public memory resource.
pub(crate) struct QueueScratchBorrow {
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
) -> Result<rocddi::gpu::queue::KernelCommand, u64> {
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
    Ok(rocddi::gpu::queue::KernelCommand {
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

#[allow(unused_unsafe)]
#[allow(
    clippy::too_many_lines,
    reason = "external admission, native attachment, and move-on-success publication form one transaction"
)]
pub(crate) unsafe extern "C" fn import(
    scope_pointer: *mut amdf_memory_scope_t,
    info: *const amdf_memory_import_info_t,
    external_pointer: *mut amdf_external_memory_t,
    out: *mut *mut amdf_memory_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let scope = object(scope_pointer.cast::<Scope>())?;
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
        if instance.closing {
            return Err(PRECONDITION);
        }
        if !uniform_requirements(accesses.iter().map(|access| access.requirements)) {
            return Err(UNSUPPORTED);
        }
        let mut staged =
            Buffer::try_with_capacity(accesses.len(), instance.allocator).map_err(|_| EXHAUSTED)?;
        let mut devices =
            Buffer::try_with_capacity(accesses.len(), instance.allocator).map_err(|_| EXHAUSTED)?;
        for access in accesses {
            let device = object(access.device.cast::<Device>())?;
            staged
                .try_push(ProfileAccess {
                    endpoint: &*device.endpoint,
                    request: access.requirements,
                    address_range: device.native.address_range(),
                    import_supported: linux_interop::supports_system_dma_buf_import(&device.native),
                })
                .map_err(|_| EXHAUSTED)?;
            devices.try_push(&device.native).map_err(|_| EXHAUSTED)?;
        }
        let (profile, capabilities) = joint_profile(
            scope,
            info.memory_profile_ordinal,
            &staged,
            instance.allocator,
        )?;
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
        let descriptor = i32::try_from(external.payload.file_descriptor).map_err(|_| INVALID)?;
        let slot = Owned::<Memory>::try_new_uninit(instance.allocator).map_err(|_| EXHAUSTED)?;
        let mut access_records =
            Buffer::try_with_capacity(accesses.len(), instance.allocator).map_err(|_| EXHAUSTED)?;
        let permissions =
            memory::DeviceAccess::from_bits(accesses[0].requirements.access).ok_or(UNSUPPORTED)?;
        let native_allocation = linux_interop::import_system_dma_buf(
            &instance.native,
            devices.as_slice(),
            descriptor,
            external.source_byte_offset,
            external.byte_length,
            alignment,
            permissions,
        )
        .map_err(|error| native(&error))?;
        let native_info = native_allocation.info();
        if has_identity(external.physical_backing_id.words)
            && external.physical_backing_id.words != native_info.physical_backing_id
        {
            return Err(PRECONDITION);
        }
        let host = native_info.host_address.ok_or(INTERNAL)?;
        for (index, (requested, capability)) in accesses.iter().zip(&capabilities).enumerate() {
            let device = &*requested.device.cast::<Device>();
            let address = native_allocation
                .device_address(&device.native)
                .map_err(|error| native(&error))?;
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
            backing: Backing::Gpu(native_allocation),
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
        list_host_backing(instance, &mut owner)?;
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

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn export(
    pointer: *mut amdf_memory_t,
    info: *const amdf_memory_export_info_t,
    out: *mut amdf_external_memory_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let memory = object(pointer.cast::<Memory>())?;
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
            linux_interop::supports_system_dma_buf_import(&device.native),
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
        let dma_buf = linux_interop::export_dma_buf(allocation).map_err(|error| native(&error))?;
        let dma_buf_info = dma_buf.info();
        if dma_buf_info.byte_length != memory.info.native_allocation_byte_length
            || dma_buf_info.physical_backing_id != memory.info.physical_backing_id.words
        {
            return Err(INTERNAL);
        }
        out.write(amdf_external_memory_t {
            r#type: export.external_memory_type,
            payload: amdf_external_memory_payload_t {
                file_descriptor: i64::from(dma_buf.into_fd().into_raw_fd()),
            },
            provenance: support.provenance,
            source_byte_offset: source_offset,
            byte_length: export.byte_length,
            physical_backing_id: memory.info.physical_backing_id,
            release: Some(close_dma_buf),
            ..Default::default()
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

/// One producer or consumer site used to derive cache-transition requirements.
#[derive(Clone, Copy)]
struct Site {
    memory: *mut Memory,
    host: bool,
    access: u32,
    host_coherent: bool,
    cacheability: u32,
    release: amdf_cache_transition_t,
    acquire: amdf_cache_transition_t,
}

fn host_transitions(
    instance: &Instance,
    cacheability: amdf_host_cacheability_t,
) -> Result<(u32, amdf_cache_transition_t, amdf_cache_transition_t), u64> {
    let (line, flush) = match cacheability {
        AMDF_HOST_CACHEABILITY_WRITE_BACK => {
            let line = instance.host_cache_line.ok_or(UNSUPPORTED)?;
            (
                line,
                amdf_cache_transition_t {
                    kind: AMDF_CACHE_TRANSITION_KIND_RANGE,
                    executor: AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
                    host_operation: AMDF_HOST_CACHE_OPERATION_FLUSH,
                    host_instruction: AMDF_HOST_CACHE_INSTRUCTION_X86_CLFLUSH,
                    host_fence_before: AMDF_HOST_CACHE_FENCE_X86_MFENCE,
                    host_fence_after: AMDF_HOST_CACHE_FENCE_X86_MFENCE,
                    range_granularity: u64::from(line),
                    ..Default::default()
                },
            )
        }
        AMDF_HOST_CACHEABILITY_WRITE_COMBINED => (
            0,
            amdf_cache_transition_t {
                kind: AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                executor: AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT,
                host_operation: AMDF_HOST_CACHE_OPERATION_FLUSH,
                host_fence_after: AMDF_HOST_CACHE_FENCE_X86_MFENCE,
                ..Default::default()
            },
        ),
        AMDF_HOST_CACHEABILITY_UNCACHED => (0, no_transition()),
        _ => (0, amdf_cache_transition_t::default()),
    };
    let mut invalidate = flush;
    if invalidate.executor == AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT {
        invalidate.host_operation = AMDF_HOST_CACHE_OPERATION_INVALIDATE;
    }
    Ok((line, flush, invalidate))
}

fn describe_pair(producer: &Site, consumer: &Site) -> Result<amdf_memory_pair_info_t, u64> {
    // Coherent GPU access can eliminate CPU line maintenance on WB mappings,
    // but not the caller's publication or completion ordering. WC views retain
    // their fences even when the peer snoops host caches.
    let release = if producer.host
        && producer.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK
        && consumer.host_coherent
    {
        no_transition()
    } else {
        producer.release
    };
    let acquire = if consumer.host
        && consumer.cacheability == AMDF_HOST_CACHEABILITY_WRITE_BACK
        && producer.host_coherent
    {
        no_transition()
    } else {
        consumer.acquire
    };
    if producer.access & AMDF_MEMORY_ACCESS_WRITE == 0
        || consumer.access & AMDF_MEMORY_ACCESS_READ == 0
        || release.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN
        || acquire.kind == AMDF_CACHE_TRANSITION_KIND_UNKNOWN
    {
        return Err(UNSUPPORTED);
    }
    Ok(amdf_memory_pair_info_t {
        flags: AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE
            | if release.kind == AMDF_CACHE_TRANSITION_KIND_NONE
                && acquire.kind == AMDF_CACHE_TRANSITION_KIND_NONE
            {
                AMDF_MEMORY_PAIR_FLAG_FIXED_COST_KNOWN
            } else {
                0
            },
        release,
        acquire,
        ..Default::default()
    })
}

fn device_site(
    memory: *mut Memory,
    memory_class: u32,
    flags: u64,
    access: u32,
    family: &amdf_queue_family_info_t,
    local_wc_single_owner: bool,
) -> Site {
    // SYSTEM uses the family-qualified cache protocol. GFX1201 PM4, AQL, and
    // SDMA also qualify host-visible WC LOCAL storage with one owner VM;
    // peer-device LOCAL visibility needs separate evidence.
    let system =
        memory_class == AMDF_MEMORY_CLASS_SYSTEM && flags & AMDF_MEMORY_FLAG_HOST_COHERENT != 0;
    let qualified_local_family = (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
        && family.format_version == AMDF_GPU_PM4_QUEUE_FORMAT_VERSION_1
        && family.format_features == AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR)
        || (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA
            && family.format_version == AMDF_GPU_SDMA_QUEUE_FORMAT_VERSION_1
            && family.format_features
                == AMDF_GPU_SDMA_FORMAT_FEATURE_GCR | AMDF_GPU_SDMA_FORMAT_FEATURE_FENCE_SYSTEM)
        || (family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
            && family.format_version == 1
            && family.format_features == 0);
    let local = memory_class == AMDF_MEMORY_CLASS_LOCAL
        && flags & AMDF_MEMORY_FLAG_HOST_VISIBLE != 0
        && local_wc_single_owner
        && qualified_local_family;
    let qualified = (system || local)
        && family.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL != 0;
    let transition = |operation, supported| {
        if qualified && family.cache_operations & supported != 0 {
            amdf_cache_transition_t {
                kind: AMDF_CACHE_TRANSITION_KIND_GLOBAL,
                executor: AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE,
                operation,
                ..Default::default()
            }
        } else {
            amdf_cache_transition_t::default()
        }
    };
    Site {
        memory,
        host: false,
        access,
        host_coherent: flags & AMDF_MEMORY_FLAG_HOST_COHERENT != 0,
        cacheability: AMDF_HOST_CACHEABILITY_UNKNOWN,
        release: transition(
            AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM,
        ),
        acquire: transition(
            AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
            AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
        ),
    }
}

unsafe fn profile_site(
    scope: &Scope,
    profile: &amdf_memory_profile_t,
    capabilities: &[amdf_memory_access_capabilities_t],
    accesses: &[amdf_memory_device_access_t],
    required_flags: amdf_memory_flags_t,
    registered_host_cacheability: amdf_host_cacheability_t,
    site: amdf_memory_profile_site_t,
) -> Result<Site, u64> {
    if site.reserved != 0 {
        return Err(INVALID);
    }
    match site.kind {
        AMDF_MEMORY_SITE_KIND_DEVICE => {
            let value = unsafe { site.value.device };
            let index = usize::try_from(value.access_ordinal).map_err(|_| RANGE)?;
            let access = accesses.get(index).ok_or(RANGE)?;
            let capability = capabilities.get(index).ok_or(INTERNAL)?;
            let device = unsafe { object(access.device.cast::<Device>())? };
            let family =
                crate::instance::family(device.native.endpoint(), value.queue_family_ordinal)?;
            Ok(device_site(
                std::ptr::null_mut(),
                profile.memory_class,
                capability.guaranteed_flags | access.requirements.flags | required_flags,
                access.requirements.access,
                &family,
                accesses.len() == 1
                    && matches!(
                        device.native.endpoint().host_local_cacheability,
                        Some(memory::HostCacheability::WriteCombined)
                    ),
            ))
        }
        AMDF_MEMORY_SITE_KIND_HOST => {
            let access = unsafe { site.value.host_access };
            if access == 0
                || access & !(AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE) != 0
            {
                return Err(INVALID);
            }
            if required_flags & AMDF_MEMORY_FLAG_HOST_VISIBLE == 0
                || profile.roles & AMDF_MEMORY_PROFILE_ROLE_HOST_MAP == 0
                || access & !profile.host_mapping.supported_access != 0
            {
                return Err(UNSUPPORTED);
            }
            let cacheability = if profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER != 0 {
                registered_host_cacheability
            } else if profile.memory_class == AMDF_MEMORY_CLASS_LOCAL {
                match unsafe { (*scope.endpoint).native.host_local_cacheability } {
                    Some(memory::HostCacheability::WriteBack) => AMDF_HOST_CACHEABILITY_WRITE_BACK,
                    Some(memory::HostCacheability::WriteCombined) => {
                        AMDF_HOST_CACHEABILITY_WRITE_COMBINED
                    }
                    None => return Err(UNSUPPORTED),
                }
            } else {
                AMDF_HOST_CACHEABILITY_WRITE_BACK
            };
            let (_, release, acquire) =
                host_transitions(unsafe { &*scope.instance }, cacheability)?;
            Ok(Site {
                memory: std::ptr::null_mut(),
                host: true,
                access: if access & AMDF_MEMORY_MAP_FLAG_READ != 0 {
                    AMDF_MEMORY_ACCESS_READ
                } else {
                    0
                } | if access & AMDF_MEMORY_MAP_FLAG_WRITE != 0 {
                    AMDF_MEMORY_ACCESS_WRITE
                } else {
                    0
                },
                host_coherent: true,
                cacheability,
                release,
                acquire,
            })
        }
        _ => Err(INVALID),
    }
}

#[allow(unused_unsafe)]
#[allow(
    clippy::too_many_lines,
    reason = "profile qualification validates one atomic caller contract"
)]
pub(crate) unsafe extern "C" fn scope_pair_info(
    pointer: *mut amdf_memory_scope_t,
    query: *const amdf_memory_profile_pair_query_t,
    out: *mut amdf_memory_pair_info_t,
) -> u64 {
    crate::support::boundary(|| unsafe {
        let scope = object(pointer.cast::<Scope>())?;
        let query = input(query, AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY)?;
        let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO)?;
        if query.reserved != 0
            || query.external_memory_reserved != 0
            || query.required_flags & !BACKING_FLAGS != 0
            || query.external_memory_type > AMDF_EXTERNAL_MEMORY_TYPE_COUNT
        {
            return Err(INVALID);
        }
        let accesses = array(query.accesses, query.access_count)?;
        if !unique_devices(accesses) {
            return Err(INVALID);
        }
        let allocator = (*scope.instance).allocator;
        let mut staged =
            Buffer::try_with_capacity(accesses.len(), allocator).map_err(|_| EXHAUSTED)?;
        for access in accesses {
            requirements(access.requirements)?;
            let device = object(access.device.cast::<Device>())?;
            if (*device.endpoint).instance != scope.instance {
                return Err(INVALID);
            }
            staged
                .try_push(ProfileAccess {
                    endpoint: &*device.endpoint,
                    request: access.requirements,
                    address_range: device.native.address_range(),
                    import_supported: linux_interop::supports_system_dma_buf_import(&device.native),
                })
                .map_err(|_| EXHAUSTED)?;
        }
        let (profile, capabilities) =
            joint_profile(scope, query.memory_profile_ordinal, &staged, allocator)?;
        if query.required_flags & !profile.supported_flags != 0 {
            return Err(UNSUPPORTED);
        }
        let registered = profile.roles & AMDF_MEMORY_PROFILE_ROLE_REGISTER != 0;
        if registered {
            if query.registered_host_cacheability
                != profile.registration.registered_host_cacheability
            {
                return Err(UNSUPPORTED);
            }
        } else if query.registered_host_cacheability != AMDF_HOST_CACHEABILITY_UNKNOWN {
            return Err(INVALID);
        }
        if query.external_memory_type == AMDF_EXTERNAL_MEMORY_TYPE_NONE {
            if has_identity(query.external_memory_provenance.words) {
                return Err(INVALID);
            }
        } else {
            let count =
                usize::try_from(profile.external_memory_support_count).map_err(|_| INTERNAL)?;
            let support = profile
                .external_memory_support
                .get(..count)
                .and_then(|values| {
                    values
                        .iter()
                        .find(|value| value.r#type == query.external_memory_type)
                })
                .ok_or(UNSUPPORTED)?;
            if support.flags & AMDF_EXTERNAL_MEMORY_SUPPORT_FLAG_IMPORT == 0 {
                return Err(UNSUPPORTED);
            }
            if support.provenance.words != query.external_memory_provenance.words {
                return Err(UNSUPPORTED);
            }
        }
        let producer = profile_site(
            scope,
            &profile,
            &capabilities,
            accesses,
            query.required_flags,
            query.registered_host_cacheability,
            query.producer,
        )?;
        let consumer = profile_site(
            scope,
            &profile,
            &capabilities,
            accesses,
            query.required_flags,
            query.registered_host_cacheability,
            query.consumer,
        )?;
        out.publish(describe_pair(&producer, &consumer)?);
        Ok(())
    })
}

unsafe fn site(pointer: *const amdf_memory_site_t) -> Result<Site, u64> {
    let site = unsafe { input(pointer, AMDF_STRUCTURE_TYPE_MEMORY_SITE)? };
    if site.reserved != 0 {
        return Err(INVALID);
    }
    match site.kind {
        AMDF_MEMORY_SITE_KIND_HOST => {
            let mapping = unsafe { object(site.value.host_mapping.cast::<Mapping>())? };
            Ok(Site {
                memory: mapping.memory,
                host: true,
                access: if mapping.info.flags & AMDF_MEMORY_MAP_FLAG_READ != 0 {
                    AMDF_MEMORY_ACCESS_READ
                } else {
                    0
                } | if mapping.info.flags & AMDF_MEMORY_MAP_FLAG_WRITE != 0 {
                    AMDF_MEMORY_ACCESS_WRITE
                } else {
                    0
                },
                host_coherent: true,
                cacheability: mapping.info.cacheability,
                release: mapping.info.flush,
                acquire: mapping.info.invalidate,
            })
        }
        AMDF_MEMORY_SITE_KIND_DEVICE => {
            let device = unsafe { site.value.device };
            let memory = unsafe { object(device.memory.cast::<Memory>())? };
            if device.access_ordinal >= memory.info.access_count {
                return Err(RANGE);
            }
            let access = memory
                .access
                .get(device.access_ordinal as usize)
                .ok_or(INTERNAL)?;
            let family = unsafe {
                crate::instance::family(
                    (*access.device).native.endpoint(),
                    device.queue_family_ordinal,
                )?
            };
            Ok(device_site(
                device.memory.cast(),
                memory.info.memory_class,
                memory.info.flags | access.info.flags,
                access.info.access,
                &family,
                memory.info.access_count == 1
                    && memory.cacheability == AMDF_HOST_CACHEABILITY_WRITE_COMBINED,
            ))
        }
        _ => Err(INVALID),
    }
}

#[allow(unused_unsafe)]
pub(crate) unsafe extern "C" fn pair_info(
    producer: *const amdf_memory_site_t,
    consumer: *const amdf_memory_site_t,
    out: *mut amdf_memory_pair_info_t,
) -> u64 {
    crate::support::boundary(|| {
        unsafe {
            let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO)?;
            let producer = site(producer)?;
            let consumer = site(consumer)?;
            if (*(*producer.memory).scope).instance != (*(*consumer.memory).scope).instance {
                return Err(PRECONDITION);
            }
            if producer.memory != consumer.memory {
                let producer_id = (*producer.memory).info.physical_backing_id.words;
                let consumer_id = (*consumer.memory).info.physical_backing_id.words;
                // Distinct resources need positive evidence of shared backing.
                // Unknown identity leaves the relation unsupported; known,
                // different identities prove that its precondition is false.
                if producer_id == [0; 2] || consumer_id == [0; 2] {
                    return Err(UNSUPPORTED);
                }
                if producer_id != consumer_id {
                    return Err(PRECONDITION);
                }
            }
            if (*producer.memory).freeing || (*consumer.memory).freeing {
                return Err(PRECONDITION);
            }
            // Shared backing alone does not tell the caller how to release or
            // acquire device caches. Publish a pair only when both recipes are
            // complete; otherwise leave the caller's output unchanged.
            out.publish(describe_pair(&producer, &consumer)?);
            Ok(())
        }
    })
}

#[cfg(all(test, target_arch = "x86_64"))]
#[path = "memory_tests.rs"]
mod tests;
