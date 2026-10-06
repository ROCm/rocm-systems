// SPDX-License-Identifier: MIT

//! HSA memory pools, pointer metadata, registration, IPC, and virtual memory.
//!
//! rocddi owns native allocations and mappings; this module owns their HSA
//! handles, compatibility metadata, deallocation callbacks, and public status
//! translation. Every lookup accepts both host and device aliases where the HSA
//! contract permits them, while range checks remain overflow-safe and bounded
//! by the original allocation or reservation.
//!
//! Destruction removes public visibility only when ownership can be transferred
//! safely. Failed native cleanup keeps the owning record available for retry,
//! and user callbacks run outside the process-global runtime lock. Linux-only
//! descriptor and AIS operations remain explicit rather than being presented
//! as portable native-handle behavior.

use crate::platform::fd::{AsFd, IntoRawFd};
use std::collections::HashMap;
use std::ffi::{c_char, c_void};
use std::mem::{MaybeUninit, align_of, size_of};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering, fence};
use std::thread;
use std::time::Duration;

use crate::platform::memory as linux_interop;
use crate::platform::memory::{
    KfdIpcMemoryHandle as IpcMemoryHandle, KfdSvmAccess as SvmAccess,
    KfdSvmAttribute as SvmAttribute, KfdSvmLocation as SvmLocation,
};
use rocddi::device::Device;
use rocddi::gpu::{CopyRect, GpuCopySequence};
use rocddi::memory::{
    Allocation, DeviceAccess, MemoryKind, VirtualAddress, VirtualDeviceMapping, VirtualHostMapping,
    VirtualMemory,
};
use rocddi::session::{Session, SessionLifetime};
use rocddi::topology::{MemoryLinkInfo, MemoryLinkType};

use crate::ffi::*;
use crate::runtime::{Runtime, boundary, initialized_mut, lock, map_error};

const GPU_POOL_COARSE: u64 = 1;
const GPU_POOL_FINE: u64 = 2;
const GPU_POOL_GROUP: u64 = 3;
const ALLOC_PCIE: u32 = 1;
const ALLOC_CONTIGUOUS: u32 = 1 << 1;
const ALLOC_EXECUTABLE: u32 = 1 << 2;
const ALLOC_UNCACHED: u32 = 1 << 3;
const ALLOC_FLAGS: u32 = ALLOC_PCIE | ALLOC_CONTIGUOUS | ALLOC_EXECUTABLE | ALLOC_UNCACHED;
const SVM_FLAG_HOST_ACCESS: u32 = 0x01;
const SVM_FLAG_COHERENT: u32 = 0x02;
const SVM_FLAG_HIVE_LOCAL: u32 = 0x04;
const SVM_FLAG_GPU_READ_ONLY: u32 = 0x08;
const SVM_FLAG_GPU_EXECUTE: u32 = 0x10;
const SVM_FLAG_GPU_READ_MOSTLY: u32 = 0x20;
const SVM_MAX_MIGRATION_GRANULARITY: u64 = 18;
const SVM_MAX_ATTRIBUTES: usize = 2044;
const AIS_MAX_TRANSFER_BYTES: u64 = 0x7fff_f000;
const EIO: i32 = 5;
const EOVERFLOW: i32 = 75;

/// Canonical metadata for every address alias of one HSA-visible allocation.
///
/// `agent_base` is always present. `host_base` exists only when CPU access is
/// valid. Range resolution preserves the offset between those aliases instead
/// of assuming that host and GPU virtual addresses are identical.
#[derive(Clone)]
struct PointerDescription {
    pointer_type: u32,
    agent_base: usize,
    host_base: Option<usize>,
    size: usize,
    owner: HsaAgent,
    global_flags: u32,
    registered: bool,
    alloc_flags: u32,
    user_data: usize,
    accessible: Vec<HsaAgent>,
}

impl PointerDescription {
    fn contains(&self, address: usize) -> bool {
        contains(self.agent_base, self.size, address)
            || self
                .host_base
                .is_some_and(|host| contains(host, self.size, address))
    }

    fn starts_at(&self, address: usize) -> bool {
        self.agent_base == address || self.host_base == Some(address)
    }

    fn host_address(&self, address: usize) -> Option<usize> {
        if self
            .host_base
            .is_some_and(|host| contains(host, self.size, address))
        {
            return Some(address);
        }
        let offset = address.checked_sub(self.agent_base)?;
        (offset < self.size)
            .then(|| self.host_base?.checked_add(offset))
            .flatten()
    }

    fn host_range(&self, address: usize, size: usize) -> Option<usize> {
        self.offset_range(address, size)?;
        self.host_address(address)
    }

    fn offset_range(&self, address: usize, size: usize) -> Option<usize> {
        let base = if contains(self.agent_base, self.size, address) {
            self.agent_base
        } else if self
            .host_base
            .is_some_and(|host| contains(host, self.size, address))
        {
            self.host_base?
        } else {
            return None;
        };
        let offset = address.checked_sub(base)?;
        (offset.checked_add(size)? <= self.size).then_some(offset)
    }
}

/// Runtime-owned allocation plus callbacks registered for its destruction.
pub(crate) struct Memory {
    allocation: Allocation,
    description: PointerDescription,
    deallocation_callbacks: Vec<RegisteredDeallocationCallback>,
}

/// Application callback invoked after an allocation loses public ownership.
#[derive(Clone, Copy)]
struct RegisteredDeallocationCallback {
    pointer: usize,
    callback: unsafe extern "C" fn(*mut c_void, *mut c_void),
    user_data: usize,
}

impl Memory {
    fn new(
        allocation: Allocation,
        size: usize,
        owner: HsaAgent,
        global_flags: u32,
        alloc_flags: u32,
        accessible: Vec<HsaAgent>,
    ) -> Self {
        let info = allocation.info();
        let agent_base = info.device_address as usize;
        Self {
            allocation,
            description: PointerDescription {
                pointer_type: POINTER_TYPE_HSA,
                agent_base,
                host_base: info.host_address,
                size,
                owner,
                global_flags,
                registered: true,
                alloc_flags,
                user_data: 0,
                accessible,
            },
            deallocation_callbacks: Vec::new(),
        }
    }

    fn contains(&self, address: usize) -> bool {
        self.description.contains(address)
    }

    fn new_interop(allocation: Allocation, owner: HsaAgent, accessible: Vec<HsaAgent>) -> Self {
        let info = allocation.info();
        let agent_base = info.device_address as usize;
        Self {
            allocation,
            description: PointerDescription {
                pointer_type: POINTER_TYPE_GRAPHICS,
                agent_base,
                host_base: info.host_address,
                size: info.size as usize,
                owner,
                global_flags: POOL_FLAG_COARSE,
                registered: true,
                alloc_flags: POINTER_ALLOC_NONPAGED
                    | if info.host_address.is_some() {
                        POINTER_ALLOC_HOST_ACCESS
                    } else {
                        0
                    },
                user_data: 0,
                accessible,
            },
            deallocation_callbacks: Vec::new(),
        }
    }

    fn new_ipc(
        allocation: Allocation,
        size: usize,
        owner: HsaAgent,
        accessible: Vec<HsaAgent>,
    ) -> Self {
        let info = allocation.info();
        Self {
            allocation,
            description: PointerDescription {
                pointer_type: POINTER_TYPE_IPC,
                agent_base: info.device_address as usize,
                host_base: info.host_address,
                size,
                owner,
                global_flags: POOL_FLAG_COARSE,
                registered: true,
                alloc_flags: POINTER_ALLOC_NONPAGED
                    | if info.host_address.is_some() {
                        POINTER_ALLOC_HOST_ACCESS
                    } else {
                        0
                    },
                user_data: 0,
                accessible,
            },
            deallocation_callbacks: Vec::new(),
        }
    }

    fn free(&mut self) -> Result<(), rocddi::Error> {
        self.allocation.free()
    }
}

/// Host memory registered or pinned for GPU access.
///
/// Some registrations require a native rocddi allocation while already-shared
/// ranges need only the pointer metadata, hence the optional owner.
pub(crate) struct LockedMemory {
    allocation: Option<Allocation>,
    description: PointerDescription,
}

/// Reserved virtual-address interval and the mappings currently occupying it.
pub(crate) struct VmemReservation {
    address: VirtualAddress,
    description: PointerDescription,
    deallocation_callbacks: Vec<RegisteredDeallocationCallback>,
    mappings: usize,
}

impl VmemReservation {
    fn new(address: VirtualAddress, registered: bool) -> Self {
        let info = address.info();
        Self {
            address,
            description: PointerDescription {
                pointer_type: POINTER_TYPE_RESERVED_ADDR,
                agent_base: 0,
                host_base: Some(info.address as usize),
                size: info.size as usize,
                owner: HsaAgent { handle: 0 },
                global_flags: 0,
                registered,
                alloc_flags: 0,
                user_data: 0,
                accessible: Vec::new(),
            },
            deallocation_callbacks: Vec::new(),
            mappings: 0,
        }
    }

    fn contains_range(&self, address: usize, size: usize) -> bool {
        address
            .checked_sub(self.address.info().address as usize)
            .and_then(|offset| offset.checked_add(size))
            .is_some_and(|end| size != 0 && end <= self.description.size)
    }

    fn free(&mut self) -> Result<(), rocddi::Error> {
        self.address.free()
    }
}

/// Retained physical-memory handle shared by public references and mappings.
pub(crate) struct VmemHandle {
    memory: VirtualMemory,
    references: usize,
    mappings: usize,
    pool: HsaMemoryPool,
    memory_type: u32,
    imported: bool,
}

impl VmemHandle {
    fn new(memory: VirtualMemory, pool: HsaMemoryPool, memory_type: u32, imported: bool) -> Self {
        Self {
            memory,
            references: 1,
            mappings: 0,
            pool,
            memory_type,
            imported,
        }
    }

    fn free(&mut self) -> Result<(), rocddi::Error> {
        self.memory.free()
    }
}

/// Native mapping owner selected by whether access belongs to a CPU or GPU.
enum VmemAccessOwner {
    Device(VirtualDeviceMapping),
    Host(VirtualHostMapping),
}

/// Per-agent permissions and the native object that enforces them.
struct VmemAccess {
    permissions: u32,
    owner: VmemAccessOwner,
}

impl VmemAccess {
    fn free(&mut self) -> Result<(), rocddi::Error> {
        match &mut self.owner {
            VmemAccessOwner::Device(mapping) => mapping.free(),
            VmemAccessOwner::Host(mapping) => mapping.free(),
        }
    }
}

/// One mapped virtual subrange and all native per-agent access mappings.
pub(crate) struct VmemMapping {
    handle: u64,
    reservation: usize,
    address: usize,
    offset: usize,
    size: usize,
    access: HashMap<u64, VmemAccess>,
}

impl VmemMapping {
    fn contains(&self, address: usize) -> bool {
        contains(self.address, self.size, address)
    }

    fn free_access(&mut self) -> Result<(), rocddi::Error> {
        let agents = self.access.keys().copied().collect::<Vec<_>>();
        for agent in agents {
            let Some(mut access) = self.access.remove(&agent) else {
                continue;
            };
            if let Err(error) = access.free() {
                self.access.insert(agent, access);
                return Err(error);
            }
        }
        Ok(())
    }
}

impl LockedMemory {
    fn new(
        allocation: Option<Allocation>,
        host_base: usize,
        device_base: usize,
        size: usize,
        global_flags: u32,
        accessible: Vec<HsaAgent>,
    ) -> Self {
        Self {
            allocation,
            description: PointerDescription {
                pointer_type: POINTER_TYPE_LOCKED,
                agent_base: device_base,
                host_base: Some(host_base),
                size,
                owner: HsaAgent { handle: CPU_AGENT },
                global_flags,
                registered: true,
                alloc_flags: POINTER_ALLOC_NONPAGED | POINTER_ALLOC_HOST_ACCESS,
                user_data: 0,
                accessible,
            },
        }
    }

    fn contains(&self, address: usize) -> bool {
        self.description.contains(address)
    }

    fn free(&mut self) -> Result<(), rocddi::Error> {
        self.allocation.as_mut().map_or(Ok(()), Allocation::free)
    }
}

fn contains(base: usize, size: usize, address: usize) -> bool {
    address
        .checked_sub(base)
        .is_some_and(|offset| offset < size)
}

fn current_host_page() -> Result<usize, Status> {
    let page =
        usize::try_from(rocddi::memory::host_page_size().map_err(map_error)?).map_err(|_| ERROR)?;
    if page < 4096 || !page.is_power_of_two() {
        return Err(ERROR);
    }
    Ok(page)
}

fn cpu_pool(pool: HsaMemoryPool) -> bool {
    matches!(
        pool.handle,
        CPU_POOL_FINE | CPU_POOL_EXTENDED | CPU_POOL_KERNARG | CPU_POOL_COARSE
    )
}

fn cpu_pool_memory_kind(pool: HsaMemoryPool, lifetime: SessionLifetime) -> MemoryKind {
    if pool.handle == CPU_POOL_KERNARG || lifetime == SessionLifetime::Session {
        MemoryKind::System
    } else {
        MemoryKind::OwnedHost
    }
}

fn pool_owner(runtime: &Runtime, pool: HsaMemoryPool) -> Option<HsaAgent> {
    if cpu_pool(pool) {
        Some(HsaAgent { handle: CPU_AGENT })
    } else {
        runtime.decode_gpu_pool(pool).map(|(index, _)| HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        })
    }
}

fn pool_global_flags(runtime: &Runtime, pool: HsaMemoryPool) -> Option<u32> {
    let flags = match pool.handle {
        CPU_POOL_FINE => POOL_FLAG_FINE,
        CPU_POOL_EXTENDED => POOL_FLAG_EXTENDED_FINE,
        CPU_POOL_KERNARG => POOL_FLAG_KERNARG | POOL_FLAG_FINE,
        CPU_POOL_COARSE => POOL_FLAG_COARSE,
        _ => match runtime.decode_gpu_pool(pool) {
            Some((_, GPU_POOL_COARSE)) => POOL_FLAG_COARSE,
            Some((_, GPU_POOL_FINE)) => POOL_FLAG_FINE,
            Some((_, GPU_POOL_GROUP)) => 0,
            _ => return None,
        },
    };
    Some(flags)
}

fn gpu_agent_pools(index: usize, fine_grain: bool) -> Vec<HsaMemoryPool> {
    let mut pools = Vec::with_capacity(usize::from(fine_grain) + 2);
    pools.push(Runtime::gpu_pool(index, GPU_POOL_COARSE));
    if fine_grain {
        pools.push(Runtime::gpu_pool(index, GPU_POOL_FINE));
    }
    pools.push(Runtime::gpu_pool(index, GPU_POOL_GROUP));
    pools
}

fn gpu_agent_regions(index: usize, fine_grain: bool) -> Vec<HsaMemoryPool> {
    let mut regions = gpu_agent_pools(index, fine_grain);
    regions.extend([
        HsaMemoryPool {
            handle: CPU_POOL_FINE,
        },
        HsaMemoryPool {
            handle: CPU_POOL_KERNARG,
        },
        HsaMemoryPool {
            handle: CPU_POOL_EXTENDED,
        },
        HsaMemoryPool {
            handle: CPU_POOL_COARSE,
        },
    ]);
    regions
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PoolStorage {
    System,
    LocalCoarse,
    LocalFine,
    Group,
}

fn pool_access(same_owner: bool, storage: PoolStorage, linked: bool, same_hive: bool) -> u32 {
    if same_owner {
        POOL_ACCESS_DEFAULT
    } else if matches!(storage, PoolStorage::Group) || !linked {
        POOL_ACCESS_NEVER
    } else {
        match storage {
            PoolStorage::System | PoolStorage::LocalCoarse => POOL_ACCESS_DISALLOWED,
            PoolStorage::LocalFine if same_hive => POOL_ACCESS_DISALLOWED,
            PoolStorage::LocalFine | PoolStorage::Group => POOL_ACCESS_NEVER,
        }
    }
}

fn agent_pool_relationship(
    runtime: &Runtime,
    agent: HsaAgent,
    owner: HsaAgent,
    storage: PoolStorage,
) -> (u32, Option<MemoryLinkInfo>) {
    let same = owner == agent;
    let requester_gpu = runtime.gpu_index(agent);
    let owner_gpu = runtime.gpu_index(owner);
    let same_hive = match (requester_gpu, owner_gpu) {
        (Some(requester), Some(owner)) => {
            runtime.gpus[requester].info.hive_id == runtime.gpus[owner].info.hive_id
        }
        (None, Some(owner)) => runtime.gpus[owner].info.hive_id == 0,
        _ => false,
    };
    let link = if same {
        None
    } else if agent.handle == CPU_AGENT {
        owner_gpu.and_then(|index| runtime.gpus[index].endpoint.memory_link_from_host())
    } else if let Some(requester) = requester_gpu {
        if matches!(storage, PoolStorage::System) {
            runtime.gpus[requester].endpoint.memory_link_to_host()
        } else {
            owner_gpu.and_then(|owner| {
                runtime.gpus[requester]
                    .endpoint
                    .memory_link_to(&runtime.gpus[owner].endpoint)
            })
        }
    } else {
        None
    };
    let linked = same || link.is_some_and(|link| link.hop_count() != 0);
    (pool_access(same, storage, linked, same_hive), link)
}

fn pool_link_info(link: Option<MemoryLinkInfo>) -> PoolLinkInfo {
    let Some(link) = link else {
        return PoolLinkInfo {
            min_latency: 0,
            max_latency: 0,
            min_bandwidth: 0,
            max_bandwidth: 0,
            atomic_support_32bit: false,
            atomic_support_64bit: false,
            coherent_support: false,
            link_type: 0,
            numa_distance: 0,
        };
    };
    PoolLinkInfo {
        min_latency: link.minimum_latency(),
        max_latency: link.maximum_latency(),
        min_bandwidth: link.minimum_bandwidth(),
        max_bandwidth: link.maximum_bandwidth(),
        atomic_support_32bit: link.supports_32bit_atomics(),
        atomic_support_64bit: link.supports_64bit_atomics(),
        coherent_support: link.is_coherent(),
        link_type: hsa_link_type(link.kind()),
        numa_distance: link.numa_distance(),
    }
}

const fn hsa_link_type(kind: MemoryLinkType) -> u32 {
    match kind {
        MemoryLinkType::HyperTransport | MemoryLinkType::Unknown => 0,
        MemoryLinkType::Qpi => 1,
        MemoryLinkType::Pcie => 2,
        MemoryLinkType::Infiniband => 3,
        MemoryLinkType::Xgmi => 4,
    }
}

fn agent_pools(runtime: &Runtime, agent: HsaAgent) -> Result<Vec<HsaMemoryPool>, Status> {
    if agent.handle == CPU_AGENT {
        Ok(vec![
            HsaMemoryPool {
                handle: CPU_POOL_FINE,
            },
            HsaMemoryPool {
                handle: CPU_POOL_EXTENDED,
            },
            HsaMemoryPool {
                handle: CPU_POOL_KERNARG,
            },
            HsaMemoryPool {
                handle: CPU_POOL_COARSE,
            },
        ])
    } else if let Some(index) = runtime.gpu_index(agent) {
        Ok(gpu_agent_pools(index, runtime.gpus[index].fine_grain_pool))
    } else {
        Err(INVALID_AGENT)
    }
}

fn agent_regions(runtime: &Runtime, agent: HsaAgent) -> Result<Vec<HsaMemoryPool>, Status> {
    if agent.handle == CPU_AGENT {
        return agent_pools(runtime, agent);
    }
    let Some(index) = runtime.gpu_index(agent) else {
        return Err(INVALID_AGENT);
    };
    Ok(gpu_agent_regions(
        index,
        runtime.gpus[index].fine_grain_pool,
    ))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_agent_iterate_memory_pools(
    agent: HsaAgent,
    callback: PoolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let pools = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match guard.as_ref() {
                Some(runtime) => runtime,
                None => return NOT_INITIALIZED,
            };
            if callback.is_none() {
                return INVALID_ARGUMENT;
            }
            match agent_pools(runtime, agent) {
                Ok(pools) => pools,
                Err(status) => return status,
            }
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        for pool in pools {
            // SAFETY: HSA defines traversal callbacks as synchronous and the
            // caller keeps its data pointer live through this call.
            let status = unsafe { callback(pool, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_agent_iterate_regions(
    agent: HsaAgent,
    callback: RegionCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let regions = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            if callback.is_none() {
                return INVALID_ARGUMENT;
            }
            match agent_regions(runtime, agent) {
                Ok(pools) => pools
                    .into_iter()
                    .map(|pool| HsaRegion {
                        handle: pool.handle,
                    })
                    .collect::<Vec<_>>(),
                Err(status) => return status,
            }
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        for region in regions {
            // SAFETY: HSA region traversal is synchronous and data stays live.
            let status = unsafe { callback(region, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_agent_iterate_caches(
    agent: HsaAgent,
    callback: CacheCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let caches = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            if !runtime.is_agent(agent) {
                return INVALID_AGENT;
            }
            let Some(_) = callback else {
                return INVALID_ARGUMENT;
            };
            runtime
                .caches
                .iter()
                .enumerate()
                .filter(|(_, cache)| cache.agent == agent)
                .map(|(index, _)| HsaCache {
                    handle: CACHE_BASE + index as u64,
                })
                .collect::<Vec<_>>()
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        for cache in caches {
            // SAFETY: Cache traversal is synchronous and data stays live.
            let status = unsafe { callback(cache, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_cache_get_info(
    cache: HsaCache,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(index) = runtime.cache_index(cache) else {
            return INVALID_CACHE;
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let cache = &runtime.caches[index];
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                CACHE_INFO_NAME_LENGTH => value.cast::<u32>().write((cache.name.len() - 1) as u32),
                CACHE_INFO_NAME => value
                    .cast::<*const c_char>()
                    .write(cache.name.as_ptr().cast()),
                CACHE_INFO_LEVEL => value.cast::<u8>().write(cache.level),
                CACHE_INFO_SIZE => value.cast::<u32>().write(cache.size),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

unsafe fn memory_pool_get_info(
    runtime: &Runtime,
    pool: HsaMemoryPool,
    attribute: u32,
    value: *mut c_void,
    invalid_pool: Status,
) -> Status {
    let gpu_pool = runtime.decode_gpu_pool(pool);
    if !cpu_pool(pool) && gpu_pool.is_none() {
        return invalid_pool;
    }
    let kind = gpu_pool.map_or(0, |(_, kind)| kind);
    let group = kind == GPU_POOL_GROUP;
    let flags = pool_global_flags(runtime, pool).unwrap_or(0);
    let page = runtime.host_page_size;
    let size = gpu_pool.map_or(runtime.host_memory_bytes, |(index, _)| {
        if group {
            usize::try_from(runtime.gpus[index].info.local_data_share_byte_length)
                .unwrap_or(usize::MAX)
        } else {
            usize::try_from(runtime.gpus[index].endpoint.local_memory_bytes).unwrap_or(usize::MAX)
        }
    });
    // SAFETY: Each arm writes the public value type for the attribute.
    unsafe {
        match attribute {
            POOL_INFO_SEGMENT => {
                value
                    .cast::<u32>()
                    .write(if group { SEGMENT_GROUP } else { SEGMENT_GLOBAL })
            }
            POOL_INFO_GLOBAL_FLAGS => value.cast::<u32>().write(flags),
            POOL_INFO_SIZE => value.cast::<usize>().write(size),
            POOL_INFO_ALLOC_MAX_SIZE => {
                value
                    .cast::<usize>()
                    .write(if group { 0 } else { size & !(page - 1) })
            }
            POOL_INFO_RUNTIME_ALLOC_ALLOWED => value.cast::<bool>().write(!group),
            POOL_INFO_RUNTIME_ALLOC_GRANULE | POOL_INFO_RUNTIME_ALLOC_ALIGNMENT => {
                value.cast::<usize>().write(if group { 0 } else { page })
            }
            POOL_INFO_RUNTIME_ALLOC_REC_GRANULE => value.cast::<usize>().write(if group {
                0
            } else if gpu_pool.is_some() {
                2 * 1024 * 1024
            } else {
                page
            }),
            POOL_INFO_ACCESSIBLE_BY_ALL => value.cast::<bool>().write(cpu_pool(pool)),
            POOL_INFO_LOCATION => {
                if group {
                    return INVALID_ARGUMENT;
                }
                value.cast::<u32>().write(u32::from(!cpu_pool(pool)));
            }
            _ => return INVALID_ARGUMENT,
        }
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_pool_get_info(
    pool: HsaMemoryPool,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller provided writable storage for the requested attribute.
        unsafe { memory_pool_get_info(runtime, pool, attribute, value, INVALID_MEMORY_POOL) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_region_get_info(
    region: HsaRegion,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let pool = HsaMemoryPool {
            handle: region.handle,
        };
        let pool_attribute = match attribute {
            POOL_INFO_SEGMENT
            | POOL_INFO_GLOBAL_FLAGS
            | POOL_INFO_SIZE
            | POOL_INFO_RUNTIME_ALLOC_ALLOWED
            | POOL_INFO_RUNTIME_ALLOC_GRANULE
            | POOL_INFO_RUNTIME_ALLOC_ALIGNMENT => Some(attribute),
            REGION_INFO_ALLOC_MAX_SIZE => Some(POOL_INFO_ALLOC_MAX_SIZE),
            _ => None,
        };
        if let Some(pool_attribute) = pool_attribute {
            // SAFETY: The caller provided writable storage for the requested attribute.
            return unsafe {
                memory_pool_get_info(runtime, pool, pool_attribute, value, INVALID_REGION)
            };
        }
        if !cpu_pool(pool) && runtime.decode_gpu_pool(pool).is_none() {
            return INVALID_REGION;
        }
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                REGION_INFO_ALLOC_MAX_PRIVATE_WORKGROUP_SIZE => return INVALID_ARGUMENT,
                AMD_REGION_INFO_HOST_ACCESSIBLE => value.cast::<bool>().write(cpu_pool(pool)),
                AMD_REGION_INFO_BASE => value.cast::<*mut c_void>().write(std::ptr::null_mut()),
                AMD_REGION_INFO_BUS_WIDTH => value.cast::<u32>().write(0),
                AMD_REGION_INFO_MAX_CLOCK_FREQUENCY => value.cast::<u32>().write(0),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_memory_allocate(
    region: HsaRegion,
    size: usize,
    pointer: *mut *mut c_void,
) -> Status {
    // SAFETY: Legacy regions and AMD pools identify the same memory resources.
    unsafe {
        memory_pool_allocate(
            HsaMemoryPool {
                handle: region.handle,
            },
            size,
            0,
            pointer,
            INVALID_REGION,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_memory_free(pointer: *mut c_void) -> Status {
    if pointer.is_null() {
        return boundary(|| {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            if guard.is_some() {
                SUCCESS
            } else {
                NOT_INITIALIZED
            }
        });
    }
    // SAFETY: Both entry points release allocations owned by this runtime.
    unsafe { hsa_amd_memory_pool_free(pointer) }
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_memory_register(pointer: *mut c_void, size: usize) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.is_none() {
            return NOT_INITIALIZED;
        }
        if !pointer.is_null() && size == 0 {
            return INVALID_ARGUMENT;
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_memory_deregister(_pointer: *mut c_void, _size: usize) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.is_some() {
            SUCCESS
        } else {
            NOT_INITIALIZED
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_memory_assign_agent(
    pointer: *mut c_void,
    agent: HsaAgent,
    access: u32,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if pointer.is_null() || !(1..=3).contains(&access) {
            return INVALID_ARGUMENT;
        }
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_agent_memory_pool_get_info(
    agent: HsaAgent,
    pool: HsaMemoryPool,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        let Some(owner) = pool_owner(runtime, pool) else {
            return INVALID_MEMORY_POOL;
        };
        let storage = runtime
            .decode_gpu_pool(pool)
            .map_or(PoolStorage::System, |(_, kind)| match kind {
                GPU_POOL_GROUP => PoolStorage::Group,
                GPU_POOL_FINE => PoolStorage::LocalFine,
                _ => PoolStorage::LocalCoarse,
            });
        let (access, link) = agent_pool_relationship(runtime, agent, owner, storage);
        let same = owner == agent;
        let hops = if same || access == POOL_ACCESS_NEVER {
            0
        } else {
            link.map_or(0, MemoryLinkInfo::hop_count)
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                AGENT_POOL_INFO_ACCESS => value.cast::<u32>().write(access),
                AGENT_POOL_INFO_NUM_LINK_HOPS => value.cast::<u32>().write(hops),
                AGENT_POOL_INFO_LINK_INFO => value
                    .cast::<PoolLinkInfo>()
                    .write(pool_link_info(if hops == 0 { None } else { link })),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

unsafe fn memory_pool_allocate(
    pool: HsaMemoryPool,
    size: usize,
    flags: u32,
    pointer: *mut *mut c_void,
    invalid_pool: Status,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if pointer.is_null() || size == 0 || flags & !ALLOC_FLAGS != 0 {
            return INVALID_ARGUMENT;
        }
        let page = runtime.host_page_size;
        let rounded = match size.checked_add(page - 1).map(|value| value & !(page - 1)) {
            Some(size) => size,
            None => return INVALID_ALLOCATION,
        };
        let (device_index, kind) = if cpu_pool(pool) {
            (0, cpu_pool_memory_kind(pool, runtime.lifetime))
        } else if let Some((index, pool_kind)) = runtime.decode_gpu_pool(pool) {
            if pool_kind == GPU_POOL_GROUP {
                return INVALID_ALLOCATION;
            }
            let uncached = flags & ALLOC_UNCACHED != 0;
            (
                index,
                MemoryKind::DeviceLocal {
                    host_visible: true,
                    coherent: pool_kind == GPU_POOL_FINE || flags & ALLOC_PCIE != 0,
                    uncached,
                    contiguous: flags & ALLOC_CONTIGUOUS != 0,
                },
            )
        } else {
            return invalid_pool;
        };
        let maximum = runtime.decode_gpu_pool(pool).map_or(
            runtime.host_memory_bytes & !(page - 1),
            |(index, pool_kind)| {
                if pool_kind == GPU_POOL_GROUP {
                    0
                } else {
                    usize::try_from(runtime.gpus[index].endpoint.local_memory_bytes)
                        .unwrap_or(usize::MAX)
                        & !(page - 1)
                }
            },
        );
        if rounded > maximum {
            return INVALID_ALLOCATION;
        }
        let owner = match pool_owner(runtime, pool) {
            Some(owner) => owner,
            None => return invalid_pool,
        };
        let global_flags = match pool_global_flags(runtime, pool) {
            Some(flags) => flags,
            None => return invalid_pool,
        };
        let permissions = if flags & ALLOC_EXECUTABLE != 0 {
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE
        } else {
            DeviceAccess::READ | DeviceAccess::WRITE
        };
        let allocation = match runtime.gpus[device_index].device.allocate(
            kind,
            rounded as u64,
            page as u64,
            permissions,
        ) {
            Ok(allocation) => allocation,
            Err(error) => return map_error(error),
        };
        let Some(host) = allocation.info().host_address else {
            return OUT_OF_RESOURCES;
        };
        let accessible = vec![HsaAgent {
            handle: GPU_AGENT_BASE + device_index as u64,
        }];
        let alloc_flags = POINTER_ALLOC_NONPAGED
            | POINTER_ALLOC_HOST_ACCESS
            | if flags & ALLOC_CONTIGUOUS != 0 {
                POINTER_ALLOC_CONTIGUOUS
            } else {
                0
            }
            | if flags & ALLOC_EXECUTABLE != 0 {
                POINTER_ALLOC_EXECUTABLE
            } else {
                0
            }
            | if global_flags & POOL_FLAG_FINE != 0 && flags & ALLOC_UNCACHED == 0 {
                POINTER_ALLOC_ATOMIC_FULL
            } else if flags & ALLOC_PCIE != 0 {
                POINTER_ALLOC_ATOMIC_PARTIAL
            } else {
                0
            };
        runtime.allocations.insert(
            host,
            Memory::new(
                allocation,
                size,
                owner,
                global_flags,
                alloc_flags,
                accessible,
            ),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { pointer.write(host as *mut c_void) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_pool_allocate(
    pool: HsaMemoryPool,
    size: usize,
    flags: u32,
    pointer: *mut *mut c_void,
) -> Status {
    // SAFETY: This forwards the public ABI arguments to the common allocator.
    unsafe { memory_pool_allocate(pool, size, flags, pointer, INVALID_MEMORY_POOL) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_register_deallocation_callback(
    pointer: *mut c_void,
    callback: DeallocationCallbackFn,
    user_data: *mut c_void,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        if let Some(memory) = runtime
            .allocations
            .values_mut()
            .find(|memory| memory.contains(pointer as usize))
        {
            if memory.deallocation_callbacks.try_reserve(1).is_err() {
                return OUT_OF_RESOURCES;
            }
            memory
                .deallocation_callbacks
                .push(RegisteredDeallocationCallback {
                    pointer: pointer as usize,
                    callback,
                    user_data: user_data as usize,
                });
            return SUCCESS;
        }
        if let Some((_, reservation)) = runtime
            .vmem_reservations
            .range_mut(..=pointer as usize)
            .next_back()
            .filter(|(_, reservation)| reservation.contains_range(pointer as usize, 1))
        {
            if reservation.deallocation_callbacks.try_reserve(1).is_err() {
                return OUT_OF_RESOURCES;
            }
            reservation
                .deallocation_callbacks
                .push(RegisteredDeallocationCallback {
                    pointer: pointer as usize,
                    callback,
                    user_data: user_data as usize,
                });
            return SUCCESS;
        }
        INVALID_ALLOCATION
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_deregister_deallocation_callback(
    pointer: *mut c_void,
    callback: DeallocationCallbackFn,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        if let Some(memory) = runtime
            .allocations
            .values_mut()
            .find(|memory| memory.contains(pointer as usize))
        {
            let previous_len = memory.deallocation_callbacks.len();
            memory.deallocation_callbacks.retain(|registered| {
                registered.pointer != pointer as usize
                    || registered.callback as usize != callback as usize
            });
            return if memory.deallocation_callbacks.len() == previous_len {
                INVALID_ARGUMENT
            } else {
                SUCCESS
            };
        }
        if let Some((_, reservation)) = runtime
            .vmem_reservations
            .range_mut(..=pointer as usize)
            .next_back()
            .filter(|(_, reservation)| reservation.contains_range(pointer as usize, 1))
        {
            let previous_len = reservation.deallocation_callbacks.len();
            reservation.deallocation_callbacks.retain(|registered| {
                registered.pointer != pointer as usize
                    || registered.callback as usize != callback as usize
            });
            return if reservation.deallocation_callbacks.len() == previous_len {
                INVALID_ARGUMENT
            } else {
                SUCCESS
            };
        }
        INVALID_ARGUMENT
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_pool_free(pointer: *mut c_void) -> Status {
    boundary(|| {
        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        let callbacks = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            let base = runtime
                .allocations
                .get(&(pointer as usize))
                .map_or(pointer as usize, |memory| memory.description.agent_base);
            if copy_borrowed(runtime, base) {
                return OUT_OF_RESOURCES;
            }
            if let Some(mut memory) = runtime.allocations.remove(&(pointer as usize)) {
                match memory.free() {
                    Ok(()) => std::mem::take(&mut memory.deallocation_callbacks),
                    Err(error) => {
                        runtime.allocations.insert(pointer as usize, memory);
                        return map_error(error);
                    }
                }
            } else {
                let Some(index) = runtime
                    .locked_allocations
                    .iter()
                    .rposition(|memory| memory.description.agent_base == pointer as usize)
                else {
                    return INVALID_ALLOCATION;
                };
                if copy_borrowed(
                    runtime,
                    runtime.locked_allocations[index].description.agent_base,
                ) {
                    return OUT_OF_RESOURCES;
                }
                let mut memory = runtime.locked_allocations.remove(index);
                return match memory.free() {
                    Ok(()) => SUCCESS,
                    Err(error) => {
                        runtime.locked_allocations.insert(index, memory);
                        map_error(error)
                    }
                };
            }
        };
        for registered in callbacks {
            // SAFETY: The caller registered this callback and its user data for
            // synchronous notification when this allocation was released.
            unsafe {
                (registered.callback)(
                    registered.pointer as *mut c_void,
                    registered.user_data as *mut c_void,
                )
            };
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_ipc_memory_create(
    pointer: *mut c_void,
    size: usize,
    handle: *mut HsaAmdIpcMemory,
) -> Status {
    boundary(|| {
        if pointer.is_null() || size == 0 || handle.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(memory) = runtime.allocations.values().find(|memory| {
            memory.description.agent_base == pointer as usize
                && memory.description.size == size
                && memory.description.owner.handle != CPU_AGENT
        }) else {
            return INVALID_ARGUMENT;
        };
        let exported = match linux_interop::export_kfd_ipc_memory(&memory.allocation) {
            Ok(handle) => handle.words(),
            Err(error) => return map_error(error),
        };
        // SAFETY: The caller supplied writable output storage and publication
        // occurs only after the native export completed successfully.
        unsafe { handle.write(HsaAmdIpcMemory { handle: exported }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_ipc_memory_attach(
    handle: *const HsaAmdIpcMemory,
    size: usize,
    num_agents: u32,
    mapping_agents: *const HsaAgent,
    mapped_pointer: *mut *mut c_void,
) -> Status {
    boundary(|| {
        if handle.is_null()
            || size == 0
            || mapped_pointer.is_null()
            || (num_agents != 0 && mapping_agents.is_null())
        {
            return INVALID_ARGUMENT;
        }
        // Read inputs before publishing output; the C ABI does not forbid
        // mapped_pointer from overlapping the handle or agent array.
        // SAFETY: The caller supplied a readable IPC handle.
        let words = unsafe { handle.read() }.handle;
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let requested = if num_agents == 0 {
            None
        } else {
            // SAFETY: The caller supplied num_agents readable entries.
            Some(unsafe { std::slice::from_raw_parts(mapping_agents, num_agents as usize) })
        };
        if requested.is_some_and(|agents| agents.iter().any(|agent| !runtime.is_agent(*agent))) {
            return INVALID_AGENT;
        }

        let mut accessible = Vec::new();
        let mut mapping_indices = Vec::new();
        if let Some(agents) = requested {
            if accessible.try_reserve(agents.len()).is_err()
                || mapping_indices.try_reserve(agents.len()).is_err()
            {
                return OUT_OF_RESOURCES;
            }
            for agent in agents {
                if !accessible.contains(agent) {
                    accessible.push(*agent);
                }
                if let Some(index) = runtime.gpu_index(*agent) {
                    if !mapping_indices.contains(&index) {
                        mapping_indices.push(index);
                    }
                }
            }
        } else {
            if accessible.try_reserve(runtime.gpus.len()).is_err()
                || mapping_indices.try_reserve(runtime.gpus.len()).is_err()
            {
                return OUT_OF_RESOURCES;
            }
            for index in 0..runtime.gpus.len() {
                accessible.push(HsaAgent {
                    handle: GPU_AGENT_BASE + index as u64,
                });
                mapping_indices.push(index);
            }
        }

        let allocation = {
            let devices = runtime
                .gpus
                .iter()
                .map(|gpu| &gpu.device)
                .collect::<Vec<_>>();
            let mapping_devices = mapping_indices
                .iter()
                .map(|index| &runtime.gpus[*index].device)
                .collect::<Vec<_>>();
            match linux_interop::import_kfd_ipc_memory(
                &runtime.session,
                &devices,
                &mapping_devices,
                IpcMemoryHandle::from_words(words),
                size as u64,
            ) {
                Ok(allocation) => allocation,
                Err(error) => {
                    return match error.kind() {
                        rocddi::ErrorKind::InvalidArgument | rocddi::ErrorKind::Unsupported => {
                            INVALID_ARGUMENT
                        }
                        rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
                        _ => ERROR,
                    };
                }
            }
        };
        let info = allocation.info();
        let imported_pointer = info.device_address as usize;
        let owner = runtime
            .gpus
            .iter()
            .enumerate()
            .find(|(_, gpu)| allocation.originates_from(&gpu.device))
            .map_or(HsaAgent { handle: 0 }, |(index, _)| HsaAgent {
                handle: GPU_AGENT_BASE + index as u64,
            });
        if runtime.ipc_allocations.try_reserve(1).is_err()
            || runtime.ipc_allocations.contains_key(&imported_pointer)
        {
            return OUT_OF_RESOURCES;
        }
        runtime.ipc_allocations.insert(
            imported_pointer,
            Memory::new_ipc(allocation, size, owner, accessible),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { mapped_pointer.write(imported_pointer as *mut c_void) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_ipc_memory_detach(mapped_pointer: *mut c_void) -> Status {
    boundary(|| {
        if mapped_pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let key = mapped_pointer as usize;
        let base = runtime
            .ipc_allocations
            .get(&key)
            .map_or(key, |memory| memory.description.agent_base);
        if copy_borrowed(runtime, base) {
            return OUT_OF_RESOURCES;
        }
        let Some(mut memory) = runtime.ipc_allocations.remove(&key) else {
            return INVALID_ARGUMENT;
        };
        if memory.free().is_ok() {
            SUCCESS
        } else {
            runtime.ipc_allocations.insert(key, memory);
            INVALID_ARGUMENT
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_external_semaphore_handle_open(
    agent: HsaAgent,
    descriptor: *const HsaAmdExternalSemaphoreHandleDescriptor,
    semaphore: *mut HsaAmdExternalSemaphore,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if descriptor.is_null() || semaphore.is_null() {
            return INVALID_ARGUMENT;
        }
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        NOT_SUPPORTED
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_external_semaphore_handle_close(
    _semaphore: HsaAmdExternalSemaphore,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.is_none() {
            return NOT_INITIALIZED;
        }
        ERROR
    })
}

#[derive(Clone, Copy)]
enum AisOperation {
    Read,
    Write,
}

unsafe fn ais_transfer(
    descriptor: i32,
    host: usize,
    size: usize,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
    operation: AisOperation,
) -> Status {
    let mut copied = 0usize;
    let mut write_retries = 3;
    let mut first = true;
    let result = loop {
        let remaining = size - copied;
        if !first && remaining == 0 {
            break Ok(());
        }
        first = false;
        let Some(offset) = i64::try_from(copied)
            .ok()
            .and_then(|copied| file_offset.checked_add(copied))
        else {
            break Err(EOVERFLOW);
        };
        // SAFETY: The caller validated the complete host range and keeps it
        // live for this synchronous transfer. The Linux provider borrows it.
        let transferred = match operation {
            AisOperation::Read => {
                let buffer = if remaining == 0 {
                    &mut []
                } else {
                    unsafe { std::slice::from_raw_parts_mut((host + copied) as *mut u8, remaining) }
                };
                linux_interop::read_descriptor_at(descriptor, buffer, offset)
            }
            AisOperation::Write => {
                let buffer = if remaining == 0 {
                    &[]
                } else {
                    unsafe { std::slice::from_raw_parts((host + copied) as *const u8, remaining) }
                };
                linux_interop::write_descriptor_at(descriptor, buffer, offset)
            }
        };
        let transferred = match transferred {
            Ok(transferred) => transferred,
            Err(error) => break Err(error.raw_os_error().unwrap_or(EIO)),
        };
        if transferred == 0 {
            if matches!(operation, AisOperation::Read) || remaining == 0 {
                break Ok(());
            }
            if write_retries == 0 {
                break Err(EIO);
            }
            write_retries -= 1;
            continue;
        }
        if transferred > remaining {
            break Err(EIO);
        }
        copied += transferred;
    };

    // SAFETY: Both outputs are optional in the public ABI and, when non-null,
    // point to caller-owned writable storage for the duration of this call.
    unsafe {
        if !size_copied.is_null() {
            size_copied.write(copied as u64);
        }
        if !operation_status.is_null() {
            operation_status.write(result.as_ref().map_or_else(|error| -*error, |()| 0));
        }
    }
    result.map_or(ERROR, |()| SUCCESS)
}

unsafe fn ais_file_io(
    handle: HsaAmdAisFileHandle,
    device_pointer: *mut c_void,
    size: u64,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
    operation: AisOperation,
) -> Status {
    // SAFETY: Linux defines the public union's active member as fd.
    let descriptor = unsafe { handle.fd };
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    if device_pointer.is_null() || descriptor < 0 {
        return INVALID_ARGUMENT;
    }
    let transfer_size = match usize::try_from(size.min(AIS_MAX_TRANSFER_BYTES)) {
        Ok(size) => size,
        Err(_) => return ERROR,
    };
    let Some(description) = pointer_description(runtime, device_pointer as usize) else {
        return ERROR;
    };
    if description.owner.handle == CPU_AGENT {
        return ERROR;
    }
    let Some(host) = description.host_range(device_pointer as usize, transfer_size) else {
        return ERROR;
    };
    // Keep the runtime lock through positioned I/O so another HSA call cannot
    // free or unmap the allocation while the provider borrows its host range.
    // SAFETY: The mapped range stays owned by the locked runtime. The public
    // call requires the descriptor to remain live for this synchronous call.
    let status = unsafe {
        ais_transfer(
            descriptor,
            host,
            transfer_size,
            file_offset,
            size_copied,
            operation_status,
            operation,
        )
    };
    drop(guard);
    status
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_ais_file_write(
    handle: HsaAmdAisFileHandle,
    device_pointer: *mut c_void,
    size: u64,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
) -> Status {
    boundary(|| {
        // SAFETY: ais_file_io validates the public arguments and allocation.
        unsafe {
            ais_file_io(
                handle,
                device_pointer,
                size,
                file_offset,
                size_copied,
                operation_status,
                AisOperation::Write,
            )
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_ais_file_read(
    handle: HsaAmdAisFileHandle,
    device_pointer: *mut c_void,
    size: u64,
    file_offset: i64,
    size_copied: *mut u64,
    operation_status: *mut i32,
) -> Status {
    boundary(|| {
        // SAFETY: ais_file_io validates the public arguments and allocation.
        unsafe {
            ais_file_io(
                handle,
                device_pointer,
                size,
                file_offset,
                size_copied,
                operation_status,
                AisOperation::Read,
            )
        }
    })
}

unsafe fn portable_export_dmabuf(
    pointer: *const c_void,
    size: usize,
    descriptor: *mut i32,
    offset: *mut u64,
    _flags: u64,
) -> Status {
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    if pointer.is_null() || descriptor.is_null() || offset.is_null() || size == 0 {
        return INVALID_ARGUMENT;
    }
    let Some((memory, pointer_offset)) = runtime.allocations.values().find_map(|memory| {
        memory
            .description
            .offset_range(pointer as usize, size)
            .map(|pointer_offset| (memory, pointer_offset))
    }) else {
        return INVALID_ALLOCATION;
    };
    if memory.description.owner.handle == CPU_AGENT {
        return INVALID_AGENT;
    }
    let dma_buf = match linux_interop::export_dma_buf(&memory.allocation) {
        Ok(dma_buf) => dma_buf,
        Err(error) => return map_error(error),
    };
    let exported_offset = match dma_buf
        .info()
        .source_offset
        .checked_add(pointer_offset as u64)
    {
        Some(exported_offset) => exported_offset,
        None => return INVALID_ALLOCATION,
    };
    let exported_descriptor = dma_buf.into_fd().into_raw_fd();
    // SAFETY: Both output pointers were validated above and are written only
    // after the export has completed successfully.
    unsafe {
        descriptor.write(exported_descriptor);
        offset.write(exported_offset);
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_portable_export_dmabuf(
    pointer: *const c_void,
    size: usize,
    descriptor: *mut i32,
    offset: *mut u64,
) -> Status {
    boundary(|| {
        // SAFETY: This forwards the validated public ABI arguments.
        unsafe { portable_export_dmabuf(pointer, size, descriptor, offset, DMABUF_MAPPING_NONE) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_portable_export_dmabuf_v2(
    pointer: *const c_void,
    size: usize,
    descriptor: *mut i32,
    offset: *mut u64,
    flags: u64,
) -> Status {
    boundary(|| {
        // SAFETY: This forwards the validated public ABI arguments.
        unsafe { portable_export_dmabuf(pointer, size, descriptor, offset, flags) }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_portable_close_dmabuf(descriptor: i32) -> Status {
    boundary(|| {
        // SAFETY: The ABI transfers responsibility for closing this descriptor.
        if unsafe { linux_interop::close_owned_descriptor(descriptor) }.is_ok() {
            SUCCESS
        } else {
            RESOURCE_FREE
        }
    })
}

fn registered_extent(host: usize, size: usize, page: usize) -> Option<usize> {
    host.checked_add(size)?;
    let page_offset = host & (page - 1);
    page_offset
        .checked_add(size)?
        .checked_add(page - 1)
        .map(|extent| extent & !(page - 1))
}

/// # Safety
/// `host_ptr` must cover `size` live host bytes through successful unlock,
/// `agents` must contain `num_agents` readable handles when non-null, and
/// `agent_ptr` must point to writable output storage. After an ambiguous
/// native registration failure, the caller must retain the host page cover
/// through process teardown.
unsafe fn memory_lock_to_pool(
    host_ptr: *mut c_void,
    size: usize,
    agents: *const HsaAgent,
    num_agents: i32,
    pool: HsaMemoryPool,
    flags: u32,
    agent_ptr: *mut *mut c_void,
) -> Status {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    if host_ptr.is_null() || size == 0 || agent_ptr.is_null() {
        return INVALID_ARGUMENT;
    }
    if num_agents < 0
        || (agents.is_null() && num_agents != 0)
        || (!agents.is_null() && num_agents == 0)
    {
        return INVALID_ARGUMENT;
    }
    if !cpu_pool(pool) {
        return INVALID_MEMORY_POOL;
    }
    let global_flags = match pool_global_flags(runtime, pool) {
        Some(flags) => flags,
        None => return INVALID_MEMORY_POOL,
    };
    if runtime.full_profile {
        // SAFETY: The caller supplied writable pointer storage.
        unsafe { agent_ptr.write(host_ptr) };
        return SUCCESS;
    }

    // Keep the output untouched while reading the caller's agent array: the
    // two may share storage under the public C ABI.
    let mut gpu_indexes = Vec::new();
    if agents.is_null() {
        gpu_indexes.extend(0..runtime.gpus.len());
    } else {
        let count = match usize::try_from(num_agents) {
            Ok(count) => count,
            Err(_) => return INVALID_ARGUMENT,
        };
        // SAFETY: The ABI requires num_agents readable handles when agents is non-null.
        let requested = unsafe { std::slice::from_raw_parts(agents, count) };
        for agent in requested {
            if agent.handle == CPU_AGENT {
                continue;
            }
            let Some(index) = runtime.gpu_index(*agent) else {
                return INVALID_AGENT;
            };
            if !gpu_indexes.contains(&index) {
                gpu_indexes.push(index);
            }
        }
    }

    if runtime.locked_allocations.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    let host_base = host_ptr as usize;
    let (allocation, device_base) = if let Some((&first_index, peer_indexes)) =
        gpu_indexes.split_first()
    {
        let Some(native_size) = registered_extent(host_base, size, runtime.host_page_size) else {
            return INVALID_ARGUMENT;
        };
        let allocation = {
            let first = &runtime.gpus[first_index].device;
            let peers: Vec<_> = peer_indexes
                .iter()
                .map(|&index| &runtime.gpus[index].device)
                .collect();
            // SAFETY: The HSA lock caller retains the complete host page
            // cover until unlock succeeds; ambiguous native failures keep
            // the registration backing caller-owned through teardown.
            match unsafe {
                first.register_host_with_peers(
                    &peers,
                    host_base,
                    flags & ALLOC_UNCACHED != 0,
                    native_size as u64,
                    runtime.host_page_size as u64,
                    DeviceAccess::READ | DeviceAccess::WRITE,
                )
            } {
                Ok(allocation) => allocation,
                Err(error) => return map_error(error),
            }
        };
        let device_base = allocation.info().device_address as usize;
        (Some(allocation), device_base)
    } else {
        (None, host_base)
    };
    let accessible = gpu_indexes
        .into_iter()
        .map(|index| HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        })
        .collect();
    runtime.locked_allocations.push(LockedMemory::new(
        allocation,
        host_base,
        device_base,
        size,
        global_flags,
        accessible,
    ));
    // SAFETY: The caller supplied writable pointer storage.
    unsafe { agent_ptr.write(device_base as *mut c_void) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_lock(
    host_ptr: *mut c_void,
    size: usize,
    agents: *const HsaAgent,
    num_agents: i32,
    agent_ptr: *mut *mut c_void,
) -> Status {
    boundary(|| {
        // SAFETY: The HSA C caller provides live input and output storage and
        // retains the host page cover through unlock or uncertain teardown.
        unsafe {
            memory_lock_to_pool(
                host_ptr,
                size,
                agents,
                num_agents,
                HsaMemoryPool {
                    handle: CPU_POOL_COARSE,
                },
                0,
                agent_ptr,
            )
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_lock_to_pool(
    host_ptr: *mut c_void,
    size: usize,
    agents: *const HsaAgent,
    num_agents: i32,
    pool: HsaMemoryPool,
    flags: u32,
    agent_ptr: *mut *mut c_void,
) -> Status {
    boundary(|| {
        // SAFETY: The HSA C caller provides live input and output storage and
        // retains the host page cover through unlock or uncertain teardown.
        unsafe { memory_lock_to_pool(host_ptr, size, agents, num_agents, pool, flags, agent_ptr) }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_unlock(host_ptr: *mut c_void) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.full_profile {
            return SUCCESS;
        }
        if host_ptr.is_null() {
            return INVALID_ARGUMENT;
        }
        let Some(index) = runtime
            .locked_allocations
            .iter()
            .rposition(|memory| memory.description.host_base == Some(host_ptr as usize))
        else {
            return INVALID_ARGUMENT;
        };
        if copy_borrowed(
            runtime,
            runtime.locked_allocations[index].description.agent_base,
        ) {
            return OUT_OF_RESOURCES;
        }
        let mut memory = runtime.locked_allocations.remove(index);
        match memory.free() {
            Ok(()) => SUCCESS,
            Err(error) => {
                runtime.locked_allocations.insert(index, memory);
                map_error(error)
            }
        }
    })
}

#[allow(clippy::too_many_arguments)]
unsafe fn interop_map_buffer(
    num_agents: u32,
    agents: *mut HsaAgent,
    interop_handle: HsaHandle,
    flags: u32,
    size_hint: usize,
    size: *mut usize,
    pointer: *mut *mut c_void,
    metadata_size: *mut usize,
    metadata: *mut *const c_void,
) -> Status {
    if num_agents == 0
        || agents.is_null()
        || interop_handle < 0
        || flags != 0
        || size.is_null()
        || pointer.is_null()
    {
        return INVALID_ARGUMENT;
    }
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    // SAFETY: The caller supplies num_agents readable handles.
    let agents = unsafe { std::slice::from_raw_parts(agents, num_agents as usize) };
    let mut devices = Vec::new();
    let mut accessible = Vec::new();
    if devices.try_reserve(agents.len()).is_err() || accessible.try_reserve(agents.len()).is_err() {
        return OUT_OF_RESOURCES;
    }
    for agent in agents {
        let Some(index) = runtime.gpu_index(*agent) else {
            return INVALID_AGENT;
        };
        devices.push(&runtime.gpus[index].device);
        accessible.push(*agent);
    }
    let descriptor = match linux_interop::duplicate_descriptor(interop_handle) {
        Ok(descriptor) => descriptor,
        Err(error) => return map_error(error),
    };
    let allocation = match linux_interop::import_graphics_dma_buf(
        &runtime.session,
        &devices,
        descriptor.as_fd(),
        size_hint as u64,
    ) {
        Ok(allocation) => allocation,
        Err(error) => {
            return match error.kind() {
                rocddi::ErrorKind::InvalidArgument | rocddi::ErrorKind::Unsupported => {
                    INVALID_ARGUMENT
                }
                rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
                _ => ERROR,
            };
        }
    };
    let info = allocation.info();
    let imported_size = match usize::try_from(info.size) {
        Ok(size) => size,
        Err(_) => return OUT_OF_RESOURCES,
    };
    let imported_pointer = info.device_address as usize;
    let owner = runtime
        .gpus
        .iter()
        .enumerate()
        .find(|(_, gpu)| allocation.originates_from(&gpu.device))
        .map_or(HsaAgent { handle: 0 }, |(index, _)| HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        });
    let metadata_bytes = allocation.metadata();
    let imported_metadata_size = metadata_bytes.len();
    let imported_metadata = if metadata_bytes.is_empty() {
        std::ptr::null()
    } else {
        metadata_bytes.as_ptr().cast::<c_void>()
    };
    if runtime.interop_allocations.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    if runtime.interop_allocations.contains_key(&imported_pointer) {
        return ERROR;
    }
    runtime.interop_allocations.insert(
        imported_pointer,
        Memory::new_interop(allocation, owner, accessible),
    );
    // SAFETY: Required outputs were validated above. Optional metadata outputs
    // are written independently, matching the public ABI.
    unsafe {
        size.write(imported_size);
        pointer.write(imported_pointer as *mut c_void);
        if !metadata_size.is_null() {
            metadata_size.write(imported_metadata_size);
        }
        if !metadata.is_null() {
            metadata.write(imported_metadata);
        }
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_interop_map_buffer(
    num_agents: u32,
    agents: *mut HsaAgent,
    interop_handle: HsaHandle,
    flags: u32,
    size: *mut usize,
    pointer: *mut *mut c_void,
    metadata_size: *mut usize,
    metadata: *mut *const c_void,
) -> Status {
    boundary(|| {
        // SAFETY: This forwards the public ABI arguments with a zero size hint.
        unsafe {
            interop_map_buffer(
                num_agents,
                agents,
                interop_handle,
                flags,
                0,
                size,
                pointer,
                metadata_size,
                metadata,
            )
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_interop_map_buffer_with_size(
    num_agents: u32,
    agents: *mut HsaAgent,
    interop_handle: HsaHandle,
    flags: u32,
    size_hint: usize,
    size: *mut usize,
    pointer: *mut *mut c_void,
    metadata_size: *mut usize,
    metadata: *mut *const c_void,
) -> Status {
    boundary(|| {
        // SAFETY: This forwards the public ABI arguments unchanged.
        unsafe {
            interop_map_buffer(
                num_agents,
                agents,
                interop_handle,
                flags,
                size_hint,
                size,
                pointer,
                metadata_size,
                metadata,
            )
        }
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_interop_unmap_buffer(pointer: *mut c_void) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if pointer.is_null() {
            return SUCCESS;
        }
        let key = pointer as usize;
        let base = runtime
            .interop_allocations
            .get(&key)
            .map_or(key, |memory| memory.description.agent_base);
        if copy_borrowed(runtime, base) {
            return OUT_OF_RESOURCES;
        }
        if let Some(mut memory) = runtime.interop_allocations.remove(&key) {
            if memory.free().is_err() {
                runtime.interop_allocations.insert(key, memory);
            }
        }
        SUCCESS
    })
}

fn vmem_reservation_base(runtime: &Runtime, address: usize, size: usize) -> Option<usize> {
    let (&base, reservation) = runtime.vmem_reservations.range(..=address).next_back()?;
    reservation.contains_range(address, size).then_some(base)
}

fn vmem_mapping_bases(
    runtime: &Runtime,
    address: usize,
    size: usize,
) -> Result<Vec<usize>, Status> {
    let end = address.checked_add(size).ok_or(INVALID_ARGUMENT)?;
    let mut bases = Vec::new();
    let mut cursor = address;
    while cursor < end {
        let mapping = runtime
            .vmem_mappings
            .get(&cursor)
            .ok_or(INVALID_ALLOCATION)?;
        bases.try_reserve(1).map_err(|_| OUT_OF_RESOURCES)?;
        bases.push(cursor);
        cursor = cursor.checked_add(mapping.size).ok_or(INVALID_ALLOCATION)?;
    }
    if cursor == end {
        Ok(bases)
    } else {
        Err(INVALID_ALLOCATION)
    }
}

fn replace_vmem_access(
    runtime: &mut Runtime,
    mapping_base: usize,
    descriptor: HsaAmdMemoryAccessDesc,
) -> Status {
    let existing_permission = runtime
        .vmem_mappings
        .get(&mapping_base)
        .and_then(|mapping| mapping.access.get(&descriptor.agent_handle.handle))
        .map(|access| access.permissions);
    if existing_permission == Some(descriptor.permissions) {
        return SUCCESS;
    }
    if descriptor.permissions != ACCESS_PERMISSION_NONE {
        let Some(mapping) = runtime.vmem_mappings.get_mut(&mapping_base) else {
            return INVALID_ALLOCATION;
        };
        if existing_permission.is_none() && mapping.access.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
    }
    let old_access = runtime
        .vmem_mappings
        .get_mut(&mapping_base)
        .and_then(|mapping| mapping.access.remove(&descriptor.agent_handle.handle));
    if let Some(mut access) = old_access {
        if let Err(error) = access.free() {
            if let Some(mapping) = runtime.vmem_mappings.get_mut(&mapping_base) {
                mapping
                    .access
                    .insert(descriptor.agent_handle.handle, access);
            }
            return map_error(error);
        }
    }
    if descriptor.permissions == ACCESS_PERMISSION_NONE {
        return SUCCESS;
    }

    let Some(mapping) = runtime.vmem_mappings.get(&mapping_base) else {
        return INVALID_ALLOCATION;
    };
    let Some(handle) = runtime.vmem_handles.get(&mapping.handle) else {
        return INVALID_ALLOCATION;
    };
    let Some(reservation) = runtime.vmem_reservations.get(&mapping.reservation) else {
        return INVALID_ALLOCATION;
    };
    let Some(permissions) = DeviceAccess::from_bits(descriptor.permissions) else {
        return INVALID_ARGUMENT;
    };
    let owner = if descriptor.agent_handle.handle == CPU_AGENT {
        match handle.memory.map_host(
            &reservation.address,
            mapping.address as u64,
            mapping.offset as u64,
            mapping.size as u64,
            permissions,
        ) {
            Ok(mapping) => VmemAccessOwner::Host(mapping),
            Err(error) => return map_error(error),
        }
    } else {
        let Some(index) = runtime.gpu_index(descriptor.agent_handle) else {
            return INVALID_AGENT;
        };
        match runtime.gpus[index].device.map_virtual_memory(
            &handle.memory,
            &reservation.address,
            mapping.address as u64,
            mapping.offset as u64,
            mapping.size as u64,
            permissions,
        ) {
            Ok(mapping) => VmemAccessOwner::Device(mapping),
            Err(error) => return map_error(error),
        }
    };
    let Some(mapping) = runtime.vmem_mappings.get_mut(&mapping_base) else {
        return ERROR;
    };
    mapping.access.insert(
        descriptor.agent_handle.handle,
        VmemAccess {
            permissions: descriptor.permissions,
            owner,
        },
    );
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_address_reserve(
    address: *mut *mut c_void,
    size: usize,
    requested: u64,
    flags: u64,
) -> Status {
    // SAFETY: This entry point is the default-alignment form of the same ABI.
    let page = match current_host_page() {
        Ok(page) => page,
        Err(status) => return status,
    };
    unsafe { hsa_amd_vmem_address_reserve_align(address, size, requested, page as u64, flags) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_address_reserve_align(
    address: *mut *mut c_void,
    size: usize,
    requested: u64,
    alignment: u64,
    flags: u64,
) -> Status {
    boundary(|| {
        let page = match current_host_page() {
            Ok(page) => page,
            Err(status) => return status,
        };
        if address.is_null()
            || size == 0
            || size % page != 0
            || alignment < page as u64
            || !alignment.is_power_of_two()
            || flags & !VMEM_ADDRESS_NO_REGISTER != 0
        {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let devices = runtime
            .gpus
            .iter()
            .map(|gpu| &gpu.device)
            .collect::<Vec<_>>();
        let reservation = match runtime.session.reserve_virtual_address(
            &devices,
            size as u64,
            alignment,
            requested,
        ) {
            Ok(reservation) => reservation,
            Err(error) => return map_error(error),
        };
        let base = reservation.info().address as usize;
        if runtime.vmem_reservations.contains_key(&base) {
            return ERROR;
        }
        runtime.vmem_reservations.insert(
            base,
            VmemReservation::new(reservation, flags & VMEM_ADDRESS_NO_REGISTER == 0),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { address.write(base as *mut c_void) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_vmem_address_free(address: *mut c_void, size: usize) -> Status {
    boundary(|| {
        if address.is_null() || size == 0 {
            return INVALID_ARGUMENT;
        }
        let callbacks = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            let key = address as usize;
            let Some(mut reservation) = runtime.vmem_reservations.remove(&key) else {
                return INVALID_ALLOCATION;
            };
            if reservation.description.size != size {
                runtime.vmem_reservations.insert(key, reservation);
                return INVALID_ARGUMENT;
            }
            if reservation.mappings != 0 {
                runtime.vmem_reservations.insert(key, reservation);
                return RESOURCE_FREE;
            }
            if let Err(error) = reservation.free() {
                runtime.vmem_reservations.insert(key, reservation);
                return map_error(error);
            }
            std::mem::take(&mut reservation.deallocation_callbacks)
        };
        for registered in callbacks {
            // SAFETY: The caller registered this callback and data for the live
            // reservation; release notification is synchronous.
            unsafe {
                (registered.callback)(
                    registered.pointer as *mut c_void,
                    registered.user_data as *mut c_void,
                )
            };
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_handle_create(
    pool: HsaMemoryPool,
    size: usize,
    memory_type: u32,
    flags: u64,
    memory_handle: *mut HsaAmdVmemAllocHandle,
) -> Status {
    boundary(|| {
        let page = match current_host_page() {
            Ok(page) => page,
            Err(status) => return status,
        };
        if size == 0
            || size % page != 0
            || !matches!(memory_type, MEMORY_TYPE_NONE | MEMORY_TYPE_PINNED)
            || flags & !u64::from(ALLOC_UNCACHED) != 0
            || memory_handle.is_null()
        {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let (device_index, kind) = if cpu_pool(pool) {
            (0, MemoryKind::System)
        } else if let Some((index, pool_kind)) = runtime.decode_gpu_pool(pool) {
            if pool_kind == GPU_POOL_GROUP {
                return INVALID_ALLOCATION;
            }
            (
                index,
                MemoryKind::DeviceLocal {
                    host_visible: false,
                    coherent: pool_kind == GPU_POOL_FINE,
                    uncached: flags & u64::from(ALLOC_UNCACHED) != 0,
                    contiguous: false,
                },
            )
        } else {
            return INVALID_ARGUMENT;
        };
        if runtime.vmem_handles.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let memory = match runtime.gpus[device_index].device.create_virtual_memory(
            kind,
            size as u64,
            memory_type == MEMORY_TYPE_PINNED,
            flags & u64::from(ALLOC_UNCACHED) != 0,
        ) {
            Ok(memory) => memory,
            Err(error) => return map_error(error),
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime
            .vmem_handles
            .insert(handle, VmemHandle::new(memory, pool, memory_type, false));
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_vmem_handle_release(memory_handle: HsaAmdVmemAllocHandle) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(handle) = runtime.vmem_handles.get_mut(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        if handle.references == 0 {
            return INVALID_ALLOCATION;
        }
        handle.references -= 1;
        if handle.references != 0 || handle.mappings != 0 {
            return SUCCESS;
        }
        let Some(mut handle) = runtime.vmem_handles.remove(&memory_handle.handle) else {
            return ERROR;
        };
        if let Err(error) = handle.free() {
            handle.references = 1;
            runtime.vmem_handles.insert(memory_handle.handle, handle);
            return map_error(error);
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_vmem_map(
    address: *mut c_void,
    size: usize,
    offset: usize,
    memory_handle: HsaAmdVmemAllocHandle,
    flags: u64,
) -> Status {
    boundary(|| {
        let page = match current_host_page() {
            Ok(page) => page,
            Err(status) => return status,
        };
        if address.is_null() || size == 0 || size % page != 0 || offset % page != 0 || flags != 0 {
            return INVALID_ARGUMENT;
        }
        let base = address as usize;
        let end = match base.checked_add(size) {
            Some(end) => end,
            None => return INVALID_ARGUMENT,
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(reservation) = vmem_reservation_base(runtime, base, size) else {
            return INVALID_ARGUMENT;
        };
        let Some(handle) = runtime.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ARGUMENT;
        };
        if offset
            .checked_add(size)
            .is_none_or(|extent| extent > handle.memory.info().size as usize)
        {
            return INVALID_ARGUMENT;
        }
        if runtime
            .vmem_mappings
            .range(..=base)
            .next_back()
            .is_some_and(|(&mapped, mapping)| {
                mapped != base
                    && mapped
                        .checked_add(mapping.size)
                        .is_none_or(|mapped_end| mapped_end > base)
            })
            || runtime
                .vmem_mappings
                .range(base..)
                .next()
                .is_some_and(|(&mapped, _)| mapped < end)
        {
            return INVALID_ARGUMENT;
        }
        let Some(handle) = runtime.vmem_handles.get_mut(&memory_handle.handle) else {
            return ERROR;
        };
        handle.mappings = match handle.mappings.checked_add(1) {
            Some(count) => count,
            None => return OUT_OF_RESOURCES,
        };
        let Some(reservation_owner) = runtime.vmem_reservations.get_mut(&reservation) else {
            handle.mappings -= 1;
            return ERROR;
        };
        let Some(reservation_mappings) = reservation_owner.mappings.checked_add(1) else {
            handle.mappings -= 1;
            return OUT_OF_RESOURCES;
        };
        reservation_owner.mappings = reservation_mappings;
        runtime.vmem_mappings.insert(
            base,
            VmemMapping {
                handle: memory_handle.handle,
                reservation,
                address: base,
                offset,
                size,
                access: HashMap::new(),
            },
        );
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_amd_vmem_unmap(address: *mut c_void, size: usize) -> Status {
    boundary(|| {
        if address.is_null() || size == 0 {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let bases = match vmem_mapping_bases(runtime, address as usize, size) {
            Ok(bases) => bases,
            Err(status) => return status,
        };
        if bases.iter().any(|&base| copy_borrowed(runtime, base)) {
            return OUT_OF_RESOURCES;
        }
        for base in bases {
            let Some(mut mapping) = runtime.vmem_mappings.remove(&base) else {
                return ERROR;
            };
            if let Err(error) = mapping.free_access() {
                runtime.vmem_mappings.insert(base, mapping);
                return map_error(error);
            }
            if let Some(reservation) = runtime.vmem_reservations.get_mut(&mapping.reservation) {
                reservation.mappings = reservation.mappings.saturating_sub(1);
            }
            let release_handle = if let Some(handle) = runtime.vmem_handles.get_mut(&mapping.handle)
            {
                handle.mappings = handle.mappings.saturating_sub(1);
                handle.mappings == 0 && handle.references == 0
            } else {
                false
            };
            if release_handle {
                let Some(mut handle) = runtime.vmem_handles.remove(&mapping.handle) else {
                    return ERROR;
                };
                if let Err(error) = handle.free() {
                    runtime.vmem_handles.insert(mapping.handle, handle);
                    return map_error(error);
                }
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_set_access(
    address: *mut c_void,
    size: usize,
    descriptors: *const HsaAmdMemoryAccessDesc,
    descriptor_count: usize,
) -> Status {
    boundary(|| {
        if address.is_null()
            || size == 0
            || descriptors.is_null()
            || descriptor_count == 0
            || descriptor_count > isize::MAX as usize / size_of::<HsaAmdMemoryAccessDesc>()
        {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let mut requested = Vec::new();
        if requested.try_reserve_exact(descriptor_count).is_err() {
            return OUT_OF_RESOURCES;
        }
        // A CPU access change can remove the mapping that holds the caller's
        // descriptors. Own them before the first native access update.
        // SAFETY: The caller supplies descriptor_count readable entries.
        requested.extend_from_slice(unsafe {
            std::slice::from_raw_parts(descriptors, descriptor_count)
        });
        for descriptor in &requested {
            if !runtime.is_agent(descriptor.agent_handle) {
                return INVALID_AGENT;
            }
            if !matches!(
                descriptor.permissions,
                ACCESS_PERMISSION_NONE
                    | ACCESS_PERMISSION_RO
                    | ACCESS_PERMISSION_WO
                    | ACCESS_PERMISSION_RW
            ) {
                return INVALID_ARGUMENT;
            }
        }
        let bases = match vmem_mapping_bases(runtime, address as usize, size) {
            Ok(bases) => bases,
            Err(status) => return status,
        };
        if bases.iter().any(|&base| copy_borrowed(runtime, base)) {
            return OUT_OF_RESOURCES;
        }
        for base in bases {
            for descriptor in &requested {
                let status = replace_vmem_access(runtime, base, *descriptor);
                if status != SUCCESS {
                    return status;
                }
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_get_access(
    address: *mut c_void,
    permissions: *mut u32,
    agent: HsaAgent,
) -> Status {
    boundary(|| {
        if address.is_null() || permissions.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !runtime.is_agent(agent) {
            return INVALID_AGENT;
        }
        let Some((_, mapping)) = runtime
            .vmem_mappings
            .range(..=address as usize)
            .next_back()
            .filter(|(_, mapping)| mapping.contains(address as usize))
        else {
            return INVALID_ALLOCATION;
        };
        let value = mapping
            .access
            .get(&agent.handle)
            .map_or(ACCESS_PERMISSION_NONE, |access| access.permissions);
        // SAFETY: The caller supplied writable output storage.
        unsafe { permissions.write(value) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_export_shareable_handle(
    descriptor: *mut i32,
    memory_handle: HsaAmdVmemAllocHandle,
    flags: u64,
) -> Status {
    boundary(|| {
        if descriptor.is_null() || flags & !DMABUF_MAPPING_PCIE != 0 {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        if handle.imported {
            return INCOMPATIBLE_ARGUMENTS;
        }
        let dma_buf = match linux_interop::export_virtual_memory(&handle.memory) {
            Ok(dma_buf) => dma_buf,
            Err(error) => return map_error(error),
        };
        // SAFETY: The caller supplied writable output storage and ownership of
        // the descriptor is transferred only after export succeeds.
        unsafe { descriptor.write(dma_buf.into_fd().into_raw_fd()) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_import_shareable_handle(
    descriptor: i32,
    memory_handle: *mut HsaAmdVmemAllocHandle,
) -> Status {
    boundary(|| {
        if descriptor < 0 || memory_handle.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.vmem_handles.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let descriptor = match linux_interop::duplicate_descriptor(descriptor) {
            Ok(descriptor) => descriptor,
            Err(error) => return map_error(error),
        };
        let memory =
            match linux_interop::import_virtual_memory(&runtime.session, descriptor.as_fd()) {
                Ok(memory) => memory,
                Err(error) => return map_error(error),
            };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.vmem_handles.insert(
            handle,
            VmemHandle::new(memory, HsaMemoryPool { handle: 0 }, MEMORY_TYPE_NONE, true),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_retain_alloc_handle(
    memory_handle: *mut HsaAmdVmemAllocHandle,
    address: *mut c_void,
) -> Status {
    boundary(|| {
        if memory_handle.is_null() || address.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(mapping) = runtime.vmem_mappings.get(&(address as usize)) else {
            return INVALID_ALLOCATION;
        };
        let handle_id = mapping.handle;
        let Some(handle) = runtime.vmem_handles.get_mut(&handle_id) else {
            return INVALID_ALLOCATION;
        };
        handle.references = match handle.references.checked_add(1) {
            Some(references) => references,
            None => return OUT_OF_RESOURCES,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle: handle_id }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_get_alloc_properties_from_handle(
    memory_handle: HsaAmdVmemAllocHandle,
    pool: *mut HsaMemoryPool,
    memory_type: *mut u32,
) -> Status {
    boundary(|| {
        if pool.is_null() || memory_type.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        // SAFETY: Both outputs were validated and are written only after lookup.
        unsafe {
            pool.write(handle.pool);
            memory_type.write(handle.memory_type);
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_export_fabric_handle(
    fabric_handle: *mut HsaFabricHandle,
    memory_handle: HsaAmdVmemAllocHandle,
    _flags: u64,
) -> Status {
    boundary(|| {
        if fabric_handle.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !runtime.vmem_handles.contains_key(&memory_handle.handle) {
            return INVALID_ALLOCATION;
        }
        NOT_SUPPORTED
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_vmem_import_fabric_handle(
    _fabric_handle: HsaFabricHandle,
    memory_handle: *mut HsaAmdVmemAllocHandle,
) -> Status {
    boundary(|| {
        if memory_handle.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.is_none() {
            NOT_INITIALIZED
        } else {
            NOT_SUPPORTED
        }
    })
}

fn pointer_description(runtime: &Runtime, address: usize) -> Option<PointerDescription> {
    vmem_pointer_description(runtime, address).or_else(|| {
        runtime
            .allocations
            .values()
            .find(|memory| memory.contains(address))
            .map(|memory| memory.description.clone())
            .or_else(|| {
                runtime
                    .ipc_allocations
                    .values()
                    .find(|memory| memory.contains(address))
                    .map(|memory| memory.description.clone())
            })
            .or_else(|| {
                runtime
                    .interop_allocations
                    .values()
                    .find(|memory| memory.contains(address))
                    .map(|memory| memory.description.clone())
            })
            .or_else(|| {
                runtime
                    .locked_allocations
                    .iter()
                    .rev()
                    .find(|memory| memory.contains(address))
                    .map(|memory| memory.description.clone())
            })
    })
}

fn vmem_pointer_description(runtime: &Runtime, address: usize) -> Option<PointerDescription> {
    if let Some((_, mapping)) = runtime
        .vmem_mappings
        .range(..=address)
        .next_back()
        .filter(|(_, mapping)| mapping.contains(address))
    {
        let handle = runtime.vmem_handles.get(&mapping.handle)?;
        let reservation = runtime.vmem_reservations.get(&mapping.reservation)?;
        let accessible = mapping
            .access
            .iter()
            .filter(|(_, access)| access.permissions != ACCESS_PERMISSION_NONE)
            .map(|(&agent, _)| HsaAgent { handle: agent })
            .collect();
        let host_access = mapping
            .access
            .get(&CPU_AGENT)
            .is_some_and(|access| access.permissions != ACCESS_PERMISSION_NONE);
        let owner = pool_owner(runtime, handle.pool).unwrap_or(HsaAgent { handle: 0 });
        let global_flags = pool_global_flags(runtime, handle.pool).unwrap_or(0);
        return Some(PointerDescription {
            pointer_type: POINTER_TYPE_HSA_VMEM,
            agent_base: mapping.address,
            host_base: host_access.then_some(mapping.address),
            size: mapping.size,
            owner,
            global_flags,
            registered: true,
            alloc_flags: if cpu_pool(handle.pool) {
                POINTER_ALLOC_NONPAGED
            } else {
                0
            } | if host_access {
                POINTER_ALLOC_HOST_ACCESS
            } else {
                0
            },
            user_data: reservation.description.user_data,
            accessible,
        });
    }
    runtime
        .vmem_reservations
        .range(..=address)
        .next_back()
        .filter(|(_, reservation)| reservation.contains_range(address, 1))
        .map(|(_, reservation)| reservation.description.clone())
}

fn host_copy_address(
    runtime: &Runtime,
    address: usize,
    size: usize,
    required_access: u32,
) -> Result<usize, Status> {
    address.checked_add(size).ok_or(INVALID_ARGUMENT)?;
    if let Some(description) = pointer_description(runtime, address) {
        if description.pointer_type == POINTER_TYPE_RESERVED_ADDR {
            return Err(INVALID_ALLOCATION);
        }
        vmem_copy_access(
            runtime,
            &description,
            HsaAgent { handle: CPU_AGENT },
            required_access,
        )?;
        return description
            .host_range(address, size)
            .ok_or(INVALID_ALLOCATION);
    }
    crate::loader::loaded_host_range(runtime, address, size).map(|host| host.unwrap_or(address))
}

fn memory_copy_has_work(
    destination: *mut c_void,
    source: *const c_void,
    size: usize,
) -> Result<bool, Status> {
    if destination.is_null() || source.is_null() || size > isize::MAX as usize {
        return Err(INVALID_ARGUMENT);
    }
    Ok(size != 0)
}

fn memory_fill_size(pointer: *mut c_void, count: usize) -> Result<Option<usize>, Status> {
    if pointer.is_null() || (pointer as usize) % align_of::<u32>() != 0 {
        return Err(INVALID_ARGUMENT);
    }
    if count == 0 {
        return Ok(None);
    }
    count
        .checked_mul(size_of::<u32>())
        .filter(|size| isize::try_from(*size).is_ok())
        .map(Some)
        .ok_or(INVALID_ARGUMENT)
}

unsafe fn write_pointer_info(
    destination: *mut HsaAmdPointerInfo,
    input_size: u32,
    description: Option<&PointerDescription>,
) {
    let returned_size = usize::try_from(input_size)
        .unwrap_or(usize::MAX)
        .min(size_of::<HsaAmdPointerInfo>());
    let mut output = HsaAmdPointerInfo {
        size: returned_size as u32,
        pointer_type: POINTER_TYPE_UNKNOWN,
        agent_base_address: std::ptr::null_mut(),
        host_base_address: std::ptr::null_mut(),
        size_in_bytes: 0,
        user_data: std::ptr::null_mut(),
        agent_owner: HsaAgent { handle: 0 },
        global_flags: 0,
        registered: false,
        registered_padding: [0; 3],
        alloc_flags: 0,
        tail_padding: [0; 4],
    };
    if let Some(description) = description {
        output.pointer_type = description.pointer_type;
        output.agent_base_address = description.agent_base as *mut c_void;
        output.host_base_address = description
            .host_base
            .map_or(std::ptr::null_mut(), |address| address as *mut c_void);
        output.size_in_bytes = description.size;
        output.agent_owner = description.owner;
        output.global_flags = description.global_flags;
        output.registered = description.registered;
        output.alloc_flags = description.alloc_flags;
        output.user_data = description.user_data as *mut c_void;
    }
    // SAFETY: The caller supplies input_size writable bytes and returned_size is
    // bounded by both that value and the supported structure size.
    unsafe {
        std::ptr::copy_nonoverlapping(
            (&raw const output).cast::<u8>(),
            destination.cast::<u8>(),
            returned_size,
        );
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_pointer_info(
    pointer: *const c_void,
    info: *mut HsaAmdPointerInfo,
    allocator: PointerAllocator,
    num_agents_accessible: *mut u32,
    accessible: *mut *mut HsaAgent,
) -> Status {
    boundary(|| {
        if pointer.is_null() || info.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires info to reference initialized size storage.
        let input_size = unsafe { info.cast::<u32>().read() };
        if input_size == 0 {
            return INVALID_ARGUMENT;
        }
        let description = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match guard.as_ref() {
                Some(runtime) => runtime,
                None => return NOT_INITIALIZED,
            };
            pointer_description(runtime, pointer as usize)
        };
        // SAFETY: The caller supplies input_size writable bytes at info.
        unsafe { write_pointer_info(info, input_size, description.as_ref()) };

        let (Some(description), Some(allocator)) = (description, allocator) else {
            return SUCCESS;
        };
        if num_agents_accessible.is_null() || accessible.is_null() {
            return SUCCESS;
        }
        let count = match u32::try_from(description.accessible.len()) {
            Ok(count) => count,
            Err(_) => return OUT_OF_RESOURCES,
        };
        // SAFETY: Both output pointers were checked above.
        unsafe { num_agents_accessible.write(count) };
        if description.accessible.is_empty() {
            // SAFETY: The caller supplied writable pointer storage.
            unsafe { accessible.write(std::ptr::null_mut()) };
            return SUCCESS;
        }
        let Some(bytes) = description
            .accessible
            .len()
            .checked_mul(size_of::<HsaAgent>())
        else {
            return OUT_OF_RESOURCES;
        };
        // SAFETY: The allocator is supplied by the caller for this exact purpose.
        let storage = unsafe { allocator(bytes) }.cast::<HsaAgent>();
        if storage.is_null() {
            return OUT_OF_RESOURCES;
        }
        // SAFETY: The allocator returned storage for every accessible handle and
        // the caller supplied writable output pointer storage.
        unsafe {
            std::ptr::copy_nonoverlapping(
                description.accessible.as_ptr(),
                storage,
                description.accessible.len(),
            );
            accessible.write(storage);
        }
        SUCCESS
    })
}

fn set_pointer_userdata(runtime: &mut Runtime, address: usize, user_data: usize) -> bool {
    if let Some((_, reservation)) = runtime
        .vmem_reservations
        .range_mut(..=address)
        .next_back()
        .filter(|(_, reservation)| reservation.contains_range(address, 1))
    {
        reservation.description.user_data = user_data;
        return true;
    }
    for memory in runtime.allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.ipc_allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.interop_allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.locked_allocations.iter_mut().rev() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    false
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_pointer_info_set_userdata(
    pointer: *const c_void,
    user_data: *mut c_void,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        if set_pointer_userdata(runtime, pointer as usize, user_data as usize) {
            SUCCESS
        } else {
            INVALID_ARGUMENT
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_agents_allow_access(
    num_agents: u32,
    agents: *const HsaAgent,
    flags: *const u32,
    pointer: *const c_void,
) -> Status {
    boundary(|| {
        if num_agents == 0 || agents.is_null() || !flags.is_null() || pointer.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        // SAFETY: The caller supplies num_agents readable entries.
        let agents = unsafe { std::slice::from_raw_parts(agents, num_agents as usize) };
        if agents.iter().any(|agent| !runtime.is_agent(*agent)) {
            return INVALID_AGENT;
        }
        let Some(key) = runtime
            .allocations
            .iter()
            .find(|(_, memory)| memory.contains(pointer as usize))
            .map(|(key, _)| *key)
        else {
            // This API is defined for memory-pool allocations. IPC, graphics,
            // and locked ranges need their own native mapping lifecycle.
            return INVALID_ALLOCATION;
        };
        let memory = &runtime.allocations[&key];
        if copy_borrowed(runtime, memory.description.agent_base) {
            return OUT_OF_RESOURCES;
        }
        let owner = memory.description.owner;
        let storage = if owner.handle == CPU_AGENT {
            PoolStorage::System
        } else if memory.description.global_flags & POOL_FLAG_FINE != 0 {
            PoolStorage::LocalFine
        } else {
            PoolStorage::LocalCoarse
        };
        if agents.iter().any(|agent| {
            agent_pool_relationship(runtime, *agent, owner, storage).0 == POOL_ACCESS_NEVER
                || (agent.handle == CPU_AGENT && memory.description.host_base.is_none())
        }) {
            return INVALID_ARGUMENT;
        }
        let mut desired = Vec::new();
        if desired.try_reserve(agents.len().saturating_add(1)).is_err() {
            return OUT_OF_RESOURCES;
        }
        for agent in agents.iter().copied().chain(std::iter::once(owner)) {
            if !desired.contains(&agent) {
                desired.push(agent);
            }
        }
        let mut devices = Vec::new();
        if devices.try_reserve(desired.len()).is_err() {
            return OUT_OF_RESOURCES;
        }
        for agent in &desired {
            if let Some(index) = runtime.gpu_index(*agent) {
                devices.push(&runtime.gpus[index].device);
            }
        }
        let Some(origin) = runtime
            .gpus
            .iter()
            .map(|gpu| &gpu.device)
            .find(|device| memory.allocation.originates_from(device))
        else {
            return ERROR;
        };
        let Some(memory) = runtime.allocations.get_mut(&key) else {
            return ERROR;
        };
        match memory.allocation.set_device_access(origin, &devices) {
            Ok(()) => {
                memory.description.accessible = desired;
                SUCCESS
            }
            Err(error) => {
                // Native calls can complete a prefix before reporting failure.
                // Report only mappings still known to be available.
                let cpu = HsaAgent { handle: CPU_AGENT };
                let cpu_accessible = owner == cpu || memory.description.accessible.contains(&cpu);
                memory.description.accessible.clear();
                if memory
                    .description
                    .accessible
                    .try_reserve(runtime.gpus.len() + usize::from(cpu_accessible))
                    .is_ok()
                {
                    if cpu_accessible {
                        memory.description.accessible.push(cpu);
                    }
                    for (index, gpu) in runtime.gpus.iter().enumerate() {
                        if memory.allocation.device_address(&gpu.device).is_ok() {
                            memory.description.accessible.push(HsaAgent {
                                handle: GPU_AGENT_BASE + index as u64,
                            });
                        }
                    }
                }
                match error.kind() {
                    rocddi::ErrorKind::InvalidArgument | rocddi::ErrorKind::Unsupported => {
                        INVALID_ARGUMENT
                    }
                    _ => map_error(error),
                }
            }
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_pool_can_migrate(
    source_pool: HsaMemoryPool,
    destination_pool: HsaMemoryPool,
    result: *mut bool,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if result.is_null() {
            return INVALID_ARGUMENT;
        }
        if pool_owner(runtime, source_pool).is_none()
            || pool_owner(runtime, destination_pool).is_none()
        {
            return INVALID_MEMORY_POOL;
        }
        // ROCr does not currently implement migration between otherwise valid
        // memory pools and publishes a negative result before reporting that.
        unsafe { result.write(false) };
        OUT_OF_RESOURCES
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_migrate(
    pointer: *const c_void,
    memory_pool: HsaMemoryPool,
    flags: u32,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if pointer.is_null() || flags != 0 {
            return INVALID_ARGUMENT;
        }
        if pool_owner(runtime, memory_pool).is_none() {
            return INVALID_MEMORY_POOL;
        }
        OUT_OF_RESOURCES
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_memory_copy(
    dst: *mut c_void,
    src: *const c_void,
    size: usize,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        match memory_copy_has_work(dst, src, size) {
            Ok(false) => return SUCCESS,
            Ok(true) => {}
            Err(status) => return status,
        }
        let host_destination = host_copy_address(runtime, dst as usize, size, ACCESS_PERMISSION_WO);
        let host_source = host_copy_address(runtime, src as usize, size, ACCESS_PERMISSION_RO);
        if let (Ok(destination), Ok(source)) = (host_destination, host_source) {
            // SAFETY: HSA requires both ranges to be valid for size bytes.
            // ptr::copy tolerates overlap. The registry lock retains owned
            // backing throughout this host copy.
            unsafe { std::ptr::copy(source as *const u8, destination as *mut u8, size) };
            fence(Ordering::SeqCst);
            return SUCCESS;
        }

        let resolved = match resolve_sync_gpu_copy(
            runtime,
            dst as usize,
            src as usize,
            size,
            host_destination,
            host_source,
        ) {
            Ok(resolved) => resolved,
            Err(status) => return status,
        };
        let Some(_inflight) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        let mut retained = [None; 2];
        for (slot, base) in retained.iter_mut().zip(resolved.borrowed_memory) {
            if let Some(base) = base {
                if let Err(status) = retain_copy_borrow(runtime, base) {
                    for base in retained.into_iter().flatten() {
                        release_copy_borrow(runtime, base);
                    }
                    return status;
                }
                *slot = Some(base);
            }
        }
        let stop = runtime.stop_workers.clone();
        let quarantine = runtime.async_copy_quarantine.clone();
        drop(guard);

        let (status, may_be_live) = match execute_sync_gpu_copy(resolved.route, &stop) {
            Ok(SyncGpuCopyOutcome::Done) => (SUCCESS, false),
            Ok(SyncGpuCopyOutcome::LoadedDestination { address, bytes }) => {
                (write_loaded_copy_destination(address, &bytes), false)
            }
            Err(failure) => (failure.status, failure.operands_may_be_live),
        };
        if may_be_live {
            quarantine.store(true, Ordering::Release);
        }
        release_async_resources(&[], &retained, may_be_live);
        status
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_fill(
    pointer: *mut c_void,
    value: u32,
    count: usize,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let size = match memory_fill_size(pointer, count) {
            Ok(Some(size)) => size,
            Ok(None) => return SUCCESS,
            Err(status) => return status,
        };
        let Some(description) = pointer_description(runtime, pointer as usize) else {
            let host = match crate::loader::loaded_host_range(runtime, pointer as usize, size) {
                Ok(Some(host)) => host,
                Ok(None) => return INVALID_ALLOCATION,
                Err(status) => return status,
            };
            if host % align_of::<u32>() != 0 {
                return INVALID_ALLOCATION;
            }
            // SAFETY: The registry lock retains the complete writable loaded
            // mapping until the fill returns.
            unsafe { std::slice::from_raw_parts_mut(host as *mut MaybeUninit<u32>, count) }
                .fill(MaybeUninit::new(value));
            fence(Ordering::SeqCst);
            return SUCCESS;
        };
        if description.pointer_type == POINTER_TYPE_RESERVED_ADDR {
            return INVALID_ALLOCATION;
        }
        if let Some(host) = description.host_range(pointer as usize, size) {
            if vmem_copy_access(
                runtime,
                &description,
                HsaAgent { handle: CPU_AGENT },
                ACCESS_PERMISSION_WO,
            )
            .is_ok()
            {
                // SAFETY: The complete aligned output range belongs to a live
                // allocation retained by the runtime lock. Its prior contents
                // need not have been initialized as u32 values.
                unsafe { std::slice::from_raw_parts_mut(host as *mut MaybeUninit<u32>, count) }
                    .fill(MaybeUninit::new(value));
                fence(Ordering::SeqCst);
                return SUCCESS;
            }
        }
        let Some((index, agent)) = selected_copy_gpu(runtime, &description) else {
            return INVALID_ALLOCATION;
        };
        let (address, base) = match gpu_copy_address(
            runtime,
            Some(&description),
            pointer as usize,
            size,
            agent,
            ACCESS_PERMISSION_WO,
        ) {
            Ok(address) => address,
            Err(status) => return status,
        };
        let Ok(native_count) = u64::try_from(count) else {
            return INVALID_ARGUMENT;
        };
        let Some(_inflight) = runtime.inflight.enter() else {
            return OUT_OF_RESOURCES;
        };
        if let Err(status) = retain_copy_borrow(runtime, base) {
            return status;
        }
        let device = runtime.gpus[index].device.clone();
        let stop = runtime.stop_workers.clone();
        let quarantine = runtime.async_copy_quarantine.clone();
        drop(guard);

        let (status, may_be_live) = match device.gpu() {
            Ok(gpu) => {
                // SAFETY: The complete writable GPU range was checked above.
                // Its allocation borrow and in-flight token remain live until
                // retirement or process teardown after an uncertain result.
                match unsafe { gpu.fill_u32(address, value, native_count, &stop) } {
                    Ok(()) => (SUCCESS, false),
                    Err(failure) => (map_error(failure.error), failure.operands_may_be_live),
                }
            }
            Err(error) => (map_error(error), false),
        };
        if may_be_live {
            quarantine.store(true, Ordering::Release);
        }
        release_async_resources(&[], &[Some(base)], may_be_live);
        status
    })
}

fn svm_range(pointer: *mut c_void, size: usize) -> Result<(u64, u64), Status> {
    if pointer.is_null() || size == 0 {
        return Err(INVALID_ARGUMENT);
    }
    let page_size = rocddi::memory::host_page_size().map_err(map_error)?;
    let page_size = usize::try_from(page_size).map_err(|_| INVALID_ARGUMENT)?;
    if page_size == 0 || !page_size.is_power_of_two() {
        return Err(ERROR);
    }
    let address = pointer as usize;
    let base = address & !(page_size - 1);
    let end = address
        .checked_add(size)
        .and_then(|end| end.checked_add(page_size - 1))
        .map(|end| end & !(page_size - 1))
        .ok_or(INVALID_ARGUMENT)?;
    let length = end.checked_sub(base).ok_or(INVALID_ARGUMENT)?;
    Ok((base as u64, length as u64))
}

fn svm_location_from_agent(
    runtime: &Runtime,
    agent: HsaAgent,
    allow_null: bool,
) -> Result<SvmLocation, Status> {
    if agent.handle == 0 && allow_null {
        return Ok(SvmLocation::Undefined);
    }
    if agent.handle == CPU_AGENT {
        return Ok(SvmLocation::System);
    }
    let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
    Ok(SvmLocation::Device(runtime.gpus[index].endpoint.id))
}

fn svm_agent_from_location(runtime: &Runtime, location: SvmLocation) -> Result<HsaAgent, Status> {
    match location {
        SvmLocation::System => Ok(HsaAgent { handle: CPU_AGENT }),
        SvmLocation::Undefined => Ok(HsaAgent { handle: 0 }),
        SvmLocation::Device(identity) => runtime
            .gpus
            .iter()
            .position(|gpu| gpu.endpoint.id == identity)
            .map(|index| HsaAgent {
                handle: GPU_AGENT_BASE + index as u64,
            })
            .ok_or(ERROR),
    }
}

fn reserve_svm_attributes(count: usize) -> Result<(Vec<SvmAttribute>, Vec<Option<usize>>), Status> {
    if count > SVM_MAX_ATTRIBUTES {
        return Err(INVALID_ARGUMENT);
    }
    let mut attributes = Vec::new();
    attributes
        .try_reserve_exact(count.saturating_add(2))
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut indices = Vec::new();
    indices
        .try_reserve_exact(count)
        .map_err(|_| OUT_OF_RESOURCES)?;
    Ok((attributes, indices))
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_svm_attributes_set(
    pointer: *mut c_void,
    size: usize,
    attribute_list: *mut HsaAmdSvmAttributePair,
    attribute_count: usize,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if attribute_count > SVM_MAX_ATTRIBUTES
            || (attribute_count != 0
                && (attribute_list.is_null()
                    || (attribute_list as usize) % align_of::<HsaAmdSvmAttributePair>() != 0))
        {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        let pairs = if attribute_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA ABI requires attribute_count readable pairs.
            unsafe { std::slice::from_raw_parts(attribute_list, attribute_count) }
        };
        let (mut attributes, _) = match reserve_svm_attributes(attribute_count) {
            Ok(storage) => storage,
            Err(status) => return status,
        };
        let mut scalar_attributes = 0_u16;
        let mut agents = Vec::new();
        if agents.try_reserve_exact(attribute_count).is_err() {
            return OUT_OF_RESOURCES;
        }
        let mut set_flags = 0_u32;
        let mut clear_flags = 0_u32;
        for pair in pairs {
            let scalar_bit = match pair.attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG
                | AMD_SVM_ATTRIB_READ_ONLY
                | AMD_SVM_ATTRIB_HIVE_LOCAL
                | AMD_SVM_ATTRIB_MIGRATION_GRANULARITY
                | AMD_SVM_ATTRIB_PREFERRED_LOCATION
                | AMD_SVM_ATTRIB_READ_MOSTLY
                | AMD_SVM_ATTRIB_GPU_EXEC => Some(1_u16 << pair.attribute),
                _ => None,
            };
            if let Some(bit) = scalar_bit {
                if scalar_attributes & bit != 0 {
                    return INCOMPATIBLE_ARGUMENTS;
                }
                scalar_attributes |= bit;
            }
            match pair.attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG => match pair.value {
                    AMD_SVM_GLOBAL_FLAG_FINE_GRAINED => set_flags |= SVM_FLAG_COHERENT,
                    AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED => clear_flags |= SVM_FLAG_COHERENT,
                    _ => return INVALID_ARGUMENT,
                },
                AMD_SVM_ATTRIB_READ_ONLY => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_READ_ONLY;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_READ_ONLY;
                    }
                }
                AMD_SVM_ATTRIB_HIVE_LOCAL => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_HIVE_LOCAL;
                    } else {
                        clear_flags |= SVM_FLAG_HIVE_LOCAL;
                    }
                }
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    attributes.push(SvmAttribute::MigrationGranularity(
                        pair.value.min(SVM_MAX_MIGRATION_GRANULARITY) as u32,
                    ))
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION => {
                    let location = match svm_location_from_agent(
                        runtime,
                        HsaAgent { handle: pair.value },
                        true,
                    ) {
                        Ok(location) => location,
                        Err(status) => return status,
                    };
                    attributes.push(SvmAttribute::PreferredLocation(location));
                }
                AMD_SVM_ATTRIB_READ_MOSTLY => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_READ_MOSTLY;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_READ_MOSTLY;
                    }
                }
                AMD_SVM_ATTRIB_GPU_EXEC => {
                    if pair.value != 0 {
                        set_flags |= SVM_FLAG_GPU_EXECUTE;
                    } else {
                        clear_flags |= SVM_FLAG_GPU_EXECUTE;
                    }
                }
                AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                | AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE
                | AMD_SVM_ATTRIB_AGENT_NO_ACCESS => {
                    let agent = HsaAgent { handle: pair.value };
                    if !runtime.is_agent(agent) {
                        return INVALID_AGENT;
                    }
                    if agents.contains(&agent.handle) {
                        return INCOMPATIBLE_ARGUMENTS;
                    }
                    agents.push(agent.handle);
                    if agent.handle == CPU_AGENT {
                        if pair.attribute == AMD_SVM_ATTRIB_AGENT_NO_ACCESS {
                            clear_flags |= SVM_FLAG_HOST_ACCESS;
                        } else {
                            set_flags |= SVM_FLAG_HOST_ACCESS;
                        }
                    } else {
                        let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT);
                        let index = match index {
                            Ok(index) => index,
                            Err(status) => return status,
                        };
                        let access = match pair.attribute {
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE => SvmAccess::Accessible,
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE => {
                                SvmAccess::AccessibleInPlace
                            }
                            _ => SvmAccess::NoAccess,
                        };
                        attributes.push(SvmAttribute::Access {
                            device: runtime.gpus[index].endpoint.id,
                            access,
                        });
                    }
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        if set_flags & SVM_FLAG_HOST_ACCESS != 0 {
            clear_flags &= !SVM_FLAG_HOST_ACCESS;
        }
        if clear_flags != 0 {
            attributes.push(SvmAttribute::ClearFlags(clear_flags));
        }
        if set_flags != 0 {
            attributes.push(SvmAttribute::SetFlags(set_flags));
        }
        if attributes.len() > SVM_MAX_ATTRIBUTES {
            return INVALID_ARGUMENT;
        }
        linux_interop::set_kfd_svm_attributes(&runtime.session, address, size, &attributes)
            .map_or_else(map_error, |()| SUCCESS)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_svm_attributes_get(
    pointer: *mut c_void,
    size: usize,
    attribute_list: *mut HsaAmdSvmAttributePair,
    attribute_count: usize,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if attribute_count > SVM_MAX_ATTRIBUTES
            || (attribute_count != 0
                && (attribute_list.is_null()
                    || (attribute_list as usize) % align_of::<HsaAmdSvmAttributePair>() != 0))
        {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        let (mut attributes, mut indices) = match reserve_svm_attributes(attribute_count) {
            Ok(storage) => storage,
            Err(status) => return status,
        };
        let mut get_flags = false;
        for index in 0..attribute_count {
            // SAFETY: The caller supplies this complete pair array. Only each
            // attribute tag is initialized for a query; its value is input
            // only for ACCESS_QUERY.
            let pair = unsafe { attribute_list.add(index) };
            let attribute = unsafe { (&raw const (*pair).attribute).read() };
            match attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG
                | AMD_SVM_ATTRIB_READ_ONLY
                | AMD_SVM_ATTRIB_HIVE_LOCAL
                | AMD_SVM_ATTRIB_READ_MOSTLY
                | AMD_SVM_ATTRIB_GPU_EXEC => {
                    get_flags = true;
                    indices.push(None);
                }
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::MigrationGranularity(0));
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::PreferredLocation(SvmLocation::Undefined));
                }
                AMD_SVM_ATTRIB_PREFETCH_LOCATION => {
                    indices.push(Some(attributes.len()));
                    attributes.push(SvmAttribute::PrefetchLocation(SvmLocation::Undefined));
                }
                AMD_SVM_ATTRIB_ACCESS_QUERY => {
                    // SAFETY: ACCESS_QUERY requires the agent value as input.
                    let agent = HsaAgent {
                        handle: unsafe { (&raw const (*pair).value).read() },
                    };
                    if !runtime.is_agent(agent) {
                        return INVALID_AGENT;
                    }
                    if agent.handle == CPU_AGENT {
                        get_flags = true;
                        indices.push(None);
                    } else {
                        let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT);
                        let index = match index {
                            Ok(index) => index,
                            Err(status) => return status,
                        };
                        indices.push(Some(attributes.len()));
                        attributes.push(SvmAttribute::Access {
                            device: runtime.gpus[index].endpoint.id,
                            access: SvmAccess::Accessible,
                        });
                    }
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        let flag_indices = if get_flags {
            let clear = attributes.len();
            attributes.push(SvmAttribute::ClearFlags(0));
            let set = attributes.len();
            attributes.push(SvmAttribute::SetFlags(0));
            Some((clear, set))
        } else {
            None
        };
        if attributes.len() > SVM_MAX_ATTRIBUTES {
            return INVALID_ARGUMENT;
        }
        if let Err(error) =
            linux_interop::get_kfd_svm_attributes(&runtime.session, address, size, &mut attributes)
        {
            return map_error(error);
        }
        let (clear_flags, set_flags) = flag_indices.map_or((0, 0), |(clear, set)| {
            let clear = match attributes.get(clear).copied() {
                Some(SvmAttribute::ClearFlags(value)) => value,
                _ => 0,
            };
            let set = match attributes.get(set).copied() {
                Some(SvmAttribute::SetFlags(value)) => value,
                _ => 0,
            };
            (clear, set)
        });
        for (index, native_index) in indices.iter().copied().enumerate() {
            // SAFETY: The caller retains the complete writable pair array.
            // Read only the initialized tag until an output value is written.
            let pair = unsafe { attribute_list.add(index) };
            let attribute = unsafe { (&raw const (*pair).attribute).read() };
            let value = match attribute {
                AMD_SVM_ATTRIB_GLOBAL_FLAG => {
                    if set_flags & SVM_FLAG_COHERENT != 0 {
                        AMD_SVM_GLOBAL_FLAG_FINE_GRAINED
                    } else if clear_flags & SVM_FLAG_COHERENT != 0 {
                        AMD_SVM_GLOBAL_FLAG_COARSE_GRAINED
                    } else {
                        AMD_SVM_GLOBAL_FLAG_INDETERMINATE
                    }
                }
                AMD_SVM_ATTRIB_READ_ONLY => u64::from(set_flags & SVM_FLAG_GPU_READ_ONLY),
                AMD_SVM_ATTRIB_HIVE_LOCAL => u64::from(set_flags & SVM_FLAG_HIVE_LOCAL),
                AMD_SVM_ATTRIB_READ_MOSTLY => u64::from(set_flags & SVM_FLAG_GPU_READ_MOSTLY),
                AMD_SVM_ATTRIB_GPU_EXEC => u64::from(set_flags & SVM_FLAG_GPU_EXECUTE),
                AMD_SVM_ATTRIB_MIGRATION_GRANULARITY => {
                    let Some(native) = native_index else {
                        return ERROR;
                    };
                    let Some(SvmAttribute::MigrationGranularity(value)) =
                        attributes.get(native).copied()
                    else {
                        return ERROR;
                    };
                    u64::from(value)
                }
                AMD_SVM_ATTRIB_PREFERRED_LOCATION | AMD_SVM_ATTRIB_PREFETCH_LOCATION => {
                    let Some(native) = native_index else {
                        return ERROR;
                    };
                    let location = match attributes.get(native).copied() {
                        Some(SvmAttribute::PreferredLocation(location))
                            if attribute == AMD_SVM_ATTRIB_PREFERRED_LOCATION =>
                        {
                            location
                        }
                        Some(SvmAttribute::PrefetchLocation(location))
                            if attribute == AMD_SVM_ATTRIB_PREFETCH_LOCATION =>
                        {
                            location
                        }
                        _ => return ERROR,
                    };
                    let agent = match svm_agent_from_location(runtime, location) {
                        Ok(agent) => agent,
                        Err(status) => return status,
                    };
                    agent.handle
                }
                AMD_SVM_ATTRIB_ACCESS_QUERY => {
                    // SAFETY: ACCESS_QUERY retains its initialized agent input.
                    let agent = HsaAgent {
                        handle: unsafe { (&raw const (*pair).value).read() },
                    };
                    let result = if agent.handle == CPU_AGENT {
                        if set_flags & SVM_FLAG_HOST_ACCESS != 0 {
                            AMD_SVM_ATTRIB_AGENT_ACCESSIBLE
                        } else {
                            AMD_SVM_ATTRIB_AGENT_NO_ACCESS
                        }
                    } else {
                        let Some(native) = native_index else {
                            return ERROR;
                        };
                        match attributes.get(native).copied() {
                            Some(SvmAttribute::Access {
                                access: SvmAccess::Accessible,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_ACCESSIBLE,
                            Some(SvmAttribute::Access {
                                access: SvmAccess::AccessibleInPlace,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_ACCESSIBLE_IN_PLACE,
                            Some(SvmAttribute::Access {
                                access: SvmAccess::NoAccess,
                                ..
                            }) => AMD_SVM_ATTRIB_AGENT_NO_ACCESS,
                            _ => return ERROR,
                        }
                    };
                    // SAFETY: The pair's initialized tag is the output field
                    // for ACCESS_QUERY; its input agent value is preserved.
                    unsafe { (&raw mut (*pair).attribute).write(result) };
                    continue;
                }
                _ => return INVALID_ARGUMENT,
            };
            // SAFETY: The caller supplied writable value storage. Its prior
            // bytes need not have been initialized for output-only queries.
            unsafe { (&raw mut (*pair).value).write(value) };
        }
        SUCCESS
    })
}

/// Rolls back signal retention if worker creation fails or unwinds.
struct PendingAsyncSignals<'a> {
    runtime: &'a mut Runtime,
    handles: Arc<Vec<HsaSignal>>,
    retained: usize,
    borrowed_memory: Vec<usize>,
}

impl<'a> PendingAsyncSignals<'a> {
    fn new(runtime: &'a mut Runtime, handles: Arc<Vec<HsaSignal>>) -> Result<Self, Status> {
        let mut pending = Self {
            runtime,
            handles,
            retained: 0,
            borrowed_memory: Vec::new(),
        };
        for signal in pending.handles.iter() {
            pending.runtime.retain_async_signal(*signal)?;
            pending.retained += 1;
        }
        Ok(pending)
    }

    fn retain_memory(&mut self, base: usize) -> Result<(), Status> {
        self.borrowed_memory
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        retain_copy_borrow(self.runtime, base)?;
        self.borrowed_memory.push(base);
        Ok(())
    }

    fn transfer_to_worker(&mut self) {
        self.retained = 0;
        self.borrowed_memory.clear();
    }
}

fn retain_copy_borrow(runtime: &mut Runtime, base: usize) -> Result<(), Status> {
    if !runtime.async_copy_borrows.contains_key(&base) {
        runtime
            .async_copy_borrows
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
    }
    let count = runtime.async_copy_borrows.entry(base).or_default();
    *count = count.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
    Ok(())
}

fn release_copy_borrow(runtime: &mut Runtime, base: usize) {
    if let Some(count) = runtime.async_copy_borrows.get_mut(&base) {
        *count -= 1;
        if *count == 0 {
            runtime.async_copy_borrows.remove(&base);
        }
    }
}

fn copy_borrowed(runtime: &Runtime, base: usize) -> bool {
    runtime
        .async_copy_borrows
        .get(&base)
        .is_some_and(|count| *count != 0)
}

impl Drop for PendingAsyncSignals<'_> {
    fn drop(&mut self) {
        for signal in self.handles.iter().take(self.retained) {
            self.runtime.release_async_signal(*signal);
        }
        for &base in &self.borrowed_memory {
            release_copy_borrow(self.runtime, base);
        }
    }
}

fn release_async_resources(
    handles: &[HsaSignal],
    borrowed_memory: &[Option<usize>],
    retain_memory: bool,
) {
    if let Ok(mut guard) = lock() {
        if let Some(runtime) = guard.as_mut() {
            for signal in handles {
                runtime.release_async_signal(*signal);
            }
            if !retain_memory {
                for &base in borrowed_memory.iter().flatten() {
                    release_copy_borrow(runtime, base);
                }
            }
        }
    }
    // Shutdown removes the runtime from the registry before joining workers;
    // it retains all signal storage until after this worker exits.
}

fn finish_async_signal(completion: HsaSignal, success: bool) {
    if completion.handle == 0 {
        return;
    }
    // SAFETY: Submission retains this signal until the worker exits, and
    // shutdown keeps its storage mapped until all workers have joined.
    let _ = unsafe {
        crate::signal::with_signal(completion, |signal| {
            if success {
                signal.value.fetch_sub(1, Ordering::Release);
            } else {
                signal.value.store(-1, Ordering::Release);
            }
        })
    };
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "the worker owns the session and stop token for its full asynchronous lifetime"
)]
fn svm_prefetch_worker(
    session: Session,
    stop: Arc<AtomicBool>,
    dependencies: &[HsaSignal],
    completion: HsaSignal,
    address: u64,
    size: u64,
    location: SvmLocation,
) {
    if !wait_async_dependencies(dependencies, &stop) {
        finish_async_signal(completion, false);
        return;
    }
    let result = linux_interop::set_kfd_svm_attributes(
        &session,
        address,
        size,
        &[SvmAttribute::PrefetchLocation(location)],
    );
    finish_async_signal(completion, result.is_ok());
}

fn wait_async_dependencies(dependencies: &[HsaSignal], stop: &AtomicBool) -> bool {
    for &dependency in dependencies {
        loop {
            if stop.load(Ordering::Acquire) {
                return false;
            }
            // SAFETY: Submission retains every dependency until the worker
            // exits; final shutdown joins workers before releasing storage.
            let Some(ready) = (unsafe {
                crate::signal::with_signal(dependency, |signal| {
                    signal.value.load(Ordering::Acquire) == 0
                })
            }) else {
                return false;
            };
            if ready {
                break;
            }
            thread::sleep(Duration::from_micros(20));
        }
    }
    true
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_svm_prefetch_async(
    pointer: *mut c_void,
    size: usize,
    agent: HsaAgent,
    dependency_count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if dependency_count != 0 && dependencies.is_null() {
            return INVALID_ARGUMENT;
        }
        let (address, size) = match svm_range(pointer, size) {
            Ok(range) => range,
            Err(status) => return status,
        };
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA ABI requires dependency_count readable handles.
            unsafe { std::slice::from_raw_parts(dependencies, dependency_count as usize) }
        };
        let location = match svm_location_from_agent(runtime, agent, false) {
            Ok(location) => location,
            Err(status) => return status,
        };
        if dependency_slice
            .iter()
            .any(|dependency| !runtime.owns_signal(*dependency))
            || (completion.handle != 0 && !runtime.owns_signal(completion))
        {
            return INVALID_SIGNAL;
        }
        let mut signal_handles = Vec::new();
        if signal_handles
            .try_reserve_exact(dependency_slice.len() + usize::from(completion.handle != 0))
            .is_err()
            || runtime.workers.try_reserve(1).is_err()
        {
            return OUT_OF_RESOURCES;
        }
        signal_handles.extend_from_slice(dependency_slice);
        if completion.handle != 0 {
            signal_handles.push(completion);
        }
        let mut pending = match PendingAsyncSignals::new(runtime, Arc::new(signal_handles)) {
            Ok(pending) => pending,
            Err(status) => return status,
        };
        let session = pending.runtime.session.clone();
        let stop = pending.runtime.stop_workers.clone();
        let worker_handles = pending.handles.clone();
        let dependency_len = dependency_slice.len();
        let worker = match thread::Builder::new()
            .name("rocddi-svm-prefetch".into())
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    svm_prefetch_worker(
                        session,
                        stop,
                        &worker_handles[..dependency_len],
                        completion,
                        address,
                        size,
                        location,
                    );
                }));
                if result.is_err() {
                    finish_async_signal(completion, false);
                }
                release_async_resources(&worker_handles, &[], false);
            }) {
            Ok(worker) => worker,
            Err(_) => return OUT_OF_RESOURCES,
        };
        pending.transfer_to_worker();
        pending.runtime.workers.push(worker);
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_svm_discard_batch_async(
    pointers: *mut *mut c_void,
    sizes: *mut usize,
    count: u32,
    _dependency_count: u32,
    _dependencies: *const HsaSignal,
    _completion: HsaSignal,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.is_none() {
            return NOT_INITIALIZED;
        }
        if pointers.is_null() || sizes.is_null() || count == 0 {
            return INVALID_ARGUMENT;
        }
        // GFX1201 reports XNACK disabled through hsa_system_get_info, so ROCr
        // rejects this operation before inspecting dependencies or ranges.
        XNACK_DISABLED
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_svm_discard_and_prefetch_batch_async(
    pointers: *mut *mut c_void,
    sizes: *mut usize,
    count: u32,
    destination_agents: *const HsaAgent,
    destination_count: u32,
    _dependency_count: u32,
    _dependencies: *const HsaSignal,
    _completion: HsaSignal,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.is_none() {
            return NOT_INITIALIZED;
        }
        if pointers.is_null()
            || sizes.is_null()
            || count == 0
            || destination_agents.is_null()
            || destination_count == 0
        {
            return INVALID_ARGUMENT;
        }
        // See hsa_amd_svm_discard_batch_async: XNACK is unavailable on the
        // currently supported target, matching ROCr's required status.
        XNACK_DISABLED
    })
}

enum AsyncCopyRoute {
    Noop,
    Host {
        destination: HostCopyDestination,
        source: HostCopySource,
        size: usize,
    },
    GpuLinear {
        device: Box<Device>,
        index: usize,
        destination: u64,
        source: u64,
        size: u64,
    },
    GpuFromHost {
        device: Box<Device>,
        index: usize,
        destination: u64,
        source: HostCopySource,
    },
    GpuToHost {
        device: Box<Device>,
        index: usize,
        destination: HostCopyDestination,
        source: u64,
        size: usize,
    },
    GpuRect {
        device: Box<Device>,
        rect: CopyRect,
    },
    GpuBatch {
        device: Box<Device>,
        entries: Vec<GpuBatchEntry>,
    },
}

enum GpuBatchEntry {
    Linear {
        destination: u64,
        source: u64,
        size: u64,
    },
    FromHost {
        destination: u64,
        source: HostCopySource,
    },
    ToHost {
        destination: HostCopyDestination,
        source: u64,
        size: usize,
    },
}

enum SyncGpuCopyRoute {
    Direct {
        device: Box<Device>,
        destination: u64,
        source: u64,
        size: u64,
    },
    FromHost {
        device: Box<Device>,
        destination: u64,
        source: HostCopySource,
    },
    ToHost {
        device: Box<Device>,
        destination: HostCopyDestination,
        source: u64,
        size: usize,
    },
}

enum HostCopySource {
    Address(usize, usize),
    LoadedAddress(usize, usize),
    LoadedBytes(Vec<u8>),
}

#[derive(Clone, Copy)]
enum HostSourceTiming {
    Immediate,
    AfterDependencies,
}

enum HostCopyDestination {
    Address(usize),
    LoadedAddress(usize),
}

enum SyncGpuCopyOutcome {
    Done,
    LoadedDestination { address: usize, bytes: Vec<u8> },
}

struct CopyRouteFailure {
    status: Status,
    operands_may_be_live: bool,
}

impl CopyRouteFailure {
    fn retired(status: Status) -> Self {
        Self {
            status,
            operands_may_be_live: false,
        }
    }
}

impl From<rocddi::gpu::CopyFailure> for CopyRouteFailure {
    fn from(failure: rocddi::gpu::CopyFailure) -> Self {
        Self {
            status: map_error(failure.error),
            operands_may_be_live: failure.operands_may_be_live,
        }
    }
}

struct ResolvedSyncGpuCopy {
    route: SyncGpuCopyRoute,
    borrowed_memory: [Option<usize>; 2],
}

struct ResolvedAsyncCopy {
    route: AsyncCopyRoute,
    borrowed_memory: [Option<usize>; 2],
}

fn copy_agent(
    runtime: &Runtime,
    requested: HsaAgent,
    description: Option<&PointerDescription>,
) -> Result<HsaAgent, Status> {
    let agent = if requested.handle == 0 {
        description.map_or(HsaAgent { handle: CPU_AGENT }, |description| {
            description.owner
        })
    } else {
        requested
    };
    runtime
        .is_agent(agent)
        .then_some(agent)
        .ok_or(INVALID_AGENT)
}

fn selected_copy_gpu(
    runtime: &Runtime,
    description: &PointerDescription,
) -> Option<(usize, HsaAgent)> {
    std::iter::once(description.owner)
        .chain(description.accessible.iter().copied())
        .filter(|agent| description.accessible.contains(agent))
        .find_map(|agent| runtime.gpu_index(agent).map(|index| (index, agent)))
}

fn copy_pointer_description(
    runtime: &Runtime,
    pointer: usize,
    size: usize,
) -> Result<Option<PointerDescription>, Status> {
    if pointer.checked_add(size).is_none() {
        return Err(INVALID_ARGUMENT);
    }
    let description = pointer_description(runtime, pointer);
    if let Some(description) = &description {
        if description.pointer_type == POINTER_TYPE_RESERVED_ADDR {
            return Err(INVALID_ALLOCATION);
        }
        if description.offset_range(pointer, size).is_none() {
            return Err(INVALID_ALLOCATION);
        }
    }
    Ok(description)
}

fn gpu_copy_address(
    runtime: &Runtime,
    description: Option<&PointerDescription>,
    pointer: usize,
    size: usize,
    gpu: HsaAgent,
    required_access: u32,
) -> Result<(u64, usize), Status> {
    let description = description.ok_or(INVALID_ALLOCATION)?;
    if !description.accessible.contains(&gpu) {
        return Err(INVALID_AGENT);
    }
    vmem_copy_access(runtime, description, gpu, required_access)?;
    let offset = description
        .offset_range(pointer, size)
        .ok_or(INVALID_ALLOCATION)?;
    let address = description
        .agent_base
        .checked_add(offset)
        .ok_or(INVALID_ALLOCATION)?;
    Ok((
        u64::try_from(address).map_err(|_| INVALID_ALLOCATION)?,
        description.agent_base,
    ))
}

fn prepare_host_copy_source(
    runtime: &Runtime,
    description: Option<&PointerDescription>,
    pointer: usize,
    host: usize,
    size: usize,
    timing: HostSourceTiming,
) -> Result<HostCopySource, Status> {
    if description.is_none() && crate::loader::loaded_host_range(runtime, pointer, size)?.is_some()
    {
        if matches!(timing, HostSourceTiming::AfterDependencies) {
            return Ok(HostCopySource::LoadedAddress(pointer, size));
        }
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(size)
            .map_err(|_| OUT_OF_RESOURCES)?;
        bytes.resize(size, 0);
        // SAFETY: The registry lock retains this loaded mapping while the
        // complete range is copied into owned host storage.
        unsafe { std::ptr::copy_nonoverlapping(host as *const u8, bytes.as_mut_ptr(), size) };
        Ok(HostCopySource::LoadedBytes(bytes))
    } else {
        Ok(HostCopySource::Address(host, size))
    }
}

fn prepare_host_copy_destination(
    runtime: &Runtime,
    description: Option<&PointerDescription>,
    pointer: usize,
    host: usize,
    size: usize,
) -> Result<HostCopyDestination, Status> {
    if description.is_none() && crate::loader::loaded_host_range(runtime, pointer, size)?.is_some()
    {
        Ok(HostCopyDestination::LoadedAddress(pointer))
    } else {
        Ok(HostCopyDestination::Address(host))
    }
}

fn vmem_copy_access(
    runtime: &Runtime,
    description: &PointerDescription,
    agent: HsaAgent,
    required_access: u32,
) -> Result<(), Status> {
    if description.pointer_type != POINTER_TYPE_HSA_VMEM {
        return Ok(());
    }
    let mapping = runtime
        .vmem_mappings
        .get(&description.agent_base)
        .ok_or(INVALID_ALLOCATION)?;
    if mapping
        .access
        .get(&agent.handle)
        .is_none_or(|access| access.permissions & required_access != required_access)
    {
        return Err(INVALID_ALLOCATION);
    }
    Ok(())
}

fn resolve_async_copy(
    runtime: &Runtime,
    destination: usize,
    dst_agent: HsaAgent,
    source: usize,
    src_agent: HsaAgent,
    size: usize,
) -> Result<ResolvedAsyncCopy, Status> {
    if size == 0 {
        if (dst_agent.handle != 0 && !runtime.is_agent(dst_agent))
            || (src_agent.handle != 0 && !runtime.is_agent(src_agent))
        {
            return Err(INVALID_AGENT);
        }
        return Ok(ResolvedAsyncCopy {
            route: AsyncCopyRoute::Noop,
            borrowed_memory: [None, None],
        });
    }
    let dst_description = copy_pointer_description(runtime, destination, size)?;
    let src_description = copy_pointer_description(runtime, source, size)?;
    let dst_agent = copy_agent(runtime, dst_agent, dst_description.as_ref())?;
    let src_agent = copy_agent(runtime, src_agent, src_description.as_ref())?;
    let dst_gpu = runtime.gpu_index(dst_agent);
    let src_gpu = runtime.gpu_index(src_agent);
    if let Some(index) = dst_gpu.or(src_gpu) {
        if !runtime.gpus[index]
            .device
            .gpu()
            .is_ok_and(|gpu| gpu.supports_linear_copy())
        {
            return Err(NOT_SUPPORTED);
        }
        let gpu_agent = HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        };
        let gpu_destination = gpu_copy_address(
            runtime,
            dst_description.as_ref(),
            destination,
            size,
            gpu_agent,
            ACCESS_PERMISSION_WO,
        );
        let gpu_source = gpu_copy_address(
            runtime,
            src_description.as_ref(),
            source,
            size,
            gpu_agent,
            ACCESS_PERMISSION_RO,
        );
        let device = Box::new(runtime.gpus[index].device.clone());
        return match (gpu_destination, gpu_source) {
            (Ok((destination, dst_base)), Ok((source, src_base))) => Ok(ResolvedAsyncCopy {
                route: AsyncCopyRoute::GpuLinear {
                    device,
                    index,
                    destination,
                    source,
                    size: u64::try_from(size).map_err(|_| INVALID_ARGUMENT)?,
                },
                borrowed_memory: [Some(dst_base), Some(src_base)],
            }),
            (Ok((destination, dst_base)), Err(_)) if src_gpu.is_none() => {
                let host = host_copy_address(runtime, source, size, ACCESS_PERMISSION_RO)?;
                let source = prepare_host_copy_source(
                    runtime,
                    src_description.as_ref(),
                    source,
                    host,
                    size,
                    HostSourceTiming::AfterDependencies,
                )?;
                Ok(ResolvedAsyncCopy {
                    route: AsyncCopyRoute::GpuFromHost {
                        device,
                        index,
                        destination,
                        source,
                    },
                    borrowed_memory: [
                        Some(dst_base),
                        src_description.map(|description| description.agent_base),
                    ],
                })
            }
            (Err(_), Ok((source, src_base))) if dst_gpu.is_none() => {
                let host = host_copy_address(runtime, destination, size, ACCESS_PERMISSION_WO)?;
                let destination = prepare_host_copy_destination(
                    runtime,
                    dst_description.as_ref(),
                    destination,
                    host,
                    size,
                )?;
                Ok(ResolvedAsyncCopy {
                    route: AsyncCopyRoute::GpuToHost {
                        device,
                        index,
                        destination,
                        source,
                        size,
                    },
                    borrowed_memory: [
                        dst_description.map(|description| description.agent_base),
                        Some(src_base),
                    ],
                })
            }
            (Err(status), _) | (_, Err(status)) => Err(status),
        };
    }
    let destination_host = host_copy_address(runtime, destination, size, ACCESS_PERMISSION_WO)?;
    let source_host = host_copy_address(runtime, source, size, ACCESS_PERMISSION_RO)?;
    let destination = prepare_host_copy_destination(
        runtime,
        dst_description.as_ref(),
        destination,
        destination_host,
        size,
    )?;
    let source = prepare_host_copy_source(
        runtime,
        src_description.as_ref(),
        source,
        source_host,
        size,
        HostSourceTiming::AfterDependencies,
    )?;
    Ok(ResolvedAsyncCopy {
        route: AsyncCopyRoute::Host {
            destination,
            source,
            size,
        },
        borrowed_memory: [
            dst_description.map(|description| description.agent_base),
            src_description.map(|description| description.agent_base),
        ],
    })
}

fn resolve_sync_gpu_copy(
    runtime: &Runtime,
    destination: usize,
    source: usize,
    size: usize,
    host_destination: Result<usize, Status>,
    host_source: Result<usize, Status>,
) -> Result<ResolvedSyncGpuCopy, Status> {
    match (host_destination, host_source) {
        (Err(_), Ok(host_source)) => {
            let description =
                copy_pointer_description(runtime, destination, size)?.ok_or(INVALID_ALLOCATION)?;
            let (index, agent) = selected_copy_gpu(runtime, &description).ok_or(INVALID_AGENT)?;
            let (destination, destination_base) = gpu_copy_address(
                runtime,
                Some(&description),
                destination,
                size,
                agent,
                ACCESS_PERMISSION_WO,
            )?;
            let source_description = copy_pointer_description(runtime, source, size)?;
            let source_base = source_description
                .as_ref()
                .map(|description| description.agent_base);
            let source = prepare_host_copy_source(
                runtime,
                source_description.as_ref(),
                source,
                host_source,
                size,
                HostSourceTiming::Immediate,
            )?;
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::FromHost {
                    device: Box::new(runtime.gpus[index].device.clone()),
                    destination,
                    source,
                },
                borrowed_memory: [Some(destination_base), source_base],
            })
        }
        (Ok(host_destination), Err(_)) => {
            let description =
                copy_pointer_description(runtime, source, size)?.ok_or(INVALID_ALLOCATION)?;
            let (index, agent) = selected_copy_gpu(runtime, &description).ok_or(INVALID_AGENT)?;
            let (source, source_base) = gpu_copy_address(
                runtime,
                Some(&description),
                source,
                size,
                agent,
                ACCESS_PERMISSION_RO,
            )?;
            let destination_description = copy_pointer_description(runtime, destination, size)?;
            let destination_base = destination_description
                .as_ref()
                .map(|description| description.agent_base);
            let destination = prepare_host_copy_destination(
                runtime,
                destination_description.as_ref(),
                destination,
                host_destination,
                size,
            )?;
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::ToHost {
                    device: Box::new(runtime.gpus[index].device.clone()),
                    destination,
                    source,
                    size,
                },
                borrowed_memory: [destination_base, Some(source_base)],
            })
        }
        (Err(_), Err(_)) => {
            let resolved = resolve_async_copy(
                runtime,
                destination,
                HsaAgent { handle: 0 },
                source,
                HsaAgent { handle: 0 },
                size,
            )?;
            let AsyncCopyRoute::GpuLinear {
                device,
                destination,
                source,
                size,
                ..
            } = resolved.route
            else {
                return Err(INVALID_ALLOCATION);
            };
            Ok(ResolvedSyncGpuCopy {
                route: SyncGpuCopyRoute::Direct {
                    device,
                    destination,
                    source,
                    size,
                },
                borrowed_memory: resolved.borrowed_memory,
            })
        }
        (Ok(_), Ok(_)) => Err(ERROR),
    }
}

fn copy_gpu(device: &Device) -> Result<rocddi::gpu::GpuDevice<'_>, CopyRouteFailure> {
    device
        .gpu()
        .map_err(|error| CopyRouteFailure::retired(map_error(error)))
}

fn loaded_source_snapshot(address: usize, size: usize) -> Result<Vec<u8>, Status> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| OUT_OF_RESOURCES)?;
    bytes.resize(size, 0);
    let guard = lock()?;
    let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
    let host =
        crate::loader::loaded_host_range(runtime, address, size)?.ok_or(INVALID_ALLOCATION)?;
    // SAFETY: The registry lock retains the complete loaded mapping while the
    // source is snapshotted after the dependency signals have been satisfied.
    unsafe { std::ptr::copy_nonoverlapping(host as *const u8, bytes.as_mut_ptr(), size) };
    Ok(bytes)
}

fn with_host_copy_source<T>(
    source: &HostCopySource,
    copy: impl FnOnce(&[u8]) -> Result<T, CopyRouteFailure>,
) -> Result<T, CopyRouteFailure> {
    let snapshot = if let HostCopySource::LoadedAddress(address, size) = source {
        loaded_source_snapshot(*address, *size).map_err(CopyRouteFailure::retired)?
    } else {
        Vec::new()
    };
    let host = match source {
        HostCopySource::Address(address, size) => {
            // SAFETY: The HSA caller keeps this valid host range live through
            // its synchronous call or until asynchronous completion.
            unsafe { std::slice::from_raw_parts(*address as *const u8, *size) }
        }
        HostCopySource::LoadedAddress(_, _) => snapshot.as_slice(),
        HostCopySource::LoadedBytes(bytes) => bytes,
    };
    copy(host)
}

fn copy_from_host_route(
    device: &Device,
    destination: u64,
    source: &HostCopySource,
    stop: &AtomicBool,
) -> Result<(), CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    with_host_copy_source(source, |host| {
        // SAFETY: HSA checked the GPU destination and retains its backing. rocddi
        // copies host bytes into owned staging before native submission.
        unsafe { gpu.copy_from_host(destination, host, stop) }.map_err(Into::into)
    })
}

fn copy_to_host_destination(
    destination: &HostCopyDestination,
    size: usize,
    copy: impl FnOnce(&mut [u8]) -> Result<(), CopyRouteFailure>,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    match destination {
        HostCopyDestination::Address(address) => {
            // SAFETY: The HSA caller keeps this valid host range live through
            // its synchronous call or until asynchronous completion.
            let host = unsafe { std::slice::from_raw_parts_mut(*address as *mut u8, size) };
            copy(host)?;
            fence(Ordering::SeqCst);
            Ok(SyncGpuCopyOutcome::Done)
        }
        HostCopyDestination::LoadedAddress(address) => {
            let mut bytes = Vec::new();
            bytes
                .try_reserve_exact(size)
                .map_err(|_| CopyRouteFailure::retired(OUT_OF_RESOURCES))?;
            bytes.resize(size, 0);
            copy(&mut bytes)?;
            Ok(SyncGpuCopyOutcome::LoadedDestination {
                address: *address,
                bytes,
            })
        }
    }
}

fn copy_to_host_route(
    device: &Device,
    destination: &HostCopyDestination,
    source: u64,
    size: usize,
    stop: &AtomicBool,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    copy_to_host_destination(destination, size, |host| {
        // SAFETY: HSA checked the GPU source and retains its backing. rocddi
        // writes the host buffer only after native retirement.
        unsafe { gpu.copy_to_host(host, source, stop) }.map_err(Into::into)
    })
}

fn execute_sync_gpu_copy(
    route: SyncGpuCopyRoute,
    stop: &AtomicBool,
) -> Result<SyncGpuCopyOutcome, CopyRouteFailure> {
    match route {
        SyncGpuCopyRoute::Direct {
            device,
            destination,
            source,
            size,
        } => {
            let gpu = copy_gpu(&device)?;
            // SAFETY: HSA checked both mapped GPU ranges before retaining
            // their backing through this synchronous call.
            unsafe { gpu.copy_linear(destination, source, size, stop) }
                .map_err(CopyRouteFailure::from)?;
            Ok(SyncGpuCopyOutcome::Done)
        }
        SyncGpuCopyRoute::FromHost {
            device,
            destination,
            source,
        } => {
            copy_from_host_route(&device, destination, &source, stop)?;
            Ok(SyncGpuCopyOutcome::Done)
        }
        SyncGpuCopyRoute::ToHost {
            device,
            destination,
            source,
            size,
        } => copy_to_host_route(&device, &destination, source, size, stop),
    }
}

fn write_loaded_copy_destination(address: usize, bytes: &[u8]) -> Status {
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    let host = match crate::loader::loaded_host_range(runtime, address, bytes.len()) {
        Ok(Some(host)) => host,
        Ok(None) | Err(_) => return INVALID_ALLOCATION,
    };
    // SAFETY: The registry lock retains the complete loaded mapping until
    // the copied bytes and their system-scope ordering are published.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), host as *mut u8, bytes.len()) };
    fence(Ordering::SeqCst);
    SUCCESS
}

fn execute_gpu_batch(
    device: &Device,
    entries: Vec<GpuBatchEntry>,
    stop: &AtomicBool,
) -> Result<(), CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    let mut sequence = GpuCopySequence::begin(gpu, stop).map_err(CopyRouteFailure::from)?;
    for entry in entries {
        let outcome = match entry {
            GpuBatchEntry::Linear {
                destination,
                source,
                size,
            } => {
                // SAFETY: Batch preparation checked both mapped GPU ranges and
                // retains their owners until native retirement is proved.
                unsafe { sequence.copy_linear(destination, source, size) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuBatchEntry::FromHost {
                destination,
                source,
            } => with_host_copy_source(&source, |host| {
                // SAFETY: HSA checked the GPU destination and retains its
                // backing. rocddi owns the staged copy of these host bytes.
                unsafe { sequence.copy_from_host(destination, host) }
                    .map_err(CopyRouteFailure::from)
            })
            .map(|()| SyncGpuCopyOutcome::Done),
            GpuBatchEntry::ToHost {
                destination,
                source,
                size,
            } => copy_to_host_destination(&destination, size, |host| {
                // SAFETY: HSA checked the GPU source and retains its backing.
                // rocddi writes the host slice after native retirement.
                unsafe { sequence.copy_to_host(host, source) }.map_err(CopyRouteFailure::from)
            }),
        }?;
        if let SyncGpuCopyOutcome::LoadedDestination { address, bytes } = outcome {
            let status = write_loaded_copy_destination(address, &bytes);
            if status != SUCCESS {
                return Err(CopyRouteFailure::retired(status));
            }
        }
    }
    Ok(())
}

fn execute_copy_route(route: AsyncCopyRoute, stop: &AtomicBool) -> (bool, bool) {
    match route {
        AsyncCopyRoute::Noop => (true, false),
        AsyncCopyRoute::Host {
            destination,
            source,
            size,
        } => {
            if stop.load(Ordering::Acquire) {
                return (false, false);
            }
            let needs_loaded_mapping = matches!(&source, HostCopySource::LoadedAddress(..))
                || matches!(&destination, HostCopyDestination::LoadedAddress(..));
            let guard = if needs_loaded_mapping {
                match lock() {
                    Ok(guard) => Some(guard),
                    Err(_) => return (false, false),
                }
            } else {
                None
            };
            let runtime = guard.as_ref().and_then(|guard| guard.as_ref());
            let loaded_host = |address| {
                runtime.and_then(|runtime| {
                    crate::loader::loaded_host_range(runtime, address, size)
                        .ok()
                        .flatten()
                })
            };
            let source_host = match &source {
                HostCopySource::Address(address, _) => *address,
                HostCopySource::LoadedAddress(address, _) => {
                    let Some(host) = loaded_host(*address) else {
                        return (false, false);
                    };
                    host
                }
                HostCopySource::LoadedBytes(bytes) => bytes.as_ptr() as usize,
            };
            let destination_host = match destination {
                HostCopyDestination::Address(address) => address,
                HostCopyDestination::LoadedAddress(address) => {
                    let Some(host) = loaded_host(address) else {
                        return (false, false);
                    };
                    host
                }
            };
            // SAFETY: The caller keeps ordinary host ranges live through
            // completion. Runtime-owned memory is borrowed, and the registry
            // lock keeps loaded mappings live for this copy. ptr::copy also
            // permits overlapping ranges.
            unsafe { std::ptr::copy(source_host as *const u8, destination_host as *mut u8, size) };
            fence(Ordering::SeqCst);
            (true, false)
        }
        AsyncCopyRoute::GpuLinear {
            device,
            index: _,
            destination,
            source,
            size,
        } => {
            let Ok(gpu) = device.gpu() else {
                return (false, false);
            };
            // SAFETY: The HSA submission validated mapped ranges and retains
            // their public owners until native retirement is proved.
            match unsafe { gpu.copy_linear(destination, source, size, stop) } {
                Ok(()) => (true, false),
                Err(failure) => (false, failure.operands_may_be_live),
            }
        }
        AsyncCopyRoute::GpuFromHost {
            device,
            index: _,
            destination,
            source,
        } => match copy_from_host_route(&device, destination, &source, stop) {
            Ok(()) => (true, false),
            Err(failure) => (false, failure.operands_may_be_live),
        },
        AsyncCopyRoute::GpuToHost {
            device,
            index: _,
            destination,
            source,
            size,
        } => match copy_to_host_route(&device, &destination, source, size, stop) {
            Ok(SyncGpuCopyOutcome::Done) => (true, false),
            Ok(SyncGpuCopyOutcome::LoadedDestination { address, bytes }) => (
                write_loaded_copy_destination(address, &bytes) == SUCCESS,
                false,
            ),
            Err(failure) => (false, failure.operands_may_be_live),
        },
        AsyncCopyRoute::GpuRect { device, rect } => {
            let Ok(gpu) = device.gpu() else {
                return (false, false);
            };
            // SAFETY: Validation covers all rows and keeps both backing owners
            // live until the native copy reports retirement.
            match unsafe { gpu.copy_rect(rect, stop) } {
                Ok(()) => (true, false),
                Err(failure) => (false, failure.operands_may_be_live),
            }
        }
        AsyncCopyRoute::GpuBatch { device, entries } => {
            match execute_gpu_batch(&device, entries, stop) {
                Ok(()) => (true, false),
                Err(failure) => (false, failure.operands_may_be_live),
            }
        }
    }
}

struct AsyncCopyTask {
    route: AsyncCopyRoute,
    borrowed_memory: Vec<Option<usize>>,
    completion: HsaSignal,
}

fn enqueue_copy_job(
    runtime: &mut Runtime,
    tasks: Vec<AsyncCopyTask>,
    dependency_slice: &[HsaSignal],
) -> Status {
    let mut handles = Vec::new();
    let Some(handle_count) = dependency_slice.len().checked_add(tasks.len()) else {
        return OUT_OF_RESOURCES;
    };
    if handles.try_reserve_exact(handle_count).is_err() || runtime.workers.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    handles.extend_from_slice(dependency_slice);
    handles.extend(tasks.iter().map(|task| task.completion));
    let mut pending = match PendingAsyncSignals::new(runtime, Arc::new(handles)) {
        Ok(pending) => pending,
        Err(status) => return status,
    };
    for task in &tasks {
        for &base in &task.borrowed_memory {
            if let Some(base) = base {
                if let Err(status) = pending.retain_memory(base) {
                    return status;
                }
            }
        }
    }
    let worker_handles = pending.handles.clone();
    let quarantine = pending.runtime.async_copy_quarantine.clone();
    let stop = pending.runtime.stop_workers.clone();
    let dependency_len = dependency_slice.len();
    let worker = match thread::Builder::new()
        .name("rocddi-hsa-copy".into())
        .spawn(move || {
            let mut execute = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                wait_async_dependencies(&worker_handles[..dependency_len], &stop)
            }))
            .unwrap_or(false);
            for task in tasks {
                let (success, may_be_live) = if execute {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute_copy_route(task.route, &stop)
                    }))
                    .unwrap_or((false, true))
                } else {
                    (false, false)
                };
                if may_be_live {
                    quarantine.store(true, Ordering::Release);
                }
                // Release only this operation's backing before publishing its
                // completion. Later operations keep their own borrows.
                release_async_resources(&[], &task.borrowed_memory, may_be_live);
                finish_async_signal(task.completion, success);
                execute = success;
            }
            release_async_resources(&worker_handles, &[], false);
        }) {
        Ok(worker) => worker,
        Err(_) => return OUT_OF_RESOURCES,
    };
    pending.transfer_to_worker();
    pending.runtime.workers.push(worker);
    SUCCESS
}

fn enqueue_async_copy(
    runtime: &mut Runtime,
    resolved: ResolvedAsyncCopy,
    dependency_slice: &[HsaSignal],
    completion: HsaSignal,
) -> Status {
    enqueue_copy_job(
        runtime,
        vec![AsyncCopyTask {
            route: resolved.route,
            borrowed_memory: resolved.borrowed_memory.to_vec(),
            completion,
        }],
        dependency_slice,
    )
}

#[allow(clippy::too_many_arguments)]
unsafe fn submit_async_copy(
    dst: *mut c_void,
    dst_agent: HsaAgent,
    src: *const c_void,
    src_agent: HsaAgent,
    size: usize,
    dependency_count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
    engine: Option<u32>,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if dst.is_null()
            || src.is_null()
            || completion.handle == 0
            || size > isize::MAX as usize
            || (dependency_count == 0 && !dependencies.is_null())
            || (dependency_count != 0 && dependencies.is_null())
        {
            return INVALID_ARGUMENT;
        }
        if !runtime.owns_signal(completion) {
            return INVALID_SIGNAL;
        }
        if engine.is_some_and(|engine| engine != 1) {
            return INVALID_ARGUMENT;
        }
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA caller supplies dependency_count readable handles.
            unsafe { std::slice::from_raw_parts(dependencies, dependency_count as usize) }
        };
        if dependency_slice
            .iter()
            .any(|signal| !runtime.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        if size == 0 {
            return if runtime.is_agent(dst_agent) && runtime.is_agent(src_agent) {
                SUCCESS
            } else {
                INVALID_AGENT
            };
        }
        let resolved = match resolve_async_copy(
            runtime,
            dst as usize,
            dst_agent,
            src as usize,
            src_agent,
            size,
        ) {
            Ok(resolved) => resolved,
            Err(status) => return status,
        };
        if engine.is_some()
            && !matches!(
                &resolved.route,
                AsyncCopyRoute::GpuLinear { .. }
                    | AsyncCopyRoute::GpuFromHost { .. }
                    | AsyncCopyRoute::GpuToHost { .. }
                    | AsyncCopyRoute::Noop
            )
        {
            return NOT_SUPPORTED;
        }
        enqueue_async_copy(runtime, resolved, dependency_slice, completion)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_async_copy(
    dst: *mut c_void,
    dst_agent: HsaAgent,
    src: *const c_void,
    src_agent: HsaAgent,
    size: usize,
    count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
) -> Status {
    // SAFETY: The caller supplies readable dependency handles and live copy
    // ranges for the HSA asynchronous operation.
    unsafe {
        submit_async_copy(
            dst,
            dst_agent,
            src,
            src_agent,
            size,
            count,
            dependencies,
            completion,
            None,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_async_copy_on_engine(
    dst: *mut c_void,
    dst_agent: HsaAgent,
    src: *const c_void,
    src_agent: HsaAgent,
    size: usize,
    count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
    engine: u32,
    _force: bool,
) -> Status {
    // SAFETY: The caller supplies readable dependency handles and live copy
    // ranges for the HSA asynchronous operation.
    unsafe {
        submit_async_copy(
            dst,
            dst_agent,
            src,
            src_agent,
            size,
            count,
            dependencies,
            completion,
            Some(engine),
        )
    }
}

#[derive(Clone, Copy)]
struct BatchCopyEntry {
    destination: usize,
    destination_agent: HsaAgent,
    source: usize,
    source_agent: HsaAgent,
    size: usize,
}

/// The C caller owns the array for the duration of the entry point. Alignment
/// and Rust slice-size limits are checked before forming a shared reference.
#[allow(unsafe_code)]
unsafe fn batch_slice<'a, T>(pointer: *const T, count: usize) -> Result<&'a [T], Status> {
    if pointer.is_null()
        || (pointer as usize) % align_of::<T>() != 0
        || count > isize::MAX as usize / size_of::<T>()
    {
        return Err(INVALID_ARGUMENT);
    }
    // SAFETY: The HSA caller supplies count initialized, readable elements.
    Ok(unsafe { std::slice::from_raw_parts(pointer, count) })
}

#[allow(unsafe_code)]
unsafe fn batch_copy_entries(op: HsaAmdMemoryCopyOp) -> Result<Vec<BatchCopyEntry>, Status> {
    let count = usize::from(op.entry_count.max(1));
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| OUT_OF_RESOURCES)?;
    if op.entry_count == 0 {
        if op.operation != 0 {
            return Err(INVALID_ARGUMENT);
        }
        // SAFETY: The scalar members of the public C unions are active here.
        let destination_agent = unsafe { op.destination_agent.agent };
        let size = unsafe { op.size.bytes };
        if op.secondary_size != 0 {
            return Err(INVALID_ARGUMENT);
        }
        entries.push(BatchCopyEntry {
            destination: op.destination as usize,
            destination_agent,
            source: op.source as usize,
            source_agent: op.source_agent,
            size,
        });
    } else {
        if op.secondary_size != 0 {
            return Err(INVALID_ARGUMENT);
        }
        // SAFETY: A multi-entry descriptor selects pointer, agent, and size
        // lists. The caller keeps those arrays readable through this call.
        let destinations = unsafe { batch_slice(op.destination.cast::<*mut c_void>(), count)? };
        let agents = unsafe { batch_slice(op.destination_agent.list, count)? };
        match op.operation {
            0 => {
                // SAFETY: LINEAR with a positive entry count selects the list
                // members of both source and size unions.
                let sources = unsafe { batch_slice(op.source.cast::<*mut c_void>(), count)? };
                let sizes = unsafe { batch_slice(op.size.list, count)? };
                for index in 0..count {
                    entries.push(BatchCopyEntry {
                        destination: destinations[index] as usize,
                        destination_agent: agents[index],
                        source: sources[index] as usize,
                        source_agent: op.source_agent,
                        size: sizes[index],
                    });
                }
            }
            1 => {
                // SAFETY: BROADCAST keeps the scalar source and size members.
                let size = unsafe { op.size.bytes };
                for index in 0..count {
                    entries.push(BatchCopyEntry {
                        destination: destinations[index] as usize,
                        destination_agent: agents[index],
                        source: op.source as usize,
                        source_agent: op.source_agent,
                        size,
                    });
                }
            }
            _ => return Err(INVALID_ARGUMENT),
        }
    }
    if entries.iter().any(|entry| {
        entry.source == 0
            || entry.destination == 0
            || entry.size > isize::MAX as usize
            || (op.entry_count != 0 && entry.size == 0)
    }) {
        return Err(INVALID_ARGUMENT);
    }
    Ok(entries)
}

fn resolve_batch_copy_entries(
    runtime: &Runtime,
    entries: &[BatchCopyEntry],
) -> Result<(AsyncCopyRoute, Vec<Option<usize>>), Status> {
    let mut prepared_entries = Vec::new();
    prepared_entries
        .try_reserve_exact(entries.len())
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut borrowed_memory = Vec::new();
    borrowed_memory
        .try_reserve_exact(entries.len().checked_mul(2).ok_or(OUT_OF_RESOURCES)?)
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut selected: Option<(usize, Box<Device>)> = None;
    for entry in entries {
        let resolved = resolve_async_copy(
            runtime,
            entry.destination,
            entry.destination_agent,
            entry.source,
            entry.source_agent,
            entry.size,
        )?;
        let (device, index, prepared) = match resolved.route {
            AsyncCopyRoute::GpuLinear {
                device,
                index,
                destination,
                source,
                size,
            } => (
                device,
                index,
                GpuBatchEntry::Linear {
                    destination,
                    source,
                    size,
                },
            ),
            AsyncCopyRoute::GpuFromHost {
                device,
                index,
                destination,
                source,
            } => (
                device,
                index,
                GpuBatchEntry::FromHost {
                    destination,
                    source,
                },
            ),
            AsyncCopyRoute::GpuToHost {
                device,
                index,
                destination,
                source,
                size,
            } => (
                device,
                index,
                GpuBatchEntry::ToHost {
                    destination,
                    source,
                    size,
                },
            ),
            AsyncCopyRoute::Noop if entries.len() == 1 => {
                return Ok((AsyncCopyRoute::Noop, borrowed_memory));
            }
            _ => return Err(INVALID_AGENT),
        };
        if selected
            .as_ref()
            .is_some_and(|(selected, _)| *selected != index)
        {
            return Err(NOT_SUPPORTED);
        }
        selected.get_or_insert((index, device));
        prepared_entries.push(prepared);
        borrowed_memory.extend(resolved.borrowed_memory);
    }
    let (_, device) = selected.ok_or(INVALID_ARGUMENT)?;
    Ok((
        AsyncCopyRoute::GpuBatch {
            device,
            entries: prepared_entries,
        },
        borrowed_memory,
    ))
}

struct PreparedBatchCopy {
    tasks: Vec<AsyncCopyTask>,
}

#[allow(unsafe_code)]
unsafe fn prepare_batch_copy(
    runtime: &Runtime,
    operations: &[HsaAmdMemoryCopyOp],
) -> Result<PreparedBatchCopy, Status> {
    if operations.is_empty() {
        return Err(INVALID_ARGUMENT);
    }
    let mut tasks = Vec::new();
    tasks
        .try_reserve_exact(operations.len())
        .map_err(|_| OUT_OF_RESOURCES)?;
    for &op in operations {
        if op.version != 1
            || op.operation > 5
            || op.reserved != [0]
            || op.wait.reserved != 0
            || op.wait.function > 6
            || op.wait.scope > 2
            || op.signal.reserved != 0
            || op.signal.operation > 3
            || op.signal.scope > 2
        {
            return Err(INVALID_ARGUMENT);
        }
        if !runtime.owns_signal(op.completion_signal) {
            return Err(INVALID_SIGNAL);
        }
        if op.operation > 1
            || op.wait.function != 0
            || op.signal.operation != 0
            || op.traffic_class != 0
        {
            return Err(NOT_SUPPORTED);
        }
        if op.wait.scope != 0
            || !op.wait.address.is_null()
            || op.wait.value != 0
            || op.wait.mask != 0
            || op.signal.scope != 0
            || !op.signal.address.is_null()
            || op.signal.data != 0
        {
            return Err(INVALID_ARGUMENT);
        }
        if op.operation == 1 && runtime.gpu_index(op.source_agent).is_none() {
            return Err(INVALID_AGENT);
        }
        // SAFETY: Each operation selects the documented scalar or array union
        // members. All array values are copied before the caller returns.
        let entries = unsafe { batch_copy_entries(op)? };
        let (route, bases) = resolve_batch_copy_entries(runtime, &entries)?;
        tasks.push(AsyncCopyTask {
            route,
            borrowed_memory: bases,
            completion: op.completion_signal,
        });
    }
    Ok(PreparedBatchCopy { tasks })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_async_batch_copy(
    copy_operations: *const HsaAmdMemoryCopyOp,
    operation_count: u32,
    dependency_count: u32,
    dependencies: *const HsaSignal,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if operation_count == 0
            || copy_operations.is_null()
            || (dependency_count == 0 && !dependencies.is_null())
            || (dependency_count != 0 && dependencies.is_null())
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The HSA caller supplies readable descriptors and dependency
        // handles. Every pointer list is copied into worker-owned storage
        // before this call returns.
        let operations = match unsafe { batch_slice(copy_operations, operation_count as usize) } {
            Ok(operations) => operations,
            Err(status) => return status,
        };
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA caller supplies readable dependency handles.
            match unsafe { batch_slice(dependencies, dependency_count as usize) } {
                Ok(dependencies) => dependencies,
                Err(status) => return status,
            }
        };
        if dependency_slice
            .iter()
            .any(|signal| !runtime.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        // SAFETY: batch_slice established the descriptor storage for this call.
        let prepared = match unsafe { prepare_batch_copy(runtime, operations) } {
            Ok(prepared) => prepared,
            Err(status) => return status,
        };
        enqueue_copy_job(runtime, prepared.tasks, dependency_slice)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_async_copy_rect(
    dst: *const HsaPitchedPtr,
    dst_offset: *const HsaDim3,
    src: *const HsaPitchedPtr,
    src_offset: *const HsaDim3,
    range: *const HsaDim3,
    copy_agent: HsaAgent,
    direction: u32,
    dependency_count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if dst.is_null()
            || dst_offset.is_null()
            || src.is_null()
            || src_offset.is_null()
            || range.is_null()
            || completion.handle == 0
            || !(1..=3).contains(&direction)
            || (dependency_count == 0 && !dependencies.is_null())
            || (dependency_count != 0 && dependencies.is_null())
        {
            return INVALID_ARGUMENT;
        }
        if !runtime.owns_signal(completion) {
            return INVALID_SIGNAL;
        }
        // SAFETY: The HSA caller supplies readable descriptor storage.
        let (dst, dst_offset, src, src_offset, range) =
            unsafe { (*dst, *dst_offset, *src, *src_offset, *range) };
        let Some(index) = runtime.gpu_index(copy_agent) else {
            return INVALID_AGENT;
        };
        let dependency_slice = if dependency_count == 0 {
            &[]
        } else {
            // SAFETY: The HSA caller supplies dependency_count readable handles.
            unsafe { std::slice::from_raw_parts(dependencies, dependency_count as usize) }
        };
        if dependency_slice
            .iter()
            .any(|signal| !runtime.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        if range.x == 0 || range.y == 0 || range.z == 0 {
            return SUCCESS;
        }
        if !runtime.gpus[index]
            .device
            .gpu()
            .is_ok_and(|gpu| gpu.supports_linear_copy())
        {
            return NOT_SUPPORTED;
        }
        let (destination, dst_extent) = match pitched_copy_range(dst, dst_offset, range) {
            Some(range) => range,
            None => return INVALID_ARGUMENT,
        };
        let (source, src_extent) = match pitched_copy_range(src, src_offset, range) {
            Some(range) => range,
            None => return INVALID_ARGUMENT,
        };
        let dst_description = match copy_pointer_description(runtime, destination, dst_extent) {
            Ok(description) => description,
            Err(status) => return status,
        };
        let src_description = match copy_pointer_description(runtime, source, src_extent) {
            Ok(description) => description,
            Err(status) => return status,
        };
        let (destination, dst_base) = match gpu_copy_address(
            runtime,
            dst_description.as_ref(),
            destination,
            dst_extent,
            copy_agent,
            ACCESS_PERMISSION_WO,
        ) {
            Ok(address) => address,
            Err(status) => return status,
        };
        let (source, src_base) = match gpu_copy_address(
            runtime,
            src_description.as_ref(),
            source,
            src_extent,
            copy_agent,
            ACCESS_PERMISSION_RO,
        ) {
            Ok(address) => address,
            Err(status) => return status,
        };
        enqueue_async_copy(
            runtime,
            ResolvedAsyncCopy {
                route: AsyncCopyRoute::GpuRect {
                    device: Box::new(runtime.gpus[index].device.clone()),
                    rect: CopyRect {
                        destination,
                        source,
                        width: u64::from(range.x),
                        height: range.y,
                        depth: range.z,
                        destination_pitch: dst.pitch as u64,
                        source_pitch: src.pitch as u64,
                        destination_slice: dst.slice as u64,
                        source_slice: src.slice as u64,
                    },
                },
                borrowed_memory: [Some(dst_base), Some(src_base)],
            },
            dependency_slice,
            completion,
        )
    })
}

fn pitched_copy_range(
    pitched: HsaPitchedPtr,
    offset: HsaDim3,
    range: HsaDim3,
) -> Option<(usize, usize)> {
    let base = pitched.base as usize;
    let pitch = pitched.pitch;
    let slice = pitched.slice;
    let width = range.x as usize;
    let row_end = (offset.x as usize).checked_add(width)?;
    if base == 0 || (base | pitch | slice | offset.x as usize | width) & 3 != 0 || row_end > pitch {
        return None;
    }
    let first = (offset.z as usize)
        .checked_mul(slice)?
        .checked_add((offset.y as usize).checked_mul(pitch)?)?
        .checked_add(offset.x as usize)?;
    let extent = ((range.z - 1) as usize)
        .checked_mul(slice)?
        .checked_add(((range.y - 1) as usize).checked_mul(pitch)?)?
        .checked_add(width)?;
    let address = base.checked_add(first)?;
    address.checked_add(extent)?;
    Some((address, extent))
}

fn copy_engine_mask(runtime: &Runtime, dst: HsaAgent, src: HsaAgent) -> u32 {
    let dst_gpu = runtime.gpu_index(dst);
    let src_gpu = runtime.gpu_index(src);
    if dst_gpu.is_some() && src_gpu.is_some() && dst_gpu != src_gpu {
        return 0;
    }
    let Some(index) = dst_gpu.or(src_gpu) else {
        return 0;
    };
    u32::from(
        runtime.gpus[index]
            .device
            .gpu()
            .is_ok_and(|gpu| gpu.supports_linear_copy()),
    )
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_copy_engine_status(
    dst: HsaAgent,
    src: HsaAgent,
    mask: *mut u32,
) -> Status {
    boundary(|| {
        if mask.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        if !runtime.is_agent(dst) || !runtime.is_agent(src) {
            return INVALID_AGENT;
        }
        let available = copy_engine_mask(runtime, dst, src);
        // SAFETY: The caller supplied writable output storage.
        unsafe { mask.write(available) };
        if available == 0 {
            OUT_OF_RESOURCES
        } else {
            SUCCESS
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_memory_get_preferred_copy_engine(
    dst: HsaAgent,
    src: HsaAgent,
    mask: *mut u32,
) -> Status {
    boundary(|| {
        if mask.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match guard.as_ref() {
            Some(runtime) => runtime,
            None => return NOT_INITIALIZED,
        };
        if !runtime.is_agent(dst) || !runtime.is_agent(src) {
            return INVALID_AGENT;
        }
        // SAFETY: The caller supplied writable output storage.
        unsafe { mask.write(copy_engine_mask(runtime, dst, src)) };
        SUCCESS
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn pointer_info_matches_the_public_x86_64_abi() {
        assert_eq!(size_of::<HsaAmdPointerInfo>(), 64);
        assert_eq!(std::mem::align_of::<HsaAmdPointerInfo>(), 8);
        assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, pointer_type), 4);
        assert_eq!(
            std::mem::offset_of!(HsaAmdPointerInfo, agent_base_address),
            8
        );
        assert_eq!(
            std::mem::offset_of!(HsaAmdPointerInfo, host_base_address),
            16
        );
        assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, size_in_bytes), 24);
        assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, agent_owner), 40);
        assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, global_flags), 48);
        assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, registered), 52);
        assert_eq!(std::mem::offset_of!(HsaAmdPointerInfo, alloc_flags), 56);
        let _: unsafe extern "C" fn(*const c_void, *mut c_void) -> Status =
            hsa_amd_pointer_info_set_userdata;
    }

    #[test]
    fn ipc_attach_does_not_clobber_an_aliased_input_on_failure() {
        #[repr(align(8))]
        struct AlignedHandle(HsaAmdIpcMemory);

        let original = [0xfeed_beef_u32; 8];
        let mut handle = AlignedHandle(HsaAmdIpcMemory { handle: original });
        // SAFETY: Both raw pointers refer to the same aligned writable storage.
        // The input is read before any successful output can be published.
        let status = unsafe {
            hsa_amd_ipc_memory_attach(
                &raw const handle.0,
                1,
                0,
                std::ptr::null(),
                (&raw mut handle.0).cast::<*mut c_void>(),
            )
        };
        assert_ne!(status, SUCCESS);
        assert_eq!(handle.0.handle, original);
    }

    #[test]
    fn allow_access_rejects_reserved_flags_before_runtime_lookup() {
        let agent = HsaAgent {
            handle: GPU_AGENT_BASE,
        };
        let flags = 0_u32;
        // SAFETY: The scalar inputs remain readable for the duration of the call.
        assert_eq!(
            unsafe {
                hsa_amd_agents_allow_access(
                    1,
                    &raw const agent,
                    &raw const flags,
                    std::ptr::dangling::<c_void>(),
                )
            },
            INVALID_ARGUMENT
        );
    }

    #[test]
    fn registered_extent_covers_unaligned_host_ranges() {
        assert_eq!(registered_extent(0x10_000, 1, 4096), Some(4096));
        assert_eq!(registered_extent(0x10_345, 1, 4096), Some(4096));
        assert_eq!(registered_extent(0x10_345, 4096, 4096), Some(8192));
        assert_eq!(registered_extent(usize::MAX, 2, 4096), None);
        assert_eq!(registered_extent(0x10_123, 1, 65_536), Some(65_536));
        assert_eq!(registered_extent(0x10_123, 65_536, 65_536), Some(131_072));
    }

    #[test]
    fn registered_gpu_alias_translates_to_host_backing() {
        let description = PointerDescription {
            pointer_type: POINTER_TYPE_LOCKED,
            agent_base: 0x20_000,
            host_base: Some(0x10_000),
            size: 4096,
            owner: HsaAgent { handle: CPU_AGENT },
            global_flags: POOL_FLAG_COARSE,
            registered: true,
            alloc_flags: POINTER_ALLOC_NONPAGED | POINTER_ALLOC_HOST_ACCESS,
            user_data: 0x1234,
            accessible: Vec::new(),
        };

        assert_eq!(description.host_address(0x20_000), Some(0x10_000));
        assert_eq!(description.host_address(0x20_123), Some(0x10_123));
        assert_eq!(description.host_address(0x10_123), Some(0x10_123));
        assert_eq!(description.host_address(0x21_000), None);
        assert_eq!(description.host_range(0x20_123, 100), Some(0x10_123));
        assert_eq!(description.host_range(0x20_ff0, 16), Some(0x10_ff0));
        assert_eq!(description.host_range(0x20_ff0, 17), None);
        assert_eq!(description.offset_range(0x20_123, 100), Some(0x123));
        assert_eq!(description.offset_range(0x10_123, 100), Some(0x123));
        let mut info = std::mem::MaybeUninit::<HsaAmdPointerInfo>::uninit();
        // SAFETY: write_pointer_info initializes the complete public structure.
        unsafe {
            write_pointer_info(
                info.as_mut_ptr(),
                size_of::<HsaAmdPointerInfo>() as u32,
                Some(&description),
            );
            assert_eq!(info.assume_init().user_data as usize, 0x1234);
        }
    }

    #[test]
    fn private_graphics_addresses_do_not_claim_a_cpu_mapping() {
        let description = PointerDescription {
            pointer_type: POINTER_TYPE_GRAPHICS,
            agent_base: 0x20_000,
            host_base: None,
            size: 4096,
            owner: HsaAgent { handle: 0 },
            global_flags: POOL_FLAG_COARSE,
            registered: true,
            alloc_flags: POINTER_ALLOC_NONPAGED,
            user_data: 0,
            accessible: Vec::new(),
        };

        assert!(description.contains(0x20_123));
        assert_eq!(description.host_address(0x20_123), None);
        assert_eq!(description.host_range(0x20_123, 1), None);
        assert_eq!(description.offset_range(0x20_123, 100), Some(0x123));
    }

    #[test]
    fn external_memory_entry_points_match_the_linux_public_abi() {
        assert_eq!(size_of::<HsaAmdIpcMemory>(), 32);
        assert_eq!(align_of::<HsaAmdIpcMemory>(), 4);
        assert_eq!(size_of::<HsaAmdExternalSemaphore>(), 8);
        assert_eq!(align_of::<HsaAmdExternalSemaphore>(), 8);
        assert_eq!(size_of::<HsaAmdExternalSemaphoreHandleDescriptor>(), 16);
        assert_eq!(align_of::<HsaAmdExternalSemaphoreHandleDescriptor>(), 8);
        let _: unsafe extern "C" fn(*mut c_void, usize, *mut HsaAmdIpcMemory) -> Status =
            hsa_amd_ipc_memory_create;
        let _: unsafe extern "C" fn(
            *const HsaAmdIpcMemory,
            usize,
            u32,
            *const HsaAgent,
            *mut *mut c_void,
        ) -> Status = hsa_amd_ipc_memory_attach;
        let _: extern "C" fn(*mut c_void) -> Status = hsa_amd_ipc_memory_detach;
        let _: unsafe extern "C" fn(
            u32,
            *mut HsaAgent,
            HsaHandle,
            u32,
            *mut usize,
            *mut *mut c_void,
            *mut usize,
            *mut *const c_void,
        ) -> Status = hsa_amd_interop_map_buffer;
        let _: unsafe extern "C" fn(
            u32,
            *mut HsaAgent,
            HsaHandle,
            u32,
            usize,
            *mut usize,
            *mut *mut c_void,
            *mut usize,
            *mut *const c_void,
        ) -> Status = hsa_amd_interop_map_buffer_with_size;
        let _: extern "C" fn(*mut c_void) -> Status = hsa_amd_interop_unmap_buffer;
        let _: unsafe extern "C" fn(
            HsaAgent,
            *const HsaAmdExternalSemaphoreHandleDescriptor,
            *mut HsaAmdExternalSemaphore,
        ) -> Status = hsa_amd_external_semaphore_handle_open;
        let _: extern "C" fn(HsaAmdExternalSemaphore) -> Status =
            hsa_amd_external_semaphore_handle_close;
        let _: unsafe extern "C" fn(
            HsaAmdAisFileHandle,
            *mut c_void,
            u64,
            i64,
            *mut u64,
            *mut i32,
        ) -> Status = hsa_amd_ais_file_read;
        let _: unsafe extern "C" fn(
            HsaAmdAisFileHandle,
            *mut c_void,
            u64,
            i64,
            *mut u64,
            *mut i32,
        ) -> Status = hsa_amd_ais_file_write;
        let _: unsafe extern "C" fn(*const c_void, usize, *mut i32, *mut u64) -> Status =
            hsa_amd_portable_export_dmabuf;
        let _: unsafe extern "C" fn(*const c_void, usize, *mut i32, *mut u64, u64) -> Status =
            hsa_amd_portable_export_dmabuf_v2;
        let _: extern "C" fn(i32) -> Status = hsa_amd_portable_close_dmabuf;
    }

    #[test]
    fn virtual_memory_entry_points_match_the_public_abi() {
        assert_eq!(size_of::<HsaAmdVmemAllocHandle>(), 8);
        assert_eq!(align_of::<HsaAmdVmemAllocHandle>(), 8);
        assert_eq!(size_of::<HsaAmdMemoryAccessDesc>(), 16);
        assert_eq!(align_of::<HsaAmdMemoryAccessDesc>(), 8);
        assert_eq!(size_of::<HsaFabricHandle>(), 16);
        let _: unsafe extern "C" fn(*mut *mut c_void, usize, u64, u64) -> Status =
            hsa_amd_vmem_address_reserve;
        let _: unsafe extern "C" fn(*mut *mut c_void, usize, u64, u64, u64) -> Status =
            hsa_amd_vmem_address_reserve_align;
        let _: extern "C" fn(*mut c_void, usize) -> Status = hsa_amd_vmem_address_free;
        let _: unsafe extern "C" fn(
            HsaMemoryPool,
            usize,
            u32,
            u64,
            *mut HsaAmdVmemAllocHandle,
        ) -> Status = hsa_amd_vmem_handle_create;
        let _: extern "C" fn(HsaAmdVmemAllocHandle) -> Status = hsa_amd_vmem_handle_release;
        let _: extern "C" fn(*mut c_void, usize, usize, HsaAmdVmemAllocHandle, u64) -> Status =
            hsa_amd_vmem_map;
        let _: extern "C" fn(*mut c_void, usize) -> Status = hsa_amd_vmem_unmap;
        let _: unsafe extern "C" fn(
            *mut c_void,
            usize,
            *const HsaAmdMemoryAccessDesc,
            usize,
        ) -> Status = hsa_amd_vmem_set_access;
        let _: unsafe extern "C" fn(*mut c_void, *mut u32, HsaAgent) -> Status =
            hsa_amd_vmem_get_access;
        let _: unsafe extern "C" fn(*mut i32, HsaAmdVmemAllocHandle, u64) -> Status =
            hsa_amd_vmem_export_shareable_handle;
        let _: unsafe extern "C" fn(i32, *mut HsaAmdVmemAllocHandle) -> Status =
            hsa_amd_vmem_import_shareable_handle;
        let _: unsafe extern "C" fn(*mut HsaAmdVmemAllocHandle, *mut c_void) -> Status =
            hsa_amd_vmem_retain_alloc_handle;
        let _: unsafe extern "C" fn(HsaAmdVmemAllocHandle, *mut HsaMemoryPool, *mut u32) -> Status =
            hsa_amd_vmem_get_alloc_properties_from_handle;
        let _: unsafe extern "C" fn(*mut HsaFabricHandle, HsaAmdVmemAllocHandle, u64) -> Status =
            hsa_amd_vmem_export_fabric_handle;
        let _: unsafe extern "C" fn(HsaFabricHandle, *mut HsaAmdVmemAllocHandle) -> Status =
            hsa_amd_vmem_import_fabric_handle;
    }

    #[test]
    fn virtual_mapping_contains_only_its_half_open_range() {
        let mapping = VmemMapping {
            handle: 1,
            reservation: 0x1_0000,
            address: 0x1_2000,
            offset: 0,
            size: 0x2000,
            access: HashMap::new(),
        };
        assert!(mapping.contains(0x1_2000));
        assert!(mapping.contains(0x1_3fff));
        assert!(!mapping.contains(0x1_4000));
    }

    #[test]
    fn ais_transfer_preserves_offsets_partial_reads_and_status()
    -> Result<(), Box<dyn std::error::Error>> {
        use std::fs::OpenOptions;
        use std::io::{Read, Seek, SeekFrom, Write};
        use std::os::fd::AsRawFd;

        let path = std::env::temp_dir().join(format!(
            "rocddi-ais-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos()
        ));
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(b"prefix........suffix")?;

        let source = *b"rocddi";
        let mut copied = u64::MAX;
        let mut status = i32::MIN;
        // SAFETY: source is readable and file remains live for the transfer.
        assert_eq!(
            unsafe {
                ais_transfer(
                    file.as_raw_fd(),
                    source.as_ptr() as usize,
                    source.len(),
                    6,
                    &raw mut copied,
                    &raw mut status,
                    AisOperation::Write,
                )
            },
            SUCCESS
        );
        assert_eq!(copied, source.len() as u64);
        assert_eq!(status, 0);

        let mut destination = [0u8; 9];
        copied = u64::MAX;
        status = i32::MIN;
        // SAFETY: destination is writable and file remains live for the transfer.
        assert_eq!(
            unsafe {
                ais_transfer(
                    file.as_raw_fd(),
                    destination.as_mut_ptr() as usize,
                    destination.len(),
                    6,
                    &raw mut copied,
                    &raw mut status,
                    AisOperation::Read,
                )
            },
            SUCCESS
        );
        assert_eq!(copied, destination.len() as u64);
        assert_eq!(status, 0);
        assert_eq!(&destination, b"rocddi..s");

        file.seek(SeekFrom::Start(0))?;
        let mut contents = Vec::new();
        file.read_to_end(&mut contents)?;
        assert_eq!(&contents, b"prefixrocddi..suffix");

        copied = u64::MAX;
        status = i32::MIN;
        // SAFETY: A zero-byte transfer borrows no host storage.
        assert_eq!(
            unsafe {
                ais_transfer(
                    file.as_raw_fd(),
                    0,
                    0,
                    0,
                    &raw mut copied,
                    &raw mut status,
                    AisOperation::Read,
                )
            },
            SUCCESS
        );
        assert_eq!((copied, status), (0, 0));

        copied = u64::MAX;
        status = i32::MIN;
        // SAFETY: The source is readable; Linux reports the invalid descriptor.
        assert_eq!(
            unsafe {
                ais_transfer(
                    -1,
                    source.as_ptr() as usize,
                    source.len(),
                    0,
                    &raw mut copied,
                    &raw mut status,
                    AisOperation::Write,
                )
            },
            ERROR
        );
        assert_eq!((copied, status), (0, -9));
        drop(file);
        std::fs::remove_file(path)?;
        Ok(())
    }

    #[test]
    fn deallocation_callback_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(*mut c_void, DeallocationCallbackFn, *mut c_void) -> Status =
            hsa_amd_register_deallocation_callback;
        let _: unsafe extern "C" fn(*mut c_void, DeallocationCallbackFn) -> Status =
            hsa_amd_deregister_deallocation_callback;
    }

    #[test]
    fn migration_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(HsaMemoryPool, HsaMemoryPool, *mut bool) -> Status =
            hsa_amd_memory_pool_can_migrate;
        let _: unsafe extern "C" fn(*const c_void, HsaMemoryPool, u32) -> Status =
            hsa_amd_memory_migrate;
    }

    #[test]
    fn base_memory_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(HsaRegion, usize, *mut *mut c_void) -> Status =
            hsa_memory_allocate;
        let _: unsafe extern "C" fn(*mut c_void) -> Status = hsa_memory_free;
        let _: extern "C" fn(*mut c_void, usize) -> Status = hsa_memory_register;
        let _: extern "C" fn(*mut c_void, usize) -> Status = hsa_memory_deregister;
        let _: extern "C" fn(*mut c_void, HsaAgent, u32) -> Status = hsa_memory_assign_agent;
        let _: unsafe extern "C" fn(*mut c_void, *const c_void, usize) -> Status = hsa_memory_copy;
    }

    #[test]
    fn zero_length_memory_operations_skip_address_resolution() {
        let aligned = align_of::<u32>() as *mut c_void;
        assert_eq!(memory_copy_has_work(aligned, aligned, 0), Ok(false));
        assert_eq!(
            memory_copy_has_work(aligned, aligned, isize::MAX as usize + 1),
            Err(INVALID_ARGUMENT)
        );
        assert_eq!(memory_fill_size(aligned, 0), Ok(None));

        assert_eq!(
            memory_copy_has_work(std::ptr::null_mut(), aligned, 0),
            Err(INVALID_ARGUMENT)
        );
        assert_eq!(
            memory_fill_size(std::ptr::null_mut(), 0),
            Err(INVALID_ARGUMENT)
        );
        assert_eq!(
            memory_fill_size((align_of::<u32>() + 1) as *mut c_void, 0),
            Err(INVALID_ARGUMENT)
        );
        assert_eq!(
            memory_fill_size(aligned, isize::MAX as usize / size_of::<u32>() + 1),
            Err(INVALID_ARGUMENT)
        );
    }

    #[test]
    fn rectangular_copy_entry_point_matches_the_public_abi() {
        let _: unsafe extern "C" fn(
            *const HsaPitchedPtr,
            *const HsaDim3,
            *const HsaPitchedPtr,
            *const HsaDim3,
            *const HsaDim3,
            HsaAgent,
            u32,
            u32,
            *const HsaSignal,
            HsaSignal,
        ) -> Status = hsa_amd_memory_async_copy_rect;
        assert_eq!(size_of::<HsaPitchedPtr>(), 24);
        assert_eq!(align_of::<HsaPitchedPtr>(), 8);
        assert_eq!(std::mem::offset_of!(HsaPitchedPtr, base), 0);
        assert_eq!(std::mem::offset_of!(HsaPitchedPtr, pitch), 8);
        assert_eq!(std::mem::offset_of!(HsaPitchedPtr, slice), 16);
    }

    #[test]
    fn batch_copy_entry_point_and_descriptor_match_the_public_abi() {
        let _: unsafe extern "C" fn(
            *const HsaAmdMemoryCopyOp,
            u32,
            u32,
            *const HsaSignal,
        ) -> Status = hsa_amd_memory_async_batch_copy;
        assert_eq!(size_of::<HsaAmdMemoryCopyWait>(), 32);
        assert_eq!(align_of::<HsaAmdMemoryCopyWait>(), 8);
        assert_eq!(size_of::<HsaAmdMemoryCopySignal>(), 24);
        assert_eq!(align_of::<HsaAmdMemoryCopySignal>(), 8);
        assert_eq!(size_of::<HsaAmdMemoryCopyOp>(), 128);
        assert_eq!(align_of::<HsaAmdMemoryCopyOp>(), 8);
        assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, source), 16);
        assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, size), 48);
        assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, wait), 64);
        assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, signal), 96);
        assert_eq!(std::mem::offset_of!(HsaAmdMemoryCopyOp, reserved), 120);
    }

    #[test]
    fn async_copy_entry_points_preserve_no_init_validation() {
        let agent = HsaAgent { handle: 0 };
        let completion = HsaSignal { handle: 1 };
        let dependency = HsaSignal { handle: 2 };
        let mut destination = 0x5a_u8;
        let source = 0xa5_u8;
        assert_eq!(
            unsafe {
                hsa_amd_memory_async_copy(
                    (&raw mut destination).cast(),
                    agent,
                    (&raw const source).cast(),
                    agent,
                    1,
                    1,
                    &raw const dependency,
                    completion,
                )
            },
            NOT_INITIALIZED
        );
        assert_eq!(
            unsafe {
                hsa_amd_memory_async_copy_on_engine(
                    (&raw mut destination).cast(),
                    agent,
                    (&raw const source).cast(),
                    agent,
                    1,
                    1,
                    &raw const dependency,
                    completion,
                    1,
                    true,
                )
            },
            NOT_INITIALIZED
        );
        assert_eq!(
            unsafe {
                hsa_amd_memory_async_batch_copy(std::ptr::null(), 1, 1, &raw const dependency)
            },
            NOT_INITIALIZED
        );
        assert_eq!(
            unsafe {
                hsa_amd_memory_async_copy_rect(
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null(),
                    agent,
                    1,
                    1,
                    &raw const dependency,
                    completion,
                )
            },
            NOT_INITIALIZED
        );
        assert_eq!(destination, 0x5a);
    }

    #[test]
    fn svm_entry_points_match_the_public_abi() {
        assert_eq!(size_of::<HsaAmdSvmAttributePair>(), 16);
        assert_eq!(align_of::<HsaAmdSvmAttributePair>(), 8);
        let _: unsafe extern "C" fn(
            *mut c_void,
            usize,
            *mut HsaAmdSvmAttributePair,
            usize,
        ) -> Status = hsa_amd_svm_attributes_set;
        let _: unsafe extern "C" fn(
            *mut c_void,
            usize,
            *mut HsaAmdSvmAttributePair,
            usize,
        ) -> Status = hsa_amd_svm_attributes_get;
        let _: unsafe extern "C" fn(
            *mut c_void,
            usize,
            HsaAgent,
            u32,
            *const HsaSignal,
            HsaSignal,
        ) -> Status = hsa_amd_svm_prefetch_async;
        let _: unsafe extern "C" fn(
            *mut *mut c_void,
            *mut usize,
            u32,
            u32,
            *const HsaSignal,
            HsaSignal,
        ) -> Status = hsa_amd_svm_discard_batch_async;
        let _: unsafe extern "C" fn(
            *mut *mut c_void,
            *mut usize,
            u32,
            *const HsaAgent,
            u32,
            u32,
            *const HsaSignal,
            HsaSignal,
        ) -> Status = hsa_amd_svm_discard_and_prefetch_batch_async;
    }

    #[test]
    fn svm_ranges_cover_each_touched_page() {
        let page = current_host_page().unwrap();
        let page_address = page as u64;
        let interior = (page + 0x123) as *mut c_void;
        assert_eq!(svm_range(interior, 1), Ok((page_address, page_address)));
        assert_eq!(
            svm_range(interior, page),
            Ok((page_address, page_address * 2))
        );
        assert_eq!(svm_range(std::ptr::null_mut(), 1), Err(INVALID_ARGUMENT));
        assert_eq!(svm_range(page as *mut c_void, 0), Err(INVALID_ARGUMENT));
    }

    #[test]
    fn portable_close_works_without_an_initialized_runtime() {
        use std::os::fd::IntoRawFd;

        let descriptor = std::fs::File::open("/dev/null").map_or(-1, IntoRawFd::into_raw_fd);
        assert_eq!(hsa_amd_portable_close_dmabuf(descriptor), SUCCESS);
        assert_eq!(hsa_amd_portable_close_dmabuf(-1), RESOURCE_FREE);
    }

    #[test]
    fn portable_export_checks_initialization_before_arguments() {
        assert!(lock().is_ok_and(|runtime| runtime.is_none()));

        let mut descriptor = -1;
        let mut offset = u64::MAX;
        // SAFETY: The runtime is uninitialized, so ROCr-compatible validation
        // returns before dereferencing any of the deliberately invalid inputs.
        assert_eq!(
            unsafe {
                hsa_amd_portable_export_dmabuf(
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            NOT_INITIALIZED
        );
        // ROCr only consumes the PCIe bit and ignores the remaining bits.
        assert_eq!(
            unsafe {
                hsa_amd_portable_export_dmabuf_v2(
                    std::ptr::dangling::<c_void>(),
                    1,
                    &raw mut descriptor,
                    &raw mut offset,
                    u64::MAX,
                )
            },
            NOT_INITIALIZED
        );
        assert_eq!(descriptor, -1);
        assert_eq!(offset, u64::MAX);
    }

    #[test]
    fn memory_lock_checks_initialization_before_arguments() {
        assert!(lock().is_ok_and(|runtime| runtime.is_none()));

        let mut agent_ptr = usize::MAX as *mut c_void;
        // SAFETY: An uninitialized runtime returns before inspecting the
        // deliberately invalid input pointers.
        assert_eq!(
            unsafe {
                hsa_amd_memory_lock(
                    std::ptr::null_mut(),
                    0,
                    std::ptr::null(),
                    0,
                    &raw mut agent_ptr,
                )
            },
            NOT_INITIALIZED
        );
        assert_eq!(agent_ptr, usize::MAX as *mut c_void);
        // The uncached pool-allocation flag is accepted by ROCr and reaches
        // its host-memory registration path.
        assert_eq!(
            unsafe {
                hsa_amd_memory_lock_to_pool(
                    std::ptr::dangling_mut::<c_void>(),
                    1,
                    std::ptr::null(),
                    0,
                    HsaMemoryPool {
                        handle: CPU_POOL_COARSE,
                    },
                    ALLOC_UNCACHED,
                    &raw mut agent_ptr,
                )
            },
            NOT_INITIALIZED
        );
        assert_eq!(agent_ptr, usize::MAX as *mut c_void);
    }

    #[test]
    fn cpu_pool_allocations_preserve_their_cache_policy() {
        for pool in [CPU_POOL_FINE, CPU_POOL_EXTENDED, CPU_POOL_COARSE] {
            assert_eq!(
                cpu_pool_memory_kind(HsaMemoryPool { handle: pool }, SessionLifetime::Process),
                MemoryKind::OwnedHost
            );
        }
        assert_eq!(
            cpu_pool_memory_kind(
                HsaMemoryPool {
                    handle: CPU_POOL_KERNARG,
                },
                SessionLifetime::Process
            ),
            MemoryKind::System
        );
        for pool in [CPU_POOL_FINE, CPU_POOL_EXTENDED, CPU_POOL_COARSE] {
            assert_eq!(
                cpu_pool_memory_kind(HsaMemoryPool { handle: pool }, SessionLifetime::Session),
                MemoryKind::System
            );
        }
    }

    #[test]
    fn gfx12_gpu_pool_and_region_enumeration_matches_rocr() {
        let index = 2;
        let pools = gpu_agent_pools(index, false)
            .into_iter()
            .map(|pool| pool.handle)
            .collect::<Vec<_>>();
        assert_eq!(
            pools,
            [
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_COARSE,
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_GROUP,
            ]
        );
        let fine_pools = gpu_agent_pools(index, true)
            .into_iter()
            .map(|pool| pool.handle)
            .collect::<Vec<_>>();
        assert_eq!(
            fine_pools,
            [
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_COARSE,
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_FINE,
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_GROUP,
            ]
        );

        let regions = gpu_agent_regions(index, true)
            .into_iter()
            .map(|pool| pool.handle)
            .collect::<Vec<_>>();
        assert_eq!(
            regions,
            [
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_COARSE,
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_FINE,
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_GROUP,
                CPU_POOL_FINE,
                CPU_POOL_KERNARG,
                CPU_POOL_EXTENDED,
                CPU_POOL_COARSE,
            ]
        );
    }

    #[test]
    fn agent_pool_access_matches_rocr_relationships() {
        assert_eq!(
            pool_access(true, PoolStorage::System, false, false),
            POOL_ACCESS_DEFAULT
        );
        assert_eq!(
            pool_access(true, PoolStorage::LocalCoarse, false, false),
            POOL_ACCESS_DEFAULT
        );
        assert_eq!(
            pool_access(true, PoolStorage::Group, false, false),
            POOL_ACCESS_DEFAULT
        );
        assert_eq!(
            pool_access(false, PoolStorage::System, true, false),
            POOL_ACCESS_DISALLOWED
        );
        assert_eq!(
            pool_access(false, PoolStorage::LocalCoarse, true, false),
            POOL_ACCESS_DISALLOWED
        );
        assert_eq!(
            pool_access(false, PoolStorage::LocalCoarse, false, false),
            POOL_ACCESS_NEVER
        );
        assert_eq!(
            pool_access(false, PoolStorage::LocalFine, true, true),
            POOL_ACCESS_DISALLOWED
        );
        assert_eq!(
            pool_access(false, PoolStorage::LocalFine, true, false),
            POOL_ACCESS_NEVER
        );
        assert_eq!(
            pool_access(false, PoolStorage::Group, true, true),
            POOL_ACCESS_NEVER
        );
    }

    #[test]
    fn topology_link_types_use_the_hsa_amd_encoding() {
        assert_eq!(hsa_link_type(MemoryLinkType::HyperTransport), 0);
        assert_eq!(hsa_link_type(MemoryLinkType::Qpi), 1);
        assert_eq!(hsa_link_type(MemoryLinkType::Pcie), 2);
        assert_eq!(hsa_link_type(MemoryLinkType::Infiniband), 3);
        assert_eq!(hsa_link_type(MemoryLinkType::Xgmi), 4);
        assert_eq!(hsa_link_type(MemoryLinkType::Unknown), 0);
    }
}
