// Copyright (c) 2026 Advanced Micro Devices, Inc.
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
use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_char, c_void};
use std::mem::{MaybeUninit, align_of, size_of};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering, fence};
use std::thread;
use std::time::Duration;

use crate::platform::memory as linux_interop;
use crate::platform::memory::AisFileOperation as AisOperation;
use crate::platform::memory::{
    KfdIpcMemoryHandle as IpcMemoryHandle, KfdSvmAccess as SvmAccess,
    KfdSvmAttribute as SvmAttribute, KfdSvmLocation as SvmLocation,
};
use rocddi::device::Device;
use rocddi::device::gpu::{
    CopyRect, GpuCopyOptions, GpuCopySequence, GpuCopySignal, GpuCopySignalOperation,
    GpuCopyTimestamps, GpuCopyWait,
};
use rocddi::memory::device::{
    Allocation, DeviceAccess, HostMappingPolicy, MemoryKind, VirtualAddress, VirtualDeviceMapping,
    VirtualHostMapping, VirtualMemory,
};
use rocddi::session::{DriverContextLifetime, Session};
use rocddi::topology::{MemoryLinkInfo, MemoryLinkType};

use crate::ffi::*;
use crate::runtime::{HsaState, boundary, initialized_mut, lock, map_error};
use crate::signal::{AsyncCopyClock, AsyncCopyProfile};

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
const EIO: i32 = 5;

/// Process setting captured when an asynchronous copy is accepted.
#[derive(Clone, Copy)]
pub(crate) enum AsyncCopyProfiling {
    Disabled,
    Enabled,
}

/// HSA memory handles and metadata around rocddi allocation owners.
pub(crate) struct MemoryState {
    /// Ordinary HSA allocations keyed by host address.
    allocations: HashMap<usize, Memory>,
    /// Imported IPC memory keyed by host address.
    ipc_allocations: HashMap<usize, Memory>,
    /// External memory imports keyed by host address.
    interop_allocations: HashMap<usize, Memory>,
    /// Host ranges locked for GPU access.
    locked_allocations: Vec<LockedMemory>,
    /// Outstanding asynchronous copy borrows per allocation address.
    async_copy_borrows: HashMap<usize, usize>,
    /// Prevents unsafe reuse after uncertain asynchronous copy retirement.
    async_copy_quarantine: Arc<AtomicBool>,
    /// Process setting captured by new asynchronous copies.
    pub(crate) async_copy_profiling: AsyncCopyProfiling,
    /// Virtual address reservations ordered by base address.
    vmem_reservations: BTreeMap<usize, VmemReservation>,
    /// Physical virtual-memory handles indexed by public ID.
    vmem_handles: HashMap<u64, VmemHandle>,
    /// Active virtual-memory mappings ordered by base address.
    vmem_mappings: BTreeMap<usize, VmemMapping>,
    /// Prefetch and copy workers joined before memory owners are released.
    workers: Vec<thread::JoinHandle<()>>,
}

impl MemoryState {
    pub(crate) fn new() -> Self {
        Self {
            allocations: HashMap::new(),
            ipc_allocations: HashMap::new(),
            interop_allocations: HashMap::new(),
            locked_allocations: Vec::new(),
            async_copy_borrows: HashMap::new(),
            async_copy_quarantine: Arc::new(AtomicBool::new(false)),
            async_copy_profiling: AsyncCopyProfiling::Disabled,
            vmem_reservations: BTreeMap::new(),
            vmem_handles: HashMap::new(),
            vmem_mappings: BTreeMap::new(),
            workers: Vec::new(),
        }
    }

    pub(crate) fn running_on_worker(&self) -> bool {
        self.workers
            .iter()
            .any(|worker| worker.thread().id() == thread::current().id())
    }

    /// Joins memory workers after the shared stop request is published.
    pub(crate) fn join_workers(&mut self) {
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }

    pub(crate) fn is_quarantined(&self) -> bool {
        self.async_copy_quarantine.load(Ordering::Acquire)
    }

    /// Releases HSA metadata after queues, copies, and signal users stop.
    pub(crate) fn clear(&mut self) {
        self.locked_allocations.clear();
        self.vmem_mappings.clear();
        self.vmem_handles.clear();
        self.vmem_reservations.clear();
        self.ipc_allocations.clear();
        self.interop_allocations.clear();
        self.allocations.clear();
        self.async_copy_borrows.clear();
    }
}

/// Canonical metadata for every address alias of one HSA-visible allocation.
///
/// `agent_base` is always present. `host_base` exists only when CPU access is
/// valid. Range resolution preserves the offset between those aliases instead
/// of assuming that host and GPU virtual addresses are identical.
#[derive(Clone)]
struct PointerDescription {
    /// HSA pointer classification reported to callers.
    pointer_type: u32,
    /// Address visible to the owning agent.
    agent_base: usize,
    /// CPU alias when this allocation is host accessible.
    host_base: Option<usize>,
    /// Byte extent common to the address aliases.
    size: usize,
    /// HSA agent that owns the allocation.
    owner: HsaAgent,
    /// Pool allocation flags exposed by pointer queries.
    global_flags: u32,
    /// Whether this address came from host registration.
    registered: bool,
    /// Native allocation flags retained for compatibility queries.
    alloc_flags: u32,
    /// Caller-set pointer metadata.
    user_data: usize,
    /// Agents with established access to the allocation.
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

/// HsaState-owned allocation plus callbacks registered for its destruction.
pub(crate) struct Memory {
    /// Native backing and its mapping lifetime.
    allocation: Allocation,
    /// Canonical metadata for every public address alias.
    description: PointerDescription,
    /// Pool flags fixed when this allocation was created.
    pool_global_flags: u32,
    /// Callbacks invoked after public ownership is removed.
    deallocation_callbacks: Vec<RegisteredDeallocationCallback>,
}

/// Application callback invoked after an allocation loses public ownership.
#[derive(Clone, Copy)]
struct RegisteredDeallocationCallback {
    /// Public allocation address supplied when the callback was registered.
    pointer: usize,
    /// Application callback invoked after public ownership ends.
    callback: unsafe extern "C" fn(*mut c_void, *mut c_void),
    /// Caller-owned argument passed back to that callback.
    user_data: usize,
}

impl Memory {
    fn new(
        allocation: Allocation,
        size: usize,
        owner: HsaAgent,
        pool_global_flags: u32,
        uncached: bool,
        flags: u32,
        accessible: Vec<HsaAgent>,
    ) -> Self {
        let info = allocation.info();
        let agent_base = info.device_address as usize;
        let global_flags =
            effective_global_flags(pool_global_flags, uncached, flags & ALLOC_PCIE != 0);
        let alloc_flags = POINTER_ALLOC_NONPAGED
            | if info.host_address.is_some() {
                POINTER_ALLOC_HOST_ACCESS
            } else {
                0
            }
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
            | if pool_global_flags & POOL_FLAG_FINE != 0 && flags & ALLOC_UNCACHED == 0 {
                POINTER_ALLOC_ATOMIC_FULL
            } else if flags & ALLOC_PCIE != 0 {
                POINTER_ALLOC_ATOMIC_PARTIAL
            } else {
                0
            };
        Self {
            allocation,
            pool_global_flags,
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
            pool_global_flags: POOL_FLAG_COARSE,
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
            pool_global_flags: POOL_FLAG_COARSE,
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
    /// Native registration when the host range needed one.
    allocation: Option<Allocation>,
    /// Host and agent aliases reported by pointer queries.
    description: PointerDescription,
}

/// Reserved virtual-address interval and the mappings currently occupying it.
pub(crate) struct VmemReservation {
    /// Native reserved address range.
    address: VirtualAddress,
    /// Public pointer metadata for the reservation.
    description: PointerDescription,
    /// Callbacks invoked when public ownership is removed.
    deallocation_callbacks: Vec<RegisteredDeallocationCallback>,
    /// Mappings that still occupy this reserved range.
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
    /// Native physical backing or imported fabric handle.
    backing: VmemBacking,
    /// Number of public references to this handle.
    references: usize,
    /// Number of active virtual-address mappings.
    mappings: usize,
    /// HSA pool from which the backing originates.
    pool: HsaMemoryPool,
    /// HSA physical-memory type exposed for this handle.
    memory_type: u32,
}

/// Origin of physical backing behind an HSA virtual-memory handle.
enum VmemBacking {
    /// Allocation created through this runtime.
    Owned(VirtualMemory),
    /// Native allocation imported from a DMA-BUF.
    DmaBuf(VirtualMemory),
    /// Fabric token with per-device imported native owners.
    Fabric {
        /// Opaque native token received from another process.
        token: [u8; 16],
        /// Imports keyed by activated GPU index in this runtime.
        imports: HashMap<usize, VirtualMemory>,
    },
}

impl VmemHandle {
    fn new(backing: VmemBacking, pool: HsaMemoryPool, memory_type: u32) -> Self {
        Self {
            backing,
            references: 1,
            mappings: 0,
            pool,
            memory_type,
        }
    }

    fn imported(&self) -> bool {
        !matches!(&self.backing, VmemBacking::Owned(_))
    }

    fn owned_memory(&self) -> Option<&VirtualMemory> {
        match &self.backing {
            VmemBacking::Owned(memory) => Some(memory),
            VmemBacking::DmaBuf(_) | VmemBacking::Fabric { .. } => None,
        }
    }

    fn known_size(&self) -> Option<u64> {
        match &self.backing {
            VmemBacking::Owned(memory) | VmemBacking::DmaBuf(memory) => Some(memory.info().size),
            VmemBacking::Fabric { .. } => None,
        }
    }

    fn memory_for_access(
        &mut self,
        gpu_index: usize,
        device: &Device,
    ) -> Result<&VirtualMemory, Status> {
        match &mut self.backing {
            VmemBacking::Owned(memory) | VmemBacking::DmaBuf(memory) => Ok(memory),
            VmemBacking::Fabric { token, imports } => {
                if !imports.contains_key(&gpu_index) {
                    imports.try_reserve(1).map_err(|_| OUT_OF_RESOURCES)?;
                    let memory = linux_interop::import_fabric_virtual_memory(device, *token)
                        .map_err(map_error)?;
                    imports.insert(gpu_index, memory);
                }
                imports.get(&gpu_index).ok_or(ERROR)
            }
        }
    }

    fn free(&mut self) -> Result<(), rocddi::Error> {
        match &mut self.backing {
            VmemBacking::Owned(memory) | VmemBacking::DmaBuf(memory) => memory.free(),
            VmemBacking::Fabric { imports, .. } => {
                while let Some(index) = imports.keys().next().copied() {
                    let Some(mut memory) = imports.remove(&index) else {
                        continue;
                    };
                    if let Err(error) = memory.free() {
                        imports.insert(index, memory);
                        return Err(error);
                    }
                }
                Ok(())
            }
        }
    }
}

/// Native mapping owner selected by whether access belongs to a CPU or GPU.
enum VmemAccessOwner {
    /// Native mapping in an activated device VM.
    Device(VirtualDeviceMapping),
    /// Native mapping in the host process address space.
    Host(VirtualHostMapping),
}

/// Per-agent permissions and the native object that enforces them.
struct VmemAccess {
    /// HSA permission mask granted to this agent.
    permissions: u32,
    /// Native mapping retained until this access is removed.
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
    /// Public physical-memory handle supplying backing.
    handle: u64,
    /// Base address of the parent reservation.
    reservation: usize,
    /// First mapped virtual byte.
    address: usize,
    /// Byte offset into physical backing.
    offset: usize,
    /// Mapped byte extent.
    size: usize,
    /// Per-agent permissions and native map owners.
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
    let page = usize::try_from(rocddi::memory::host::host_page_size().map_err(map_error)?)
        .map_err(|_| ERROR)?;
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

fn cpu_pool_memory_kind(
    pool: HsaMemoryPool,
    lifetime: DriverContextLifetime,
    flags: u32,
) -> MemoryKind {
    if pool.handle == CPU_POOL_KERNARG || lifetime == DriverContextLifetime::Session {
        MemoryKind::System
    } else {
        MemoryKind::OwnedHost {
            policy: cpu_pool_mapping_policy(pool, flags),
        }
    }
}

fn cpu_pool_mapping_policy(pool: HsaMemoryPool, flags: u32) -> HostMappingPolicy {
    if pool.handle == CPU_POOL_KERNARG || flags & ALLOC_UNCACHED != 0 {
        HostMappingPolicy::Uncached
    } else if pool.handle == CPU_POOL_EXTENDED {
        HostMappingPolicy::Extended
    } else if pool.handle == CPU_POOL_FINE || flags & ALLOC_PCIE != 0 {
        HostMappingPolicy::Fine
    } else {
        HostMappingPolicy::Coarse
    }
}

fn effective_global_flags(pool_flags: u32, uncached: bool, pcie: bool) -> u32 {
    if uncached {
        (pool_flags & !(POOL_FLAG_COARSE | POOL_FLAG_EXTENDED_FINE))
            | POOL_FLAG_FINE
            | POOL_FLAG_KERNARG
    } else if pcie {
        (pool_flags & !POOL_FLAG_COARSE) | POOL_FLAG_FINE
    } else {
        pool_flags
    }
}

fn pool_owner(runtime: &HsaState, pool: HsaMemoryPool) -> Option<HsaAgent> {
    if cpu_pool(pool) {
        Some(HsaAgent { handle: CPU_AGENT })
    } else {
        runtime.decode_gpu_pool(pool).map(|(index, _)| HsaAgent {
            handle: GPU_AGENT_BASE + index as u64,
        })
    }
}

fn pool_global_flags(runtime: &HsaState, pool: HsaMemoryPool) -> Option<u32> {
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
    pools.push(HsaState::gpu_pool(index, GPU_POOL_COARSE));
    if fine_grain {
        pools.push(HsaState::gpu_pool(index, GPU_POOL_FINE));
    }
    pools.push(HsaState::gpu_pool(index, GPU_POOL_GROUP));
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

/// Backing and coherence class represented by one HSA memory pool.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PoolStorage {
    /// Host system memory.
    System,
    /// GPU local memory with coarse-grain coherence.
    LocalCoarse,
    /// GPU local memory with fine-grain coherence.
    LocalFine,
    /// Per-workgroup memory.
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

fn gpu_pool_host_visible(runtime: &HsaState, index: usize) -> bool {
    let endpoint = &runtime.gpus[index].endpoint;
    endpoint.local_memory_bytes != 0
        && endpoint.host_visible_local_memory_bytes >= endpoint.local_memory_bytes
}

fn gpu_pool_coherent(pool_kind: u64, uncached: bool, pcie: bool) -> bool {
    pool_kind == GPU_POOL_FINE || uncached || pcie
}

fn agent_pool_relationship(
    runtime: &HsaState,
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
    let cpu_cannot_access_local = agent.handle == CPU_AGENT
        && owner_gpu.is_some_and(|index| !gpu_pool_host_visible(runtime, index))
        && matches!(storage, PoolStorage::LocalCoarse | PoolStorage::LocalFine);
    let access = if cpu_cannot_access_local {
        POOL_ACCESS_NEVER
    } else {
        pool_access(same, storage, linked, same_hive)
    };
    (access, link)
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

fn agent_pools(runtime: &HsaState, agent: HsaAgent) -> Result<Vec<HsaMemoryPool>, Status> {
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

fn agent_regions(runtime: &HsaState, agent: HsaAgent) -> Result<Vec<HsaMemoryPool>, Status> {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agent_iterate_memory_pools")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_iterate_regions")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_agent_iterate_caches")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_cache_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
    runtime: &HsaState,
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_pool_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_region_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_allocate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_free")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_memory_free(pointer: *mut c_void) -> Status {
    if pointer.is_null() {
        return boundary(|| {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            if guard.as_ref().is_some() {
                SUCCESS
            } else {
                NOT_INITIALIZED
            }
        });
    }
    // SAFETY: Both entry points release allocations owned by this runtime.
    unsafe { hsa_amd_memory_pool_free(pointer) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_register")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_memory_register(pointer: *mut c_void, size: usize) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        if !pointer.is_null() && size == 0 {
            return INVALID_ARGUMENT;
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_deregister")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_memory_deregister(_pointer: *mut c_void, _size: usize) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_some() {
            SUCCESS
        } else {
            NOT_INITIALIZED
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_assign_agent")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agent_memory_pool_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let (
            generation,
            device,
            kind,
            rounded,
            page,
            permissions,
            owner,
            global_flags,
            uncached,
            device_index,
            tracker,
            _token,
        ) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(generation) = guard.active_generation() else {
                return NOT_INITIALIZED;
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
                (0, cpu_pool_memory_kind(pool, runtime.lifetime, flags))
            } else if let Some((index, pool_kind)) = runtime.decode_gpu_pool(pool) {
                if pool_kind == GPU_POOL_GROUP {
                    return INVALID_ALLOCATION;
                }
                let uncached = flags & ALLOC_UNCACHED != 0;
                (
                    index,
                    MemoryKind::DeviceLocal {
                        host_visible: gpu_pool_host_visible(runtime, index),
                        coherent: gpu_pool_coherent(pool_kind, uncached, flags & ALLOC_PCIE != 0),
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
            let uncached = flags & ALLOC_UNCACHED != 0 || matches!(kind, MemoryKind::System);
            let tracker = runtime.inflight.clone();
            let Some(token) = tracker.enter() else {
                return OUT_OF_RESOURCES;
            };
            (
                generation,
                runtime.gpus[device_index].device.clone(),
                kind,
                rounded,
                page,
                permissions,
                owner,
                global_flags,
                uncached,
                device_index,
                tracker,
                token,
            )
        };
        let allocation = match device.allocate(kind, rounded as u64, page as u64, permissions) {
            Ok(allocation) => allocation,
            Err(error) => return map_error(error),
        };
        let info = allocation.info();
        let pointer_value = info.host_address.unwrap_or(info.device_address as usize);
        let accessible = vec![HsaAgent {
            handle: GPU_AGENT_BASE + device_index as u64,
        }];
        let memory = Memory::new(
            allocation,
            size,
            owner,
            global_flags,
            uncached,
            flags,
            accessible,
        );
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return retire_unpublished_memory(memory, &tracker, status),
        };
        if guard.active_generation() != Some(generation) {
            drop(guard);
            return retire_unpublished_memory(memory, &tracker, INVALID_RUNTIME_STATE);
        }
        let Some(runtime) = guard.as_mut() else {
            drop(guard);
            return retire_unpublished_memory(memory, &tracker, NOT_INITIALIZED);
        };
        if runtime.memory_state.allocations.try_reserve(1).is_err()
            || runtime
                .memory_state
                .allocations
                .contains_key(&pointer_value)
        {
            drop(guard);
            return retire_unpublished_memory(memory, &tracker, OUT_OF_RESOURCES);
        }
        runtime
            .memory_state
            .allocations
            .insert(pointer_value, memory);
        // SAFETY: The caller supplied writable output storage, and the
        // allocation is registered before its address is exposed.
        unsafe { pointer.write(pointer_value as *mut c_void) };
        SUCCESS
    })
}

fn retire_unpublished_memory(
    mut memory: Memory,
    tracker: &crate::runtime::InFlightTracker,
    status: Status,
) -> Status {
    match memory.free() {
        Ok(()) => status,
        Err(error) => {
            tracker.quarantine();
            std::mem::forget(memory);
            map_error(error)
        }
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_pool_allocate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_memory_pool_allocate(
    pool: HsaMemoryPool,
    size: usize,
    flags: u32,
    pointer: *mut *mut c_void,
) -> Status {
    // SAFETY: This forwards the public ABI arguments to the common allocator.
    unsafe { memory_pool_allocate(pool, size, flags, pointer, INVALID_MEMORY_POOL) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_register_deallocation_callback")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .memory_state
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
            .memory_state
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_deregister_deallocation_callback")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .memory_state
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
            .memory_state
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_pool_free")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_memory_pool_free(pointer: *mut c_void) -> Status {
    boundary(|| {
        enum Removed {
            Allocation { key: usize, memory: Memory },
            Locked { index: usize, memory: LockedMemory },
        }

        if pointer.is_null() {
            return INVALID_ARGUMENT;
        }

        // Transfer the native owner out of the public registry before calling
        // the driver. The token keeps its session alive if shutdown starts.
        let (mut removed, token, tracker) = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            let base = runtime
                .memory_state
                .allocations
                .get(&(pointer as usize))
                .map_or(pointer as usize, |memory| memory.description.agent_base);
            if copy_borrowed(runtime, base) {
                return OUT_OF_RESOURCES;
            }
            let removed = if let Some(memory) =
                runtime.memory_state.allocations.remove(&(pointer as usize))
            {
                Removed::Allocation {
                    key: pointer as usize,
                    memory,
                }
            } else {
                let Some(index) = runtime
                    .memory_state
                    .locked_allocations
                    .iter()
                    .rposition(|memory| memory.description.agent_base == pointer as usize)
                else {
                    return INVALID_ALLOCATION;
                };
                if copy_borrowed(
                    runtime,
                    runtime.memory_state.locked_allocations[index]
                        .description
                        .agent_base,
                ) {
                    return OUT_OF_RESOURCES;
                }
                Removed::Locked {
                    index,
                    memory: runtime.memory_state.locked_allocations.remove(index),
                }
            };
            let tracker = runtime.inflight.clone();
            let Some(token) = tracker.enter() else {
                match removed {
                    Removed::Allocation { key, memory } => {
                        runtime.memory_state.allocations.insert(key, memory);
                    }
                    Removed::Locked { index, memory } => {
                        runtime
                            .memory_state
                            .locked_allocations
                            .insert(index, memory);
                    }
                }
                return OUT_OF_RESOURCES;
            };
            (removed, token, tracker)
        };
        let result = match &mut removed {
            Removed::Allocation { memory, .. } => memory.free(),
            Removed::Locked { memory, .. } => memory.free(),
        };
        if let Err(error) = result {
            let status = map_error(error);
            let mut slot = Some(removed);
            if let Ok(mut guard) = lock() {
                if let Some(runtime) = guard.as_mut() {
                    let can_restore = slot.as_ref().is_some_and(|owner| match owner {
                        Removed::Allocation { key, .. } => {
                            !runtime.memory_state.allocations.contains_key(key)
                        }
                        Removed::Locked { memory, .. } => !runtime
                            .memory_state
                            .locked_allocations
                            .iter()
                            .any(|current| {
                                current.description.agent_base == memory.description.agent_base
                            }),
                    });
                    if can_restore {
                        if let Some(owner) = slot.take() {
                            match owner {
                                Removed::Allocation { key, memory } => {
                                    runtime.memory_state.allocations.insert(key, memory);
                                }
                                Removed::Locked { index, memory } => {
                                    let index =
                                        index.min(runtime.memory_state.locked_allocations.len());
                                    runtime
                                        .memory_state
                                        .locked_allocations
                                        .insert(index, memory);
                                }
                            }
                        }
                    }
                }
            }
            if let Some(owner) = slot {
                // Shutdown already detached the generation, or the registry
                // is poisoned. Its native context cannot now be retired.
                tracker.quarantine();
                std::mem::forget(owner);
            }
            drop(token);
            return status;
        }
        let callbacks = match removed {
            Removed::Allocation { mut memory, .. } => {
                std::mem::take(&mut memory.deallocation_callbacks)
            }
            Removed::Locked { .. } => Vec::new(),
        };
        drop(token);
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_memory_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(memory) = runtime.memory_state.allocations.values().find(|memory| {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_memory_attach")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        if runtime.memory_state.ipc_allocations.try_reserve(1).is_err()
            || runtime
                .memory_state
                .ipc_allocations
                .contains_key(&imported_pointer)
        {
            return OUT_OF_RESOURCES;
        }
        runtime.memory_state.ipc_allocations.insert(
            imported_pointer,
            Memory::new_ipc(allocation, size, owner, accessible),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { mapped_pointer.write(imported_pointer as *mut c_void) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ipc_memory_detach")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .memory_state
            .ipc_allocations
            .get(&key)
            .map_or(key, |memory| memory.description.agent_base);
        if copy_borrowed(runtime, base) {
            return OUT_OF_RESOURCES;
        }
        let Some(mut memory) = runtime.memory_state.ipc_allocations.remove(&key) else {
            return INVALID_ARGUMENT;
        };
        if memory.free().is_ok() {
            SUCCESS
        } else {
            runtime.memory_state.ipc_allocations.insert(key, memory);
            INVALID_ARGUMENT
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_external_semaphore_handle_open")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_external_semaphore_handle_close")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_amd_external_semaphore_handle_close(
    _semaphore: HsaAmdExternalSemaphore,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        if guard.as_ref().is_none() {
            return NOT_INITIALIZED;
        }
        ERROR
    })
}

unsafe fn publish_ais_result(
    result: linux_interop::AisFileResult,
    size_copied: *mut u64,
    operation_status: *mut i32,
) -> Status {
    // SAFETY: Both outputs are optional in the public ABI and, when non-null,
    // point to caller-owned writable storage for the duration of this call.
    unsafe {
        if !size_copied.is_null() {
            size_copied.write(result.size_copied);
        }
        if !operation_status.is_null() {
            operation_status.write(result.status);
        }
    }
    if result.status == 0 { SUCCESS } else { ERROR }
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
    if size > 0 && (host == 0 || host.checked_add(size).is_none() || size > isize::MAX as usize) {
        return ERROR;
    }
    // SAFETY: ais_file_io validates the full host mapping and holds the
    // runtime lock until this synchronous positioned I/O call returns.
    let buffer = match operation {
        AisOperation::Read => linux_interop::AisHostBuffer::Read(if size == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(host as *mut u8, size) }
        }),
        AisOperation::Write => linux_interop::AisHostBuffer::Write(if size == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(host as *const u8, size) }
        }),
    };
    let result = linux_interop::ais_host_transfer(descriptor, buffer, file_offset);
    // SAFETY: The caller owns the optional output storage for this call.
    unsafe { publish_ais_result(result, size_copied, operation_status) }
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
    let transfer_size = match usize::try_from(size.min(linux_interop::AIS_MAX_TRANSFER_BYTES)) {
        Ok(size) => size,
        Err(_) => return ERROR,
    };
    let Some(description) = pointer_description(runtime, device_pointer as usize) else {
        return ERROR;
    };
    if description.owner.handle == CPU_AGENT {
        return ERROR;
    }
    let address = device_pointer as usize;
    let Some(offset) = description.offset_range(address, transfer_size) else {
        return ERROR;
    };
    // Hold the runtime lock through either synchronous transfer, preventing
    // another HSA call from freeing the allocation during the borrow.
    let status = if let Some(host) = description.host_address(address) {
        // SAFETY: The mapped host range stays owned by the locked runtime.
        // The public call keeps its descriptor live until this call returns.
        unsafe {
            ais_transfer(
                descriptor,
                host,
                transfer_size,
                file_offset,
                size_copied,
                operation_status,
                operation,
            )
        }
    } else {
        let Some(memory) = owned_memory(runtime, address).filter(|memory| {
            memory.description.agent_base == description.agent_base
                && memory.description.host_base == description.host_base
                && memory.description.offset_range(address, transfer_size) == Some(offset)
        }) else {
            return ERROR;
        };
        let offset = match u64::try_from(offset) {
            Ok(offset) => offset,
            Err(_) => return ERROR,
        };
        let size = match u64::try_from(transfer_size) {
            Ok(size) => size,
            Err(_) => return ERROR,
        };
        match linux_interop::ais_transfer(
            &memory.allocation,
            descriptor,
            offset,
            size,
            file_offset,
            operation,
        ) {
            Ok(result) => {
                // SAFETY: Optional outputs belong to the caller for this call.
                unsafe { publish_ais_result(result, size_copied, operation_status) }
            }
            Err(error) => {
                // The ioctl input overlaps its output. On error KFD's copied
                // count cannot be trusted, so leave size_copied untouched.
                let errno = error
                    .native_error_code()
                    .filter(|errno| *errno > 0)
                    .unwrap_or(EIO);
                if !operation_status.is_null() {
                    // SAFETY: The non-null optional output is caller-owned.
                    unsafe { operation_status.write(-errno) };
                }
                ERROR
            }
        }
    };
    drop(guard);
    status
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ais_file_write")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_ais_file_read")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
    let Some((memory, pointer_offset)) =
        runtime
            .memory_state
            .allocations
            .values()
            .find_map(|memory| {
                memory
                    .description
                    .offset_range(pointer as usize, size)
                    .map(|pointer_offset| (memory, pointer_offset))
            })
    else {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_portable_export_dmabuf")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_portable_export_dmabuf_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_portable_close_dmabuf")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

    if runtime
        .memory_state
        .locked_allocations
        .try_reserve(1)
        .is_err()
    {
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
                    cpu_pool_mapping_policy(pool, flags & ALLOC_UNCACHED),
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
    runtime
        .memory_state
        .locked_allocations
        .push(LockedMemory::new(
            allocation,
            host_base,
            device_base,
            size,
            effective_global_flags(global_flags, flags & ALLOC_UNCACHED != 0, false),
            accessible,
        ));
    // SAFETY: The caller supplied writable pointer storage.
    unsafe { agent_ptr.write(device_base as *mut c_void) };
    SUCCESS
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_lock")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_lock_to_pool")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_unlock")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .memory_state
            .locked_allocations
            .iter()
            .rposition(|memory| memory.description.host_base == Some(host_ptr as usize))
        else {
            return INVALID_ARGUMENT;
        };
        if copy_borrowed(
            runtime,
            runtime.memory_state.locked_allocations[index]
                .description
                .agent_base,
        ) {
            return OUT_OF_RESOURCES;
        }
        let mut memory = runtime.memory_state.locked_allocations.remove(index);
        match memory.free() {
            Ok(()) => SUCCESS,
            Err(error) => {
                runtime
                    .memory_state
                    .locked_allocations
                    .insert(index, memory);
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
    if runtime
        .memory_state
        .interop_allocations
        .try_reserve(1)
        .is_err()
    {
        return OUT_OF_RESOURCES;
    }
    if runtime
        .memory_state
        .interop_allocations
        .contains_key(&imported_pointer)
    {
        return ERROR;
    }
    runtime.memory_state.interop_allocations.insert(
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_interop_map_buffer")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_interop_map_buffer_with_size")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_interop_unmap_buffer")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .memory_state
            .interop_allocations
            .get(&key)
            .map_or(key, |memory| memory.description.agent_base);
        if copy_borrowed(runtime, base) {
            return OUT_OF_RESOURCES;
        }
        if let Some(mut memory) = runtime.memory_state.interop_allocations.remove(&key) {
            if memory.free().is_err() {
                runtime.memory_state.interop_allocations.insert(key, memory);
            }
        }
        SUCCESS
    })
}

fn vmem_reservation_base(runtime: &HsaState, address: usize, size: usize) -> Option<usize> {
    let (&base, reservation) = runtime
        .memory_state
        .vmem_reservations
        .range(..=address)
        .next_back()?;
    reservation.contains_range(address, size).then_some(base)
}

fn vmem_mapping_bases(
    runtime: &HsaState,
    address: usize,
    size: usize,
) -> Result<Vec<usize>, Status> {
    let end = address.checked_add(size).ok_or(INVALID_ARGUMENT)?;
    let mut bases = Vec::new();
    let mut cursor = address;
    while cursor < end {
        let mapping = runtime
            .memory_state
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
    runtime: &mut HsaState,
    mapping_base: usize,
    descriptor: HsaAmdMemoryAccessDesc,
) -> Status {
    let existing_permission = runtime
        .memory_state
        .vmem_mappings
        .get(&mapping_base)
        .and_then(|mapping| mapping.access.get(&descriptor.agent_handle.handle))
        .map(|access| access.permissions);
    if existing_permission == Some(descriptor.permissions) {
        return SUCCESS;
    }
    if descriptor.permissions != ACCESS_PERMISSION_NONE {
        let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) else {
            return INVALID_ALLOCATION;
        };
        if existing_permission.is_none() && mapping.access.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
    }
    let old_access = runtime
        .memory_state
        .vmem_mappings
        .get_mut(&mapping_base)
        .and_then(|mapping| mapping.access.remove(&descriptor.agent_handle.handle));
    if let Some(mut access) = old_access {
        if let Err(error) = access.free() {
            if let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) {
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

    let Some(mapping) = runtime.memory_state.vmem_mappings.get(&mapping_base) else {
        return INVALID_ALLOCATION;
    };
    let gpu_index = if descriptor.agent_handle.handle == CPU_AGENT {
        runtime
            .gpus
            .iter()
            .position(|gpu| gpu.info.fabric_handles_supported)
            .unwrap_or(0)
    } else {
        let Some(index) = runtime.gpu_index(descriptor.agent_handle) else {
            return INVALID_AGENT;
        };
        index
    };
    let Some(gpu) = runtime.gpus.get(gpu_index) else {
        return NOT_SUPPORTED;
    };
    let Some(handle) = runtime.memory_state.vmem_handles.get_mut(&mapping.handle) else {
        return INVALID_ALLOCATION;
    };
    if matches!(&handle.backing, VmemBacking::Fabric { .. }) && !gpu.info.fabric_handles_supported {
        return NOT_SUPPORTED;
    }
    let memory = match handle.memory_for_access(gpu_index, &gpu.device) {
        Ok(memory) => memory,
        Err(status) => return status,
    };
    let Some(reservation) = runtime
        .memory_state
        .vmem_reservations
        .get(&mapping.reservation)
    else {
        return INVALID_ALLOCATION;
    };
    let Some(permissions) = DeviceAccess::from_bits(descriptor.permissions) else {
        return INVALID_ARGUMENT;
    };
    let owner = if descriptor.agent_handle.handle == CPU_AGENT {
        match memory.map_host(
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
        match gpu.device.map_virtual_memory(
            memory,
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
    let Some(mapping) = runtime.memory_state.vmem_mappings.get_mut(&mapping_base) else {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_address_reserve")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_address_reserve_align")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let reservation = match VirtualAddress::reserve(&devices, size as u64, alignment, requested)
        {
            Ok(reservation) => reservation,
            Err(error) => return map_error(error),
        };
        let base = reservation.info().address as usize;
        if runtime.memory_state.vmem_reservations.contains_key(&base) {
            return ERROR;
        }
        runtime.memory_state.vmem_reservations.insert(
            base,
            VmemReservation::new(reservation, flags & VMEM_ADDRESS_NO_REGISTER == 0),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { address.write(base as *mut c_void) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_address_free")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            let Some(mut reservation) = runtime.memory_state.vmem_reservations.remove(&key) else {
                return INVALID_ALLOCATION;
            };
            if reservation.description.size != size {
                runtime
                    .memory_state
                    .vmem_reservations
                    .insert(key, reservation);
                return INVALID_ARGUMENT;
            }
            if reservation.mappings != 0 {
                runtime
                    .memory_state
                    .vmem_reservations
                    .insert(key, reservation);
                return RESOURCE_FREE;
            }
            if let Err(error) = reservation.free() {
                runtime
                    .memory_state
                    .vmem_reservations
                    .insert(key, reservation);
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_handle_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let uncached = flags & u64::from(ALLOC_UNCACHED) != 0;
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
                    coherent: gpu_pool_coherent(pool_kind, uncached, false),
                    uncached,
                    contiguous: false,
                },
            )
        } else {
            return INVALID_ARGUMENT;
        };
        if runtime.memory_state.vmem_handles.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let memory = match runtime.gpus[device_index].device.create_virtual_memory(
            kind,
            size as u64,
            memory_type == MEMORY_TYPE_PINNED,
            uncached,
        ) {
            Ok(memory) => memory,
            Err(error) => return map_error(error),
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.memory_state.vmem_handles.insert(
            handle,
            VmemHandle::new(VmemBacking::Owned(memory), pool, memory_type),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_handle_release")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(handle) = runtime
            .memory_state
            .vmem_handles
            .get_mut(&memory_handle.handle)
        else {
            return INVALID_ALLOCATION;
        };
        if handle.references == 0 {
            return INVALID_ALLOCATION;
        }
        handle.references -= 1;
        if handle.references != 0 || handle.mappings != 0 {
            return SUCCESS;
        }
        let Some(mut handle) = runtime
            .memory_state
            .vmem_handles
            .remove(&memory_handle.handle)
        else {
            return ERROR;
        };
        if let Err(error) = handle.free() {
            handle.references = 1;
            runtime
                .memory_state
                .vmem_handles
                .insert(memory_handle.handle, handle);
            return map_error(error);
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_map")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ARGUMENT;
        };
        let Some(extent) = offset.checked_add(size) else {
            return INVALID_ARGUMENT;
        };
        if handle
            .known_size()
            .is_some_and(|known_size| extent as u64 > known_size)
        {
            return INVALID_ARGUMENT;
        }
        if runtime
            .memory_state
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
                .memory_state
                .vmem_mappings
                .range(base..)
                .next()
                .is_some_and(|(&mapped, _)| mapped < end)
        {
            return INVALID_ARGUMENT;
        }
        let Some(handle) = runtime
            .memory_state
            .vmem_handles
            .get_mut(&memory_handle.handle)
        else {
            return ERROR;
        };
        handle.mappings = match handle.mappings.checked_add(1) {
            Some(count) => count,
            None => return OUT_OF_RESOURCES,
        };
        let Some(reservation_owner) = runtime.memory_state.vmem_reservations.get_mut(&reservation)
        else {
            handle.mappings -= 1;
            return ERROR;
        };
        let Some(reservation_mappings) = reservation_owner.mappings.checked_add(1) else {
            handle.mappings -= 1;
            return OUT_OF_RESOURCES;
        };
        reservation_owner.mappings = reservation_mappings;
        runtime.memory_state.vmem_mappings.insert(
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_unmap")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            let Some(mut mapping) = runtime.memory_state.vmem_mappings.remove(&base) else {
                return ERROR;
            };
            if let Err(error) = mapping.free_access() {
                runtime.memory_state.vmem_mappings.insert(base, mapping);
                return map_error(error);
            }
            if let Some(reservation) = runtime
                .memory_state
                .vmem_reservations
                .get_mut(&mapping.reservation)
            {
                reservation.mappings = reservation.mappings.saturating_sub(1);
            }
            let release_handle =
                if let Some(handle) = runtime.memory_state.vmem_handles.get_mut(&mapping.handle) {
                    handle.mappings = handle.mappings.saturating_sub(1);
                    handle.mappings == 0 && handle.references == 0
                } else {
                    false
                };
            if release_handle {
                let Some(mut handle) = runtime.memory_state.vmem_handles.remove(&mapping.handle)
                else {
                    return ERROR;
                };
                if let Err(error) = handle.free() {
                    runtime
                        .memory_state
                        .vmem_handles
                        .insert(mapping.handle, handle);
                    return map_error(error);
                }
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_set_access")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_get_access")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .memory_state
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_export_shareable_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        if handle.imported() {
            return INCOMPATIBLE_ARGUMENTS;
        }
        let Some(memory) = handle.owned_memory() else {
            return ERROR;
        };
        let dma_buf = match linux_interop::export_virtual_memory(memory) {
            Ok(dma_buf) => dma_buf,
            Err(error) => return map_error(error),
        };
        // SAFETY: The caller supplied writable output storage and ownership of
        // the descriptor is transferred only after export succeeds.
        unsafe { descriptor.write(dma_buf.into_fd().into_raw_fd()) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_import_shareable_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        if runtime.memory_state.vmem_handles.try_reserve(1).is_err() {
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
        runtime.memory_state.vmem_handles.insert(
            handle,
            VmemHandle::new(
                VmemBacking::DmaBuf(memory),
                HsaMemoryPool { handle: 0 },
                MEMORY_TYPE_NONE,
            ),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_retain_alloc_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(mapping) = runtime.memory_state.vmem_mappings.get(&(address as usize)) else {
            return INVALID_ALLOCATION;
        };
        let handle_id = mapping.handle;
        let Some(handle) = runtime.memory_state.vmem_handles.get_mut(&handle_id) else {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_get_alloc_properties_from_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_export_fabric_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_export_fabric_handle(
    fabric_handle: *mut HsaFabricHandle,
    memory_handle: HsaAmdVmemAllocHandle,
    flags: u64,
) -> Status {
    boundary(|| {
        if fabric_handle.is_null() || flags != 0 {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime.memory_state.vmem_handles.get(&memory_handle.handle) else {
            return INVALID_ALLOCATION;
        };
        if handle.imported() {
            return INCOMPATIBLE_ARGUMENTS;
        }
        let Some((index, _)) = runtime.decode_gpu_pool(handle.pool) else {
            return INVALID_ARGUMENT;
        };
        if !runtime.gpus[index].info.fabric_handles_supported {
            return NOT_SUPPORTED;
        }
        let Some(memory) = handle.owned_memory() else {
            return ERROR;
        };
        match linux_interop::fabric_virtual_memory_ready(memory) {
            Ok(true) => (),
            Ok(false) => return RESOURCE_NOT_READY,
            Err(error) => return map_error(error),
        }
        let fabric = match linux_interop::export_fabric_virtual_memory(memory) {
            Ok(fabric) => fabric,
            Err(error) if error.kind() == rocddi::ErrorKind::Busy => return RESOURCE_NOT_READY,
            Err(error) => return map_error(error),
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { fabric_handle.write(HsaFabricHandle { handle: fabric }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_vmem_import_fabric_handle")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_amd_vmem_import_fabric_handle(
    fabric_handle: HsaFabricHandle,
    memory_handle: *mut HsaAmdVmemAllocHandle,
) -> Status {
    boundary(|| {
        if memory_handle.is_null() {
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
        if runtime.memory_state.vmem_handles.try_reserve(1).is_err() {
            return OUT_OF_RESOURCES;
        }
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.memory_state.vmem_handles.insert(
            handle,
            VmemHandle::new(
                VmemBacking::Fabric {
                    token: fabric_handle.handle,
                    imports: HashMap::new(),
                },
                HsaMemoryPool { handle: 0 },
                MEMORY_TYPE_NONE,
            ),
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { memory_handle.write(HsaAmdVmemAllocHandle { handle }) };
        SUCCESS
    })
}

fn owned_memory(runtime: &HsaState, address: usize) -> Option<&Memory> {
    runtime
        .memory_state
        .allocations
        .values()
        .find(|memory| memory.contains(address))
        .or_else(|| {
            runtime
                .memory_state
                .ipc_allocations
                .values()
                .find(|memory| memory.contains(address))
        })
        .or_else(|| {
            runtime
                .memory_state
                .interop_allocations
                .values()
                .find(|memory| memory.contains(address))
        })
}

fn pointer_description(runtime: &HsaState, address: usize) -> Option<PointerDescription> {
    vmem_pointer_description(runtime, address).or_else(|| {
        owned_memory(runtime, address)
            .map(|memory| memory.description.clone())
            .or_else(|| {
                runtime
                    .memory_state
                    .locked_allocations
                    .iter()
                    .rev()
                    .find(|memory| memory.contains(address))
                    .map(|memory| memory.description.clone())
            })
    })
}

fn vmem_pointer_description(runtime: &HsaState, address: usize) -> Option<PointerDescription> {
    if let Some((_, mapping)) = runtime
        .memory_state
        .vmem_mappings
        .range(..=address)
        .next_back()
        .filter(|(_, mapping)| mapping.contains(address))
    {
        let handle = runtime.memory_state.vmem_handles.get(&mapping.handle)?;
        let reservation = runtime
            .memory_state
            .vmem_reservations
            .get(&mapping.reservation)?;
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
        .memory_state
        .vmem_reservations
        .range(..=address)
        .next_back()
        .filter(|(_, reservation)| reservation.contains_range(address, 1))
        .map(|(_, reservation)| reservation.description.clone())
}

fn host_copy_address(
    runtime: &HsaState,
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
    crate::loader::loaded_host_range(&runtime.loader, address, size)
        .map(|host| host.unwrap_or(address))
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_pointer_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

fn set_pointer_userdata(runtime: &mut HsaState, address: usize, user_data: usize) -> bool {
    if let Some((_, reservation)) = runtime
        .memory_state
        .vmem_reservations
        .range_mut(..=address)
        .next_back()
        .filter(|(_, reservation)| reservation.contains_range(address, 1))
    {
        reservation.description.user_data = user_data;
        return true;
    }
    for memory in runtime.memory_state.allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.memory_state.ipc_allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.memory_state.interop_allocations.values_mut() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    for memory in runtime.memory_state.locked_allocations.iter_mut().rev() {
        if memory.description.starts_at(address) {
            memory.description.user_data = user_data;
            return true;
        }
    }
    false
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_pointer_info_set_userdata")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_agents_allow_access")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .memory_state
            .allocations
            .iter()
            .find(|(_, memory)| memory.contains(pointer as usize))
            .map(|(key, _)| *key)
        else {
            // This API is defined for memory-pool allocations. IPC, graphics,
            // and locked ranges need their own native mapping lifecycle.
            return INVALID_ALLOCATION;
        };
        let memory = &runtime.memory_state.allocations[&key];
        if copy_borrowed(runtime, memory.description.agent_base) {
            return OUT_OF_RESOURCES;
        }
        let owner = memory.description.owner;
        let storage = if owner.handle == CPU_AGENT {
            PoolStorage::System
        } else if memory.pool_global_flags & POOL_FLAG_FINE != 0 {
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
        let Some(memory) = runtime.memory_state.allocations.get_mut(&key) else {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_pool_can_migrate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_migrate")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_memory_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let quarantine = runtime.memory_state.async_copy_quarantine.clone();
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_fill")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            let host =
                match crate::loader::loaded_host_range(&runtime.loader, pointer as usize, size) {
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
        let quarantine = runtime.memory_state.async_copy_quarantine.clone();
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
    let page_size = rocddi::memory::host::host_page_size().map_err(map_error)?;
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
    runtime: &HsaState,
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

fn svm_agent_from_location(runtime: &HsaState, location: SvmLocation) -> Result<HsaAgent, Status> {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_attributes_set")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_attributes_get")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
    /// `HsaState` whose signal and allocation counts are being updated.
    runtime: &'a mut HsaState,
    /// Signals retained for the pending worker.
    handles: Arc<Vec<HsaSignal>>,
    /// Prefix successfully retained before a possible failure.
    retained: usize,
    /// Allocation bases borrowed until worker ownership is published.
    borrowed_memory: Vec<usize>,
}

impl<'a> PendingAsyncSignals<'a> {
    fn new(runtime: &'a mut HsaState, handles: Arc<Vec<HsaSignal>>) -> Result<Self, Status> {
        let mut pending = Self {
            runtime,
            handles,
            retained: 0,
            borrowed_memory: Vec::new(),
        };
        for signal in pending.handles.iter() {
            pending.runtime.signal_state.retain_async_signal(*signal)?;
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

fn retain_copy_borrow(runtime: &mut HsaState, base: usize) -> Result<(), Status> {
    if !runtime.memory_state.async_copy_borrows.contains_key(&base) {
        runtime
            .memory_state
            .async_copy_borrows
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
    }
    let count = runtime
        .memory_state
        .async_copy_borrows
        .entry(base)
        .or_default();
    *count = count.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
    Ok(())
}

fn release_copy_borrow(runtime: &mut HsaState, base: usize) {
    if let Some(count) = runtime.memory_state.async_copy_borrows.get_mut(&base) {
        *count -= 1;
        if *count == 0 {
            runtime.memory_state.async_copy_borrows.remove(&base);
        }
    }
}

fn copy_borrowed(runtime: &HsaState, base: usize) -> bool {
    runtime
        .memory_state
        .async_copy_borrows
        .get(&base)
        .is_some_and(|count| *count != 0)
}

impl Drop for PendingAsyncSignals<'_> {
    fn drop(&mut self) {
        for signal in self.handles.iter().take(self.retained) {
            self.runtime.signal_state.release_async_signal(*signal);
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
                runtime.signal_state.release_async_signal(*signal);
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

fn complete_async_signal_value(signal: &crate::signal::AmdSignal, success: bool) {
    if success {
        signal.value.fetch_sub(1, Ordering::Release);
    } else {
        signal.value.store(-1, Ordering::Release);
    }
}

fn finish_async_signal(completion: HsaSignal, success: bool) {
    if completion.handle == 0 {
        return;
    }
    // SAFETY: Submission retains this signal until the worker exits, and
    // shutdown keeps its storage mapped until all workers have joined.
    let _ = unsafe {
        crate::signal::with_signal(completion, |signal| {
            complete_async_signal_value(signal, success);
        })
    };
}

fn finish_async_copy_signal(
    completion: HsaSignal,
    success: bool,
    profile: Option<AsyncCopyProfile>,
) {
    if completion.handle == 0 {
        return;
    }
    let profile = if success { profile } else { None };
    let mut guard = lock().ok();
    if let Some(runtime) = guard.as_mut().and_then(|guard| guard.as_mut()) {
        if let Some(record) = runtime
            .signal_state
            .async_signal_refs
            .get_mut(&(completion.handle as usize))
        {
            record.copy_profile = profile;
        }
    }
    // SAFETY: Submission retains this signal until the worker exits, and
    // shutdown keeps its storage mapped until all workers have joined. The
    // registry lock orders profile publication with the completion update.
    let _ = unsafe {
        crate::signal::with_signal(completion, |signal| {
            if let Some(profile) = profile {
                signal.start_ts.store(profile.start, Ordering::Relaxed);
                signal.end_ts.store(profile.end, Ordering::Relaxed);
            }
            complete_async_signal_value(signal, success);
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_prefetch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .any(|dependency| !runtime.signal_state.owns_signal(*dependency))
            || (completion.handle != 0 && !runtime.signal_state.owns_signal(completion))
        {
            return INVALID_SIGNAL;
        }
        let mut signal_handles = Vec::new();
        if signal_handles
            .try_reserve_exact(dependency_slice.len() + usize::from(completion.handle != 0))
            .is_err()
            || runtime.memory_state.workers.try_reserve(1).is_err()
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
        pending.runtime.memory_state.workers.push(worker);
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_discard_batch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        if guard.as_ref().is_none() {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_svm_discard_and_prefetch_batch_async")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        if guard.as_ref().is_none() {
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

/// Execution path selected for an asynchronous HSA copy request.
enum AsyncCopyRoute {
    /// Zero-byte copy that only completes dependencies and its signal.
    Noop,
    /// CPU copy performed after dependency signals complete.
    Host {
        destination: HostCopyDestination,
        source: HostCopySource,
        size: usize,
    },
    /// One native SDMA operation on an activated GPU.
    Gpu {
        device: Box<Device>,
        index: usize,
        entry: GpuCopyEntry,
    },
    /// Several ordered SDMA operations sharing one native sequence.
    GpuBatch {
        device: Box<Device>,
        index: usize,
        entries: Vec<GpuCopyEntry>,
        controls: BatchCopyControls,
    },
}

/// One operation within a native GPU copy route.
enum GpuCopyEntry {
    /// Rectangular copy with row and slice strides.
    Rect(CopyRect),
    /// Direct linear copy.
    Linear {
        destination: u64,
        source: u64,
        size: u64,
    },
    /// Exchange between two GPU address ranges.
    Swap {
        address_a: u64,
        address_b: u64,
        size: u64,
    },
    /// Copy whose addresses may come from GPU-visible pointer slots.
    Indirect {
        destination: u64,
        source: u64,
        size: u64,
        indirect_source: bool,
        indirect_destination: bool,
    },
    /// Copy from host storage staged for SDMA access.
    FromHost {
        destination: u64,
        source: HostCopySource,
    },
    /// Copy to a host destination staged for SDMA access.
    ToHost {
        destination: HostCopyDestination,
        source: u64,
        size: usize,
    },
}

/// Optional wait and signal attached to a native batch copy.
#[derive(Clone, Copy, Default)]
struct BatchCopyControls {
    /// GPU comparison that must pass before the batch begins.
    wait: Option<GpuCopyWait>,
    /// GPU atomic notification after the final packet.
    signal: Option<GpuCopySignal>,
}

/// Native route selected for one synchronous HSA copy.
enum SyncGpuCopyRoute {
    /// Direct copy between GPU-addressable ranges.
    Direct {
        device: Box<Device>,
        destination: u64,
        source: u64,
        size: u64,
    },
    /// Copy from a CPU-only source through staging.
    FromHost {
        device: Box<Device>,
        destination: u64,
        source: HostCopySource,
    },
    /// Copy into a CPU-only destination through staging.
    ToHost {
        device: Box<Device>,
        destination: HostCopyDestination,
        source: u64,
        size: usize,
    },
}

/// CPU source storage and when its bytes must be captured.
enum HostCopySource {
    /// Direct host pointer and byte extent.
    Address(usize, usize),
    /// Loader-owned mapping resolved after dependencies finish.
    LoadedAddress(usize, usize),
    /// Owned snapshot of a loader mapping for immediate copy.
    LoadedBytes(Vec<u8>),
}

/// Point at which a loader-owned source is read.
#[derive(Clone, Copy)]
enum HostSourceTiming {
    /// Snapshot before releasing the registry lock.
    Immediate,
    /// Resolve the live loader mapping after dependency completion.
    AfterDependencies,
}

/// CPU destination, including loader mappings needing a later lookup.
enum HostCopyDestination {
    /// Direct host address.
    Address(usize),
    /// Loader-owned address resolved while the registry lock is held.
    LoadedAddress(usize),
}

/// Completion data that must be published after a synchronous GPU copy.
enum SyncGpuCopyOutcome {
    /// Native and host copies completed without deferred publication.
    Done,
    /// Staged bytes to write to a loader-owned mapping.
    LoadedDestination { address: usize, bytes: Vec<u8> },
}

/// Frontend status and whether native work may still read its operands.
struct CopyRouteFailure {
    /// HSA status translated from validation or native failure.
    status: Status,
    /// Requires retaining borrowed memory until retirement is proved.
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

impl From<rocddi::device::gpu::CopyFailure> for CopyRouteFailure {
    fn from(failure: rocddi::device::gpu::CopyFailure) -> Self {
        Self {
            status: map_error(failure.error),
            operands_may_be_live: failure.operands_may_be_live,
        }
    }
}

/// Synchronous route and public allocations borrowed by its operands.
struct ResolvedSyncGpuCopy {
    /// Native copy plan selected after pointer validation.
    route: SyncGpuCopyRoute,
    /// Base addresses whose allocation owners remain borrowed.
    borrowed_memory: [Option<usize>; 2],
}

/// Asynchronous route and public allocations borrowed by its operands.
struct ResolvedAsyncCopy {
    /// Native or host copy plan selected after pointer validation.
    route: AsyncCopyRoute,
    /// Base addresses whose allocation owners remain borrowed.
    borrowed_memory: [Option<usize>; 2],
}

fn copy_agent(
    runtime: &HsaState,
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
    runtime: &HsaState,
    description: &PointerDescription,
) -> Option<(usize, HsaAgent)> {
    std::iter::once(description.owner)
        .chain(description.accessible.iter().copied())
        .filter(|agent| description.accessible.contains(agent))
        .find_map(|agent| runtime.gpu_index(agent).map(|index| (index, agent)))
}

fn copy_pointer_description(
    runtime: &HsaState,
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
    runtime: &HsaState,
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

/// Resolves a caller-owned image backing range to this agent's GPU address.
/// Image data lifetime remains the caller's responsibility under HSA.
pub(crate) fn image_device_address(
    runtime: &HsaState,
    pointer: usize,
    size: usize,
    agent: HsaAgent,
    permission: u32,
) -> Result<u64, Status> {
    let description = copy_pointer_description(runtime, pointer, size)?;
    gpu_copy_address(
        runtime,
        description.as_ref(),
        pointer,
        size,
        agent,
        permission,
    )
    .map(|(address, _)| address)
}

fn prepare_host_copy_source(
    runtime: &HsaState,
    description: Option<&PointerDescription>,
    pointer: usize,
    host: usize,
    size: usize,
    timing: HostSourceTiming,
) -> Result<HostCopySource, Status> {
    if description.is_none()
        && crate::loader::loaded_host_range(&runtime.loader, pointer, size)?.is_some()
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
    runtime: &HsaState,
    description: Option<&PointerDescription>,
    pointer: usize,
    host: usize,
    size: usize,
) -> Result<HostCopyDestination, Status> {
    if description.is_none()
        && crate::loader::loaded_host_range(&runtime.loader, pointer, size)?.is_some()
    {
        Ok(HostCopyDestination::LoadedAddress(pointer))
    } else {
        Ok(HostCopyDestination::Address(host))
    }
}

fn vmem_copy_access(
    runtime: &HsaState,
    description: &PointerDescription,
    agent: HsaAgent,
    required_access: u32,
) -> Result<(), Status> {
    if description.pointer_type != POINTER_TYPE_HSA_VMEM {
        return Ok(());
    }
    let mapping = runtime
        .memory_state
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
    runtime: &HsaState,
    destination: usize,
    dst_agent: HsaAgent,
    source: usize,
    src_agent: HsaAgent,
    size: usize,
    copy_gpu_index: Option<usize>,
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
    if let Some(index) = copy_gpu_index.or(src_gpu.or(dst_gpu)) {
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
                route: AsyncCopyRoute::Gpu {
                    device,
                    index,
                    entry: GpuCopyEntry::Linear {
                        destination,
                        source,
                        size: u64::try_from(size).map_err(|_| INVALID_ARGUMENT)?,
                    },
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
                    route: AsyncCopyRoute::Gpu {
                        device,
                        index,
                        entry: GpuCopyEntry::FromHost {
                            destination,
                            source,
                        },
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
                    route: AsyncCopyRoute::Gpu {
                        device,
                        index,
                        entry: GpuCopyEntry::ToHost {
                            destination,
                            source,
                            size,
                        },
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
    runtime: &HsaState,
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
                None,
            )?;
            let AsyncCopyRoute::Gpu {
                device,
                entry:
                    GpuCopyEntry::Linear {
                        destination,
                        source,
                        size,
                    },
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

fn copy_gpu(device: &Device) -> Result<rocddi::device::gpu::GpuDevice<'_>, CopyRouteFailure> {
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
    let host = crate::loader::loaded_host_range(&runtime.loader, address, size)?
        .ok_or(INVALID_ALLOCATION)?;
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
    let host = match crate::loader::loaded_host_range(&runtime.loader, address, bytes.len()) {
        Ok(Some(host)) => host,
        Ok(None) | Err(_) => return INVALID_ALLOCATION,
    };
    // SAFETY: The registry lock retains the complete loaded mapping until
    // the copied bytes and their system-scope ordering are published.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), host as *mut u8, bytes.len()) };
    fence(Ordering::SeqCst);
    SUCCESS
}

fn execute_gpu_entries(
    device: &Device,
    entries: impl ExactSizeIterator<Item = GpuCopyEntry>,
    controls: BatchCopyControls,
    sdma_engine: Option<u32>,
    timed: bool,
    stop: &AtomicBool,
) -> Result<Option<GpuCopyTimestamps>, CopyRouteFailure> {
    let gpu = copy_gpu(device)?;
    let mut sequence = match sdma_engine {
        Some(engine) => GpuCopySequence::begin_on_sdma_engine(gpu, stop, engine),
        None => GpuCopySequence::begin(gpu, stop),
    }
    .map_err(CopyRouteFailure::from)?;
    if timed {
        sequence.enable_timing().map_err(CopyRouteFailure::from)?;
    }
    let count = entries.len();
    for (ordinal, entry) in entries.enumerate() {
        let options = GpuCopyOptions {
            wait: if ordinal == 0 { controls.wait } else { None },
            signal: if ordinal + 1 == count {
                controls.signal
            } else {
                None
            },
            ..GpuCopyOptions::default()
        };
        let outcome = match entry {
            GpuCopyEntry::Rect(rect) => {
                // SAFETY: Preparation validated every selected row and keeps
                // its mapped backing live until native retirement.
                unsafe { sequence.copy_rect(rect) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Linear {
                destination,
                source,
                size,
            } => {
                // SAFETY: Copy preparation checked both mapped GPU ranges and
                // retains their owners until native retirement is proved.
                unsafe { sequence.copy_linear_with_options(destination, source, size, options) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Swap {
                address_a,
                address_b,
                size,
            } => {
                // SAFETY: Preparation verified read and write access to both
                // ranges on the selected GPU and retained their owners.
                unsafe { sequence.swap_linear_with_options(address_a, address_b, size, options) }
                    .map(|()| SyncGpuCopyOutcome::Done)
                    .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::Indirect {
                destination,
                source,
                size,
                indirect_source,
                indirect_destination,
            } => {
                // SAFETY: Preparation checked access to every direct range and
                // indirect pointer slot. The caller owns the resolved targets
                // through completion, including after an uncertain submission.
                unsafe {
                    sequence.copy_linear_with_options(
                        destination,
                        source,
                        size,
                        GpuCopyOptions {
                            indirect_source,
                            indirect_destination,
                            ..options
                        },
                    )
                }
                .map(|()| SyncGpuCopyOutcome::Done)
                .map_err(CopyRouteFailure::from)
            }
            GpuCopyEntry::FromHost {
                destination,
                source,
            } => with_host_copy_source(&source, |host| {
                // SAFETY: HSA checked the GPU destination and retains its
                // backing. rocddi owns the staged copy of these host bytes.
                unsafe { sequence.copy_from_host(destination, host) }
                    .map_err(CopyRouteFailure::from)
            })
            .map(|()| SyncGpuCopyOutcome::Done),
            GpuCopyEntry::ToHost {
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
    if timed {
        sequence
            .finish_timing()
            .map(Some)
            .map_err(CopyRouteFailure::from)
    } else {
        Ok(None)
    }
}

fn system_copy_tick(device: Option<&Device>) -> Result<u64, CopyRouteFailure> {
    let device = device.ok_or_else(|| CopyRouteFailure::retired(ERROR))?;
    let counters = copy_gpu(device)?
        .clock_counters()
        .map_err(|error| CopyRouteFailure::retired(map_error(error)))?;
    if counters.system == 0 {
        return Err(CopyRouteFailure::retired(ERROR));
    }
    Ok(counters.system)
}

fn gpu_copy_profile(index: usize, ticks: Option<GpuCopyTimestamps>) -> Option<AsyncCopyProfile> {
    ticks.map(|ticks| AsyncCopyProfile {
        clock: AsyncCopyClock::Gpu(index),
        start: ticks.start,
        end: ticks.end,
    })
}

fn execute_copy_route(
    route: AsyncCopyRoute,
    sdma_engine: Option<u32>,
    profile_enabled: bool,
    host_clock: Option<&Device>,
    stop: &AtomicBool,
) -> Result<Option<AsyncCopyProfile>, CopyRouteFailure> {
    match route {
        AsyncCopyRoute::Noop => Ok(None),
        AsyncCopyRoute::Host {
            destination,
            source,
            size,
        } => {
            if stop.load(Ordering::Acquire) {
                return Err(CopyRouteFailure::retired(ERROR));
            }
            let start = profile_enabled
                .then(|| system_copy_tick(host_clock))
                .transpose()?;
            let needs_loaded_mapping = matches!(&source, HostCopySource::LoadedAddress(..))
                || matches!(&destination, HostCopyDestination::LoadedAddress(..));
            let guard = if needs_loaded_mapping {
                match lock() {
                    Ok(guard) => Some(guard),
                    Err(_) => return Err(CopyRouteFailure::retired(ERROR)),
                }
            } else {
                None
            };
            let runtime = guard.as_ref().and_then(|guard| guard.as_ref());
            let loaded_host = |address| {
                runtime.and_then(|runtime| {
                    crate::loader::loaded_host_range(&runtime.loader, address, size)
                        .ok()
                        .flatten()
                })
            };
            let source_host = match &source {
                HostCopySource::Address(address, _) => *address,
                HostCopySource::LoadedAddress(address, _) => {
                    let Some(host) = loaded_host(*address) else {
                        return Err(CopyRouteFailure::retired(INVALID_ALLOCATION));
                    };
                    host
                }
                HostCopySource::LoadedBytes(bytes) => bytes.as_ptr() as usize,
            };
            let destination_host = match destination {
                HostCopyDestination::Address(address) => address,
                HostCopyDestination::LoadedAddress(address) => {
                    let Some(host) = loaded_host(address) else {
                        return Err(CopyRouteFailure::retired(INVALID_ALLOCATION));
                    };
                    host
                }
            };
            // SAFETY: The caller keeps ordinary host ranges live through
            // completion. HsaState-owned memory is borrowed, and the registry
            // lock keeps loaded mappings live for this copy. ptr::copy also
            // permits overlapping ranges.
            unsafe { std::ptr::copy(source_host as *const u8, destination_host as *mut u8, size) };
            fence(Ordering::SeqCst);
            drop(guard);
            let end = profile_enabled
                .then(|| system_copy_tick(host_clock))
                .transpose()?;
            let profile = start.zip(end).map(|(start, end)| AsyncCopyProfile {
                clock: AsyncCopyClock::System,
                start,
                end,
            });
            if profile.is_some_and(|profile| profile.end < profile.start) {
                return Err(CopyRouteFailure::retired(ERROR));
            }
            Ok(profile)
        }
        AsyncCopyRoute::Gpu {
            device,
            index,
            entry,
        } => execute_gpu_entries(
            &device,
            std::iter::once(entry),
            BatchCopyControls::default(),
            sdma_engine,
            profile_enabled,
            stop,
        )
        .map(|ticks| gpu_copy_profile(index, ticks)),
        AsyncCopyRoute::GpuBatch {
            device,
            index,
            entries,
            controls,
        } => execute_gpu_entries(
            &device,
            entries.into_iter(),
            controls,
            None,
            profile_enabled,
            stop,
        )
        .map(|ticks| gpu_copy_profile(index, ticks)),
    }
}

/// Enqueued copy with its operands retained until completion signaling.
struct AsyncCopyTask {
    /// Validated host or native copy route.
    route: AsyncCopyRoute,
    /// Optional dedicated SDMA engine selected by the caller.
    sdma_engine: Option<u32>,
    /// Allocation borrows released after native retirement.
    borrowed_memory: Vec<Option<usize>>,
    /// HSA signal decremented when the copy task completes.
    completion: HsaSignal,
}

fn enqueue_copy_job(
    runtime: &mut HsaState,
    tasks: Vec<AsyncCopyTask>,
    dependency_slice: &[HsaSignal],
) -> Status {
    let mut handles = Vec::new();
    let Some(handle_count) = dependency_slice.len().checked_add(tasks.len()) else {
        return OUT_OF_RESOURCES;
    };
    if handles.try_reserve_exact(handle_count).is_err()
        || runtime.memory_state.workers.try_reserve(1).is_err()
    {
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
    let profile_enabled = matches!(
        pending.runtime.memory_state.async_copy_profiling,
        AsyncCopyProfiling::Enabled
    );
    let host_clock = if profile_enabled
        && tasks
            .iter()
            .any(|task| matches!(&task.route, AsyncCopyRoute::Host { .. }))
    {
        pending.runtime.gpus.first().map(|gpu| gpu.device.clone())
    } else {
        None
    };
    let worker_handles = pending.handles.clone();
    let quarantine = pending.runtime.memory_state.async_copy_quarantine.clone();
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
                let result = if execute {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute_copy_route(
                            task.route,
                            task.sdma_engine,
                            profile_enabled,
                            host_clock.as_ref(),
                            &stop,
                        )
                    }))
                    .unwrap_or(Err(CopyRouteFailure {
                        status: ERROR,
                        operands_may_be_live: true,
                    }))
                } else {
                    Err(CopyRouteFailure::retired(ERROR))
                };
                let (success, may_be_live, profile) = match result {
                    Ok(profile) => (true, false, profile),
                    Err(failure) => (false, failure.operands_may_be_live, None),
                };
                if may_be_live {
                    quarantine.store(true, Ordering::Release);
                }
                // Release only this operation's backing before publishing its
                // completion. Later operations keep their own borrows.
                release_async_resources(&[], &task.borrowed_memory, may_be_live);
                finish_async_copy_signal(task.completion, success, profile);
                execute = success;
            }
            release_async_resources(&worker_handles, &[], false);
        }) {
        Ok(worker) => worker,
        Err(_) => return OUT_OF_RESOURCES,
    };
    for completion in pending.handles.iter().skip(dependency_len) {
        if let Some(record) = pending
            .runtime
            .signal_state
            .async_signal_refs
            .get_mut(&(completion.handle as usize))
        {
            record.copy_profile = None;
        }
    }
    pending.transfer_to_worker();
    pending.runtime.memory_state.workers.push(worker);
    SUCCESS
}

fn enqueue_async_copy(
    runtime: &mut HsaState,
    resolved: ResolvedAsyncCopy,
    dependency_slice: &[HsaSignal],
    completion: HsaSignal,
    sdma_engine: Option<u32>,
) -> Status {
    enqueue_copy_job(
        runtime,
        vec![AsyncCopyTask {
            route: resolved.route,
            sdma_engine,
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
        if !runtime.signal_state.owns_signal(completion) {
            return INVALID_SIGNAL;
        }
        if engine.is_some_and(|engine| engine.count_ones() != 1 || engine > u16::MAX.into()) {
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
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        if size == 0 {
            if !runtime.is_agent(dst_agent) || !runtime.is_agent(src_agent) {
                return INVALID_AGENT;
            }
            if let Some(mask) = engine {
                match copy_engine_mask(runtime, dst_agent, src_agent) {
                    Ok(available) if available & mask != 0 => {}
                    Ok(_) => return INVALID_ARGUMENT,
                    Err(status) => return status,
                }
            }
            return SUCCESS;
        }
        let resolved = match resolve_async_copy(
            runtime,
            dst as usize,
            dst_agent,
            src as usize,
            src_agent,
            size,
            None,
        ) {
            Ok(resolved) => resolved,
            Err(status) => return status,
        };
        let sdma_engine = match (engine, &resolved.route) {
            (Some(mask), AsyncCopyRoute::Gpu { index, .. }) => {
                let available = match copy_engine_mask_for_gpu(runtime, *index) {
                    Ok(available) => available,
                    Err(status) => return status,
                };
                if available & mask == 0 {
                    return INVALID_ARGUMENT;
                }
                Some(mask.trailing_zeros())
            }
            (Some(_), _) => return NOT_SUPPORTED,
            (None, _) => None,
        };
        enqueue_async_copy(runtime, resolved, dependency_slice, completion, sdma_engine)
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_async_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_async_copy_on_engine")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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

/// One validated input operation in an HSA async copy batch.
#[derive(Clone, Copy)]
struct BatchCopyEntry {
    /// Destination address supplied by the caller.
    destination: usize,
    /// Agent expected to reach the destination.
    destination_agent: HsaAgent,
    /// Source address supplied by the caller.
    source: usize,
    /// Agent expected to reach the source.
    source_agent: HsaAgent,
    /// Number of bytes transferred by this operation.
    size: usize,
    /// Packet shape and address mode.
    mode: BatchCopyMode,
}

/// Native packet shape selected for a batch entry.
#[derive(Clone, Copy)]
enum BatchCopyMode {
    /// Ordinary linear copy.
    Linear,
    /// Exchange two ranges.
    Swap,
    /// Resolve source or destination through GPU pointer slots.
    Indirect { source: bool, destination: bool },
}

fn batch_copy_mode(operation: u16) -> Result<BatchCopyMode, Status> {
    match operation {
        0 | 1 => Ok(BatchCopyMode::Linear),
        2 => Ok(BatchCopyMode::Swap),
        3 => Ok(BatchCopyMode::Indirect {
            source: true,
            destination: false,
        }),
        4 => Ok(BatchCopyMode::Indirect {
            source: false,
            destination: true,
        }),
        5 => Ok(BatchCopyMode::Indirect {
            source: true,
            destination: true,
        }),
        _ => Err(INVALID_ARGUMENT),
    }
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
    let mode = batch_copy_mode(op.operation)?;
    if matches!(mode, BatchCopyMode::Indirect { .. }) && op.entry_count > 1024 {
        return Err(INVALID_ARGUMENT);
    }
    let count = usize::from(op.entry_count.max(1));
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| OUT_OF_RESOURCES)?;
    if op.entry_count == 0 {
        if op.operation == 1 {
            return Err(INVALID_ARGUMENT);
        }
        // SAFETY: The scalar members of the public C unions are active here.
        let destination_agent = unsafe { op.destination_agent.agent };
        let size = unsafe { op.size.bytes };
        if op.secondary_size
            != if matches!(mode, BatchCopyMode::Swap) {
                size
            } else {
                0
            }
        {
            return Err(INVALID_ARGUMENT);
        }
        entries.push(BatchCopyEntry {
            destination: op.destination as usize,
            destination_agent,
            source: op.source as usize,
            source_agent: op.source_agent,
            size,
            mode,
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
                        mode,
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
                        mode,
                    });
                }
            }
            2..=5 => {
                let sources = unsafe { batch_slice(op.source.cast::<*mut c_void>(), count)? };
                let sizes = unsafe { batch_slice(op.size.list, count)? };
                for index in 0..count {
                    entries.push(BatchCopyEntry {
                        destination: destinations[index] as usize,
                        destination_agent: agents[index],
                        source: sources[index] as usize,
                        source_agent: op.source_agent,
                        size: sizes[index],
                        mode,
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
            || (entry.size == 0
                && (op.entry_count != 0 || !matches!(entry.mode, BatchCopyMode::Linear)))
    }) {
        return Err(INVALID_ARGUMENT);
    }
    Ok(entries)
}

fn resolve_indirect_batch_entry(
    runtime: &HsaState,
    entry: &BatchCopyEntry,
    indirect_source: bool,
    indirect_destination: bool,
    selected: Option<usize>,
) -> Result<ResolvedAsyncCopy, Status> {
    if entry.size > 0x3fff_ffe0 {
        return Err(INVALID_ARGUMENT);
    }
    let source_size = if indirect_source { 8 } else { entry.size };
    let destination_size = if indirect_destination { 8 } else { entry.size };
    let source_description = copy_pointer_description(runtime, entry.source, source_size)?;
    let destination_description =
        copy_pointer_description(runtime, entry.destination, destination_size)?;
    let source_agent = copy_agent(runtime, entry.source_agent, source_description.as_ref())?;
    let destination_agent = copy_agent(
        runtime,
        entry.destination_agent,
        destination_description.as_ref(),
    )?;
    let index = selected
        .or(runtime.gpu_index(source_agent))
        .or(runtime.gpu_index(destination_agent))
        .ok_or(INVALID_AGENT)?;
    if runtime.gpus[index].info.gfx_minor != 5 {
        return Err(NOT_SUPPORTED);
    }
    let gpu_agent = HsaAgent {
        handle: GPU_AGENT_BASE + index as u64,
    };
    let (source, source_base) = gpu_copy_address(
        runtime,
        source_description.as_ref(),
        entry.source,
        source_size,
        gpu_agent,
        ACCESS_PERMISSION_RO,
    )?;
    let (destination, destination_base) = gpu_copy_address(
        runtime,
        destination_description.as_ref(),
        entry.destination,
        destination_size,
        gpu_agent,
        if indirect_destination {
            ACCESS_PERMISSION_RO
        } else {
            ACCESS_PERMISSION_WO
        },
    )?;
    if indirect_source && source % 8 != 0 || indirect_destination && destination % 8 != 0 {
        return Err(INVALID_ARGUMENT);
    }
    Ok(ResolvedAsyncCopy {
        route: AsyncCopyRoute::Gpu {
            device: Box::new(runtime.gpus[index].device.clone()),
            index,
            entry: GpuCopyEntry::Indirect {
                destination,
                source,
                size: u64::try_from(entry.size).map_err(|_| INVALID_ARGUMENT)?,
                indirect_source,
                indirect_destination,
            },
        },
        borrowed_memory: [Some(destination_base), Some(source_base)],
    })
}

fn resolve_batch_copy_entries(
    runtime: &HsaState,
    entries: &[BatchCopyEntry],
) -> Result<(AsyncCopyRoute, Vec<Option<usize>>), Status> {
    let mut prepared_entries = Vec::new();
    prepared_entries
        .try_reserve_exact(entries.len())
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut borrowed_memory = Vec::new();
    borrowed_memory
        .try_reserve_exact(
            entries
                .len()
                .checked_mul(4)
                .and_then(|count| count.checked_add(2))
                .ok_or(OUT_OF_RESOURCES)?,
        )
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut selected: Option<(usize, Box<Device>)> = None;
    for entry in entries {
        if entry.size != 0
            && runtime.gpu_index(entry.source_agent).is_none()
            && runtime.gpu_index(entry.destination_agent).is_none()
        {
            return Err(INVALID_AGENT);
        }
        let selected_index = selected.as_ref().map(|(index, _)| *index);
        let resolved = match entry.mode {
            BatchCopyMode::Indirect {
                source,
                destination,
            } => resolve_indirect_batch_entry(runtime, entry, source, destination, selected_index)?,
            BatchCopyMode::Linear | BatchCopyMode::Swap => resolve_async_copy(
                runtime,
                entry.destination,
                entry.destination_agent,
                entry.source,
                entry.source_agent,
                entry.size,
                selected_index,
            )?,
        };
        let (device, index, prepared) = match resolved.route {
            AsyncCopyRoute::Gpu {
                device,
                index,
                entry,
            } => (device, index, entry),
            AsyncCopyRoute::Noop if entries.len() == 1 => {
                return Ok((AsyncCopyRoute::Noop, borrowed_memory));
            }
            _ => return Err(INVALID_AGENT),
        };
        let prepared = if matches!(entry.mode, BatchCopyMode::Swap) {
            let GpuCopyEntry::Linear {
                destination,
                source,
                size,
            } = prepared
            else {
                return Err(NOT_SUPPORTED);
            };
            if runtime.gpus[index].info.gfx_minor != 5 {
                return Err(NOT_SUPPORTED);
            }
            let end_a = source.checked_add(size).ok_or(INVALID_ARGUMENT)?;
            let end_b = destination.checked_add(size).ok_or(INVALID_ARGUMENT)?;
            if source % 32 != 0 || destination % 32 != 0 || (source < end_b && destination < end_a)
            {
                return Err(INVALID_ARGUMENT);
            }
            let reverse = resolve_async_copy(
                runtime,
                entry.source,
                entry.source_agent,
                entry.destination,
                entry.destination_agent,
                entry.size,
                Some(index),
            )?;
            if !matches!(
                reverse.route,
                AsyncCopyRoute::Gpu {
                    index: reverse_index,
                    entry: GpuCopyEntry::Linear { .. },
                    ..
                } if reverse_index == index
            ) {
                return Err(NOT_SUPPORTED);
            }
            borrowed_memory.extend(reverse.borrowed_memory);
            GpuCopyEntry::Swap {
                address_a: source,
                address_b: destination,
                size,
            }
        } else {
            prepared
        };
        // A multi-entry operation uses the source GPU, or its first
        // destination GPU for host sources. Later peer destinations must be
        // accessible from that same GPU before one ordered DDI submission.
        selected.get_or_insert((index, device));
        prepared_entries.push(prepared);
        borrowed_memory.extend(resolved.borrowed_memory);
    }
    let (index, device) = selected.ok_or(INVALID_ARGUMENT)?;
    Ok((
        AsyncCopyRoute::GpuBatch {
            device,
            index,
            entries: prepared_entries,
            controls: BatchCopyControls::default(),
        },
        borrowed_memory,
    ))
}

/// Copy tasks staged for publication as one async batch.
struct PreparedBatchCopy {
    /// Tasks to enqueue after every entry validates.
    tasks: Vec<AsyncCopyTask>,
}

fn batch_control_address(
    runtime: &HsaState,
    pointer: usize,
    gpu: HsaAgent,
    permission: u32,
) -> Result<(u64, usize), Status> {
    let description = copy_pointer_description(runtime, pointer, 8)?;
    let (address, base) =
        gpu_copy_address(runtime, description.as_ref(), pointer, 8, gpu, permission)?;
    if address % 8 != 0 {
        return Err(INVALID_ARGUMENT);
    }
    Ok((address, base))
}

#[allow(unsafe_code)]
unsafe fn prepare_batch_copy(
    runtime: &HsaState,
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
        if !runtime.signal_state.owns_signal(op.completion_signal) {
            return Err(INVALID_SIGNAL);
        }
        if op.traffic_class != 0 {
            return Err(NOT_SUPPORTED);
        }
        if (op.wait.function == 0
            && (!op.wait.address.is_null()
                || op.wait.value != 0
                || op.wait.mask != 0
                || op.wait.scope != 0 && op.operation < 3))
            || (op.wait.function != 0 && op.wait.address.is_null())
            || (op.signal.operation == 0
                && (op.signal.scope != 0 || !op.signal.address.is_null() || op.signal.data != 0))
            || (op.signal.operation != 0 && op.signal.address.is_null())
        {
            return Err(INVALID_ARGUMENT);
        }
        if op.operation == 1 && runtime.gpu_index(op.source_agent).is_none() {
            return Err(INVALID_AGENT);
        }
        // SAFETY: Each operation selects the documented scalar or array union
        // members. All array values are copied before the caller returns.
        let entries = unsafe { batch_copy_entries(op)? };
        let (mut route, mut bases) = resolve_batch_copy_entries(runtime, &entries)?;
        if let AsyncCopyRoute::GpuBatch {
            index,
            entries,
            controls,
            ..
        } = &mut route
        {
            if (op.wait.function != 0 || op.signal.operation != 0)
                && (runtime.gpus[*index].info.gfx_minor != 5
                    || entries.iter().any(|entry| {
                        !matches!(
                            entry,
                            GpuCopyEntry::Linear { .. }
                                | GpuCopyEntry::Swap { .. }
                                | GpuCopyEntry::Indirect { .. }
                        )
                    }))
            {
                return Err(NOT_SUPPORTED);
            }
            let gpu = HsaAgent {
                handle: GPU_AGENT_BASE + *index as u64,
            };
            if op.wait.function != 0 {
                let (address, base) = batch_control_address(
                    runtime,
                    op.wait.address as usize,
                    gpu,
                    ACCESS_PERMISSION_RO,
                )?;
                controls.wait = Some(GpuCopyWait {
                    address,
                    function: u8::try_from(op.wait.function).map_err(|_| INVALID_ARGUMENT)?,
                    reference: op.wait.value,
                    mask: op.wait.mask,
                });
                bases.push(Some(base));
            }
            if op.signal.operation != 0 {
                let (address, base) = batch_control_address(
                    runtime,
                    op.signal.address as usize,
                    gpu,
                    ACCESS_PERMISSION_RW,
                )?;
                controls.signal = Some(GpuCopySignal {
                    address,
                    operation: match op.signal.operation {
                        1 => GpuCopySignalOperation::Write,
                        2 => GpuCopySignalOperation::Add,
                        3 => GpuCopySignalOperation::Subtract,
                        _ => return Err(INVALID_ARGUMENT),
                    },
                    data: op.signal.data,
                });
                bases.push(Some(base));
            }
        } else if op.wait.function != 0 || op.signal.operation != 0 {
            return Err(INVALID_ARGUMENT);
        }
        tasks.push(AsyncCopyTask {
            route,
            sdma_engine: None,
            borrowed_memory: bases,
            completion: op.completion_signal,
        });
    }
    Ok(PreparedBatchCopy { tasks })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_async_batch_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_async_copy_rect")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        if !runtime.signal_state.owns_signal(completion) {
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
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
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
                route: AsyncCopyRoute::Gpu {
                    device: Box::new(runtime.gpus[index].device.clone()),
                    index,
                    entry: GpuCopyEntry::Rect(CopyRect {
                        destination,
                        source,
                        width: u64::from(range.x),
                        height: range.y,
                        depth: range.z,
                        destination_pitch: dst.pitch as u64,
                        source_pitch: src.pitch as u64,
                        destination_slice: dst.slice as u64,
                        source_slice: src.slice as u64,
                    }),
                },
                borrowed_memory: [Some(dst_base), Some(src_base)],
            },
            dependency_slice,
            completion,
            None,
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

fn copy_engine_mask_for_gpu(runtime: &HsaState, index: usize) -> Result<u32, Status> {
    let gpu = runtime.gpus[index].device.gpu().map_err(map_error)?;
    if !gpu.supports_linear_copy() {
        return Ok(0);
    }
    let count = gpu.info().queues.sdma_engine_count.min(16);
    let valid_ids = (1_u32 << count) - 1;
    Ok(gpu.available_sdma_engines().map_err(map_error)? & valid_ids)
}

fn copy_engine_mask(runtime: &HsaState, dst: HsaAgent, src: HsaAgent) -> Result<u32, Status> {
    let Some(index) = runtime.gpu_index(src).or_else(|| runtime.gpu_index(dst)) else {
        return Ok(0);
    };
    copy_engine_mask_for_gpu(runtime, index)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_copy_engine_status")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let available = match copy_engine_mask(runtime, dst, src) {
            Ok(available) => available,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { mask.write(available) };
        if available == 0 {
            OUT_OF_RESOURCES
        } else {
            SUCCESS
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_memory_get_preferred_copy_engine")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
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
        let available = match copy_engine_mask(runtime, dst, src) {
            Ok(available) => available,
            Err(status) => return status,
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { mask.write(available & available.wrapping_neg()) };
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
        assert!(lock().is_ok_and(|runtime| runtime.as_ref().is_none()));

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
        assert!(lock().is_ok_and(|runtime| runtime.as_ref().is_none()));

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
        for (pool, cache) in [
            (CPU_POOL_FINE, HostMappingPolicy::Fine),
            (CPU_POOL_EXTENDED, HostMappingPolicy::Extended),
            (CPU_POOL_COARSE, HostMappingPolicy::Coarse),
        ] {
            assert_eq!(
                cpu_pool_memory_kind(
                    HsaMemoryPool { handle: pool },
                    DriverContextLifetime::Process,
                    0
                ),
                MemoryKind::OwnedHost { policy: cache }
            );
        }
        assert_eq!(
            cpu_pool_memory_kind(
                HsaMemoryPool {
                    handle: CPU_POOL_KERNARG,
                },
                DriverContextLifetime::Process,
                0
            ),
            MemoryKind::System
        );
        for pool in [CPU_POOL_FINE, CPU_POOL_EXTENDED, CPU_POOL_COARSE] {
            assert_eq!(
                cpu_pool_memory_kind(
                    HsaMemoryPool { handle: pool },
                    DriverContextLifetime::Session,
                    0
                ),
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
