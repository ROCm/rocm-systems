// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA memory ownership, pools, allocations, and pointer metadata.
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
//! as portable native-handle behavior. Child modules own IPC and interop,
//! virtual memory, and copy/SVM operations; this root retains the shared
//! `MemoryState` and its shutdown order.

mod copy_svm;
mod interop;
pub(crate) use copy_svm::hsa_memory_copy;
pub(crate) use copy_svm::image_device_address;
use copy_svm::*;
mod virtual_memory;

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
    VirtualHostMapping, VirtualMemory, VirtualMemoryPlacement,
};
use rocddi::session::{DriverContextLifetime, Session};
use rocddi::topology::{MemoryLinkInfo, MemoryLinkType};

use crate::ffi::*;
use crate::runtime::{
    HsaState, InFlightGeneration, boundary, cache_handles_in_level_order, initialized_mut, lock,
    map_error,
};
use crate::signal::{AsyncCopyClock, AsyncCopyProfile, SignalState};

const GPU_POOL_COARSE: u64 = 1;
pub(crate) const GPU_POOL_FINE: u64 = 2;
const GPU_POOL_GROUP: u64 = 3;
pub(crate) const GPU_POOL_EXTENDED: u64 = 4;
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

fn valid_access_permissions(access: u32) -> bool {
    access & !(ACCESS_PERMISSION_RW | ACCESS_PERMISSION_EX) == 0
}

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

    /// Join completed workers before accepting another asynchronous operation.
    /// `is_finished` ensures joining cannot wait for work that needs the runtime
    /// lock held by the submitter.
    fn reap_finished_workers(&mut self) {
        let mut index = 0;
        while index < self.workers.len() {
            if self.workers[index].is_finished() {
                let _ = self.workers.swap_remove(index).join();
            } else {
                index += 1;
            }
        }
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
        // Mappings borrow handles and reservations. Release them first, then
        // the physical handles and finally their reserved address ranges.
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
            Some((_, GPU_POOL_EXTENDED)) => POOL_FLAG_EXTENDED_FINE,
            Some((_, GPU_POOL_GROUP)) => 0,
            _ => return None,
        },
    };
    Some(flags)
}

fn gpu_agent_pools(index: usize, fine_grain: bool, extended: bool) -> Vec<HsaMemoryPool> {
    let mut pools = Vec::with_capacity(usize::from(fine_grain) + usize::from(extended) + 2);
    pools.push(HsaState::gpu_pool(index, GPU_POOL_COARSE));
    if extended {
        pools.push(HsaState::gpu_pool(index, GPU_POOL_EXTENDED));
    }
    if fine_grain {
        pools.push(HsaState::gpu_pool(index, GPU_POOL_FINE));
    }
    pools.push(HsaState::gpu_pool(index, GPU_POOL_GROUP));
    pools
}

fn gpu_agent_regions(index: usize, fine_grain: bool, extended: bool) -> Vec<HsaMemoryPool> {
    let mut regions = gpu_agent_pools(index, fine_grain, extended);
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

/// A partial visible VRAM aperture can back individual KFD PUBLIC allocations.
/// KFD decides whether each requested allocation fits that host-mappable pool.
fn gpu_pool_has_host_aperture(runtime: &HsaState, index: usize) -> bool {
    runtime.gpus[index].endpoint.host_visible_local_memory_bytes != 0
}

/// Queue and signal mechanisms that can address arbitrary VRAM need full BAR
/// visibility rather than a host-mappable subset.
fn gpu_pool_fully_host_visible(runtime: &HsaState, index: usize) -> bool {
    let endpoint = &runtime.gpus[index].endpoint;
    endpoint.local_memory_bytes != 0
        && endpoint.host_visible_local_memory_bytes >= endpoint.local_memory_bytes
}

#[derive(Clone, Copy)]
pub(crate) enum SignalBatchPool {
    System,
    Device {
        index: usize,
        uncached: bool,
        flush: bool,
    },
}

pub(crate) fn signal_batch_pool(
    runtime: &HsaState,
    pool: HsaMemoryPool,
) -> Option<SignalBatchPool> {
    if matches!(pool.handle, CPU_POOL_FINE | CPU_POOL_KERNARG) {
        return Some(SignalBatchPool::System);
    }
    let (index, kind) = runtime.decode_gpu_pool(pool)?;
    let gpu = &runtime.gpus[index];
    let uncached =
        kind == GPU_POOL_EXTENDED && gpu.info.gfx_major == 9 && matches!(gpu.info.gfx_minor, 4 | 5);
    let coarse = kind == GPU_POOL_COARSE && gpu.info.gfx_major == 12 && gpu.info.gfx_minor >= 5;
    let host_atomic_link = runtime
        .topology
        .route_from_host(&gpu.endpoint)
        .is_some_and(|link| {
            link.kind == MemoryLinkType::Xgmi && link.coherent_support && link.atomic_support_64bit
        });
    (gpu_pool_fully_host_visible(runtime, index) && (uncached || coarse)).then_some(
        SignalBatchPool::Device {
            index,
            uncached,
            flush: !host_atomic_link,
        },
    )
}

fn gpu_pcie_dmabuf_supported(runtime: &HsaState, index: usize) -> bool {
    // Full VRAM visibility supplies a large BAR. A CPU-to-GPU XGMI link is
    // the other route that can satisfy a requested PCIe DMA-BUF mapping.
    gpu_pool_fully_host_visible(runtime, index)
        || runtime
            .topology
            .route_from_host(&runtime.gpus[index].endpoint)
            .is_some_and(|link| link.kind == MemoryLinkType::Xgmi)
}

fn gpu_pool_coherent(pool_kind: u64, uncached: bool, pcie: bool) -> bool {
    matches!(pool_kind, GPU_POOL_FINE | GPU_POOL_EXTENDED) || uncached || pcie
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
        owner_gpu.and_then(|index| {
            runtime
                .topology
                .route_from_host(&runtime.gpus[index].endpoint)
        })
    } else if let Some(requester) = requester_gpu {
        if matches!(storage, PoolStorage::System) {
            runtime
                .topology
                .route_to_host(&runtime.gpus[requester].endpoint)
        } else {
            owner_gpu.and_then(|owner| {
                runtime.topology.route_between(
                    &runtime.gpus[requester].endpoint,
                    &runtime.gpus[owner].endpoint,
                )
            })
        }
    } else {
        None
    };
    let linked = same
        || match (requester_gpu, owner_gpu) {
            (Some(requester), Some(owner))
                if matches!(storage, PoolStorage::LocalCoarse | PoolStorage::LocalFine) =>
            {
                runtime.session.can_access_local_memory(
                    &runtime.gpus[requester].endpoint,
                    &runtime.gpus[owner].endpoint,
                )
            }
            _ => link.is_some_and(|link| link.hop_count() != 0),
        };
    let cpu_cannot_access_local = agent.handle == CPU_AGENT
        && owner_gpu.is_some_and(|index| !gpu_pool_has_host_aperture(runtime, index))
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
        min_latency: link.minimum_latency,
        max_latency: link.maximum_latency,
        min_bandwidth: link.minimum_bandwidth,
        max_bandwidth: link.maximum_bandwidth,
        atomic_support_32bit: link.atomic_support_32bit,
        atomic_support_64bit: link.atomic_support_64bit,
        coherent_support: link.coherent_support,
        link_type: hsa_link_type(link.kind),
        numa_distance: link.numa_distance,
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
        let gpu = &runtime.gpus[index];
        Ok(gpu_agent_pools(
            index,
            gpu.fine_grain_pool,
            gpu.info.gfx_major == 9 && matches!(gpu.info.gfx_minor, 4 | 5),
        ))
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
        runtime.gpus[index].info.gfx_major == 9
            && matches!(runtime.gpus[index].info.gfx_minor, 4 | 5),
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
            cache_handles_in_level_order(&runtime.caches, agent)
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
            POOL_INFO_SIGNAL_CAPABLE => value
                .cast::<bool>()
                .write(signal_batch_pool(runtime, pool).is_some()),
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
        if pointer.is_null()
            || access == ACCESS_PERMISSION_NONE
            || !valid_access_permissions(access)
        {
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
            device,
            kind,
            rounded,
            page,
            permissions,
            owner,
            global_flags,
            uncached,
            device_index,
            in_flight,
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
                let uncached = flags & ALLOC_UNCACHED != 0 || pool_kind == GPU_POOL_EXTENDED;
                // A partial BAR can map an individual public allocation, but
                // the rest of local VRAM must remain available for private
                // allocations from the same HSA pool.
                let host_visible = u64::try_from(rounded).is_ok_and(|size| {
                    size <= runtime.gpus[index].endpoint.host_visible_local_memory_bytes
                });
                (
                    index,
                    MemoryKind::DeviceLocal {
                        host_visible,
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
            let in_flight = match InFlightGeneration::enter(generation, &runtime.inflight) {
                Ok(in_flight) => in_flight,
                Err(status) => return status,
            };
            (
                runtime.gpus[device_index].device.clone(),
                kind,
                rounded,
                page,
                permissions,
                owner,
                global_flags,
                uncached,
                device_index,
                in_flight,
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
            Err(status) => return retire_unpublished_memory(memory, &in_flight, status),
        };
        if !in_flight.is_current(&guard) {
            drop(guard);
            return retire_unpublished_memory(memory, &in_flight, INVALID_RUNTIME_STATE);
        }
        let Some(runtime) = guard.as_mut() else {
            drop(guard);
            return retire_unpublished_memory(memory, &in_flight, NOT_INITIALIZED);
        };
        if runtime.memory_state.allocations.try_reserve(1).is_err()
            || runtime
                .memory_state
                .allocations
                .contains_key(&pointer_value)
        {
            drop(guard);
            return retire_unpublished_memory(memory, &in_flight, OUT_OF_RESOURCES);
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
    in_flight: &InFlightGeneration,
    status: Status,
) -> Status {
    match memory.free() {
        Ok(()) => status,
        Err(error) => {
            in_flight.quarantine();
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
            if runtime.memory_state.copy_borrowed(base) {
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
                if runtime.memory_state.copy_borrowed(
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
        if runtime
            .memory_state
            .copy_borrowed(memory.description.agent_base)
        {
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::interop::*;
    use super::virtual_memory::*;
    use super::*;

    #[test]
    fn execute_access_permission_matches_the_native_bitmask() {
        assert_eq!(ACCESS_PERMISSION_EX, DeviceAccess::EXECUTE.bits());
        assert_eq!(
            ACCESS_PERMISSION_RW,
            (DeviceAccess::READ | DeviceAccess::WRITE).bits()
        );
        for access in 0..=7 {
            assert!(valid_access_permissions(access));
        }
        for access in [8, 12, u32::MAX] {
            assert!(!valid_access_permissions(access));
        }
    }

    #[test]
    fn completed_memory_workers_are_reaped_without_joining_active_work() {
        let mut state = MemoryState::new();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = stop.clone();
        let active = thread::spawn(move || {
            while !worker_stop.load(Ordering::Acquire) {
                thread::yield_now();
            }
        });
        let active_id = active.thread().id();
        let complete = thread::spawn(|| {});
        while !complete.is_finished() {
            thread::yield_now();
        }
        state.workers.extend([active, complete]);

        state.reap_finished_workers();
        assert_eq!(state.workers.len(), 1);
        assert_eq!(state.workers[0].thread().id(), active_id);

        stop.store(true, Ordering::Release);
        state.join_workers();
        assert!(state.workers.is_empty());
    }

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
    fn registered_range_aligns_native_pages_and_preserves_host_offset() {
        assert_eq!(
            registered_range(0x10_000, 1, 4096),
            Some((0x10_000, 4096, 0))
        );
        assert_eq!(
            registered_range(0x10_345, 1, 4096),
            Some((0x10_000, 4096, 0x345))
        );
        assert_eq!(
            registered_range(0x10_345, 4096, 4096),
            Some((0x10_000, 8192, 0x345))
        );
        assert_eq!(registered_range(usize::MAX, 2, 4096), None);
        assert_eq!(
            registered_range(0x10_123, 1, 65_536),
            Some((0x10_000, 65_536, 0x123))
        );
        assert_eq!(
            registered_range(0x10_123, 65_536, 65_536),
            Some((0x10_000, 131_072, 0x123))
        );
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
    fn virtual_map_rejects_unaligned_addresses_before_runtime_lookup() {
        let page = current_host_page().unwrap();
        assert_eq!(
            hsa_amd_vmem_map(
                (page + 1) as *mut c_void,
                page,
                0,
                HsaAmdVmemAllocHandle { handle: 0 },
                0,
            ),
            INVALID_ARGUMENT
        );
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
        let pools = gpu_agent_pools(index, false, false)
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
        let fine_pools = gpu_agent_pools(index, true, false)
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

        let extended_pools = gpu_agent_pools(index, true, true)
            .into_iter()
            .map(|pool| pool.handle)
            .collect::<Vec<_>>();
        assert_eq!(
            extended_pools,
            [
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_COARSE,
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_EXTENDED,
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_FINE,
                GPU_POOL_BASE + index as u64 * 0x10 + GPU_POOL_GROUP,
            ]
        );

        let regions = gpu_agent_regions(index, true, false)
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
