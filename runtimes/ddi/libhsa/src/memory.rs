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
//! as portable native-handle behavior. Child modules own pool projections,
//! allocations, pointer metadata, IPC and interop, virtual memory, and
//! copy/SVM operations. This root retains shared state and shutdown order.

pub(crate) mod copy_svm;
pub(crate) mod interop;
pub(crate) use crate::api::memory::copy_svm::hsa_memory_copy;
pub(crate) use copy_svm::image_device_address;
use copy_svm::*;
pub(crate) mod virtual_memory;

use crate::platform::fd::{AsFd, IntoRawFd};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::mem::{align_of, size_of};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering, fence};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

use crate::platform::memory as linux_interop;
use crate::platform::memory::AisFileOperation as AisOperation;
use crate::platform::memory::{KfdSvmAttribute as SvmAttribute, KfdSvmLocation as SvmLocation};
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
use crate::runtime::{HsaState, InFlightGeneration, boundary, initialized_mut, lock, map_error};
use crate::signal::{AsyncCopyClock, AsyncCopyProfile, SignalState};

/// Handle slots reserved for each agent's memory pools.
pub(crate) const POOL_HANDLE_STRIDE: u64 = 0x10;

/// CPU memory-pool class encoded in an HSA pool handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u64)]
pub(crate) enum CpuPoolKind {
    /// Fine-grained host memory.
    Fine = 1,
    /// Extended fine-grained host memory.
    ExtendedFine = 2,
    /// Uncached kernel-argument memory.
    Kernarg = 3,
    /// Coarse-grained host memory.
    Coarse = 4,
}

impl CpuPoolKind {
    /// CPU pools in the order exposed to HSA clients.
    pub(crate) const ALL: [Self; 4] = [Self::Fine, Self::ExtendedFine, Self::Kernarg, Self::Coarse];

    /// Decodes the kind slot within one CPU agent's handle range.
    pub(crate) fn from_slot(slot: u64) -> Option<Self> {
        match slot {
            1 => Some(Self::Fine),
            2 => Some(Self::ExtendedFine),
            3 => Some(Self::Kernarg),
            4 => Some(Self::Coarse),
            _ => None,
        }
    }
}

/// GPU memory-pool class encoded in an HSA pool handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u64)]
pub(crate) enum GpuPoolKind {
    /// Coarse-grained local memory.
    Coarse = 1,
    /// Fine-grained local memory.
    Fine = 2,
    /// Per-workgroup memory.
    Group = 3,
    /// Extended fine-grained local memory.
    ExtendedFine = 4,
}

impl GpuPoolKind {
    /// Decodes the kind slot within one GPU agent's handle range.
    pub(crate) fn from_slot(slot: u64) -> Option<Self> {
        match slot {
            1 => Some(Self::Coarse),
            2 => Some(Self::Fine),
            3 => Some(Self::Group),
            4 => Some(Self::ExtendedFine),
            _ => None,
        }
    }
}

/// Native allocation flag requesting PCIe-visible memory.
const ALLOC_PCIE: u32 = 1;
/// Native allocation flag requesting contiguous memory.
const ALLOC_CONTIGUOUS: u32 = 1 << 1;
/// Native allocation flag permitting code execution.
const ALLOC_EXECUTABLE: u32 = 1 << 2;
/// Native allocation flag requesting uncached memory.
pub(crate) const ALLOC_UNCACHED: u32 = 1 << 3;
/// Native allocation flags accepted by this frontend.
const ALLOC_FLAGS: u32 = ALLOC_PCIE | ALLOC_CONTIGUOUS | ALLOC_EXECUTABLE | ALLOC_UNCACHED;
/// SVM policy bit allowing host access.
pub(crate) const SVM_FLAG_HOST_ACCESS: u32 = 0x01;
/// SVM policy bit requesting coherent access.
pub(crate) const SVM_FLAG_COHERENT: u32 = 0x02;
/// SVM policy bit restricting placement to the local hive.
pub(crate) const SVM_FLAG_HIVE_LOCAL: u32 = 0x04;
/// SVM policy bit making GPU access read-only.
pub(crate) const SVM_FLAG_GPU_READ_ONLY: u32 = 0x08;
/// SVM policy bit allowing GPU execution.
pub(crate) const SVM_FLAG_GPU_EXECUTE: u32 = 0x10;
/// SVM policy bit marking a range read-mostly for GPUs.
pub(crate) const SVM_FLAG_GPU_READ_MOSTLY: u32 = 0x20;
/// Largest accepted SVM migration-granularity encoding.
pub(crate) const SVM_MAX_MIGRATION_GRANULARITY: u64 = 18;
/// Maximum number of SVM attributes accepted in one request.
pub(crate) const SVM_MAX_ATTRIBUTES: usize = 2044;
/// Linux I/O error number used when native cleanup cannot be proved.
const EIO: i32 = 5;

/// Accepts only the HSA virtual-memory access bits implemented here.
pub(crate) fn valid_access_permissions(access: u32) -> bool {
    access & !(ACCESS_PERMISSION_RW | ACCESS_PERMISSION_EX) == 0
}

/// Process setting captured when an asynchronous copy is accepted.
#[derive(Clone, Copy)]
pub(crate) enum AsyncCopyProfiling {
    /// Async copies complete without retaining profiling timestamps.
    Disabled,
    /// Async copies retain timestamps for later profiling queries.
    Enabled,
}

/// HSA memory handles and metadata around rocddi allocation owners.
pub(crate) struct MemoryState {
    /// Ordinary HSA allocations keyed by host address.
    pub(crate) allocations: HashMap<usize, Memory>,
    /// Imported IPC memory keyed by host address.
    pub(crate) ipc_allocations: HashMap<usize, Memory>,
    /// External memory imports keyed by host address.
    pub(crate) interop_allocations: HashMap<usize, Memory>,
    /// Host ranges locked for GPU access.
    pub(crate) locked_allocations: Vec<LockedMemory>,
    /// Outstanding asynchronous copy borrows per allocation address.
    async_copy_borrows: HashMap<usize, usize>,
    /// Prevents unsafe reuse after uncertain asynchronous copy retirement.
    pub(crate) health: Arc<crate::runtime::GenerationHealth>,
    /// Process setting captured by new asynchronous copies.
    pub(crate) async_copy_profiling: AsyncCopyProfiling,
    /// Virtual address reservations ordered by base address.
    pub(crate) vmem_reservations: BTreeMap<usize, VmemReservation>,
    /// Physical virtual-memory handles indexed by public ID.
    pub(crate) vmem_handles: HashMap<u64, VmemHandle>,
    /// Active virtual-memory mappings ordered by base address.
    pub(crate) vmem_mappings: BTreeMap<usize, VmemMapping>,
    /// Prefetch and copy workers joined before memory owners are released.
    pub(crate) workers: Vec<thread::JoinHandle<()>>,
}

impl MemoryState {
    /// Initializes the allocation, registration, and virtual memory registries.
    pub(crate) fn new(health: Arc<crate::runtime::GenerationHealth>) -> Self {
        Self {
            allocations: HashMap::new(),
            ipc_allocations: HashMap::new(),
            interop_allocations: HashMap::new(),
            locked_allocations: Vec::new(),
            async_copy_borrows: HashMap::new(),
            health,
            async_copy_profiling: AsyncCopyProfiling::Disabled,
            vmem_reservations: BTreeMap::new(),
            vmem_handles: HashMap::new(),
            vmem_mappings: BTreeMap::new(),
            workers: Vec::new(),
        }
    }

    /// Checks whether the current thread is an asynchronous memory worker.
    pub(crate) fn running_on_worker(&self) -> bool {
        self.workers
            .iter()
            .any(|worker| worker.thread().id() == thread::current().id())
    }

    /// Join completed workers before accepting another asynchronous operation.
    /// `is_finished` ensures joining cannot wait for work that needs the runtime
    /// lock held by the submitter.
    pub(crate) fn reap_finished_workers(&mut self) {
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
        for worker in self.take_workers() {
            let _ = worker.join();
        }
    }

    /// Detaches workers so shutdown can join them outside the lifecycle lock.
    pub(crate) fn take_workers(&mut self) -> Vec<JoinHandle<()>> {
        std::mem::take(&mut self.workers)
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
pub(crate) struct PointerDescription {
    /// HSA pointer classification reported to callers.
    pub(crate) pointer_type: u32,
    /// Address visible to the owning agent.
    pub(crate) agent_base: usize,
    /// CPU alias when this allocation is host accessible.
    pub(crate) host_base: Option<usize>,
    /// Byte extent common to the address aliases.
    pub(crate) size: usize,
    /// HSA agent that owns the allocation.
    pub(crate) owner: HsaAgent,
    /// Pool allocation flags exposed by pointer queries.
    global_flags: u32,
    /// Whether this address came from host registration.
    registered: bool,
    /// Native allocation flags retained for compatibility queries.
    alloc_flags: u32,
    /// Caller-set pointer metadata.
    user_data: usize,
    /// Agents with established access to the allocation.
    pub(crate) accessible: Vec<HsaAgent>,
}

impl PointerDescription {
    /// Checks whether the requested address belongs to this object.
    fn contains(&self, address: usize) -> bool {
        contains(self.agent_base, self.size, address)
            || self
                .host_base
                .is_some_and(|host| contains(host, self.size, address))
    }

    /// Checks whether this allocation starts at the given address.
    fn starts_at(&self, address: usize) -> bool {
        self.agent_base == address || self.host_base == Some(address)
    }

    /// Translates a device address to its host mapping when available.
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

    /// Returns a host mapping for a complete device address range.
    pub(crate) fn host_range(&self, address: usize, size: usize) -> Option<usize> {
        self.offset_range(address, size)?;
        self.host_address(address)
    }

    /// Returns the offset of a complete range within this allocation.
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
    pub(crate) allocation: Allocation,
    /// Canonical metadata for every public address alias.
    pub(crate) description: PointerDescription,
    /// Pool flags fixed when this allocation was created.
    pub(crate) pool_global_flags: u32,
    /// Callbacks invoked after public ownership is removed.
    pub(crate) deallocation_callbacks: Vec<RegisteredDeallocationCallback>,
}

/// Application callback invoked after an allocation loses public ownership.
#[derive(Clone, Copy)]
pub(crate) struct RegisteredDeallocationCallback {
    /// Public allocation address supplied when the callback was registered.
    pub(crate) pointer: usize,
    /// Application callback invoked after public ownership ends.
    pub(crate) callback: unsafe extern "C" fn(*mut c_void, *mut c_void),
    /// Caller-owned argument passed back to that callback.
    pub(crate) user_data: usize,
}

impl Memory {
    /// Records an allocation’s native backing and HSA visibility.
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

    /// Checks whether the requested address belongs to this object.
    pub(crate) fn contains(&self, address: usize) -> bool {
        self.description.contains(address)
    }

    /// Wraps an imported graphics allocation and its accessible agents.
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

    /// Wraps an attached IPC allocation and its accessible agents.
    pub(crate) fn new_ipc(
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

    /// Releases this object’s native backing.
    pub(crate) fn free(&mut self) -> Result<(), rocddi::Error> {
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
    pub(crate) description: PointerDescription,
}

/// Reserved virtual-address interval and the mappings currently occupying it.
pub(crate) struct VmemReservation {
    /// Native reserved address range.
    pub(crate) address: VirtualAddress,
    /// Public pointer metadata for the reservation.
    pub(crate) description: PointerDescription,
    /// Callbacks invoked when public ownership is removed.
    pub(crate) deallocation_callbacks: Vec<RegisteredDeallocationCallback>,
    /// Mappings that still occupy this reserved range.
    pub(crate) mappings: usize,
}

impl VmemReservation {
    /// Records a reserved virtual range and its registration state.
    pub(crate) fn new(address: VirtualAddress, registered: bool) -> Self {
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

    /// Checks whether the requested address belongs to this object.
    pub(crate) fn contains_range(&self, address: usize, size: usize) -> bool {
        address
            .checked_sub(self.address.info().address as usize)
            .and_then(|offset| offset.checked_add(size))
            .is_some_and(|end| size != 0 && end <= self.description.size)
    }

    /// Releases this object’s native backing.
    pub(crate) fn free(&mut self) -> Result<(), rocddi::Error> {
        self.address.free()
    }
}

/// Retained physical-memory handle shared by public references and mappings.
pub(crate) struct VmemHandle {
    /// Native physical backing or imported fabric handle.
    pub(crate) backing: VmemBacking,
    /// Number of public references to this handle.
    pub(crate) references: usize,
    /// Number of active virtual-address mappings.
    pub(crate) mappings: usize,
    /// HSA pool from which the backing originates.
    pub(crate) pool: HsaMemoryPool,
    /// HSA physical-memory type exposed for this handle.
    pub(crate) memory_type: u32,
}

/// Origin of physical backing behind an HSA virtual-memory handle.
pub(crate) enum VmemBacking {
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
    /// Records virtual memory backing, pool, and memory type.
    pub(crate) fn new(backing: VmemBacking, pool: HsaMemoryPool, memory_type: u32) -> Self {
        Self {
            backing,
            references: 1,
            mappings: 0,
            pool,
            memory_type,
        }
    }

    /// Checks whether the virtual memory handle came from an import.
    pub(crate) fn imported(&self) -> bool {
        !matches!(&self.backing, VmemBacking::Owned(_))
    }

    /// Returns native backing when this frontend owns the allocation.
    pub(crate) fn owned_memory(&self) -> Option<&VirtualMemory> {
        match &self.backing {
            VmemBacking::Owned(memory) => Some(memory),
            VmemBacking::DmaBuf(_) | VmemBacking::Fabric { .. } => None,
        }
    }

    /// Returns the known allocation size, if native metadata provides it.
    pub(crate) fn known_size(&self) -> Option<u64> {
        match &self.backing {
            VmemBacking::Owned(memory) | VmemBacking::DmaBuf(memory) => Some(memory.info().size),
            VmemBacking::Fabric { .. } => None,
        }
    }

    /// Returns backing memory after validating the requested access.
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

    /// Releases this object’s native backing.
    pub(crate) fn free(&mut self) -> Result<(), rocddi::Error> {
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

/// Native device mapping owner; CPU accesses share the host map on `VmemMapping`.
pub(crate) enum VmemAccessOwner {
    /// Native mapping in an activated device VM.
    Device(VirtualDeviceMapping),
    /// Permission record for the process-wide host mapping.
    Host,
}

/// Per-agent permissions and the native object that enforces them.
pub(crate) struct VmemAccess {
    /// HSA permission mask granted to this agent.
    pub(crate) permissions: u32,
    /// Native mapping retained until this access is removed.
    pub(crate) owner: VmemAccessOwner,
}

impl VmemAccess {
    /// Releases this object’s native backing.
    fn free(&mut self) -> Result<(), rocddi::Error> {
        match &mut self.owner {
            VmemAccessOwner::Device(mapping) => mapping.free(),
            VmemAccessOwner::Host => Ok(()),
        }
    }
}

/// One mapped virtual subrange and all native per-agent access mappings.
pub(crate) struct VmemMapping {
    /// Public physical-memory handle supplying backing.
    pub(crate) handle: u64,
    /// Base address of the parent reservation.
    pub(crate) reservation: usize,
    /// First mapped virtual byte.
    pub(crate) address: usize,
    /// Byte offset into physical backing.
    pub(crate) offset: usize,
    /// Mapped byte extent.
    pub(crate) size: usize,
    /// Per-agent permissions and native map owners.
    pub(crate) access: HashMap<u64, VmemAccess>,
    /// One process-wide host map shared by all CPU-agent access records.
    pub(crate) host_mapping: Option<VirtualHostMapping>,
}

impl VmemMapping {
    /// Returns the process-wide CPU permission union after one agent changes.
    fn host_permissions_after(&self, agent: u64, permissions: u32) -> u32 {
        self.access
            .iter()
            .filter(|(other, access)| {
                **other != agent && matches!(access.owner, VmemAccessOwner::Host)
            })
            .fold(permissions, |bits, (_, access)| bits | access.permissions)
    }

    /// Returns the currently granted process-wide CPU permission union.
    fn host_permissions(&self) -> u32 {
        self.access
            .values()
            .filter(|access| matches!(access.owner, VmemAccessOwner::Host))
            .fold(ACCESS_PERMISSION_NONE, |bits, access| {
                bits | access.permissions
            })
    }

    /// Checks whether the requested address belongs to this object.
    pub(crate) fn contains(&self, address: usize) -> bool {
        contains(self.address, self.size, address)
    }

    /// Releases access.
    pub(crate) fn free_access(&mut self) -> Result<(), rocddi::Error> {
        let agents = self
            .access
            .iter()
            .filter(|(_, access)| matches!(access.owner, VmemAccessOwner::Device(_)))
            .map(|(agent, _)| *agent)
            .collect::<Vec<_>>();
        for agent in agents {
            let Some(mut access) = self.access.remove(&agent) else {
                continue;
            };
            if let Err(error) = access.free() {
                self.access.insert(agent, access);
                return Err(error);
            }
        }
        if let Some(mapping) = &mut self.host_mapping {
            mapping.free()?;
            self.host_mapping = None;
        }
        self.access
            .retain(|_, access| !matches!(access.owner, VmemAccessOwner::Host));
        Ok(())
    }
}

impl LockedMemory {
    /// Records pinned host storage and its accessible agents.
    fn new(
        allocation: Option<Allocation>,
        host_base: usize,
        device_base: usize,
        size: usize,
        owner: HsaAgent,
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
                owner,
                global_flags,
                registered: true,
                alloc_flags: POINTER_ALLOC_NONPAGED | POINTER_ALLOC_HOST_ACCESS,
                user_data: 0,
                accessible,
            },
        }
    }

    /// Checks whether the requested address belongs to this object.
    fn contains(&self, address: usize) -> bool {
        self.description.contains(address)
    }

    /// Releases this object’s native backing.
    pub(crate) fn free(&mut self) -> Result<(), rocddi::Error> {
        self.allocation.as_mut().map_or(Ok(()), Allocation::free)
    }
}

/// Checks whether the requested address belongs to this object.
fn contains(base: usize, size: usize, address: usize) -> bool {
    address
        .checked_sub(base)
        .is_some_and(|offset| offset < size)
}

/// Returns the current host page.
pub(crate) fn current_host_page() -> Result<usize, Status> {
    let page = usize::try_from(rocddi::memory::host::host_page_size().map_err(map_error)?)
        .map_err(|_| ERROR)?;
    if page < 4096 || !page.is_power_of_two() {
        return Err(ERROR);
    }
    Ok(page)
}

pub(crate) mod allocation;
pub(crate) mod pointer;
pub(crate) mod pool;

#[cfg(test)]
use crate::api::memory::copy_svm::async_copy::*;
#[cfg(test)]
use crate::api::memory::{allocation::*, copy_svm::*, interop::*, pointer::*, virtual_memory::*};
use pointer::*;
use pool::*;
pub(crate) use pool::{SignalBatchPool, signal_batch_pool};

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
