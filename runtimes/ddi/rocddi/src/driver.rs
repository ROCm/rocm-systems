// SPDX-License-Identifier: MIT

//! Native control used by the implementation-neutral rocddi core.
//!
//! Discovery, activation, allocation, queue control, and host maintenance enter
//! through domain-specific capability traits. Cached metadata queries stay in
//! the core and need no native call. Resource owners carry their concrete cleanup state so a failed
//! destruction can resume without replaying released IDs or requiring a global
//! registry. These are internal implementation boundaries, not a second ABI.
mod builtin;
#[cfg(target_os = "linux")]
pub(crate) mod linux_interop;
use crate::host_storage::Owned;
use crate::kernel_queue::{KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait};
#[cfg(test)]
use crate::memory::MemoryKind;
use crate::memory::{
    AllocationInfo, DeviceAccess, HostAllocationInfo, HostRegistration, OwnedMemoryKind,
    VirtualAddressInfo, VirtualMemoryInfo,
};
use crate::profiling::ClockCounters;
use crate::queue::{QueueRequest, QueueScratch, QueueTransport};
use crate::session::SessionLifetime;
use crate::topology::{Endpoint, GpuPresentation};
use crate::{Error, ErrorKind};
#[cfg(target_os = "linux")]
pub(crate) use builtin::NativeSignalEvent;
pub(crate) use builtin::PlatformDriver;
use std::sync::atomic::{AtomicU64, Ordering};

/// Cached lifecycle facts required for every activated endpoint kind.
pub(crate) trait DeviceStateInfo: Clone {
    fn has_observed_loss(&self) -> bool;
}

/// Address-space facts required only by providers with device memory.
pub(crate) trait AddressSpaceInfo: DeviceStateInfo {
    fn address_range(&self) -> (u64, u64);
    fn shares_address_domain(&self, other: &Self) -> bool;
}

pub(crate) trait HostOwnerInfo {
    fn cached_info(&self) -> HostAllocationInfo;
}

pub(crate) trait AllocationOwnerInfo {
    fn cached_info(&self) -> AllocationInfo;
}

pub(crate) trait VirtualAddressOwnerInfo {
    fn cached_info(&self) -> VirtualAddressInfo;
}

pub(crate) trait VirtualMemoryOwnerInfo {
    fn cached_info(&self) -> VirtualMemoryInfo;
}

pub(crate) trait QueueOwnerInfo {
    fn cached_info(&self) -> QueueTransport;
}

/// Native owner types are split by capability. A CPU or NPU backend can
/// implement discovery and host memory without supplying GPU queue owners.
pub(crate) trait ProviderTypes {
    type DeviceState: DeviceStateInfo;
}

pub(crate) trait HostTypes {
    type HostAllocation: HostOwnerInfo;
}

pub(crate) trait AllocationTypes {
    type Allocation: AllocationOwnerInfo;
}

pub(crate) trait VirtualMemoryTypes {
    type VirtualAddress: VirtualAddressOwnerInfo;
    type VirtualDeviceMapping;
    type VirtualHostMapping;
    type VirtualMemory: VirtualMemoryOwnerInfo;
}

pub(crate) trait QueueTypes {
    type Queue: QueueOwnerInfo;
}

pub(crate) trait KernelQueueTypes {
    type KernelQueue;
}

pub(crate) type DeviceState = <PlatformDriver as ProviderTypes>::DeviceState;
pub(crate) use builtin::EndpointSelector;
pub(crate) type NativeAllocation = <PlatformDriver as AllocationTypes>::Allocation;
pub(crate) type NativeKernelQueue = <PlatformDriver as KernelQueueTypes>::KernelQueue;

pub(crate) trait ProviderDriver: ProviderTypes + Send + Sync {
    fn provider_instance(&self) -> u64;
    fn supports_host_registration(&self, endpoint: &Endpoint, lifetime: SessionLifetime) -> bool;
    fn shutdown(&mut self) -> Result<(), Error>;
    fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error>;
    fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error>;
    fn activate(
        &self,
        endpoint: &Endpoint,
        lifetime: SessionLifetime,
    ) -> Result<Self::DeviceState, Error>;
}

/// Optional GPU display facts; non-GPU providers need no implementation.
pub(crate) trait GpuPresentationDriver: ProviderDriver {
    fn gpu_presentation(&self, endpoint: &Endpoint) -> GpuPresentation;
}

pub(crate) trait HostDriver: HostTypes + Send + Sync {
    fn allocate_host(
        &self,
        size: u64,
        alignment: u64,
    ) -> Result<Owned<Self::HostAllocation>, Error>;
    fn free_host(allocation: &mut Self::HostAllocation) -> Result<(), Error>;
    fn host_page_size() -> Result<u64, Error>;
    fn host_cache_line_size() -> Result<u32, Error>;
    #[allow(unsafe_code)]
    unsafe fn host_cache_control(pointer: usize, length: u64, line_size: u32) -> Result<(), Error>;
}

pub(crate) trait AllocationDriver: ProviderTypes + AllocationTypes + Send + Sync {
    fn check_allocation(allocation: &Self::Allocation) -> Result<(), Error>;
    fn allocation_is_device_local(allocation: &Self::Allocation) -> bool;
    fn set_allocation_access(
        _allocation: &mut Self::Allocation,
        _devices: &[&Self::DeviceState],
    ) -> Result<(), Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "provider does not support changing allocation access",
        })
    }
    fn allocation_device_address(
        allocation: &Self::Allocation,
        device: &Self::DeviceState,
    ) -> Result<u64, Error>;
    fn allocation_is_owned_by(allocation: &Self::Allocation, device: &Self::DeviceState) -> bool;
    fn free_allocation(allocation: &mut Self::Allocation) -> Result<(), Error>;
    fn allocate_owned(
        &self,
        device: &Self::DeviceState,
        peers: &[&Self::DeviceState],
        kind: OwnedMemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self::Allocation>, Error>;
    /// # Safety
    /// The caller keeps the complete host page cover mapped and synchronized
    /// through successful free or, after ambiguous native failure, process exit.
    #[allow(unsafe_code)]
    unsafe fn register_host(
        &self,
        _device: &Self::DeviceState,
        _peers: &[&Self::DeviceState],
        _request: HostRegistration,
    ) -> Result<Owned<Self::Allocation>, Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "provider does not support host registration",
        })
    }
}

/// GPU-only scratch aperture and MMIO page operations.
pub(crate) trait GpuAuxAllocationDriver:
    ProviderTypes + AllocationTypes + Send + Sync
{
    fn allocate_queue_scratch(
        &self,
        device: &Self::DeviceState,
        size: u64,
    ) -> Result<Owned<Self::Allocation>, Error>;
    fn map_mmio_remap(&self, device: &Self::DeviceState) -> Result<Owned<Self::Allocation>, Error>;
}

pub(crate) trait VirtualMemoryDriver:
    ProviderTypes + VirtualMemoryTypes + Send + Sync
{
    fn reserve_virtual_address(
        &self,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<Owned<Self::VirtualAddress>, Error>;
    fn free_virtual_address(address: &mut Self::VirtualAddress) -> Result<(), Error>;
    fn create_virtual_memory(
        &self,
        device: &Self::DeviceState,
        kind: OwnedMemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<Owned<Self::VirtualMemory>, Error>;
    fn free_virtual_memory(memory: &mut Self::VirtualMemory) -> Result<(), Error>;
    /// An error must leave no usable mapping owner. If native submission is
    /// ambiguous, the provider marks the reservation unusable before returning.
    fn map_virtual_device(
        memory: &Self::VirtualMemory,
        reservation: &Self::VirtualAddress,
        device: &Self::DeviceState,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<Owned<Self::VirtualDeviceMapping>, Error>;
    fn free_virtual_device_mapping(mapping: &mut Self::VirtualDeviceMapping) -> Result<(), Error>;
    /// An error must leave the process range unmapped or restored to its
    /// reservation state so the core may release tentative interval occupancy.
    fn map_virtual_host(
        memory: &Self::VirtualMemory,
        reservation: &Self::VirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
        allocator: crate::host_storage::Allocator,
    ) -> Result<Owned<Self::VirtualHostMapping>, Error>;
    fn free_virtual_host_mapping(mapping: &mut Self::VirtualHostMapping) -> Result<(), Error>;
}

pub(crate) trait QueueDriver: ProviderTypes + QueueTypes + Send + Sync {
    fn supports_expert_scheduling(&self, device: &Self::DeviceState) -> Result<bool, Error>;
    fn check_queue(queue: &Self::Queue) -> Result<(), Error>;
    fn queue_progress(queue: &Self::Queue) -> Result<(u64, u64), Error>;
    fn inactivate_queue(queue: &mut Self::Queue) -> Result<(), Error>;
    fn set_queue_priority(
        queue: &mut Self::Queue,
        priority: crate::queue::QueuePriority,
    ) -> Result<(), Error>;
    fn set_queue_cu_mask(queue: &mut Self::Queue, mask: &[u32]) -> Result<(), Error>;
    /// # Safety
    /// Firmware has stopped the queue, and the caller retains the new scratch
    /// backing until a later successful replacement or native destruction.
    #[allow(unsafe_code)]
    unsafe fn set_queue_scratch(
        queue: &mut Self::Queue,
        scratch: QueueScratch,
    ) -> Result<(), Error>;
    /// # Safety
    /// Producers and published transport mappings have been retired.
    #[allow(unsafe_code)]
    unsafe fn destroy_queue(queue: &mut Self::Queue) -> Result<(), Error>;
    /// # Safety
    /// The caller retains all raw signal and scratch addresses that firmware
    /// may reach, including after an ambiguous native creation result.
    #[allow(unsafe_code)]
    unsafe fn create_queue(
        &self,
        device: &Self::DeviceState,
        desc: QueueRequest,
    ) -> Result<Owned<Self::Queue>, Error>;
    fn map_queue(queue: &Self::Queue, device: &Self::DeviceState) -> Result<QueueTransport, Error>;
}

pub(crate) trait KernelQueueDriver: ProviderTypes + KernelQueueTypes + Send + Sync {
    fn create_kernel_queue(
        &self,
        device: &Self::DeviceState,
        format: KernelQueueFormat,
    ) -> Result<Owned<Self::KernelQueue>, Error>;
    /// # Safety
    /// The command range remains device-accessible, executable, and unchanged
    /// until retirement or conclusive native teardown. An ambiguous native
    /// outcome must be returned as an accepted submission; `Err` proves that
    /// the command was rejected.
    #[allow(unsafe_code)]
    unsafe fn submit_kernel_queue(
        queue: &Self::KernelQueue,
        command: KernelCommand,
    ) -> Result<u64, Error>;
    fn kernel_queue_status(queue: &Self::KernelQueue) -> KernelQueueStatus;
    fn wait_kernel_queue(
        queue: &Self::KernelQueue,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error>;
    fn destroy_kernel_queue(queue: &mut Self::KernelQueue) -> Result<(), Error>;
}

/// Endpoint services that do not require a GPU execution engine.
pub(crate) trait DeviceDriver: ProviderTypes + Send + Sync {
    fn check(&self, device: &Self::DeviceState) -> Result<(), Error>;
    fn available_memory(&self, device: &Self::DeviceState) -> Result<u64, Error>;
    fn set_persisting_l2_cache_size(
        &self,
        device: &Self::DeviceState,
        size_bytes: u32,
    ) -> Result<(), Error>;
}

/// GPU-only timing, trap, and stream-monitor services.
pub(crate) trait GpuProfilingDriver: DeviceDriver {
    fn clock_counters(&self, device: &Self::DeviceState) -> Result<ClockCounters, Error>;
    /// # Safety
    /// The handler code and argument storage remain backed and valid while
    /// traps can reach them, including after an ambiguous native update.
    #[allow(unsafe_code)]
    unsafe fn set_trap_handler(
        &self,
        device: &Self::DeviceState,
        handler_address: u64,
        memory_address: u64,
    ) -> Result<(), Error>;
    fn spm_acquire(&self, device: &Self::DeviceState) -> Result<(), Error>;
    fn spm_release(&self, device: &Self::DeviceState) -> Result<(), Error>;
    /// # Safety
    /// The previous and new writable destinations remain live and exclude
    /// conflicting access until replacement, unset, or conclusive teardown.
    #[allow(unsafe_code)]
    unsafe fn spm_set_destination(
        &self,
        device: &Self::DeviceState,
        size: u32,
        timeout: &mut u32,
        bytes_copied: &mut u32,
        destination: Option<usize>,
        data_loss: &mut bool,
    ) -> Result<(), Error>;
}

static NEXT_PROVIDER_INSTANCE: AtomicU64 = AtomicU64::new(1);

pub(crate) fn new_provider_instance() -> u64 {
    NEXT_PROVIDER_INSTANCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .unwrap_or_else(|_| std::process::abort())
}
#[cfg(test)]
#[allow(unsafe_code)]
mod backend_contract_tests {
    use super::*;
    use crate::host_storage::Allocator;
    use crate::memory::{HostCacheability, ProviderAllocation};
    use crate::session::ProviderSession;
    use crate::topology::{EndpointKind, TopologyKey};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;

    const ID: [u8; 16] = [0x42; 16];

    struct FakeDriver {
        instance: u64,
        host_drops: Arc<AtomicUsize>,
        allocation_drops: Arc<AtomicUsize>,
        fail_activation: bool,
        fail_allocation: bool,
        fail_shutdown: bool,
    }

    #[derive(Clone)]
    struct FakeDevice {
        id: [u8; 16],
    }

    struct FakeHost {
        drops: Arc<AtomicUsize>,
        fail_free: bool,
        released: bool,
    }

    impl Drop for FakeHost {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    struct FakeAllocation {
        device_id: [u8; 16],
        drops: Arc<AtomicUsize>,
        fail_free: bool,
        released: bool,
    }

    impl Drop for FakeAllocation {
        fn drop(&mut self) {
            self.drops.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn failure(kind: crate::ErrorKind, detail: &'static str) -> Error {
        Error::Operation { kind, detail }
    }

    impl FakeDriver {
        fn new() -> Self {
            Self {
                instance: new_provider_instance(),
                host_drops: Arc::new(AtomicUsize::new(0)),
                allocation_drops: Arc::new(AtomicUsize::new(0)),
                fail_activation: false,
                fail_allocation: false,
                fail_shutdown: false,
            }
        }

        fn endpoint(&self) -> Endpoint {
            let mut name = [0; 128];
            name[..8].copy_from_slice(b"fake-cpu");
            Endpoint {
                id: ID,
                name,
                pci: None,
                kind: EndpointKind::Cpu,
                local_memory_bytes: 0,
                host_visible_local_memory_bytes: 0,
                host_local_cacheability: None::<HostCacheability>,
                allocation_granularity: 4096,
                address_bit_count: 48,
                minimum_address: 0,
                maximum_address: u64::MAX,
                supported_permissions: DeviceAccess::READ | DeviceAccess::WRITE,
                caches: None,
                memory_links: None,
                topology_key: TopologyKey {
                    group: 0,
                    member: 0,
                },
                provider_instance: self.instance,
                native: EndpointSelector::Opaque(7),
            }
        }
    }

    impl DeviceStateInfo for FakeDevice {
        fn has_observed_loss(&self) -> bool {
            false
        }
    }

    impl HostOwnerInfo for FakeHost {
        fn cached_info(&self) -> HostAllocationInfo {
            HostAllocationInfo {
                host_address: 0,
                size: if self.released { 0 } else { 4096 },
            }
        }
    }

    impl AllocationOwnerInfo for FakeAllocation {
        fn cached_info(&self) -> AllocationInfo {
            AllocationInfo {
                device_address: if self.released { 0 } else { 0x10000 },
                host_address: None,
                size: if self.released { 0 } else { 4096 },
                native_size: if self.released { 0 } else { 4096 },
                physical_backing_id: [0; 2],
            }
        }
    }

    impl ProviderTypes for FakeDriver {
        type DeviceState = FakeDevice;
    }

    impl HostTypes for FakeDriver {
        type HostAllocation = FakeHost;
    }

    impl AllocationTypes for FakeDriver {
        type Allocation = FakeAllocation;
    }

    impl ProviderDriver for FakeDriver {
        fn provider_instance(&self) -> u64 {
            self.instance
        }

        fn supports_host_registration(&self, _: &Endpoint, _: SessionLifetime) -> bool {
            false
        }

        fn shutdown(&mut self) -> Result<(), Error> {
            if self.fail_shutdown {
                self.fail_shutdown = false;
                return Err(failure(
                    crate::ErrorKind::Driver,
                    "injected shutdown failure",
                ));
            }
            Ok(())
        }

        fn enumerate(
            &self,
            visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
        ) -> Result<(), Error> {
            visitor(self.endpoint())
        }

        fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
            if id != ID {
                return Err(failure(
                    crate::ErrorKind::InvalidArgument,
                    "unknown fake endpoint",
                ));
            }
            Ok(self.endpoint())
        }

        fn activate(&self, endpoint: &Endpoint, _: SessionLifetime) -> Result<FakeDevice, Error> {
            if endpoint.provider_instance != self.instance
                || endpoint.id != ID
                || endpoint.native != EndpointSelector::Opaque(7)
                || endpoint.kind != EndpointKind::Cpu
            {
                return Err(failure(
                    crate::ErrorKind::InvalidArgument,
                    "foreign fake endpoint",
                ));
            }
            if self.fail_activation {
                return Err(failure(
                    crate::ErrorKind::Driver,
                    "injected activation failure",
                ));
            }
            Ok(FakeDevice { id: endpoint.id })
        }
    }

    impl DeviceDriver for FakeDriver {
        fn check(&self, device: &FakeDevice) -> Result<(), Error> {
            if device.id == ID {
                Ok(())
            } else {
                Err(failure(
                    crate::ErrorKind::InvalidArgument,
                    "foreign fake device",
                ))
            }
        }

        fn available_memory(&self, device: &FakeDevice) -> Result<u64, Error> {
            self.check(device)?;
            Ok(0)
        }

        fn set_persisting_l2_cache_size(
            &self,
            device: &FakeDevice,
            _size_bytes: u32,
        ) -> Result<(), Error> {
            self.check(device)
        }
    }

    impl HostDriver for FakeDriver {
        fn allocate_host(&self, size: u64, alignment: u64) -> Result<Owned<FakeHost>, Error> {
            if size != 4096 || alignment != 4096 {
                return Err(failure(
                    crate::ErrorKind::InvalidArgument,
                    "fake host extent",
                ));
            }
            Ok(Owned::new(
                FakeHost {
                    drops: self.host_drops.clone(),
                    fail_free: true,
                    released: false,
                },
                Allocator::system(),
            )?)
        }

        fn free_host(allocation: &mut FakeHost) -> Result<(), Error> {
            if allocation.fail_free {
                allocation.fail_free = false;
                return Err(failure(
                    crate::ErrorKind::Driver,
                    "injected host free failure",
                ));
            }
            allocation.released = true;
            Ok(())
        }

        fn host_page_size() -> Result<u64, Error> {
            Ok(4096)
        }

        fn host_cache_line_size() -> Result<u32, Error> {
            Ok(64)
        }

        unsafe fn host_cache_control(_: usize, _: u64, _: u32) -> Result<(), Error> {
            Err(failure(
                crate::ErrorKind::Unsupported,
                "no fake cache control",
            ))
        }
    }

    impl AllocationDriver for FakeDriver {
        fn allocation_is_device_local(_: &FakeAllocation) -> bool {
            false
        }
        fn check_allocation(allocation: &FakeAllocation) -> Result<(), Error> {
            if allocation.released {
                Err(failure(
                    crate::ErrorKind::InvalidArgument,
                    "released fake allocation",
                ))
            } else {
                Ok(())
            }
        }

        fn allocation_device_address(
            allocation: &FakeAllocation,
            device: &FakeDevice,
        ) -> Result<u64, Error> {
            Self::check_allocation(allocation)?;
            if allocation.device_id != device.id {
                return Err(failure(
                    crate::ErrorKind::InvalidArgument,
                    "foreign fake device",
                ));
            }
            Ok(0x10000)
        }

        fn allocation_is_owned_by(allocation: &FakeAllocation, device: &FakeDevice) -> bool {
            allocation.device_id == device.id && !allocation.released
        }

        fn free_allocation(allocation: &mut FakeAllocation) -> Result<(), Error> {
            if allocation.fail_free {
                allocation.fail_free = false;
                return Err(failure(
                    crate::ErrorKind::Driver,
                    "injected allocation free failure",
                ));
            }
            allocation.released = true;
            Ok(())
        }

        fn allocate_owned(
            &self,
            device: &FakeDevice,
            peers: &[&FakeDevice],
            kind: OwnedMemoryKind,
            size: u64,
            alignment: u64,
            permissions: DeviceAccess,
        ) -> Result<Owned<FakeAllocation>, Error> {
            if !peers.is_empty()
                || kind.get() != MemoryKind::System
                || size != 4096
                || alignment != 4096
                || !permissions.contains(DeviceAccess::READ)
            {
                return Err(failure(
                    crate::ErrorKind::InvalidArgument,
                    "fake allocation request",
                ));
            }
            if self.fail_allocation {
                return Err(failure(
                    crate::ErrorKind::Driver,
                    "injected allocation failure",
                ));
            }
            Ok(Owned::new(
                FakeAllocation {
                    device_id: device.id,
                    drops: self.allocation_drops.clone(),
                    fail_free: true,
                    released: false,
                },
                Allocator::system(),
            )?)
        }
    }

    #[test]
    fn non_gpu_provider_runs_through_core_session_lifecycle() -> Result<(), Error> {
        let mut fake = FakeDriver::new();
        let host_drops = fake.host_drops.clone();
        let allocation_drops = fake.allocation_drops.clone();
        fake.fail_shutdown = true;
        let mut session =
            ProviderSession::new(fake, SessionLifetime::Session, Allocator::system())?;
        let mut discovered = Vec::new();
        session.enumerate(&mut |endpoint| {
            discovered.push(endpoint);
            Ok(())
        })?;
        assert_eq!(discovered.len(), 1);
        let endpoint = session.open_endpoint(discovered[0].id)?;
        assert_eq!(endpoint.kind, EndpointKind::Cpu);
        assert!(endpoint.pci.is_none());
        assert!(!session.supports_host_registration(&endpoint));
        assert!(session.activate(&FakeDriver::new().endpoint()).is_err());

        let active = session.activate(&endpoint)?;
        active.driver.check(&active.state)?;
        assert_eq!(active.driver.available_memory(&active.state)?, 0);
        assert!(matches!(
            session.destroy(),
            Err(error) if error.kind() == crate::ErrorKind::Busy
        ));
        let device_state = active.state.clone();
        let active_driver = active.driver.clone();
        drop(active);

        let mut host = session.allocate_host(4096, 4096)?;
        assert_eq!(host.info().size, 4096);
        assert!(host.free().is_err());
        assert_eq!(host_drops.load(Ordering::Relaxed), 0);
        host.free()?;
        drop(host);
        assert_eq!(host_drops.load(Ordering::Relaxed), 1);

        let mut allocation = ProviderAllocation::<FakeDriver>::new(active_driver.allocate_owned(
            &device_state,
            &[],
            OwnedMemoryKind::try_from(MemoryKind::System)?,
            4096,
            4096,
            DeviceAccess::READ,
        )?);
        assert_eq!(allocation.info().device_address, 0x10000);
        allocation.check()?;
        assert_eq!(allocation.device_address(&device_state)?, 0x10000);
        assert!(allocation.originates_from(&device_state));
        assert!(allocation.free().is_err());
        assert_eq!(allocation_drops.load(Ordering::Relaxed), 0);
        allocation.free()?;
        drop(allocation);
        assert_eq!(allocation_drops.load(Ordering::Relaxed), 1);
        drop(active_driver);
        assert!(session.destroy().is_err());
        session.destroy()?;
        Ok(())
    }

    #[test]
    fn non_gpu_backend_contract_covers_discovery_activation_and_failed_cleanup() -> Result<(), Error>
    {
        let mut backend = FakeDriver::new();
        let mut discovered = Vec::new();
        backend.enumerate(&mut |endpoint| {
            discovered.push(endpoint.id);
            Ok(())
        })?;
        assert_eq!(discovered, [ID]);
        let endpoint = backend.open_endpoint(ID)?;
        assert!(endpoint.pci.is_none());
        assert!(endpoint.gpu().is_none());
        assert!(endpoint.linux_kfd_drm_info().is_none());
        assert!(!backend.supports_host_registration(&endpoint, SessionLifetime::Session));

        backend.fail_activation = true;
        assert!(
            backend
                .activate(&endpoint, SessionLifetime::Session)
                .is_err()
        );
        backend.fail_activation = false;
        let device = backend.activate(&endpoint, SessionLifetime::Session)?;
        assert!(!device.has_observed_loss());
        backend.check(&device)?;
        assert_eq!(backend.available_memory(&device)?, 0);

        let mut host = backend.allocate_host(4096, 4096)?;
        assert_eq!(host.cached_info().size, 4096);
        assert!(FakeDriver::free_host(&mut host).is_err());
        assert_eq!(backend.host_drops.load(Ordering::Relaxed), 0);
        FakeDriver::free_host(&mut host)?;
        drop(host);
        assert_eq!(backend.host_drops.load(Ordering::Relaxed), 1);

        backend.fail_allocation = true;
        assert!(
            backend
                .allocate_owned(
                    &device,
                    &[],
                    OwnedMemoryKind::try_from(MemoryKind::System)?,
                    4096,
                    4096,
                    DeviceAccess::READ
                )
                .is_err()
        );
        assert_eq!(backend.allocation_drops.load(Ordering::Relaxed), 0);
        backend.fail_allocation = false;
        let mut allocation = backend.allocate_owned(
            &device,
            &[],
            OwnedMemoryKind::try_from(MemoryKind::System)?,
            4096,
            4096,
            DeviceAccess::READ,
        )?;
        assert_eq!(
            FakeDriver::allocation_device_address(&allocation, &device)?,
            0x10000
        );
        assert_eq!(allocation.cached_info().device_address, 0x10000);
        assert!(FakeDriver::free_allocation(&mut allocation).is_err());
        assert_eq!(backend.allocation_drops.load(Ordering::Relaxed), 0);
        FakeDriver::free_allocation(&mut allocation)?;
        drop(allocation);
        assert_eq!(backend.allocation_drops.load(Ordering::Relaxed), 1);

        backend.fail_shutdown = true;
        assert!(backend.shutdown().is_err());
        backend.shutdown()?;
        Ok(())
    }
}
