// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Concrete resource routing for drivers installed in a session.
//!
//! The public owners and their validation rules live in the memory and queue
//! modules. This private module pairs those owners with the activated driver
//! that created them. Adding a driver variant changes routing here while the
//! generic ownership and rollback algorithms remain shared.

#[cfg(test)]
use super::test_driver::{TestAllocation, TestDriver};
use super::{
    AllocationOperations, DeviceDriverState, GpuDriver, GpuDriverRef, KfdAllocation, KfdDriver,
    KfdKernelQueue, KfdQueue, VirtualMemoryOperations,
};
use crate::Error;
#[cfg(test)]
use crate::ErrorKind;
use crate::device::Device;
use crate::device::gpu::kernel_queue::{
    DriverKernelQueue, KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait,
};
use crate::device::gpu::queue::{
    DriverQueue, QueuePriority, QueueRequest, QueueScratch, QueueTransport,
};
use crate::memory::storage::{Owned, Shared};
use crate::memory::{
    Allocation, AllocationInfo, DeviceAccess, DriverAllocation, DriverVirtualAddress,
    DriverVirtualDeviceMapping, DriverVirtualHostMapping, DriverVirtualMemory, HostRegistration,
    MemoryKind, VirtualAddressInfo, VirtualMapRequest, VirtualMemory, VirtualMemoryInfo,
    set_device_access_for,
};

#[cfg(test)]
fn unsupported_memory<T>() -> Result<T, Error> {
    Err(Error::Operation {
        kind: ErrorKind::Unsupported,
        detail: "activated driver has no matching memory capability",
    })
}

/// Driver-selected owner of a virtual-address reservation.
pub(crate) enum VirtualAddressState {
    LinuxKfd(DriverVirtualAddress<KfdDriver>),
}

impl VirtualAddressState {
    #[cfg(test)]
    pub(crate) fn from_linux_kfd(address: DriverVirtualAddress<KfdDriver>) -> Self {
        Self::LinuxKfd(address)
    }

    #[cfg(test)]
    pub(crate) fn linux_kfd(&self) -> &DriverVirtualAddress<KfdDriver> {
        match self {
            Self::LinuxKfd(address) => address,
        }
    }

    pub(crate) fn info(&self) -> VirtualAddressInfo {
        match self {
            Self::LinuxKfd(address) => address.info(),
        }
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(address) => address.free(),
        }
    }
}

impl DeviceDriverState {
    pub(crate) fn reserve_virtual_address(
        &self,
        bounds: (u64, u64),
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<VirtualAddressState, Error> {
        match self {
            Self::LinuxKfd { driver, .. } => Ok(VirtualAddressState::LinuxKfd(
                DriverVirtualAddress::reserve(driver, bounds, size, alignment, address)?,
            )),
            #[cfg(test)]
            Self::Test { .. } => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "device driver has no virtual-memory capability",
            }),
        }
    }
}

/// Driver-selected owner of detached physical backing.
pub(crate) enum VirtualMemoryState {
    LinuxKfd(DriverVirtualMemory<KfdDriver>),
}

impl VirtualMemoryState {
    pub(crate) fn from_linux_kfd(memory: DriverVirtualMemory<KfdDriver>) -> Self {
        Self::LinuxKfd(memory)
    }

    pub(crate) fn linux_kfd(&self) -> &DriverVirtualMemory<KfdDriver> {
        match self {
            Self::LinuxKfd(memory) => memory,
        }
    }

    pub(crate) fn info(&self) -> VirtualMemoryInfo {
        match self {
            Self::LinuxKfd(memory) => memory.info(),
        }
    }

    pub(crate) fn map_host(
        &self,
        reservation: &VirtualAddressState,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<VirtualHostMappingState, Error> {
        match (self, reservation) {
            (Self::LinuxKfd(memory), VirtualAddressState::LinuxKfd(reservation)) => {
                Ok(VirtualHostMappingState::LinuxKfd(memory.map_host(
                    reservation,
                    address,
                    offset,
                    size,
                    permissions,
                )?))
            }
        }
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(memory) => memory.free(),
        }
    }
}

/// Driver-selected owner of one device virtual-memory mapping.
pub(crate) enum VirtualDeviceMappingState {
    LinuxKfd(DriverVirtualDeviceMapping<KfdDriver>),
}

impl VirtualDeviceMappingState {
    pub(crate) fn free(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(mapping) => mapping.free(),
        }
    }
}

/// Driver-selected owner of one host virtual-memory mapping.
pub(crate) enum VirtualHostMappingState {
    LinuxKfd(DriverVirtualHostMapping<KfdDriver>),
}

impl VirtualHostMappingState {
    pub(crate) fn free(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(mapping) => mapping.free(),
        }
    }
}

/// Driver-selected owner of device-accessible backing.
pub(crate) enum AllocationState {
    LinuxKfd(DriverAllocation<KfdDriver>),
    #[cfg(test)]
    Test(DriverAllocation<TestDriver>),
}

impl AllocationState {
    pub(crate) fn from_linux_kfd(driver: &Shared<KfdDriver>, inner: Owned<KfdAllocation>) -> Self {
        Self::LinuxKfd(DriverAllocation::new(driver.clone(), inner))
    }

    #[cfg(test)]
    pub(crate) fn from_test(driver: &Shared<TestDriver>, inner: Owned<TestAllocation>) -> Self {
        Self::Test(DriverAllocation::new(driver.clone(), inner))
    }

    #[allow(
        clippy::unnecessary_wraps,
        reason = "other allocation drivers require an error"
    )]
    pub(crate) fn linux_kfd(&self) -> Result<&DriverAllocation<KfdDriver>, Error> {
        match self {
            Self::LinuxKfd(allocation) => Ok(allocation),
            #[cfg(test)]
            Self::Test(_) => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "allocation does not belong to Linux KFD",
            }),
        }
    }

    #[allow(
        clippy::unnecessary_wraps,
        reason = "other allocation drivers require an error"
    )]
    pub(crate) fn linux_kfd_mut(&mut self) -> Result<&mut DriverAllocation<KfdDriver>, Error> {
        match self {
            Self::LinuxKfd(allocation) => Ok(allocation),
            #[cfg(test)]
            Self::Test(_) => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "allocation does not belong to Linux KFD",
            }),
        }
    }

    pub(crate) fn info(&self) -> AllocationInfo {
        match self {
            Self::LinuxKfd(allocation) => allocation.info(),
            #[cfg(test)]
            Self::Test(allocation) => allocation.info(),
        }
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(allocation) => allocation.check(),
            #[cfg(test)]
            Self::Test(allocation) => allocation.check(),
        }
    }

    pub(crate) fn device_address(&self, device: &DeviceDriverState) -> Result<u64, Error> {
        match (self, device) {
            (Self::LinuxKfd(allocation), DeviceDriverState::LinuxKfd { state, .. }) => {
                allocation.device_address(state)
            }
            #[cfg(test)]
            (Self::Test(allocation), DeviceDriverState::Test { state, .. }) => {
                allocation.device_address(state)
            }
            #[cfg(test)]
            _ => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "device driver cannot address this allocation",
            }),
        }
    }

    pub(crate) fn set_device_access(
        &mut self,
        driver_instance: u64,
        origin: &Device,
        devices: &[&Device],
    ) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(allocation) => set_device_access_for::<KfdDriver, _>(
                allocation,
                driver_instance,
                origin,
                devices,
                |device| device.driver_state.linux_kfd().ok().map(|(_, state)| state),
            ),
            #[cfg(test)]
            Self::Test(_) => unsupported_memory(),
        }
    }

    pub(crate) fn originates_from(&self, device: &DeviceDriverState) -> bool {
        match (self, device) {
            (Self::LinuxKfd(allocation), DeviceDriverState::LinuxKfd { state, .. }) => {
                allocation.originates_from(state)
            }
            #[cfg(test)]
            (Self::Test(allocation), DeviceDriverState::Test { state, .. }) => {
                allocation.originates_from(state)
            }
            #[cfg(test)]
            _ => false,
        }
    }

    pub(crate) fn metadata(&self) -> &[u8] {
        match self {
            Self::LinuxKfd(allocation) => allocation.driver_state().metadata(),
            #[cfg(test)]
            Self::Test(_) => &[],
        }
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(allocation) => allocation.free(),
            #[cfg(test)]
            Self::Test(allocation) => allocation.free(),
        }
    }
}

impl DeviceDriverState {
    pub(crate) fn allocate_owned(
        &self,
        driver_instance: u64,
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        match self {
            Self::LinuxKfd { driver, state } => Ok(Allocation::new(
                AllocationState::from_linux_kfd(
                    driver,
                    driver.allocate_owned(state, &[], kind, size, alignment, permissions)?,
                ),
                driver_instance,
            )),
            #[cfg(test)]
            Self::Test { .. } => unsupported_memory(),
        }
    }

    /// # Safety
    /// The caller retains the complete registered page cover until release.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn register_host(
        &self,
        driver_instance: u64,
        request: HostRegistration,
    ) -> Result<Allocation, Error> {
        match self {
            Self::LinuxKfd { driver, state } => {
                // SAFETY: The caller preserves the page cover across native use.
                let inner = unsafe { driver.register_host(state, &[], request)? };
                Ok(Allocation::new(
                    AllocationState::from_linux_kfd(driver, inner),
                    driver_instance,
                ))
            }
            #[cfg(test)]
            Self::Test { driver, state } => {
                // SAFETY: The caller preserves the complete page cover.
                let inner = unsafe { driver.register_host(state, &[], request)? };
                Ok(Allocation::new(
                    AllocationState::from_test(driver, inner),
                    driver_instance,
                ))
            }
        }
    }

    pub(crate) fn create_virtual_memory(
        &self,
        kind: MemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<VirtualMemory, Error> {
        match self {
            Self::LinuxKfd { driver, state } => {
                let owner = Shared::try_new_uninit()?;
                let inner = driver.create_virtual_memory(state, kind, size, pinned, uncached)?;
                Ok(VirtualMemory::new(VirtualMemoryState::from_linux_kfd(
                    DriverVirtualMemory::new(driver.clone(), owner.write(inner)),
                )))
            }
            #[cfg(test)]
            Self::Test { .. } => unsupported_memory(),
        }
    }

    pub(crate) fn map_virtual_memory(
        &self,
        memory: &VirtualMemoryState,
        reservation: &VirtualAddressState,
        request: VirtualMapRequest,
    ) -> Result<VirtualDeviceMappingState, Error> {
        match (self, memory, reservation) {
            (
                Self::LinuxKfd { driver, state },
                VirtualMemoryState::LinuxKfd(memory),
                VirtualAddressState::LinuxKfd(reservation),
            ) => Ok(VirtualDeviceMappingState::LinuxKfd(memory.map_device(
                driver,
                state,
                reservation,
                request,
            )?)),
            #[cfg(test)]
            _ => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "device driver cannot map this virtual memory",
            }),
        }
    }

    pub(crate) fn allocate_with_peers(
        &self,
        origin: &Device,
        peers: &[&Device],
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        match self {
            Self::LinuxKfd { driver, state } => {
                let states = origin.peer_states_for(
                    peers,
                    matches!(kind, MemoryKind::DeviceLocal { .. }),
                    driver,
                    state,
                    |peer| peer.driver_state.linux_kfd().ok(),
                )?;
                let inner = driver.allocate_owned(
                    state,
                    states.as_slice(),
                    kind,
                    size,
                    alignment,
                    permissions,
                )?;
                Ok(Allocation::new(
                    AllocationState::from_linux_kfd(driver, inner),
                    origin.endpoint.driver_instance,
                ))
            }
            #[cfg(test)]
            Self::Test { .. } => unsupported_memory(),
        }
    }

    /// # Safety
    /// The caller retains the complete registered page cover until release.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn register_host_with_peers(
        &self,
        origin: &Device,
        peers: &[&Device],
        request: HostRegistration,
    ) -> Result<Allocation, Error> {
        match self {
            Self::LinuxKfd { driver, state } => {
                let states = origin.peer_states_for(peers, false, driver, state, |peer| {
                    peer.driver_state.linux_kfd().ok()
                })?;
                // SAFETY: The caller preserves the page cover across native use.
                let inner = unsafe { driver.register_host(state, states.as_slice(), request)? };
                Ok(Allocation::new(
                    AllocationState::from_linux_kfd(driver, inner),
                    origin.endpoint.driver_instance,
                ))
            }
            #[cfg(test)]
            Self::Test { driver, state } => {
                let states = origin.peer_states_for(peers, false, driver, state, |peer| {
                    peer.driver_state.test_driver()
                })?;
                // SAFETY: The caller preserves the complete page cover.
                let inner = unsafe { driver.register_host(state, states.as_slice(), request)? };
                Ok(Allocation::new(
                    AllocationState::from_test(driver, inner),
                    origin.endpoint.driver_instance,
                ))
            }
        }
    }
}

impl GpuDriverRef<'_> {
    pub(crate) fn allocate_queue_scratch(
        &self,
        driver_instance: u64,
        size: u64,
    ) -> Result<Allocation, Error> {
        match self {
            Self::LinuxKfd { driver, state } => Ok(Allocation::new(
                AllocationState::from_linux_kfd(
                    driver,
                    driver.allocate_queue_scratch(state, size)?,
                ),
                driver_instance,
            )),
        }
    }

    pub(crate) fn map_mmio_remap(&self, driver_instance: u64) -> Result<Allocation, Error> {
        match self {
            Self::LinuxKfd { driver, state } => Ok(Allocation::new(
                AllocationState::from_linux_kfd(driver, driver.map_mmio_remap(state)?),
                driver_instance,
            )),
        }
    }
}

/// Driver-selected owner of a user-mode GPU queue.
pub(crate) enum QueueState {
    LinuxKfd(DriverQueue<KfdDriver, KfdQueue>),
}

impl QueueState {
    pub(crate) fn info(&self) -> QueueTransport {
        match self {
            Self::LinuxKfd(queue) => queue.info(),
        }
    }

    pub(crate) fn map_device(&self, device: &Device) -> Result<QueueTransport, Error> {
        match (self, &device.driver_state) {
            (Self::LinuxKfd(queue), DeviceDriverState::LinuxKfd { driver, state }) => {
                queue.map_device(driver, state)
            }
            #[cfg(test)]
            _ => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "device driver cannot map this GPU queue",
            }),
        }
    }

    pub(crate) fn progress(&self) -> Result<(u64, u64), Error> {
        match self {
            Self::LinuxKfd(queue) => queue.progress(),
        }
    }

    pub(crate) fn inactivate(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(queue) => queue.inactivate(),
        }
    }

    pub(crate) fn set_priority(&mut self, priority: QueuePriority) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(queue) => queue.set_priority(priority),
        }
    }

    pub(crate) fn set_cu_mask(&mut self, mask: &[u32]) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(queue) => queue.set_cu_mask(mask),
        }
    }

    /// # Safety
    /// The command processor has stopped the queue and scratch remains live.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn set_scratch(&mut self, scratch: QueueScratch) -> Result<(), Error> {
        match self {
            // SAFETY: The caller preserves the stopped queue and backing.
            Self::LinuxKfd(queue) => unsafe { queue.set_scratch(scratch) },
        }
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(queue) => queue.check(),
        }
    }

    /// # Safety
    /// The caller has retired producers and transport mappings.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn destroy(&mut self) -> Result<(), Error> {
        match self {
            // SAFETY: The caller has retired all queue reachability.
            Self::LinuxKfd(queue) => unsafe { queue.destroy() },
        }
    }
}

/// Driver-selected owner of a kernel-mediated GPU queue.
pub(crate) enum KernelQueueState {
    LinuxKfd(DriverKernelQueue<KfdDriver, KfdKernelQueue>),
}

impl KernelQueueState {
    /// # Safety
    /// Command storage remains reachable until native retirement is proved.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        match self {
            // SAFETY: The caller preserves the command backing.
            Self::LinuxKfd(queue) => unsafe { queue.submit(command) },
        }
    }

    pub(crate) fn status(&self) -> KernelQueueStatus {
        match self {
            Self::LinuxKfd(queue) => queue.status(),
        }
    }

    pub(crate) fn refresh_status(&self) -> Result<KernelQueueStatus, Error> {
        match self {
            Self::LinuxKfd(queue) => queue.refresh_status(),
        }
    }

    pub(crate) fn wait(
        &self,
        submission: u64,
        timeout_nanoseconds: u64,
        poll_duration_nanoseconds: u64,
    ) -> Result<KernelQueueWait, Error> {
        match self {
            Self::LinuxKfd(queue) => {
                queue.wait(submission, timeout_nanoseconds, poll_duration_nanoseconds)
            }
        }
    }

    pub(crate) fn destroy(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(queue) => queue.destroy(),
        }
    }
}

impl GpuDriverRef<'_> {
    /// # Safety
    /// Every GPU address in the request remains backed through queue teardown.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn create_queue(&self, desc: QueueRequest) -> Result<QueueState, Error> {
        match self {
            Self::LinuxKfd { driver, state } => {
                // SAFETY: The caller preserves every request address.
                let inner = unsafe { driver.create_user_queue(state, desc) }?;
                Ok(QueueState::LinuxKfd(DriverQueue::new(
                    (*driver).clone(),
                    inner,
                )))
            }
        }
    }

    pub(crate) fn create_kernel_queue(
        &self,
        format: KernelQueueFormat,
    ) -> Result<KernelQueueState, Error> {
        match self {
            Self::LinuxKfd { driver, state } => {
                let inner = driver.create_kernel_queue(state, format)?;
                Ok(KernelQueueState::LinuxKfd(DriverKernelQueue::new(
                    (*driver).clone(),
                    inner,
                )))
            }
        }
    }
}
