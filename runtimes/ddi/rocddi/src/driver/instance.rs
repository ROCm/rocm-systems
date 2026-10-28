// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Installed driver instances and activated-device routing.
//!
//! A session can own several concrete drivers. The common [`Driver`] trait
//! supplies discovery and activated-device lifecycle operations, while the
//! instance and device-state enums retain their concrete types at the session
//! boundary. Resource owners retain the concrete driver needed for cleanup.

#[cfg(test)]
use super::test_driver;
use super::{
    AddressSpaceInfo, AllocationOperations, AllocationState, Driver, GpuDriver,
    GpuInstrumentationOperations, GpuKernelQueueOperations, GpuUserQueueOperations, KfdDriver,
    KfdEventSubscription, linux_kfd,
};
use crate::device::event::DeviceEvent;
use crate::device::gpu::profiling::{
    ClockCounters, PcSamplingConfiguration, PcSamplingConfigurationSet, SpmDestinationFailure,
    SpmDestinationReport,
};
use crate::memory::Allocation;
use crate::memory::storage::Shared;
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, TopologySnapshot};
use crate::{Error, ErrorKind};
use std::sync::atomic::{AtomicU64, Ordering};

/// One driver instance installed in a session.
///
/// Each variant owns a shared reference to a concrete driver implementation.
/// Adding a driver implementation adds a variant here and in [`DeviceDriverState`]
/// without adding GPU capabilities to the common [`Driver`] trait.
pub(crate) enum DriverInstance {
    /// One Linux KFD backend constructed for this session.
    LinuxKfd(Shared<KfdDriver>),
    /// Test backend used to exercise routing without KFD.
    #[cfg(test)]
    Test(Shared<test_driver::TestDriver>),
}

impl DriverInstance {
    /// Borrows the concrete KFD backend for Linux-specific interop.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "other installed drivers have no Linux KFD projection"
    )]
    pub(crate) fn linux_kfd(&self) -> Option<&Shared<KfdDriver>> {
        match self {
            Self::LinuxKfd(driver) => Some(driver),
            #[cfg(test)]
            Self::Test(_) => None,
        }
    }

    /// Returns the process-unique identity assigned to this driver instance.
    pub(crate) fn instance_id(&self) -> DriverInstanceId {
        match self {
            Self::LinuxKfd(driver) => driver.driver_instance(),
            #[cfg(test)]
            Self::Test(driver) => driver.driver_instance(),
        }
    }

    pub(crate) fn context_lifetime(&self) -> DriverContextLifetime {
        match self {
            Self::LinuxKfd(driver) => driver.context_lifetime(),
            #[cfg(test)]
            Self::Test(driver) => driver.context_lifetime(),
        }
    }

    /// Checks whether an allocation holds this exact installed driver.
    pub(crate) fn owns_allocation(&self, allocation: &Allocation) -> bool {
        match (self, &allocation.inner) {
            (Self::LinuxKfd(driver), AllocationState::LinuxKfd(allocation)) => {
                allocation.shares_driver_instance(driver)
            }
            #[cfg(test)]
            (Self::Test(driver), AllocationState::Test(allocation)) => {
                allocation.shares_driver_instance(driver)
            }
            #[cfg(test)]
            _ => false,
        }
    }

    /// Counts only the driver reference held by a recognized internal page.
    pub(crate) fn has_other_owners_except(&self, page: Option<&Allocation>) -> bool {
        let expected = 1 + usize::from(page.is_some_and(|page| self.owns_allocation(page)));
        match self {
            Self::LinuxKfd(driver) => Shared::strong_count(driver) != expected,
            #[cfg(test)]
            Self::Test(driver) => Shared::strong_count(driver) != expected,
        }
    }

    pub(crate) fn shutdown(&mut self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(driver) => Shared::get_mut(driver)
                .ok_or(Error::Operation {
                    kind: ErrorKind::Internal,
                    detail: "driver acquired another owner during shutdown",
                })?
                .shutdown(),
            #[cfg(test)]
            Self::Test(driver) => Shared::get_mut(driver)
                .ok_or(Error::Operation {
                    kind: ErrorKind::Internal,
                    detail: "driver acquired another owner during shutdown",
                })?
                .shutdown(),
        }
    }

    pub(crate) fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(driver) => driver.enumerate(visitor),
            #[cfg(test)]
            Self::Test(driver) => driver.enumerate(visitor),
        }
    }

    pub(crate) fn contribute_topology(&self, snapshot: &mut TopologySnapshot) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(driver) => driver.contribute_topology(snapshot),
            #[cfg(test)]
            Self::Test(driver) => driver.contribute_topology(snapshot),
        }
    }

    pub(crate) fn can_access_local_memory(&self, consumer: &Endpoint, owner: &Endpoint) -> bool {
        if consumer.driver_instance != self.instance_id()
            || owner.driver_instance != self.instance_id()
        {
            return false;
        }
        match self {
            Self::LinuxKfd(driver) => driver.can_access_local_memory(consumer, owner),
            #[cfg(test)]
            Self::Test(driver) => driver.can_access_local_memory(consumer, owner),
        }
    }

    pub(crate) fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        match self {
            Self::LinuxKfd(driver) => driver.open_endpoint(id),
            #[cfg(test)]
            Self::Test(driver) => driver.open_endpoint(id),
        }
    }

    pub(crate) fn probe_endpoint(&self, id: [u8; 16]) -> Result<Option<Endpoint>, Error> {
        match self {
            Self::LinuxKfd(driver) => driver.probe_endpoint(id),
            #[cfg(test)]
            Self::Test(driver) => driver.probe_endpoint(id),
        }
    }

    pub(crate) fn activate(&self, endpoint: &Endpoint) -> Result<DeviceDriverState, Error> {
        match self {
            Self::LinuxKfd(driver) => Ok(DeviceDriverState::LinuxKfd {
                driver: driver.clone(),
                state: driver.activate(endpoint)?,
            }),
            #[cfg(test)]
            Self::Test(driver) => Ok(DeviceDriverState::Test {
                driver: driver.clone(),
                state: driver.activate(endpoint)?,
            }),
        }
    }

    pub(crate) fn populate_endpoint(&self, endpoint: &mut Endpoint) -> Result<(), Error> {
        match self {
            Self::LinuxKfd(driver) => driver.populate_endpoint(endpoint),
            #[cfg(test)]
            Self::Test(driver) => driver.populate_endpoint(endpoint),
        }
    }

    pub(crate) fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        match self {
            Self::LinuxKfd(driver) => driver.supports_host_registration(endpoint),
            #[cfg(test)]
            Self::Test(driver) => driver.supports_host_registration(endpoint),
        }
    }
}

/// Driver state acquired for one endpoint, with its owning driver instance.
///
/// One driver instance can activate several devices. This value retains the
/// specific state for one device and keeps its driver alive until the device
/// and all resources that borrow the driver have been released.
#[derive(Clone)]
pub(crate) enum DeviceDriverState {
    /// Activated KFD GPU state and the driver that owns it.
    LinuxKfd {
        /// Retains the exact driver instance through device cleanup.
        driver: Shared<KfdDriver>,
        /// Native VM and endpoint facts acquired at activation.
        state: KfdDeviceState,
    },
    /// Activated non-GPU fixture state and its owner.
    #[cfg(test)]
    Test {
        /// Retains the fixture driver through resource cleanup.
        driver: Shared<test_driver::TestDriver>,
        /// Fixture device state selected at activation.
        state: test_driver::TestDevice,
    },
}

impl DeviceDriverState {
    /// Uses the retained activation driver to check its cached local-memory
    /// access policy for two devices of that same driver instance.
    pub(crate) fn can_access_local_memory(
        &self,
        other: &Self,
        consumer: &Endpoint,
        owner: &Endpoint,
    ) -> bool {
        if !self.shares_driver_instance(other) {
            return false;
        }
        match self {
            Self::LinuxKfd { driver, .. } => driver.can_access_local_memory(consumer, owner),
            #[cfg(test)]
            Self::Test { driver, .. } => driver.can_access_local_memory(consumer, owner),
        }
    }

    /// Reports whether this driver can register host pages for an endpoint.
    pub(crate) fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        match self {
            Self::LinuxKfd { driver, .. } => driver.supports_host_registration(endpoint),
            #[cfg(test)]
            Self::Test { driver, .. } => driver.supports_host_registration(endpoint),
        }
    }

    /// Checks whether two activated devices retain the same driver instance.
    /// A matching numeric endpoint identity alone cannot establish ownership.
    pub(crate) fn shares_driver_instance(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::LinuxKfd { driver: left, .. }, Self::LinuxKfd { driver: right, .. }) => {
                Shared::ptr_eq(left, right)
            }
            #[cfg(test)]
            (Self::Test { driver: left, .. }, Self::Test { driver: right, .. }) => {
                Shared::ptr_eq(left, right)
            }
            #[cfg(test)]
            _ => false,
        }
    }

    /// Projects the activated state into its GPU capability, when present.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "non-GPU driver variants return no GPU projection"
    )]
    pub(crate) fn gpu_driver(&self) -> Option<GpuDriverRef<'_>> {
        match self {
            Self::LinuxKfd { driver, state } => Some(GpuDriverRef::LinuxKfd { driver, state }),
            #[cfg(test)]
            Self::Test { .. } => None,
        }
    }
    /// Subscribes to the activated driver's retained system notifications.
    #[allow(
        clippy::unnecessary_wraps,
        reason = "other driver variants may not support events"
    )]
    pub(crate) fn subscribe_events(&self) -> Result<EventSubscription, Error> {
        match self {
            Self::LinuxKfd { state, .. } => {
                Ok(EventSubscription::LinuxKfd(state.subscribe_events()?))
            }
            #[cfg(test)]
            Self::Test { .. } => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "driver does not expose system events",
            }),
        }
    }

    /// Checks both driver ownership and native address-space identity.
    pub(crate) fn shares_address_domain(&self, other: &Self) -> bool {
        if !self.shares_driver_instance(other) {
            return false;
        }
        match (self, other) {
            (Self::LinuxKfd { state: a, .. }, Self::LinuxKfd { state: b, .. }) => {
                AddressSpaceInfo::shares_address_domain(a, b)
            }
            #[cfg(test)]
            (Self::Test { state: a, .. }, Self::Test { state: b, .. }) => {
                AddressSpaceInfo::shares_address_domain(a, b)
            }
            #[cfg(test)]
            _ => false,
        }
    }

    /// Returns inclusive device-address bounds cached at activation.
    pub(crate) fn address_range(&self) -> (u64, u64) {
        match self {
            Self::LinuxKfd { state, .. } => AddressSpaceInfo::address_range(state),
            #[cfg(test)]
            Self::Test { state, .. } => AddressSpaceInfo::address_range(state),
        }
    }

    /// Borrows a Linux KFD projection or rejects another driver family.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn linux_kfd(&self) -> Result<(&Shared<KfdDriver>, &KfdDeviceState), Error> {
        match self {
            Self::LinuxKfd { driver, state } => Ok((driver, state)),
            #[cfg(test)]
            Self::Test { .. } => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "device does not use the Linux KFD driver",
            }),
        }
    }

    #[cfg(test)]
    pub(crate) fn test_driver(
        &self,
    ) -> Option<(&Shared<test_driver::TestDriver>, &test_driver::TestDevice)> {
        match self {
            Self::Test { driver, state } => Some((driver, state)),
            Self::LinuxKfd { .. } => None,
        }
    }
}

/// Concrete observer retained behind the device-independent event contract.
pub(crate) enum EventSubscription {
    /// Linux KFD observer retained by the generic device event handle.
    LinuxKfd(KfdEventSubscription),
}

impl EventSubscription {
    /// Reads the next event available to this observer.
    pub(crate) fn poll(&mut self) -> Result<Option<DeviceEvent>, Error> {
        match self {
            Self::LinuxKfd(subscription) => subscription.poll(),
        }
    }
}

/// Driver-specific identity retained by a passive endpoint snapshot.
///
/// The selector contains discovery facts, not a live driver handle. Activation
/// checks the driver instance and reopens the endpoint before using it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum EndpointSelector {
    /// KFD node identity and cached native peer-access evidence.
    LinuxKfd {
        native: linux_kfd::LinuxSelector,
        access_targets: Option<Shared<Vec<KfdAccessTarget>>>,
    },
    /// Driver-private selector for a future non-KFD backend.
    #[allow(dead_code)]
    Opaque(u64),
}

/// KFD node and GPU identity used only for driver-qualified access policy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct KfdAccessTarget {
    pub node: u32,
    pub gpu_id: u32,
}

#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
compile_error!("rocddi currently supports Linux x86-64 and AArch64");

/// Activated device state returned by the concrete KFD driver.
pub(crate) type KfdDeviceState = <KfdDriver as Driver>::DeviceState;
/// Allocation owner returned by the concrete KFD driver.
pub(crate) type KfdAllocation = <KfdDriver as AllocationOperations>::Allocation;

static NEXT_DRIVER_INSTANCE: AtomicU64 = AtomicU64::new(1);

/// Opaque identity of a constructed driver, retained by passive endpoints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct DriverInstanceId(u64);

#[cfg(test)]
impl DriverInstanceId {
    // The generator aborts before returning this reserved test identity.
    pub(crate) const TEST: Self = Self(u64::MAX);
}

/// Allocates an identity for one constructed driver instance in this process.
pub(crate) fn new_driver_instance() -> DriverInstanceId {
    DriverInstanceId(
        NEXT_DRIVER_INSTANCE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .unwrap_or_else(|_| std::process::abort()),
    )
}

/// Borrowed GPU-only driver state established from an activated device.
/// Non-GPU drivers cannot enter this operation surface.
#[derive(Clone, Copy)]
pub(crate) enum GpuDriverRef<'a> {
    LinuxKfd {
        driver: &'a Shared<KfdDriver>,
        state: &'a KfdDeviceState,
    },
}

/// Routes GPU-only operations through a checked GPU driver projection.
impl GpuDriverRef<'_> {
    pub(crate) fn xnack_enabled(&self) -> Result<bool, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.xnack_enabled(state),
        }
    }

    pub(crate) fn supports_expert_scheduling(&self) -> Result<bool, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.supports_expert_scheduling(state),
        }
    }

    pub(crate) fn set_persisting_l2_cache_size(&self, size: u32) -> Result<(), Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.set_persisting_l2_cache_size(state, size),
        }
    }

    pub(crate) fn available_memory(&self) -> Result<u64, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.available_memory(state),
        }
    }

    pub(crate) fn available_sdma_engines(&self) -> Result<u32, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.available_sdma_engines(state),
        }
    }

    pub(crate) fn clock_counters(&self) -> Result<ClockCounters, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.clock_counters(state),
        }
    }

    pub(crate) fn pc_sampling_configurations(
        &self,
        set: PcSamplingConfigurationSet,
    ) -> Result<Vec<PcSamplingConfiguration>, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.pc_sampling_configurations(state, set),
        }
    }

    pub(crate) fn surface_config(&self) -> Result<crate::device::gpu::GpuSurfaceConfig, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.surface_config(state),
        }
    }

    /// # Safety
    /// The caller retains handler code and argument storage while traps can
    /// reach them, including after an ambiguous driver update.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn set_trap_handler(
        &self,
        handler_address: u64,
        memory_address: u64,
    ) -> Result<(), Error> {
        match self {
            // SAFETY: The caller preserves both native-reachable allocations.
            Self::LinuxKfd { driver, state } => unsafe {
                driver.set_trap_handler(state, handler_address, memory_address)
            },
        }
    }

    pub(crate) fn spm_acquire(&self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.spm_acquire(state),
        }
    }

    pub(crate) fn spm_release(&self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.spm_release(state),
        }
    }

    /// # Safety
    /// The caller retains both potentially reachable destinations until a
    /// successful replacement, unset, or conclusive teardown.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn spm_set_destination(
        &self,
        size: u32,
        timeout: u32,
        destination: Option<usize>,
    ) -> Result<SpmDestinationReport, SpmDestinationFailure> {
        match self {
            // SAFETY: The caller preserves previous and new destinations.
            Self::LinuxKfd { driver, state } => unsafe {
                driver.spm_set_destination(state, size, timeout, destination)
            },
        }
    }
}
