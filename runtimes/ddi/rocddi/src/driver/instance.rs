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
    AddressSpaceInfo, AllocationOperations, Driver, GpuDriver, KfdDriver, KfdEventSubscription,
    linux_kfd,
};
use crate::event::DeviceEvent;
use crate::host_storage::Shared;
use crate::profiling::ClockCounters;
use crate::session::DriverContextLifetime;
use crate::topology::{Endpoint, GpuPresentation};
use crate::{Error, ErrorKind};
use std::sync::atomic::{AtomicU64, Ordering};

/// One driver instance installed in a session.
///
/// Each variant owns a shared reference to a concrete driver implementation.
/// Adding a driver implementation adds a variant here and in [`DeviceDriverState`]
/// without adding GPU capabilities to the common [`Driver`] trait.
pub(crate) enum DriverInstance {
    LinuxKfd(Shared<KfdDriver>),
    #[cfg(test)]
    Test(Shared<test_driver::TestDriver>),
}

impl DriverInstance {
    /// Returns the process-unique identity assigned to this driver instance.
    pub(crate) fn instance_id(&self) -> u64 {
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

    pub(crate) fn has_other_owners(&self) -> bool {
        match self {
            Self::LinuxKfd(driver) => Shared::strong_count(driver) != 1,
            #[cfg(test)]
            Self::Test(driver) => Shared::strong_count(driver) != 1,
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

    pub(crate) fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        match self {
            Self::LinuxKfd(driver) => driver.open_endpoint(id),
            #[cfg(test)]
            Self::Test(driver) => driver.open_endpoint(id),
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

    // A non-GPU driver variant needs the same fallible routing contract.
    #[allow(clippy::unnecessary_wraps)]
    pub(crate) fn gpu_presentation(&self, endpoint: &Endpoint) -> Result<GpuPresentation, Error> {
        match self {
            Self::LinuxKfd(driver) => Ok(driver.gpu_presentation(endpoint)),
            #[cfg(test)]
            Self::Test(_) => Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "driver does not present GPU endpoints",
            }),
        }
    }

    pub(crate) fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        match self {
            Self::LinuxKfd(driver) => driver.supports_host_registration(endpoint),
            #[cfg(test)]
            Self::Test(_) => false,
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
    LinuxKfd {
        driver: Shared<KfdDriver>,
        state: KfdDeviceState,
    },
    #[cfg(test)]
    Test {
        driver: Shared<test_driver::TestDriver>,
        state: test_driver::TestDevice,
    },
}

impl DeviceDriverState {
    /// Checks the exact installed driver owner of an activated device.
    /// A matching numeric endpoint identity alone cannot establish ownership.
    pub(crate) fn belongs_to(&self, instance: &DriverInstance) -> bool {
        match (self, instance) {
            (Self::LinuxKfd { driver, .. }, DriverInstance::LinuxKfd(installed)) => {
                Shared::ptr_eq(driver, installed)
            }
            #[cfg(test)]
            (Self::Test { driver, .. }, DriverInstance::Test(installed)) => {
                Shared::ptr_eq(driver, installed)
            }
            #[cfg(test)]
            _ => false,
        }
    }

    pub(crate) fn is_gpu(&self) -> bool {
        match self {
            Self::LinuxKfd { .. } => true,
            #[cfg(test)]
            Self::Test { .. } => false,
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

    pub(crate) fn shares_address_domain(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::LinuxKfd {
                    driver: left,
                    state: a,
                },
                Self::LinuxKfd {
                    driver: right,
                    state: b,
                },
            ) => Shared::ptr_eq(left, right) && AddressSpaceInfo::shares_address_domain(a, b),
            #[cfg(test)]
            (
                Self::Test {
                    driver: left,
                    state: a,
                },
                Self::Test {
                    driver: right,
                    state: b,
                },
            ) => Shared::ptr_eq(left, right) && AddressSpaceInfo::shares_address_domain(a, b),
            #[cfg(test)]
            _ => false,
        }
    }

    pub(crate) fn address_range(&self) -> (u64, u64) {
        match self {
            Self::LinuxKfd { state, .. } => AddressSpaceInfo::address_range(state),
            #[cfg(test)]
            Self::Test { state, .. } => AddressSpaceInfo::address_range(state),
        }
    }

    pub(crate) fn allocator(&self) -> crate::host_storage::Allocator {
        match self {
            Self::LinuxKfd { driver, .. } => driver.allocator(),
            #[cfg(test)]
            Self::Test { driver, .. } => driver.allocator(),
        }
    }

    // The Result rejects other driver variants when they are installed.
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
}

/// Concrete observer retained behind the device-independent event contract.
pub(crate) enum EventSubscription {
    LinuxKfd(KfdEventSubscription),
}

impl EventSubscription {
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
    LinuxKfd(linux_kfd::LinuxSelector),
    #[allow(dead_code)]
    Opaque(u64),
}

#[cfg(not(all(
    target_os = "linux",
    any(target_arch = "x86_64", target_arch = "aarch64")
)))]
compile_error!("rocddi currently supports Linux x86-64 and AArch64");

pub(crate) type KfdDeviceState = <KfdDriver as Driver>::DeviceState;
pub(crate) type KfdAllocation = <KfdDriver as AllocationOperations>::Allocation;

static NEXT_DRIVER_INSTANCE: AtomicU64 = AtomicU64::new(1);

/// Allocates an identity for one constructed driver instance in this process.
pub(crate) fn new_driver_instance() -> u64 {
    NEXT_DRIVER_INSTANCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .unwrap_or_else(|_| std::process::abort())
}

#[cfg(test)]
fn unsupported<T>() -> Result<T, Error> {
    Err(Error::Operation {
        kind: ErrorKind::Unsupported,
        detail: "activated driver has no GPU capability",
    })
}

/// Routes GPU-only operations through the driver that activated this device.
/// Other device-family variants report `Unsupported` without acquiring GPU state.
impl DeviceDriverState {
    pub(crate) fn supports_expert_scheduling(&self) -> Result<bool, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.supports_expert_scheduling(state),
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
        }
    }

    pub(crate) fn set_persisting_l2_cache_size(&self, size: u32) -> Result<(), Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.set_persisting_l2_cache_size(state, size),
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
        }
    }

    pub(crate) fn available_memory(&self) -> Result<u64, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.available_memory(state),
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
        }
    }

    pub(crate) fn available_sdma_engines(&self) -> Result<u32, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.available_sdma_engines(state),
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
        }
    }

    pub(crate) fn clock_counters(&self) -> Result<ClockCounters, Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.clock_counters(state),
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
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
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
        }
    }

    pub(crate) fn spm_acquire(&self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.spm_acquire(state),
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
        }
    }

    pub(crate) fn spm_release(&self) -> Result<(), Error> {
        match self {
            Self::LinuxKfd { driver, state } => driver.spm_release(state),
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
        }
    }

    /// # Safety
    /// The caller retains both potentially reachable destinations until a
    /// successful replacement, unset, or conclusive teardown.
    #[allow(unsafe_code)]
    pub(crate) unsafe fn spm_set_destination(
        &self,
        size: u32,
        timeout: &mut u32,
        bytes_copied: &mut u32,
        destination: Option<usize>,
        data_loss: &mut bool,
    ) -> Result<(), Error> {
        match self {
            // SAFETY: The caller preserves previous and new destinations.
            Self::LinuxKfd { driver, state } => unsafe {
                driver.spm_set_destination(
                    state,
                    size,
                    timeout,
                    bytes_copied,
                    destination,
                    data_loss,
                )
            },
            #[cfg(test)]
            Self::Test { .. } => unsupported(),
        }
    }
}
