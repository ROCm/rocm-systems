// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Session identity, driver context lifetime, and root coordination for rocddi.
//!
//! A session is a caller's logical scope for discovering endpoints, activating
//! devices, and eventually shutting down its installed drivers. Each driver
//! applies the session's maximum context lifetime to state it acquires.
//! Creating a session does not activate a device or acquire a driver context.
//! Device, memory, and queue owners live in their respective modules.

use crate::device::{Device, DeviceResources};
use crate::driver::{self, DriverInstance};
use crate::memory::storage::Shared;
use crate::memory::storage::TryPush;
use crate::topology::{Endpoint, EndpointDetails};
use crate::{Error, ErrorKind};

/// Maximum lifetime allowed for a driver context acquired through a session.
///
/// A driver context is the driver-managed state needed to operate activated
/// devices, including any connections and device bindings. This policy bounds
/// how long that state may remain acquired. Callers still release device,
/// memory, and queue owners according to their own lifetimes under either
/// policy. A driver may retain backing that its context can still access when
/// release cannot be proved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DriverContextLifetime {
    /// Allows a process-scoped driver context and its bindings to survive
    /// session destruction. Callers still release their resource owners;
    /// ambiguous driver results may require retaining backing until process
    /// teardown.
    Process,
    /// Requires the driver context and every acquired binding to be reclaimable
    /// within the session's lifetime. A driver must reject an operation before
    /// acquisition when it cannot satisfy this bound.
    Session,
}

/// Caller-owned scope for discovery, activation, and driver shutdown.
///
/// Each session has a requested [`DriverContextLifetime`] and one or more
/// independently identified drivers. Clones share the same driver set;
/// separate constructors create independent driver instances. Creating a
/// session does not activate an endpoint. An activated device retains exactly
/// its owning driver instance, while the session coordinates discovery,
/// routing, and shutdown.
pub struct Session {
    /// Shared registry. Each activated device retains only its owning driver.
    registry: Shared<DriverRegistry>,
}

/// Set of drivers installed in one logical session. The shared owner makes
/// clones cheap and lets destruction first prove that no clone is active.
struct DriverRegistry {
    /// Installed drivers; after teardown starts, only pending drivers remain.
    drivers: Vec<DriverInstance>,
    /// Policy validated against every installed driver at construction.
    lifetime: DriverContextLifetime,
    /// Process that constructed this registry; inherited sessions cannot run.
    process_id: u32,
    /// Prevents new work once any driver shutdown has started.
    closing: bool,
}

impl Clone for Session {
    /// Shares this session's driver set, identities, and lifetime policy.
    /// A separate call to [`Session::new`] creates an independent session.
    fn clone(&self) -> Self {
        Self {
            registry: self.registry.clone(),
        }
    }
}

impl Session {
    /// Queries optional information from the endpoint's owning driver.
    /// Identity and discovered capabilities remain in [`Endpoint`]. This
    /// query does not activate or revalidate the endpoint snapshot.
    ///
    /// # Errors
    /// Rejects an endpoint owned by another session or a closed session, and
    /// propagates any driver query error.
    pub fn query_endpoint_details(&self, endpoint: &Endpoint) -> Result<EndpointDetails, Error> {
        self.ensure_open()?;
        let Some(driver) = self.driver_for_endpoint(endpoint) else {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "endpoint details require an endpoint in this session",
            });
        };
        driver.query_endpoint_details(endpoint)
    }

    /// Selects the driver named by a passive endpoint snapshot without
    /// refreshing the endpoint or acquiring driver state.
    fn driver_for_endpoint(&self, endpoint: &Endpoint) -> Option<&DriverInstance> {
        self.registry
            .drivers
            .iter()
            .find(|driver| driver.instance_id() == endpoint.driver_instance)
    }

    /// Rejects new session operations after driver cleanup has started.
    fn ensure_open(&self) -> Result<(), Error> {
        self.ensure_process()?;
        if self.registry.closing {
            Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "session driver shutdown has begun",
            })
        } else {
            Ok(())
        }
    }

    /// Rejects a session inherited through fork before touching driver state
    /// in the child process.
    fn ensure_process(&self) -> Result<(), Error> {
        if self.registry.process_id == std::process::id() {
            Ok(())
        } else {
            Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "session belongs to another process",
            })
        }
    }

    /// Borrows the Linux KFD driver instance for Linux-specific memory interop.
    /// A session without KFD cannot use KFD descriptor or SVM operations.
    #[allow(clippy::unnecessary_find_map)]
    pub(crate) fn linux_kfd(&self) -> Result<&Shared<driver::KfdDriver>, Error> {
        self.ensure_open()?;
        self.registry
            .drivers
            .iter()
            .find_map(|driver| match driver {
                DriverInstance::LinuxKfd(driver) => Some(driver),

                #[cfg(test)]
                DriverInstance::Test(_) => None,
            })
            .ok_or(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "session has no Linux KFD driver",
            })
    }

    /// Creates one instance of every compiled driver backend without opening,
    /// enumerating, or activating an endpoint.
    ///
    /// # Errors
    /// Fails with `ResourceExhausted` if driver registry metadata cannot be allocated.
    pub fn new(lifetime: DriverContextLifetime) -> Result<Self, Error> {
        let mut drivers = Vec::new();
        drivers
            .try_reserve_exact(driver::BUILTIN_DRIVER_FACTORIES.len())
            .map_err(|_| Error::Capacity {
                resource: "session drivers",
            })?;
        for construct in driver::BUILTIN_DRIVER_FACTORIES {
            drivers.push(construct(lifetime)?);
        }
        Self::from_drivers(drivers, lifetime)
    }

    /// Takes already constructed driver instances as one session registry.
    /// Every driver must have been configured with the same lifetime policy.
    fn from_drivers(
        drivers: Vec<DriverInstance>,
        lifetime: DriverContextLifetime,
    ) -> Result<Self, Error> {
        if drivers.is_empty() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "session requires at least one driver",
            });
        }
        for (index, driver) in drivers.iter().enumerate() {
            if driver.context_lifetime() != lifetime {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "session drivers require one context lifetime",
                });
            }
            if drivers[..index]
                .iter()
                .any(|other| other.instance_id() == driver.instance_id())
            {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "session drivers require distinct instance identities",
                });
            }
        }
        Ok(Self {
            registry: Shared::new(DriverRegistry {
                drivers,
                lifetime,
                process_id: std::process::id(),
                closing: false,
            })?,
        })
    }

    /// Discharges this session's driver context dependencies without waiting
    /// for device work. The adapter must have discharged its public borrowing
    /// obligations first. [`DriverContextLifetime::Process`] may leave the
    /// process context and its device bindings alive. Dependencies retained
    /// by devices and resources must be released before destruction can
    /// succeed, even when an ambiguous driver failure prevents recovery.
    ///
    /// # Errors
    /// `Busy` leaves pending drivers unchanged while another session clone, an
    /// activated device, or another driver-backed owner retains a driver instance.
    /// Once cleanup starts, every pending driver is attempted. The first error
    /// is returned, and only failed drivers remain for a destroy retry. The
    /// session may then be used only for another destroy attempt.
    pub fn destroy(&mut self) -> Result<(), Error> {
        self.ensure_process()?;
        let registry = Shared::get_mut(&mut self.registry).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "another owner still borrows this session",
        })?;
        if registry
            .drivers
            .iter()
            .any(DriverInstance::has_other_owners)
        {
            return Err(Error::Operation {
                kind: ErrorKind::Busy,
                detail: "a device or resource still borrows a session driver",
            });
        }
        registry.closing = true;
        let mut first_error = None;
        registry
            .drivers
            .retain_mut(|driver| match driver.shutdown() {
                Ok(()) => false,
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(error);
                    }
                    true
                }
            });
        // Retain failed drivers for retry. Every driver gets an attempt even
        // when another fails, so shutdown progress is independent of order.
        first_error.map_or(Ok(()), Err)
    }

    /// Visits passive endpoint records from every installed driver. Each
    /// driver's records are generation-consistent; independent drivers have no
    /// shared native generation. With multiple drivers, the session stages all
    /// records and rejects duplicate endpoint IDs before calling `visitor`.
    /// Enumeration does not activate an endpoint or acquire execution state.
    ///
    /// # Errors
    /// Reports malformed or unsupported driver metadata, discovery errors,
    /// allocation failure, duplicate IDs, or topology churn. Driver discovery
    /// failures call no visitor. A visitor error stops delivery after any
    /// records already visited and is returned unchanged.
    pub fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.ensure_open()?;
        if self.registry.drivers.len() == 1 {
            return self.registry.drivers[0].enumerate(visitor);
        }
        let mut records = Vec::new();
        for driver in &self.registry.drivers {
            driver.enumerate(&mut |endpoint| {
                if records
                    .iter()
                    .any(|record: &Endpoint| record.id == endpoint.id)
                {
                    return Err(Error::Operation {
                        kind: ErrorKind::InvalidData,
                        detail: "drivers reported the same endpoint ID",
                    });
                }
                records.try_push(endpoint).map_err(Error::from)
            })?;
        }
        for endpoint in records.iter().cloned() {
            visitor(endpoint)?;
        }
        Ok(())
    }

    /// Reads the exact endpoint selected by `id` and verifies its identity.
    /// A multi-driver session first discovers which driver owns the ID and
    /// rejects an ID collision. No execution endpoint is acquired.
    ///
    /// # Errors
    /// Reports removal or identity mismatch, inconsistent discovery state,
    /// malformed metadata, duplicate IDs, or the driver error from the
    /// selected endpoint. A driver whose teardown has begun rejects the
    /// query.
    pub fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        self.ensure_open()?;
        if self.registry.drivers.len() == 1 {
            return self.registry.drivers[0].open_endpoint(id);
        }
        let mut owner = None;
        for (index, driver) in self.registry.drivers.iter().enumerate() {
            driver.enumerate(&mut |endpoint| {
                if endpoint.id == id && owner.replace(index).is_some() {
                    return Err(Error::Operation {
                        kind: ErrorKind::InvalidData,
                        detail: "drivers reported the same endpoint ID",
                    });
                }
                Ok(())
            })?;
        }
        let index = owner.ok_or(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "endpoint ID is not present in this session",
        })?;
        self.registry.drivers[index].open_endpoint(id)
    }

    /// Revalidates a passive endpoint with its owning driver and acquires or
    /// reuses the driver context and device state required for operations. The
    /// returned [`Device`] retains only that driver instance. The driver may
    /// retain reusable context resources for later activation, subject to this
    /// session's context lifetime policy.
    ///
    /// # Errors
    /// Reports changed endpoint metadata, a lifetime policy the driver cannot
    /// honor, unsupported driver interfaces or address-space layouts,
    /// incompatible existing ownership, allocation failure, or driver I/O.
    /// Partial driver setup remains owned for safe cleanup or a later retry.
    pub fn activate(&self, endpoint: &Endpoint) -> Result<Device, Error> {
        self.ensure_open()?;
        let driver = self.driver_for_endpoint(endpoint).ok_or(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "endpoint belongs to another driver instance",
        })?;
        let driver_state = driver.activate(endpoint)?;
        let endpoint = endpoint.clone();
        // Construct family-owned services before publishing the device so a
        // metadata allocation failure cannot leave a partial public handle.
        let resources = DeviceResources::for_activated(&endpoint, &driver_state)?;
        Ok(Device {
            driver_state,
            endpoint,
            resources,
        })
    }

    /// Returns the policy used to qualify services that depend on the driver
    /// context.
    /// This is the policy requested at construction, regardless of whether a
    /// driver context or device has been activated yet.
    #[must_use]
    pub fn driver_context_lifetime(&self) -> DriverContextLifetime {
        self.registry.lifetime
    }

    /// Returns whether the endpoint's driver can register caller-owned host
    /// pages for the selected endpoint and context lifetime. This capability
    /// query does not activate the device; a registration can still
    /// fail if its caller pages cannot be bound by the driver.
    #[must_use]
    pub fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        !self.registry.closing
            && self
                .driver_for_endpoint(endpoint)
                .is_some_and(|driver| driver.supports_host_registration(endpoint))
    }
}

#[cfg(all(test, target_os = "linux"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::driver::VirtualMemoryOperations;
    use crate::driver::test_driver::TestDriver;
    use crate::memory::device::VirtualAddress;
    use crate::memory::host::HostAllocation;
    use crate::memory::{
        DeviceAccess, DeviceIntervals, DriverVirtualAddress, HostIntervals, MemoryKind,
    };
    use crate::topology::{EndpointDetailsKind, EndpointKind};
    use std::sync::atomic::Ordering;

    #[test]
    fn builtin_catalog_constructs_independent_driver_instances() {
        let mut first = Session::new(DriverContextLifetime::Session).unwrap();
        let mut second = Session::new(DriverContextLifetime::Session).unwrap();
        let shared = first.clone();

        assert_eq!(
            first.registry.drivers.len(),
            driver::BUILTIN_DRIVER_FACTORIES.len()
        );
        assert_eq!(second.registry.drivers.len(), first.registry.drivers.len());
        for ((first_driver, second_driver), shared_driver) in first
            .registry
            .drivers
            .iter()
            .zip(&second.registry.drivers)
            .zip(&shared.registry.drivers)
        {
            assert_ne!(first_driver.instance_id(), second_driver.instance_id());
            assert_eq!(first_driver.instance_id(), shared_driver.instance_id());
        }

        drop(shared);
        first.destroy().unwrap();
        second.destroy().unwrap();
    }

    #[test]
    fn cloned_session_blocks_driver_shutdown_until_released() {
        let mut session = Session::new(DriverContextLifetime::Session).unwrap();
        let clone = session.clone();
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
        drop(clone);
        session.destroy().unwrap();
    }

    #[test]
    fn live_reservation_lease_blocks_release_until_its_mapping_owner_finishes() {
        let mut session = Session::new(DriverContextLifetime::Session).unwrap();
        let driver = session.linux_kfd().unwrap();
        // A process VA reservation needs no GPU endpoint, so this exercises the
        // public owner and its driver cleanup on CPU-only test hosts.
        let inner = driver
            .reserve_virtual_address((0x1_0000, isize::MAX as u64), 4096, 4096, 0)
            .unwrap();
        let info = driver::CachedInfo::cached_info(&*inner);
        let mut address = VirtualAddress {
            inner: driver::VirtualAddressState::from_linux_kfd(DriverVirtualAddress::new(
                driver.clone(),
                Shared::new(inner).unwrap(),
                Shared::new(HostIntervals::new()).unwrap(),
                Shared::new(DeviceIntervals::new()).unwrap(),
            )),
        };
        let mapping_lease = address.inner.linux_kfd().inner.clone();
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(address.free().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(address.info(), info);
        drop(mapping_lease);
        address.free().unwrap();
        drop(address);
        session.destroy().unwrap();
    }

    fn test_session(ids: &[[u8; 16]]) -> (Session, Vec<TestDriverProbe>) {
        test_session_with_host_registration(ids, false)
    }

    fn test_session_with_host_registration(
        ids: &[[u8; 16]],
        host_registration: bool,
    ) -> (Session, Vec<TestDriverProbe>) {
        let mut drivers = Vec::with_capacity(ids.len());
        let mut probes = Vec::new();
        for id in ids {
            let mut driver =
                TestDriver::new(*id, EndpointKind::Npu, DriverContextLifetime::Session);
            if host_registration {
                driver = driver.with_host_registration();
            }
            probes.push(TestDriverProbe {
                shutdowns: driver.shutdowns.clone(),
                fail_shutdown: driver.fail_shutdown.clone(),
            });
            drivers.push(DriverInstance::Test(Shared::new(driver).unwrap()));
        }
        (
            Session::from_drivers(drivers, DriverContextLifetime::Session).unwrap(),
            probes,
        )
    }

    struct TestDriverProbe {
        shutdowns: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        fail_shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    #[test]
    fn multiple_drivers_route_endpoints_and_retry_only_unfinished_shutdown() {
        let ids = [[0x11; 16], [0x22; 16], [0x33; 16]];
        let (mut session, probes) = test_session(&ids);
        let mut endpoints = Vec::new();
        session
            .enumerate(&mut |endpoint| {
                endpoints.push(endpoint);
                Ok(())
            })
            .unwrap();
        assert_eq!(endpoints.len(), 3);
        assert_eq!(endpoints[0].id, ids[0]);
        assert_eq!(endpoints[1].id, ids[1]);
        assert_eq!(endpoints[2].id, ids[2]);
        assert_eq!(session.open_endpoint(ids[1]).unwrap(), endpoints[1]);
        for endpoint in &endpoints {
            let details = session.query_endpoint_details(endpoint).unwrap();
            assert_eq!(details.product_name, None);
            assert_eq!(details.kind, EndpointDetailsKind::Other);
        }
        let (mut other_session, _) = test_session(&ids);
        assert_eq!(
            other_session
                .query_endpoint_details(&endpoints[0])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            other_session.activate(&endpoints[0]).err().unwrap().kind(),
            ErrorKind::InvalidArgument
        );
        other_session.destroy().unwrap();
        let first = session.activate(&endpoints[0]).unwrap();
        let second = session.activate(&endpoints[1]).unwrap();
        assert!(!first.shares_address_domain(&second));
        assert_eq!(first.address_range(), (0x1000, 0xffff));
        assert_eq!(first.gpu().err().unwrap().kind(), ErrorKind::Unsupported);
        assert!(!session.supports_host_registration(&endpoints[0]));
        assert_eq!(
            first
                .allocate(MemoryKind::System, 4096, 4096, DeviceAccess::READ)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            VirtualAddress::reserve(&[&first], 4096, 4096, 0)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(probes[0].shutdowns.load(Ordering::Relaxed), 0);
        assert_eq!(probes[1].shutdowns.load(Ordering::Relaxed), 0);
        assert_eq!(probes[2].shutdowns.load(Ordering::Relaxed), 0);
        drop(first);
        drop(second);
        probes[1].fail_shutdown.store(true, Ordering::Relaxed);
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Driver);
        assert_eq!(probes[0].shutdowns.load(Ordering::Relaxed), 1);
        assert_eq!(probes[1].shutdowns.load(Ordering::Relaxed), 1);
        assert_eq!(probes[2].shutdowns.load(Ordering::Relaxed), 1);
        assert_eq!(
            session.open_endpoint(ids[0]).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            session
                .query_endpoint_details(&endpoints[0])
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
        session.destroy().unwrap();
        assert_eq!(probes[0].shutdowns.load(Ordering::Relaxed), 1);
        assert_eq!(probes[1].shutdowns.load(Ordering::Relaxed), 2);
        assert_eq!(probes[2].shutdowns.load(Ordering::Relaxed), 1);
        session.destroy().unwrap();
        assert_eq!(probes[0].shutdowns.load(Ordering::Relaxed), 1);
        assert_eq!(probes[1].shutdowns.load(Ordering::Relaxed), 2);
        assert_eq!(probes[2].shutdowns.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn virtual_address_reservation_validates_driver_identity_before_capability() {
        let own_id = [0x55; 16];
        let foreign_id = [0x66; 16];
        let (mut session, _) = test_session(&[own_id]);
        let (mut other, _) = test_session(&[foreign_id]);
        let own = session
            .activate(&session.open_endpoint(own_id).unwrap())
            .unwrap();
        let foreign = other
            .activate(&other.open_endpoint(foreign_id).unwrap())
            .unwrap();

        assert_eq!(
            VirtualAddress::reserve(&[], 4096, 4096, 0)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            VirtualAddress::reserve(&[&foreign], 4096, 4096, 0)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(
            VirtualAddress::reserve(&[&own, &foreign], 4096, 4096, 0)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            VirtualAddress::reserve(&[&own], 4096, 4096, 0)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );

        drop(own);
        drop(foreign);
        session.destroy().unwrap();
        other.destroy().unwrap();
    }

    #[test]
    fn host_allocation_is_available_without_a_gpu_driver() {
        let (mut session, _) = test_session(&[[0x44; 16]]);
        let page = crate::memory::host_page_size().unwrap();
        let mut allocation = HostAllocation::allocate(page, page).unwrap();
        let info = allocation.info();
        assert_ne!(info.host_address, 0);
        assert_eq!(info.size, page);
        let device = session
            .activate(&session.open_endpoint([0x44; 16]).unwrap())
            .unwrap();
        assert_eq!(
            device
                .register_host_allocation(
                    &allocation,
                    &[],
                    crate::memory::HostMappingPolicy::Fine,
                    page,
                    DeviceAccess::READ,
                )
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
        allocation.free().unwrap();
        drop(allocation);
        drop(device);
        session.destroy().unwrap();
    }

    #[test]
    fn rejected_host_registration_releases_its_backing_lease() {
        let id = [0x45; 16];
        let (mut session, _) = test_session_with_host_registration(&[id], true);
        let device = session
            .activate(&session.open_endpoint(id).unwrap())
            .unwrap();
        let page = crate::memory::host_page_size().unwrap();
        let mut host = HostAllocation::allocate(page, page).unwrap();

        assert_eq!(
            device
                .register_host_allocation(
                    &host,
                    &[],
                    crate::memory::HostMappingPolicy::Fine,
                    page,
                    DeviceAccess::WRITE,
                )
                .err()
                .unwrap()
                .kind(),
            ErrorKind::InvalidArgument
        );
        host.free().unwrap();
        drop(host);
        drop(device);
        session.destroy().unwrap();
    }

    #[test]
    fn allocation_keeps_its_driver_live_after_the_device_is_dropped() {
        for lifetime in [
            DriverContextLifetime::Session,
            DriverContextLifetime::Process,
        ] {
            let id = [0x46; 16];
            let driver = TestDriver::new(id, EndpointKind::Npu, lifetime).with_host_registration();
            let shutdowns = driver.shutdowns.clone();
            let mut session = Session::from_drivers(
                vec![DriverInstance::Test(Shared::new(driver).unwrap())],
                lifetime,
            )
            .unwrap();
            let device = session
                .activate(&session.open_endpoint(id).unwrap())
                .unwrap();
            let page = crate::memory::host_page_size().unwrap();
            let mut host = HostAllocation::allocate(page, page).unwrap();
            let mut registration = device
                .register_host_allocation(
                    &host,
                    &[],
                    crate::memory::HostMappingPolicy::Fine,
                    page,
                    DeviceAccess::READ,
                )
                .unwrap();

            drop(device);
            assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
            assert_eq!(shutdowns.load(Ordering::Relaxed), 0);
            registration.free().unwrap();
            assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
            drop(registration);
            host.free().unwrap();
            session.destroy().unwrap();
            assert_eq!(shutdowns.load(Ordering::Relaxed), 1);
        }
    }

    #[test]
    fn one_host_backing_supports_independent_driver_registrations() {
        let ids = [[0x11; 16], [0x22; 16]];
        let (mut session, _) = test_session_with_host_registration(&ids, true);
        let first = session
            .activate(&session.open_endpoint(ids[0]).unwrap())
            .unwrap();
        let second = session
            .activate(&session.open_endpoint(ids[1]).unwrap())
            .unwrap();
        let page = crate::memory::host_page_size().unwrap();
        let mut host = HostAllocation::allocate(page, page).unwrap();
        let permissions = DeviceAccess::READ | DeviceAccess::WRITE;
        let mut first_mapping = first
            .register_host_allocation(
                &host,
                &[],
                crate::memory::HostMappingPolicy::Fine,
                page,
                permissions,
            )
            .unwrap();
        let second_mapping = second
            .register_host_allocation(
                &host,
                &[],
                crate::memory::HostMappingPolicy::Fine,
                page,
                permissions,
            )
            .unwrap();
        assert_eq!(
            first_mapping.info().host_address,
            Some(host.info().host_address)
        );
        assert_eq!(
            second_mapping.info().host_address,
            Some(host.info().host_address)
        );
        assert_ne!(
            first_mapping.info().device_address,
            second_mapping.info().device_address
        );
        assert_eq!(
            first_mapping.device_address(&second).unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(host.free().unwrap_err().kind(), ErrorKind::Busy);
        first_mapping.free().unwrap();
        assert_eq!(host.free().unwrap_err().kind(), ErrorKind::Busy);
        drop(second_mapping);
        host.free().unwrap();
        drop(first_mapping);
        drop(first);
        drop(second);
        session.destroy().unwrap();
    }

    #[test]
    fn grouped_host_memory_owns_registrations_across_drivers() {
        let ids = [[0x51; 16], [0x52; 16]];
        let (mut session, _) = test_session_with_host_registration(&ids, true);
        let first = session
            .activate(&session.open_endpoint(ids[0]).unwrap())
            .unwrap();
        let second = session
            .activate(&session.open_endpoint(ids[1]).unwrap())
            .unwrap();
        let page = crate::memory::host_page_size().unwrap();
        let host = HostAllocation::allocate(page, page).unwrap();
        let mut memory = crate::memory::device::RegisteredHostMemory::new(host);
        let policy = crate::memory::HostMappingPolicy::Fine;
        let permissions = DeviceAccess::READ | DeviceAccess::WRITE;
        memory
            .register(&first, &[], policy, page, permissions)
            .unwrap();
        memory
            .register(&second, &[], policy, page, permissions)
            .unwrap();
        assert_ne!(
            memory.device_address(&first).unwrap(),
            memory.device_address(&second).unwrap()
        );
        assert_eq!(
            memory
                .register(&first, &[], policy, page, permissions)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
        memory.free().unwrap();
        drop(memory);
        drop(first);
        drop(second);
        session.destroy().unwrap();
    }

    #[test]
    #[allow(unsafe_code)]
    fn device_registration_keeps_host_pages_mapped_after_host_owner_drop() {
        let id = [0x33; 16];
        let (mut session, _) = test_session_with_host_registration(&[id], true);
        let device = session
            .activate(&session.open_endpoint(id).unwrap())
            .unwrap();
        let page = crate::memory::host_page_size().unwrap();
        let host = HostAllocation::allocate(page, page).unwrap();
        let address = host.info().host_address;
        let mut mapping = device
            .register_host_allocation(
                &host,
                &[],
                crate::memory::HostMappingPolicy::Fine,
                page,
                DeviceAccess::READ | DeviceAccess::WRITE,
            )
            .unwrap();
        drop(host);
        // SAFETY: The registration owner retains this complete mapped page.
        unsafe {
            (address as *mut u8).write(0x5a);
            assert_eq!((address as *const u8).read(), 0x5a);
        }
        mapping.free().unwrap();
        drop(mapping);
        drop(device);
        session.destroy().unwrap();
    }

    #[test]
    fn duplicate_ids_are_rejected_before_any_endpoint_is_delivered() {
        let id = [0x33; 16];
        let (session, _) = test_session(&[id, id]);
        let mut delivered = 0;
        assert_eq!(
            session
                .enumerate(&mut |_| {
                    delivered += 1;
                    Ok(())
                })
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidData
        );
        assert_eq!(delivered, 0);
        assert_eq!(
            session.open_endpoint(id).unwrap_err().kind(),
            ErrorKind::InvalidData
        );
    }
}
