// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Session identity, driver context lifetime, and root coordination for rocddi.
//!
//! A session is a caller's logical scope for discovering endpoints, activating
//! devices, and eventually shutting down its installed drivers. Each driver
//! applies the session's maximum context lifetime to state it acquires.
//! Creating a session does not activate a device or acquire a driver context.
//! Device, memory, and queue owners live in their respective modules.

use crate::device::Device;
use crate::driver::{self, DriverInstance, HostMemoryDriver, VirtualMemoryDriver};
use crate::host_storage::{Allocator, Buffer, Shared};
use crate::memory::{
    DeviceIntervals, DriverHostAllocation, DriverVirtualAddress, HostAllocation, HostIntervals,
    VirtualAddress,
};
use crate::topology::{Endpoint, GpuPresentation};
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
    /// Shared session registry. Each entry owns one concrete driver instance;
    /// each activated device retains only its owning driver instance.
    drivers: Shared<DriverRegistry>,
}

/// One installed driver instance and its completed-shutdown marker.
/// A successful shutdown is never repeated if another driver later fails.
struct DriverEntry {
    driver: DriverInstance,
    stopped: bool,
}

/// Set of drivers installed in one logical session. The shared owner makes
/// clones cheap and lets destruction first prove that no clone is active.
struct DriverRegistry {
    entries: Buffer<DriverEntry>,
    lifetime: DriverContextLifetime,
    closing: bool,
}

impl Clone for Session {
    /// Shares this session's driver set, identities, and lifetime policy.
    /// A separate call to [`Session::new`] creates an independent session.
    fn clone(&self) -> Self {
        Self {
            drivers: self.drivers.clone(),
        }
    }
}

impl Session {
    /// Returns optional display metadata from the endpoint's driver.
    /// The GPU target and optional PCI identity remain in [`Endpoint`].
    ///
    /// # Errors
    /// Rejects an endpoint owned by another session or a non-GPU endpoint.
    pub fn gpu_presentation(&self, endpoint: &Endpoint) -> Result<GpuPresentation, Error> {
        self.ensure_open()?;
        let Some(entry) = self.owns_endpoint(endpoint) else {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "GPU presentation requires a GPU in this session",
            });
        };
        if endpoint.gpu().is_none() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "GPU presentation requires a GPU in this session",
            });
        }
        entry.driver.gpu_presentation(endpoint)
    }

    /// Finds the driver instance named by a passive endpoint snapshot. This
    /// does not refresh the endpoint or acquire driver state.
    fn owns_endpoint(&self, endpoint: &Endpoint) -> Option<&DriverEntry> {
        self.drivers
            .entries
            .iter()
            .find(|entry| entry.driver.instance_id() == endpoint.driver_instance)
    }

    /// Rejects new session operations after driver cleanup has started.
    fn ensure_open(&self) -> Result<(), Error> {
        if self.drivers.closing {
            Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "session driver shutdown has begun",
            })
        } else {
            Ok(())
        }
    }

    /// Borrows the Linux KFD driver instance for Linux-specific memory interop.
    /// A session without KFD cannot use KFD descriptor or SVM operations.
    #[allow(clippy::unnecessary_find_map)]
    pub(crate) fn linux_kfd(&self) -> Result<&Shared<driver::KfdDriver>, Error> {
        self.ensure_open()?;
        self.drivers
            .entries
            .iter()
            .find_map(|entry| match &entry.driver {
                DriverInstance::LinuxKfd(driver) => Some(driver),
                #[cfg(test)]
                DriverInstance::Test(_) => None,
            })
            .ok_or(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "session has no Linux KFD driver",
            })
    }

    /// Creates a session and its installed drivers using the system allocator
    /// for metadata. No endpoint is opened, enumerated, or activated.
    ///
    /// # Errors
    /// Fails with `ResourceExhausted` if driver registry metadata cannot be
    /// allocated.
    pub fn new(lifetime: DriverContextLifetime) -> Result<Self, Error> {
        Self::with_allocator(lifetime, Allocator::default())
    }

    /// Creates a session and copies the supplied allocator into its drivers.
    /// No endpoint is opened, enumerated, or activated.
    /// Later metadata owners inherit these callbacks; the driver acquires any
    /// separate backing required for device operations. Callback code and user
    /// data must remain valid for every owner created with them, as required by
    /// [`Allocator`].
    ///
    /// # Errors
    /// Fails with `ResourceExhausted` if the allocator declines registry or
    /// driver metadata. Failure acquires no endpoint or driver address-space
    /// state.
    pub fn with_allocator(
        lifetime: DriverContextLifetime,
        allocator: Allocator,
    ) -> Result<Self, Error> {
        let mut entries = Buffer::try_with_capacity(1, allocator)?;
        entries.try_push(DriverEntry {
            driver: DriverInstance::LinuxKfd(Shared::new(
                driver::KfdDriver::with_context_lifetime(allocator, lifetime),
                allocator,
            )?),
            stopped: false,
        })?;
        Self::from_drivers(entries, lifetime, allocator)
    }

    /// Takes already constructed driver instances as one session registry.
    /// Every driver must have been configured with the same lifetime policy.
    fn from_drivers(
        entries: Buffer<DriverEntry>,
        lifetime: DriverContextLifetime,
        allocator: Allocator,
    ) -> Result<Self, Error> {
        if entries.is_empty() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "session requires at least one driver",
            });
        }
        for (index, entry) in entries.iter().enumerate() {
            if entry.driver.context_lifetime() != lifetime {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "session drivers require one context lifetime",
                });
            }
            if entries.as_slice()[..index]
                .iter()
                .any(|other| other.driver.instance_id() == entry.driver.instance_id())
            {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "session drivers require distinct instance identities",
                });
            }
        }
        Ok(Self {
            drivers: Shared::new(
                DriverRegistry {
                    entries,
                    lifetime,
                    closing: false,
                },
                allocator,
            )?,
        })
    }

    /// Discharges this session's driver context dependencies without waiting
    /// for device work. The adapter must have discharged its public borrowing
    /// obligations first. [`DriverContextLifetime::Process`] may leave the
    /// process context and its device bindings alive. Dependencies backed by
    /// caller callbacks must be released before destruction can succeed, even
    /// when an ambiguous driver failure prevents recovery.
    ///
    /// # Errors
    /// `Busy` leaves every driver unchanged while another session clone, an
    /// activated device, or another driver-backed owner retains a driver instance.
    /// Once cleanup starts, an error preserves only the unfinished work; the
    /// session may then be used only for another destroy attempt. A driver that
    /// already shut down successfully is never shut down again on retry.
    pub fn destroy(&mut self) -> Result<(), Error> {
        let registry = Shared::get_mut(&mut self.drivers).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "another owner still borrows this session",
        })?;
        if registry
            .entries
            .iter()
            .any(|entry| entry.driver.has_other_owners())
        {
            return Err(Error::Operation {
                kind: ErrorKind::Busy,
                detail: "a device or resource still borrows a session driver",
            });
        }
        registry.closing = true;
        for entry in &mut registry.entries {
            if !entry.stopped {
                entry.driver.shutdown()?;
                entry.stopped = true;
            }
        }
        Ok(())
    }

    /// Allocates host-only storage under either driver context lifetime policy.
    /// The current Linux host allocator is supplied by the KFD driver, and a
    /// session without KFD reports `Unsupported`.
    /// `size` is a nonzero multiple of
    /// [`host_page_size`](crate::memory::host_page_size); `alignment` is a power
    /// of two at least that large. The driver may reserve a larger backing
    /// extent while preserving the requested logical range. This owner needs no
    /// activated device and retains only its storage and allocator.
    ///
    /// # Errors
    /// Rejects invalid extents or a driver whose teardown has begun. Metadata
    /// exhaustion and driver allocation failure leave no published owner; driver
    /// errors retain their original cause.
    pub fn allocate_host(&self, size: u64, alignment: u64) -> Result<HostAllocation, Error> {
        Ok(HostAllocation {
            inner: DriverHostAllocation::new(self.linux_kfd()?.allocate_host(size, alignment)?),
        })
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
        if self.drivers.entries.len() == 1 {
            return self.drivers.entries[0].driver.enumerate(visitor);
        }
        let mut records = Buffer::new(self.drivers.allocator());
        for entry in &self.drivers.entries {
            entry.driver.enumerate(&mut |endpoint| {
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
        if self.drivers.entries.len() == 1 {
            return self.drivers.entries[0].driver.open_endpoint(id);
        }
        let mut owner = None;
        for (index, entry) in self.drivers.entries.iter().enumerate() {
            entry.driver.enumerate(&mut |endpoint| {
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
        self.drivers.entries[index].driver.open_endpoint(id)
    }

    /// Reserves one process virtual-address range common to every supplied
    /// activated device. A nonzero requested address is a hint and may fall
    /// back to another address in the common aperture. The current reservation
    /// implementation requires KFD devices in this session.
    ///
    /// # Errors
    /// Rejects an empty or cross-session device set, incompatible apertures,
    /// invalid page-aligned extents, a non-KFD device, or address-space
    /// exhaustion.
    pub fn reserve_virtual_address(
        &self,
        devices: &[&Device],
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<VirtualAddress, Error> {
        // Every device must come from this exact driver instance. Intersect their
        // address apertures so the reservation can use one device virtual
        // address on all of them; an empty set cannot define a usable aperture.
        self.ensure_open()?;
        let driver = self.linux_kfd()?;
        let mut bounds: Option<(u64, u64)> = None;
        for device in devices {
            let (device_driver, _) = device.linux_kfd()?;
            if !Shared::ptr_eq(driver, device_driver) {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "virtual-address devices must belong to one session",
                });
            }
            let device_bounds = device.address_range();
            bounds = Some(bounds.map_or(device_bounds, |bounds| {
                (bounds.0.max(device_bounds.0), bounds.1.min(device_bounds.1))
            }));
        }
        let bounds = bounds.ok_or(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "virtual-address reservation requires an activated device",
        })?;
        if bounds.0 > bounds.1 {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "activated devices have no common virtual-address aperture",
            });
        }
        // Prepare every callback-backed owner before reserving driver-managed
        // address space. Allocation failure then cannot strand a reservation.
        let owner = Shared::try_new_uninit(driver.allocator())?;
        let intervals = Shared::new(HostIntervals::new(driver.allocator()), driver.allocator())?;
        let device_intervals =
            Shared::new(DeviceIntervals::new(driver.allocator()), driver.allocator())?;
        let inner = driver.reserve_virtual_address(bounds, size, alignment, address)?;
        Ok(VirtualAddress {
            inner: DriverVirtualAddress::new(
                driver.clone(),
                owner.write(inner),
                intervals,
                device_intervals,
            ),
        })
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
        let entry = self.owns_endpoint(endpoint).ok_or(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "endpoint belongs to another driver instance",
        })?;
        let activation = entry.driver.activate(endpoint)?;
        let endpoint = endpoint.clone();
        // Only GPU devices need the shared copy-resource pool. Build it before
        // publishing Device so allocation failure leaves no partial handle.
        let copy_pool = endpoint
            .gpu()
            .map(|_| {
                Shared::new(
                    crate::gpu::CopyResourcePool::default(),
                    activation.allocator(),
                )
            })
            .transpose()?;
        Ok(Device {
            activation,
            endpoint,
            copy_pool,
        })
    }

    /// Returns the policy used to qualify services that depend on the driver
    /// context.
    /// This is the policy requested at construction, regardless of whether a
    /// driver context or device has been activated yet.
    #[must_use]
    pub fn driver_context_lifetime(&self) -> DriverContextLifetime {
        self.drivers.lifetime
    }

    /// Returns whether the endpoint's driver can register caller-owned host
    /// pages for the selected endpoint and context lifetime. This capability
    /// query does not activate the device; a registration can still
    /// fail if its caller pages cannot be bound by the driver.
    #[must_use]
    pub fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        !self.drivers.closing
            && self
                .owns_endpoint(endpoint)
                .is_some_and(|entry| entry.driver.supports_host_registration(endpoint))
    }
}

#[cfg(all(test, target_os = "linux"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::driver::test_driver::TestDriver;
    use crate::memory::{DeviceAccess, MemoryKind};
    use crate::topology::EndpointKind;
    use std::sync::atomic::Ordering;

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
            inner: DriverVirtualAddress::new(
                driver.clone(),
                Shared::new(inner, driver.allocator()).unwrap(),
                Shared::new(HostIntervals::new(driver.allocator()), driver.allocator()).unwrap(),
                Shared::new(DeviceIntervals::new(driver.allocator()), driver.allocator()).unwrap(),
            ),
        };
        let mapping_lease = address.inner.inner.clone();
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(address.free().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(address.info(), info);
        drop(mapping_lease);
        address.free().unwrap();
        drop(address);
        session.destroy().unwrap();
    }

    fn test_session(ids: &[[u8; 16]]) -> (Session, Vec<TestDriverProbe>) {
        let allocator = Allocator::default();
        let mut entries = Buffer::try_with_capacity(ids.len(), allocator).unwrap();
        let mut probes = Vec::new();
        for id in ids {
            let driver = TestDriver::new(*id, EndpointKind::Npu, DriverContextLifetime::Session);
            probes.push(TestDriverProbe {
                shutdowns: driver.shutdowns.clone(),
                fail_shutdown: driver.fail_shutdown.clone(),
            });
            entries
                .try_push(DriverEntry {
                    driver: DriverInstance::Test(Shared::new(driver, allocator).unwrap()),
                    stopped: false,
                })
                .unwrap();
        }
        (
            Session::from_drivers(entries, DriverContextLifetime::Session, allocator).unwrap(),
            probes,
        )
    }

    struct TestDriverProbe {
        shutdowns: std::sync::Arc<std::sync::atomic::AtomicUsize>,
        fail_shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
    }

    #[test]
    fn multiple_drivers_route_endpoints_and_retry_only_unfinished_shutdown() {
        let ids = [[0x11; 16], [0x22; 16]];
        let (mut session, probes) = test_session(&ids);
        let mut endpoints = Vec::new();
        session
            .enumerate(&mut |endpoint| {
                endpoints.push(endpoint);
                Ok(())
            })
            .unwrap();
        assert_eq!(endpoints.len(), 2);
        assert_eq!(endpoints[0].id, ids[0]);
        assert_eq!(endpoints[1].id, ids[1]);
        assert_eq!(session.open_endpoint(ids[1]).unwrap(), endpoints[1]);
        let (mut other_session, _) = test_session(&ids);
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
            session
                .reserve_virtual_address(&[&first], 4096, 4096, 0)
                .err()
                .unwrap()
                .kind(),
            ErrorKind::Unsupported
        );
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(probes[0].shutdowns.load(Ordering::Relaxed), 0);
        assert_eq!(probes[1].shutdowns.load(Ordering::Relaxed), 0);
        drop(first);
        drop(second);
        probes[1].fail_shutdown.store(true, Ordering::Relaxed);
        assert_eq!(session.destroy().unwrap_err().kind(), ErrorKind::Driver);
        assert_eq!(probes[0].shutdowns.load(Ordering::Relaxed), 1);
        assert_eq!(probes[1].shutdowns.load(Ordering::Relaxed), 1);
        assert_eq!(
            session.open_endpoint(ids[0]).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
        session.destroy().unwrap();
        assert_eq!(probes[0].shutdowns.load(Ordering::Relaxed), 1);
        assert_eq!(probes[1].shutdowns.load(Ordering::Relaxed), 2);
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
