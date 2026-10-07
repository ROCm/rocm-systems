// SPDX-License-Identifier: MIT

//! Session lifetime and root coordination for the rocddi interface.
//!
//! A session owns the selected platform driver and coordinates passive
//! discovery, explicit activation, and resources whose scope spans more than
//! one device. Domain-specific value types and resource owners live in their
//! respective modules rather than sharing this lifecycle namespace.

use crate::device::Device;
use crate::driver::{self, GpuPresentationDriver, HostDriver, ProviderDriver, VirtualMemoryDriver};
use crate::host_storage::{Allocator, Shared};
use crate::memory::{
    DeviceIntervals, HostAllocation, HostIntervals, ProviderHostAllocation, ProviderVirtualAddress,
    VirtualAddress,
};
use crate::topology::{Endpoint, GpuPresentation};
use crate::{Error, ErrorKind};

/// Upper bound on the lifetime of native state acquired by this session.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SessionLifetime {
    /// Allows native state whose ownership is inherently process-scoped to
    /// survive session destruction. Ordinary owners still release their
    /// resources; ambiguous native results may require retaining backing until
    /// process teardown.
    Process,
    /// Requires every acquisition to be reclaimable within the session's
    /// lifetime. A backend must reject an operation before acquisition when its
    /// native ownership model cannot satisfy this bound.
    Session,
}

/// Coordinates native connections and reusable bindings for one core session.
/// Construction allocates only the controller record. Both lifetime policies
/// permit passive queries and host storage; only an explicit activation
/// acquires endpoint state. Dropping a [`Device`] leaves any backend-retained
/// binding available for recreation under the selected lifetime policy.
pub(crate) struct ProviderSession<D: ProviderDriver> {
    driver: Shared<D>,
    lifetime: SessionLifetime,
}

pub(crate) struct Activated<D: ProviderDriver> {
    pub(crate) driver: Shared<D>,
    pub(crate) state: D::DeviceState,
    pub(crate) endpoint: Endpoint,
}

impl<D: ProviderDriver> Clone for ProviderSession<D> {
    fn clone(&self) -> Self {
        Self {
            driver: self.driver.clone(),
            lifetime: self.lifetime,
        }
    }
}

impl<D: ProviderDriver> ProviderSession<D> {
    pub(crate) fn new(
        driver: D,
        lifetime: SessionLifetime,
        allocator: Allocator,
    ) -> Result<Self, Error> {
        Ok(Self {
            driver: Shared::new(driver, allocator)?,
            lifetime,
        })
    }

    pub(crate) fn destroy(&mut self) -> Result<(), Error> {
        let driver = Shared::get_mut(&mut self.driver).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "activated devices still borrow this session",
        })?;
        driver.shutdown()
    }

    pub(crate) fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.driver.enumerate(visitor)
    }

    pub(crate) fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        self.driver.open_endpoint(id)
    }

    pub(crate) fn activate(&self, endpoint: &Endpoint) -> Result<Activated<D>, Error> {
        if endpoint.provider_instance != self.driver.provider_instance() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "endpoint belongs to another provider instance",
            });
        }
        let state = self.driver.activate(endpoint, self.lifetime)?;
        Ok(Activated {
            driver: self.driver.clone(),
            state,
            endpoint: endpoint.clone(),
        })
    }

    pub(crate) fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        endpoint.provider_instance == self.driver.provider_instance()
            && self
                .driver
                .supports_host_registration(endpoint, self.lifetime)
    }
}

impl<D: GpuPresentationDriver> ProviderSession<D> {
    fn gpu_presentation(&self, endpoint: &Endpoint) -> Result<GpuPresentation, Error> {
        if endpoint.provider_instance != self.driver.provider_instance() || endpoint.gpu().is_none()
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "GPU presentation requires a GPU in this session",
            });
        }
        Ok(self.driver.gpu_presentation(endpoint))
    }
}

impl<D: ProviderDriver + HostDriver> ProviderSession<D> {
    pub(crate) fn allocate_host(
        &self,
        size: u64,
        alignment: u64,
    ) -> Result<ProviderHostAllocation<D>, Error> {
        Ok(ProviderHostAllocation::new(
            self.driver.allocate_host(size, alignment)?,
        ))
    }
}

/// Public session using the native provider selected for this target.
/// The provider-independent discovery and activation lifecycle is shared with
/// non-GPU provider contract tests.
pub struct Session {
    inner: ProviderSession<driver::PlatformDriver>,
}

impl Clone for Session {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl Session {
    /// Returns optional display metadata from this session's native provider.
    /// The stable target and PCI identity remain in [`Endpoint`].
    ///
    /// # Errors
    /// Rejects an endpoint owned by another session or a non-GPU endpoint.
    pub fn gpu_presentation(&self, endpoint: &Endpoint) -> Result<GpuPresentation, Error> {
        self.inner.gpu_presentation(endpoint)
    }

    /// Borrows the private native controller for domain-specific crate layers.
    pub(crate) fn driver(&self) -> &Shared<driver::PlatformDriver> {
        &self.inner.driver
    }

    /// Creates an inert controller with the system allocator for metadata.
    /// No endpoint is opened or enumerated, and either lifetime is accepted.
    ///
    /// # Errors
    /// Fails with `ResourceExhausted` if the controller record cannot be allocated.
    pub fn new(lifetime: SessionLifetime) -> Result<Self, Error> {
        Self::with_allocator(lifetime, Allocator::default())
    }

    /// Creates an inert controller and copies the supplied allocator into it.
    /// Later metadata owners inherit these callbacks; platform and device calls
    /// still provide native backing. Callback code and user data must remain
    /// valid for every owner created with them, as required by [`Allocator`].
    ///
    /// # Errors
    /// Fails with `ResourceExhausted` if the allocator declines the controller
    /// record. Failure acquires no native endpoint or address-space state.
    pub fn with_allocator(lifetime: SessionLifetime, allocator: Allocator) -> Result<Self, Error> {
        Ok(Self {
            inner: ProviderSession::new(
                driver::PlatformDriver::with_lifetime(allocator, lifetime),
                lifetime,
                allocator,
            )?,
        })
    }

    /// Discharges this session's native dependencies without waiting
    /// for device work. The adapter must have discharged its public borrowing
    /// obligations first. [`SessionLifetime::Process`] may leave native state
    /// alive; callback-backed native dependencies must be released before
    /// destruction can succeed, even when an ambiguous driver failure prevents
    /// recovery.
    ///
    /// # Errors
    /// `Busy` leaves the session unchanged while an activated Device borrows it.
    /// Once native cleanup starts, an error preserves only the remaining work
    /// and the session may be used only for another destroy attempt. Native
    /// close errors retain their cause; consumed native handles are never
    /// replayed.
    pub fn destroy(&mut self) -> Result<(), Error> {
        self.inner.destroy()
    }

    /// Allocates host-only storage under either native lifetime policy.
    /// `size` is a nonzero multiple of
    /// [`host_page_size`](crate::memory::host_page_size); `alignment` is a power
    /// of two at least that large. The backend may reserve a larger native
    /// extent while preserving the requested logical range. This owner needs no
    /// activated device or session connection and retains only its storage and
    /// allocator.
    ///
    /// # Errors
    /// Rejects invalid extents or a controller whose teardown has begun. Metadata
    /// exhaustion and native allocation failure leave no published owner; native
    /// errors retain their original cause.
    pub fn allocate_host(&self, size: u64, alignment: u64) -> Result<HostAllocation, Error> {
        Ok(HostAllocation {
            inner: self.inner.allocate_host(size, alignment)?,
        })
    }

    /// Visits one generation-consistent set of passive endpoint records. The
    /// backend stages records with the session allocator before calling
    /// `visitor`; enumeration does not activate an endpoint or acquire its
    /// execution and memory resources. A platform with no supported endpoints
    /// yields an empty enumeration.
    ///
    /// # Errors
    /// Reports malformed or unsupported native metadata, discovery errors,
    /// allocation failure, or topology churn across the backend's consistency
    /// checks. Those failures call no visitor. A visitor error stops delivery
    /// after any records already visited and is returned unchanged.
    pub fn enumerate(
        &self,
        visitor: &mut dyn FnMut(Endpoint) -> Result<(), Error>,
    ) -> Result<(), Error> {
        self.inner.enumerate(visitor)
    }

    /// Reads the exact native endpoint selected by `id` and verifies its
    /// identity. Other endpoints are not enumerated, and no execution endpoint
    /// is acquired.
    ///
    /// # Errors
    /// Reports removal or identity mismatch, inconsistent discovery state,
    /// malformed metadata, or the native error from the selected endpoint. A
    /// controller whose teardown has begun rejects the query.
    pub fn open_endpoint(&self, id: [u8; 16]) -> Result<Endpoint, Error> {
        self.inner.open_endpoint(id)
    }

    /// Reserves one process virtual-address range common to every supplied
    /// activated device. A nonzero requested address is a hint and may fall
    /// back to another address in the common aperture.
    ///
    /// # Errors
    /// Rejects an empty or cross-session device set, incompatible apertures,
    /// invalid page-aligned extents, or native address-space exhaustion.
    pub fn reserve_virtual_address(
        &self,
        devices: &[&Device],
        size: u64,
        alignment: u64,
        address: u64,
    ) -> Result<VirtualAddress, Error> {
        let mut bounds: Option<(u64, u64)> = None;
        for device in devices {
            if !Shared::ptr_eq(&self.inner.driver, &device.driver) {
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
        let owner = Shared::try_new_uninit(self.inner.driver.allocator())?;
        let intervals = Shared::new(
            HostIntervals::new(self.inner.driver.allocator()),
            self.inner.driver.allocator(),
        )?;
        let device_intervals = Shared::new(
            DeviceIntervals::new(self.inner.driver.allocator()),
            self.inner.driver.allocator(),
        )?;
        let inner = self
            .inner
            .driver
            .reserve_virtual_address(bounds, size, alignment, address)?;
        Ok(VirtualAddress {
            inner: ProviderVirtualAddress::new(
                self.inner.driver.clone(),
                owner.write(inner),
                intervals,
                device_intervals,
            ),
        })
    }

    /// Revalidates a passive endpoint and acquires the native state required for
    /// device operations. The returned [`Device`] borrows this controller. A
    /// backend may retain reusable bindings for later activation, subject to
    /// this session's lifetime policy.
    ///
    /// # Errors
    /// Reports changed endpoint metadata, a lifetime policy the backend cannot
    /// honor, unsupported native interfaces or address-space layouts,
    /// incompatible existing ownership, allocation failure, or native I/O.
    /// Partial native setup remains owned for safe cleanup or a later retry.
    pub fn activate(&self, endpoint: &Endpoint) -> Result<Device, Error> {
        let activated = self.inner.activate(endpoint)?;
        let copy_pool = activated
            .endpoint
            .gpu()
            .map(|_| {
                Shared::new(
                    crate::gpu::CopyResourcePool::default(),
                    activated.driver.allocator(),
                )
            })
            .transpose()?;
        Ok(Device {
            driver: activated.driver,
            state: activated.state,
            endpoint: activated.endpoint,
            copy_pool,
        })
    }

    /// Lifetime policy used when qualifying device services.
    #[must_use]
    pub fn state_lifetime(&self) -> SessionLifetime {
        self.inner.lifetime
    }

    /// Returns whether this backend can register caller-owned host pages for
    /// the selected endpoint and native lifetime. This cached qualification
    /// does not activate the device; a specific registration can still fail
    /// if its caller pages cannot be bound by the native driver.
    #[must_use]
    pub fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        self.inner.supports_host_registration(endpoint)
    }
}

#[cfg(all(test, target_os = "linux"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn live_reservation_lease_blocks_release_until_its_mapping_owner_finishes() {
        let mut session = Session::new(SessionLifetime::Session).unwrap();
        // A process VA reservation needs no GPU endpoint, so this exercises the
        // public owner and its native cleanup on CPU-only test hosts.
        let inner = session
            .inner
            .driver
            .reserve_virtual_address((0x1_0000, isize::MAX as u64), 4096, 4096, 0)
            .unwrap();
        let info = driver::VirtualAddressOwnerInfo::cached_info(&*inner);
        let mut address = VirtualAddress {
            inner: ProviderVirtualAddress::new(
                session.inner.driver.clone(),
                Shared::new(inner, session.inner.driver.allocator()).unwrap(),
                Shared::new(
                    HostIntervals::new(session.inner.driver.allocator()),
                    session.inner.driver.allocator(),
                )
                .unwrap(),
                Shared::new(
                    DeviceIntervals::new(session.inner.driver.allocator()),
                    session.inner.driver.allocator(),
                )
                .unwrap(),
            ),
        };
        let mapping_lease = address.inner.inner.clone();
        assert_eq!(address.free().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(address.info(), info);
        drop(mapping_lease);
        address.free().unwrap();
        drop(address);
        session.destroy().unwrap();
    }
}
