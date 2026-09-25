//! Session lifetime and root coordination for the rocddi interface.
//!
//! A session owns the selected platform driver and coordinates passive
//! discovery, explicit activation, and resources whose scope spans more than
//! one device. Domain-specific value types and resource owners live in their
//! respective modules rather than sharing this lifecycle namespace.

use crate::device::Device;
use crate::driver::{self, HostDriver, ProviderDriver, VirtualMemoryDriver};
use crate::host_storage::{Allocator, Shared};
use crate::memory::{HostAllocation, VirtualAddress, VirtualAddressInfo};
use crate::topology::Endpoint;
use crate::{Error, ErrorKind};

/// Linux descriptor calls shared by native-handle ABI adapters.
#[cfg(target_os = "linux")]
pub mod linux {
    use std::io;

    /// Reads the current length of a borrowed descriptor without closing it.
    ///
    /// # Errors
    /// Returns the native descriptor or metadata error.
    pub fn descriptor_length(descriptor: i32) -> io::Result<u64> {
        crate::driver::descriptor_length(descriptor)
    }

    /// Reads exactly `bytes.len()` bytes from a borrowed descriptor at `offset`.
    ///
    /// # Errors
    /// Returns a native read error or `UnexpectedEof` for a short file.
    pub fn read_descriptor_exact(descriptor: i32, bytes: &mut [u8], offset: i64) -> io::Result<()> {
        crate::driver::read_descriptor_exact(descriptor, bytes, offset)
    }

    /// Closes one descriptor whose ownership was transferred by the caller.
    ///
    /// # Errors
    /// Returns the native close error. Linux may still consume the descriptor.
    pub fn close_descriptor(descriptor: i32) -> io::Result<()> {
        crate::driver::close_descriptor(descriptor)
    }

    /// Reads one chunk at `offset` without changing the descriptor position.
    ///
    /// # Safety
    /// `address..address + size` must be a live writable host range for the
    /// duration of the call, and `descriptor` must remain open.
    ///
    /// # Errors
    /// Returns an invalid argument or native read error.
    #[allow(unsafe_code)]
    pub unsafe fn read_descriptor(
        descriptor: i32,
        address: usize,
        size: usize,
        offset: i64,
    ) -> io::Result<usize> {
        // SAFETY: The caller upholds the range and descriptor contract.
        unsafe { crate::driver::read_descriptor(descriptor, address, size, offset) }
    }

    /// Writes one chunk at `offset` without changing the descriptor position.
    ///
    /// # Safety
    /// `address..address + size` must be a live readable host range for the
    /// duration of the call, and `descriptor` must remain open.
    ///
    /// # Errors
    /// Returns an invalid argument or native write error.
    #[allow(unsafe_code)]
    pub unsafe fn write_descriptor(
        descriptor: i32,
        address: usize,
        size: usize,
        offset: i64,
    ) -> io::Result<usize> {
        // SAFETY: The caller upholds the range and descriptor contract.
        unsafe { crate::driver::write_descriptor(descriptor, address, size, offset) }
    }

    #[cfg(test)]
    #[allow(clippy::unwrap_used)]
    mod tests {
        use super::*;
        use std::io::Write;
        use std::os::fd::AsRawFd;

        #[test]
        fn exact_read_preserves_the_borrowed_descriptor() {
            let path = std::env::temp_dir().join(format!(
                "rocddi-descriptor-{}-{:?}",
                std::process::id(),
                std::thread::current().id()
            ));
            let mut file = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&path)
                .unwrap();
            std::fs::remove_file(path).unwrap();
            file.write_all(b"ABCD").unwrap();
            let descriptor = file.as_raw_fd();
            assert_eq!(descriptor_length(descriptor).unwrap(), 4);
            let mut bytes = [0; 2];
            read_descriptor_exact(descriptor, &mut bytes, 1).unwrap();
            assert_eq!(&bytes, b"BC");
            assert_eq!(
                read_descriptor_exact(descriptor, &mut bytes, 3)
                    .unwrap_err()
                    .kind(),
                io::ErrorKind::UnexpectedEof
            );
            assert_eq!(file.metadata().unwrap().len(), 4);
        }
    }
}

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

/// Owns native connections and reusable device bindings for one core session.
/// Construction allocates only the controller record. Both lifetime policies
/// permit passive queries and host storage; only an explicit activation
/// acquires endpoint state. Dropping a [`Device`] leaves any backend-retained
/// binding available for recreation within this session.
pub struct Session {
    driver: Shared<driver::PlatformDriver>,
    lifetime: SessionLifetime,
}

impl Clone for Session {
    fn clone(&self) -> Self {
        Self {
            driver: self.driver.clone(),
            lifetime: self.lifetime,
        }
    }
}

impl Session {
    /// Borrows the private native controller for domain-specific crate layers.
    pub(crate) fn driver(&self) -> &Shared<driver::PlatformDriver> {
        &self.driver
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
            driver: Shared::new(driver::PlatformDriver::new(allocator), allocator)?,
            lifetime,
        })
    }

    /// Releases native connection owners in dependency order without waiting
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
        let driver = Shared::get_mut(&mut self.driver).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "activated devices still borrow this session",
        })?;
        driver.shutdown()
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
        let inner = self.driver.allocate_host(size, alignment)?;
        let info = inner.cached_info();
        Ok(HostAllocation { inner, info })
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
        self.driver.enumerate(visitor)
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
        self.driver.open_endpoint(id)
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
            if !Shared::ptr_eq(&self.driver, &device.driver) {
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
        let owner = Shared::try_new_uninit(self.driver.allocator())?;
        let inner = self
            .driver
            .reserve_virtual_address(bounds, size, alignment, address)?;
        let info = VirtualAddressInfo {
            address: inner.address(),
            size: inner.size(),
            mapping_granularity: inner.mapping_granularity(),
        };
        Ok(VirtualAddress {
            driver: self.driver.clone(),
            inner: owner.write(inner),
            info,
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
        if endpoint.provider_instance != self.driver.provider_instance() {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "endpoint belongs to another provider instance",
            });
        }
        let state = self.driver.activate(endpoint, self.lifetime)?;
        Ok(Device {
            driver: self.driver.clone(),
            state,
            endpoint: endpoint.clone(),
        })
    }

    /// Lifetime policy used when qualifying device services.
    #[must_use]
    pub fn state_lifetime(&self) -> SessionLifetime {
        self.lifetime
    }

    /// Returns whether this backend can register caller-owned host pages for
    /// the selected endpoint and native lifetime. This cached qualification
    /// does not activate the device; a specific registration can still fail
    /// if its caller pages cannot be bound by the native driver.
    #[must_use]
    pub fn supports_host_registration(&self, endpoint: &Endpoint) -> bool {
        endpoint.provider_instance == self.driver.provider_instance()
            && self
                .driver
                .supports_host_registration(endpoint, self.lifetime)
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
            .driver
            .reserve_virtual_address((0x1_0000, isize::MAX as u64), 4096, 4096, 0)
            .unwrap();
        let info = VirtualAddressInfo {
            address: inner.address(),
            size: inner.size(),
            mapping_granularity: inner.mapping_granularity(),
        };
        let mut address = VirtualAddress {
            driver: session.driver.clone(),
            inner: Shared::new(inner, session.driver.allocator()).unwrap(),
            info,
        };
        let mapping_lease = address.inner.clone();
        assert_eq!(address.free().unwrap_err().kind(), ErrorKind::Busy);
        assert_eq!(address.info(), info);
        drop(mapping_lease);
        address.free().unwrap();
        drop(address);
        session.destroy().unwrap();
    }
}
