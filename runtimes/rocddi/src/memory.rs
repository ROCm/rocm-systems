//! Memory placement, ownership, mapping, sharing, and host-cache services.
//!
//! Public owners retain the exact native dependencies required for teardown.
//! The platform driver supplies backing and mappings; this module enforces
//! cross-session ownership and range validation. Platform-specific sharing
//! transports live under [`interop`] so native handles do not leak into the
//! implementation-neutral allocation model.

mod access;
pub use access::DeviceAccess;
pub mod interop;
mod types;
pub(crate) use types::{AllocationDesc, AllocationLimits};

use crate::device::Device;
use crate::driver::{self, AllocationDriver, DeviceDriver, HostDriver, VirtualMemoryDriver};
use crate::gpu::GpuDevice;
use crate::host_storage::{Buffer, Owned, Shared};
use crate::{Error, ErrorKind};

/// Host mapping cache policy qualified by the native backend.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostCacheability {
    /// Ordinary write-back host mapping.
    WriteBack,
    /// Write-combined host mapping.
    WriteCombined,
}

/// Platform-neutral backing and placement requested from the native backend.
///
/// These variants describe ownership, visibility, and address behavior that a
/// caller can observe. They deliberately do not encode operating-system handle
/// types or driver allocation mechanisms; each platform backend selects the
/// native mechanism that satisfies the requested contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MemoryKind {
    /// Platform-managed, host-accessible system backing with device access
    /// established in the activated address space. The native platform memory
    /// manager owns the physical backing; rocddi owns its returned allocation
    /// and persistent host mapping. Host-only allocations instead use
    /// [`Session::allocate_host`](crate::session::Session::allocate_host).
    System,
    /// rocddi-owned write-back host pages made accessible to the device at the
    /// same host and device virtual address. The backend may pin, register, or
    /// otherwise bind those pages without exposing that native mechanism.
    OwnedHost,
    /// Caller-owned host pages made accessible to the device. `address` is the
    /// logical host base and may be subpage aligned. The caller keeps the
    /// complete page cover live until [`Allocation::free`] succeeds.
    RegisteredHost {
        /// Borrowed logical host address; ownership remains with the caller.
        address: usize,
        /// Bypass device caches for this registration.
        uncached: bool,
    },
    /// Device-local storage; host visibility is an explicit requirement.
    DeviceLocal {
        /// Require a persistent host mapping of the local storage.
        host_visible: bool,
        /// Request coherent device mappings for fine-grained access.
        coherent: bool,
        /// Bypass device caches for this allocation.
        uncached: bool,
        /// Request physically contiguous backing.
        contiguous: bool,
    },
}

/// Cached addresses and extents of one successfully mapped native allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AllocationInfo {
    /// Address in the activated device's native address space.
    pub device_address: u64,
    /// Existing host mapping when requested and supported.
    pub host_address: Option<usize>,
    /// Exact requested bytes.
    pub size: u64,
    /// Complete native physical allocation or imported backing extent.
    pub native_size: u64,
    /// Stable native identity for imported or shareable backing, when available.
    pub physical_backing_id: [u64; 2],
}

/// Immutable facts describing one detached virtual-memory allocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualMemoryInfo {
    /// Complete physical backing extent.
    pub size: u64,
    /// Required alignment and size multiple for mappings of this backing.
    pub mapping_granularity: u64,
    /// Stable native identity shared by equivalent handles for this backing.
    pub physical_backing_id: [u64; 2],
}

/// Cached facts for one reserved process virtual-address range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VirtualAddressInfo {
    /// First byte in the reserved range.
    pub address: u64,
    /// Exact reserved extent.
    pub size: u64,
    /// Required alignment and size multiple for mappings in this reservation.
    pub mapping_granularity: u64,
}

/// Owns a process virtual-address reservation used by detached memory mappings.
pub struct VirtualAddress {
    pub(crate) driver: Shared<driver::PlatformDriver>,
    pub(crate) inner: Shared<Owned<driver::NativeVirtualAddress>>,
    pub(crate) info: VirtualAddressInfo,
}

impl VirtualAddress {
    /// Returns the immutable base and extent of this reservation.
    #[must_use]
    pub fn info(&self) -> VirtualAddressInfo {
        self.info
    }

    /// Releases an unused address range. All device and host mapping owners
    /// that refer to this range must have been freed first.
    ///
    /// # Errors
    /// Reports a native release failure while retaining the owner for retry.
    pub fn free(&mut self) -> Result<(), Error> {
        let inner = Shared::get_mut(&mut self.inner).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "virtual-address reservation has live mappings",
        })?;
        driver::PlatformDriver::free_virtual_address(inner)
    }
}

/// Owns detached physical backing used by virtual-memory mappings.
pub struct VirtualMemory {
    pub(crate) driver: Shared<driver::PlatformDriver>,
    pub(crate) inner: Shared<Owned<driver::NativeVirtualMemory>>,
    pub(crate) info: VirtualMemoryInfo,
}

impl VirtualMemory {
    /// Returns immutable backing facts.
    #[must_use]
    pub fn info(&self) -> VirtualMemoryInfo {
        self.info
    }

    /// Maps a subrange of this backing into the process host page tables at a
    /// range owned by `reservation` with exactly the requested permissions.
    ///
    /// # Errors
    /// Rejects cross-session or out-of-range mappings and reports a native
    /// host-mapping failure.
    pub fn map_host(
        &self,
        reservation: &VirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<VirtualHostMapping, Error> {
        if !Shared::ptr_eq(&self.driver, &reservation.driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "virtual-memory owners must belong to one session",
            });
        }
        validate_virtual_mapping(self.info, reservation.info, address, offset, size)?;
        let inner = driver::PlatformDriver::map_virtual_host(
            &self.inner,
            &reservation.inner,
            address,
            offset,
            size,
            permissions,
            self.inner.allocator(),
        )?;
        Ok(VirtualHostMapping {
            inner,
            driver: Some(self.driver.clone()),
            reservation: Some(reservation.inner.clone()),
            backing: Some(self.inner.clone()),
        })
    }

    /// Releases detached physical backing after every mapping has been freed.
    ///
    /// # Errors
    /// Reports native cleanup failure while retaining unfinished ownership.
    pub fn free(&mut self) -> Result<(), Error> {
        let inner = Shared::get_mut(&mut self.inner).ok_or(Error::Operation {
            kind: ErrorKind::Busy,
            detail: "virtual-memory backing has live mappings",
        })?;
        driver::PlatformDriver::free_virtual_memory(inner)
    }
}

/// Owns one virtual-memory mapping in an activated device VM.
pub struct VirtualDeviceMapping {
    pub(crate) inner: Owned<driver::NativeVirtualDeviceMapping>,
    driver: Option<Shared<driver::PlatformDriver>>,
    reservation: Option<Shared<Owned<driver::NativeVirtualAddress>>>,
    backing: Option<Shared<Owned<driver::NativeVirtualMemory>>>,
}

impl VirtualDeviceMapping {
    /// Removes the device mapping and releases its imported native-memory
    /// reference.
    ///
    /// # Errors
    /// Reports an unmap or handle-release failure while retaining cleanup state.
    pub fn free(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::free_virtual_device_mapping(&mut self.inner)?;
        self.reservation = None;
        self.backing = None;
        self.driver = None;
        Ok(())
    }
}

impl Drop for VirtualDeviceMapping {
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            if let Some(backing) = self.backing.take() {
                std::mem::forget(backing);
            }
            if let Some(driver) = self.driver.take() {
                std::mem::forget(driver);
            }
        }
    }
}

/// Owns one virtual-memory mapping in the process host page tables.
pub struct VirtualHostMapping {
    pub(crate) inner: Owned<driver::NativeVirtualHostMapping>,
    driver: Option<Shared<driver::PlatformDriver>>,
    reservation: Option<Shared<Owned<driver::NativeVirtualAddress>>>,
    backing: Option<Shared<Owned<driver::NativeVirtualMemory>>>,
}

impl VirtualHostMapping {
    /// Removes the host mapping while preserving the encompassing address
    /// reservation for reuse.
    ///
    /// # Errors
    /// Reports a native mapping failure while retaining cleanup state.
    pub fn free(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::free_virtual_host_mapping(&mut self.inner)?;
        self.reservation = None;
        self.backing = None;
        self.driver = None;
        Ok(())
    }
}

impl Drop for VirtualHostMapping {
    fn drop(&mut self) {
        if self.free().is_err() {
            if let Some(reservation) = self.reservation.take() {
                std::mem::forget(reservation);
            }
            if let Some(backing) = self.backing.take() {
                std::mem::forget(backing);
            }
            if let Some(driver) = self.driver.take() {
                std::mem::forget(driver);
            }
        }
    }
}

pub(crate) fn validate_virtual_mapping(
    memory: VirtualMemoryInfo,
    reservation: VirtualAddressInfo,
    address: u64,
    offset: u64,
    size: u64,
) -> Result<(), Error> {
    if !memory.mapping_granularity.is_power_of_two()
        || !reservation.mapping_granularity.is_power_of_two()
        || memory.size == 0
        || memory.size % memory.mapping_granularity != 0
        || reservation.address % reservation.mapping_granularity != 0
        || reservation.size == 0
        || reservation.size % reservation.mapping_granularity != 0
    {
        return Err(Error::Operation {
            kind: ErrorKind::DriverContract,
            detail: "virtual-memory owner has invalid mapping granularity",
        });
    }
    let granularity = memory
        .mapping_granularity
        .max(reservation.mapping_granularity);
    let mapping_end = address.checked_add(size).ok_or(Error::Operation {
        kind: ErrorKind::InvalidArgument,
        detail: "virtual-memory mapping address overflows",
    })?;
    let reservation_end =
        reservation
            .address
            .checked_add(reservation.size)
            .ok_or(Error::Operation {
                kind: ErrorKind::DriverContract,
                detail: "virtual-address reservation extent overflows",
            })?;
    let backing_end = offset.checked_add(size).ok_or(Error::Operation {
        kind: ErrorKind::InvalidArgument,
        detail: "virtual-memory backing offset overflows",
    })?;
    if size == 0
        || address % granularity != 0
        || offset % granularity != 0
        || size % granularity != 0
        || address < reservation.address
        || mapping_end > reservation_end
        || backing_end > memory.size
    {
        return Err(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "virtual-memory mapping is outside its aligned owners",
        });
    }
    Ok(())
}

/// Owns one native allocation, its exact VM dependencies, and cleanup progress.
/// Explicit free reports failures. If final Drop cannot complete cleanup, it
/// retains potentially reachable native backing instead of recycling its VA.
pub struct Allocation {
    pub(crate) inner: Owned<driver::NativeAllocation>,
    pub(crate) info: AllocationInfo,
}

impl Allocation {
    /// Returns creation-time addresses and extent without touching the driver.
    /// The adapter controls their public lifetime. This snapshot remains cached
    /// after cleanup begins, when its addresses must no longer be used.
    #[must_use]
    pub fn info(&self) -> AllocationInfo {
        self.info
    }
    /// Checks native loss and whether this allocation still permits access.
    /// No mapping is added and no cached address is refreshed.
    ///
    /// # Errors
    /// Returns `DeviceLost` for latched loss, `Unsupported` once freeing has begun
    /// or the mapping is unavailable, and the native cause if observation fails.
    pub fn check(&self) -> Result<(), Error> {
        driver::PlatformDriver::check_allocation(&self.inner)
    }

    /// Returns the established address for one device in this allocation's
    /// immutable access set.
    ///
    /// # Errors
    /// Returns `Unsupported` if the device has no mapping, `DeviceLost` for a
    /// latched native loss, or the native error from the availability check.
    pub fn device_address(&self, device: &Device) -> Result<u64, Error> {
        driver::PlatformDriver::allocation_device_address(&self.inner, &device.state)
    }

    /// Returns whether this allocation's physical backing originated on the
    /// supplied device. Devices from another session never match.
    #[must_use]
    pub fn originates_from(&self, device: &Device) -> bool {
        driver::PlatformDriver::allocation_is_owned_by(&self.inner, &device.state)
    }

    /// Returns opaque provider metadata retained with an imported native
    /// resource. Its format is defined by the interop operation that created
    /// the allocation; ordinary allocations return an empty slice.
    #[must_use]
    pub fn metadata(&self) -> &[u8] {
        self.inner.metadata()
    }
    /// Releases device mappings, the native allocation, and then its virtual
    /// address reservation.
    /// The adapter must first ensure all host mappings and future device uses
    /// have ended. Cleanup performs no execution wait or cache transition, and
    /// resumes only the unfinished native steps after a retryable failure.
    ///
    /// # Errors
    /// Returns the failing native cleanup operation without discarding remaining
    /// ownership. Ambiguous native outcomes return `DriverContract` and retain
    /// backing for process teardown; retrying cannot make an unsafe replay safe.
    /// Cached addresses must not be used once freeing has begun.
    pub fn free(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::free_allocation(&mut self.inner)
    }
}

/// Cached host-only storage facts.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HostAllocationInfo {
    /// Host address of the mapped storage.
    pub host_address: usize,
    /// Requested mapped bytes.
    pub size: u64,
}
/// Owns host-only storage and the allocator used for its ownership record.
/// It has no activated-device or session-connection dependency. The adapter
/// still enforces its public scope and mapping lifetimes, and callback state
/// must outlive this owner. Both native lifetime policies use the same host
/// cleanup path.
pub struct HostAllocation {
    pub(crate) inner: Owned<driver::NativeHostAllocation>,
    pub(crate) info: HostAllocationInfo,
}

impl HostAllocation {
    /// Copies the original host address and native extent without a system call.
    /// Freeing the storage does not rewrite this snapshot; its address is valid
    /// only until cleanup begins.
    #[must_use]
    pub fn info(&self) -> HostAllocationInfo {
        self.info
    }
    /// Releases the host mapping after all accesses through its address have ended.
    /// The adapter owns logical mapping borrows and must discharge them first.
    /// Repeated successful cleanup is harmless; no device operation is involved.
    ///
    /// # Errors
    /// A native unmapping error leaves the reservation owned for retry.
    /// Inherited process state is rejected before unmapping. This operation
    /// allocates no metadata.
    pub fn free(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::free_host(&mut self.inner)
    }
}
/// Returns the native host virtual-memory page size for allocation rounding.
/// This query does not acquire an endpoint and is independent of any device
/// queue or address-space page layout.
///
/// # Errors
/// Returns a native error if the platform does not report a valid page size.
pub fn host_page_size() -> Result<u64, Error> {
    driver::PlatformDriver::host_page_size()
}

/// Qualifies explicit host-cache maintenance and returns its line granularity.
/// The adapter uses success to advertise a mapping's maintenance recipe. A
/// backend qualifies both an available maintenance operation and its exact
/// granularity; this call opens no native endpoint and allocates no metadata.
///
/// # Errors
/// Returns `Unsupported` when the target has no implemented recipe or lacks the
/// required instruction and valid line size. Callers must not advertise a cache
/// operation using an unqualified nominal hardware line size.
pub fn host_cache_line_size() -> Result<u32, Error> {
    driver::PlatformDriver::host_cache_line_size()
}
/// Executes the qualified host-cache maintenance recipe over a nonempty host
/// range. The backend provides the writeback, invalidation, and ordering
/// sequence required by the current host architecture. The adapter checks
/// mapping permissions and handles any public empty-range no-op before calling
/// this bridge. This operation neither waits for device work nor supplies its
/// retirement. A zero `line_size` selects an ordering-only recipe for a
/// write-combined view; nonzero values select the qualified write-back recipe.
///
/// # Errors
/// Rejects an unsupported recipe, a line size different from the qualified
/// value, a null address, an empty range, or numeric address overflow before
/// executing cache instructions. Address validity itself is a caller obligation.
///
/// # Safety
/// Every intersecting cache line, including bytes outside the logical range,
/// must remain mapped through the call. The caller must externally synchronize
/// access to those complete lines and prevent concurrent unmapping.
#[allow(unsafe_code)]
pub unsafe fn host_cache_control(pointer: usize, length: u64, line_size: u32) -> Result<(), Error> {
    // SAFETY: The public caller provides the live mapped range required by the
    // private native boundary; the backend validates the numeric extent.
    unsafe { driver::PlatformDriver::host_cache_control(pointer, length, line_size) }
}

impl Device {
    /// Returns the current bytes available for allocation on this device.
    #[doc(hidden)]
    pub fn available_memory(&self) -> Result<u64, Error> {
        self.driver.available_memory(&self.state)
    }

    /// Creates or registers backing and establishes its device mapping before
    /// returning. `kind` selects owned system/local placement or borrowed caller
    /// pages; permissions are never widened. The native extent is page-aligned
    /// and separate from the logical byte range maintained by the adapter. The
    /// owner retains its address-space and platform dependencies, and takes its
    /// metadata allocator from that address space. Registered pages remain
    /// caller-owned and live through successful [`Allocation::free`].
    ///
    /// # Errors
    /// Rejects unsupported permissions or placement and invalid extents before
    /// mutation. Native allocation, host mapping, device mapping, or loss checks
    /// can then fail. Partial mapping progress is preserved during rollback; an
    /// ambiguous kernel result retains any possibly reachable backing.
    pub fn allocate(
        &self,
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        let inner = self
            .driver
            .allocate(&self.state, &[], kind, size, alignment, permissions)?;
        let info = inner.cached_info();
        Ok(Allocation { inner, info })
    }

    /// Creates device-local backing inside this process's native scratch
    /// aperture. The returned owner keeps both the physical allocation and its
    /// aperture range live until [`Allocation::free`] succeeds.
    ///
    /// # Errors
    /// Rejects invalid extents or unavailable local storage. Native scratch-base
    /// programming, allocation, mapping, and loss checks may also fail.
    pub(crate) fn allocate_queue_scratch(&self, size: u64) -> Result<Allocation, Error> {
        let inner = self.driver.allocate_queue_scratch(&self.state, size)?;
        let info = inner.cached_info();
        Ok(Allocation { inner, info })
    }

    /// Maps the device's process-level MMIO remap page.
    #[doc(hidden)]
    pub(crate) fn map_mmio_remap(&self) -> Result<Allocation, Error> {
        let inner = self.driver.map_mmio_remap(&self.state)?;
        let info = inner.cached_info();
        Ok(Allocation { inner, info })
    }

    /// Creates detached physical backing for later virtual-address mappings.
    /// The allocation has no usable device or host address until a mapping
    /// owner is created with explicit access permissions.
    ///
    /// # Errors
    /// Rejects unsupported placement or invalid extents and reports native
    /// allocation or native share-handle creation failures.
    pub fn create_virtual_memory(
        &self,
        kind: MemoryKind,
        size: u64,
        pinned: bool,
        uncached: bool,
    ) -> Result<VirtualMemory, Error> {
        let owner = Shared::try_new_uninit(self.driver.allocator())?;
        let inner = self
            .driver
            .create_virtual_memory(&self.state, kind, size, pinned, uncached)?;
        let info = inner.info();
        Ok(VirtualMemory {
            driver: self.driver.clone(),
            inner: owner.write(inner),
            info,
        })
    }

    /// Maps a subrange of detached physical backing into this device's VM at a
    /// range owned by `reservation` with exactly the requested permissions.
    ///
    /// # Errors
    /// Rejects cross-session owners, ranges outside either owner, invalid
    /// permissions, device loss, or native import and mapping failures.
    pub fn map_virtual_memory(
        &self,
        memory: &VirtualMemory,
        reservation: &VirtualAddress,
        address: u64,
        offset: u64,
        size: u64,
        permissions: DeviceAccess,
    ) -> Result<VirtualDeviceMapping, Error> {
        if !Shared::ptr_eq(&self.driver, &memory.driver)
            || !Shared::ptr_eq(&self.driver, &reservation.driver)
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "virtual-memory owners must belong to one session",
            });
        }
        validate_virtual_mapping(memory.info, reservation.info, address, offset, size)?;
        let inner = driver::PlatformDriver::map_virtual_device(
            &memory.inner,
            &reservation.inner,
            &self.state,
            address,
            offset,
            size,
            permissions,
        )?;
        Ok(VirtualDeviceMapping {
            inner,
            driver: Some(self.driver.clone()),
            reservation: Some(reservation.inner.clone()),
            backing: Some(memory.inner.clone()),
        })
    }

    /// Creates or registers one allocation mapped into this device and every
    /// peer address space before returning. All devices must belong to this
    /// core session, and the native backend establishes one common device
    /// virtual address. Repeated live-device wrappers sharing one native
    /// address space reuse that mapping. Device-local placement additionally
    /// requires a cached directional route from every peer to this physical
    /// owner. Per-device permission differences are not representable by the
    /// current native path.
    ///
    /// # Errors
    /// Rejects a session mismatch, differing permissions, unsupported
    /// placement, or an empty common address envelope before native allocation.
    /// Native acquisition and rollback failures have the same ownership behavior
    /// as [`Self::allocate`].
    pub fn allocate_with_peers(
        &self,
        peers: &[&Self],
        kind: MemoryKind,
        size: u64,
        alignment: u64,
        permissions: DeviceAccess,
    ) -> Result<Allocation, Error> {
        for peer in peers {
            if !Shared::ptr_eq(&self.driver, &peer.driver) {
                return Err(Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "peer devices must belong to one session",
                });
            }
            if matches!(kind, MemoryKind::DeviceLocal { .. })
                && !peer.endpoint.can_access_local_memory(&self.endpoint)
            {
                return Err(Error::Operation {
                    kind: ErrorKind::Unsupported,
                    detail: "peer device has no qualified route to local memory",
                });
            }
        }
        let mut states = Buffer::try_with_capacity(peers.len(), self.driver.allocator())?;
        for peer in peers {
            if self.state.shares_vm(&peer.state)
                || states
                    .iter()
                    .any(|state: &&driver::DeviceState| state.shares_vm(&peer.state))
            {
                continue;
            }
            states.try_push(&peer.state)?;
        }
        let inner = self.driver.allocate(
            &self.state,
            states.as_slice(),
            kind,
            size,
            alignment,
            permissions,
        )?;
        let info = inner.cached_info();
        Ok(Allocation { inner, info })
    }
}

impl GpuDevice<'_> {
    /// Creates device-local backing inside this GPU's native scratch aperture.
    /// The returned owner keeps both the physical allocation and its aperture
    /// range live until [`Allocation::free`] succeeds.
    ///
    /// # Errors
    /// Rejects invalid extents or unavailable local storage. Native
    /// scratch-base programming, allocation, mapping, and loss checks may also
    /// fail.
    pub fn allocate_queue_scratch(&self, size: u64) -> Result<Allocation, Error> {
        self.device.allocate_queue_scratch(size)
    }

    /// Maps this GPU's process-level MMIO remap page.
    ///
    /// This is a GPU transport capability rather than a universal device
    /// memory operation. Callers must keep the returned allocation alive for
    /// every use of addresses derived from the mapping.
    ///
    /// # Errors
    /// Returns `Unsupported` when the backend or GPU exposes no MMIO remap page,
    /// and otherwise reports native allocation or mapping failures.
    pub fn map_mmio_remap(&self) -> Result<Allocation, Error> {
        self.device.map_mmio_remap()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn virtual_mapping_ranges_must_fit_both_owners() {
        let memory = VirtualMemoryInfo {
            size: 0x4000,
            mapping_granularity: 0x1000,
            physical_backing_id: [1, 2],
        };
        let reservation = VirtualAddressInfo {
            address: 0x1_0000,
            size: 0x8000,
            mapping_granularity: 0x1000,
        };

        assert!(validate_virtual_mapping(memory, reservation, 0x1_1000, 0x2000, 0x2000).is_ok());
        for (address, offset, size) in [
            (0x1_1001, 0x2000, 0x1000),
            (0x1_1000, 0x2001, 0x1000),
            (0x1_1000, 0x2000, 0),
            (0x0_f000, 0, 0x1000),
            (0x1_7000, 0, 0x2000),
            (0x1_1000, 0x3000, 0x2000),
        ] {
            assert_eq!(
                validate_virtual_mapping(memory, reservation, address, offset, size)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidArgument
            );
        }
    }

    #[test]
    fn virtual_mapping_uses_the_stricter_owner_granularity() {
        let memory = VirtualMemoryInfo {
            size: 0x2_0000,
            mapping_granularity: 0x1_0000,
            physical_backing_id: [1, 2],
        };
        let reservation = VirtualAddressInfo {
            address: 0x10_0000,
            size: 0x4_0000,
            mapping_granularity: 0x1000,
        };

        assert!(validate_virtual_mapping(memory, reservation, 0x10_0000, 0, 0x1_0000).is_ok());
        assert_eq!(
            validate_virtual_mapping(memory, reservation, 0x10_1000, 0, 0x1_0000)
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );

        let invalid = VirtualMemoryInfo {
            mapping_granularity: 0,
            ..memory
        };
        assert_eq!(
            validate_virtual_mapping(invalid, reservation, 0x10_0000, 0, 0x1_0000)
                .unwrap_err()
                .kind(),
            ErrorKind::DriverContract
        );

        let invalid = VirtualAddressInfo {
            address: reservation.address + 1,
            ..reservation
        };
        assert_eq!(
            validate_virtual_mapping(memory, invalid, 0x10_0000, 0, 0x1_0000)
                .unwrap_err()
                .kind(),
            ErrorKind::DriverContract
        );
    }
}
