// SPDX-License-Identifier: MIT

//! Linux DMA-BUF import and export.
//!
//! DMA-BUF is a Linux file-descriptor transport, not a portable memory kind.
//! This module keeps descriptor ownership, duplication, physical-identity
//! checks, and KFD/DRM interop outside rocddi's platform-neutral memory model.

use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd, RawFd};
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;

use crate::device::Device;
use crate::driver::ProviderDriver;
use crate::driver::linux_interop::LinuxMemoryInteropDriver;
use crate::host_storage::{Buffer, Shared};
use crate::memory::{Allocation, DeviceAccess, VirtualMemory};
use crate::session::Session;
use crate::{Error, ErrorKind};

/// Closes a descriptor whose ownership the caller has transferred.
///
/// # Safety
/// If `descriptor` names an open file, the caller must own it and must not
/// use it after this call, including when close reports an error. An invalid
/// descriptor is permitted and reported by the kernel.
///
/// # Errors
/// Preserves the Linux close error, if any.
#[allow(unsafe_code)]
pub unsafe fn close_owned_descriptor(descriptor: RawFd) -> io::Result<()> {
    // SAFETY: The caller transfers descriptor ownership to this operation.
    unsafe { crate::driver::PlatformDriver::close_owned_descriptor(descriptor) }
}

/// Validates and duplicates a caller descriptor before constructing a Rust
/// borrowed descriptor. The returned owner is independent of the caller's fd.
///
/// # Errors
/// Reports an invalid or closed descriptor, or native descriptor exhaustion.
pub fn duplicate_descriptor(descriptor: RawFd) -> Result<OwnedFd, Error> {
    crate::driver::PlatformDriver::duplicate_descriptor(descriptor)
}

/// Returns the length of a borrowed descriptor without taking its ownership.
///
/// # Errors
/// Reports an invalid descriptor or native metadata failure.
pub fn descriptor_length(descriptor: RawFd) -> io::Result<u64> {
    crate::driver::PlatformDriver::descriptor_length(descriptor)
}

/// Resolves a reopenable filesystem path for a borrowed Linux descriptor.
///
/// The returned path owns its bytes and does not depend on the descriptor
/// remaining open. An anonymous or unlinked file has no such path. The caller
/// must keep the descriptor valid during this call.
///
/// # Errors
/// Reports an invalid descriptor or a missing or changed filesystem path.
pub fn descriptor_path(descriptor: RawFd) -> io::Result<PathBuf> {
    if descriptor < 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    let link = format!("/proc/self/fd/{descriptor}");
    let path = std::fs::read_link(&link)?;
    if !path.is_absolute() {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    let source = std::fs::metadata(link)?;
    let target = std::fs::metadata(&path)?;
    if (source.dev(), source.ino()) != (target.dev(), target.ino()) {
        return Err(io::Error::from(io::ErrorKind::InvalidData));
    }
    Ok(path)
}

/// Reads an entire fixed range from a borrowed descriptor without changing its
/// shared file position.
///
/// # Errors
/// Reports an invalid descriptor, short read, or native read failure.
pub fn read_descriptor_exact_at(
    descriptor: RawFd,
    buffer: &mut [u8],
    offset: u64,
) -> io::Result<()> {
    crate::driver::PlatformDriver::read_descriptor_exact_at(descriptor, buffer, offset)
}

/// Reads from a borrowed descriptor at a fixed offset.
///
/// # Errors
/// Preserves the Linux read error, including its errno.
pub fn read_descriptor_at(descriptor: RawFd, buffer: &mut [u8], offset: i64) -> io::Result<usize> {
    crate::driver::PlatformDriver::read_descriptor_at(descriptor, buffer, offset)
}

/// Writes to a borrowed descriptor at a fixed offset.
///
/// # Errors
/// Preserves the Linux write error, including its errno.
pub fn write_descriptor_at(descriptor: RawFd, buffer: &[u8], offset: i64) -> io::Result<usize> {
    crate::driver::PlatformDriver::write_descriptor_at(descriptor, buffer, offset)
}

/// Maximum bytes submitted in one AIS operation, matching Linux `MAX_RW_COUNT`.
pub const AIS_MAX_TRANSFER_BYTES: u64 = 0x7fff_f000;

/// Direction of a Linux AIS transfer between a file and device VRAM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AisFileOperation {
    /// Read file bytes into the allocation.
    Read,
    /// Write allocation bytes into the file.
    Write,
}

/// Completed KFD AIS transfer. A nonzero status is a negative Linux errno.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AisFileResult {
    /// Bytes KFD reports as copied, bounded by the submitted transfer size.
    pub size_copied: u64,
    /// Operation status returned by KFD, zero on success.
    pub status: i32,
}

/// Transfers between a borrowed file descriptor and live mapped VRAM.
///
/// `allocation_offset` is relative to the allocation's logical device address.
/// One native operation transfers at most Linux's `MAX_RW_COUNT` bytes. The
/// caller retains the descriptor through this synchronous call; neither its
/// ownership nor its shared file position changes. If the ioctl fails, the
/// copied-byte count is unknown and the operation is never replayed here.
///
/// # Errors
/// Rejects invalid ranges, descriptors, offsets, and unsupported backing.
/// Preserves the native errno if KFD reports an ioctl failure.
pub fn ais_transfer(
    allocation: &Allocation,
    descriptor: RawFd,
    allocation_offset: u64,
    size: u64,
    file_offset: i64,
    operation: AisFileOperation,
) -> Result<AisFileResult, Error> {
    crate::driver::PlatformDriver::ais_transfer(
        allocation.inner.native(),
        descriptor,
        allocation_offset,
        size,
        file_offset,
        operation,
    )
}

/// Opaque 256-bit KFD IPC identifier for one shareable allocation.
///
/// This layout belongs to the Linux KFD userspace contract. It deliberately
/// does not appear in rocddi's platform-neutral allocation API, where a Windows
/// backend may use a HANDLE-based shared-resource representation instead.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct KfdIpcMemoryHandle {
    words: [u32; 8],
}

impl KfdIpcMemoryHandle {
    /// Wraps the eight words supplied by a KFD-compatible public API.
    #[must_use]
    pub const fn from_words(words: [u32; 8]) -> Self {
        Self { words }
    }

    /// Returns the eight words carried by the KFD IPC contract.
    #[must_use]
    pub const fn words(self) -> [u32; 8] {
        self.words
    }
}

/// Location value accepted by Linux KFD SVM attribute operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KfdSvmLocation {
    /// Host system memory.
    System,
    /// No uniform or preferred location.
    Undefined,
    /// One activated endpoint, identified without exposing KFD's GPU ID.
    Device([u8; 16]),
}

/// Per-device accessibility accepted by Linux KFD SVM operations.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KfdSvmAccess {
    /// Faulting access with migration permitted.
    Accessible,
    /// Non-faulting access without migration.
    AccessibleInPlace,
    /// Access denied.
    NoAccess,
}

/// One Linux KFD SVM operation attribute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum KfdSvmAttribute {
    /// Preferred physical location.
    PreferredLocation(KfdSvmLocation),
    /// Immediate migration target or most recent prefetch location.
    PrefetchLocation(KfdSvmLocation),
    /// Access mode for one activated GPU endpoint.
    Access {
        /// Opaque rocddi endpoint identity.
        device: [u8; 16],
        /// Requested or returned access mode.
        access: KfdSvmAccess,
    },
    /// KFD SVM flags to set.
    SetFlags(u32),
    /// KFD SVM flags to clear.
    ClearFlags(u32),
    /// Base-two logarithm of the migration page count.
    MigrationGranularity(u32),
}

/// One owned DMA-BUF file descriptor plus immutable backing facts.
pub struct DmaBuf {
    descriptor: OwnedFd,
    info: DmaBufInfo,
}

impl DmaBuf {
    pub(crate) fn new(descriptor: OwnedFd, info: DmaBufInfo) -> Self {
        Self { descriptor, info }
    }

    /// Borrows the live file descriptor without transferring ownership.
    #[must_use]
    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.descriptor.as_fd()
    }

    /// Returns immutable physical backing facts.
    #[must_use]
    pub fn info(&self) -> DmaBufInfo {
        self.info
    }

    /// Transfers ownership of the DMA-BUF file descriptor to the caller.
    #[must_use]
    pub fn into_fd(self) -> OwnedFd {
        self.descriptor
    }
}

/// Immutable facts established from a live DMA-BUF file descriptor.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DmaBufInfo {
    /// Complete DMA-BUF physical backing extent.
    pub byte_length: u64,
    /// Byte offset of the exported logical allocation within that backing.
    pub source_offset: u64,
    /// File identity shared by duplicate descriptors for this live backing.
    pub physical_backing_id: [u64; 2],
}

/// Imports detached virtual-memory backing from a borrowed DMA-BUF.
///
/// The backend duplicates `descriptor` before returning, so the caller retains
/// ownership and may close its descriptor independently after this operation.
///
/// # Errors
/// Rejects an invalid descriptor or reports descriptor and metadata acquisition
/// failures without consuming the caller's descriptor.
pub fn import_virtual_memory(
    session: &Session,
    descriptor: BorrowedFd<'_>,
) -> Result<VirtualMemory, Error> {
    let owner = crate::host_storage::Shared::try_new_uninit(session.driver().allocator())?;
    let inner = session.driver().import_virtual_memory(descriptor)?;
    Ok(VirtualMemory {
        inner: crate::memory::ProviderVirtualMemory::new(
            session.driver().clone(),
            owner.write(inner),
        ),
    })
}

/// Exports detached virtual-memory backing as an independently owned DMA-BUF.
///
/// # Errors
/// Reports descriptor duplication or backing-validation failures.
pub fn export_virtual_memory(memory: &VirtualMemory) -> Result<DmaBuf, Error> {
    crate::driver::PlatformDriver::export_virtual_memory(memory.inner.native())
}

/// Imports one same-provider system allocation from a borrowed DMA-BUF.
///
/// The logical range is established in `device`'s address space with exactly
/// `permissions`. The backend duplicates `descriptor` before native acquisition,
/// so failure never consumes the caller's descriptor.
///
/// # Errors
/// Rejects invalid ranges, placement, origin devices, permissions, or alignment
/// before publication. Native failures retain or release partial state under the
/// same rules as ordinary allocation.
pub fn import_dma_buf(
    device: &Device,
    descriptor: BorrowedFd<'_>,
    source_offset: u64,
    byte_length: u64,
    alignment: u64,
    permissions: DeviceAccess,
) -> Result<Allocation, Error> {
    let inner = device.driver.import_dma_buf(
        &device.state,
        descriptor,
        source_offset,
        byte_length,
        alignment,
        permissions,
    )?;
    Ok(Allocation::from_native(
        inner,
        device.endpoint.provider_instance,
    ))
}

/// Whether an activated Linux GPU can attach a qualified same-device GTT
/// DMA-BUF with explicit GPU permissions and a write-back host view.
#[must_use]
pub fn supports_system_dma_buf_import(device: &Device) -> bool {
    crate::driver::PlatformDriver::supports_system_dma_buf_import(&device.state)
}

/// Imports a qualified SYSTEM DMA-BUF into one native GPU address domain.
///
/// The raw descriptor is borrowed and may be invalid. The native owner first
/// duplicates it, then establishes an independent host view plus exact GPU
/// access before this function returns.
/// Repeated device wrappers sharing one VM receive the same address; distinct
/// GPU VMs are currently unsupported by this qualified profile.
///
/// # Errors
/// Rejects unqualified placement, cache policy, source ranges, permissions, or
/// a mismatched native GPU. Native failures retain ambiguous mapping state.
pub fn import_system_dma_buf(
    session: &Session,
    devices: &[&Device],
    descriptor: RawFd,
    source_offset: u64,
    byte_length: u64,
    alignment: u64,
    permissions: DeviceAccess,
) -> Result<Allocation, Error> {
    if devices.is_empty() {
        return Err(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "system import requires a device",
        });
    }
    let mut states = Buffer::try_with_capacity(devices.len(), session.driver().allocator())?;
    for device in devices {
        if !Shared::ptr_eq(session.driver(), &device.driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "system import devices must belong to this session",
            });
        }
        states.try_push(&device.state)?;
    }
    let inner = session.driver().import_system_dma_buf(
        states.as_slice(),
        descriptor,
        source_offset,
        byte_length,
        alignment,
        permissions,
    )?;
    Ok(Allocation::from_native(
        inner,
        session.driver().provider_instance(),
    ))
}

/// Imports a Linux graphics DMA-BUF into the common address range of `devices`.
///
/// The complete backing is mapped, and native graphics metadata remains owned
/// by the returned allocation until it is freed. The backend duplicates the
/// borrowed descriptor before this function returns.
///
/// # Errors
/// Rejects an empty device list, devices from another session, or invalid
/// DMA-BUF state. Native import and mapping failures preserve the same cleanup
/// ownership guarantees as ordinary allocations.
pub fn import_graphics_dma_buf(
    session: &Session,
    devices: &[&Device],
    descriptor: BorrowedFd<'_>,
    size_hint: u64,
) -> Result<Allocation, Error> {
    if devices.is_empty() {
        return Err(Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "graphics import requires at least one device",
        });
    }
    let mut states = Buffer::try_with_capacity(devices.len(), session.driver().allocator())?;
    for device in devices {
        if !Shared::ptr_eq(session.driver(), &device.driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "graphics import devices must belong to this session",
            });
        }
        states.try_push(&device.state)?;
    }
    let inner =
        session
            .driver()
            .import_graphics_dma_buf(states.as_slice(), descriptor, size_hint)?;
    Ok(Allocation::from_native(
        inner,
        session.driver().provider_instance(),
    ))
}

/// Exports a live allocation as an independently owned DMA-BUF.
///
/// # Errors
/// Returns a native error if Linux cannot export the allocation, or a driver
/// contract error if the resulting file does not describe the same backing.
pub fn export_dma_buf(allocation: &Allocation) -> Result<DmaBuf, Error> {
    crate::driver::PlatformDriver::export_dma_buf(allocation.inner.native())
}

/// Imports one KFD IPC allocation and maps it to the requested GPU devices.
///
/// `devices` supplies every activated GPU available for resolving the exporter,
/// while `mapping_devices` is the exact access set requested by the frontend.
/// All devices must belong to `session`. The returned allocation owns the KFD
/// import, mappings, optional host view, and address reservation.
///
/// # Errors
/// Rejects malformed handles, invalid extents, an unavailable exporting GPU,
/// cross-session devices, or incompatible address spaces. Native import and
/// mapping failures retain partial ownership for safe cleanup.
pub fn import_kfd_ipc_memory(
    session: &Session,
    devices: &[&Device],
    mapping_devices: &[&Device],
    handle: KfdIpcMemoryHandle,
    size: u64,
) -> Result<Allocation, Error> {
    let mut states = Buffer::try_with_capacity(devices.len(), session.driver().allocator())?;
    for device in devices {
        if !Shared::ptr_eq(session.driver(), &device.driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "KFD IPC import devices must belong to this session",
            });
        }
        states.try_push(&device.state)?;
    }
    let mut mapping_states =
        Buffer::try_with_capacity(mapping_devices.len(), session.driver().allocator())?;
    for device in mapping_devices {
        if !Shared::ptr_eq(session.driver(), &device.driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "KFD IPC mapping devices must belong to this session",
            });
        }
        mapping_states.try_push(&device.state)?;
    }
    let inner = session.driver().import_kfd_ipc_memory(
        states.as_slice(),
        mapping_states.as_slice(),
        handle,
        size,
    )?;
    Ok(Allocation::from_native(
        inner,
        session.driver().provider_instance(),
    ))
}

/// Exports a live native allocation as a process-independent KFD IPC handle.
/// The allocation must remain alive until every importer attaches.
///
/// # Errors
/// Rejects unsupported backing or an unavailable allocation and reports the
/// native export failure without changing ownership.
pub fn export_kfd_ipc_memory(allocation: &Allocation) -> Result<KfdIpcMemoryHandle, Error> {
    crate::driver::PlatformDriver::export_kfd_ipc_memory(allocation.inner.native())
}

/// Applies Linux KFD SVM attributes to a process virtual-address range.
///
/// # Errors
/// Rejects invalid ranges, endpoint identities, flags, or attributes and
/// reports the native KFD operation failure.
pub fn set_kfd_svm_attributes(
    session: &Session,
    address: u64,
    size: u64,
    attributes: &[KfdSvmAttribute],
) -> Result<(), Error> {
    session
        .driver()
        .set_kfd_svm_attributes(address, size, attributes)
}

/// Queries Linux KFD SVM attributes for a process virtual-address range.
///
/// # Errors
/// Rejects invalid ranges or query attributes and reports malformed or failed
/// native KFD results without publishing partial translated output.
pub fn get_kfd_svm_attributes(
    session: &Session,
    address: u64,
    size: u64,
    attributes: &mut [KfdSvmAttribute],
) -> Result<(), Error> {
    session
        .driver()
        .get_kfd_svm_attributes(address, size, attributes)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod descriptor_tests {
    use super::*;
    use std::os::fd::AsRawFd;

    #[test]
    fn duplicate_rejects_closed_descriptors_and_retains_its_own_owner() {
        assert_eq!(
            duplicate_descriptor(-1).unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
        let source = std::fs::File::open("/dev/null").unwrap();
        let duplicate = duplicate_descriptor(source.as_raw_fd()).unwrap();
        drop(source);
        assert_eq!(
            duplicate_descriptor(i32::MAX).unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
        assert!(std::fs::File::from(duplicate).metadata().is_ok());
    }

    #[test]
    fn descriptor_path_remains_owned_after_the_caller_closes_its_file() {
        let mut path = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("rocddi path #{}-{nonce}", std::process::id()));
        let file = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&path)
            .unwrap();
        assert!(descriptor_path(-1).is_err());
        let resolved = descriptor_path(file.as_raw_fd()).unwrap();
        assert_eq!(resolved, path);
        drop(file);
        let _other = std::fs::File::open("/dev/null").unwrap();
        assert_eq!(resolved, path);
        assert!(std::fs::File::open(&resolved).is_ok());
        let unlinked = std::fs::File::open(&path).unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(descriptor_path(unlinked.as_raw_fd()).is_err());
    }
}
