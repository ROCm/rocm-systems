// Copyright (c) 2026 Advanced Micro Devices, Inc.
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
use crate::driver::Driver;
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
    unsafe { crate::driver::linux_fd::close_owned_descriptor(descriptor) }
}

/// Validates and duplicates a caller descriptor before constructing a Rust
/// borrowed descriptor. The returned owner is independent of the caller's fd.
///
/// # Errors
/// Reports an invalid or closed descriptor, or native descriptor exhaustion.
pub fn duplicate_descriptor(descriptor: RawFd) -> Result<OwnedFd, Error> {
    crate::driver::linux_fd::duplicate_descriptor(descriptor)
}

/// Returns the length of a borrowed descriptor without taking its ownership.
///
/// # Errors
/// Reports an invalid descriptor or native metadata failure.
pub fn descriptor_length(descriptor: RawFd) -> io::Result<u64> {
    crate::driver::linux_fd::descriptor_length(descriptor)
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
    crate::driver::linux_fd::read_descriptor_exact_at(descriptor, buffer, offset)
}

/// Reads from a borrowed descriptor at a fixed offset.
///
/// # Errors
/// Preserves the Linux read error, including its errno.
pub fn read_descriptor_at(descriptor: RawFd, buffer: &mut [u8], offset: i64) -> io::Result<usize> {
    crate::driver::linux_fd::read_descriptor_at(descriptor, buffer, offset)
}

/// Writes to a borrowed descriptor at a fixed offset.
///
/// # Errors
/// Preserves the Linux write error, including its errno.
pub fn write_descriptor_at(descriptor: RawFd, buffer: &[u8], offset: i64) -> io::Result<usize> {
    crate::driver::linux_fd::write_descriptor_at(descriptor, buffer, offset)
}

/// Maximum bytes submitted in one AIS operation, matching Linux `MAX_RW_COUNT`.
pub const AIS_MAX_TRANSFER_BYTES: u64 = 0x7fff_f000;
const EIO: i32 = 5;
const EOVERFLOW: i32 = 75;

/// Direction of a Linux AIS transfer between a file and device VRAM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AisFileOperation {
    /// Read file bytes into the allocation.
    Read,
    /// Write allocation bytes into the file.
    Write,
}

/// AIS transfer result with known copied-byte progress.
/// A nonzero status is a negative Linux errno.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AisFileResult {
    /// Bytes reported as copied, bounded by the submitted transfer size.
    pub size_copied: u64,
    /// Operation status, zero on success or a negative Linux errno.
    pub status: i32,
}

/// CPU-visible storage borrowed for one positioned AIS file transfer.
pub enum AisHostBuffer<'a> {
    /// Read file bytes into writable host storage.
    Read(&'a mut [u8]),
    /// Write readable host storage into the file.
    Write(&'a [u8]),
}

/// Transfers one bounded chunk through positioned host file I/O.
///
/// A short read at end of file succeeds. A short write continues from the
/// reported position; a write with no progress is retried at most three times.
/// Partial progress and Linux errno remain available on failure. A zero-byte
/// request still issues one positioned call to validate the descriptor.
#[must_use]
pub fn ais_host_transfer(
    descriptor: RawFd,
    mut buffer: AisHostBuffer<'_>,
    file_offset: i64,
) -> AisFileResult {
    let size = match &buffer {
        AisHostBuffer::Read(bytes) => bytes.len(),
        AisHostBuffer::Write(bytes) => bytes.len(),
    }
    .min(usize::try_from(AIS_MAX_TRANSFER_BYTES).unwrap_or(usize::MAX));
    let mut copied = 0;
    let mut write_retries = 3;
    let mut first = true;
    let status = loop {
        let remaining = size - copied;
        if !first && remaining == 0 {
            break 0;
        }
        first = false;
        let Some(offset) = i64::try_from(copied)
            .ok()
            .and_then(|copied| file_offset.checked_add(copied))
        else {
            break -EOVERFLOW;
        };
        let transferred = match &mut buffer {
            AisHostBuffer::Read(bytes) => {
                read_descriptor_at(descriptor, &mut bytes[copied..size], offset)
            }
            AisHostBuffer::Write(bytes) => {
                write_descriptor_at(descriptor, &bytes[copied..size], offset)
            }
        };
        let transferred = match transferred {
            Ok(transferred) => transferred,
            Err(error) => break -error.raw_os_error().unwrap_or(EIO),
        };
        if transferred == 0 {
            if matches!(&buffer, AisHostBuffer::Read(_)) || remaining == 0 {
                break 0;
            }
            if write_retries == 0 {
                break -EIO;
            }
            write_retries -= 1;
            continue;
        }
        if transferred > remaining {
            break -EIO;
        }
        copied += transferred;
    };
    AisFileResult {
        size_copied: copied as u64,
        status,
    }
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
    crate::driver::KfdDriver::ais_transfer(
        allocation.inner.driver_state(),
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
    let driver = session.linux_kfd()?;
    let owner = crate::host_storage::Shared::try_new_uninit(driver.allocator())?;
    let inner = driver.import_virtual_memory(descriptor)?;
    Ok(VirtualMemory {
        inner: crate::memory::DriverVirtualMemory::new(driver.clone(), owner.write(inner)),
    })
}

/// Exports detached virtual-memory backing as an independently owned DMA-BUF.
///
/// # Errors
/// Reports descriptor duplication or backing-validation failures.
pub fn export_virtual_memory(memory: &VirtualMemory) -> Result<DmaBuf, Error> {
    crate::driver::KfdDriver::export_virtual_memory(memory.inner.driver_state())
}

/// Imports one same-driver system allocation from a borrowed DMA-BUF.
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
    let (driver, state) = device.linux_kfd()?;
    let inner = driver.import_dma_buf(
        state,
        descriptor,
        source_offset,
        byte_length,
        alignment,
        permissions,
    )?;
    Ok(Allocation::from_native(
        inner,
        device.endpoint.driver_instance,
    ))
}

/// Whether an activated Linux GPU can attach a qualified same-device GTT
/// DMA-BUF with explicit GPU permissions and a write-back host view.
#[must_use]
pub fn supports_system_dma_buf_import(device: &Device) -> bool {
    device
        .linux_kfd()
        .is_ok_and(|(_, state)| crate::driver::KfdDriver::supports_system_dma_buf_import(state))
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
    let driver = session.linux_kfd()?;
    let mut states = Buffer::try_with_capacity(devices.len(), driver.allocator())?;
    for device in devices {
        let (device_driver, state) = device.linux_kfd()?;
        if !Shared::ptr_eq(driver, device_driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "system import devices must belong to this session",
            });
        }
        states.try_push(state)?;
    }
    let inner = driver.import_system_dma_buf(
        states.as_slice(),
        descriptor,
        source_offset,
        byte_length,
        alignment,
        permissions,
    )?;
    Ok(Allocation::from_native(inner, driver.driver_instance()))
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
    let driver = session.linux_kfd()?;
    let mut states = Buffer::try_with_capacity(devices.len(), driver.allocator())?;
    for device in devices {
        let (device_driver, state) = device.linux_kfd()?;
        if !Shared::ptr_eq(driver, device_driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "graphics import devices must belong to this session",
            });
        }
        states.try_push(state)?;
    }
    let inner = driver.import_graphics_dma_buf(states.as_slice(), descriptor, size_hint)?;
    Ok(Allocation::from_native(inner, driver.driver_instance()))
}

/// Exports a live allocation as an independently owned DMA-BUF.
///
/// # Errors
/// Returns a native error if Linux cannot export the allocation, or a driver
/// contract error if the resulting file does not describe the same backing.
pub fn export_dma_buf(allocation: &Allocation) -> Result<DmaBuf, Error> {
    crate::driver::KfdDriver::export_dma_buf(allocation.inner.driver_state())
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
    let driver = session.linux_kfd()?;
    let mut states = Buffer::try_with_capacity(devices.len(), driver.allocator())?;
    for device in devices {
        let (device_driver, state) = device.linux_kfd()?;
        if !Shared::ptr_eq(driver, device_driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "KFD IPC import devices must belong to this session",
            });
        }
        states.try_push(state)?;
    }
    let mut mapping_states = Buffer::try_with_capacity(mapping_devices.len(), driver.allocator())?;
    for device in mapping_devices {
        let (device_driver, state) = device.linux_kfd()?;
        if !Shared::ptr_eq(driver, device_driver) {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "KFD IPC mapping devices must belong to this session",
            });
        }
        mapping_states.try_push(state)?;
    }
    let inner =
        driver.import_kfd_ipc_memory(states.as_slice(), mapping_states.as_slice(), handle, size)?;
    Ok(Allocation::from_native(inner, driver.driver_instance()))
}

/// Exports a live native allocation as a process-independent KFD IPC handle.
/// The allocation must remain alive until every importer attaches.
///
/// # Errors
/// Rejects unsupported backing or an unavailable allocation and reports the
/// native export failure without changing ownership.
pub fn export_kfd_ipc_memory(allocation: &Allocation) -> Result<KfdIpcMemoryHandle, Error> {
    crate::driver::KfdDriver::export_kfd_ipc_memory(allocation.inner.driver_state())
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
        .linux_kfd()?
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
        .linux_kfd()?
        .get_kfd_svm_attributes(address, size, attributes)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod descriptor_tests {
    use super::*;
    use std::io::{Seek, SeekFrom};
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::FileExt;

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

    #[test]
    fn host_ais_reports_partial_progress_without_moving_the_file_position() {
        let mut path = std::env::temp_dir();
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        path.push(format!("rocddi-host-ais-{}-{nonce}", std::process::id()));
        let mut file = std::fs::OpenOptions::new()
            .create_new(true)
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        file.write_all_at(b"abc", 0).unwrap();
        file.seek(SeekFrom::Start(2)).unwrap();

        let mut bytes = [0xa5_u8; 5];
        let result = ais_host_transfer(file.as_raw_fd(), AisHostBuffer::Read(&mut bytes), 0);
        assert_eq!((result.size_copied, result.status), (3, 0));
        assert_eq!(&bytes, b"abc\xa5\xa5");
        assert_eq!(file.stream_position().unwrap(), 2);

        let result = ais_host_transfer(file.as_raw_fd(), AisHostBuffer::Write(b"de"), 1);
        assert_eq!((result.size_copied, result.status), (2, 0));
        assert_eq!(file.stream_position().unwrap(), 2);
        let mut updated = [0_u8; 3];
        file.read_exact_at(&mut updated, 0).unwrap();
        assert_eq!(&updated, b"ade");

        let result = ais_host_transfer(-1, AisHostBuffer::Write(b"x"), 0);
        assert_eq!((result.size_copied, result.status), (0, -9));
        drop(file);
        std::fs::remove_file(path).unwrap();
    }
}
