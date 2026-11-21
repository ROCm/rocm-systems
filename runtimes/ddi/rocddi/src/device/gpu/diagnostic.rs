// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Fatal GPU diagnostics. Snapshots retain memory owners but do not make a
//! running application's host writes atomic. Queue capture permanently stops
//! execution; it is intended for a fatal-error path, not debugger suspension.

use crate::{Error, ErrorKind};
use std::marker::PhantomData;

/// One retained GPU interval and its optional CPU alias.
#[derive(Clone, Copy, Debug)]
pub struct DiagnosticMemory<'a> {
    /// GPU virtual address used by saved waves and packets.
    pub(crate) address: u64,
    /// Logical byte extent of the allocation.
    pub(crate) size: u64,
    /// CPU mapping, which can differ from the GPU address.
    pub(crate) host_address: Option<usize>,
    /// Prevents release of the owner while this view is used.
    owner: PhantomData<&'a ()>,
}

impl<'a> DiagnosticMemory<'a> {
    /// GPU virtual address of the first byte.
    #[must_use]
    pub const fn address(self) -> u64 {
        self.address
    }
    /// Logical byte extent retained by the owner.
    #[must_use]
    pub const fn size(self) -> u64 {
        self.size
    }
    /// CPU alias for reading, when the native placement is host visible.
    #[must_use]
    pub const fn host_address(self) -> Option<usize> {
        self.host_address
    }

    /// Borrows an externally owned CPU mapping for diagnostic reads.
    ///
    /// # Safety
    /// `owner` must retain the entire host interval until this view expires.
    /// It must contain ordinary memory, not MMIO. The GPU address must denote
    /// the same bytes. This does not synchronize concurrent CPU/GPU writes.
    ///
    /// # Errors
    /// Rejects empty or overflowing ranges.
    #[allow(unsafe_code)]
    pub unsafe fn borrow<T: ?Sized>(
        _owner: &'a T,
        address: u64,
        host_address: Option<usize>,
        size: u64,
    ) -> Result<Self, Error> {
        if size == 0
            || address.checked_add(size).is_none()
            || host_address.is_some_and(|host| {
                (host as u64)
                    .checked_add(size)
                    .is_none_or(|end| end > isize::MAX as u64)
            })
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "invalid diagnostic memory interval",
            });
        }
        Ok(Self {
            address,
            size,
            host_address,
            owner: PhantomData,
        })
    }
}

/// KFD runtime registration actually accepted by the driver.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoreRuntimeInfo {
    /// Address of the registered runtime debugger rendezvous.
    pub r_debug: u64,
    /// KFD runtime-state enum (zero disabled, one enabled).
    pub runtime_state: u32,
    /// Whether the registered trap handler initializes TTMP state.
    pub ttmp_setup: u32,
}

/// Native KFD device information consumed by AMD GPU core debuggers.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct CoreDeviceInfo {
    /// Native `exception_status` field in `kfd_dbg_device_info_entry`.
    pub exception_status: u64,
    /// Native `lds_base` field in `kfd_dbg_device_info_entry`.
    pub lds_base: u64,
    /// Native `lds_limit` field in `kfd_dbg_device_info_entry`.
    pub lds_limit: u64,
    /// Native `scratch_base` field in `kfd_dbg_device_info_entry`.
    pub scratch_base: u64,
    /// Native `scratch_limit` field in `kfd_dbg_device_info_entry`.
    pub scratch_limit: u64,
    /// Native `gpuvm_base` field in `kfd_dbg_device_info_entry`.
    pub gpuvm_base: u64,
    /// Native `gpuvm_limit` field in `kfd_dbg_device_info_entry`.
    pub gpuvm_limit: u64,
    /// Native `gpu_id` field in `kfd_dbg_device_info_entry`.
    pub gpu_id: u32,
    /// Native `location_id` field in `kfd_dbg_device_info_entry`.
    pub location_id: u32,
    /// Native `vendor_id` field in `kfd_dbg_device_info_entry`.
    pub vendor_id: u32,
    /// Native `device_id` field in `kfd_dbg_device_info_entry`.
    pub device_id: u32,
    /// Native `revision_id` field in `kfd_dbg_device_info_entry`.
    pub revision_id: u32,
    /// Native `subsystem_vendor_id` field in `kfd_dbg_device_info_entry`.
    pub subsystem_vendor_id: u32,
    /// Native `subsystem_device_id` field in `kfd_dbg_device_info_entry`.
    pub subsystem_device_id: u32,
    /// Native `fw_version` field in `kfd_dbg_device_info_entry`.
    pub fw_version: u32,
    /// Native `gfx_target_version` field in `kfd_dbg_device_info_entry`.
    pub gfx_target_version: u32,
    /// Native `simd_count` field in `kfd_dbg_device_info_entry`.
    pub simd_count: u32,
    /// Native `max_waves_per_simd` field in `kfd_dbg_device_info_entry`.
    pub max_waves_per_simd: u32,
    /// Native `array_count` field in `kfd_dbg_device_info_entry`.
    pub array_count: u32,
    /// Native `simd_arrays_per_engine` field in `kfd_dbg_device_info_entry`.
    pub simd_arrays_per_engine: u32,
    /// Native `num_xcc` field in `kfd_dbg_device_info_entry`.
    pub num_xcc: u32,
    /// Native `capability` field in `kfd_dbg_device_info_entry`.
    pub capability: u32,
    /// Native `debug_prop` field in `kfd_dbg_device_info_entry`.
    pub debug_prop: u32,
}

/// Process and device metadata obtained without attaching a debugger.
#[derive(Clone, Copy, Debug)]
pub struct CoreDeviceSnapshot {
    /// KFD version queried on the active connection.
    pub version: [u32; 2],
    /// Effective process registration on that connection.
    pub runtime: CoreRuntimeInfo,
    /// Device identity, shader geometry, and process apertures.
    pub device: CoreDeviceInfo,
}

/// An inactivated compute queue and retained diagnostic memory.
pub struct CoreQueueSnapshot<'a> {
    /// Kernel queue identifier.
    pub queue_id: u32,
    /// Kernel GPU identifier.
    pub gpu_id: u32,
    /// Native KFD queue-type enum.
    pub queue_type: u32,
    /// Ring extent submitted to KFD, excluding allocation page padding.
    pub ring_size_bytes: u32,
    /// Device address of the write index.
    pub write_pointer: u64,
    /// Device address of the read index.
    pub read_pointer: u64,
    /// Context-save allocation size per XCC.
    pub context_size: u32,
    /// Ring, index/control, and refreshed CWSR memory, in that order.
    pub memory: [DiagnosticMemory<'a>; 3],
}

/// A process-local diagnostic reader, opened afresh for each capture.
#[cfg(target_os = "linux")]
pub struct DiagnosticReader {
    /// This descriptor must never be reused in a fork child.
    file: std::fs::File,
    /// PID whose address space the descriptor reads.
    process: u32,
}

#[cfg(target_os = "linux")]
impl DiagnosticReader {
    /// Opens the current address space without creating Rust references to
    /// memory that application threads or the device may still modify.
    ///
    /// # Errors
    /// Propagates procfs access failures.
    pub fn new() -> std::io::Result<Self> {
        Ok(Self {
            file: std::fs::File::open("/proc/self/mem")?,
            process: std::process::id(),
        })
    }

    /// Reads a bounded portion of retained memory through its CPU alias.
    ///
    /// # Errors
    /// Rejects unavailable aliases, foreign processes, invalid ranges and
    /// native short reads. No device copy engine is used after a GPU fault.
    pub fn read(
        &self,
        memory: DiagnosticMemory<'_>,
        offset: u64,
        bytes: &mut [u8],
    ) -> std::io::Result<()> {
        use std::os::unix::fs::FileExt;
        if self.process != std::process::id()
            || offset
                .checked_add(bytes.len() as u64)
                .is_none_or(|end| end > memory.size)
        {
            return Err(std::io::ErrorKind::InvalidInput.into());
        }
        let host = memory.host_address.ok_or(std::io::ErrorKind::Unsupported)? as u64;
        self.file.read_exact_at(bytes, host + offset)
    }
}
