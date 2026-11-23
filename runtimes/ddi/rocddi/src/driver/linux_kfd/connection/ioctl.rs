// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Typed KFD ioctl dispatch and temporary argument marshalling.
//!
//! Production calls and test hooks share descriptor checks, borrowed argument
//! storage, and output handling. The connection and runtime transaction remain
//! owned by the parent module; resource owners decide retry and cleanup policy.

#![allow(unsafe_code)]

use super::{Kfd, uapi};
use crate::memory::storage::TryPush;
use crate::os::linux::invalid_data;
use std::ffi::{c_int, c_ulong, c_void};
use std::io;
use std::mem::{size_of, size_of_val};
use std::os::fd::AsRawFd;
use std::ptr;

unsafe extern "C" {
    fn ioctl(fd: c_int, request: c_ulong, ...) -> c_int;
}

/// The typed call also keeps indirect buffers borrowed during fault injection.
/// Test doubles can inspect and modify outputs without following raw pointers.
/// Production dispatch goes straight from this match to the kernel ioctl.
pub(in super::super) enum Call<'a> {
    Version(&'a mut uapi::Version),
    VersionExtension(&'a mut uapi::VersionExtension),
    ClockCounters(&'a mut uapi::ClockCounters),
    AvailableMemory(&'a mut uapi::AvailableMemory),
    XnackMode(&'a mut uapi::XnackMode),
    TileConfig(&'a mut uapi::TileConfig, &'a mut [u32], &'a mut [u32]),
    Apertures(&'a mut uapi::Apertures, &'a mut [uapi::Aperture]),
    AcquireVm(&'a mut uapi::AcquireVm),
    CreateProcess(&'a mut uapi::CreateProcess),
    RuntimeEnable(&'a mut uapi::RuntimeEnable),
    Allocate(&'a mut uapi::AllocMemory),
    Free(&'a mut uapi::FreeMemory),
    Map(&'a mut uapi::MapMemory, &'a [u32]),
    Unmap(&'a mut uapi::MapMemory, &'a [u32]),
    DmaBufInfo(&'a mut uapi::DmaBufInfo, &'a mut [u8]),
    ImportDmaBuf(&'a mut uapi::ImportDmaBuf),
    ExportDmaBuf(&'a mut uapi::ExportDmaBuf),
    IpcImportHandle(&'a mut uapi::IpcImportHandle),
    IpcExportHandle(&'a mut uapi::IpcExportHandle),
    Svm(&'a mut uapi::SvmArgs, &'a mut [uapi::SvmAttribute]),
    Spm(&'a mut uapi::Spm),
    PcSample(&'a mut uapi::PcSampleArgs, &'a mut [uapi::PcSampleInfo]),
    Ais(&'a mut uapi::AisArgs),
    CreateEvent(&'a mut uapi::CreateEvent),
    DestroyEvent(&'a mut uapi::DestroyEvent),
    SetEvent(&'a mut uapi::ResetEvent),
    ResetEvent(&'a mut uapi::ResetEvent),
    Wait(&'a mut uapi::WaitEvents, &'a mut uapi::EventData),
    WaitSignal(&'a mut uapi::WaitEvents, &'a mut [uapi::EventData; 3]),
    WaitExceptions(&'a mut uapi::WaitEvents, &'a mut [uapi::EventData; 2]),
    SetScratchBackingVa(&'a mut uapi::SetScratchBackingVa),
    SetTrapHandler(&'a mut uapi::SetTrapHandler),
    CreateQueue(&'a mut uapi::CreateQueue),
    DestroyQueue(&'a mut uapi::DestroyQueue),
    UpdateQueue(&'a mut uapi::UpdateQueue),
    QueueWaveState(&'a mut uapi::QueueWaveState),
    SetCuMask(&'a mut uapi::SetCuMask, &'a [u32]),
    AllocQueueGws(&'a mut uapi::AllocQueueGws),
}

fn prepare_tile_config(call: &mut Call<'_>) -> io::Result<()> {
    if let Call::TileConfig(args, tiles, macro_tiles) = call {
        args.tiles_ptr = tiles.as_mut_ptr() as u64;
        args.macro_tiles_ptr = macro_tiles.as_mut_ptr() as u64;
        args.num_tile_configs = u32::try_from(tiles.len()).map_err(invalid_data)?;
        args.num_macro_tile_configs = u32::try_from(macro_tiles.len()).map_err(invalid_data)?;
    }
    Ok(())
}

#[cfg(test)]
pub(in super::super) type IoctlHook =
    std::sync::Arc<dyn Fn(&mut Call<'_>) -> io::Result<()> + Send + Sync>;

impl Kfd {
    pub(super) fn call(&self, call: Call<'_>) -> io::Result<()> {
        self.call_with_page_offer(call, None)
    }

    #[allow(
        clippy::too_many_lines,
        reason = "one typed match audits every native KFD ioctl body and borrowed buffer"
    )]
    pub(super) fn call_with_page_offer(
        &self,
        mut call: Call<'_>,
        page_offered: Option<&mut bool>,
    ) -> io::Result<()> {
        self.check_process()?;
        let descriptor = self
            .file
            .as_ref()
            .ok_or_else(|| io::Error::from(io::ErrorKind::NotConnected))?
            .as_raw_fd();
        if let Call::PcSample(args, info) = &mut call {
            args.num_sample_info = u32::try_from(info.len()).map_err(invalid_data)?;
            args.sample_info_ptr = if info.is_empty() {
                0
            } else {
                info.as_mut_ptr() as u64
            };
        }
        prepare_tile_config(&mut call)?;
        if let Some(page_offered) = page_offered {
            // Validation has finished. KFD may retain this page even when the
            // ioctl fails, so record the offer before dispatch or its test hook.
            *page_offered = true;
        }
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            return hook(&mut call);
        }
        if let Call::Svm(args, attributes) = &mut call {
            return Self::call_svm(descriptor, args, attributes);
        }
        let (request, body) = match &mut call {
            Call::Version(args) => (uapi::GET_VERSION, ptr::from_mut(*args).cast::<c_void>()),
            Call::VersionExtension(args) => (
                uapi::GET_VERSION_EXTENSION,
                ptr::from_mut(*args).cast::<c_void>(),
            ),
            Call::ClockCounters(args) => (uapi::GET_CLOCK_COUNTERS, ptr::from_mut(*args).cast()),
            Call::AvailableMemory(args) => {
                (uapi::GET_AVAILABLE_MEMORY, ptr::from_mut(*args).cast())
            }
            Call::XnackMode(args) => (uapi::SET_XNACK_MODE, ptr::from_mut(*args).cast()),
            Call::TileConfig(args, _, _) => (uapi::GET_TILE_CONFIG, ptr::from_mut(*args).cast()),
            Call::Apertures(args, entries) => {
                args.pointer = if entries.is_empty() {
                    0
                } else {
                    entries.as_mut_ptr() as u64
                };
                (uapi::GET_APERTURES, ptr::from_mut(*args).cast())
            }
            Call::AcquireVm(args) => (uapi::ACQUIRE_VM, ptr::from_mut(*args).cast()),
            Call::CreateProcess(args) => (uapi::CREATE_PROCESS, ptr::from_mut(*args).cast()),
            Call::RuntimeEnable(args) => (uapi::RUNTIME_ENABLE, ptr::from_mut(*args).cast()),
            Call::Allocate(args) => (uapi::ALLOC_MEMORY, ptr::from_mut(*args).cast()),
            Call::Free(args) => (uapi::FREE_MEMORY, ptr::from_mut(*args).cast()),
            Call::Map(args, devices) => {
                args.devices = devices.as_ptr() as u64;
                (uapi::MAP_MEMORY, ptr::from_mut(*args).cast())
            }
            Call::Unmap(args, devices) => {
                args.devices = devices.as_ptr() as u64;
                (uapi::UNMAP_MEMORY, ptr::from_mut(*args).cast())
            }
            Call::DmaBufInfo(args, metadata) => {
                args.metadata = if metadata.is_empty() {
                    0
                } else {
                    metadata.as_mut_ptr() as u64
                };
                args.metadata_size = u32::try_from(metadata.len()).map_err(invalid_data)?;
                (uapi::GET_DMABUF_INFO, ptr::from_mut(*args).cast())
            }
            Call::ImportDmaBuf(args) => (uapi::IMPORT_DMABUF, ptr::from_mut(*args).cast()),
            Call::ExportDmaBuf(args) => (uapi::EXPORT_DMABUF, ptr::from_mut(*args).cast()),
            Call::IpcImportHandle(args) => (uapi::IPC_IMPORT_HANDLE, ptr::from_mut(*args).cast()),
            Call::IpcExportHandle(args) => (uapi::IPC_EXPORT_HANDLE, ptr::from_mut(*args).cast()),
            Call::Svm(_, _) => return Err(invalid_data("SVM dispatch path was not selected")),
            Call::Spm(args) => (uapi::SPM, ptr::from_mut(*args).cast()),
            Call::PcSample(args, _) => (uapi::PC_SAMPLE, ptr::from_mut(*args).cast()),
            Call::Ais(args) => (uapi::AIS, ptr::from_mut(*args).cast()),
            Call::CreateEvent(args) => (uapi::CREATE_EVENT, ptr::from_mut(*args).cast()),
            Call::DestroyEvent(args) => (uapi::DESTROY_EVENT, ptr::from_mut(*args).cast()),
            Call::SetEvent(args) => (uapi::SET_EVENT, ptr::from_mut(*args).cast()),
            Call::ResetEvent(args) => (uapi::RESET_EVENT, ptr::from_mut(*args).cast()),
            Call::WaitSignal(args, events) => {
                args.events = events.as_mut_ptr() as u64;
                (uapi::WAIT_EVENTS, ptr::from_mut(*args).cast())
            }
            Call::WaitExceptions(args, events) => {
                args.events = events.as_mut_ptr() as u64;
                (uapi::WAIT_EVENTS, ptr::from_mut(*args).cast())
            }
            Call::SetScratchBackingVa(args) => {
                (uapi::SET_SCRATCH_BACKING_VA, ptr::from_mut(*args).cast())
            }
            Call::SetTrapHandler(args) => (uapi::SET_TRAP_HANDLER, ptr::from_mut(*args).cast()),
            Call::CreateQueue(args) => (uapi::CREATE_QUEUE, ptr::from_mut(*args).cast()),
            Call::DestroyQueue(args) => (uapi::DESTROY_QUEUE, ptr::from_mut(*args).cast()),
            Call::QueueWaveState(args) => (uapi::GET_QUEUE_WAVE_STATE, ptr::from_mut(*args).cast()),
            Call::UpdateQueue(args) => (uapi::UPDATE_QUEUE, ptr::from_mut(*args).cast()),
            Call::SetCuMask(args, mask) => {
                args.mask = mask.as_ptr() as u64;
                (uapi::SET_CU_MASK, ptr::from_mut(*args).cast())
            }
            Call::AllocQueueGws(args) => (uapi::ALLOC_QUEUE_GWS, ptr::from_mut(*args).cast()),
            Call::Wait(args, event) => {
                args.events = ptr::from_mut(*event) as u64;
                (uapi::WAIT_EVENTS, ptr::from_mut(*args).cast())
            }
        };
        // SAFETY: Every variant supplies the repr(C) record belonging to this
        // request. Records and indirect arrays remain exclusively borrowed and
        // valid until ioctl returns; their lengths were set from actual storage.
        // The descriptor is owned here and has been checked against the PID.
        // KFD may update outputs on failure, so nothing rewrites them afterward.
        let result = unsafe { ioctl(descriptor, request as c_ulong, body) };
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }

    fn call_svm(
        descriptor: c_int,
        args: &mut uapi::SvmArgs,
        attributes: &mut [uapi::SvmAttribute],
    ) -> io::Result<()> {
        let request = uapi::svm_request(attributes.len())
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let words = 3_usize
            .checked_add(attributes.len())
            .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
        let mut buffer = crate::memory::storage::try_vec_with_capacity::<u64>(words)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        for _ in 0..words {
            buffer
                .try_push(0)
                .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        }
        let words = buffer.as_mut_slice().as_mut_ptr();
        let body = words.cast::<u8>();
        // SAFETY: The u64 buffer is eight-byte aligned and exactly covers the
        // 24-byte header followed by every eight-byte flexible-array element.
        unsafe {
            words.cast::<uapi::SvmArgs>().write(*args);
            ptr::copy_nonoverlapping(
                attributes.as_ptr().cast::<u8>(),
                body.add(size_of::<uapi::SvmArgs>()),
                size_of_val(attributes),
            );
        }
        // SAFETY: The request size encodes this complete live buffer. KFD makes
        // one synchronous copy and may update both header and attributes.
        let result = unsafe { ioctl(descriptor, request as c_ulong, body.cast::<c_void>()) };
        // Preserve KFD outputs even when the ioctl reports an error.
        // SAFETY: Both destinations are valid exclusive borrows of the exact
        // initialized record types copied into the aligned buffer above.
        unsafe {
            *args = words.cast::<uapi::SvmArgs>().read();
            ptr::copy_nonoverlapping(
                body.add(size_of::<uapi::SvmArgs>()),
                attributes.as_mut_ptr().cast::<u8>(),
                size_of_val(attributes),
            );
        }
        if result == 0 {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
}
