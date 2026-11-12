// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Positioned disk I/O through an independently reopened overlapped handle.
//!
//! `ReOpenFile` can fail under access or sharing restrictions even when the
//! original handle is valid. No seek/read/restore sequence changes its cursor.
//! All request state and buffers remain live until native completion.

#![allow(unsafe_code)]

use std::ffi::OsString;
use std::fs::File;
use std::io;
use std::os::windows::ffi::OsStringExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle, RawHandle};
use std::path::PathBuf;
use std::ptr;
use windows_sys::Win32::Foundation::{
    DUPLICATE_SAME_ACCESS, DuplicateHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, GENERIC_READ,
    GENERIC_WRITE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::{
    BY_HANDLE_FILE_INFORMATION, FILE_FLAG_OVERLAPPED, FILE_SHARE_DELETE, FILE_SHARE_READ,
    FILE_SHARE_WRITE, GetFileInformationByHandle, GetFinalPathNameByHandleW, ReOpenFile, ReadFile,
    WriteFile,
};
use windows_sys::Win32::System::IO::{
    GetOverlappedResult, OVERLAPPED, OVERLAPPED_0, OVERLAPPED_0_0,
};
use windows_sys::Win32::System::Threading::{CreateEventW, GetCurrentProcess};

pub(crate) fn duplicate_file(handle: RawHandle) -> io::Result<File> {
    let mut duplicate = ptr::null_mut();
    // SAFETY: DuplicateHandle validates raw C input before any Rust owner is
    // created. Neither pseudo current-process handle is transferred.
    if unsafe {
        DuplicateHandle(
            GetCurrentProcess(),
            handle,
            GetCurrentProcess(),
            &raw mut duplicate,
            0,
            0,
            DUPLICATE_SAME_ACCESS,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: Success produced a new owned handle; the caller keeps its input.
    Ok(unsafe { File::from_raw_handle(duplicate) })
}

fn reopen(file: &File, access: u32) -> io::Result<File> {
    // SAFETY: The File retains the original handle. OVERLAPPED mode gives
    // this independent reference explicit per-operation offsets.
    let handle = unsafe {
        ReOpenFile(
            file.as_raw_handle(),
            access,
            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
            FILE_FLAG_OVERLAPPED,
        )
    };
    if handle == INVALID_HANDLE_VALUE {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: ReOpenFile returns a new owned native file handle.
    Ok(unsafe { File::from_raw_handle(handle) })
}

/// The private event cannot be reset or closed by another runtime request.
fn request(offset: u64) -> io::Result<(OwnedHandle, OVERLAPPED)> {
    // SAFETY: No security/name pointer is borrowed; create a manual-reset event.
    let event = unsafe { CreateEventW(ptr::null(), 1, 0, ptr::null()) };
    if event.is_null() {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: The successful creation returns a new owned event.
    let owner = unsafe { OwnedHandle::from_raw_handle(event) };
    let bytes = offset.to_le_bytes();
    let overlapped = OVERLAPPED {
        Anonymous: OVERLAPPED_0 {
            Anonymous: OVERLAPPED_0_0 {
                Offset: u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
                OffsetHigh: u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]),
            },
        },
        hEvent: event,
        ..OVERLAPPED::default()
    };
    Ok((owner, overlapped))
}

fn raw_status(error: &io::Error) -> Option<u32> {
    error
        .raw_os_error()
        .map(|code| u32::from_ne_bytes(code.to_ne_bytes()))
}

fn completed(file: &File, request: &OVERLAPPED, accepted: i32) -> io::Result<usize> {
    if accepted == 0 {
        let error = io::Error::last_os_error();
        if raw_status(&error) == Some(ERROR_HANDLE_EOF) {
            return Ok(0);
        }
        if raw_status(&error) != Some(ERROR_IO_PENDING) {
            return Err(error);
        }
    }
    let mut count = 0;
    // SAFETY: The caller retains the private file, event, request, and buffer.
    // TRUE waits until this operation completes, including failed completion;
    // no cancellation or timeout releases its borrowed storage prematurely.
    if unsafe { GetOverlappedResult(file.as_raw_handle(), request, &raw mut count, 1) } == 0 {
        let error = io::Error::last_os_error();
        if raw_status(&error) == Some(ERROR_HANDLE_EOF) {
            return Ok(0);
        }
        return Err(error);
    }
    Ok(count as usize)
}

pub(crate) fn read_at(file: &File, bytes: &mut [u8], offset: u64) -> io::Result<usize> {
    let file = reopen(file, GENERIC_READ)?;
    let (_event, mut request) = request(offset)?;
    let count = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    let mut immediate = 0;
    // SAFETY: The writable slice, OVERLAPPED and event remain live through
    // completed(), which waits for pending I/O before this scope exits.
    let accepted = unsafe {
        ReadFile(
            file.as_raw_handle(),
            bytes.as_mut_ptr(),
            count,
            &raw mut immediate,
            &raw mut request,
        )
    };
    completed(&file, &request, accepted)
}

pub(crate) fn write_at(file: &File, bytes: &[u8], offset: u64) -> io::Result<usize> {
    let file = reopen(file, GENERIC_WRITE)?;
    let (_event, mut request) = request(offset)?;
    let count = u32::try_from(bytes.len()).unwrap_or(u32::MAX);
    let mut immediate = 0;
    // SAFETY: The readable slice and private completion state remain live
    // until completed() proves that native I/O no longer accesses them.
    let accepted = unsafe {
        WriteFile(
            file.as_raw_handle(),
            bytes.as_ptr(),
            count,
            &raw mut immediate,
            &raw mut request,
        )
    };
    completed(&file, &request, accepted)
}

fn identity(file: &File) -> io::Result<(u32, u32, u32)> {
    let mut info = BY_HANDLE_FILE_INFORMATION::default();
    // SAFETY: The retained file and complete output record live for the call.
    if unsafe { GetFileInformationByHandle(file.as_raw_handle(), &raw mut info) } == 0 {
        return Err(io::Error::last_os_error());
    }
    Ok((
        info.dwVolumeSerialNumber,
        info.nFileIndexHigh,
        info.nFileIndexLow,
    ))
}

pub(crate) fn path(file: &File) -> io::Result<PathBuf> {
    let mut path = vec![0_u16; 512];
    for _ in 0..3 {
        let capacity = u32::try_from(path.len()).map_err(|_| io::ErrorKind::InvalidData)?;
        // SAFETY: The output buffer is writable for capacity UTF-16 units.
        let length = unsafe {
            GetFinalPathNameByHandleW(file.as_raw_handle(), path.as_mut_ptr(), capacity, 0)
        };
        if length == 0 {
            return Err(io::Error::last_os_error());
        }
        if length >= capacity {
            if length > 32_768 {
                return Err(io::ErrorKind::InvalidData.into());
            }
            path.resize(length as usize + 1, 0);
            continue;
        }
        path.truncate(length as usize);
        let path = PathBuf::from(OsString::from_wide(&path));
        if identity(file)? != identity(&File::open(&path)?)? {
            return Err(io::ErrorKind::InvalidData.into());
        }
        return Ok(path);
    }
    Err(io::ErrorKind::InvalidData.into())
}
