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
    DUPLICATE_SAME_ACCESS, DuplicateHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING,
    INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::Storage::FileSystem::{
    FILE_FLAG_OVERLAPPED, FILE_ID_INFO, FILE_READ_DATA, FILE_SHARE_DELETE, FILE_SHARE_READ,
    FILE_SHARE_WRITE, FILE_WRITE_DATA, FileIdInfo, GetFileInformationByHandleEx,
    GetFinalPathNameByHandleW, ReOpenFile, ReadFile, WriteFile,
};
use windows_sys::Win32::System::IO::{
    GetOverlappedResult, OVERLAPPED, OVERLAPPED_0, OVERLAPPED_0_0,
};
use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
use windows_sys::Win32::System::Threading::{CreateEventW, GetCurrentProcess};
use windows_sys::Win32::System::WindowsProgramming::PUBLIC_OBJECT_BASIC_INFORMATION;

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

/// Queries the access granted to the retained handle, without reopening it.
fn granted_access(file: &File) -> io::Result<u32> {
    // NtQueryObject's public contract requires dynamic resolution. Ntdll is a
    // process-lifetime system module; GetModuleHandleW does not acquire a lease.
    type Query =
        unsafe extern "system" fn(RawHandle, i32, *mut std::ffi::c_void, u32, *mut u32) -> i32;
    // SAFETY: Both names are constant and terminated; the module remains loaded.
    let address = unsafe {
        let module = GetModuleHandleW(windows_sys::w!("ntdll.dll"));
        if module.is_null() {
            return Err(io::Error::last_os_error());
        }
        GetProcAddress(module, c"NtQueryObject".as_ptr().cast())
    }
    .ok_or_else(io::Error::last_os_error)?;
    // SAFETY: The documented symbol has the Query signature and system ABI.
    let query: Query = unsafe { std::mem::transmute(address) };
    let mut info = PUBLIC_OBJECT_BASIC_INFORMATION::default();
    // SAFETY: Class zero is ObjectBasicInformation; the complete output record
    // and retained handle live through this synchronous query.
    let status = unsafe {
        query(
            file.as_raw_handle(),
            0,
            (&raw mut info).cast(),
            u32::try_from(size_of::<PUBLIC_OBJECT_BASIC_INFORMATION>())
                .map_err(|_| io::ErrorKind::InvalidData)?,
            ptr::null_mut(),
        )
    };
    if status < 0 {
        return Err(io::Error::other(crate::Error::DriverOperation {
            kind: crate::ErrorKind::Driver,
            operation: "NtQueryObject file access",
            status: crate::NativeStatus {
                domain: crate::NativeStatusDomain::NtStatus,
                code: u32::from_ne_bytes(status.to_ne_bytes()),
            },
        }));
    }
    Ok(info.GrantedAccess)
}

fn reopen(file: &File, access: u32) -> io::Result<File> {
    if granted_access(file)? & access != access {
        return Err(io::ErrorKind::PermissionDenied.into());
    }
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

fn completed(file: &File, request: *const OVERLAPPED, accepted: i32) -> io::Result<usize> {
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
    let file = reopen(file, FILE_READ_DATA)?;
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
    completed(&file, &raw const request, accepted)
}

pub(crate) fn write_at(file: &File, bytes: &[u8], offset: u64) -> io::Result<usize> {
    let file = reopen(file, FILE_WRITE_DATA)?;
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
    completed(&file, &raw const request, accepted)
}

fn identity(file: &File) -> io::Result<(u64, [u8; 16])> {
    let mut info = FILE_ID_INFO::default();
    // SAFETY: FileIdInfo writes the complete record. The full 128-bit ID is
    // required for uniqueness on ReFS; legacy 64-bit file IDs are insufficient.
    if unsafe {
        GetFileInformationByHandleEx(
            file.as_raw_handle(),
            FileIdInfo,
            (&raw mut info).cast(),
            u32::try_from(size_of::<FILE_ID_INFO>()).map_err(|_| io::ErrorKind::InvalidData)?,
        )
    } == 0
    {
        return Err(io::Error::last_os_error());
    }
    Ok((info.VolumeSerialNumber, info.FileId.Identifier))
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
