// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Windows loader references pinned before exported addresses can be published.

#![allow(unsafe_code)]

use crate::{Error, ErrorKind};
use std::ffi::{CStr, OsStr, OsString, c_void};
use std::os::windows::ffi::OsStrExt;
use std::ptr::{self, NonNull};
use windows_sys::Win32::Foundation::{ERROR_PROC_NOT_FOUND, FreeLibrary};
use windows_sys::Win32::System::LibraryLoader::{
    GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_PIN, GetModuleHandleExW,
    GetProcAddress, LoadLibraryExW,
};

pub(crate) struct ResidentModule(usize);

fn last_error(operation: &'static str) -> Error {
    super::native_error(operation, std::io::Error::last_os_error())
}

impl ResidentModule {
    pub(crate) unsafe fn acquire(name: &OsStr, loaded_only: bool) -> Result<Self, Error> {
        let mut wide: Vec<u16> = name.encode_wide().collect();
        if wide.contains(&0) {
            return Err(crate::error::error(
                ErrorKind::InvalidArgument,
                "module name contains NUL",
            ));
        }
        wide.push(0);
        let mut handle = ptr::null_mut();
        if loaded_only {
            // SAFETY: The terminated native name is live. Flags zero acquires
            // an owned reference, closing the enumeration/unload race.
            if unsafe { GetModuleHandleExW(0, wide.as_ptr(), &raw mut handle) } == 0 {
                return Err(last_error("GetModuleHandleExW resident module"));
            }
        } else {
            // SAFETY: The caller accepts foreign initialization by the OS loader.
            handle = unsafe { LoadLibraryExW(wide.as_ptr(), ptr::null_mut(), 0) };
            if handle.is_null() {
                return Err(last_error("LoadLibraryExW resident module"));
            }
        }
        let owner = Self(handle as usize);
        let mut pinned = ptr::null_mut();
        // SAFETY: The acquired reference keeps the image live while pinning by
        // an address within it. PIN establishes residency until process exit.
        if unsafe {
            GetModuleHandleExW(
                GET_MODULE_HANDLE_EX_FLAG_PIN | GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                handle.cast(),
                &raw mut pinned,
            )
        } == 0
        {
            return Err(last_error("GetModuleHandleExW pin module"));
        }
        Ok(owner)
    }

    pub(crate) fn id(&self) -> usize {
        self.0
    }

    pub(crate) unsafe fn symbol(&self, name: &CStr) -> Result<Option<NonNull<c_void>>, Error> {
        // SAFETY: The owned image is pinned and export name is terminated.
        let address = unsafe { GetProcAddress(self.0 as *mut c_void, name.as_ptr().cast()) };
        if let Some(address) = address {
            return Ok(NonNull::new(address as *mut c_void));
        }
        let source = std::io::Error::last_os_error();
        if source
            .raw_os_error()
            .map(|code| u32::from_ne_bytes(code.to_ne_bytes()))
            == Some(ERROR_PROC_NOT_FOUND)
        {
            return Ok(None);
        }
        Err(super::native_error("GetProcAddress", source))
    }
}

impl Drop for ResidentModule {
    fn drop(&mut self) {
        // SAFETY: One loader reference is owned; process pinning keeps exported
        // code valid even after this reference is released.
        unsafe { FreeLibrary(self.0 as *mut c_void) };
    }
}

pub(crate) fn loaded_names() -> Result<Vec<OsString>, Error> {
    Err(crate::error::error(
        ErrorKind::Unsupported,
        "Windows loaded-module discovery is not implemented",
    ))
}
