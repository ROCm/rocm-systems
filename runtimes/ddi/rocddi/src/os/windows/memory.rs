// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Windows CPU backing. GPU VA and native GPU allocation remain provider-owned.

#![allow(unsafe_code)]

use super::{native_error, process_identity::ProcessIdentity};
use crate::memory::constraints::{AllocationDesc, AllocationLimits};
use crate::memory::storage::Shared;
use crate::{Error, ErrorKind};
use std::{convert::Infallible, io, ptr};
use windows_sys::Win32::System::Memory::{
    MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER, MEM_EXTENDED_PARAMETER_0,
    MEM_EXTENDED_PARAMETER_1, MEM_RELEASE, MEM_RESERVE, MemExtendedParameterAddressRequirements,
    PAGE_READWRITE, VirtualAlloc2, VirtualFree,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};

fn system_info() -> SYSTEM_INFO {
    let mut info = SYSTEM_INFO::default();
    // SAFETY: GetSystemInfo fills a complete writable system-information slot.
    unsafe { GetSystemInfo(&raw mut info) };
    info
}

pub(crate) fn host_page_size() -> Result<u64, Error> {
    let page = u64::from(system_info().dwPageSize);
    if !page.is_power_of_two() {
        return Err(Error::Operation {
            kind: ErrorKind::DriverContract,
            detail: "invalid Windows host page size",
        });
    }
    Ok(page)
}

pub(crate) struct HostAllocation {
    address: usize,
    size: u64,
    process: ProcessIdentity,
}

impl HostAllocation {
    pub(crate) fn create(size: u64, alignment: u64, process: u32) -> Result<Shared<Self>, Error> {
        if process != std::process::id() {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "foreign process host allocation",
            });
        }
        let info = system_info();
        let limits = AllocationLimits {
            alignment: host_page_size()?,
            granularity: host_page_size()?,
            maximum_size: isize::MAX as u64,
        };
        if !limits.supports(AllocationDesc { size, alignment })
            || usize::try_from(alignment).is_err()
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "invalid Windows allocation geometry",
            });
        }
        let process = ProcessIdentity::capture()
            .map_err(|error| native_error("host process identity", error))?;
        let owner = Shared::try_new_uninit()?;
        let mut requirements = MEM_ADDRESS_REQUIREMENTS {
            LowestStartingAddress: ptr::null_mut(),
            HighestEndingAddress: ptr::null_mut(),
            Alignment: usize::try_from(alignment)
                .map_err(|_| Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "alignment exceeds host width",
                })?
                .max(info.dwAllocationGranularity as usize),
        };
        let mut parameter = MEM_EXTENDED_PARAMETER {
            Anonymous1: MEM_EXTENDED_PARAMETER_0 {
                _bitfield: MemExtendedParameterAddressRequirements as u64,
            },
            Anonymous2: MEM_EXTENDED_PARAMETER_1 {
                Pointer: (&raw mut requirements).cast(),
            },
        };
        // SAFETY: This acquires fresh committed CPU pages, never replaces an
        // existing range, and borrows the address requirements only for the call.
        let mapped = unsafe {
            VirtualAlloc2(
                ptr::null_mut(),
                ptr::null(),
                usize::try_from(size).map_err(|_| Error::Operation {
                    kind: ErrorKind::InvalidArgument,
                    detail: "allocation size exceeds host width",
                })?,
                MEM_RESERVE | MEM_COMMIT,
                PAGE_READWRITE,
                &raw mut parameter,
                1,
            )
        };
        if mapped.is_null() {
            return Err(native_error(
                "VirtualAlloc2 host allocation",
                io::Error::last_os_error(),
            ));
        }
        Ok(owner.write(Self {
            address: mapped as usize,
            size,
            process,
        }))
    }

    pub(crate) fn create_remappable(
        _size: u64,
        _alignment: u64,
        _process: u32,
    ) -> Result<Shared<Self>, Error> {
        Err(Error::Operation {
            kind: ErrorKind::Unsupported,
            detail: "Windows remappable host backing is not implemented",
        })
    }

    #[allow(clippy::unused_self, reason = "OS-selected owner contract")]
    pub(crate) fn is_remappable(&self) -> bool {
        false
    }
    pub(crate) fn check_process(&self) -> io::Result<()> {
        self.process.check()
    }
    pub(crate) fn extent(&self) -> (usize, u64) {
        (self.address, self.size)
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        self.check_process()
            .map_err(|error| native_error("host allocation process identity", error))?;
        if self.address != 0 {
            // SAFETY: This is the exact base returned by VirtualAlloc2. Zero
            // size releases the entire reservation; failure retains this owner.
            if unsafe { VirtualFree(self.address as *mut _, 0, MEM_RELEASE) } == 0 {
                return Err(native_error(
                    "VirtualFree host allocation",
                    io::Error::last_os_error(),
                ));
            }
            self.address = 0;
            self.size = 0;
        }
        Ok(())
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        let _ = self.free();
    }
}

/// No owner is constructed until placeholder replacement/restore is implemented.
pub(crate) struct HostReservation(Infallible);

impl HostReservation {
    pub(crate) fn create(_size: usize, _alignment: usize, _process: u32) -> io::Result<Self> {
        Err(io::ErrorKind::Unsupported.into())
    }
    pub(crate) fn extent(&self) -> (usize, usize) {
        match self.0 {}
    }
    pub(crate) fn replace(
        &mut self,
        _offset: u64,
        _length: u64,
        _source: &HostAllocation,
        _source_offset: u64,
        _writable: bool,
    ) -> io::Result<()> {
        match self.0 {}
    }
    pub(crate) fn clear(&mut self, _offset: u64, _length: u64) -> io::Result<()> {
        match self.0 {}
    }
    pub(crate) fn free(&mut self) -> io::Result<()> {
        match self.0 {}
    }
}

pub(crate) fn host_memory_capabilities()
-> Result<crate::memory::host::HostMemoryCapabilities, Error> {
    let page = host_page_size()?;
    Ok(crate::memory::host::HostMemoryCapabilities {
        allocation: AllocationLimits {
            alignment: page,
            granularity: page,
            maximum_size: isize::MAX as u64,
        },
        remappable_backing: false,
        reservation: None,
    })
}
