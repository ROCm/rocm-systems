// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Windows CPU backing, pagefile sections, and placeholder reservations.
//! GPU virtual addresses and GPU backing remain provider-owned.

#![allow(unsafe_code)]

mod regions;
use super::{native_error, process_identity::ProcessIdentity};
use crate::memory::constraints::{AllocationDesc, AllocationLimits, CheckedRange, RangeTail};
use crate::memory::host::{HostMemoryCapabilities, HostReservationCapabilities};
use crate::memory::storage::Shared;
use crate::{Error, ErrorKind};
use regions::{Operations, Regions};
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::{io, ptr};
use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_EXTENDED_PARAMETER_0, MEM_EXTENDED_PARAMETER_1, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE,
    MEM_REPLACE_PLACEHOLDER, MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS,
    MapViewOfFile3, MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READONLY,
    PAGE_READWRITE, SEC_COMMIT, UnmapViewOfFile2, VirtualAlloc2, VirtualFree,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

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

fn address_requirements(alignment: usize) -> MEM_ADDRESS_REQUIREMENTS {
    MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: ptr::null_mut(),
        HighestEndingAddress: ptr::null_mut(),
        Alignment: alignment.max(system_info().dwAllocationGranularity as usize),
    }
}

fn address_parameter(requirements: &mut MEM_ADDRESS_REQUIREMENTS) -> MEM_EXTENDED_PARAMETER {
    MEM_EXTENDED_PARAMETER {
        Anonymous1: MEM_EXTENDED_PARAMETER_0 {
            _bitfield: MemExtendedParameterAddressRequirements as u64,
        },
        Anonymous2: MEM_EXTENDED_PARAMETER_1 {
            Pointer: ptr::from_mut(requirements).cast(),
        },
    }
}

enum Storage {
    Private,
    Section(OwnedHandle),
}

pub(crate) struct HostAllocation {
    address: usize,
    size: u64,
    process: ProcessIdentity,
    storage: Storage,
}

impl HostAllocation {
    pub(crate) fn create(size: u64, alignment: u64, process: u32) -> Result<Shared<Self>, Error> {
        Self::create_backing(size, alignment, process, false)
    }

    pub(crate) fn create_remappable(
        size: u64,
        alignment: u64,
        process: u32,
    ) -> Result<Shared<Self>, Error> {
        Self::create_backing(size, alignment, process, true)
    }

    fn create_backing(
        size: u64,
        alignment: u64,
        process: u32,
        remappable: bool,
    ) -> Result<Shared<Self>, Error> {
        if process != std::process::id() {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "foreign process host allocation",
            });
        }
        let limits = host_memory_capabilities()?.allocation;
        if !limits.supports(AllocationDesc { size, alignment })
            || usize::try_from(alignment).is_err()
        {
            return Err(Error::Operation {
                kind: ErrorKind::InvalidArgument,
                detail: "invalid Windows allocation geometry",
            });
        }
        let alignment = usize::try_from(alignment).map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "alignment exceeds address width",
        })?;
        let native_size = usize::try_from(size).map_err(|_| Error::Operation {
            kind: ErrorKind::InvalidArgument,
            detail: "size exceeds address width",
        })?;
        let process = ProcessIdentity::capture()
            .map_err(|error| native_error("host process identity", error))?;
        let owner = Shared::try_new_uninit()?;
        let mut requirements = address_requirements(alignment);
        let mut parameter = address_parameter(&mut requirements);
        let (mapped, storage) = if remappable {
            #[allow(
                clippy::cast_possible_truncation,
                reason = "Win32 takes separate high and low words of the section size"
            )]
            let size_low = size as u32;
            // SAFETY: An unnamed pagefile section acquires new committed backing.
            let section = unsafe {
                CreateFileMappingW(
                    INVALID_HANDLE_VALUE,
                    ptr::null(),
                    PAGE_READWRITE | SEC_COMMIT,
                    (size >> 32) as u32,
                    size_low,
                    ptr::null(),
                )
            };
            if section.is_null() {
                return Err(native_error(
                    "CreateFileMappingW host backing",
                    io::Error::last_os_error(),
                ));
            }
            // SAFETY: The successful native call transferred one handle to us.
            let section = unsafe { OwnedHandle::from_raw_handle(section) };
            // SAFETY: The fresh section covers size bytes and requirements live
            // through this synchronous acquisition. No existing VA is replaced.
            let mapped = unsafe {
                MapViewOfFile3(
                    section.as_raw_handle(),
                    GetCurrentProcess(),
                    ptr::null(),
                    0,
                    native_size,
                    0,
                    PAGE_READWRITE,
                    &raw mut parameter,
                    1,
                )
            };
            (mapped.Value, Storage::Section(section))
        } else {
            // SAFETY: This acquires fresh CPU backing without replacing VA.
            let mapped = unsafe {
                VirtualAlloc2(
                    GetCurrentProcess(),
                    ptr::null(),
                    native_size,
                    MEM_RESERVE | MEM_COMMIT,
                    PAGE_READWRITE,
                    &raw mut parameter,
                    1,
                )
            };
            (mapped, Storage::Private)
        };
        if mapped.is_null() {
            return Err(native_error(
                "Windows host mapping",
                io::Error::last_os_error(),
            ));
        }
        Ok(owner.write(Self {
            address: mapped as usize,
            size,
            process,
            storage,
        }))
    }

    pub(crate) fn is_remappable(&self) -> bool {
        matches!(self.storage, Storage::Section(_))
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
            // SAFETY: This is the exact acquired view/private allocation base.
            // Failure keeps all fields live so explicit cleanup can retry.
            let released = unsafe {
                match self.storage {
                    Storage::Private => VirtualFree(self.address as *mut _, 0, MEM_RELEASE),
                    Storage::Section(_) => UnmapViewOfFile2(
                        GetCurrentProcess(),
                        MEMORY_MAPPED_VIEW_ADDRESS {
                            Value: self.address as *mut _,
                        },
                        0,
                    ),
                }
            };
            if released == 0 {
                return Err(native_error(
                    "Windows host unmapping",
                    io::Error::last_os_error(),
                ));
            }
            self.address = 0;
            self.size = 0;
            self.storage = Storage::Private;
        }
        Ok(())
    }
}

impl Drop for HostAllocation {
    fn drop(&mut self) {
        if let Err(error) = self.free() {
            crate::memory::device::record_rollback(error, self.size);
        }
        // A live mapped view retains its own native section reference even
        // if unmapping failed and the section handle is closed here.
    }
}

pub(crate) struct HostReservation {
    base: usize,
    size: usize,
    regions: Regions,
    process: ProcessIdentity,
}

impl HostReservation {
    pub(crate) fn create(
        size: usize,
        alignment: usize,
        unit: usize,
        process: u32,
    ) -> io::Result<Self> {
        if process != std::process::id() {
            return Err(io::ErrorKind::Unsupported.into());
        }
        let page = usize::try_from(host_page_size().map_err(io::Error::other)?)
            .map_err(|_| io::ErrorKind::InvalidInput)?;
        let unit = if unit == 0 {
            system_info().dwAllocationGranularity as usize
        } else {
            unit
        };
        if size == 0
            || size % page != 0
            || size > isize::MAX as usize
            || !alignment.is_power_of_two()
            || alignment < page
            || !unit.is_power_of_two()
            || unit < page
        {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let process = ProcessIdentity::capture()?;
        let regions = Regions::new(size, unit)?;
        let mut requirements = address_requirements(alignment);
        let mut parameter = address_parameter(&mut requirements);
        // SAFETY: A new placeholder reservation owns fresh, inaccessible VA.
        // All metadata needed for future native views has already been reserved.
        let base = unsafe {
            VirtualAlloc2(
                GetCurrentProcess(),
                ptr::null(),
                size,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                &raw mut parameter,
                1,
            )
        };
        if base.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Self {
            base: base as usize,
            size,
            regions,
            process,
        })
    }

    pub(crate) fn extent(&self) -> (usize, usize) {
        (self.base, self.size)
    }
    pub(crate) fn mapping_granularity(&self) -> usize {
        self.regions.unit()
    }

    pub(crate) fn replace(
        &mut self,
        offset: u64,
        length: u64,
        source: &HostAllocation,
        source_offset: u64,
        writable: bool,
    ) -> io::Result<()> {
        self.process.check()?;
        source.check_process()?;
        let (offset, length) = self.regions.validate(offset, length)?;
        let page = host_page_size().map_err(io::Error::other)?;
        if source.address == 0
            || source_offset % self.regions.unit() as u64 != 0
            || CheckedRange::new(
                source.size,
                source_offset,
                length as u64,
                page,
                RangeTail::WholeUnits,
            )
            .is_none()
        {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let Storage::Section(section) = &source.storage else {
            return Err(io::ErrorKind::Unsupported.into());
        };
        self.regions.replace(
            offset,
            length,
            source_offset,
            writable,
            &mut NativeViews {
                base: self.base,
                section: Some(section),
            },
        )
    }

    pub(crate) fn clear(&mut self, offset: u64, length: u64) -> io::Result<()> {
        self.process.check()?;
        let (offset, length) = self.regions.validate(offset, length)?;
        self.regions.clear(
            offset,
            length,
            &mut NativeViews {
                base: self.base,
                section: None,
            },
        )
    }

    pub(crate) fn free(&mut self) -> io::Result<()> {
        self.process.check()?;
        self.regions.free(&mut NativeViews {
            base: self.base,
            section: None,
        })
    }
}

impl Drop for HostReservation {
    fn drop(&mut self) {
        if let Err(error) = self.free() {
            crate::memory::device::record_rollback(
                native_error("Windows reservation release", error),
                0,
            );
        }
    }
}

struct NativeViews<'a> {
    base: usize,
    section: Option<&'a OwnedHandle>,
}

impl Operations for NativeViews<'_> {
    fn split(&mut self, offset: usize, prefix: usize) -> io::Result<()> {
        // SAFETY: The region owner supplies a complete placeholder prefix;
        // preserving it splits native ownership without releasing the VA.
        if unsafe {
            VirtualFree(
                (self.base + offset) as *mut _,
                prefix,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            )
        } == 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
    fn map(&mut self, offset: usize, length: usize, source: u64, writable: bool) -> io::Result<()> {
        let section = self.section.ok_or(io::ErrorKind::InvalidInput)?;
        // SAFETY: This replaces exactly one owned placeholder. Windows permits
        // page-aligned offsets/bases with MEM_REPLACE_PLACEHOLDER. Source backing
        // covers the entire range and is borrowed until mapping completes.
        let mapped = unsafe {
            MapViewOfFile3(
                section.as_raw_handle(),
                GetCurrentProcess(),
                (self.base + offset) as *const _,
                source,
                length,
                MEM_REPLACE_PLACEHOLDER,
                if writable {
                    PAGE_READWRITE
                } else {
                    PAGE_READONLY
                },
                ptr::null_mut(),
                0,
            )
        };
        if mapped.Value.is_null() {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
    fn unmap(&mut self, offset: usize, preserve: bool) -> io::Result<()> {
        // SAFETY: The exact view base is recorded. Preserving the placeholder
        // retains VA; terminal release removes only this independently owned view.
        if unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: (self.base + offset) as *mut _,
                },
                if preserve {
                    MEM_PRESERVE_PLACEHOLDER
                } else {
                    0
                },
            )
        } == 0
        {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
    fn release_placeholder(&mut self, offset: usize) -> io::Result<()> {
        // SAFETY: This is one entire still-owned placeholder. Released holes
        // have already been removed from the record set and cannot reach here.
        if unsafe { VirtualFree((self.base + offset) as *mut _, 0, MEM_RELEASE) } == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(())
        }
    }
}

pub(crate) fn host_memory_capabilities() -> Result<HostMemoryCapabilities, Error> {
    let page = host_page_size()?;
    let system = system_info();
    Ok(HostMemoryCapabilities {
        allocation: AllocationLimits {
            alignment: page,
            granularity: page,
            maximum_size: isize::MAX as u64,
        },
        remappable_backing: true,
        reservation: Some(HostReservationCapabilities {
            reservation_granularity: page,
            mapping_granularity: page,
            maximum_size: (isize::MAX as u64) & !(page - 1),
            maximum_alignment: 1_u64 << (usize::BITS - 2),
            minimum_address: system.lpMinimumApplicationAddress as u64,
            maximum_address: system.lpMaximumApplicationAddress as u64,
        }),
    })
}
