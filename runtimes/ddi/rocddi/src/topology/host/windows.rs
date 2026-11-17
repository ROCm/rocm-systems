// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Independent Win32 observations of host processors and physical memory.

use super::{HostTopologyGraphSnapshot, windows_records};
use windows_sys::Win32::Foundation::{ERROR_INSUFFICIENT_BUFFER, GetLastError};
use windows_sys::Win32::System::SystemInformation::{
    CACHE_RELATIONSHIP, GROUP_AFFINITY, GetLogicalProcessorInformationEx, GlobalMemoryStatusEx,
    MEMORYSTATUSEX, NUMA_NODE_RELATIONSHIP, PROCESSOR_RELATIONSHIP, RelationAll,
    SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
};

// The decoder implements the native 64-bit SDK record layout, not WOW64.
const _: () = {
    assert!(std::mem::size_of::<GROUP_AFFINITY>() == 16);
    assert!(std::mem::offset_of!(GROUP_AFFINITY, Group) == 8);
    assert!(std::mem::offset_of!(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX, Anonymous) == 8);
    assert!(std::mem::offset_of!(PROCESSOR_RELATIONSHIP, GroupCount) == 22);
    assert!(std::mem::offset_of!(PROCESSOR_RELATIONSHIP, GroupMask) == 24);
    assert!(std::mem::offset_of!(NUMA_NODE_RELATIONSHIP, GroupCount) == 22);
    assert!(std::mem::offset_of!(NUMA_NODE_RELATIONSHIP, Anonymous) == 24);
    assert!(std::mem::offset_of!(CACHE_RELATIONSHIP, GroupCount) == 30);
    assert!(std::mem::offset_of!(CACHE_RELATIONSHIP, Anonymous) == 32);
};

#[allow(unsafe_code)]
fn processors() -> Option<HostTopologyGraphSnapshot> {
    let mut length = 0_u32;
    // SAFETY: Null storage requests the required byte length.
    if unsafe {
        GetLogicalProcessorInformationEx(RelationAll, std::ptr::null_mut(), &raw mut length)
    } != 0
        || unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER
    {
        return None;
    }
    for _ in 0..4 {
        let bytes = usize::try_from(length).ok()?;
        if bytes == 0 {
            return None;
        }
        let words = bytes.div_ceil(8);
        let mut buffer = Vec::<u64>::new();
        buffer.try_reserve_exact(words).ok()?;
        buffer.resize(words, 0);
        // SAFETY: The initialized storage is aligned for native records and
        // covers the requested length. The API updates length on success or
        // insufficient-buffer failure; other failures do not expose contents.
        if unsafe {
            GetLogicalProcessorInformationEx(
                RelationAll,
                buffer.as_mut_ptr().cast(),
                &raw mut length,
            )
        } != 0
        {
            let returned = usize::try_from(length).ok()?;
            if returned > bytes {
                return None;
            }
            // SAFETY: Successful returned bytes are within initialized storage.
            let records = unsafe { std::slice::from_raw_parts(buffer.as_ptr().cast(), returned) };
            return Some(windows_records::decode(records));
        }
        // SAFETY: Reads the status of the failed call on this thread.
        if unsafe { GetLastError() } != ERROR_INSUFFICIENT_BUFFER {
            return None;
        }
    }
    None
}

#[allow(unsafe_code)]
pub(super) fn graph_snapshot() -> HostTopologyGraphSnapshot {
    let mut snapshot = processors().unwrap_or_else(|| {
        let mut snapshot = windows_records::decode(&[]);
        snapshot.caches.complete = false;
        snapshot.domains_complete = false;
        snapshot
    });
    let mut memory = MEMORYSTATUSEX {
        dwLength: u32::try_from(std::mem::size_of::<MEMORYSTATUSEX>()).unwrap_or(0),
        ..MEMORYSTATUSEX::default()
    };
    // SAFETY: The initialized record declares its complete SDK size.
    if unsafe { GlobalMemoryStatusEx(&raw mut memory) } != 0 {
        snapshot.memory_bytes = usize::try_from(memory.ullTotalPhys)
            .ok()
            .filter(|size| *size != 0);
    }
    snapshot
}
