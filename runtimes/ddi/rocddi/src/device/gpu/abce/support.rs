// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host atomics, thread-local packet storage and explicit topology loading.

use super::abi::{abce_topology_data_t, abce_topology_load_result_t};
#[cfg(target_os = "linux")]
use super::{abce_topology_data_initialize, abce_topology_data_set_device};
use core::ffi::c_char;
#[cfg(not(unix))]
use std::cell::RefCell;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering, fence};

#[must_use]
pub(super) unsafe extern "C" fn abce_host_load_relaxed_u64(ptr: *const u64) -> u64 {
    unsafe { AtomicU64::from_ptr(ptr.cast_mut()).load(Ordering::Relaxed) }
}
#[must_use]
pub(super) unsafe extern "C" fn abce_host_load_acquire_u64(ptr: *const u64) -> u64 {
    unsafe { AtomicU64::from_ptr(ptr.cast_mut()).load(Ordering::Acquire) }
}
pub(super) unsafe extern "C" fn abce_host_store_relaxed_u64(ptr: *mut u64, value: u64) {
    unsafe { AtomicU64::from_ptr(ptr).store(value, Ordering::Relaxed) }
}
pub(super) unsafe extern "C" fn abce_host_store_release_u64(ptr: *mut u64, value: u64) {
    unsafe { AtomicU64::from_ptr(ptr).store(value, Ordering::Release) }
}
pub(super) unsafe extern "C" fn abce_host_store_release_u32(ptr: *mut u32, value: u32) {
    unsafe { AtomicU32::from_ptr(ptr).store(value, Ordering::Release) }
}
pub(super) unsafe extern "C" fn abce_host_cas_relaxed_u64(
    ptr: *mut u64,
    expected: *mut u64,
    desired: u64,
) -> bool {
    unsafe {
        match AtomicU64::from_ptr(ptr).compare_exchange(
            *expected,
            desired,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => true,
            Err(value) => {
                *expected = value;
                false
            }
        }
    }
}
/// Issues the C API release fence.
///
/// # Safety
/// No pointer preconditions; the caller supplies the surrounding synchronization.
pub unsafe extern "C" fn abce_host_fence_release() {
    fence(Ordering::Release);
}
pub(super) unsafe extern "C" fn abce_host_fence_seq_cst() {
    fence(Ordering::SeqCst);
}

pub(super) unsafe extern "C" fn abce_ring_host_store_fence() {
    // A release fence alone does not drain x86 write-combining stores.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_mm_sfence();
    }
    #[cfg(not(target_arch = "x86_64"))]
    fence(Ordering::SeqCst);
}

#[cfg(not(unix))]
thread_local! {
    // Word storage preserves the packet emitter's alignment precondition.
    static PACKETS: RefCell<Vec<u64>> = const { RefCell::new(Vec::new()) };
}

fn grow_packets(words: &mut Vec<u64>, bytes: usize) -> *mut c_char {
    let Some(count) = bytes.checked_add(7).map(|n| n / 8) else {
        return core::ptr::null_mut();
    };
    if count > words.len() {
        let additional = count - words.len();
        if words.try_reserve(additional).is_err() {
            return core::ptr::null_mut();
        }
        words.resize(count, 0);
    }
    if words.is_empty() {
        core::ptr::null_mut()
    } else {
        words.as_mut_ptr().cast()
    }
}

#[cfg(unix)]
mod packet_tls {
    use super::{Ordering, c_char, grow_packets};
    use std::sync::atomic::{AtomicI32, AtomicU32};

    static KEY: AtomicU32 = AtomicU32::new(0);
    static ERROR: AtomicI32 = AtomicI32::new(0);
    static mut ONCE: libc::pthread_once_t = libc::PTHREAD_ONCE_INIT;

    unsafe extern "C" fn destroy(pointer: *mut core::ffi::c_void) {
        if !pointer.is_null() {
            unsafe {
                drop(Box::from_raw(pointer.cast::<Vec<u64>>()));
            }
        }
    }

    extern "C" fn initialize() {
        let mut key = 0;
        let error = unsafe { libc::pthread_key_create(&raw mut key, Some(destroy)) };
        KEY.store(key, Ordering::Relaxed);
        ERROR.store(error, Ordering::Relaxed);
    }

    pub(super) unsafe fn get(bytes: usize) -> *mut c_char {
        // POSIX key destructors run after C++ thread_local destructors. Rust
        // thread_local Vec destruction would make a later C++ destructor's
        // map abort with AccessError. pthread_once also handles fork during
        // first initialization, matching the original header implementation.
        unsafe {
            if libc::pthread_once(core::ptr::addr_of_mut!(ONCE), initialize) != 0
                || ERROR.load(Ordering::Relaxed) != 0
            {
                return core::ptr::null_mut();
            }
            let key = KEY.load(Ordering::Relaxed);
            let mut state = libc::pthread_getspecific(key).cast::<Vec<u64>>();
            if state.is_null() {
                state = std::alloc::alloc(std::alloc::Layout::new::<Vec<u64>>()).cast();
                if state.is_null() {
                    return core::ptr::null_mut();
                }
                state.write(Vec::new());
                if libc::pthread_setspecific(key, state.cast()) != 0 {
                    destroy(state.cast());
                    return core::ptr::null_mut();
                }
            }
            // This thread exclusively owns the Vec. No borrow crosses any
            // policy or ring callback; only the packet view leaves this call.
            grow_packets(&mut *state, bytes)
        }
    }
}

#[must_use]
pub(super) unsafe extern "C" fn abce_acquire_packet_scratch(bytes: usize) -> *mut c_char {
    #[cfg(unix)]
    {
        unsafe { packet_tls::get(bytes) }
    }
    #[cfg(not(unix))]
    {
        PACKETS
            .try_with(|cell| {
                cell.try_borrow_mut()
                    .map_or(core::ptr::null_mut(), |mut words| {
                        grow_packets(&mut words, bytes)
                    })
            })
            .unwrap_or(core::ptr::null_mut())
    }
}

#[cfg(target_os = "linux")]
fn properties(path: &std::path::Path) -> Option<std::collections::BTreeMap<String, u64>> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut words = text.split_whitespace();
    let mut result = std::collections::BTreeMap::new();
    while let (Some(key), Some(value)) = (words.next(), words.next()) {
        let Ok(value) = value.parse() else { break };
        result.insert(key.to_owned(), value);
    }
    Some(result)
}

/// Loads explicitly requested KFD topology, without opening a native device.
///
/// # Safety
/// A nonnull path points to a terminated C string. Output pointers are writable,
/// correctly aligned records, exclusively
/// owned by the caller during this call.
#[allow(
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "bounded node indices and byte fields follow the existing C topology contract"
)]
pub unsafe extern "C" fn abce_topology_load_from_kfd(
    base_path: *const c_char,
    out_topology: *mut abce_topology_data_t,
    out_result: *mut abce_topology_load_result_t,
) -> bool {
    #[cfg(not(target_os = "linux"))]
    let _ = base_path;
    if out_topology.is_null() || out_result.is_null() {
        return false;
    }
    unsafe {
        out_result.write(core::mem::zeroed());
    }
    #[cfg(target_os = "linux")]
    unsafe {
        use std::os::unix::ffi::OsStrExt;
        let root = if base_path.is_null() {
            std::path::Path::new("/sys/class/kfd/kfd/topology/nodes")
        } else {
            std::path::Path::new(std::ffi::OsStr::from_bytes(
                std::ffi::CStr::from_ptr(base_path).to_bytes(),
            ))
        };
        let Ok(entries) = std::fs::read_dir(root) else {
            return false;
        };
        let mut loaded: abce_topology_data_t = core::mem::zeroed();
        abce_topology_data_initialize(&raw mut loaded);
        for entry in entries.flatten() {
            let filename = entry.file_name();
            let Some(name) = filename.to_str() else {
                continue;
            };
            let Ok(node) = name.parse::<usize>() else {
                continue;
            };
            if node >= 16 || entry.path().as_os_str().len() >= 512 {
                continue;
            }
            let Some(p) = properties(&entry.path().join("properties")) else {
                continue;
            };
            let get = |key: &str| p.get(key).copied().unwrap_or(0);
            if get("simd_count") == 0 {
                continue;
            }
            let mut physical_id = -1;
            if let Some(location) = p.get("location_id").filter(|&&n| n != u64::MAX) {
                let bdf = format!(
                    "{:04x}:{:02x}:{:02x}.{:01x}",
                    get("domain"),
                    (location >> 8) & 255,
                    (location >> 3) & 31,
                    location & 7
                );
                if let Ok(text) =
                    std::fs::read_to_string(format!("/sys/bus/pci/devices/{bdf}/xgmi_physical_id"))
                {
                    physical_id = text.trim().parse().unwrap_or(-1);
                }
            }
            abce_topology_data_set_device(
                &raw mut loaded,
                node as u32,
                physical_id,
                get("hive_id"),
            );
            for links in ["io_links", "p2p_links"] {
                let Ok(entries) = std::fs::read_dir(entry.path().join(links)) else {
                    continue;
                };
                for link in entries.flatten() {
                    if link
                        .file_name()
                        .to_str()
                        .and_then(|s| s.parse::<u32>().ok())
                        .is_none()
                    {
                        continue;
                    }
                    let Some(p) = properties(&link.path().join("properties")) else {
                        continue;
                    };
                    if let (Some(&dst), Some(&mask)) =
                        (p.get("node_to"), p.get("recommended_sdma_engine_id_mask"))
                    {
                        if dst < 16 && mask != 0 {
                            loaded.recommended_mask[node][dst as usize] = mask;
                        }
                    }
                }
            }
            (*out_result).found_gpu = true;
            let target = get("gfx_target_version");
            if (*out_result).total_sdma == 0 && target != 0 {
                (*out_result).gfx_minor = ((target / 100) % 100) as u8;
                (*out_result).total_sdma =
                    get("num_sdma_engines").wrapping_add(get("num_sdma_xgmi_engines")) as u8;
                (*out_result).num_non_xgmi_sdma = get("num_sdma_engines") as u8;
            }
        }
        if (*out_result).found_gpu {
            out_topology.write(loaded);
        }
    }
    unsafe { (*out_result).found_gpu }
}
