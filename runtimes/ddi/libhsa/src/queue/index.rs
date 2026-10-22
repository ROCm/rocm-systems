// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Lock-free public queue index operations and their live-handle contract.

use super::*;

/// Applies one atomic index operation without exporting a queue reference.
///
/// # Safety
/// The public queue handle and its index mapping must remain live throughout
/// `operation`; the caller must prevent concurrent destruction.
#[inline]
unsafe fn with_queue_index<R>(
    queue: *const HsaQueue,
    offset: usize,
    operation: impl for<'a> FnOnce(&'a AtomicU64) -> R,
) -> Option<R> {
    if queue.is_null() {
        return None;
    }
    let address = if queue as usize & COUNTED_QUEUE_HANDLE_BIT != 0 {
        // SAFETY: A live counted handle points at CountedQueuePublic::header,
        // whose preceding word holds the stable hardware queue address.
        let hardware = unsafe {
            (*queue
                .cast::<u8>()
                .sub(COUNTED_QUEUE_HANDLE_BIT)
                .cast::<CountedQueuePublic>())
            .hardware_queue
        };
        hardware.checked_add(offset)?
    } else if queue as usize & SDMA_QUEUE_HANDLE_BIT != 0 {
        // SAFETY: A live SDMA header begins 128 bytes into its private
        // control. The private words borrow rocddi's aligned index mappings.
        let control = unsafe {
            &*queue
                .cast::<u8>()
                .sub(SDMA_QUEUE_HANDLE_BIT)
                .cast::<SdmaQueueControl>()
        };
        match offset {
            READ_INDEX_OFFSET => control.read_index_host_address,
            WRITE_INDEX_OFFSET => control.write_index_host_address,
            _ => return None,
        }
    } else {
        (queue as usize).checked_add(offset)?
    };
    // SAFETY: The caller retains a live queue handle. AQL and soft controls
    // contain indices at the ABI offsets; counted and SDMA handles resolve to
    // the native index mapping retained by their queue owner.
    let index = unsafe { &*(address as *const AtomicU64) };
    Some(operation(index))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_load_read_index_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_queue_load_read_index_relaxed(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, READ_INDEX_OFFSET, |index| {
            index.load(Ordering::Relaxed)
        })
    }
    .unwrap_or(0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_load_read_index_scacquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_queue_load_read_index_scacquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, READ_INDEX_OFFSET, |index| {
            index.load(Ordering::Acquire)
        })
    }
    .unwrap_or(0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_load_read_index_acquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_queue_load_read_index_acquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe { hsa_queue_load_read_index_scacquire(queue) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_load_write_index_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_queue_load_write_index_relaxed(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, WRITE_INDEX_OFFSET, |index| {
            index.load(Ordering::Relaxed)
        })
    }
    .unwrap_or(0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_load_write_index_scacquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_queue_load_write_index_scacquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, WRITE_INDEX_OFFSET, |index| {
            index.load(Ordering::Acquire)
        })
    }
    .unwrap_or(0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_load_write_index_acquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub unsafe extern "C" fn hsa_queue_load_write_index_acquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe { hsa_queue_load_write_index_scacquire(queue) }
}

unsafe fn queue_store(queue: *const HsaQueue, offset: usize, value: u64, order: Ordering) {
    // SAFETY: The caller owns a live queue while accessing its index.
    let _ = unsafe { with_queue_index(queue, offset, |index| index.store(value, order)) };
}

unsafe fn queue_compare_exchange(
    queue: *const HsaQueue,
    expected: u64,
    value: u64,
    success: Ordering,
    failure: Ordering,
) -> u64 {
    // SAFETY: The caller owns a live queue while reserving packet slots.
    unsafe {
        with_queue_index(queue, WRITE_INDEX_OFFSET, |index| {
            index
                .compare_exchange(expected, value, success, failure)
                .unwrap_or_else(|observed| observed)
        })
    }
    .unwrap_or(0)
}

unsafe fn queue_add(queue: *const HsaQueue, value: u64, order: Ordering) -> u64 {
    // SAFETY: The caller owns a live queue while reserving packet slots.
    unsafe {
        with_queue_index(queue, WRITE_INDEX_OFFSET, |index| {
            index.fetch_add(value, order)
        })
    }
    .unwrap_or(0)
}

macro_rules! queue_store_entry {
    ($name:ident, $offset:expr, $order:expr) => {
        #[cfg_attr(all(target_arch = "x86_64", not(test)), unsafe(export_name = concat!("__hsa_impl_", stringify!($name))))]
        #[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
        pub unsafe extern "C" fn $name(queue: *const HsaQueue, value: u64) {
            // SAFETY: The public entry point preserves the HSA queue contract.
            unsafe { queue_store(queue, $offset, value, $order) };
        }
    };
}

macro_rules! queue_compare_exchange_entry {
    ($name:ident, $success:expr, $failure:expr) => {
        #[cfg_attr(all(target_arch = "x86_64", not(test)), unsafe(export_name = concat!("__hsa_impl_", stringify!($name))))]
        #[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
        pub unsafe extern "C" fn $name(queue: *const HsaQueue, expected: u64, value: u64) -> u64 {
            // SAFETY: The public entry point preserves the HSA queue contract.
            unsafe { queue_compare_exchange(queue, expected, value, $success, $failure) }
        }
    };
}

macro_rules! queue_add_entry {
    ($name:ident, $order:expr) => {
        #[cfg_attr(all(target_arch = "x86_64", not(test)), unsafe(export_name = concat!("__hsa_impl_", stringify!($name))))]
        #[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
        pub unsafe extern "C" fn $name(queue: *const HsaQueue, value: u64) -> u64 {
            // SAFETY: The public entry point preserves the HSA queue contract.
            unsafe { queue_add(queue, value, $order) }
        }
    };
}

queue_store_entry!(
    hsa_queue_store_write_index_relaxed,
    WRITE_INDEX_OFFSET,
    Ordering::Relaxed
);
queue_store_entry!(
    hsa_queue_store_write_index_screlease,
    WRITE_INDEX_OFFSET,
    Ordering::Release
);
queue_store_entry!(
    hsa_queue_store_write_index_release,
    WRITE_INDEX_OFFSET,
    Ordering::Release
);

queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_scacq_screl,
    Ordering::AcqRel,
    Ordering::Acquire
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_acq_rel,
    Ordering::AcqRel,
    Ordering::Acquire
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_scacquire,
    Ordering::Acquire,
    Ordering::Acquire
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_acquire,
    Ordering::Acquire,
    Ordering::Acquire
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_relaxed,
    Ordering::Relaxed,
    Ordering::Relaxed
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_screlease,
    Ordering::Release,
    Ordering::Relaxed
);
queue_compare_exchange_entry!(
    hsa_queue_cas_write_index_release,
    Ordering::Release,
    Ordering::Relaxed
);

queue_add_entry!(hsa_queue_add_write_index_scacq_screl, Ordering::AcqRel);
queue_add_entry!(hsa_queue_add_write_index_acq_rel, Ordering::AcqRel);
queue_add_entry!(hsa_queue_add_write_index_scacquire, Ordering::Acquire);
queue_add_entry!(hsa_queue_add_write_index_acquire, Ordering::Acquire);
queue_add_entry!(hsa_queue_add_write_index_relaxed, Ordering::Relaxed);
queue_add_entry!(hsa_queue_add_write_index_screlease, Ordering::Release);
queue_add_entry!(hsa_queue_add_write_index_release, Ordering::Release);

queue_store_entry!(
    hsa_queue_store_read_index_relaxed,
    READ_INDEX_OFFSET,
    Ordering::Relaxed
);
queue_store_entry!(
    hsa_queue_store_read_index_screlease,
    READ_INDEX_OFFSET,
    Ordering::Release
);
queue_store_entry!(
    hsa_queue_store_read_index_release,
    READ_INDEX_OFFSET,
    Ordering::Release
);
