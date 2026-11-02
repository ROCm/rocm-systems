// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA queue index C entry points and atomic ordering variants.

use std::sync::atomic::Ordering;

use crate::ffi::HsaQueue;
use crate::queue::index::{QueueIndex, with_queue_index};

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_queue_load_read_index_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Loads the queue read index with relaxed ordering.
pub unsafe extern "C" fn hsa_queue_load_read_index_relaxed(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, QueueIndex::Read, |index| {
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
/// Loads the queue read index with the HSA SC-acquire contract.
pub unsafe extern "C" fn hsa_queue_load_read_index_scacquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, QueueIndex::Read, |index| {
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
/// Uses the deprecated acquire spelling for the queue read index.
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
/// Loads the queue write index with relaxed ordering.
pub unsafe extern "C" fn hsa_queue_load_write_index_relaxed(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, QueueIndex::Write, |index| {
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
/// Loads the queue write index with the HSA SC-acquire contract.
pub unsafe extern "C" fn hsa_queue_load_write_index_scacquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: The caller owns a live queue while accessing its index.
    unsafe {
        with_queue_index(queue, QueueIndex::Write, |index| {
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
/// Uses the deprecated acquire spelling for the queue write index.
pub unsafe extern "C" fn hsa_queue_load_write_index_acquire(queue: *const HsaQueue) -> u64 {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe { hsa_queue_load_write_index_scacquire(queue) }
}

/// Stores a queue index with the selected atomic ordering.
unsafe fn queue_store(queue: *const HsaQueue, index: QueueIndex, value: u64, order: Ordering) {
    // SAFETY: The caller owns a live queue while accessing its index.
    let _ = unsafe { with_queue_index(queue, index, |word| word.store(value, order)) };
}

/// Compares and exchanges a queue index with the selected ordering.
unsafe fn queue_compare_exchange(
    queue: *const HsaQueue,
    expected: u64,
    value: u64,
    success: Ordering,
    failure: Ordering,
) -> u64 {
    // SAFETY: The caller owns a live queue while reserving packet slots.
    unsafe {
        with_queue_index(queue, QueueIndex::Write, |index| {
            index
                .compare_exchange(expected, value, success, failure)
                .unwrap_or_else(|observed| observed)
        })
    }
    .unwrap_or(0)
}

/// Atomically adds to the queue write index with the selected ordering.
unsafe fn queue_add(queue: *const HsaQueue, value: u64, order: Ordering) -> u64 {
    // SAFETY: The caller owns a live queue while reserving packet slots.
    unsafe {
        with_queue_index(queue, QueueIndex::Write, |index| {
            index.fetch_add(value, order)
        })
    }
    .unwrap_or(0)
}

/// Generates the queue store entry ABI entry points.
macro_rules! queue_store_entry {
    ($name:ident, $index:expr, $order:expr) => {
        #[doc = concat!("Stores a queue index through `", stringify!($name), "`.")]
        #[cfg_attr(all(target_arch = "x86_64", not(test)), unsafe(export_name = concat!("__hsa_impl_", stringify!($name))))]
        #[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
        pub unsafe extern "C" fn $name(queue: *const HsaQueue, value: u64) {
            // SAFETY: The public entry point preserves the HSA queue contract.
            unsafe { queue_store(queue, $index, value, $order) };
        }
    };
}

/// Generates the queue compare exchange entry ABI entry points.
macro_rules! queue_compare_exchange_entry {
    ($name:ident, $success:expr, $failure:expr) => {
        #[doc = concat!("Conditionally exchanges a queue index through `", stringify!($name), "`.")]
        #[cfg_attr(all(target_arch = "x86_64", not(test)), unsafe(export_name = concat!("__hsa_impl_", stringify!($name))))]
        #[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
        pub unsafe extern "C" fn $name(queue: *const HsaQueue, expected: u64, value: u64) -> u64 {
            // SAFETY: The public entry point preserves the HSA queue contract.
            unsafe { queue_compare_exchange(queue, expected, value, $success, $failure) }
        }
    };
}

/// Generates the queue add entry ABI entry points.
macro_rules! queue_add_entry {
    ($name:ident, $order:expr) => {
        #[doc = concat!("Atomically increments a queue index through `", stringify!($name), "`.")]
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
    QueueIndex::Write,
    Ordering::Relaxed
);
queue_store_entry!(
    hsa_queue_store_write_index_screlease,
    QueueIndex::Write,
    Ordering::Release
);
queue_store_entry!(
    hsa_queue_store_write_index_release,
    QueueIndex::Write,
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
    QueueIndex::Read,
    Ordering::Relaxed
);
queue_store_entry!(
    hsa_queue_store_read_index_screlease,
    QueueIndex::Read,
    Ordering::Release
);
queue_store_entry!(
    hsa_queue_store_read_index_release,
    QueueIndex::Read,
    Ordering::Release
);
