// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Lock-free public queue index operations and their live-handle contract.

use super::*;

/// Native index selected by one HSA queue operation.
#[derive(Clone, Copy)]
pub(crate) enum QueueIndex {
    /// Device-maintained consumer index.
    Read,
    /// Producer-maintained reservation index.
    Write,
}

/// Applies one atomic index operation without exporting a queue reference.
///
/// # Safety
/// The public queue handle and its index mapping must remain live throughout
/// `operation`; the caller must prevent concurrent destruction.
#[inline]
pub(crate) unsafe fn with_queue_index<R>(
    queue: *const HsaQueue,
    index: QueueIndex,
    operation: impl for<'a> FnOnce(&'a AtomicU64) -> R,
) -> Option<R> {
    let _admission = crate::runtime::enter_public_raw_handle()?;
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
        // SAFETY: The counted handle retains this validated hardware control.
        match index {
            QueueIndex::Read => unsafe { AqlControlRegion::read_index_address_at(hardware) },
            QueueIndex::Write => unsafe { AqlControlRegion::write_index_address_at(hardware) },
        }
    } else if queue as usize & SDMA_QUEUE_HANDLE_BIT != 0 {
        // SAFETY: A live SDMA header begins 128 bytes into its private
        // control. The private words borrow rocddi's aligned index mappings.
        let control = unsafe {
            &*queue
                .cast::<u8>()
                .sub(SDMA_QUEUE_HANDLE_BIT)
                .cast::<SdmaQueueControl>()
        };
        match index {
            QueueIndex::Read => control.read_index_host_address,
            QueueIndex::Write => control.write_index_host_address,
        }
    } else {
        // SAFETY: The public AQL or soft handle retains its complete control.
        match index {
            QueueIndex::Read => unsafe { AqlControlRegion::read_index_address_at(queue as usize) },
            QueueIndex::Write => unsafe {
                AqlControlRegion::write_index_address_at(queue as usize)
            },
        }
    };
    // SAFETY: The caller retains a live queue handle. AQL and soft controls
    // expose named indices; counted and SDMA handles resolve to
    // the native index mapping retained by their queue owner.
    let index = unsafe { &*(address as *const AtomicU64) };
    Some(operation(index))
}
