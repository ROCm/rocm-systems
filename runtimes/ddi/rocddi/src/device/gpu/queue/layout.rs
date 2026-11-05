// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Typed AMD AQL control layout shared by native backends and frontends.

use super::AqlControlRegion;

/// HSA queue header prefix used to pin offsets in the native control page.
#[allow(dead_code, clippy::missing_docs_in_private_items)]
#[repr(C)]
pub(crate) struct AqlQueueHeader {
    pub(crate) queue_type: u32,
    pub(crate) features: u32,
    pub(crate) base_address: u64,
    pub(crate) doorbell_signal: u64,
    pub(crate) size: u32,
    pub(crate) reserved: u32,
    pub(crate) id: u64,
}

/// Native 256-byte AQL control layout shared with the command processor.
#[allow(dead_code, clippy::missing_docs_in_private_items)]
#[repr(C, align(64))]
pub(crate) struct AqlControlLayout {
    pub(crate) queue_header: AqlQueueHeader,
    pub(crate) caps: u32,
    pub(crate) reserved1: [u32; 3],
    pub(crate) write_dispatch_id: u64,
    pub(crate) group_segment_aperture_base_hi: u32,
    pub(crate) private_segment_aperture_base_hi: u32,
    pub(crate) max_cu_id: u32,
    pub(crate) max_wave_id: u32,
    pub(crate) max_legacy_doorbell_dispatch_id_plus_1: u64,
    pub(crate) legacy_doorbell_lock: u32,
    pub(crate) reserved2: [u32; 9],
    pub(crate) read_dispatch_id: u64,
    pub(crate) read_dispatch_id_field_base_byte_offset: u32,
    pub(crate) compute_tmpring_size: u32,
    pub(crate) scratch_resource_descriptor: [u32; 4],
    pub(crate) scratch_backing_memory_location: u64,
    pub(crate) scratch_backing_memory_byte_size: u64,
    pub(crate) scratch_wave64_lane_byte_size: u32,
    pub(crate) queue_properties: u32,
    pub(crate) scratch_max_use_index: u64,
    pub(crate) queue_inactive_signal: u64,
    pub(crate) alt_scratch_max_use_index: u64,
    pub(crate) alt_scratch_resource_descriptor: [u32; 4],
    pub(crate) alt_scratch_backing_memory_location: u64,
    pub(crate) alt_scratch_dispatch_limit: [u32; 3],
    pub(crate) alt_scratch_wave64_lane_byte_size: u32,
    pub(crate) alt_compute_tmpring_size: u32,
    pub(crate) reserved5: u32,
}

#[allow(unsafe_code)]
impl AqlControlRegion {
    /// Finds the read-index field in a live AQL control allocation.
    ///
    /// # Safety
    /// `base` must be a live, aligned AQL control base retained by the caller.
    #[inline]
    #[must_use]
    pub unsafe fn read_index_address_at(base: usize) -> usize {
        let layout = base as *const AqlControlLayout;
        // SAFETY: The caller guarantees the complete typed control allocation.
        unsafe { std::ptr::addr_of!((*layout).read_dispatch_id) as usize }
    }

    /// Finds the write-index field in a live AQL control allocation.
    ///
    /// # Safety
    /// `base` must be a live, aligned AQL control base retained by the caller.
    #[inline]
    #[must_use]
    pub unsafe fn write_index_address_at(base: usize) -> usize {
        let layout = base as *const AqlControlLayout;
        // SAFETY: The caller guarantees the complete typed control allocation.
        unsafe { std::ptr::addr_of!((*layout).write_dispatch_id) as usize }
    }

    /// Finds the properties field in a live AQL control allocation.
    ///
    /// # Safety
    /// `base` must be a live, aligned AQL control base retained by the caller.
    #[inline]
    #[must_use]
    pub unsafe fn properties_address_at(base: usize) -> usize {
        let layout = base as *const AqlControlLayout;
        // SAFETY: The caller guarantees the complete typed control allocation.
        unsafe { std::ptr::addr_of!((*layout).queue_properties) as usize }
    }
}
