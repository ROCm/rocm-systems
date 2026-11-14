// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Rust ABCE ring mechanisms.

// These entry points project caller-owned C objects through raw pointers.
// Initialization/reconfiguration requires exclusive caller access; submission
// accesses individual fields and shared atomic cursors, never whole-object
// exclusive references. Pointer and buffer preconditions are specified by the
// corresponding public C header. The packet/ring tests run both providers.

// The C-facing projection keeps C integer conversion and alignment contracts.
// Safety requirements and callback preconditions are specified in the public
// header named by this module; bodies intentionally follow those algorithms.
#![allow(
    clippy::missing_safety_doc,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_ptr_alignment,
    clippy::too_many_lines
)]

use super::abi::{
    ABCE_STATUS_INVALID_ARGUMENT, ABCE_STATUS_NOT_READY, ABCE_STATUS_OK, ABCE_STATUS_OUT_OF_RANGE,
    abce_doorbell_notify_fn_t, abce_ring_config_t, abce_ring_producer_t, abce_ring_publisher_t,
    abce_ring_queue_ops_t, abce_ring_reservation_t, abce_ring_t, abce_status_t,
};
use super::{
    abce_align_up, abce_host_cas_relaxed_u64, abce_host_fence_release, abce_host_fence_seq_cst,
    abce_host_load_acquire_u64, abce_host_load_relaxed_u64, abce_host_store_relaxed_u64,
    abce_host_store_release_u64, abce_is_power_of_two, abce_ring_host_store_fence,
    abce_status_is_ok,
};

pub(super) unsafe extern "C" fn abce_ring_atomic_load_hw(ptr: *const u64) -> u64 {
    unsafe {
        let stable_ptr: *const u64 =
            ((core::ptr::read_volatile(core::ptr::addr_of!(ptr))) as usize) as *const u64;
        abce_host_load_relaxed_u64(stable_ptr)
    }
}

pub(super) unsafe extern "C" fn abce_ring_cursor_load(ptr: *const u64) -> u64 {
    unsafe { abce_host_load_relaxed_u64(ptr) }
}

pub(super) unsafe extern "C" fn abce_ring_cursor_load_acquire(ptr: *const u64) -> u64 {
    unsafe { abce_host_load_acquire_u64(ptr) }
}

pub(super) unsafe extern "C" fn abce_ring_cursor_cas(
    ptr: *mut u64,
    expected: *mut u64,
    desired: u64,
) -> bool {
    unsafe { abce_host_cas_relaxed_u64(ptr, expected, desired) }
}

pub(super) unsafe extern "C" fn abce_ring_cursor_store(ptr: *mut u64, value: u64) {
    unsafe {
        abce_host_store_relaxed_u64(ptr, value);
    }
}

pub(super) unsafe extern "C" fn abce_ring_cursor_store_release(ptr: *mut u64, value: u64) {
    unsafe {
        abce_host_store_release_u64(ptr, value);
    }
}

pub(super) unsafe extern "C" fn abce_ring_commit_load(commit_cursor: *const u64) -> u64 {
    unsafe { abce_ring_cursor_load_acquire(commit_cursor) }
}

pub(super) unsafe extern "C" fn abce_ring_commit_store(commit_cursor: *mut u64, value: u64) {
    unsafe {
        abce_ring_cursor_store_release(commit_cursor, value);
    }
}

pub(super) unsafe extern "C" fn abce_ring_cached_read_load(cached_read_index: *const u64) -> u64 {
    unsafe { abce_ring_cursor_load_acquire(cached_read_index) }
}

pub(super) unsafe extern "C" fn abce_ring_cached_read_store(
    cached_read_index: *mut u64,
    read_index: u64,
) {
    unsafe {
        abce_ring_cursor_store_release(cached_read_index, read_index);
    }
}

pub(super) unsafe extern "C" fn abce_ring_pause() {
    std::thread::yield_now();
}

#[must_use]
pub unsafe extern "C" fn abce_ring_reservation_is_valid(
    reservation: *const abce_ring_reservation_t,
) -> bool {
    unsafe { ((*(reservation)).end) > ((*(reservation)).start) }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_reservation_bytes(
    reservation: *const abce_ring_reservation_t,
) -> u64 {
    unsafe { ((*(reservation)).end).wrapping_sub((*(reservation)).start) }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_wrap(index: u64, size: u64) -> u64 {
    index & ((size).wrapping_sub(1_u64))
}

#[must_use]
pub unsafe extern "C" fn abce_ring_can_write_upto(read_index: u64, size: u64, upto: u64) -> bool {
    ((upto).wrapping_sub(read_index)) < (size)
}

#[must_use]
pub unsafe extern "C" fn abce_ring_pad_bytes(start: u64, bytes: u64, size: u64) -> u64 {
    unsafe {
        let offset: u64 = abce_ring_wrap(start, size);
        if ((offset).wrapping_add(bytes)) > (size) {
            (size).wrapping_sub(offset)
        } else {
            0_u64
        }
    }
}

pub(super) unsafe extern "C" fn abce_ring_reserve_(
    producer: *const abce_ring_producer_t,
    bytes: u64,
    block: bool,
    out_reservation: *mut abce_ring_reservation_t,
) -> abce_status_t {
    unsafe {
        if (((producer).is_null() || ((*(producer)).reserve_cursor).is_null())
            || ((*(producer)).load_read_index).is_none())
            || (out_reservation).is_null()
        {
            return ABCE_STATUS_INVALID_ARGUMENT;
        }
        let size: u64 = (*(producer)).size;
        if !(abce_is_power_of_two(size)) || ((bytes) == (0_u64)) {
            return ABCE_STATUS_INVALID_ARGUMENT;
        }
        let fold_padding: bool = ((*(producer)).publish_padding).is_none();
        if ((bytes) >= (size)) || (fold_padding && ((bytes) > (size / 2_u64))) {
            return ABCE_STATUS_OUT_OF_RANGE;
        }
        {
            loop {
                {
                    let mut current: u64 =
                        abce_ring_cursor_load(((*(producer)).reserve_cursor).cast_const());
                    let pad_bytes: u64 = abce_ring_pad_bytes(current, bytes, size);
                    let reservation_bytes: u64 = if fold_padding {
                        (pad_bytes).wrapping_add(bytes)
                    } else if (pad_bytes) == (0_u64) {
                        bytes
                    } else {
                        pad_bytes
                    };
                    let next: u64 = (current).wrapping_add(reservation_bytes);
                    let mut has_room: bool = !((*(producer)).cached_read_index).is_null()
                        && abce_ring_can_write_upto(
                            abce_ring_cached_read_load(
                                ((*(producer)).cached_read_index).cast_const(),
                            ),
                            size,
                            next,
                        );
                    if !(has_room) {
                        let read_index: u64 = (match (*(producer)).load_read_index {
                            Some(callback) => callback,
                            None => std::process::abort(),
                        })((*(producer)).user_data);
                        if !((*(producer)).cached_read_index).is_null() {
                            abce_ring_cached_read_store(
                                (*(producer)).cached_read_index,
                                read_index,
                            );
                        }
                        has_room = abce_ring_can_write_upto(read_index, size, next);
                        if !(has_room) && !(block) {
                            return ABCE_STATUS_NOT_READY;
                        }
                    }
                    if has_room
                        && abce_ring_cursor_cas(
                            (*(producer)).reserve_cursor,
                            core::ptr::addr_of_mut!(current),
                            next,
                        )
                    {
                        if ((pad_bytes) != (0_u64)) && !(fold_padding) {
                            let mut padding: abce_ring_reservation_t = { core::mem::zeroed() };
                            (padding).start = current;
                            (padding).end = next;
                            let status: abce_status_t = (match (*(producer)).publish_padding {
                                Some(callback) => callback,
                                None => std::process::abort(),
                            })(
                                (*(producer)).user_data,
                                core::ptr::addr_of_mut!(padding).cast_const(),
                            );
                            if !(abce_status_is_ok(status)) {
                                return status;
                            }
                            continue;
                        }
                        (*(out_reservation)).start = current;
                        (*(out_reservation)).end = next;
                        return ABCE_STATUS_OK;
                    }
                    abce_ring_pause();
                }
            }
        }
    }
}

pub unsafe extern "C" fn abce_ring_reserve(
    producer: *const abce_ring_producer_t,
    bytes: u64,
    out_reservation: *mut abce_ring_reservation_t,
) -> abce_status_t {
    unsafe { abce_ring_reserve_(producer, bytes, true, out_reservation) }
}

pub unsafe extern "C" fn abce_ring_try_reserve(
    producer: *const abce_ring_producer_t,
    bytes: u64,
    out_reservation: *mut abce_ring_reservation_t,
) -> abce_status_t {
    unsafe { abce_ring_reserve_(producer, bytes, false, out_reservation) }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_publish(
    publisher: *const abce_ring_publisher_t,
    reservation: *const abce_ring_reservation_t,
) -> abce_status_t {
    unsafe {
        if ((((publisher).is_null() || ((*(publisher)).commit_cursor).is_null())
            || ((*(publisher)).publish).is_none())
            || (reservation).is_null())
            || !(abce_ring_reservation_is_valid(reservation))
        {
            return ABCE_STATUS_INVALID_ARGUMENT;
        }
        let mut observed_commit_index: u64 =
            abce_ring_commit_load(((*(publisher)).commit_cursor).cast_const());
        while (observed_commit_index) != ((*(reservation)).start) {
            if ((*(publisher)).wait).is_some() {
                (match (*(publisher)).wait {
                    Some(callback) => callback,
                    None => std::process::abort(),
                })(
                    (*(publisher)).user_data,
                    ((*(publisher)).commit_cursor).cast_const(),
                    observed_commit_index,
                );
            } else {
                abce_ring_pause();
            }
            observed_commit_index =
                abce_ring_commit_load(((*(publisher)).commit_cursor).cast_const());
        }
        (match (*(publisher)).publish {
            Some(callback) => callback,
            None => std::process::abort(),
        })((*(publisher)).user_data, (*(reservation)).end);
        abce_ring_commit_store((*(publisher)).commit_cursor, (*(reservation)).end);
        ABCE_STATUS_OK
    }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_queue_ops_empty(ops: *const abce_ring_queue_ops_t) -> bool {
    unsafe {
        ((((*(ops)).load_read_index).is_none() && ((*(ops)).load_write_index).is_none())
            && ((*(ops)).publish_write_index).is_none())
            && ((*(ops)).user_data).is_null()
    }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_queue_ops_complete(ops: *const abce_ring_queue_ops_t) -> bool {
    unsafe {
        (((*(ops)).load_read_index).is_some() && ((*(ops)).load_write_index).is_some())
            && ((*(ops)).publish_write_index).is_some()
    }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_wrap_index(ring: *const abce_ring_t, index: u64) -> u32 {
    unsafe { (abce_ring_wrap(index, (((*(ring)).config).size) as u64)) as u32 }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_load_read_index(ring: *const abce_ring_t) -> u64 {
    unsafe {
        if abce_ring_queue_ops_complete(core::ptr::addr_of!(((*(ring)).config).queue_ops)) {
            return (match (((*(ring)).config).queue_ops).load_read_index {
                Some(callback) => callback,
                None => std::process::abort(),
            })((((*(ring)).config).queue_ops).user_data);
        }
        abce_ring_atomic_load_hw(
            (core::ptr::read_volatile(core::ptr::addr_of!(((*(ring)).config).read_ptr)))
                .cast_const(),
        )
    }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_load_write_index(ring: *const abce_ring_t) -> u64 {
    unsafe {
        if abce_ring_queue_ops_complete(core::ptr::addr_of!(((*(ring)).config).queue_ops)) {
            return (match (((*(ring)).config).queue_ops).load_write_index {
                Some(callback) => callback,
                None => std::process::abort(),
            })((((*(ring)).config).queue_ops).user_data);
        }
        abce_ring_atomic_load_hw(
            (core::ptr::read_volatile(core::ptr::addr_of!(((*(ring)).config).write_ptr)))
                .cast_const(),
        )
    }
}

pub unsafe extern "C" fn abce_ring_publish_write_index(
    ring: *const abce_ring_t,
    new_write_index: u64,
) {
    unsafe {
        if abce_ring_queue_ops_complete(core::ptr::addr_of!(((*(ring)).config).queue_ops)) {
            (match (((*(ring)).config).queue_ops).publish_write_index {
                Some(callback) => callback,
                None => std::process::abort(),
            })((((*(ring)).config).queue_ops).user_data, new_write_index);
        } else if (((*(ring)).config).publish).is_some() {
            (match ((*(ring)).config).publish {
                Some(callback) => callback,
                None => std::process::abort(),
            })(
                ((*(ring)).config).publish_user_data,
                core::ptr::read_volatile(core::ptr::addr_of!(((*(ring)).config).write_ptr)),
                core::ptr::read_volatile(core::ptr::addr_of!(((*(ring)).config).doorbell)),
                new_write_index,
            );
        } else {
            core::ptr::write_volatile(
                core::ptr::addr_of_mut!(
                    (*(core::ptr::read_volatile(core::ptr::addr_of!(
                        ((*(ring)).config).write_ptr
                    ))))
                ),
                new_write_index,
            );
            abce_host_fence_release();
            core::ptr::write_volatile(
                core::ptr::addr_of_mut!(
                    (*(core::ptr::read_volatile(core::ptr::addr_of!(((*(ring)).config).doorbell))))
                ),
                new_write_index,
            );
        }
    }
}

pub unsafe extern "C" fn abce_ring_initialize(
    config: *const abce_ring_config_t,
    notify: abce_doorbell_notify_fn_t,
    notify_user_data: *mut core::ffi::c_void,
    out_ring: *mut abce_ring_t,
) {
    unsafe {
        {};
        let has_raw_backend: bool =
            (!(core::ptr::read_volatile(core::ptr::addr_of!((*(config)).write_ptr))).is_null()
                || !(core::ptr::read_volatile(core::ptr::addr_of!((*(config)).read_ptr)))
                    .is_null())
                || !(core::ptr::read_volatile(core::ptr::addr_of!((*(config)).doorbell))).is_null();
        let raw_backend_complete: bool =
            (!(core::ptr::read_volatile(core::ptr::addr_of!((*(config)).write_ptr))).is_null()
                && !(core::ptr::read_volatile(core::ptr::addr_of!((*(config)).read_ptr)))
                    .is_null())
                && !(core::ptr::read_volatile(core::ptr::addr_of!((*(config)).doorbell))).is_null();
        let has_ops_backend: bool =
            !(abce_ring_queue_ops_empty(core::ptr::addr_of!((*(config)).queue_ops)));
        let _ = has_raw_backend;
        let _ = raw_backend_complete;
        let _ = has_ops_backend;
        {};
        {};
        {};
        let _ = libc::memset(
            (out_ring).cast::<core::ffi::c_void>(),
            0i32,
            core::mem::size_of::<abce_ring_t>() as usize,
        );
        (*(out_ring)).config = *(config);
        (*(out_ring)).notify = notify;
        (*(out_ring)).notify_user_data = notify_user_data;
        let initial_write_index: u64 = abce_ring_load_write_index((out_ring).cast_const());
        abce_ring_cursor_store(
            core::ptr::addr_of_mut!(((*(out_ring)).control).reserve_cursor),
            initial_write_index,
        );
        abce_ring_cursor_store(
            core::ptr::addr_of_mut!(((*(out_ring)).control).commit_cursor),
            initial_write_index,
        );
        abce_ring_cursor_store(
            core::ptr::addr_of_mut!(((*(out_ring)).control).max_write_index),
            initial_write_index,
        );
    }
}

pub unsafe extern "C" fn abce_ring_padded_size(
    ring: *const abce_ring_t,
    payload_bytes: u32,
    out_reserve_bytes: *mut u32,
) -> abce_status_t {
    unsafe {
        if (payload_bytes) == (0_u32) {
            return ABCE_STATUS_INVALID_ARGUMENT;
        }
        let mut padded: u64 = u64::from(payload_bytes);
        if (padded) < ((((*(ring)).config).min_submission_size) as u64) {
            padded = (((*(ring)).config).min_submission_size) as u64;
        }
        if ((*(ring)).config).align64 {
            padded = abce_align_up(padded, 64_u64);
        }
        if ((padded) > u64::from(4_294_967_295_u32))
            || ((padded) >= ((((*(ring)).config).size) as u64))
        {
            return ABCE_STATUS_OUT_OF_RANGE;
        }
        (*(out_reserve_bytes)) = (padded) as u32;
        ABCE_STATUS_OK
    }
}

pub(super) unsafe extern "C" fn abce_ring_load_read_index_callback(
    user_data: *mut core::ffi::c_void,
) -> u64 {
    unsafe { abce_ring_load_read_index((user_data) as *const abce_ring_t) }
}

pub unsafe extern "C" fn abce_ring_fill(
    ring: *const abce_ring_t,
    buffer: *mut core::ffi::c_char,
    src: *const core::ffi::c_void,
    src_bytes: usize,
    reserve_bytes: usize,
) {
    unsafe {
        if (((*(ring)).config).write).is_some() {
            (match ((*(ring)).config).write {
                Some(callback) => callback,
                None => std::process::abort(),
            })(
                ((*(ring)).config).write_user_data,
                buffer,
                src,
                src_bytes,
                reserve_bytes,
            );
        } else if (src_bytes) != (0_usize) {
            let _ = libc::memcpy((buffer).cast::<core::ffi::c_void>(), src, src_bytes);
        }
    }
}

pub(super) unsafe extern "C" fn abce_ring_read_back(byte: *const core::ffi::c_void) {
    unsafe {
        abce_host_fence_seq_cst();
        let _ = core::ptr::read_volatile(core::ptr::addr_of!((*(byte).cast::<u8>())));
    }
}

pub(super) unsafe extern "C" fn abce_ring_region_last_byte(
    buffer: *const core::ffi::c_char,
    bytes: usize,
) -> *const core::ffi::c_char {
    ((buffer).wrapping_add(bytes)).wrapping_offset(-(1i32) as isize)
}

pub(super) unsafe extern "C" fn abce_ring_order_region(
    ring: *const abce_ring_t,
    region: *const abce_ring_reservation_t,
    filled_by_writer: bool,
) {
    unsafe {
        if ((*(ring)).config).readback_before_doorbell {
            let buffer: *const core::ffi::c_char = ((((*(ring)).config).base)
                .wrapping_offset((abce_ring_wrap_index(ring, (*(region)).start)) as isize))
            .cast_const();
            abce_ring_read_back(
                (abce_ring_region_last_byte(
                    buffer,
                    (abce_ring_reservation_bytes(region)) as usize,
                ))
                .cast::<core::ffi::c_void>(),
            );
        } else if !(filled_by_writer) {
            abce_ring_host_store_fence();
        }
    }
}

pub(super) unsafe extern "C" fn abce_ring_publish_padding_callback(
    user_data: *mut core::ffi::c_void,
    padding: *const abce_ring_reservation_t,
) -> abce_status_t {
    unsafe {
        let ring: *mut abce_ring_t = (user_data).cast::<abce_ring_t>();
        let buffer: *mut core::ffi::c_char = (((*(ring)).config).base).wrapping_offset(
            (abce_ring_wrap_index((ring).cast_const(), (*(padding)).start)) as isize,
        );
        let bytes: usize = (abce_ring_reservation_bytes(padding)) as usize;
        if (((*(ring)).config).write).is_some() {
            (match ((*(ring)).config).write {
                Some(callback) => callback,
                None => std::process::abort(),
            })(
                ((*(ring)).config).write_user_data,
                buffer,
                core::ptr::null(),
                0_usize,
                bytes,
            );
        } else {
            let _ = libc::memset((buffer).cast::<core::ffi::c_void>(), 0i32, bytes);
        }
        abce_ring_order_region(
            (ring).cast_const(),
            padding,
            (((*(ring)).config).write).is_some(),
        );
        abce_ring_release_ordered(ring, padding)
    }
}

pub(super) unsafe extern "C" fn abce_ring_publish_callback(
    user_data: *mut core::ffi::c_void,
    new_write_index: u64,
) {
    unsafe {
        let ring: *mut abce_ring_t = (user_data).cast::<abce_ring_t>();
        abce_ring_publish_write_index((ring).cast_const(), new_write_index);
        abce_ring_cursor_store_release(
            core::ptr::addr_of_mut!(((*(ring)).control).max_write_index),
            new_write_index,
        );
        if ((*(ring)).notify).is_some() {
            (match (*(ring)).notify {
                Some(callback) => callback,
                None => std::process::abort(),
            })((*(ring)).notify_user_data, new_write_index);
        }
    }
}

pub(super) unsafe extern "C" fn abce_ring_commit_wait_callback(
    user_data: *mut core::ffi::c_void,
    commit_cursor: *const u64,
    observed_commit_index: u64,
) {
    unsafe {
        let ring: *mut abce_ring_t = (user_data).cast::<abce_ring_t>();
        if (((*(ring)).config).commit_wait).is_some() {
            (match ((*(ring)).config).commit_wait {
                Some(callback) => callback,
                None => std::process::abort(),
            })(
                ((*(ring)).config).commit_wait_user_data,
                commit_cursor,
                observed_commit_index,
            );
        } else {
            abce_ring_pause();
        }
    }
}

pub unsafe extern "C" fn abce_ring_acquire(
    ring: *mut abce_ring_t,
    reserve_bytes: u32,
    out_reservation: *mut abce_ring_reservation_t,
    out_buffer: *mut *mut core::ffi::c_char,
) -> abce_status_t {
    unsafe {
        (*(out_buffer)) = core::ptr::null_mut();
        let mut producer: abce_ring_producer_t = { core::mem::zeroed() };
        (producer).reserve_cursor = core::ptr::addr_of_mut!(((*(ring)).control).reserve_cursor);
        (producer).cached_read_index =
            core::ptr::addr_of_mut!(((*(ring)).control).cached_read_index);
        (producer).size = (((*(ring)).config).size) as u64;
        (producer).load_read_index = Some(abce_ring_load_read_index_callback);
        (producer).publish_padding = Some(abce_ring_publish_padding_callback);
        (producer).user_data = (ring).cast::<core::ffi::c_void>();
        let status: abce_status_t = abce_ring_reserve(
            core::ptr::addr_of_mut!(producer).cast_const(),
            u64::from(reserve_bytes),
            out_reservation,
        );
        if !(abce_status_is_ok(status)) {
            return status;
        }
        let buffer: *mut core::ffi::c_char = (((*(ring)).config).base).wrapping_offset(
            (abce_ring_wrap_index((ring).cast_const(), (*(out_reservation)).start)) as isize,
        );
        if (((*(ring)).config).write).is_none() {
            let _ = libc::memset(
                (buffer).cast::<core::ffi::c_void>(),
                0i32,
                (reserve_bytes) as usize,
            );
        }
        (*(out_buffer)) = buffer;
        ABCE_STATUS_OK
    }
}

pub unsafe extern "C" fn abce_ring_release_ordered(
    ring: *mut abce_ring_t,
    reservation: *const abce_ring_reservation_t,
) -> abce_status_t {
    unsafe {
        let mut publisher: abce_ring_publisher_t = { core::mem::zeroed() };
        (publisher).commit_cursor = core::ptr::addr_of_mut!(((*(ring)).control).commit_cursor);
        (publisher).publish = Some(abce_ring_publish_callback);
        (publisher).wait = Some(abce_ring_commit_wait_callback);
        (publisher).user_data = (ring).cast::<core::ffi::c_void>();
        abce_ring_publish(core::ptr::addr_of_mut!(publisher).cast_const(), reservation)
    }
}

pub unsafe extern "C" fn abce_ring_release(
    ring: *mut abce_ring_t,
    reservation: *const abce_ring_reservation_t,
) -> abce_status_t {
    unsafe {
        abce_ring_order_region((ring).cast_const(), reservation, false);
        abce_ring_release_ordered(ring, reservation)
    }
}

pub unsafe extern "C" fn abce_ring_cancel(
    ring: *mut abce_ring_t,
    reservation: *const abce_ring_reservation_t,
) -> abce_status_t {
    unsafe {
        if (((*(ring)).config).write).is_some() {
            let buffer: *mut core::ffi::c_char = (((*(ring)).config).base).wrapping_offset(
                (abce_ring_wrap_index((ring).cast_const(), (*(reservation)).start)) as isize,
            );
            abce_ring_fill(
                (ring).cast_const(),
                buffer,
                core::ptr::null(),
                0_usize,
                (abce_ring_reservation_bytes(reservation)) as usize,
            );
        }
        abce_ring_order_region(
            (ring).cast_const(),
            reservation,
            (((*(ring)).config).write).is_some(),
        );
        abce_ring_release_ordered(ring, reservation)
    }
}

#[must_use]
pub unsafe extern "C" fn abce_ring_drain(ring: *const abce_ring_t) -> abce_status_t {
    unsafe {
        let target: u64 =
            abce_ring_cursor_load_acquire(core::ptr::addr_of!(((*(ring)).control).max_write_index));
        while (abce_ring_load_read_index(ring)) < (target) {
            abce_ring_pause();
        }
        ABCE_STATUS_OK
    }
}
