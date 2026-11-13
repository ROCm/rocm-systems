// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use amdsmi::{
    AmdsmiClkInfoT, AmdsmiClkTypeT, AmdsmiEngineUsageT, AmdsmiErrorCountT, AmdsmiGpuBlockT,
    AmdsmiPowerInfoT, AmdsmiProcessorHandle, AmdsmiStatusT,
};
use std::hint::black_box;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, Ordering};

static ECC_COUNT_ZEROED: AtomicBool = AtomicBool::new(false);
static TOTAL_ECC_COUNT_ZEROED: AtomicBool = AtomicBool::new(false);
static ACTIVITY_ZEROED: AtomicBool = AtomicBool::new(false);
static POWER_INFO_ZEROED: AtomicBool = AtomicBool::new(false);
static CLOCK_INFO_ZEROED: AtomicBool = AtomicBool::new(false);

unsafe fn is_zeroed<T>(out: *const T) -> bool {
    std::slice::from_raw_parts(out.cast::<u8>(), std::mem::size_of::<T>())
        .iter()
        .all(|&byte| byte == 0)
}

#[no_mangle]
unsafe extern "C" fn amdsmi_get_gpu_ecc_count(
    _handle: AmdsmiProcessorHandle,
    _block: AmdsmiGpuBlockT,
    out: *mut AmdsmiErrorCountT,
) -> AmdsmiStatusT {
    ECC_COUNT_ZEROED.store(is_zeroed(out), Ordering::SeqCst);
    AmdsmiStatusT::AmdsmiStatusSuccess
}

// The real function adds the per-block counts into *out, so a dirty buffer corrupts the totals.
#[no_mangle]
unsafe extern "C" fn amdsmi_get_gpu_total_ecc_count(
    _handle: AmdsmiProcessorHandle,
    out: *mut AmdsmiErrorCountT,
) -> AmdsmiStatusT {
    TOTAL_ECC_COUNT_ZEROED.store(is_zeroed(out), Ordering::SeqCst);
    (*out).correctable_count += 3;
    (*out).uncorrectable_count += 2;
    (*out).deferred_count += 1;
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[no_mangle]
unsafe extern "C" fn amdsmi_get_gpu_activity(
    _handle: AmdsmiProcessorHandle,
    out: *mut AmdsmiEngineUsageT,
) -> AmdsmiStatusT {
    ACTIVITY_ZEROED.store(is_zeroed(out), Ordering::SeqCst);
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[no_mangle]
unsafe extern "C" fn amdsmi_get_power_info(
    _handle: AmdsmiProcessorHandle,
    out: *mut AmdsmiPowerInfoT,
) -> AmdsmiStatusT {
    POWER_INFO_ZEROED.store(is_zeroed(out), Ordering::SeqCst);
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[no_mangle]
unsafe extern "C" fn amdsmi_get_clock_info(
    _handle: AmdsmiProcessorHandle,
    _clk_type: AmdsmiClkTypeT,
    out: *mut AmdsmiClkInfoT,
) -> AmdsmiStatusT {
    CLOCK_INFO_ZEROED.store(is_zeroed(out), Ordering::SeqCst);
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[inline(never)]
fn dirty_stack() {
    let mut scratch = [0xA5u8; 16384];
    black_box(&mut scratch);
}

fn handle() -> AmdsmiProcessorHandle {
    NonNull::dangling().as_ptr()
}

#[test]
fn total_ecc_count_starts_from_zero() {
    dirty_stack();
    let ec = amdsmi::amdsmi_get_gpu_total_ecc_count(handle()).unwrap();
    assert!(TOTAL_ECC_COUNT_ZEROED.load(Ordering::SeqCst));
    assert_eq!(ec.correctable_count, 3);
    assert_eq!(ec.uncorrectable_count, 2);
    assert_eq!(ec.deferred_count, 1);
}

#[test]
fn ecc_count_output_is_zeroed() {
    dirty_stack();
    amdsmi::amdsmi_get_gpu_ecc_count(handle(), AmdsmiGpuBlockT::AmdsmiGpuBlockUmc).unwrap();
    assert!(ECC_COUNT_ZEROED.load(Ordering::SeqCst));
}

#[test]
fn activity_output_is_zeroed() {
    dirty_stack();
    amdsmi::amdsmi_get_gpu_activity(handle()).unwrap();
    assert!(ACTIVITY_ZEROED.load(Ordering::SeqCst));
}

#[test]
fn power_info_output_is_zeroed() {
    dirty_stack();
    amdsmi::amdsmi_get_power_info(handle()).unwrap();
    assert!(POWER_INFO_ZEROED.load(Ordering::SeqCst));
}

#[test]
fn clock_info_output_is_zeroed() {
    dirty_stack();
    amdsmi::amdsmi_get_clock_info(handle(), AmdsmiClkTypeT::AmdsmiClkTypeGfx).unwrap();
    assert!(CLOCK_INFO_ZEROED.load(Ordering::SeqCst));
}
