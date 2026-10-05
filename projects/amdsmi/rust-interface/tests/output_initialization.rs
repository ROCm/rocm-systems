// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use amdsmi::{
    AmdsmiClkInfoT, AmdsmiClkTypeT, AmdsmiEngineUsageT, AmdsmiErrorCountT, AmdsmiGpuBlockT,
    AmdsmiPowerInfoT, AmdsmiProcessorHandle, AmdsmiStatusT,
};
use std::ptr::{addr_of_mut, null_mut, NonNull};

// These symbols stand in for the C library. Like the real queries, they leave
// reserved fields untouched. Miri detects the regression even if an ordinary
// test run happens to find zeroes in uninitialized stack memory.
#[no_mangle]
unsafe extern "C" fn amdsmi_get_gpu_activity(
    handle: AmdsmiProcessorHandle,
    out: *mut AmdsmiEngineUsageT,
) -> AmdsmiStatusT {
    if handle.is_null() {
        return AmdsmiStatusT::AmdsmiStatusInval;
    }
    addr_of_mut!((*out).gfx_activity).write(17);
    addr_of_mut!((*out).umc_activity).write(23);
    addr_of_mut!((*out).mm_activity).write(5);
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[no_mangle]
unsafe extern "C" fn amdsmi_get_power_info(
    _: AmdsmiProcessorHandle,
    out: *mut AmdsmiPowerInfoT,
) -> AmdsmiStatusT {
    addr_of_mut!((*out).socket_power).write(123);
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[no_mangle]
unsafe extern "C" fn amdsmi_get_clock_info(
    _: AmdsmiProcessorHandle,
    _: AmdsmiClkTypeT,
    out: *mut AmdsmiClkInfoT,
) -> AmdsmiStatusT {
    addr_of_mut!((*out).clk).write(1700);
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[no_mangle]
unsafe extern "C" fn amdsmi_get_gpu_ecc_count(
    _: AmdsmiProcessorHandle,
    _: AmdsmiGpuBlockT,
    out: *mut AmdsmiErrorCountT,
) -> AmdsmiStatusT {
    addr_of_mut!((*out).correctable_count).write(2);
    addr_of_mut!((*out).uncorrectable_count).write(3);
    addr_of_mut!((*out).deferred_count).write(4);
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[no_mangle]
unsafe extern "C" fn amdsmi_get_gpu_total_ecc_count(
    _: AmdsmiProcessorHandle,
    out: *mut AmdsmiErrorCountT,
) -> AmdsmiStatusT {
    // The total-count API adds to the caller's existing counters.
    (*out).correctable_count += 2;
    (*out).uncorrectable_count += 3;
    (*out).deferred_count += 4;
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[test]
fn activity_initializes_reserved_fields() {
    let activity = amdsmi::amdsmi_get_gpu_activity(NonNull::dangling().as_ptr()).unwrap();
    assert_eq!(activity.gfx_activity, 17);
    assert_eq!(activity.umc_activity, 23);
    assert_eq!(activity.mm_activity, 5);
    assert_eq!(activity.reserved, [0; 13]);
}

#[test]
fn power_initializes_unwritten_fields() {
    let power = amdsmi::amdsmi_get_power_info(null_mut()).unwrap();
    assert_eq!(power.socket_power, 123);
    assert_eq!(power.reserved, [0; 18]);
}

#[test]
fn clock_initializes_unwritten_fields() {
    let clock =
        amdsmi::amdsmi_get_clock_info(null_mut(), AmdsmiClkTypeT::AmdsmiClkTypeGfx).unwrap();
    assert_eq!(clock.clk, 1700);
    assert_eq!(clock.reserved, [0; 4]);
}

#[test]
fn ecc_initializes_reserved_fields() {
    let errors =
        amdsmi::amdsmi_get_gpu_ecc_count(null_mut(), AmdsmiGpuBlockT::AmdsmiGpuBlockUmc).unwrap();
    assert_eq!(errors.correctable_count, 2);
    assert_eq!(errors.uncorrectable_count, 3);
    assert_eq!(errors.deferred_count, 4);
    assert_eq!(errors.reserved, [0; 5]);
}

#[test]
fn total_ecc_initializes_reserved_fields() {
    let errors = amdsmi::amdsmi_get_gpu_total_ecc_count(null_mut()).unwrap();
    assert_eq!(errors.deferred_count, 4);
    assert_eq!(errors.reserved, [0; 5]);
}

#[test]
fn query_errors_are_preserved() {
    assert_eq!(
        amdsmi::amdsmi_get_gpu_activity(null_mut()).unwrap_err(),
        AmdsmiStatusT::AmdsmiStatusInval
    );
}
