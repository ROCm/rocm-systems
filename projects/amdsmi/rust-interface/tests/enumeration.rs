// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use amdsmi::{AmdsmiEnumerationInfoT, AmdsmiProcessorHandle, AmdsmiStatusT};
use std::ffi::CStr;
use std::ptr::{addr_of_mut, null_mut, NonNull};

#[no_mangle]
unsafe extern "C" fn amdsmi_get_gpu_enumeration_info(
    handle: AmdsmiProcessorHandle,
    out: *mut AmdsmiEnumerationInfoT,
) -> AmdsmiStatusT {
    if handle.is_null() {
        return AmdsmiStatusT::AmdsmiStatusNotSupported;
    }
    addr_of_mut!((*out).drm_render).write(141);
    addr_of_mut!((*out).drm_card).write(7);
    addr_of_mut!((*out).hsa_id).write(9);
    addr_of_mut!((*out).hip_id).write(2);
    addr_of_mut!((*out).oam_id).write(3);
    addr_of_mut!((*out).physical_acc_id).write(4);
    let uuid = b"GPU-0123456789abcdef\0";
    std::ptr::copy_nonoverlapping(
        uuid.as_ptr(),
        addr_of_mut!((*out).hip_uuid).cast::<u8>(),
        uuid.len(),
    );
    AmdsmiStatusT::AmdsmiStatusSuccess
}

#[test]
fn enumeration_returns_compute_identity_and_render_node() {
    let info: AmdsmiEnumerationInfoT =
        amdsmi::amdsmi_get_gpu_enumeration_info(NonNull::dangling().as_ptr()).unwrap();
    assert_eq!(info.drm_render, 141);
    assert_eq!(info.drm_card, 7);
    assert_eq!(info.hsa_id, 9);
    assert_eq!(info.hip_id, 2);
    assert_eq!(info.oam_id, 3);
    assert_eq!(info.physical_acc_id, 4);
    assert_eq!(
        unsafe { CStr::from_ptr(info.hip_uuid.as_ptr()) }.to_bytes(),
        b"GPU-0123456789abcdef"
    );
    assert!(info.hip_uuid[20..].iter().all(|&byte| byte == 0));
}

#[test]
fn enumeration_preserves_query_errors() {
    assert_eq!(
        amdsmi::amdsmi_get_gpu_enumeration_info(null_mut()).unwrap_err(),
        AmdsmiStatusT::AmdsmiStatusNotSupported
    );
}
