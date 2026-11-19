// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use crate::AmdsmiEnumerationInfoT;

#[test]
fn hip_uuid_getter_returns_c_string() {
    let mut info: AmdsmiEnumerationInfoT = unsafe { std::mem::zeroed() };
    let uuid = b"GPU-0123456789abcdef\0";
    for (dst, src) in info.hip_uuid.iter_mut().zip(uuid) {
        *dst = *src as std::os::raw::c_char;
    }
    assert_eq!(info.hip_uuid(), "GPU-0123456789abcdef");
}
