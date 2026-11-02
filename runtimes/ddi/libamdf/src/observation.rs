// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMDF observation extension over the shared DDI profiling clock.

use crate::generated::amdf::*;
use crate::support::{boundary, native, output_pointer};

pub(crate) unsafe extern "C" fn host_clock_info(out: *mut amdf_observation_clock_info_t) -> u64 {
    boundary(|| {
        output_pointer(out)?;
        let info = rocddi::host::clock::info();
        // SAFETY: The caller supplied a writable clock-info record.
        unsafe {
            out.write(amdf_observation_clock_info_t {
                frequency_hz: info.frequency_hz,
                counter_bit_count: info.counter_bit_count,
                reserved: 0,
            });
        }
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn host_clock(out: *mut u64) -> u64 {
    boundary(|| {
        output_pointer(out)?;
        let value = rocddi::host::clock::read().map_err(|error| native(&error))?;
        // SAFETY: The caller supplied a writable timestamp slot.
        unsafe { out.write(value) };
        Ok(())
    })
}
