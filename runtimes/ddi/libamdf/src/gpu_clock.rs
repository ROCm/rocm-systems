// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GPU clock metadata and native host/device correlation samples.

use crate::generated::amdf::*;
use crate::instance::{self, Device};
use crate::support::{UNSUPPORTED, boundary, native, object, output_pointer};

fn clock_info(frequency_hz: u64) -> Result<amdf_observation_clock_info_t, u64> {
    if frequency_hz == 0 {
        return Err(UNSUPPORTED);
    }
    Ok(amdf_observation_clock_info_t {
        frequency_hz,
        counter_bit_count: 0,
        reserved: 0,
    })
}

pub(crate) unsafe extern "C" fn queue_clock_info(
    pointer: *mut amdf_device_t,
    ordinal: u32,
    out: *mut amdf_observation_clock_info_t,
) -> u64 {
    boundary(|| unsafe {
        output_pointer(out)?;
        let device = object(pointer.cast::<Device>())?;
        let _family = instance::family(device.native.endpoint(), ordinal)?;
        let gpu = device
            .native
            .gpu()
            .map_err(|error| device.native_error_status(&error))?;
        let counters = gpu
            .clock_counters()
            .map_err(|error| device.native_error_status(&error))?;
        out.write(clock_info(counters.gpu_frequency)?);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn clock_sample(
    pointer: *mut amdf_device_t,
    ordinal: u32,
    out: *mut amdf_observation_clock_sample_t,
) -> u64 {
    boundary(|| unsafe {
        output_pointer(out)?;
        let device = object(pointer.cast::<Device>())?;
        let _family = instance::family(device.native.endpoint(), ordinal)?;
        let gpu = device
            .native
            .gpu()
            .map_err(|error| device.native_error_status(&error))?;
        let before = rocddi::host::clock::read().map_err(|error| native(&error))?;
        let counters = gpu
            .clock_counters()
            .map_err(|error| device.native_error_status(&error))?;
        let after = rocddi::host::clock::read().map_err(|error| native(&error))?;
        let clock = clock_info(counters.gpu_frequency)?;
        out.write(amdf_observation_clock_sample_t {
            clock,
            timestamp: counters.gpu,
            host: amdf_observation_clock_sample_t__host {
                before,
                timestamp: counters.host,
                after,
            },
            system: amdf_observation_clock_sample_t__system {
                timestamp: counters.system,
                frequency_hz: counters.system_frequency,
            },
        });
        Ok(())
    })
}
