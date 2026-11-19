// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA sampler policy and descriptor ownership.

use super::*;
use rocddi::device::gpu::image::{
    SamplerAddressMode, SamplerAnisotropy, SamplerCoordinates, SamplerDescription, SamplerFilter,
    SamplerMipFilter, sampler_descriptor,
};

/// Validates HSA modes and normalizes its anisotropy policy before device lookup.
fn sampler_description(
    descriptor: HsaExtSamplerDescriptorV3,
) -> Result<SamplerDescription, Status> {
    let coordinates = match descriptor.coordinate_mode {
        0 => SamplerCoordinates::Unnormalized,
        1 => SamplerCoordinates::Normalized,
        _ => return Err(SAMPLER_DESCRIPTOR_UNSUPPORTED),
    };
    let filter = match descriptor.filter_mode {
        0 => SamplerFilter::Point,
        1 => SamplerFilter::Linear,
        _ => return Err(SAMPLER_DESCRIPTOR_UNSUPPORTED),
    };
    let mut address = [SamplerAddressMode::Repeat; 3];
    for (axis, mode) in address.iter_mut().zip(descriptor.address_modes) {
        *axis = match mode {
            0 | 3 => SamplerAddressMode::Repeat,
            1 => SamplerAddressMode::ClampEdge,
            2 => SamplerAddressMode::ClampBorder,
            4 => SamplerAddressMode::MirroredRepeat,
            _ => return Err(SAMPLER_DESCRIPTOR_UNSUPPORTED),
        };
    }
    let mip_filter = match descriptor.mipmap_filter_mode {
        0 => SamplerMipFilter::Point,
        1 => SamplerMipFilter::Linear,
        2 => SamplerMipFilter::Disabled,
        _ => return Err(SAMPLER_DESCRIPTOR_UNSUPPORTED),
    };
    // HSA sampler v3 rounds down non-power-of-two requests and clamps to 1..=16.
    let anisotropy = match descriptor.max_anisotropy.clamp(1, 16).ilog2() {
        0 => SamplerAnisotropy::One,
        1 => SamplerAnisotropy::Two,
        2 => SamplerAnisotropy::Four,
        3 => SamplerAnisotropy::Eight,
        _ => SamplerAnisotropy::Sixteen,
    };
    Ok(SamplerDescription {
        address,
        coordinates,
        filter,
        mip_filter,
        anisotropy,
    })
}

#[allow(unsafe_code)]
/// Creates sampler and records its ownership.
pub(crate) unsafe fn create_sampler(
    agent: HsaAgent,
    descriptor: HsaExtSamplerDescriptorV3,
    output: *mut HsaExtSampler,
) -> Status {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let description = match sampler_description(descriptor) {
        Ok(description) => description,
        Err(status) => return status,
    };
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    let Some(index) = runtime.agents.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    let info = runtime.agents.gpus[index].info();
    if !info.target_profile.supports_image_sampler() {
        return NOT_SUPPORTED;
    }
    let Some(generation) = Generation::from_gpu(&info) else {
        return NOT_SUPPORTED;
    };
    let words = sampler_descriptor(generation, description);
    if runtime.image_state.samplers.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    let device = runtime.agents.gpus[index].device.clone();
    let page = match u64::try_from(runtime.agents.host_page_size) {
        Ok(page) if page >= 32 => page,
        _ => return ERROR,
    };
    let allocation = match device.allocate(MemoryKind::System, page, page, DeviceAccess::READ) {
        Ok(allocation) => allocation,
        Err(error) => return map_error(error),
    };
    let Some(host) = allocation.info().host_address else {
        return NOT_SUPPORTED;
    };
    let address = match allocation.device_address(&device) {
        Ok(address) if address != 0 && address % 16 == 0 => address,
        Ok(_) => return ERROR,
        Err(error) => return map_error(error),
    };
    if runtime
        .image_state
        .samplers
        .contains_key(&(agent.handle, address))
    {
        return ERROR;
    }
    // SAFETY: The allocation owns a writable host mapping of at least one
    // page. The 32-byte descriptor fits and cannot be seen before publication.
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), host as *mut u32, words.len()) };
    fence(Ordering::Release);
    runtime
        .image_state
        .samplers
        .insert((agent.handle, address), SamplerRecord { allocation });
    // SAFETY: The caller supplied writable output storage for one HSA handle.
    unsafe { output.write(HsaExtSampler { handle: address }) };
    SUCCESS
}

/// Releases a sampler descriptor under image state ownership.
pub(crate) fn destroy_sampler(agent: HsaAgent, sampler: HsaExtSampler) -> Status {
    if agent.handle == 0 {
        return INVALID_AGENT;
    }
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    if runtime.agents.gpu_index(agent).is_none() {
        return INVALID_AGENT;
    }
    let key = (agent.handle, sampler.handle);
    let Some(record) = runtime.image_state.samplers.get_mut(&key) else {
        return INVALID_ARGUMENT;
    };
    if let Err(error) = record.allocation.free() {
        return map_error(error);
    }
    runtime.image_state.samplers.remove(&key);
    SUCCESS
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn sampler_anisotropy_rounds_down_and_saturates() {
        for filter_mode in 0..=1 {
            for (requested, ratio) in [
                (0, 0),
                (1, 0),
                (2, 1),
                (3, 1),
                (4, 2),
                (7, 2),
                (8, 3),
                (15, 3),
                (16, 4),
                (u32::MAX, 4),
            ] {
                let description = sampler_description(HsaExtSamplerDescriptorV3 {
                    coordinate_mode: 1,
                    filter_mode,
                    mipmap_filter_mode: 2,
                    address_modes: [1, 2, 4],
                    max_anisotropy: requested,
                })
                .unwrap();
                let words = sampler_descriptor(Generation::Gfx12, description);
                assert_eq!((words[0] >> 9) & 7, ratio);
                assert_eq!((words[0] >> 16) & 7, ratio >> 1);
                assert_eq!((words[0] >> 21) & 0x3f, ratio);
                assert_eq!(words[0] & 0x1ff, 2 | (6 << 3) | (1 << 6));
                let filter = filter_mode | if ratio == 0 { 0 } else { 2 };
                assert_eq!((words[2] >> 20) & 3, filter);
                assert_eq!((words[2] >> 22) & 3, filter);
                assert_eq!((words[2] >> 26) & 3, 0);
            }
        }
    }
}
