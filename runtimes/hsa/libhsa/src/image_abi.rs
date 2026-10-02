//! Image and sampler ABI entry points retained for binary compatibility.
//!
//! The image extension is never advertised. These symbols return
//! `HSA_STATUS_ERROR_NOT_SUPPORTED` without reading or changing caller storage.

use std::ffi::c_void;

use crate::ffi::*;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_get_capability(
    _agent: HsaAgent,
    _geometry: u32,
    _image_format: *const HsaExtImageFormat,
    _capability_mask: *mut u32,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_get_capability_with_layout(
    _agent: HsaAgent,
    _geometry: u32,
    _image_format: *const HsaExtImageFormat,
    _image_data_layout: u32,
    _capability_mask: *mut u32,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_data_get_info(
    _agent: HsaAgent,
    _image_descriptor: *const HsaExtImageDescriptor,
    _access_permission: u32,
    _image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_data_get_info_with_layout(
    _agent: HsaAgent,
    _image_descriptor: *const HsaExtImageDescriptor,
    _access_permission: u32,
    _image_data_layout: u32,
    _image_data_row_pitch: usize,
    _image_data_slice_pitch: usize,
    _image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_data_get_info_v2(
    _agent: HsaAgent,
    _image_descriptor: *const HsaExtImageDescriptorV2,
    _access_permission: u32,
    _image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_create(
    _agent: HsaAgent,
    _image_descriptor: *const HsaExtImageDescriptor,
    _image_data: *const c_void,
    _access_permission: u32,
    _image: *mut HsaExtImage,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_create_with_layout(
    _agent: HsaAgent,
    _image_descriptor: *const HsaExtImageDescriptor,
    _image_data: *const c_void,
    _access_permission: u32,
    _image_data_layout: u32,
    _image_data_row_pitch: usize,
    _image_data_slice_pitch: usize,
    _image: *mut HsaExtImage,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_create_v2(
    _agent: HsaAgent,
    _image_descriptor: *const HsaExtImageDescriptorV2,
    _image_data: *const c_void,
    _access_permission: u32,
    _image: *mut HsaExtImage,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_image_create(
    _agent: HsaAgent,
    _image_descriptor: *const HsaExtImageDescriptor,
    _image_layout: *const HsaAmdImageDescriptor,
    _image_data: *const c_void,
    _access_permission: u32,
    _image: *mut HsaExtImage,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_image_create_v2(
    _agent: HsaAgent,
    _image_descriptor: *const HsaExtImageDescriptorV2,
    _image_layout: *const HsaAmdImageDescriptor,
    _image_data: *const c_void,
    _access_permission: u32,
    _image: *mut HsaExtImage,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ext_image_destroy(_agent: HsaAgent, _image: HsaExtImage) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ext_image_destroy_v2(_agent: HsaAgent, _image: HsaExtImage) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_mipmap_array_get_level(
    _agent: HsaAgent,
    _mipmapped_array: *const HsaExtImage,
    _mip_level: u32,
    _image_descriptor: *const HsaExtImageDescriptorV2,
    _level_image_out: *mut HsaExtImage,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_copy(
    _agent: HsaAgent,
    _src_image: HsaExtImage,
    _src_offset: *const HsaDim3,
    _dst_image: HsaExtImage,
    _dst_offset: *const HsaDim3,
    _range: *const HsaDim3,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_import(
    _agent: HsaAgent,
    _src_memory: *const c_void,
    _src_row_pitch: usize,
    _src_slice_pitch: usize,
    _dst_image: HsaExtImage,
    _image_region: *const HsaExtImageRegion,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_export(
    _agent: HsaAgent,
    _src_image: HsaExtImage,
    _dst_memory: *mut c_void,
    _dst_row_pitch: usize,
    _dst_slice_pitch: usize,
    _image_region: *const HsaExtImageRegion,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_image_clear(
    _agent: HsaAgent,
    _image: HsaExtImage,
    _data: *const c_void,
    _image_region: *const HsaExtImageRegion,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_amd_image_get_info_max_dim(
    _agent: HsaAgent,
    _attribute: u32,
    _value: *mut c_void,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_sampler_create(
    _agent: HsaAgent,
    _sampler_descriptor: *const HsaExtSamplerDescriptor,
    _sampler: *mut HsaExtSampler,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_sampler_create_v2(
    _agent: HsaAgent,
    _sampler_descriptor: *const HsaExtSamplerDescriptorV2,
    _sampler: *mut HsaExtSampler,
) -> Status {
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ext_sampler_destroy(_agent: HsaAgent, _sampler: HsaExtSampler) -> Status {
    NOT_SUPPORTED
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsupported_image_entry_points_leave_outputs_untouched() {
        let agent = HsaAgent { handle: 0 };
        let mut capability = 0xfeed_u32;
        let mut image = HsaExtImage { handle: 0xfeed };
        let mut sampler = HsaExtSampler { handle: 0xfeed };
        // SAFETY: The stubs do not dereference any pointer arguments.
        unsafe {
            assert_eq!(
                hsa_ext_image_get_capability(agent, 0, std::ptr::null(), &raw mut capability),
                NOT_SUPPORTED
            );
            assert_eq!(
                hsa_ext_image_create(agent, std::ptr::null(), std::ptr::null(), 0, &raw mut image),
                NOT_SUPPORTED
            );
            assert_eq!(
                hsa_ext_sampler_create(agent, std::ptr::null(), &raw mut sampler),
                NOT_SUPPORTED
            );
        }
        assert_eq!(hsa_ext_image_destroy(agent, image), NOT_SUPPORTED);
        assert_eq!(hsa_ext_sampler_destroy(agent, sampler), NOT_SUPPORTED);
        assert_eq!(capability, 0xfeed);
        assert_eq!(image.handle, 0xfeed);
        assert_eq!(sampler.handle, 0xfeed);
    }
}
