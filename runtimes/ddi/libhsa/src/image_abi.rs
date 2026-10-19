// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA image handles and gfx12 image and sampler descriptors.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};

use rocaddr3::{ResourceType, SurfaceLayout, SurfaceRequest};
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};

use crate::ffi::*;
use crate::image_gfx12::{self, Format, Layout};
use crate::runtime::{HsaState, InFlightToken, boundary, initialized_mut, lock, map_error};

const IMAGE_LAYOUT_OPAQUE: u32 = 0;
const IMAGE_LAYOUT_LINEAR: u32 = 1;
const IMAGE_CAPABILITY_READ_ONLY: u32 = 1;
const IMAGE_CAPABILITY_WRITE_ONLY: u32 = 2;
const IMAGE_CAPABILITY_READ_WRITE: u32 = 4;
const IMAGE_CAPABILITY_INVARIANT_LAYOUT: u32 = 16;

/// One GPU-visible image descriptor. HSA owns the descriptor and the caller
/// retains ownership of the image data for the handle's lifetime.
pub(crate) struct ImageRecord {
    /// GPU-visible resource descriptor backing owned by the HSA handle.
    allocation: Allocation,
    /// Image dimensions and format supplied by the caller.
    descriptor: HsaExtImageDescriptor,
    /// Hardware format and channel selector encoding.
    format: Format,
    /// Checked pitches and byte extent.
    layout: Layout,
    /// Caller-owned GPU address of image data.
    data: u64,
    /// Resource descriptor words written into `allocation`.
    words: [u32; 12],
    /// Number of levels in the image's mip chain.
    mip_levels: u32,
    /// Whether the data uses a linear hardware layout.
    linear: bool,
    /// `AddrLib3` surface details for non-linear images.
    native: Option<NativeImage>,
}

/// `AddrLib3` request and retained surface layout for a tiled image.
#[derive(Clone)]
struct NativeImage {
    /// Dimensions and GPU address configuration used for layout.
    request: SurfaceRequest,
    /// Calculated mip offsets, strides, and total size.
    layout: Arc<SurfaceLayout>,
    /// Mip level represented by this HSA image view.
    level: u32,
}

/// Mip layout tied to the HSA generation whose GPU supplied its address config.
struct NativeMipLayout {
    /// Format encoded in the resource descriptor.
    format: Format,
    /// Checked pitches and total backing extent.
    surface: Layout,
    /// Device geometry passed to `AddrLib3`.
    request: SurfaceRequest,
    /// Computed native offsets and swizzle mode.
    native: SurfaceLayout,
    /// Runtime generation that owns the queried device.
    generation: u64,
    /// Keeps that generation's device state alive through publication.
    _inflight: InFlightToken,
}

impl From<HsaExtImageDescriptorV2> for HsaExtImageDescriptor {
    fn from(value: HsaExtImageDescriptorV2) -> Self {
        Self {
            geometry: value.geometry,
            width: value.width,
            height: value.height,
            depth: value.depth,
            array_size: value.array_size,
            format: value.format,
        }
    }
}

fn gfx12_index(runtime: &HsaState, agent: HsaAgent) -> Result<usize, Status> {
    let index = runtime.gpu_index(agent).ok_or(INVALID_AGENT)?;
    let info = runtime.gpus[index].info;
    if !image_gfx12::supported_gpu(&info) {
        return Err(NOT_SUPPORTED);
    }
    Ok(index)
}

fn image_capability(
    runtime: &HsaState,
    agent: HsaAgent,
    geometry: u32,
    format: HsaExtImageFormat,
) -> Result<u32, Status> {
    gfx12_index(runtime, agent)?;
    if geometry > 7 {
        return Err(INVALID_ARGUMENT);
    }
    if !image_gfx12::supported_geometry(geometry) || image_gfx12::format(format).is_none() {
        return Ok(0);
    }
    Ok(IMAGE_CAPABILITY_READ_ONLY
        | IMAGE_CAPABILITY_WRITE_ONLY
        | IMAGE_CAPABILITY_READ_WRITE
        | IMAGE_CAPABILITY_INVARIANT_LAYOUT)
}

fn image_layout(
    runtime: &HsaState,
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    permission: u32,
    layout: u32,
    row_pitch: usize,
    slice_pitch: usize,
) -> Result<(usize, Format, Layout), Status> {
    let index = gfx12_index(runtime, agent)?;
    if !matches!(
        permission,
        ACCESS_PERMISSION_RO | ACCESS_PERMISSION_WO | ACCESS_PERMISSION_RW
    ) || !matches!(layout, IMAGE_LAYOUT_OPAQUE | IMAGE_LAYOUT_LINEAR)
    {
        return Err(INVALID_ARGUMENT);
    }
    let format = image_gfx12::format(descriptor.format).ok_or(IMAGE_FORMAT_UNSUPPORTED)?;
    let surface = image_gfx12::layout(descriptor, format, row_pitch, slice_pitch)?;
    Ok((index, format, surface))
}

fn maximum_mip_levels(descriptor: HsaExtImageDescriptor, surface: Layout) -> usize {
    let maximum_dimension = match descriptor.geometry {
        0 | 3 => surface.width,
        1 | 4 => surface.width.max(surface.height),
        2 => surface.width.max(surface.height).max(surface.slices),
        _ => 0,
    };
    (usize::BITS - maximum_dimension.leading_zeros()) as usize
}

fn native_mip_layout(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptorV2,
    permission: u32,
) -> Result<NativeMipLayout, Status> {
    let simple = descriptor.into();
    let (format, mut surface, info, device, generation, inflight) = {
        let guard = lock()?;
        let generation = guard.active_generation().ok_or(NOT_INITIALIZED)?;
        let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
        let (index, format, surface) = image_layout(
            runtime,
            agent,
            simple,
            permission,
            IMAGE_LAYOUT_OPAQUE,
            0,
            0,
        )?;
        let token = runtime.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
        (
            format,
            surface,
            runtime.gpus[index].info,
            runtime.gpus[index].device.clone(),
            generation,
            token,
        )
    };
    if descriptor.mipmap_levels < 2
        || descriptor.mipmap_levels > maximum_mip_levels(simple, surface)
        || descriptor.geometry == 5
    {
        return Err(IMAGE_SIZE_UNSUPPORTED);
    }
    let config = device
        .gpu()
        .and_then(|gpu| gpu.surface_config())
        .map_err(map_error)?;
    let resource_type = match descriptor.geometry {
        0 | 3 => ResourceType::Texture1D,
        1 | 4 => ResourceType::Texture2D,
        2 => ResourceType::Texture3D,
        _ => return Err(IMAGE_SIZE_UNSUPPORTED),
    };
    let request = SurfaceRequest {
        family: info.asic_family_id,
        gb_addr_config: config.gb_addr_config,
        bits_per_pixel: u32::try_from(format.bytes.checked_mul(8).ok_or(IMAGE_SIZE_UNSUPPORTED)?)
            .map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        width: u32::try_from(surface.width).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        height: u32::try_from(surface.height).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        slices: u32::try_from(surface.slices).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        mip_levels: u32::try_from(descriptor.mipmap_levels).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?,
        resource_type,
    };
    let native = rocaddr3::compute(request).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    surface.row_pitch = usize::try_from(native.pitch)
        .ok()
        .and_then(|pitch| pitch.checked_mul(format.bytes))
        .ok_or(IMAGE_SIZE_UNSUPPORTED)?;
    surface.slice_pitch = usize::try_from(native.slice_size).map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
    surface.size = usize::try_from(native.size)
        .ok()
        .filter(|size| isize::try_from(*size).is_ok())
        .ok_or(IMAGE_SIZE_UNSUPPORTED)?;
    Ok(NativeMipLayout {
        format,
        surface,
        request,
        native,
        generation,
        _inflight: inflight,
    })
}

#[allow(unsafe_code)]
unsafe fn create_native_mip_image(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptorV2,
    data: *const c_void,
    permission: u32,
    output: *mut HsaExtImage,
) -> Status {
    if data.is_null() || output.is_null() {
        return INVALID_ARGUMENT;
    }
    let NativeMipLayout {
        format,
        surface,
        request,
        native,
        generation,
        _inflight,
    } = match native_mip_layout(agent, descriptor, permission) {
        Ok(layout) => layout,
        Err(status) => return status,
    };
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    if guard.active_generation() != Some(generation) {
        return INVALID_RUNTIME_STATE;
    }
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    let data_address = match crate::memory::image_device_address(
        runtime,
        data as usize,
        surface.size,
        agent,
        permission,
    ) {
        Ok(address) => address,
        Err(status) => return status,
    };
    let words = match image_gfx12::native_mip_descriptor(
        descriptor.into(),
        format,
        &native,
        data_address,
    ) {
        Ok(words) => words,
        Err(status) => return status,
    };
    let native = NativeImage {
        request,
        layout: Arc::new(native),
        level: 0,
    };
    // SAFETY: The live runtime owns the descriptor allocation, and the C
    // caller provides one writable output image handle.
    unsafe {
        publish_image(
            runtime,
            agent,
            PreparedImage {
                descriptor: descriptor.into(),
                format,
                surface,
                data_address,
                words,
                mip_levels: descriptor.mipmap_levels as u32,
                linear: false,
                native: Some(native),
            },
            output,
        )
    }
}

#[allow(unsafe_code)]
unsafe fn get_info(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    permission: u32,
    layout: u32,
    row_pitch: usize,
    slice_pitch: usize,
    output: *mut HsaExtImageDataInfo,
) -> Status {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let Some(runtime) = guard.as_ref() else {
        return NOT_INITIALIZED;
    };
    let (_, _, surface) = match image_layout(
        runtime,
        agent,
        descriptor,
        permission,
        layout,
        row_pitch,
        slice_pitch,
    ) {
        Ok(result) => result,
        Err(status) => return status,
    };
    // SAFETY: The caller supplied writable output storage for one info record.
    unsafe { output.write(image_gfx12::info(surface)) };
    SUCCESS
}

/// Caller input copied before creating and publishing an image handle.
#[derive(Clone, Copy)]
struct ImageRequest {
    /// Requested HSA image geometry and format.
    descriptor: HsaExtImageDescriptor,
    /// Caller-owned image data address.
    data: *const c_void,
    /// Requested image access permission.
    permission: u32,
    /// Linear or opaque hardware layout selector.
    layout: u32,
    /// Optional caller-specified row stride in bytes.
    row_pitch: usize,
    /// Optional caller-specified slice stride in bytes.
    slice_pitch: usize,
    /// Caller-owned output slot for the new image handle.
    output: *mut HsaExtImage,
}

/// Validated descriptor and native backing prepared before publication.
struct PreparedImage {
    /// Validated HSA image dimensions and format.
    descriptor: HsaExtImageDescriptor,
    /// Hardware format encoding.
    format: Format,
    /// Checked image pitches and byte extent.
    surface: Layout,
    /// GPU address of caller-owned image data.
    data_address: u64,
    /// Resource descriptor words to publish to the GPU.
    words: [u32; 12],
    /// Number of mip levels represented by this image.
    mip_levels: u32,
    /// Whether the data uses a linear hardware layout.
    linear: bool,
    /// `AddrLib3` layout retained for a tiled image.
    native: Option<NativeImage>,
}

#[allow(unsafe_code)]
unsafe fn publish_image(
    runtime: &mut HsaState,
    agent: HsaAgent,
    prepared: PreparedImage,
    output: *mut HsaExtImage,
) -> Status {
    let PreparedImage {
        descriptor,
        format,
        surface,
        data_address,
        words,
        mip_levels,
        linear,
        native,
    } = prepared;
    if runtime.image_state.images.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    let index = match runtime.gpu_index(agent) {
        Some(index) => index,
        None => return INVALID_AGENT,
    };
    let device = runtime.gpus[index].device.clone();
    let page = match u64::try_from(runtime.host_page_size) {
        Ok(page) if page >= 64 => page,
        _ => return ERROR,
    };
    let allocation = match device.allocate(MemoryKind::System, page, page, DeviceAccess::READ) {
        Ok(allocation) => allocation,
        Err(error) => return map_error(error),
    };
    let Some(host) = allocation.info().host_address else {
        return NOT_SUPPORTED;
    };
    let handle = match allocation.device_address(&device) {
        Ok(address) if address != 0 && address % 16 == 0 => address,
        Ok(_) => return ERROR,
        Err(error) => return map_error(error),
    };
    if runtime
        .image_state
        .images
        .contains_key(&(agent.handle, handle))
    {
        return ERROR;
    }
    // SAFETY: The mapping owns a writable page and the 48-byte descriptor fits.
    unsafe { std::ptr::copy_nonoverlapping(words.as_ptr(), host as *mut u32, words.len()) };
    fence(Ordering::Release);
    runtime.image_state.images.insert(
        (agent.handle, handle),
        ImageRecord {
            allocation,
            descriptor,
            format,
            layout: surface,
            data: data_address,
            words,
            mip_levels,
            linear,
            native,
        },
    );
    // SAFETY: The caller provided output storage for one image handle.
    unsafe { output.write(HsaExtImage { handle }) };
    SUCCESS
}

#[allow(unsafe_code)]
unsafe fn create_image(agent: HsaAgent, request: ImageRequest) -> Status {
    let ImageRequest {
        descriptor,
        data,
        permission,
        layout,
        row_pitch,
        slice_pitch,
        output,
    } = request;
    if data.is_null() || output.is_null() {
        return INVALID_ARGUMENT;
    }
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    let (_, format, surface) = match image_layout(
        runtime,
        agent,
        descriptor,
        permission,
        layout,
        row_pitch,
        slice_pitch,
    ) {
        Ok(result) => result,
        Err(status) => return status,
    };
    let data_address = match crate::memory::image_device_address(
        runtime,
        data as usize,
        surface.size,
        agent,
        permission,
    ) {
        Ok(address) => address,
        Err(status) => return status,
    };
    let words = match image_gfx12::descriptor(descriptor, format, surface, data_address) {
        Ok(words) => words,
        Err(status) => return status,
    };
    // SAFETY: `publish_image` owns the descriptor allocation and the caller
    // supplied one writable output handle.
    unsafe {
        publish_image(
            runtime,
            agent,
            PreparedImage {
                descriptor,
                format,
                surface,
                data_address,
                words,
                mip_levels: 1,
                linear: true,
                native: None,
            },
            output,
        )
    }
}

/// Consumes a version-1 AMD descriptor containing eight gfx12 SRD words.
/// The producer owns the surface extent and compression metadata; HSA rebases
/// its address and checks that the importing device can reach the base. The
/// caller retains the complete backing described by the vendor SRD.
#[allow(unsafe_code)]
unsafe fn create_imported_image(
    agent: HsaAgent,
    descriptor: HsaExtImageDescriptor,
    mip_levels: usize,
    native_layout: *const HsaAmdImageDescriptor,
    data: *const c_void,
    permission: u32,
    output: *mut HsaExtImage,
) -> Status {
    if data.is_null() || output.is_null() {
        return INVALID_ARGUMENT;
    }
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    let (index, format, surface) = match image_layout(
        runtime,
        agent,
        descriptor,
        permission,
        IMAGE_LAYOUT_OPAQUE,
        0,
        0,
    ) {
        Ok(result) => result,
        Err(status) => return status,
    };
    if mip_levels > 1 {
        if descriptor.geometry == 5 {
            return IMAGE_SIZE_UNSUPPORTED;
        }
        if mip_levels > maximum_mip_levels(descriptor, surface) {
            return IMAGE_SIZE_UNSUPPORTED;
        }
    }
    // SAFETY: The caller supplies a readable AMD descriptor. Version 1 means
    // its flexible data member contains eight readable SRD words.
    let native = unsafe { &*native_layout };
    let Some(pci) = runtime.gpus[index].endpoint.pci else {
        return NOT_SUPPORTED;
    };
    if native.version != 1
        || pci.vendor_id != 0x1002
        || native.device_id != ((0x1002 << 16) | pci.device_id)
    {
        return IMAGE_FORMAT_UNSUPPORTED;
    }
    let data_address =
        match crate::memory::image_device_address(runtime, data as usize, 1, agent, permission) {
            Ok(address) => address,
            Err(status) => return status,
        };
    // SAFETY: Version-1 descriptors contain the eight source words, and the
    // caller keeps the descriptor valid for this synchronous call.
    let source = unsafe { std::slice::from_raw_parts(native.data.as_ptr(), 8) };
    let mut srd = [0_u32; 8];
    srd.copy_from_slice(source);
    let words =
        match image_gfx12::imported_descriptor(descriptor, format, surface, data_address, srd) {
            Ok(words) => words,
            Err(status) => return status,
        };
    let words = if mip_levels > 1 {
        match image_gfx12::imported_mip_chain(words, descriptor, srd, mip_levels) {
            Ok(words) => words,
            Err(status) => return status,
        }
    } else {
        words
    };
    // SAFETY: The output handle is writable and the descriptor allocation is
    // retained by the runtime until explicit image destruction.
    unsafe {
        publish_image(
            runtime,
            agent,
            PreparedImage {
                descriptor,
                format,
                surface,
                data_address,
                words,
                mip_levels: mip_levels as u32,
                linear: false,
                native: None,
            },
            output,
        )
    }
}

/// GPU-visible backing for one sampler descriptor. The public handle is its
/// GPU address, so the allocation must outlive every dispatch using it.
pub(crate) struct SamplerRecord {
    /// Descriptor allocation retained while its GPU address is public.
    allocation: Allocation,
}

/// HSA image and sampler handles around device-visible descriptors.
pub(crate) struct ImageState {
    /// Samplers keyed by agent and public sampler handle.
    samplers: HashMap<(u64, u64), SamplerRecord>,
    /// Images keyed by agent and public image handle.
    images: HashMap<(u64, u64), ImageRecord>,
}

impl ImageState {
    pub(crate) fn new() -> Self {
        Self {
            samplers: HashMap::new(),
            images: HashMap::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.samplers.clear();
        self.images.clear();
    }
}

fn sampler_words(descriptor: HsaExtSamplerDescriptorV2) -> Result<[u32; 8], Status> {
    if descriptor.coordinate_mode > 1 || descriptor.filter_mode > 1 {
        return Err(SAMPLER_DESCRIPTOR_UNSUPPORTED);
    }
    let mut clamp = [0_u32; 3];
    for (axis, mode) in clamp.iter_mut().zip(descriptor.address_modes) {
        *axis = match mode {
            0 | 3 => 0, // Undefined and repeat wrap.
            1 => 2,     // Clamp to last texel.
            2 => 6,     // Clamp to border.
            4 => 1,     // Mirrored repeat.
            _ => return Err(SAMPLER_DESCRIPTOR_UNSUPPORTED),
        };
    }
    let mip_filter = match descriptor.mipmap_filter_mode {
        0 => 1, // Point.
        1 => 2, // Linear.
        2 => 0, // Disabled.
        _ => return Err(SAMPLER_DESCRIPTOR_UNSUPPORTED),
    };
    let word0 =
        clamp[0] | clamp[1] << 3 | clamp[2] << 6 | u32::from(descriptor.coordinate_mode == 0) << 15;
    let word1 = 4095 << 13;
    let word2 =
        (descriptor.filter_mode << 20) | (descriptor.filter_mode << 22) | (mip_filter << 26);
    Ok([word0, word1, word2, 0, 0, 0, 0, 0])
}

#[allow(unsafe_code)]
unsafe fn create_sampler(
    agent: HsaAgent,
    descriptor: HsaExtSamplerDescriptorV2,
    output: *mut HsaExtSampler,
) -> Status {
    if output.is_null() {
        return INVALID_ARGUMENT;
    }
    let words = match sampler_words(descriptor) {
        Ok(words) => words,
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
    let Some(index) = runtime.gpu_index(agent) else {
        return INVALID_AGENT;
    };
    let info = runtime.gpus[index].info;
    if !matches!(
        (info.gfx_major, info.gfx_minor, info.gfx_stepping),
        (12, 0, 0 | 1)
    ) {
        return NOT_SUPPORTED;
    }
    if runtime.image_state.samplers.try_reserve(1).is_err() {
        return OUT_OF_RESOURCES;
    }
    let device = runtime.gpus[index].device.clone();
    let page = match u64::try_from(runtime.host_page_size) {
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

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_get_capability")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_get_capability(
    agent: HsaAgent,
    geometry: u32,
    image_format: *const HsaExtImageFormat,
    capability_mask: *mut u32,
) -> Status {
    boundary(|| {
        if image_format.is_null() || capability_mask.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        // SAFETY: The caller supplies one readable format and one writable mask.
        let format = unsafe { *image_format };
        let mask = match image_capability(runtime, agent, geometry, format) {
            Ok(mask) => mask,
            Err(status) => return status,
        };
        unsafe { capability_mask.write(mask) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_get_capability_with_layout")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_get_capability_with_layout(
    agent: HsaAgent,
    geometry: u32,
    image_format: *const HsaExtImageFormat,
    image_data_layout: u32,
    capability_mask: *mut u32,
) -> Status {
    if image_data_layout != IMAGE_LAYOUT_LINEAR {
        return INVALID_ARGUMENT;
    }
    unsafe { hsa_ext_image_get_capability(agent, geometry, image_format, capability_mask) }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_data_get_info")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_data_get_info(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    access_permission: u32,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        unsafe {
            get_info(
                agent,
                descriptor,
                access_permission,
                IMAGE_LAYOUT_OPAQUE,
                0,
                0,
                image_data_info,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_data_get_info_with_layout")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_data_get_info_with_layout(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    access_permission: u32,
    image_data_layout: u32,
    image_data_row_pitch: usize,
    image_data_slice_pitch: usize,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data_layout != IMAGE_LAYOUT_LINEAR {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        unsafe {
            get_info(
                agent,
                descriptor,
                access_permission,
                image_data_layout,
                image_data_row_pitch,
                image_data_slice_pitch,
                image_data_info,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_data_get_info_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_data_get_info_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    access_permission: u32,
    image_data_info: *mut HsaExtImageDataInfo,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        if descriptor.mipmap_levels > 1 {
            if image_data_info.is_null() {
                return INVALID_ARGUMENT;
            }
            let layout = match native_mip_layout(agent, descriptor, access_permission) {
                Ok(layout) => layout,
                Err(status) => return status,
            };
            // SAFETY: The caller supplied writable output storage for one
            // image data information record.
            unsafe {
                image_data_info.write(HsaExtImageDataInfo {
                    size: layout.surface.size,
                    alignment: layout.native.alignment as usize,
                });
            }
            return SUCCESS;
        }
        unsafe {
            get_info(
                agent,
                descriptor.into(),
                access_permission,
                IMAGE_LAYOUT_OPAQUE,
                0,
                0,
                image_data_info,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_create(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        unsafe {
            create_image(
                agent,
                ImageRequest {
                    descriptor,
                    data: image_data,
                    permission: access_permission,
                    layout: IMAGE_LAYOUT_OPAQUE,
                    row_pitch: 0,
                    slice_pitch: 0,
                    output: image,
                },
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_create_with_layout")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_create_with_layout(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image_data_layout: u32,
    image_data_row_pitch: usize,
    image_data_slice_pitch: usize,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() || image_data_layout != IMAGE_LAYOUT_LINEAR {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        unsafe {
            create_image(
                agent,
                ImageRequest {
                    descriptor,
                    data: image_data,
                    permission: access_permission,
                    layout: image_data_layout,
                    row_pitch: image_data_row_pitch,
                    slice_pitch: image_data_slice_pitch,
                    output: image,
                },
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_create_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_create_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        if descriptor.mipmap_levels > 1 {
            // SAFETY: The caller supplies image backing and one writable
            // output handle for this synchronous creation call.
            return unsafe {
                create_native_mip_image(agent, descriptor, image_data, access_permission, image)
            };
        }
        unsafe {
            create_image(
                agent,
                ImageRequest {
                    descriptor: descriptor.into(),
                    data: image_data,
                    permission: access_permission,
                    layout: IMAGE_LAYOUT_OPAQUE,
                    row_pitch: 0,
                    slice_pitch: 0,
                    output: image,
                },
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_image_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_amd_image_create(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptor,
    image_layout: *const HsaAmdImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        if !image_layout.is_null() {
            // SAFETY: The C caller supplies a readable versioned layout,
            // image backing, and output storage for this synchronous call.
            return unsafe {
                create_imported_image(
                    agent,
                    descriptor,
                    1,
                    image_layout,
                    image_data,
                    access_permission,
                    image,
                )
            };
        }
        unsafe {
            create_image(
                agent,
                ImageRequest {
                    descriptor,
                    data: image_data,
                    permission: access_permission,
                    layout: IMAGE_LAYOUT_OPAQUE,
                    row_pitch: 0,
                    slice_pitch: 0,
                    output: image,
                },
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_image_create_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_amd_image_create_v2(
    agent: HsaAgent,
    image_descriptor: *const HsaExtImageDescriptorV2,
    image_layout: *const HsaAmdImageDescriptor,
    image_data: *const c_void,
    access_permission: u32,
    image: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if image_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        let descriptor = unsafe { *image_descriptor };
        if descriptor.mipmap_levels > 1 && image_layout.is_null() {
            // SAFETY: The caller supplies image backing and one writable
            // output handle for this synchronous creation call.
            return unsafe {
                create_native_mip_image(agent, descriptor, image_data, access_permission, image)
            };
        }
        if !image_layout.is_null() {
            // SAFETY: The C caller supplies a readable versioned layout,
            // image backing, and output storage for this synchronous call.
            return unsafe {
                create_imported_image(
                    agent,
                    descriptor.into(),
                    descriptor.mipmap_levels.max(1),
                    image_layout,
                    image_data,
                    access_permission,
                    image,
                )
            };
        }
        unsafe {
            create_image(
                agent,
                ImageRequest {
                    descriptor: descriptor.into(),
                    data: image_data,
                    permission: access_permission,
                    layout: IMAGE_LAYOUT_OPAQUE,
                    row_pitch: 0,
                    slice_pitch: 0,
                    output: image,
                },
            )
        }
    })
}

fn destroy_image(agent: HsaAgent, image: HsaExtImage) -> Status {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(status) => return status,
    };
    let runtime = match initialized_mut(&mut guard) {
        Ok(runtime) => runtime,
        Err(status) => return status,
    };
    if runtime.gpu_index(agent).is_none() {
        return INVALID_AGENT;
    }
    let key = (agent.handle, image.handle);
    let Some(record) = runtime.image_state.images.get_mut(&key) else {
        return INVALID_ARGUMENT;
    };
    if let Err(error) = record.allocation.free() {
        return map_error(error);
    }
    runtime.image_state.images.remove(&key);
    SUCCESS
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_ext_image_destroy(agent: HsaAgent, image: HsaExtImage) -> Status {
    boundary(|| destroy_image(agent, image))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_destroy_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_ext_image_destroy_v2(agent: HsaAgent, image: HsaExtImage) -> Status {
    boundary(|| destroy_image(agent, image))
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_mipmap_array_get_level")
)]
/// # Safety
/// The non-null pointers must address readable parent and optional descriptor
/// records and writable storage for one output image handle.
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_mipmap_array_get_level(
    agent: HsaAgent,
    mipmapped_array: *const HsaExtImage,
    mip_level: u32,
    image_descriptor: *const HsaExtImageDescriptorV2,
    level_image_out: *mut HsaExtImage,
) -> Status {
    boundary(|| {
        if mipmapped_array.is_null() || level_image_out.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller provides a readable parent handle and, when
        // present, one readable level descriptor for this synchronous call.
        let parent_handle = unsafe { *mipmapped_array };
        let requested = if image_descriptor.is_null() {
            None
        } else {
            Some(unsafe { *image_descriptor })
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if let Err(status) = gfx12_index(runtime, agent) {
            return status;
        }
        let Some(parent) = runtime
            .image_state
            .images
            .get(&(agent.handle, parent_handle.handle))
        else {
            return INVALID_ARGUMENT;
        };
        if parent.mip_levels <= 1 || mip_level >= parent.mip_levels {
            return INVALID_ARGUMENT;
        }
        let mut descriptor = parent.descriptor;
        let level_dimension = |extent: usize| {
            if extent == 0 {
                0
            } else {
                extent.checked_shr(mip_level).unwrap_or(0).max(1)
            }
        };
        descriptor.width = level_dimension(descriptor.width);
        descriptor.height = level_dimension(descriptor.height);
        descriptor.depth = level_dimension(descriptor.depth);
        let format = if let Some(requested) = requested {
            if requested.mipmap_levels > 1
                || requested.geometry != descriptor.geometry
                || requested.width != descriptor.width
                || requested.height != descriptor.height
                || requested.depth != descriptor.depth
                || requested.array_size != descriptor.array_size
            {
                return INVALID_ARGUMENT;
            }
            descriptor.format = requested.format;
            match image_gfx12::format(descriptor.format) {
                Some(format) if format.bytes == parent.format.bytes => format,
                _ => return IMAGE_FORMAT_UNSUPPORTED,
            }
        } else {
            parent.format
        };
        let surface = match image_gfx12::layout(descriptor, format, 0, 0) {
            Ok(surface) => surface,
            Err(status) => return status,
        };
        let words = image_gfx12::mip_level_view(parent.words, mip_level, descriptor, format);
        let data_address = parent.data;
        let native = parent.native.as_ref().map(|parent| NativeImage {
            request: parent.request,
            layout: Arc::clone(&parent.layout),
            level: mip_level,
        });
        // SAFETY: The output points to one writable image handle. The view
        // receives its own GPU-visible SRD and remains valid after the parent
        // handle is destroyed while the caller retains the imported backing.
        unsafe {
            publish_image(
                runtime,
                agent,
                PreparedImage {
                    descriptor,
                    format,
                    surface,
                    data_address,
                    words,
                    mip_levels: 1,
                    linear: false,
                    native,
                },
                level_image_out,
            )
        }
    })
}

/// Image facts copied while its HSA owner remains protected by an in-flight token.
#[derive(Clone)]
struct ImageSnapshot {
    /// Public image geometry and format.
    descriptor: HsaExtImageDescriptor,
    /// Hardware format encoding.
    format: Format,
    /// Checked image pitches and byte extent.
    layout: Layout,
    /// GPU address of caller-owned image data.
    data: u64,
    /// Retained tiled layout, when present.
    native: Option<NativeImage>,
}

fn image_snapshot(
    agent: HsaAgent,
    image: HsaExtImage,
) -> Result<(ImageSnapshot, InFlightToken), Status> {
    let guard = lock()?;
    let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
    if runtime.gpu_index(agent).is_none() {
        return Err(INVALID_AGENT);
    }
    let record = runtime
        .image_state
        .images
        .get(&(agent.handle, image.handle))
        .ok_or(INVALID_ARGUMENT)?;
    if !record.linear && record.native.is_none() {
        return Err(NOT_SUPPORTED);
    }
    let token = runtime.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
    Ok((
        ImageSnapshot {
            descriptor: record.descriptor,
            format: record.format,
            layout: record.layout,
            data: record.data,
            native: record.native.clone(),
        },
        token,
    ))
}

impl ImageSnapshot {
    fn row_address(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        usize::try_from(self.data)
            .ok()?
            .checked_add(z.checked_mul(self.layout.slice_pitch)?)?
            .checked_add(y.checked_mul(self.layout.row_pitch)?)?
            .checked_add(x.checked_mul(self.format.bytes)?)
    }

    fn region(&self, region: HsaExtImageRegion) -> Result<Region, Status> {
        let offset = [
            region.offset.x as usize,
            region.offset.y as usize,
            region.offset.z as usize,
        ];
        let range = [
            region.range.x as usize,
            region.range.y as usize,
            region.range.z as usize,
        ];
        let extent = if self.descriptor.geometry == 3 {
            [self.layout.width, self.layout.slices, 1]
        } else {
            [self.layout.width, self.layout.height, self.layout.slices]
        };
        if range.contains(&0)
            || (0..3).any(|axis| {
                offset[axis]
                    .checked_add(range[axis])
                    .is_none_or(|end| end > extent[axis])
            })
        {
            return Err(INVALID_ARGUMENT);
        }
        Ok(Region { offset, range })
    }

    fn region_row_address(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if self.descriptor.geometry == 3 {
            self.row_address(x, 0, y)
        } else {
            self.row_address(x, y, z)
        }
    }
}

/// Validated three-dimensional image subregion.
#[derive(Clone, Copy)]
struct Region {
    /// Starting element coordinates by dimension.
    offset: [usize; 3],
    /// Element counts by dimension.
    range: [usize; 3],
}

fn external_pitches(
    row_bytes: usize,
    rows: usize,
    row_pitch: usize,
    slice_pitch: usize,
) -> Result<(usize, usize), Status> {
    let row_pitch = if row_pitch == 0 { row_bytes } else { row_pitch };
    let minimum_slice = row_pitch.checked_mul(rows).ok_or(INVALID_ARGUMENT)?;
    let slice_pitch = if slice_pitch == 0 {
        minimum_slice
    } else {
        slice_pitch
    };
    if row_pitch < row_bytes || slice_pitch < minimum_slice {
        return Err(IMAGE_PITCH_UNSUPPORTED);
    }
    Ok((row_pitch, slice_pitch))
}

fn row_offset(z: usize, y: usize, row_pitch: usize, slice_pitch: usize) -> Option<usize> {
    z.checked_mul(slice_pitch)?
        .checked_add(y.checked_mul(row_pitch)?)
}

const TILED_TRANSFER_CHUNK_BYTES: usize = 1024 * 1024;

fn zeroed_bytes(size: usize) -> Result<Vec<u8>, Status> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(size)
        .map_err(|_| OUT_OF_RESOURCES)?;
    bytes.resize(size, 0);
    Ok(bytes)
}

fn tiled_mapper(snapshot: &ImageSnapshot) -> Result<rocaddr3::SurfaceMapper, Status> {
    let native = snapshot.native.as_ref().ok_or(NOT_SUPPORTED)?;
    rocaddr3::SurfaceMapper::new(native.request, &native.layout).map_err(|_| IMAGE_SIZE_UNSUPPORTED)
}

/// Entries are sorted by physical byte offset so adjacent texels can use one
/// HSA copy. Each entry's second field selects a texel in the row-major chunk.
fn tiled_chunk_addresses(
    mapper: &mut rocaddr3::SurfaceMapper,
    snapshot: &ImageSnapshot,
    region: Region,
    z: usize,
    first_row: usize,
    rows: usize,
) -> Result<Vec<(u64, usize)>, Status> {
    let native = snapshot.native.as_ref().ok_or(NOT_SUPPORTED)?;
    let width = region.range[0];
    let count = width.checked_mul(rows).ok_or(INVALID_ARGUMENT)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(count)
        .map_err(|_| OUT_OF_RESOURCES)?;
    let mut offsets = Vec::<u64>::new();
    offsets
        .try_reserve_exact(width)
        .map_err(|_| OUT_OF_RESOURCES)?;
    offsets.resize(width, 0);
    for row in 0..rows {
        let logical_y = region.offset[1] + first_row + row;
        let (y, slice) = if snapshot.descriptor.geometry == 3 {
            (0, logical_y)
        } else {
            (logical_y, region.offset[2] + z)
        };
        mapper
            .map_row(
                native.level,
                u32::try_from(region.offset[0]).map_err(|_| INVALID_ARGUMENT)?,
                u32::try_from(y).map_err(|_| INVALID_ARGUMENT)?,
                u32::try_from(slice).map_err(|_| INVALID_ARGUMENT)?,
                &mut offsets,
            )
            .map_err(|_| IMAGE_SIZE_UNSUPPORTED)?;
        for (x, &offset) in offsets.iter().enumerate() {
            entries.push((offset, row * width + x));
        }
    }
    entries.sort_unstable_by_key(|entry| entry.0);
    if entries.windows(2).any(|pair| {
        pair[0]
            .0
            .checked_add(snapshot.format.bytes as u64)
            .is_none_or(|end| end > pair[1].0)
    }) {
        return Err(ERROR);
    }
    Ok(entries)
}

fn next_tiled_run(entries: &[(u64, usize)], first: usize, element_bytes: usize) -> usize {
    let mut end = first + 1;
    while end < entries.len()
        && entries[end - 1].0.checked_add(element_bytes as u64) == Some(entries[end].0)
    {
        end += 1;
    }
    end
}

/// Source for writing one tiled image region.
#[derive(Clone, Copy)]
enum TiledSource<'a> {
    /// Strided caller-owned host rows and slices.
    Rows {
        address: usize,
        row_pitch: usize,
        slice_pitch: usize,
    },
    /// One pixel repeated across the region.
    Pixel(&'a [u8]),
}

#[allow(unsafe_code)]
unsafe fn write_tiled_rows(
    snapshot: &ImageSnapshot,
    region: Region,
    source: TiledSource<'_>,
) -> Status {
    let element_bytes = snapshot.format.bytes;
    let Some(row_bytes) = region.range[0].checked_mul(element_bytes) else {
        return INVALID_ARGUMENT;
    };
    if matches!(source, TiledSource::Pixel(pixel) if pixel.len() != element_bytes) {
        return INVALID_ARGUMENT;
    }
    let chunk_rows = (TILED_TRANSFER_CHUNK_BYTES / row_bytes).max(1);
    let Ok(base) = usize::try_from(snapshot.data) else {
        return INVALID_ALLOCATION;
    };
    let mut mapper = match tiled_mapper(snapshot) {
        Ok(mapper) => mapper,
        Err(status) => return status,
    };
    for z in 0..region.range[2] {
        for first_row in (0..region.range[1]).step_by(chunk_rows) {
            let rows = (region.range[1] - first_row).min(chunk_rows);
            let Some(chunk_bytes) = row_bytes.checked_mul(rows) else {
                return INVALID_ARGUMENT;
            };
            let mut chunk = match zeroed_bytes(chunk_bytes) {
                Ok(bytes) => bytes,
                Err(status) => return status,
            };
            match source {
                TiledSource::Rows {
                    address,
                    row_pitch,
                    slice_pitch,
                } => {
                    for row in 0..rows {
                        let Some(offset) = row_offset(z, first_row + row, row_pitch, slice_pitch)
                        else {
                            return INVALID_ARGUMENT;
                        };
                        let Some(source) = address.checked_add(offset) else {
                            return INVALID_ARGUMENT;
                        };
                        // SAFETY: HSA validates registered GPU ranges; the
                        // caller owns each external row for this operation.
                        let status = unsafe {
                            crate::memory::hsa_memory_copy(
                                chunk[row * row_bytes..].as_mut_ptr().cast(),
                                source as *const c_void,
                                row_bytes,
                            )
                        };
                        if status != SUCCESS {
                            return status;
                        }
                    }
                }
                TiledSource::Pixel(pixel) => {
                    for element in chunk.chunks_exact_mut(element_bytes) {
                        element.copy_from_slice(pixel);
                    }
                }
            }
            let entries =
                match tiled_chunk_addresses(&mut mapper, snapshot, region, z, first_row, rows) {
                    Ok(entries) => entries,
                    Err(status) => return status,
                };
            let mut run = match zeroed_bytes(chunk_bytes) {
                Ok(bytes) => bytes,
                Err(status) => return status,
            };
            let mut first = 0;
            while first < entries.len() {
                let end = next_tiled_run(&entries, first, element_bytes);
                let bytes = (end - first) * element_bytes;
                for (index, &(_, logical)) in entries[first..end].iter().enumerate() {
                    let source = logical * element_bytes;
                    let target = index * element_bytes;
                    run[target..target + element_bytes]
                        .copy_from_slice(&chunk[source..source + element_bytes]);
                }
                let Ok(offset) = usize::try_from(entries[first].0) else {
                    return INVALID_ALLOCATION;
                };
                let Some(destination) = base.checked_add(offset) else {
                    return INVALID_ALLOCATION;
                };
                // SAFETY: AddrLib mapped every texel inside the caller-owned
                // image backing and the copy source is initialized host data.
                let status = unsafe {
                    crate::memory::hsa_memory_copy(
                        destination as *mut c_void,
                        run.as_ptr().cast(),
                        bytes,
                    )
                };
                if status != SUCCESS {
                    return status;
                }
                first = end;
            }
        }
    }
    SUCCESS
}

#[allow(unsafe_code)]
unsafe fn read_tiled_rows(
    snapshot: &ImageSnapshot,
    region: Region,
    destination: usize,
    row_pitch: usize,
    slice_pitch: usize,
) -> Status {
    let element_bytes = snapshot.format.bytes;
    let Some(row_bytes) = region.range[0].checked_mul(element_bytes) else {
        return INVALID_ARGUMENT;
    };
    let chunk_rows = (TILED_TRANSFER_CHUNK_BYTES / row_bytes).max(1);
    let Ok(base) = usize::try_from(snapshot.data) else {
        return INVALID_ALLOCATION;
    };
    let mut mapper = match tiled_mapper(snapshot) {
        Ok(mapper) => mapper,
        Err(status) => return status,
    };
    for z in 0..region.range[2] {
        for first_row in (0..region.range[1]).step_by(chunk_rows) {
            let rows = (region.range[1] - first_row).min(chunk_rows);
            let Some(chunk_bytes) = row_bytes.checked_mul(rows) else {
                return INVALID_ARGUMENT;
            };
            let mut chunk = match zeroed_bytes(chunk_bytes) {
                Ok(bytes) => bytes,
                Err(status) => return status,
            };
            let entries =
                match tiled_chunk_addresses(&mut mapper, snapshot, region, z, first_row, rows) {
                    Ok(entries) => entries,
                    Err(status) => return status,
                };
            let mut run = match zeroed_bytes(chunk_bytes) {
                Ok(bytes) => bytes,
                Err(status) => return status,
            };
            let mut first = 0;
            while first < entries.len() {
                let end = next_tiled_run(&entries, first, element_bytes);
                let bytes = (end - first) * element_bytes;
                let Ok(offset) = usize::try_from(entries[first].0) else {
                    return INVALID_ALLOCATION;
                };
                let Some(source) = base.checked_add(offset) else {
                    return INVALID_ALLOCATION;
                };
                // SAFETY: AddrLib mapped this contiguous texel span inside
                // the caller-owned image backing.
                let status = unsafe {
                    crate::memory::hsa_memory_copy(
                        run.as_mut_ptr().cast(),
                        source as *const c_void,
                        bytes,
                    )
                };
                if status != SUCCESS {
                    return status;
                }
                for (index, &(_, logical)) in entries[first..end].iter().enumerate() {
                    let source = index * element_bytes;
                    let target = logical * element_bytes;
                    chunk[target..target + element_bytes]
                        .copy_from_slice(&run[source..source + element_bytes]);
                }
                first = end;
            }
            for row in 0..rows {
                let Some(offset) = row_offset(z, first_row + row, row_pitch, slice_pitch) else {
                    return INVALID_ARGUMENT;
                };
                let Some(address) = destination.checked_add(offset) else {
                    return INVALID_ARGUMENT;
                };
                // SAFETY: The caller owns the external row and HSA validates
                // any registered GPU destination range.
                let status = unsafe {
                    crate::memory::hsa_memory_copy(
                        address as *mut c_void,
                        chunk[row * row_bytes..].as_ptr().cast(),
                        row_bytes,
                    )
                };
                if status != SUCCESS {
                    return status;
                }
            }
        }
    }
    SUCCESS
}

#[allow(unsafe_code)]
unsafe fn copy_image_rows(
    snapshot: &ImageSnapshot,
    external: usize,
    row_pitch: usize,
    slice_pitch: usize,
    region: Region,
    import: bool,
) -> Status {
    if snapshot.native.is_some() {
        // SAFETY: The external range and image backing are caller-owned for
        // the synchronous operation.
        return unsafe {
            if import {
                write_tiled_rows(
                    snapshot,
                    region,
                    TiledSource::Rows {
                        address: external,
                        row_pitch,
                        slice_pitch,
                    },
                )
            } else {
                read_tiled_rows(snapshot, region, external, row_pitch, slice_pitch)
            }
        };
    }
    let Some(row_bytes) = region.range[0].checked_mul(snapshot.format.bytes) else {
        return INVALID_ARGUMENT;
    };
    for z in 0..region.range[2] {
        for y in 0..region.range[1] {
            let Some(image_address) = snapshot.region_row_address(
                region.offset[0],
                region.offset[1] + y,
                region.offset[2] + z,
            ) else {
                return INVALID_ARGUMENT;
            };
            let Some(external_offset) = row_offset(z, y, row_pitch, slice_pitch) else {
                return INVALID_ARGUMENT;
            };
            let Some(external_address) = external.checked_add(external_offset) else {
                return INVALID_ARGUMENT;
            };
            // SAFETY: The HSA caller owns both memory ranges and the image
            // backing. hsa_memory_copy validates registered GPU ranges.
            let status = if import {
                unsafe {
                    crate::memory::hsa_memory_copy(
                        image_address as *mut c_void,
                        external_address as *const c_void,
                        row_bytes,
                    )
                }
            } else {
                unsafe {
                    crate::memory::hsa_memory_copy(
                        external_address as *mut c_void,
                        image_address as *const c_void,
                        row_bytes,
                    )
                }
            };
            if status != SUCCESS {
                return status;
            }
        }
    }
    SUCCESS
}

#[allow(unsafe_code)]
unsafe fn transfer_image(
    agent: HsaAgent,
    image: HsaExtImage,
    external: *mut c_void,
    row_pitch: usize,
    slice_pitch: usize,
    region: *const HsaExtImageRegion,
    import: bool,
) -> Status {
    if external.is_null() || region.is_null() {
        return INVALID_ARGUMENT;
    }
    let (snapshot, _inflight) = match image_snapshot(agent, image) {
        Ok(snapshot) => snapshot,
        Err(status) => return status,
    };
    // SAFETY: The caller supplies one initialized region record.
    let region = match snapshot.region(unsafe { *region }) {
        Ok(region) => region,
        Err(status) => return status,
    };
    let row_bytes = match region.range[0].checked_mul(snapshot.format.bytes) {
        Some(bytes) => bytes,
        None => return INVALID_ARGUMENT,
    };
    let (row_pitch, slice_pitch) =
        match external_pitches(row_bytes, region.range[1], row_pitch, slice_pitch) {
            Ok(pitches) => pitches,
            Err(status) => return status,
        };
    // SAFETY: The caller provides the external range for this synchronous
    // copy; the image snapshot retains the runtime while it is in flight.
    unsafe {
        copy_image_rows(
            &snapshot,
            external as usize,
            row_pitch,
            slice_pitch,
            region,
            import,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_copy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_copy(
    agent: HsaAgent,
    src_image: HsaExtImage,
    src_offset: *const HsaDim3,
    dst_image: HsaExtImage,
    dst_offset: *const HsaDim3,
    range: *const HsaDim3,
) -> Status {
    boundary(|| {
        if src_offset.is_null() || dst_offset.is_null() || range.is_null() {
            return INVALID_ARGUMENT;
        }
        let (source, _source_inflight) = match image_snapshot(agent, src_image) {
            Ok(s) => s,
            Err(status) => return status,
        };
        let (destination, _destination_inflight) = match image_snapshot(agent, dst_image) {
            Ok(s) => s,
            Err(status) => return status,
        };
        if source.descriptor.format != destination.descriptor.format {
            return IMAGE_FORMAT_UNSUPPORTED;
        }
        let src_region = HsaExtImageRegion {
            offset: unsafe { *src_offset },
            range: unsafe { *range },
        };
        let dst_region = HsaExtImageRegion {
            offset: unsafe { *dst_offset },
            range: unsafe { *range },
        };
        let src_region = match source.region(src_region) {
            Ok(region) => region,
            Err(status) => return status,
        };
        let dst_region = match destination.region(dst_region) {
            Ok(region) => region,
            Err(status) => return status,
        };
        let row_bytes = match src_region.range[0].checked_mul(source.format.bytes) {
            Some(bytes) => bytes,
            None => return INVALID_ARGUMENT,
        };
        let row_count = match src_region.range[1].checked_mul(src_region.range[2]) {
            Some(rows) => rows,
            None => return INVALID_ARGUMENT,
        };
        let size = match row_bytes.checked_mul(row_count) {
            Some(size) => size,
            None => return INVALID_ARGUMENT,
        };
        let mut staging = Vec::<u8>::new();
        if staging.try_reserve_exact(size).is_err() {
            return OUT_OF_RESOURCES;
        }
        staging.resize(size, 0);
        let Some(slice_pitch) = row_bytes.checked_mul(src_region.range[1]) else {
            return INVALID_ARGUMENT;
        };
        // SAFETY: Staging owns the entire tightly packed source region until
        // both synchronous image copies finish.
        let status = unsafe {
            copy_image_rows(
                &source,
                staging.as_mut_ptr() as usize,
                row_bytes,
                slice_pitch,
                src_region,
                false,
            )
        };
        if status != SUCCESS {
            return status;
        }
        // SAFETY: The same staging buffer remains live for the destination
        // copy, including when the two images share backing.
        unsafe {
            copy_image_rows(
                &destination,
                staging.as_mut_ptr() as usize,
                row_bytes,
                slice_pitch,
                dst_region,
                true,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_import")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_import(
    agent: HsaAgent,
    src_memory: *const c_void,
    src_row_pitch: usize,
    src_slice_pitch: usize,
    dst_image: HsaExtImage,
    image_region: *const HsaExtImageRegion,
) -> Status {
    boundary(|| unsafe {
        transfer_image(
            agent,
            dst_image,
            src_memory.cast_mut(),
            src_row_pitch,
            src_slice_pitch,
            image_region,
            true,
        )
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_export")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_export(
    agent: HsaAgent,
    src_image: HsaExtImage,
    dst_memory: *mut c_void,
    dst_row_pitch: usize,
    dst_slice_pitch: usize,
    image_region: *const HsaExtImageRegion,
) -> Status {
    boundary(|| unsafe {
        transfer_image(
            agent,
            src_image,
            dst_memory,
            dst_row_pitch,
            dst_slice_pitch,
            image_region,
            false,
        )
    })
}

fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exponent = i32::try_from((bits >> 23) & 0xff).unwrap_or(0);
    let mantissa = bits & 0x007f_ffff;
    if exponent == 255 {
        return sign | if mantissa == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exponent = exponent - 127 + 15;
    if half_exponent >= 31 {
        return sign | 0x7c00;
    }
    if half_exponent <= 0 {
        if half_exponent < -10 {
            return sign;
        }
        let significand = mantissa | 0x0080_0000;
        let shift = (14 - half_exponent) as u32;
        let truncated = significand >> shift;
        let remainder = significand & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        let rounded = truncated
            + u32::from(remainder > halfway || (remainder == halfway && truncated & 1 != 0));
        return sign | rounded as u16;
    }
    let truncated = mantissa >> 13;
    let remainder = mantissa & 0x1fff;
    let rounded =
        truncated + u32::from(remainder > 0x1000 || (remainder == 0x1000 && truncated & 1 != 0));
    let mut exponent = half_exponent as u32;
    let mantissa = if rounded == 0x400 {
        exponent += 1;
        0
    } else {
        rounded
    };
    if exponent >= 31 {
        sign | 0x7c00
    } else {
        sign | ((exponent as u16) << 10) | mantissa as u16
    }
}

#[allow(unsafe_code)]
unsafe fn clear_pixel(snapshot: &ImageSnapshot, data: *const c_void) -> Result<[u8; 16], Status> {
    let channels = match snapshot.descriptor.format.channel_order {
        0 | 1 => 1,
        3 | 5 => 2,
        8..=10 => 4,
        _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
    };
    let component_bytes = snapshot.format.bytes / channels;
    let mut pixel = [0_u8; 16];
    for index in 0..channels {
        let channel_type = snapshot.descriptor.format.channel_type;
        let encoded = if matches!(channel_type, 0..=3 | 14 | 15) {
            // SAFETY: The HSA clear pattern contains one f32 access value per
            // component for normalized and floating formats.
            let value = unsafe { data.cast::<f32>().add(index).read_unaligned() };
            match channel_type {
                0 => u32::from(((value.clamp(-1.0, 1.0) * 127.0).round_ties_even() as i8) as u8),
                1 => {
                    u32::from(((value.clamp(-1.0, 1.0) * 32767.0).round_ties_even() as i16) as u16)
                }
                2 => (value.clamp(0.0, 1.0) * 255.0).round_ties_even() as u32,
                3 => (value.clamp(0.0, 1.0) * 65535.0).round_ties_even() as u32,
                14 => u32::from(f32_to_f16(value)),
                15 => value.to_bits(),
                _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
            }
        } else if matches!(channel_type, 8..=10) {
            // SAFETY: Signed image access values are 32-bit integers.
            let value = unsafe { data.cast::<i32>().add(index).read_unaligned() };
            match channel_type {
                8 => u32::from(value.clamp(i32::from(i8::MIN), i32::from(i8::MAX)) as i8 as u8),
                9 => u32::from(value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16 as u16),
                10 => value as u32,
                _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
            }
        } else {
            // SAFETY: Unsigned image access values are 32-bit integers.
            let value = unsafe { data.cast::<u32>().add(index).read_unaligned() };
            match channel_type {
                11 => value.min(u32::from(u8::MAX)),
                12 => value.min(u32::from(u16::MAX)),
                13 => value,
                _ => return Err(IMAGE_FORMAT_UNSUPPORTED),
            }
        };
        let start = index * component_bytes;
        pixel[start..start + component_bytes]
            .copy_from_slice(&encoded.to_le_bytes()[..component_bytes]);
    }
    Ok(pixel)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_image_clear")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_image_clear(
    agent: HsaAgent,
    image: HsaExtImage,
    data: *const c_void,
    image_region: *const HsaExtImageRegion,
) -> Status {
    boundary(|| {
        if data.is_null() || image_region.is_null() {
            return INVALID_ARGUMENT;
        }
        let (snapshot, _inflight) = match image_snapshot(agent, image) {
            Ok(s) => s,
            Err(status) => return status,
        };
        let region = match snapshot.region(unsafe { *image_region }) {
            Ok(region) => region,
            Err(status) => return status,
        };
        let pixel = match unsafe { clear_pixel(&snapshot, data) } {
            Ok(pixel) => pixel,
            Err(status) => return status,
        };
        let row_bytes = match region.range[0].checked_mul(snapshot.format.bytes) {
            Some(bytes) => bytes,
            None => return INVALID_ARGUMENT,
        };
        if snapshot.native.is_some() {
            // SAFETY: The caller owns the image backing during this
            // synchronous clear; the clear pattern was copied above.
            return unsafe {
                write_tiled_rows(
                    &snapshot,
                    region,
                    TiledSource::Pixel(&pixel[..snapshot.format.bytes]),
                )
            };
        }
        let mut row = Vec::new();
        if row.try_reserve_exact(row_bytes).is_err() {
            return OUT_OF_RESOURCES;
        }
        row.resize(row_bytes, 0);
        for element in row.chunks_exact_mut(snapshot.format.bytes) {
            element.copy_from_slice(&pixel[..snapshot.format.bytes]);
        }
        for z in 0..region.range[2] {
            for y in 0..region.range[1] {
                let Some(address) = snapshot.region_row_address(
                    region.offset[0],
                    region.offset[1] + y,
                    region.offset[2] + z,
                ) else {
                    return INVALID_ARGUMENT;
                };
                let status = unsafe {
                    crate::memory::hsa_memory_copy(
                        address as *mut c_void,
                        row.as_ptr().cast(),
                        row_bytes,
                    )
                };
                if status != SUCCESS {
                    return status;
                }
            }
        }
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_image_get_info_max_dim")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_amd_image_get_info_max_dim(
    agent: HsaAgent,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    if !(EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS..=EXT_AGENT_INFO_IMAGE_ARRAY_MAX_LAYERS)
        .contains(&attribute)
    {
        return INVALID_ARGUMENT;
    }
    // SAFETY: The HSA caller provides storage of the type associated with
    // this image limit; hsa_agent_get_info validates the agent and output.
    unsafe { crate::hsa_agent_get_info(agent, attribute, value) }
}

/// Function table in the exact order of `hsa_ext_images_1_pfn_t`.
#[cfg(any(test, not(target_arch = "x86_64")))]
#[repr(C)]
pub(crate) struct ImageExtensionTable {
    get_capability:
        unsafe extern "C" fn(HsaAgent, u32, *const HsaExtImageFormat, *mut u32) -> Status,
    data_get_info: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtImageDescriptor,
        u32,
        *mut HsaExtImageDataInfo,
    ) -> Status,
    create: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtImageDescriptor,
        *const c_void,
        u32,
        *mut HsaExtImage,
    ) -> Status,
    destroy: extern "C" fn(HsaAgent, HsaExtImage) -> Status,
    copy: unsafe extern "C" fn(
        HsaAgent,
        HsaExtImage,
        *const HsaDim3,
        HsaExtImage,
        *const HsaDim3,
        *const HsaDim3,
    ) -> Status,
    import: unsafe extern "C" fn(
        HsaAgent,
        *const c_void,
        usize,
        usize,
        HsaExtImage,
        *const HsaExtImageRegion,
    ) -> Status,
    export: unsafe extern "C" fn(
        HsaAgent,
        HsaExtImage,
        *mut c_void,
        usize,
        usize,
        *const HsaExtImageRegion,
    ) -> Status,
    clear: unsafe extern "C" fn(
        HsaAgent,
        HsaExtImage,
        *const c_void,
        *const HsaExtImageRegion,
    ) -> Status,
    sampler_create: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtSamplerDescriptor,
        *mut HsaExtSampler,
    ) -> Status,
    sampler_destroy: extern "C" fn(HsaAgent, HsaExtSampler) -> Status,
    get_capability_with_layout:
        unsafe extern "C" fn(HsaAgent, u32, *const HsaExtImageFormat, u32, *mut u32) -> Status,
    data_get_info_with_layout: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtImageDescriptor,
        u32,
        u32,
        usize,
        usize,
        *mut HsaExtImageDataInfo,
    ) -> Status,
    create_with_layout: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtImageDescriptor,
        *const c_void,
        u32,
        u32,
        usize,
        usize,
        *mut HsaExtImage,
    ) -> Status,
    sampler_create_v2: unsafe extern "C" fn(
        HsaAgent,
        *const HsaExtSamplerDescriptorV2,
        *mut HsaExtSampler,
    ) -> Status,
}

#[cfg(any(test, not(target_arch = "x86_64")))]
const _: () =
    assert!(std::mem::size_of::<ImageExtensionTable>() == 14 * std::mem::size_of::<usize>());

#[cfg(any(test, not(target_arch = "x86_64")))]
pub(crate) fn image_extension_table() -> ImageExtensionTable {
    ImageExtensionTable {
        get_capability: hsa_ext_image_get_capability,
        data_get_info: hsa_ext_image_data_get_info,
        create: hsa_ext_image_create,
        destroy: hsa_ext_image_destroy,
        copy: hsa_ext_image_copy,
        import: hsa_ext_image_import,
        export: hsa_ext_image_export,
        clear: hsa_ext_image_clear,
        sampler_create: hsa_ext_sampler_create,
        sampler_destroy: hsa_ext_sampler_destroy,
        get_capability_with_layout: hsa_ext_image_get_capability_with_layout,
        data_get_info_with_layout: hsa_ext_image_data_get_info_with_layout,
        create_with_layout: hsa_ext_image_create_with_layout,
        sampler_create_v2: hsa_ext_sampler_create_v2,
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_sampler_create(
    agent: HsaAgent,
    sampler_descriptor: *const HsaExtSamplerDescriptor,
    sampler: *mut HsaExtSampler,
) -> Status {
    boundary(|| {
        if agent.handle == 0 {
            return INVALID_AGENT;
        }
        if sampler_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The HSA caller supplies one initialized descriptor.
        let descriptor = unsafe { *sampler_descriptor };
        // SAFETY: create_sampler checks output before writing it.
        unsafe {
            create_sampler(
                agent,
                HsaExtSamplerDescriptorV2 {
                    coordinate_mode: descriptor.coordinate_mode,
                    filter_mode: descriptor.filter_mode,
                    mipmap_filter_mode: 2,
                    address_modes: [descriptor.address_mode; 3],
                },
                sampler,
            )
        }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_create_v2")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
#[allow(unsafe_code)]
pub unsafe extern "C" fn hsa_ext_sampler_create_v2(
    agent: HsaAgent,
    sampler_descriptor: *const HsaExtSamplerDescriptorV2,
    sampler: *mut HsaExtSampler,
) -> Status {
    boundary(|| {
        if agent.handle == 0 {
            return INVALID_AGENT;
        }
        if sampler_descriptor.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The HSA caller supplies one initialized descriptor.
        let descriptor = unsafe { *sampler_descriptor };
        // SAFETY: create_sampler checks output before writing it.
        unsafe { create_sampler(agent, descriptor, sampler) }
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_ext_sampler_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
pub extern "C" fn hsa_ext_sampler_destroy(agent: HsaAgent, sampler: HsaExtSampler) -> Status {
    boundary(|| {
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
        if runtime.gpu_index(agent).is_none() {
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
    })
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
        assert_eq!(
            unsafe {
                hsa_ext_image_get_capability(agent, 0, std::ptr::null(), &raw mut capability)
            },
            INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe {
                hsa_ext_image_create(agent, std::ptr::null(), std::ptr::null(), 0, &raw mut image)
            },
            INVALID_ARGUMENT
        );
        assert_eq!(
            unsafe { hsa_ext_sampler_create(agent, std::ptr::null(), &raw mut sampler) },
            INVALID_AGENT
        );
        assert_eq!(hsa_ext_image_destroy(agent, image), NOT_INITIALIZED);
        assert_eq!(hsa_ext_sampler_destroy(agent, sampler), INVALID_AGENT);
        assert_eq!(capability, 0xfeed);
        assert_eq!(image.handle, 0xfeed);
        assert_eq!(sampler.handle, 0xfeed);
    }
}
