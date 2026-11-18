// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Image creation, view publication, and checked descriptor destruction.

use super::*;

/// Creates native mip image and records its ownership.
pub(crate) unsafe fn create_native_mip_image(
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
        architecture,
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
    let words = match descriptor::native_mip_descriptor(
        architecture,
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

/// Caller input copied before creating and publishing an image handle.
#[derive(Clone, Copy)]
pub(crate) struct ImageRequest {
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

impl ImageRequest {
    /// Collects copied HSA inputs before validating and publishing an image.
    pub(crate) fn new(
        descriptor: HsaExtImageDescriptor,
        data: *const c_void,
        permission: u32,
        layout: u32,
        row_pitch: usize,
        slice_pitch: usize,
        output: *mut HsaExtImage,
    ) -> Self {
        Self {
            descriptor,
            data,
            permission,
            layout,
            row_pitch,
            slice_pitch,
            output,
        }
    }
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
    /// `AddrLib` layout retained for a tiled image.
    native: Option<NativeImage>,
}

#[allow(unsafe_code)]
/// Publishes a created native image under a new HSA handle.
unsafe fn publish_image(
    runtime: &mut HsaSession,
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
    let index = match runtime.agents.gpu_index(agent) {
        Some(index) => index,
        None => return INVALID_AGENT,
    };
    let device = runtime.agents.gpus[index].device.clone();
    let page = match u64::try_from(runtime.agents.host_page_size) {
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
        Arc::new(ImageRecord {
            architecture: Generation::from_gpu(&runtime.agents.gpus[index].info())
                .unwrap_or(Generation::Gfx12),
            allocation,
            descriptor,
            format,
            layout: surface,
            data: data_address,
            words,
            mip_levels,
            linear,
            native,
        }),
    );
    // SAFETY: The caller provided output storage for one image handle.
    unsafe { output.write(HsaExtImage { handle }) };
    SUCCESS
}

#[allow(unsafe_code)]
/// Creates image and records its ownership.
pub(crate) unsafe fn create_image(agent: HsaAgent, request: ImageRequest) -> Status {
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
    let (index, format, surface) = match image_layout(
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
    let words = match descriptor::descriptor(
        Generation::from_gpu(&runtime.agents.gpus[index].info()).unwrap_or(Generation::Gfx12),
        descriptor,
        format,
        surface,
        data_address,
    ) {
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
pub(crate) unsafe fn create_imported_image(
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
    let Some(pci) = runtime.agents.gpus[index].endpoint.pci else {
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
    let words = match descriptor::imported_descriptor(
        Generation::from_gpu(&runtime.agents.gpus[index].info()).unwrap_or(Generation::Gfx12),
        descriptor,
        format,
        surface,
        data_address,
        srd,
    ) {
        Ok(words) => words,
        Err(status) => return status,
    };
    let words = if mip_levels > 1 {
        match descriptor::imported_mip_chain(
            Generation::from_gpu(&runtime.agents.gpus[index].info()).unwrap_or(Generation::Gfx12),
            words,
            descriptor,
            srd,
            mip_levels,
        ) {
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
                linear: descriptor.geometry == 5,
                native: None,
            },
            output,
        )
    }
}

/// Destroys image and releases its owned resources.
pub(crate) fn destroy_image(agent: HsaAgent, image: HsaExtImage) -> Status {
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
    let key = (agent.handle, image.handle);
    let Some(record) = runtime.image_state.images.get_mut(&key) else {
        return INVALID_ARGUMENT;
    };
    // An internal transfer can retain this SRD through native retirement.
    let Some(record) = Arc::get_mut(record) else {
        return RESOURCE_BUSY;
    };
    if let Err(error) = record.allocation.free() {
        return map_error(error);
    }
    runtime.image_state.images.remove(&key);
    SUCCESS
}

/// Validates a mip view and publishes its independently owned descriptor.
#[allow(unsafe_code)]
pub(crate) unsafe fn image_mipmap_array_get_level(
    agent: HsaAgent,
    mipmapped_array: *const HsaExtImage,
    mip_level: u32,
    image_descriptor: *const HsaExtImageDescriptorV2,
    level_image_out: *mut HsaExtImage,
) -> Status {
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
    if let Err(status) = image_gpu_index(runtime, agent) {
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
        if descriptor::access_capabilities(descriptor.geometry, descriptor.format) == 0 {
            return IMAGE_FORMAT_UNSUPPORTED;
        }
        match descriptor::format(descriptor.format) {
            Some(format) if format.bytes == parent.format.bytes => format,
            _ => return IMAGE_FORMAT_UNSUPPORTED,
        }
    } else {
        parent.format
    };
    let surface = match descriptor::layout(descriptor, format, 0, 0) {
        Ok(surface) => surface,
        Err(status) => return status,
    };
    let level =
        rocddi::device::gpu::image::base_mip_level(parent.architecture, &parent.words) + mip_level;
    let words = match descriptor::mip_level_view(
        parent.architecture,
        parent.words,
        level,
        descriptor,
        format,
    ) {
        Ok(words) => words,
        Err(status) => return status,
    };
    let data_address = parent.data;
    let native = parent.native.as_ref().map(|parent| NativeImage {
        request: parent.request,
        layout: Arc::clone(&parent.layout),
        level: parent.level + mip_level,
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
}
