// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Image handle records and leases retained by image operations.

use super::*;

/// One GPU-visible image descriptor. HSA owns the descriptor and the caller
/// retains ownership of the image data for the handle's lifetime.
pub(super) struct ImageRecord {
    /// Register generation used by this image resource.
    pub(super) architecture: Generation,
    /// GPU-visible resource descriptor backing owned by the HSA handle.
    pub(super) allocation: Allocation,
    /// Image dimensions and format supplied by the caller.
    pub(super) descriptor: HsaExtImageDescriptor,
    /// Hardware format and channel selector encoding.
    pub(super) format: Format,
    /// Checked pitches and byte extent.
    pub(super) layout: Layout,
    /// Caller-owned GPU address of image data.
    pub(super) data: u64,
    /// Resource descriptor words written into `allocation`.
    pub(super) words: [u32; 12],
    /// Number of levels in the image's mip chain.
    pub(super) mip_levels: u32,
    /// Whether the data uses a linear hardware layout.
    pub(super) linear: bool,
    /// `AddrLib` surface details for non-linear images.
    pub(super) native: Option<NativeImage>,
}

/// `AddrLib` request and retained surface layout for a tiled image.
#[derive(Clone)]
pub(super) struct NativeImage {
    /// Dimensions and GPU address configuration used for layout.
    pub(super) request: SurfaceRequest,
    /// Calculated mip offsets, strides, and total size.
    pub(super) layout: Arc<SurfaceLayout>,
    /// Mip level represented by this HSA image view.
    pub(super) level: u32,
}

/// GPU-visible backing for one sampler descriptor. The public handle is its
/// GPU address, so the allocation must outlive every dispatch using it.
pub(super) struct SamplerRecord {
    /// Descriptor allocation retained while its GPU address is public.
    pub(super) allocation: Allocation,
}

/// HSA image and sampler handles around device-visible descriptors.
pub(crate) struct ImageState {
    /// Samplers keyed by agent and public sampler handle.
    pub(super) samplers: HashMap<(u64, u64), SamplerRecord>,
    /// Images keyed by agent and public image handle.
    pub(super) images: HashMap<(u64, u64), Arc<ImageRecord>>,
}

impl ImageState {
    /// Initializes empty image and sampler handle registries.
    pub(crate) fn new() -> Self {
        Self {
            samplers: HashMap::new(),
            images: HashMap::new(),
        }
    }

    /// Releases all frontend image and sampler records.
    pub(crate) fn clear(&mut self) {
        self.samplers.clear();
        self.images.clear();
    }
}
/// Validates a GPU agent and returns its image-capable runtime index.
pub(super) fn image_gpu_index(runtime: &HsaSession, agent: HsaAgent) -> Result<usize, Status> {
    let index = runtime.agents.gpu_index(agent).ok_or(INVALID_AGENT)?;
    let info = runtime.agents.gpus[index].info();
    if !descriptor::supported_gpu(&info) || Generation::from_gpu(&info).is_none() {
        return Err(NOT_SUPPORTED);
    }
    Ok(index)
}

/// Image facts copied while its HSA owner remains protected by an in-flight token.
#[derive(Clone)]
pub(super) struct ImageSnapshot {
    /// Agent whose address space contains the image.
    pub(super) agent: HsaAgent,
    /// Keeps the public SRD allocation live through private dispatch teardown.
    pub(super) record: Arc<ImageRecord>,
    /// Public image geometry and format.
    pub(super) descriptor: HsaExtImageDescriptor,
    /// Hardware format encoding.
    pub(super) format: Format,
    /// Checked image pitches and byte extent.
    pub(super) layout: Layout,
    /// GPU address of caller-owned image data.
    pub(super) data: u64,
    /// Retained tiled layout, when present.
    pub(super) native: Option<NativeImage>,
}

/// Copies image ownership and layout data for work outside the lock.
pub(super) fn image_snapshot(
    agent: HsaAgent,
    image: HsaExtImage,
) -> Result<(ImageSnapshot, InFlightToken), Status> {
    let guard = lock()?;
    let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
    if runtime.agents.gpu_index(agent).is_none() {
        return Err(INVALID_AGENT);
    }
    let record = runtime
        .image_state
        .images
        .get(&(agent.handle, image.handle))
        .ok_or(INVALID_ARGUMENT)?;
    let token = runtime.control.inflight.enter().ok_or(OUT_OF_RESOURCES)?;
    Ok((
        ImageSnapshot {
            agent,
            record: record.clone(),
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
    /// Computes a row address in the image backing, checking overflow.
    pub(super) fn row_address(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        usize::try_from(self.data)
            .ok()?
            .checked_add(z.checked_mul(self.layout.slice_pitch)?)?
            .checked_add(y.checked_mul(self.layout.row_pitch)?)?
            .checked_add(x.checked_mul(self.format.bytes)?)
    }

    /// Validates an image region against the image dimensions.
    pub(super) fn region(&self, region: HsaExtImageRegion) -> Result<Region, Status> {
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

    /// Computes a row address within a validated image region.
    pub(super) fn region_row_address(&self, x: usize, y: usize, z: usize) -> Option<usize> {
        if self.descriptor.geometry == 3 {
            self.row_address(x, 0, y)
        } else {
            self.row_address(x, y, z)
        }
    }
}

/// Validated three-dimensional image subregion.
#[derive(Clone, Copy)]
pub(super) struct Region {
    /// Starting element coordinates by dimension.
    pub(super) offset: [usize; 3],
    /// Element counts by dimension.
    pub(super) range: [usize; 3],
}
