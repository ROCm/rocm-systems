// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA image operations over retained descriptors and caller-owned pixel storage.
//!
//! Public C adapters call this subsystem; only this subsystem owns its handle
//! records. Native descriptor encoders and allocations remain in rocddi.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};

use rocaddr3::{ResourceType, SurfaceLayout, SurfaceRequest};
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};

use crate::admission::InFlightToken;
use crate::ffi::*;
use crate::lifecycle::{initialized_mut, lock};
use crate::session::HsaSession;
use crate::status::map_error;
use descriptor::{Format, Generation, Layout};

mod color;
mod create;
pub(crate) mod descriptor;
mod layout;
mod sampler;
mod state;
mod transfer;

use layout::{NativeMipLayout, image_layout, maximum_mip_levels, native_mip_layout};
use state::{
    ImageRecord, ImageSnapshot, NativeImage, Region, SamplerRecord, image_gpu_index, image_snapshot,
};

pub(crate) use create::{
    ImageRequest, create_image, create_imported_image, create_native_mip_image, destroy_image,
    image_mipmap_array_get_level,
};
pub(crate) use layout::{get_info, image_capability, native_mip_data_info};
pub(crate) use sampler::{create_sampler, destroy_sampler};
pub(crate) use state::ImageState;
pub(crate) use transfer::{clear_image, copy_images, transfer_image};

/// Image layout selector for native tiled storage.
pub(crate) const IMAGE_LAYOUT_OPAQUE: u32 = 0;
/// Image layout selector for linear storage.
pub(crate) const IMAGE_LAYOUT_LINEAR: u32 = 1;
/// Capability bit for read-only image access.
const IMAGE_CAPABILITY_READ_ONLY: u32 = 1;
/// Capability bit for write-only image access.
const IMAGE_CAPABILITY_WRITE_ONLY: u32 = 2;
/// Capability bit for read/write image access.
const IMAGE_CAPABILITY_READ_WRITE: u32 = 4;
/// Capability bit for a stable image data layout.
const IMAGE_CAPABILITY_INVARIANT_LAYOUT: u32 = 16;
