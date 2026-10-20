// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Device allocation, relocation, and ownership of loaded AMD HSA images.

use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};

use rocddi::device::Device;
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};
use rocelf::{Gfx1250TrampolinePlan, LoadSegment, align_up};

use crate::{CodeObject, Error, Symbol, SymbolKind, SymbolLinkage, SymbolTableOrder, Target};

/// One contiguous loaded address interval reported to loader clients.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadedSegment {
    /// Byte offset into the source code-object storage.
    pub storage_offset: usize,
    /// Device address of the loaded interval.
    pub address: u64,
    /// Size of the loaded interval.
    pub size: usize,
}

/// Combines load segments into the address interval exposed by AMD loader APIs.
///
/// # Errors
/// Returns `InvalidObject` when the source extents cannot be represented.
pub fn loaded_segments(
    loads: &[LoadSegment],
    device_base: u64,
    virtual_base: u64,
) -> Result<Vec<LoadedSegment>, Error> {
    let first = loads
        .iter()
        .min_by_key(|segment| segment.virtual_address)
        .ok_or(Error::InvalidObject)?;
    let virtual_end = loads
        .iter()
        .try_fold(virtual_base, |end, segment| {
            segment
                .virtual_address
                .checked_add(segment.memory_size)
                .map(|segment_end| end.max(segment_end))
        })
        .ok_or(Error::InvalidObject)?;
    let size = usize::try_from(
        virtual_end
            .checked_sub(virtual_base)
            .ok_or(Error::InvalidObject)?,
    )
    .map_err(|_| Error::InvalidObject)?;
    Ok(vec![LoadedSegment {
        storage_offset: usize::try_from(first.file_offset).map_err(|_| Error::InvalidObject)?,
        address: device_base,
        size,
    }])
}

/// One relocated device image and all allocations required to keep it live.
///
/// Dropping this value releases the image and any gfx1250 entry trampolines.
pub struct LoadedImage {
    /// Keeps executable image backing alive.
    _allocation: Allocation,
    /// Keeps optional GFX1250 entry stubs alive.
    _trampoline: Option<Allocation>,
    /// CPU mapping used while constructing the loaded image.
    host_base: usize,
    /// GPU address of the image's first byte.
    device_base: u64,
    /// Logical loaded byte extent.
    size: u64,
    /// ELF virtual address corresponding to `device_base`.
    virtual_base: u64,
    /// Source bytes retained for symbol and segment queries.
    code_object: Arc<[u8]>,
    /// Address intervals exposed to loader clients.
    segments: Vec<LoadedSegment>,
}

impl LoadedImage {
    /// Returns the device address of the loaded image.
    #[must_use]
    pub fn device_base(&self) -> u64 {
        self.device_base
    }

    /// Returns the virtual ELF address mapped at `device_base`.
    #[must_use]
    pub fn virtual_base(&self) -> u64 {
        self.virtual_base
    }

    /// Returns the loaded image's logical byte size.
    #[must_use]
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Borrows the retained source code-object bytes.
    #[must_use]
    pub fn code_object(&self) -> &[u8] {
        &self.code_object
    }

    /// Returns the loaded intervals exposed to loader queries.
    #[must_use]
    pub fn segments(&self) -> &[LoadedSegment] {
        &self.segments
    }

    /// Returns whether a device address lies in this image's logical range.
    #[must_use]
    pub fn contains(&self, address: u64) -> bool {
        address
            .checked_sub(self.device_base)
            .is_some_and(|offset| offset < self.size)
    }

    /// Resolves a complete device range to its CPU mapping when it lies here.
    ///
    /// # Errors
    /// Returns `InvalidRange` if the first byte lies here but the requested
    /// extent exceeds the image or its CPU address cannot be represented.
    pub fn host_range(&self, address: u64, size: usize) -> Result<Option<usize>, Error> {
        let Some(offset) = address.checked_sub(self.device_base) else {
            return Ok(None);
        };
        if offset >= self.size {
            return Ok(None);
        }
        let offset = usize::try_from(offset).map_err(|_| Error::InvalidRange)?;
        let end = offset.checked_add(size).ok_or(Error::InvalidRange)?;
        if end > usize::try_from(self.size).map_err(|_| Error::InvalidRange)? {
            return Err(Error::InvalidRange);
        }
        self.host_base
            .checked_add(offset)
            .map(Some)
            .ok_or(Error::InvalidRange)
    }
}

fn install_gfx1250_trampolines(
    device: &Device,
    symbols: &[Symbol],
    image: &mut [u8],
    virtual_base: u64,
    device_base: u64,
    page: u64,
) -> Result<Option<Allocation>, Error> {
    let descriptors = symbols
        .iter()
        .filter(|symbol| {
            symbol.kind == SymbolKind::Kernel
                && symbol.is_definition
                && std::path::Path::new(&symbol.full_name)
                    .extension()
                    .is_some_and(|extension| extension.eq_ignore_ascii_case("kd"))
        })
        .map(|symbol| symbol.value);
    let plan = Gfx1250TrampolinePlan::new(image, virtual_base, descriptors)?;
    if plan.is_empty() {
        return Ok(None);
    }
    let size = align_up(
        u64::try_from(plan.required_bytes()).map_err(|_| Error::OutOfResources)?,
        page,
    )
    .ok_or(Error::OutOfResources)?;
    let host_size = usize::try_from(size)
        .ok()
        .filter(|size| isize::try_from(*size).is_ok())
        .ok_or(Error::OutOfResources)?;
    let allocation = device
        .allocate(
            MemoryKind::System,
            size,
            page,
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
        )
        .map_err(Error::Device)?;
    let info = allocation.info();
    let host = info.host_address.ok_or(Error::OutOfResources)?;
    // SAFETY: The live allocation owns host_size writable bytes beginning at host.
    let stub = unsafe { std::slice::from_raw_parts_mut(host as *mut u8, host_size) };
    stub.fill(0);
    plan.write(image, device_base, stub, info.device_address)?;
    Ok(Some(allocation))
}

/// Allocates and relocates one AMD HSA code object for an activated GPU.
///
/// The returned image owns its native allocations. Symbols become public only
/// when the frontend commits the image to an executable. The resolver supplies
/// previously defined external symbol addresses in the caller's scope.
///
/// # Errors
/// Returns code-object, target, symbol, allocation, or native driver failures.
pub fn load_image(
    device: &Device,
    code_object: Arc<[u8]>,
    memory_kind: MemoryKind,
    page: u64,
    resolve_external: impl Fn(&str) -> Option<u64>,
) -> Result<(LoadedImage, Vec<Symbol>), Error> {
    let target = Target::from(device.endpoint().gpu().ok_or(Error::InvalidTarget)?);
    let object = CodeObject::parse(&code_object)?;
    object.validate_target(target)?;
    let layout = object.layout();
    let plan = rocelf::LoadPlan::new(&code_object, layout, page)?;
    let virtual_base = plan.virtual_base();
    let virtual_end = plan.virtual_end();
    let mut symbols = object.symbols(SymbolTableOrder::DynamicFirst)?;
    if symbols.iter().any(|symbol| {
        !symbol.is_definition
            && symbol.linkage == SymbolLinkage::Program
            && resolve_external(&symbol.full_name).is_none()
    }) {
        return Err(Error::UndefinedSymbol);
    }
    let allocation_size =
        u64::try_from(plan.allocation_size()).map_err(|_| Error::InvalidObject)?;
    let allocation = device
        .allocate(
            memory_kind,
            allocation_size,
            page,
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
        )
        .map_err(Error::Device)?;
    let info = allocation.info();
    let host_base = info.host_address.ok_or(Error::OutOfResources)?;
    // SAFETY: The live, host-visible allocation spans the full page-aligned
    // PT_LOAD range and owns plan.allocation_size() writable bytes.
    let image =
        unsafe { std::slice::from_raw_parts_mut(host_base as *mut u8, plan.allocation_size()) };
    plan.write_amdgpu_image(info.device_address, image, &resolve_external)?;
    let trampoline = if (target.gfx_major, target.gfx_minor, target.gfx_stepping) == (12, 5, 0) {
        install_gfx1250_trampolines(
            device,
            &symbols,
            image,
            virtual_base,
            info.device_address,
            page,
        )?
    } else {
        None
    };
    // The destination may be write-combined. Publish segment data and
    // relocations before the frontend makes the object visible to submitters.
    fence(Ordering::SeqCst);

    symbols.retain_mut(|symbol| {
        if !symbol.is_definition || symbol.value < virtual_base || symbol.value >= virtual_end {
            return false;
        }
        let Some(address) = info.device_address.checked_add(symbol.value - virtual_base) else {
            return false;
        };
        symbol.address = address;
        true
    });
    let segments = loaded_segments(plan.segments(), info.device_address, virtual_base)?;
    let size = u64::try_from(segments[0].size).map_err(|_| Error::InvalidObject)?;
    Ok((
        LoadedImage {
            _allocation: allocation,
            _trampoline: trampoline,
            host_base,
            device_base: info.device_address,
            size,
            virtual_base,
            code_object,
            segments,
        },
        symbols,
    ))
}
