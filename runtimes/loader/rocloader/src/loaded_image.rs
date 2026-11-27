// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Device allocation, relocation, and ownership of loaded AMD HSA images.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering, fence};

use crate::elf::{LoadSegment, align_up};
use crate::target::gfx1250::TrampolinePlan;
use rocddi::device::Device;
use rocddi::device::gpu::CopyFailure;
use rocddi::device::gpu::diagnostic::DiagnosticMemory;
use rocddi::memory::device::{Allocation, DeviceAccess, MemoryKind};

use crate::{CodeObject, Error, Symbol, SymbolKind, SymbolLinkage, SymbolTableOrder, Target};

/// Code-image placement alignment, independent of native allocation granularity.
pub(crate) const CODE_ALIGNMENT: u64 = 256;

const PUBLICATION_PENDING: u8 = 0;
const PUBLICATION_ACTIVE: u8 = 1;
const PUBLICATION_COMPLETE: u8 = 2;
const PUBLICATION_UNCERTAIN: u8 = 3;

/// Executable storage and work that makes its bytes visible to every GPU.
struct PendingImage {
    /// Retains the device and its copy service until publication retires.
    device: Device,
    /// Other GPUs that can consume this image in program scope.
    peers: Vec<Device>,
    /// Writable, executable GPU range.
    destination: u64,
    /// Logical loaded extent, independent of native allocation padding.
    size: u64,
    /// Relocated source for a borrowed destination; never aliases its host mapping.
    bytes: Option<Vec<u8>>,
    /// Owned executable backing survives all clones of the publication work.
    allocation: Option<Allocation>,
    /// The full-range cache operation also publishes optional entry stubs.
    trampoline: Option<Allocation>,
    /// Publication and native-retirement state.
    state: AtomicU8,
}

/// Retained code publication work for an owned or caller-owned destination.
///
/// Caller-owned destinations follow `load_image_at_address`'s unsafe lifetime
/// contract. Owned allocations remain retained by every clone of this work.
#[derive(Clone)]
pub struct ImagePublication(Arc<PendingImage>);

impl ImagePublication {
    /// Publishes staged bytes and invalidates code caches on every consuming GPU.
    /// A successful repeat is a no-op; a retired failure permits a retry.
    ///
    /// # Errors
    /// Returns device errors or `Busy` for concurrent or uncertain publication.
    pub fn publish(&self, cancel: &AtomicBool) -> Result<(), Error> {
        publish_once(&self.0.state, || {
            let retired = |error| CopyFailure {
                error,
                operands_may_be_live: false,
            };
            let gpu = self.0.device.gpu().map_err(retired)?;
            if let Some(bytes) = &self.0.bytes {
                // SAFETY: load_image_at_address's caller retains the destination
                // through every publication owner and any uncertain native use.
                // copy_from_host owns native staging even on ambiguous failure.
                unsafe { gpu.copy_from_host(self.0.destination, bytes, cancel) }?;
            }
            // Ordinary images and entry stubs were written before construction
            // completed. Cache invalidation must follow all writes, including a
            // caller-owned image's GPU copy, before the executable is frozen.
            gpu.invalidate_code_cache(self.0.destination, self.0.size, cancel)?;
            for peer in &self.0.peers {
                peer.gpu().map_err(retired)?.invalidate_code_cache(
                    self.0.destination,
                    self.0.size,
                    cancel,
                )?;
            }
            Ok(())
        })
    }

    /// Whether no native operation can still access the destination.
    #[must_use]
    pub fn can_release(&self) -> bool {
        matches!(
            self.0.state.load(Ordering::Acquire),
            PUBLICATION_PENDING | PUBLICATION_COMPLETE
        )
    }

    /// Whether bytes and all consuming GPUs' code caches have been published.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.0.state.load(Ordering::Acquire) == PUBLICATION_COMPLETE
    }
}

/// Serializes publication and preserves uncertain native ownership on failure.
fn publish_once(
    state: &AtomicU8,
    work: impl FnOnce() -> Result<(), CopyFailure>,
) -> Result<(), Error> {
    match state.compare_exchange(
        PUBLICATION_PENDING,
        PUBLICATION_ACTIVE,
        Ordering::AcqRel,
        Ordering::Acquire,
    ) {
        Ok(_) => {}
        Err(PUBLICATION_COMPLETE) => return Ok(()),
        Err(_) => return Err(Error::Busy),
    }
    match work() {
        Ok(()) => {
            state.store(PUBLICATION_COMPLETE, Ordering::Release);
            Ok(())
        }
        Err(failure) => {
            state.store(
                if failure.operands_may_be_live {
                    PUBLICATION_UNCERTAIN
                } else {
                    PUBLICATION_PENDING
                },
                Ordering::Release,
            );
            Err(Error::Device(failure.error))
        }
    }
}

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
fn loaded_segments(
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
    /// Retains executable backing and publication for all consuming GPUs.
    publication: Option<ImagePublication>,
    /// CPU mapping used while constructing the loaded image.
    host_base: Option<usize>,
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
    /// Borrows executable allocations and the retained source ELF for a fatal
    /// diagnostic. Source bytes use their own host address, as advertised by
    /// memory-backed loader URIs; they never substitute for device contents.
    ///
    /// # Errors
    /// Returns a native lifetime or invalid-range failure.
    #[allow(unsafe_code)]
    pub fn diagnostic_memories(&self) -> Result<Vec<DiagnosticMemory<'_>>, rocddi::Error> {
        let publication = self.publication.as_ref().map(|work| &*work.0);
        let mut memory = Vec::with_capacity(3);
        memory.push(
            if let Some(allocation) = publication.and_then(|work| work.allocation.as_ref()) {
                allocation.diagnostic_memory()?
            } else {
                // SAFETY: Borrowed image constructors require callers to retain
                // destination storage through this owner's life.
                unsafe {
                    DiagnosticMemory::borrow(self, self.device_base, self.host_base, self.size)?
                }
            },
        );
        if let Some(trampoline) = publication.and_then(|work| work.trampoline.as_ref()) {
            memory.push(trampoline.diagnostic_memory()?);
        }
        let source = self.code_object();
        // SAFETY: This image retains the immutable source Arc. Its actual host
        // address is the address published in the memory-backed debugger URI.
        memory.push(unsafe {
            DiagnosticMemory::borrow(
                self,
                source.as_ptr() as u64,
                Some(source.as_ptr() as usize),
                source.len() as u64,
            )?
        });
        Ok(memory)
    }

    /// Clones pending publication work without exposing staging as destination memory.
    #[must_use]
    pub fn publication(&self) -> Option<ImagePublication> {
        self.publication.clone()
    }

    /// Whether native publication has retired and this owner may be released.
    #[must_use]
    pub fn can_release(&self) -> bool {
        self.publication
            .as_ref()
            .is_none_or(ImagePublication::can_release)
    }
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
        let Some(host_base) = self.host_base else {
            return Ok(None);
        };
        host_base
            .checked_add(offset)
            .map(Some)
            .ok_or(Error::InvalidRange)
    }

    /// Resolves a range for read-only loader inspection, including retained staging.
    /// This address must not be used as a writable alias of a caller-owned image.
    ///
    /// # Errors
    /// Returns `InvalidRange` if a range starts inside the image but exceeds it.
    pub fn query_host_range(&self, address: u64, size: usize) -> Result<Option<usize>, Error> {
        if let Some(bytes) = self
            .publication
            .as_ref()
            .and_then(|work| work.0.bytes.as_ref())
        {
            let Some(offset) = address.checked_sub(self.device_base) else {
                return Ok(None);
            };
            if offset >= self.size {
                return Ok(None);
            }
            let offset = usize::try_from(offset).map_err(|_| Error::InvalidRange)?;
            let end = offset.checked_add(size).ok_or(Error::InvalidRange)?;
            if end > bytes.len() {
                return Err(Error::InvalidRange);
            }
            return (bytes.as_ptr() as usize)
                .checked_add(offset)
                .map(Some)
                .ok_or(Error::InvalidRange);
        }
        self.host_range(address, size)
    }
}

fn install_gfx1250_trampolines(
    device: &Device,
    peers: &[&Device],
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
    let plan = TrampolinePlan::new(image, virtual_base, descriptors)?;
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
        .allocate_with_peers(
            peers,
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
/// The returned image owns its native allocations. Its publication work must
/// complete before dispatch. Symbols become public only
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
    load_image_with_peers(
        device,
        &[],
        code_object,
        memory_kind,
        page,
        resolve_external,
    )
}

/// Allocates a code image at one common address in the device and all peer VMs.
///
/// # Errors
/// Returns code-object, target, symbol, peer-mapping, or allocation failures.
pub fn load_image_with_peers(
    device: &Device,
    peers: &[&Device],
    code_object: Arc<[u8]>,
    memory_kind: MemoryKind,
    page: u64,
    resolve_external: impl Fn(&str) -> Option<u64>,
) -> Result<(LoadedImage, Vec<Symbol>), Error> {
    load_image_impl(
        device,
        peers,
        code_object,
        memory_kind,
        page,
        None,
        resolve_external,
    )
}

/// Stages a relocated code image for an application-owned GPU destination.
/// The destination is written only when its publication work is executed.
///
/// # Safety
/// `address..address + size` must remain mapped and writable/executable to
/// `device` until the image and all publication work are released. After an
/// uncertain publication failure it must remain live until conclusive native
/// teardown. No writable host mapping is required or accessed. The caller must
/// not dispatch from the destination before publication completes.
///
/// # Errors
/// Returns invalid range, object, target, symbol, or staging-allocation errors.
pub unsafe fn load_image_at_address(
    device: &Device,
    code_object: Arc<[u8]>,
    address: u64,
    size: usize,
    page: u64,
    resolve_external: impl Fn(&str) -> Option<u64>,
) -> Result<(LoadedImage, Vec<Symbol>), Error> {
    load_image_impl(
        device,
        &[],
        code_object,
        MemoryKind::System,
        page,
        Some((address, size)),
        resolve_external,
    )
}

fn validate_destination(address: u64, size: usize, load_size: usize) -> Result<(), Error> {
    if address == 0
        || address % CODE_ALIGNMENT != 0
        || load_size > size
        || address.checked_add(size as u64).is_none()
    {
        return Err(Error::InvalidRange);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn load_image_impl(
    device: &Device,
    peers: &[&Device],
    code_object: Arc<[u8]>,
    memory_kind: MemoryKind,
    page: u64,
    destination: Option<(u64, usize)>,
    resolve_external: impl Fn(&str) -> Option<u64>,
) -> Result<(LoadedImage, Vec<Symbol>), Error> {
    if page < CODE_ALIGNMENT || !page.is_power_of_two() {
        return Err(Error::InvalidRange);
    }
    let mut target = Target::from(device.endpoint().gpu().ok_or(Error::InvalidTarget)?);
    if target.gfx_major == 9 || (target.gfx_major, target.gfx_minor) == (10, 1) {
        target.xnack_enabled = Some(
            device
                .gpu()
                .map_err(Error::Device)?
                .xnack_enabled()
                .map_err(Error::Device)?,
        );
    }
    let object = CodeObject::parse(&code_object)?;
    object.validate_target(target)?;
    let plan = crate::elf::LoadPlan::new(&code_object, object.layout(), CODE_ALIGNMENT)?;
    let virtual_base = plan.virtual_base();
    let virtual_end = plan.virtual_end();
    let load_size = object.load_size()?;
    if let Some((address, size)) = destination {
        validate_destination(address, size, load_size)?;
    }
    let mut symbols = object.symbols(SymbolTableOrder::DynamicFirst)?;
    if symbols.iter().any(|symbol| {
        !symbol.is_definition
            && symbol.linkage == SymbolLinkage::Program
            && resolve_external(&symbol.full_name).is_none()
    }) {
        return Err(Error::UndefinedSymbol);
    }
    // Always stage relocation in owned host bytes. This keeps the destination
    // contract identical for ordinary allocations and host-read-only mappings.
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(plan.allocation_size())
        .map_err(|_| Error::OutOfResources)?;
    bytes.resize(plan.allocation_size(), 0);
    let allocation = if destination.is_none() {
        let size = align_up(plan.allocation_size() as u64, page).ok_or(Error::OutOfResources)?;
        Some(
            device
                .allocate_with_peers(
                    peers,
                    memory_kind,
                    size,
                    page,
                    DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
                )
                .map_err(Error::Device)?,
        )
    } else {
        None
    };
    let device_base = destination
        .map(|(address, _)| address)
        .or_else(|| {
            allocation
                .as_ref()
                .map(|allocation| allocation.info().device_address)
        })
        .ok_or(Error::InvalidRange)?;
    plan.write_amdgpu_image(device_base, &mut bytes, &resolve_external)?;
    let trampoline = if (target.gfx_major, target.gfx_minor, target.gfx_stepping) == (12, 5, 0) {
        install_gfx1250_trampolines(
            device,
            peers,
            &symbols,
            &mut bytes,
            virtual_base,
            device_base,
            page,
        )?
    } else {
        None
    };
    let (host_base, bytes) = if let Some(allocation) = &allocation {
        let host = allocation
            .info()
            .host_address
            .ok_or(Error::OutOfResources)?;
        // SAFETY: The newly allocated, unpublished backing owns at least the
        // complete staged extent and has a writable host mapping.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), host as *mut u8, bytes.len()) };
        fence(Ordering::SeqCst);
        (Some(host), None)
    } else {
        // Only logical image bytes belong to the caller's range; page padding
        // must never be written past the size returned by get_load_size.
        bytes.truncate(load_size);
        (None, Some(bytes))
    };
    let mut consuming_peers = Vec::new();
    consuming_peers
        .try_reserve_exact(peers.len())
        .map_err(|_| Error::OutOfResources)?;
    consuming_peers.extend(peers.iter().map(|peer| (*peer).clone()));
    let publication = Some(ImagePublication(Arc::new(PendingImage {
        device: device.clone(),
        peers: consuming_peers,
        destination: device_base,
        size: load_size as u64,
        bytes,
        allocation,
        trampoline,
        state: AtomicU8::new(PUBLICATION_PENDING),
    })));
    symbols.retain_mut(|symbol| {
        if !symbol.is_definition || symbol.value < virtual_base || symbol.value >= virtual_end {
            return false;
        }
        let Some(address) = device_base.checked_add(symbol.value - virtual_base) else {
            return false;
        };
        symbol.address = address;
        true
    });
    let segments = loaded_segments(plan.segments(), device_base, virtual_base)?;
    Ok((
        LoadedImage {
            publication,
            host_base,
            device_base,
            size: load_size as u64,
            virtual_base,
            code_object,
            segments,
        },
        symbols,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "linux")]
    fn diagnostic_source_keeps_its_uri_address_and_full_elf_contents()
    -> Result<(), Box<dyn std::error::Error>> {
        use rocddi::device::gpu::diagnostic::DiagnosticReader;

        let mut elf = crate::test_support::gfx1201_code_object();
        // This tail is not part of the device load image. A debugger still
        // needs the original ELF, including its nonloaded metadata.
        elf.extend_from_slice(b"source-only diagnostic sentinel");
        let device_bytes = [0xa5_u8; 128];
        let reader = DiagnosticReader::new()?;
        for host_visible in [true, false] {
            let source: Arc<[u8]> = elf.clone().into();
            let source_address = source.as_ptr() as u64;
            let weak_source = Arc::downgrade(&source);
            let image = LoadedImage {
                publication: None,
                host_base: host_visible.then_some(device_bytes.as_ptr() as usize),
                device_base: 0x1234_0000,
                size: device_bytes.len() as u64,
                virtual_base: 0,
                code_object: source,
                segments: Vec::new(),
            };
            assert_eq!(weak_source.strong_count(), 1);
            let regions = image.diagnostic_memories()?;
            assert_eq!(regions.len(), 2);
            assert_eq!(regions[0].address(), image.device_base());
            assert_eq!(regions[0].size(), device_bytes.len() as u64);
            let mut actual_device = [0; 128];
            if host_visible {
                reader.read(regions[0], 0, &mut actual_device)?;
                assert_eq!(actual_device, device_bytes);
            } else {
                assert!(regions[0].host_address().is_none());
                assert!(reader.read(regions[0], 0, &mut actual_device).is_err());
            }
            assert_eq!(regions[1].address(), source_address);
            assert_eq!(regions[1].size(), elf.len() as u64);
            let mut actual_source = vec![0; elf.len()];
            reader.read(regions[1], 0, &mut actual_source)?;
            assert_eq!(actual_source, elf.as_slice());
            drop(image);
            assert!(weak_source.upgrade().is_none());
        }
        Ok(())
    }

    #[test]
    #[cfg(target_os = "linux")]
    #[ignore = "requires a GFX1201 GPU, KFD, and a bound DRM render node"]
    fn diagnostic_regions_include_retained_native_trampoline()
    -> Result<(), Box<dyn std::error::Error>> {
        use rocddi::device::gpu::diagnostic::DiagnosticReader;
        use rocddi::memory::host::host_page_size;
        use rocddi::session::{DriverContextLifetime, Session};

        let mut session = Session::new(DriverContextLifetime::Process)?;
        let mut selected = None;
        session.enumerate(&mut |endpoint| {
            if endpoint
                .gpu()
                .is_some_and(|gpu| (gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping) == (12, 0, 1))
            {
                selected = Some(endpoint);
            }
            Ok(())
        })?;
        let endpoint = selected.ok_or_else(|| std::io::Error::other("GFX1201 unavailable"))?;
        let device = session.activate(&endpoint)?;
        let page = host_page_size()?;
        let access = DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE;
        let main = device.allocate(MemoryKind::System, page, page, access)?;
        let trampoline = device.allocate(MemoryKind::System, page, page, access)?;
        let main_info = main.info();
        let trampoline_info = trampoline.info();
        for (info, value) in [(main_info, 0xa5), (trampoline_info, 0x5a)] {
            let host = info
                .host_address
                .ok_or_else(|| std::io::Error::other("System allocation has no host alias"))?;
            // SAFETY: These live System allocations retain page writable bytes.
            // No GPU work is submitted; all access remains on this test thread.
            unsafe { std::ptr::write_bytes(host as *mut u8, value, usize::try_from(page)?) };
        }
        let source: Arc<[u8]> = crate::test_support::gfx1201_code_object().into();
        let image = LoadedImage {
            publication: Some(ImagePublication(Arc::new(PendingImage {
                device: device.clone(),
                peers: Vec::new(),
                destination: main_info.device_address,
                size: page,
                bytes: None,
                allocation: Some(main),
                trampoline: Some(trampoline),
                state: AtomicU8::new(PUBLICATION_PENDING),
            }))),
            host_base: main_info.host_address,
            device_base: main_info.device_address,
            size: page,
            virtual_base: 0,
            code_object: source.clone(),
            segments: Vec::new(),
        };
        let regions = image.diagnostic_memories()?;
        assert_eq!(regions.len(), 3);
        let reader = DiagnosticReader::new()?;
        for (region, info, value) in [
            (regions[0], main_info, 0xa5),
            (regions[1], trampoline_info, 0x5a),
        ] {
            assert_eq!(region.address(), info.device_address);
            assert_eq!(region.size(), page);
            let mut bytes = vec![0; usize::try_from(page)?];
            reader.read(region, 0, &mut bytes)?;
            assert!(bytes.iter().all(|byte| *byte == value));
        }
        assert_eq!(regions[2].address(), source.as_ptr() as u64);
        let mut bytes = vec![0; source.len()];
        reader.read(regions[2], 0, &mut bytes)?;
        assert_eq!(bytes, source.as_ref());
        drop(image);
        drop(device);
        session.destroy()?;
        Ok(())
    }

    fn copy_failure(operands_may_be_live: bool) -> CopyFailure {
        CopyFailure {
            error: rocddi::Error::Operation {
                kind: rocddi::ErrorKind::Busy,
                detail: "injected native publication failure",
            },
            operands_may_be_live,
        }
    }

    #[test]
    /// Checks reporting of one combined segment for a loaded object.
    fn reports_one_combined_segment_for_load_extent() {
        let loads = [
            LoadSegment {
                file_offset: 0,
                virtual_address: 0,
                file_size: 0x17c0,
                memory_size: 0x17c0,
            },
            LoadSegment {
                file_offset: 0x1800,
                virtual_address: 0x2800,
                file_size: 0x6c8,
                memory_size: 0x6c8,
            },
            LoadSegment {
                file_offset: 0x1ec8,
                virtual_address: 0x3ec8,
                file_size: 0x70,
                memory_size: 0x138,
            },
            LoadSegment {
                file_offset: 0x1f38,
                virtual_address: 0x4f38,
                file_size: 4,
                memory_size: 4,
            },
        ];
        let segments = loaded_segments(&loads, 0x0010_0000, 0);
        assert!(segments.is_ok());
        let Some(segments) = segments.ok() else {
            return;
        };
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].storage_offset, 0);
        assert_eq!(segments[0].address, 0x0010_0000);
        assert_eq!(segments[0].size, 0x4f3c);
    }

    #[test]
    fn retired_publication_retries_but_completed_work_does_not_repeat() {
        let state = AtomicU8::new(PUBLICATION_PENDING);
        assert!(matches!(
            publish_once(&state, || Err(copy_failure(false))),
            Err(Error::Device(_))
        ));
        assert_eq!(state.load(Ordering::Acquire), PUBLICATION_PENDING);
        let mut operations = 0;
        assert!(
            publish_once(&state, || {
                operations += 1;
                assert!(matches!(publish_once(&state, || Ok(())), Err(Error::Busy)));
                Ok(())
            })
            .is_ok()
        );
        assert_eq!(state.load(Ordering::Acquire), PUBLICATION_COMPLETE);
        assert!(
            publish_once(&state, || {
                operations += 1;
                Ok(())
            })
            .is_ok()
        );
        assert_eq!(operations, 1);
    }

    #[test]
    fn uncertain_publication_blocks_reuse_and_preserves_retention_state() {
        let state = AtomicU8::new(PUBLICATION_PENDING);
        assert!(matches!(
            publish_once(&state, || Err(copy_failure(true))),
            Err(Error::Device(_))
        ));
        assert_eq!(state.load(Ordering::Acquire), PUBLICATION_UNCERTAIN);
        let mut retried = false;
        assert!(matches!(
            publish_once(&state, || {
                retried = true;
                Ok(())
            }),
            Err(Error::Busy)
        ));
        assert!(!retried);
        assert_eq!(state.load(Ordering::Acquire), PUBLICATION_UNCERTAIN);
    }

    #[test]
    fn interrupted_publication_cannot_be_retried_as_retired_work() {
        let state = AtomicU8::new(PUBLICATION_PENDING);
        let result = std::panic::catch_unwind(|| {
            publish_once(&state, || std::panic::resume_unwind(Box::new(())))
        });
        assert!(result.is_err());
        assert_eq!(state.load(Ordering::Acquire), PUBLICATION_ACTIVE);
        let mut retried = false;
        assert!(matches!(
            publish_once(&state, || {
                retried = true;
                Ok(())
            }),
            Err(Error::Busy)
        ));
        assert!(!retried);
    }

    #[test]
    fn caller_destination_uses_logical_extent_without_page_padding() {
        assert!(validate_destination(0x100, 257, 257).is_ok());
        assert!(validate_destination(0x100, 256, 257).is_err());
        assert!(validate_destination(0x101, 257, 257).is_err());
        assert!(validate_destination(0, 257, 257).is_err());
        assert!(validate_destination(u64::MAX - 255, 257, 257).is_err());
    }
}
