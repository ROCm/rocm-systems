// SPDX-License-Identifier: MIT

//! Native rocddi contract checks for the first GPU target.
#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "the test pattern deliberately wraps each byte index modulo 256"
)]

use std::error::Error;
use std::fs::OpenOptions;
use std::io;
use std::os::fd::AsRawFd;
use std::os::unix::fs::FileExt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU16, AtomicU64, Ordering, fence};
use std::time::{Duration, Instant};

use rocddi::gpu::queue::{
    QueueAccessWidth, QueueParameters, QueuePriority, QueueProducerMode, QueueRequest,
    QueueRingMemory, SdmaEngineSelection, ring_doorbell,
};
use rocddi::gpu::{CopyRect, GpuCopySequence};
use rocddi::memory::interop::linux::{AisFileOperation, ais_transfer};
use rocddi::memory::{DeviceAccess, HostCachePolicy, MemoryKind};
use rocddi::session::{Session, SessionLifetime};

#[test]
#[ignore = "requires GFX1201, KFD AIS, and ROCDDI_CTS_AIS_DIR on P2P-capable storage"]
fn gfx1201_ais_vram_file_contract() -> Result<(), Box<dyn Error>> {
    const BYTES: usize = 4096;

    struct RemoveFile(PathBuf);
    impl Drop for RemoveFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    let directory = std::env::var_os("ROCDDI_CTS_AIS_DIR")
        .ok_or_else(|| io::Error::other("ROCDDI_CTS_AIS_DIR is required for AIS CTS"))?;
    let mut path = PathBuf::from(directory);
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_nanos();
    path.push(format!("rocddi-ais-{}-{nonce}", std::process::id()));
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)?;
    let _cleanup = RemoveFile(path);

    let mut expected = [0_u8; BYTES];
    for (index, byte) in expected.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(7).wrapping_add(3);
    }
    file.write_all_at(&expected, 0)?;
    file.write_all_at(&[0_u8; BYTES], BYTES as u64)?;
    file.sync_all()?;

    let mut session = Session::new(SessionLifetime::Process)?;
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
    let endpoint = selected.ok_or_else(|| io::Error::other("GFX1201 endpoint is unavailable"))?;
    let device = session.activate(&endpoint)?;
    let mut allocation = device.allocate(
        MemoryKind::DeviceLocal {
            host_visible: false,
            coherent: false,
            uncached: false,
            contiguous: false,
        },
        BYTES as u64,
        BYTES as u64,
        DeviceAccess::READ | DeviceAccess::WRITE,
    )?;
    assert_eq!(allocation.info().host_address, None);

    let read = ais_transfer(
        &allocation,
        file.as_raw_fd(),
        0,
        BYTES as u64,
        0,
        AisFileOperation::Read,
    )?;
    assert_eq!((read.size_copied, read.status), (BYTES as u64, 0));
    let write = ais_transfer(
        &allocation,
        file.as_raw_fd(),
        0,
        BYTES as u64,
        i64::try_from(BYTES)?,
        AisFileOperation::Write,
    )?;
    assert_eq!((write.size_copied, write.status), (BYTES as u64, 0));
    let mut actual = [0_u8; BYTES];
    file.read_exact_at(&mut actual, BYTES as u64)?;
    assert_eq!(actual, expected);

    allocation.free()?;
    drop(allocation);
    drop(device);
    session.destroy()?;
    Ok(())
}

#[test]
#[ignore = "requires a GFX1201 GPU, KFD, and a bound DRM render node"]
#[allow(
    clippy::too_many_lines,
    reason = "one native session checks linear, pitched, and virtual-memory copies"
)]
fn gfx1201_sdma_copy_contract() -> Result<(), Box<dyn Error>> {
    let mut session = Session::new(SessionLifetime::Process)?;
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
    let endpoint = selected.ok_or_else(|| io::Error::other("GFX1201 endpoint is unavailable"))?;
    let device = session.activate(&endpoint)?;
    let gpu = device.gpu()?;
    assert!(gpu.supports_linear_copy());
    gpu.preload_linear_copy().map_err(|failure| failure.error)?;
    gpu.preload_linear_copy().map_err(|failure| failure.error)?;
    let counters = gpu.clock_counters()?;
    assert!(counters.gpu_frequency > 0);

    let access = DeviceAccess::READ | DeviceAccess::WRITE;
    let mut source = device.allocate(MemoryKind::System, 4096, 4096, access)?;
    let mut destination = device.allocate(MemoryKind::System, 4096, 4096, access)?;
    let source_info = source.info();
    let destination_info = destination.info();
    let source_host = source_info
        .host_address
        .ok_or_else(|| io::Error::other("source has no host mapping"))?;
    let destination_host = destination_info
        .host_address
        .ok_or_else(|| io::Error::other("destination has no host mapping"))?;
    // SAFETY: Each live allocation provides a writable 4096-byte host mapping.
    let (source_bytes, destination_bytes) = unsafe {
        (
            std::slice::from_raw_parts_mut(source_host as *mut u8, 4096),
            std::slice::from_raw_parts_mut(destination_host as *mut u8, 4096),
        )
    };
    for (index, byte) in source_bytes.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(7).wrapping_add(3);
    }
    destination_bytes.fill(0xa5);
    let cancel = AtomicBool::new(false);
    // SAFETY: Both mapped allocations remain live until native retirement.
    if let Err(failure) = unsafe {
        gpu.copy_linear(
            destination_info.device_address,
            source_info.device_address,
            4096,
            &cancel,
        )
    } {
        if failure.operands_may_be_live {
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    assert_eq!(destination_bytes, source_bytes);

    source_bytes[..64].fill(0x5a);
    destination_bytes[..64].fill(0xa5);
    // SAFETY: The two mapped allocations remain live through this second
    // default-ring copy after the first operation returned its native context.
    if let Err(failure) = unsafe {
        gpu.copy_linear(
            destination_info.device_address,
            source_info.device_address,
            64,
            &cancel,
        )
    } {
        if failure.operands_may_be_live {
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    assert_eq!(&destination_bytes[..64], &source_bytes[..64]);

    let ring_mask = gpu.available_sdma_rings()?;
    assert_ne!(ring_mask & 1, 0);
    for ring in 0..u32::BITS {
        if ring_mask & (1_u32 << ring) == 0 {
            continue;
        }
        destination_bytes[..64].fill(0xa5);
        let mut sequence = match GpuCopySequence::begin_on_sdma_ring(gpu, &cancel, ring) {
            Ok(sequence) => sequence,
            Err(failure) => return Err(Box::new(failure.error)),
        };
        // SAFETY: Both system allocations remain GPU-mapped until this
        // selected-ring submission retires or its owners are retained.
        if let Err(failure) = unsafe {
            sequence.copy_linear(
                destination_info.device_address,
                source_info.device_address,
                64,
            )
        } {
            drop(sequence);
            if failure.operands_may_be_live {
                std::mem::forget(source);
                std::mem::forget(destination);
                std::mem::forget(device);
                std::mem::forget(session);
            }
            return Err(Box::new(failure.error));
        }
        drop(sequence);
        assert_eq!(&destination_bytes[..64], &source_bytes[..64]);
    }

    destination_bytes[..64].fill(0xa5);
    let before = gpu.clock_counters()?.gpu;
    let mut timed = GpuCopySequence::begin(gpu, &cancel).map_err(|failure| failure.error)?;
    timed.enable_timing().map_err(|failure| failure.error)?;
    // SAFETY: Both mapped allocations stay live until the timed copy retires.
    if let Err(failure) = unsafe {
        timed.copy_rect(CopyRect {
            destination: destination_info.device_address,
            source: source_info.device_address,
            width: 32,
            height: 2,
            depth: 1,
            destination_pitch: 32,
            source_pitch: 32,
            destination_slice: 64,
            source_slice: 64,
        })
    } {
        drop(timed);
        if failure.operands_may_be_live {
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    let ticks = timed.finish_timing().map_err(|failure| failure.error)?;
    drop(timed);
    let after = gpu.clock_counters()?.gpu;
    assert!(before <= ticks.start && ticks.start <= ticks.end && ticks.end <= after);
    assert_eq!(&destination_bytes[..64], &source_bytes[..64]);

    destination_bytes.fill(0xa5);
    let rect = CopyRect {
        destination: destination_info.device_address + 4,
        source: source_info.device_address + 4,
        width: 32,
        height: 3,
        depth: 2,
        destination_pitch: 64,
        source_pitch: 64,
        destination_slice: 256,
        source_slice: 256,
    };
    // SAFETY: Every selected row lies within the two live allocations.
    if let Err(failure) = unsafe { gpu.copy_rect(rect, &cancel) } {
        if failure.operands_may_be_live {
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    for layer in 0..2 {
        for row in 0..3 {
            let first = 4 + layer * 256 + row * 64;
            assert_eq!(
                &destination_bytes[first..first + 32],
                &source_bytes[first..first + 32]
            );
            assert!(
                destination_bytes[first + 32..first + 64]
                    .iter()
                    .all(|byte| *byte == 0xa5)
            );
        }
    }
    assert_eq!(destination_bytes[3], 0xa5);
    assert_eq!(destination_bytes[4 + 2 * 256 + 3 * 64], 0xa5);

    destination_bytes.fill(0xa5);
    let batch = [
        CopyRect::linear(
            destination_info.device_address,
            source_info.device_address,
            96,
        ),
        CopyRect::linear(
            destination_info.device_address + 512,
            source_info.device_address + 1024,
            128,
        ),
    ];
    // SAFETY: Both ranges are disjoint, mapped, and retained until retirement.
    if let Err(failure) = unsafe { gpu.copy_rects(&batch, &cancel) } {
        if failure.operands_may_be_live {
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    assert_eq!(&destination_bytes[..96], &source_bytes[..96]);
    assert_eq!(&destination_bytes[512..640], &source_bytes[1024..1152]);
    assert!(destination_bytes[96..512].iter().all(|byte| *byte == 0xa5));
    assert!(destination_bytes[640..].iter().all(|byte| *byte == 0xa5));

    destination_bytes.fill(0xa5);
    // SAFETY: The selected dword range stays mapped and writable until native
    // retirement; on uncertain retirement its owner is retained below.
    if let Err(failure) = unsafe {
        gpu.fill_u32(
            destination_info.device_address + 1024,
            0x1122_3344,
            64,
            &cancel,
        )
    } {
        if failure.operands_may_be_live {
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    assert!(destination_bytes[..1024].iter().all(|byte| *byte == 0xa5));
    assert!(
        destination_bytes[1024..1280]
            .chunks_exact(4)
            .all(|word| word == 0x1122_3344_u32.to_ne_bytes())
    );
    assert!(destination_bytes[1280..].iter().all(|byte| *byte == 0xa5));

    let mut staged_output = [0_u8; 256];
    // SAFETY: The selected GPU ranges remain mapped through retirement. The
    // host input is copied into rocddi staging before native submission, and
    // the output slice is written only after native retirement.
    let staged_copy = unsafe {
        gpu.copy_from_host(
            destination_info.device_address + 2048,
            &source_bytes[..256],
            &cancel,
        )
        .and_then(|()| {
            gpu.copy_to_host(
                &mut staged_output,
                destination_info.device_address + 2048,
                &cancel,
            )
        })
    };
    if let Err(failure) = staged_copy {
        if failure.operands_may_be_live {
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    assert_eq!(staged_output, source_bytes[..256]);
    assert_eq!(&destination_bytes[2048..2304], &source_bytes[..256]);

    let mut host_input = [0_u8; 4096];
    for (index, byte) in host_input.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(11).wrapping_add(9);
    }
    let mut host_output = [0xa5_u8; 4096];
    for size in [1, 63, 257, 4096] {
        source_bytes.fill(0x5a);
        destination_bytes.fill(0xa5);
        host_output.fill(0xa5);
        let mut sequence = match GpuCopySequence::begin(gpu, &cancel) {
            Ok(sequence) => sequence,
            Err(failure) => return Err(Box::new(failure.error)),
        };
        // SAFETY: The two mapped allocations retain their GPU addresses and
        // permissions through native retirement. The host slices remain live
        // until the sequence returns, and the sequence stages each host leg.
        let result = unsafe {
            sequence
                .copy_from_host(source_info.device_address, &host_input[..size])
                .and_then(|()| {
                    sequence.copy_linear(
                        destination_info.device_address,
                        source_info.device_address,
                        size as u64,
                    )
                })
                .and_then(|()| {
                    sequence.copy_to_host(&mut host_output[..size], destination_info.device_address)
                })
        };
        if let Err(failure) = result {
            drop(sequence);
            if failure.operands_may_be_live {
                std::mem::forget(source);
                std::mem::forget(destination);
                std::mem::forget(device);
                std::mem::forget(session);
            }
            return Err(Box::new(failure.error));
        }
        assert_eq!(&host_output[..size], &host_input[..size]);
        assert_eq!(&destination_bytes[..size], &host_input[..size]);
        assert!(destination_bytes[size..].iter().all(|byte| *byte == 0xa5));
        assert!(host_output[size..].iter().all(|byte| *byte == 0xa5));
    }

    let cancelled = AtomicBool::new(true);
    let mut sequence = match GpuCopySequence::begin(gpu, &cancelled) {
        Ok(sequence) => sequence,
        Err(failure) => return Err(Box::new(failure.error)),
    };
    // SAFETY: Both ranges remain mapped. Cancellation rejects the packet
    // before submission, so no operand retention is required.
    let result = unsafe {
        sequence.copy_linear(
            destination_info.device_address,
            source_info.device_address,
            64,
        )
    };
    let Err(failure) = result else {
        return Err(io::Error::other("cancelled sequence accepted a packet").into());
    };
    assert_eq!(failure.error.kind(), rocddi::ErrorKind::Busy);
    assert!(!failure.operands_may_be_live);
    cancelled.store(false, Ordering::Release);
    // SAFETY: Both ranges remain mapped; a failed sequence cannot submit
    // another packet even after cancellation is lifted.
    let result = unsafe {
        sequence.copy_linear(
            destination_info.device_address,
            source_info.device_address,
            64,
        )
    };
    let Err(failure) = result else {
        return Err(io::Error::other("failed sequence accepted another packet").into());
    };
    assert_eq!(failure.error.kind(), rocddi::ErrorKind::Busy);
    assert!(!failure.operands_may_be_live);
    drop(sequence);

    let mut private = device.allocate(
        MemoryKind::DeviceLocal {
            host_visible: false,
            coherent: false,
            uncached: false,
            contiguous: false,
        },
        4096,
        4096,
        access,
    )?;
    let private_info = private.info();
    assert_eq!(private_info.host_address, None);
    let mut uncached = device.allocate(
        MemoryKind::DeviceLocal {
            host_visible: false,
            coherent: true,
            uncached: true,
            contiguous: false,
        },
        4096,
        4096,
        access,
    )?;
    let uncached_info = uncached.info();
    assert_eq!(uncached_info.host_address, None);
    destination_bytes.fill(0xa5);
    // SAFETY: All allocations remain mapped through the native submissions.
    // Neither private VRAM address has a CPU mapping to fall back to.
    let private_copy = unsafe {
        gpu.copy_linear(
            private_info.device_address,
            source_info.device_address,
            4096,
            &cancel,
        )
        .and_then(|()| {
            gpu.copy_linear(
                uncached_info.device_address,
                private_info.device_address,
                4096,
                &cancel,
            )
        })
        .and_then(|()| {
            gpu.copy_linear(
                destination_info.device_address,
                uncached_info.device_address,
                4096,
                &cancel,
            )
        })
    };
    if let Err(failure) = private_copy {
        if failure.operands_may_be_live {
            std::mem::forget(private);
            std::mem::forget(uncached);
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    assert_eq!(destination_bytes, source_bytes);
    private.free()?;
    uncached.free()?;
    drop(private);
    drop(uncached);

    let mut virtual_memory =
        device.create_virtual_memory(MemoryKind::System, 4096, false, false)?;
    let mut reservation = session.reserve_virtual_address(&[&device], 4096, 4096, 0)?;
    let virtual_address = reservation.info().address;
    let mut mapping = device.map_virtual_memory(
        &virtual_memory,
        &reservation,
        virtual_address,
        0,
        4096,
        access,
    )?;
    destination_bytes.fill(0xa5);
    // SAFETY: The virtual mapping and both allocations stay live through each
    // retired submission. Both directions have the required GPU permissions.
    let virtual_copy = unsafe {
        gpu.copy_linear(virtual_address, source_info.device_address, 4096, &cancel)
            .and_then(|()| {
                gpu.copy_linear(
                    destination_info.device_address,
                    virtual_address,
                    4096,
                    &cancel,
                )
            })
    };
    if let Err(failure) = virtual_copy {
        if failure.operands_may_be_live {
            std::mem::forget(mapping);
            std::mem::forget(virtual_memory);
            std::mem::forget(reservation);
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    assert_eq!(destination_bytes, source_bytes);
    mapping.free()?;
    virtual_memory.free()?;
    reservation.free()?;
    drop(mapping);
    drop(virtual_memory);
    drop(reservation);

    let mut uncached_memory = device.create_virtual_memory(
        MemoryKind::DeviceLocal {
            host_visible: false,
            coherent: true,
            uncached: true,
            contiguous: false,
        },
        4096,
        false,
        true,
    )?;
    let mut uncached_reservation = session.reserve_virtual_address(&[&device], 4096, 4096, 0)?;
    let uncached_address = uncached_reservation.info().address;
    let mut uncached_mapping = device.map_virtual_memory(
        &uncached_memory,
        &uncached_reservation,
        uncached_address,
        0,
        4096,
        access,
    )?;
    destination_bytes.fill(0xa5);
    // SAFETY: The coherent, uncached VRAM mapping and both operands remain
    // live through the native copies or are retained on uncertain retirement.
    let uncached_copy = unsafe {
        gpu.copy_linear(uncached_address, source_info.device_address, 4096, &cancel)
            .and_then(|()| {
                gpu.copy_linear(
                    destination_info.device_address,
                    uncached_address,
                    4096,
                    &cancel,
                )
            })
    };
    if let Err(failure) = uncached_copy {
        if failure.operands_may_be_live {
            std::mem::forget(uncached_mapping);
            std::mem::forget(uncached_memory);
            std::mem::forget(uncached_reservation);
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
        }
        return Err(Box::new(failure.error));
    }
    assert_eq!(destination_bytes, source_bytes);
    uncached_mapping.free()?;
    uncached_memory.free()?;
    uncached_reservation.free()?;
    drop(uncached_mapping);
    drop(uncached_memory);
    drop(uncached_reservation);

    let mut uncached_host = device.allocate(
        MemoryKind::OwnedHost {
            cache: HostCachePolicy::Uncached,
        },
        4096,
        4096,
        access,
    )?;
    let uncached_host_info = uncached_host.info();
    let host_address = uncached_host_info
        .host_address
        .ok_or_else(|| io::Error::other("owned host allocation has no host mapping"))?;
    // SAFETY: The live allocation owns a writable 4096-byte host mapping.
    let host_bytes = unsafe { std::slice::from_raw_parts_mut(host_address as *mut u8, 4096) };
    host_bytes.fill(0x7d);
    destination_bytes.fill(0xa5);
    // SAFETY: All three allocations stay mapped through both retired copies.
    // Uncertain retirement retains their owners for process teardown.
    if let Err(failure) = unsafe {
        gpu.copy_linear(
            destination_info.device_address,
            uncached_host_info.device_address,
            4096,
            &cancel,
        )
        .and_then(|()| {
            gpu.copy_linear(
                uncached_host_info.device_address,
                source_info.device_address,
                4096,
                &cancel,
            )
        })
    } {
        std::mem::forget(uncached_host);
        std::mem::forget(source);
        std::mem::forget(destination);
        std::mem::forget(device);
        std::mem::forget(session);
        return Err(Box::new(failure.error));
    }
    let read_matches = destination_bytes.iter().all(|byte| *byte == 0x7d);
    let write_matches = host_bytes == source_bytes;
    uncached_host.free()?;
    assert!(read_matches);
    assert!(write_matches);
    drop(uncached_host);

    for cache in [
        HostCachePolicy::Coarse,
        HostCachePolicy::Fine,
        HostCachePolicy::Extended,
    ] {
        let mut host = device.allocate(MemoryKind::OwnedHost { cache }, 4096, 4096, access)?;
        let info = host.info();
        assert_eq!(info.host_address, Some(info.device_address as usize));
        host.free()?;
    }

    let mut host_pages = Box::new([0_u8; 8192]);
    let host_base = host_pages.as_mut_ptr() as usize;
    let host_address = (host_base + 4095) & !4095;
    let host_offset = host_address - host_base;
    host_pages[host_offset..host_offset + 4096].fill(0x3c);
    // SAFETY: The aligned 4096-byte page lies inside the boxed 8192-byte
    // extent. The box stays mapped until native deregistration succeeds.
    let mut registered = match unsafe {
        device.register_host(host_address, HostCachePolicy::Uncached, 4096, 4096, access)
    } {
        Ok(registered) => registered,
        Err(error) => {
            // Native acquisition may have retained the caller's page cover.
            std::mem::forget(host_pages);
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
            return Err(Box::new(error));
        }
    };
    let registered_info = registered.info();
    destination_bytes.fill(0xa5);
    // SAFETY: The registered page, source, and destination stay mapped until
    // both copies retire. Uncertain retirement retains their native owners
    // and the page.
    if let Err(failure) = unsafe {
        gpu.copy_linear(
            destination_info.device_address,
            registered_info.device_address,
            4096,
            &cancel,
        )
        .and_then(|()| {
            gpu.copy_linear(
                registered_info.device_address,
                source_info.device_address,
                4096,
                &cancel,
            )
        })
    } {
        std::mem::forget(registered);
        std::mem::forget(host_pages);
        std::mem::forget(source);
        std::mem::forget(destination);
        std::mem::forget(device);
        std::mem::forget(session);
        return Err(Box::new(failure.error));
    }
    let read_matches = destination_bytes.iter().all(|byte| *byte == 0x3c);
    let write_matches = host_pages[host_offset..host_offset + 4096] == *source_bytes;
    if let Err(error) = registered.free() {
        std::mem::forget(registered);
        std::mem::forget(host_pages);
        std::mem::forget(source);
        std::mem::forget(destination);
        std::mem::forget(device);
        std::mem::forget(session);
        return Err(Box::new(error));
    }
    assert_eq!(registered_info.host_address, Some(host_address));
    assert!(read_matches);
    assert!(write_matches);
    drop(registered);

    for cache in [
        HostCachePolicy::Coarse,
        HostCachePolicy::Fine,
        HostCachePolicy::Extended,
    ] {
        // SAFETY: The boxed page cover remains live through successful
        // deregistration or is retained on uncertain native cleanup.
        let mut registration =
            match unsafe { device.register_host(host_address, cache, 4096, 4096, access) } {
                Ok(registration) => registration,
                Err(error) => {
                    std::mem::forget(host_pages);
                    std::mem::forget(source);
                    std::mem::forget(destination);
                    std::mem::forget(device);
                    std::mem::forget(session);
                    return Err(Box::new(error));
                }
            };
        assert_eq!(registration.info().host_address, Some(host_address));
        if let Err(error) = registration.free() {
            std::mem::forget(registration);
            std::mem::forget(host_pages);
            std::mem::forget(source);
            std::mem::forget(destination);
            std::mem::forget(device);
            std::mem::forget(session);
            return Err(Box::new(error));
        }
    }

    source.free()?;
    destination.free()?;
    drop(source);
    drop(destination);
    drop(device);
    session.destroy()?;
    Ok(())
}

#[test]
#[ignore = "requires a GFX1201 GPU, KFD 1.20+, and a bound DRM render node"]
fn gfx1201_gpu_capability_contract() -> Result<(), Box<dyn Error>> {
    let mut session = Session::new(SessionLifetime::Process)?;
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
    let endpoint = selected.ok_or_else(|| io::Error::other("GFX1201 endpoint is unavailable"))?;
    let device = session.activate(&endpoint)?;
    let gpu = device.gpu()?;
    assert!(gpu.supports_expert_scheduling()?);
    assert_ne!(gpu.info().xcc_count, 0);
    assert_eq!(
        gpu.info().maximum_scratch_aperture_bytes(),
        (8_u64 << 30) * u64::from(gpu.info().xcc_count)
    );
    drop(device);
    session.destroy()?;
    Ok(())
}

#[test]
#[ignore = "requires a GFX1201 GPU, KFD, and a bound DRM render node"]
fn gfx1201_user_sdma_queue_contract() -> Result<(), Box<dyn Error>> {
    gfx1201_user_sdma_queue(QueueRingMemory::System)
}

#[test]
#[ignore = "requires GFX1201 with CPU-visible VRAM, KFD, and a bound DRM render node"]
fn gfx1201_local_ring_sdma_queue_contract() -> Result<(), Box<dyn Error>> {
    gfx1201_user_sdma_queue(QueueRingMemory::HostVisibleLocal)
}

#[allow(
    clippy::too_many_lines,
    reason = "one native queue lifetime covers packet publication, retirement, and cleanup"
)]
fn gfx1201_user_sdma_queue(ring_memory: QueueRingMemory) -> Result<(), Box<dyn Error>> {
    const COPY_BYTES: usize = 256;
    const PACKET_BYTES: usize = 68;
    const GCR: u32 = 0x11 | (1 << 8);
    const WRITEBACK: u32 = (1 << 31) | (1 << 22);
    const INVALIDATE: u32 = (1 << 30) | (1 << 25) | (1 << 24) | (1 << 23);

    let mut session = Session::new(SessionLifetime::Process)?;
    let mut selected = None;
    session.enumerate(&mut |endpoint| {
        if endpoint.gpu().is_some_and(|gpu| {
            (gpu.gfx_major, gpu.gfx_minor, gpu.gfx_stepping) == (12, 0, 1)
                && gpu.queues.sdma_system_cache_control
        }) {
            selected = Some(endpoint);
        }
        Ok(())
    })?;
    let endpoint = selected.ok_or_else(|| io::Error::other("GFX1201 endpoint is unavailable"))?;
    if ring_memory == QueueRingMemory::HostVisibleLocal {
        assert!(endpoint.host_visible_local_memory_bytes >= 4096);
    }
    let device = session.activate(&endpoint)?;
    let access = DeviceAccess::READ | DeviceAccess::WRITE;
    let mut source = device.allocate(MemoryKind::System, 4096, 4096, access)?;
    let mut destination = device.allocate(MemoryKind::System, 4096, 4096, access)?;
    let source_info = source.info();
    let destination_info = destination.info();
    let source_host = source_info
        .host_address
        .ok_or_else(|| io::Error::other("source has no host mapping"))?;
    let destination_host = destination_info
        .host_address
        .ok_or_else(|| io::Error::other("destination has no host mapping"))?;
    // SAFETY: Both live allocations expose at least COPY_BYTES writable host bytes.
    let (source_bytes, destination_bytes) = unsafe {
        (
            std::slice::from_raw_parts_mut(source_host as *mut u8, COPY_BYTES),
            std::slice::from_raw_parts_mut(destination_host as *mut u8, COPY_BYTES),
        )
    };
    for (index, byte) in source_bytes.iter_mut().enumerate() {
        *byte = (index as u8).wrapping_mul(13).wrapping_add(7);
    }
    destination_bytes.fill(0xa5);

    // SAFETY: This request contains no external GPU pointers. The single
    // producer writes one complete packet and stops before destruction.
    let mut queue = unsafe {
        device.gpu()?.create_queue(QueueRequest {
            ring_size_bytes: 4096,
            parameters: QueueParameters::SdmaByEngine {
                selection: SdmaEngineSelection::Id(0),
                ring_memory,
            },
            priority: QueuePriority::Normal,
            device_producer: true,
        })?
    };
    let transport = queue.info();
    assert_eq!(transport.sdma_engine_id, Some(0));
    assert_eq!(transport.index_unit_bytes, 1);
    assert_eq!(transport.read_index_width, QueueAccessWidth::Bits64);
    assert_eq!(transport.write_index_width, QueueAccessWidth::Bits64);
    assert_eq!(transport.doorbell_width, QueueAccessWidth::Bits64);
    assert!(transport.ring_size_bytes >= PACKET_BYTES as u64);
    assert_eq!(transport.read_index_host_address % 8, 0);
    assert_eq!(transport.write_index_host_address % 8, 0);
    assert_eq!(transport.doorbell_host_address % 8, 0);
    assert_ne!(transport.read_index_device_address, 0);
    assert_ne!(transport.write_index_device_address, 0);
    assert!(
        transport
            .doorbell_device_address
            .is_some_and(|address| address != 0)
    );

    // GFX1201 OSS5: a GCR cache envelope around one linear copy packet.
    let mut packet = [0_u32; PACKET_BYTES / 4];
    packet[0] = GCR;
    packet[2] = WRITEBACK | INVALIDATE;
    packet[5] = 1;
    packet[6] = COPY_BYTES as u32 - 1;
    packet[8] = source_info.device_address as u32;
    packet[9] = (source_info.device_address >> 32) as u32;
    packet[10] = destination_info.device_address as u32;
    packet[11] = (destination_info.device_address >> 32) as u32;
    packet[12] = GCR;
    packet[14] = WRITEBACK;
    // SAFETY: The live queue owns a writable ring of at least PACKET_BYTES.
    // The packet and both operands remain live until the read pointer retires.
    unsafe {
        std::ptr::copy_nonoverlapping(
            packet.as_ptr().cast::<u8>(),
            transport.ring_host_address as *mut u8,
            PACKET_BYTES,
        );
    }
    // SAFETY: The live queue exposes aligned 64-bit write and doorbell words.
    // The doorbell operation drains packet and index stores before notifying SDMA.
    unsafe {
        (&*(transport.write_index_host_address as *const AtomicU64))
            .store(PACKET_BYTES as u64, Ordering::Release);
        ring_doorbell(transport.doorbell_host_address, PACKET_BYTES as u64);
    }

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match queue.progress() {
            Ok((read, write)) if read == PACKET_BYTES as u64 && write == PACKET_BYTES as u64 => {
                break;
            }
            Ok((read, write)) if read <= write && Instant::now() < deadline => {
                std::thread::yield_now();
            }
            Ok((read, write)) => {
                // Native retirement is unproved. Keep every GPU-reachable
                // owner live through process teardown.
                std::mem::forget(queue);
                std::mem::forget(source);
                std::mem::forget(destination);
                std::mem::forget(device);
                std::mem::forget(session);
                return Err(io::Error::other(format!(
                    "SDMA queue did not retire packet: read={read}, write={write}"
                ))
                .into());
            }
            Err(error) => {
                std::mem::forget(queue);
                std::mem::forget(source);
                std::mem::forget(destination);
                std::mem::forget(device);
                std::mem::forget(session);
                return Err(Box::new(error));
            }
        }
    }
    fence(Ordering::Acquire);
    assert_eq!(destination_bytes, source_bytes);
    // SAFETY: The sole producer has stopped and the native read pointer
    // reached the complete write frontier before teardown.
    unsafe { queue.destroy()? };
    source.free()?;
    destination.free()?;
    drop(queue);
    drop(source);
    drop(destination);
    drop(device);
    session.destroy()?;
    Ok(())
}

#[test]
#[ignore = "requires a GFX1201 GPU, KFD, and a bound DRM render node"]
fn gfx1201_aql_barrier_contract() -> Result<(), Box<dyn Error>> {
    gfx1201_aql_barrier(false, QueueRingMemory::System)
}

#[test]
#[ignore = "requires GFX1201 with KFD GWS and a bound DRM render node"]
fn gfx1201_gws_aql_barrier_contract() -> Result<(), Box<dyn Error>> {
    gfx1201_aql_barrier(true, QueueRingMemory::System)
}

#[test]
#[ignore = "requires GFX1201 with CPU-visible VRAM, KFD, and a bound DRM render node"]
fn gfx1201_local_ring_aql_barrier_contract() -> Result<(), Box<dyn Error>> {
    gfx1201_aql_barrier(false, QueueRingMemory::HostVisibleLocal)
}

#[test]
#[ignore = "requires GFX1201 with CPU-visible VRAM, KFD GWS, and a DRM render node"]
fn gfx1201_local_ring_gws_aql_barrier_contract() -> Result<(), Box<dyn Error>> {
    gfx1201_aql_barrier(true, QueueRingMemory::HostVisibleLocal)
}

#[allow(
    clippy::too_many_lines,
    reason = "one native queue lifetime covers AQL publication, completion, and teardown"
)]
fn gfx1201_aql_barrier(
    global_work_sync: bool,
    ring_memory: QueueRingMemory,
) -> Result<(), Box<dyn Error>> {
    const AQL_PACKET_BYTES: usize = 64;
    const BARRIER_HEADER: u16 = 3 | (1 << 8) | (2 << 9) | (2 << 11);

    let mut session = Session::new(SessionLifetime::Process)?;
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
    let endpoint = selected.ok_or_else(|| io::Error::other("GFX1201 endpoint is unavailable"))?;
    let device = session.activate(&endpoint)?;
    let gpu = device.gpu()?;
    if global_work_sync {
        assert_ne!(gpu.info().gws_count, 0);
    }
    if ring_memory == QueueRingMemory::HostVisibleLocal {
        assert!(endpoint.host_visible_local_memory_bytes >= 4096);
    }
    let mut signal = device.allocate(
        MemoryKind::System,
        4096,
        4096,
        DeviceAccess::READ | DeviceAccess::WRITE,
    )?;
    let signal_info = signal.info();
    let signal_host = signal_info
        .host_address
        .ok_or_else(|| io::Error::other("signal has no host mapping"))?;
    assert_eq!(signal_host % 64, 0);
    assert_eq!(signal_info.device_address % 64, 0);
    // AMD's 64-byte user signal record has its kind at byte 0 and its
    // completion value at byte 8. Both addresses belong to the live allocation.
    let signal_value = unsafe {
        std::ptr::write_bytes(signal_host as *mut u8, 0, 64);
        (&*(signal_host as *const AtomicI64)).store(1, Ordering::Relaxed);
        let value = &*((signal_host + 8) as *const AtomicI64);
        value.store(1, Ordering::Relaxed);
        value
    };

    // SAFETY: The sole producer writes one complete packet before ringing the
    // doorbell. Its completion signal stays mapped until native retirement.
    let mut queue = unsafe {
        gpu.create_queue(QueueRequest {
            ring_size_bytes: 4096,
            parameters: QueueParameters::Aql {
                producer_mode: if global_work_sync {
                    QueueProducerMode::Multiple
                } else {
                    QueueProducerMode::Single
                },
                ring_memory,
                global_work_sync,
                inactive_signal: None,
                error_event: None,
                scratch: None,
            },
            priority: QueuePriority::Normal,
            device_producer: false,
        })?
    };
    let transport = queue.info();
    assert_eq!(transport.index_unit_bytes, AQL_PACKET_BYTES as u32);
    assert_eq!(transport.read_index_width, QueueAccessWidth::Bits64);
    assert_eq!(transport.write_index_width, QueueAccessWidth::Bits64);
    assert_eq!(transport.doorbell_width, QueueAccessWidth::Bits64);
    assert_eq!(transport.ring_host_address % AQL_PACKET_BYTES, 0);
    assert_eq!(transport.write_index_host_address % 8, 0);
    assert_eq!(transport.doorbell_host_address % 8, 0);
    assert!(transport.ring_size_bytes >= AQL_PACKET_BYTES as u64);

    let mut packet = [0_u8; AQL_PACKET_BYTES];
    packet[56..64].copy_from_slice(&signal_info.device_address.to_ne_bytes());
    // SAFETY: The queue owns a writable AQL ring and 64-bit index and doorbell
    // words. The release header publishes the initialized packet, the release
    // write index publishes slot 0, and the doorbell drains CPU stores before
    // notifying firmware.
    unsafe {
        std::ptr::copy_nonoverlapping(
            packet.as_ptr(),
            transport.ring_host_address as *mut u8,
            AQL_PACKET_BYTES,
        );
        (&*(transport.ring_host_address as *const AtomicU16))
            .store(BARRIER_HEADER, Ordering::Release);
        (&*(transport.write_index_host_address as *const AtomicU64)).store(1, Ordering::Release);
        ring_doorbell(transport.doorbell_host_address, 0);
    }

    let deadline = Instant::now() + Duration::from_secs(10);
    let completion: Result<(), Box<dyn Error>> = loop {
        let value = signal_value.load(Ordering::Acquire);
        match queue.progress() {
            Ok((read, write)) if value == 0 && read == 1 && write == 1 => break Ok(()),
            Ok((read, write)) if read <= write && Instant::now() < deadline => {
                std::thread::yield_now();
            }
            Ok((read, write)) => {
                break Err(io::Error::other(format!(
                    "AQL barrier did not retire: signal={value}, read={read}, write={write}"
                ))
                .into());
            }
            Err(error) => break Err(Box::new(error)),
        }
    };
    if let Err(error) = completion {
        // Native reachability is unresolved, so retain the queue and signal.
        std::mem::forget(queue);
        std::mem::forget(signal);
        std::mem::forget(device);
        std::mem::forget(session);
        return Err(error);
    }
    // SAFETY: The completion signal changed and native read progress reached
    // the one published packet; the sole producer has stopped.
    if let Err(error) = unsafe { queue.destroy() } {
        std::mem::forget(queue);
        std::mem::forget(signal);
        std::mem::forget(device);
        std::mem::forget(session);
        return Err(Box::new(error));
    }
    signal.free()?;
    drop(queue);
    drop(device);
    session.destroy()?;
    Ok(())
}
