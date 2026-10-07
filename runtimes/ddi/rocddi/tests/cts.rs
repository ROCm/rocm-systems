// SPDX-License-Identifier: MIT

//! Native rocddi contract checks for the first GPU target.
#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "the test pattern deliberately wraps each byte index modulo 256"
)]

use std::error::Error;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU16, AtomicU64, Ordering, fence};
use std::time::{Duration, Instant};

use rocddi::gpu::queue::{
    QueueAccessWidth, QueueParameters, QueuePriority, QueueProducerMode, QueueRequest,
    SdmaEngineSelection,
};
use rocddi::gpu::{CopyRect, GpuCopySequence};
use rocddi::memory::{DeviceAccess, MemoryKind};
use rocddi::session::{Session, SessionLifetime};

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
#[allow(
    clippy::too_many_lines,
    reason = "one native queue lifetime covers packet publication, retirement, and cleanup"
)]
fn gfx1201_user_sdma_queue_contract() -> Result<(), Box<dyn Error>> {
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
    fence(Ordering::SeqCst);
    // SAFETY: The live queue exposes aligned 64-bit write and doorbell words.
    // Packet stores precede the release write index and the MMIO doorbell.
    unsafe {
        (&*(transport.write_index_host_address as *const AtomicU64))
            .store(PACKET_BYTES as u64, Ordering::Release);
        fence(Ordering::SeqCst);
        std::ptr::write_volatile(
            transport.doorbell_host_address as *mut u64,
            PACKET_BYTES as u64,
        );
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
    drop(device);
    session.destroy()?;
    Ok(())
}

#[test]
#[ignore = "requires a GFX1201 GPU, KFD, and a bound DRM render node"]
#[allow(
    clippy::too_many_lines,
    reason = "one native queue lifetime covers AQL publication, completion, and teardown"
)]
fn gfx1201_aql_barrier_contract() -> Result<(), Box<dyn Error>> {
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
        device.gpu()?.create_queue(QueueRequest {
            ring_size_bytes: 4096,
            parameters: QueueParameters::Aql {
                producer_mode: QueueProducerMode::Single,
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
    // write index publishes slot 0, and the MMIO doorbell notifies firmware.
    unsafe {
        std::ptr::copy_nonoverlapping(
            packet.as_ptr(),
            transport.ring_host_address as *mut u8,
            AQL_PACKET_BYTES,
        );
        (&*(transport.ring_host_address as *const AtomicU16))
            .store(BARRIER_HEADER, Ordering::Release);
        (&*(transport.write_index_host_address as *const AtomicU64)).store(1, Ordering::Release);
        fence(Ordering::SeqCst);
        std::ptr::write_volatile(transport.doorbell_host_address as *mut u64, 0);
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
