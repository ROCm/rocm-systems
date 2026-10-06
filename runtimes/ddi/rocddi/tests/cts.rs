// SPDX-License-Identifier: MIT

//! Native rocddi contract checks for the first GPU target.
#![allow(unsafe_code)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "the test pattern deliberately wraps each byte index modulo 256"
)]

use std::error::Error;
use std::io;
use std::sync::atomic::AtomicBool;

use rocddi::gpu::CopyRect;
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
