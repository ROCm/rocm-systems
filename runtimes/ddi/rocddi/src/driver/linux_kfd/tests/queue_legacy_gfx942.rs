// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Missing-size gfx942 context planning through production queue admission.

use super::*;

fn legacy_gfx942_node() -> sysfs::KfdNode {
    let mut native = native_node();
    native.queues = sysfs::KfdQueueProperties {
        gfx_target: 90402,
        target_profile: GpuTargetProfile::Gfx9,
        compute_units: 304,
        maximum_wave_count_per_compute_unit: 32,
        maximum_scratch_wave_count_per_compute_unit: 32,
        wavefront_size: 64,
        xcc_count: 8,
        shader_engine_count_per_xcc: 4,
        local_data_share_byte_length: 65536,
        compute_queues: 24,
        ..Default::default()
    };
    native
}

#[test]
fn gfx942_missing_sizes_use_per_xcc_trap_save_layout() {
    // ROCr's gfx942 formula gives the same per-XCC layout for SPX, DPX,
    // QPX and CPX nodes with 38 enabled CUs per XCC, and six-XCC APUs.
    for (xcc_count, total_size) in [
        (1, 23_252_992),
        (2, 46_505_984),
        (4, 93_011_968),
        (6, 139_517_952),
        (8, 186_019_840),
    ] {
        let mut native = legacy_gfx942_node();
        native.queues.xcc_count = xcc_count;
        native.queues.compute_units = 38 * xcc_count;
        assert!(supports_aql(&native));
        assert!(!supports_pm4(&native));
        let request =
            Request::validate(&native, descriptor(aql(QueueProducerMode::Single))).unwrap();
        let compute = request.compute.as_ref().unwrap();
        assert_eq!(compute.context_size, 0x0162_1000);
        assert_eq!(compute.control_stack_size, 12288);
        assert_eq!(compute.debug_size, 48640);
        assert_eq!(compute.xcc_count, xcc_count);
        assert_eq!(compute.total_size, total_size);
        let reservation = os_memory::Reservation::new_aligned_host(
            usize::try_from(total_size).unwrap(),
            CWSR_ALIGNMENT,
            (0, isize::MAX as u64),
        )
        .unwrap();
        let address = reservation.address() as u64;
        let mut storage = QueueContext::Svm(reservation);
        initialize_queue_context(&mut storage, &request, compute, 0).unwrap();
        let memory = File::open("/proc/self/mem").unwrap();
        for xcc in 0..xcc_count {
            let mut header = [0_u8; 40];
            memory
                .read_exact_at(&mut header, address + u64::from(xcc * compute.context_size))
                .unwrap();
            let mut expected = [0_u8; 40];
            expected[16..20]
                .copy_from_slice(&((xcc_count - xcc) * compute.context_size).to_ne_bytes());
            expected[20..24].copy_from_slice(&(48640 * xcc_count).to_ne_bytes());
            assert_eq!(header, expected);
        }
        storage.release().unwrap();
    }
}

#[test]
fn gfx942_legacy_stack_and_debugger_share_kernel_wave_cap() {
    let mut native = legacy_gfx942_node();
    native.queues.compute_units = 40;
    native.queues.xcc_count = 1;
    native.queues.shader_engine_count_per_xcc = 2;
    let compute = compute_storage(native.queues).unwrap();
    assert_eq!(compute.control_stack_size, 12288);
    assert_eq!(compute.context_size, 0x0174_b000);
    assert_eq!(compute.debug_size, 32768);
    assert_eq!(compute.total_size, 24_457_216);
}

#[test]
fn reported_context_sizes_remain_authoritative() {
    let fixture = Fixture::new(true);
    let mut native = legacy_gfx942_node();
    native.queues.context_size = 65536;
    native.queues.control_stack_size = 4096;
    // Explicit kernel sizes do not depend on legacy register-file qualification.
    native.queues.local_data_share_byte_length = 128 * 1024;
    let compute = compute_storage(native.queues).unwrap();
    assert_eq!(compute.context_size, 65536);
    assert_eq!(compute.control_stack_size, 4096);
    assert_eq!(compute.debug_size, 48640);
    assert_eq!(compute.total_size, 913_408);
    for (context_size, control_stack_size) in [
        (0, 4096),
        (65536, 0),
        (65537, 4096),
        (65536, 4097),
        (4096, 8192),
    ] {
        native.queues.context_size = context_size;
        native.queues.control_stack_size = control_stack_size;
        assert!(!supports_aql(&native));
        assert!(Request::validate(&native, descriptor(aql(QueueProducerMode::Single))).is_err());
        assert!(
            create(
                fixture.vm.clone(),
                &native,
                descriptor(aql(QueueProducerMode::Single)),
                DriverContextLifetime::Process,
            )
            .is_err()
        );
    }
    assert_eq!(fixture.state.lock().unwrap().next_handle, 0);
    fixture.assert_released();
}

#[test]
fn legacy_context_rejects_unqualified_geometry_and_overflow_before_acquisition() {
    let fixture = Fixture::new(true);
    let base = legacy_gfx942_node().queues;
    for queues in [
        sysfs::KfdQueueProperties {
            gfx_target: 90500,
            ..base
        },
        sysfs::KfdQueueProperties {
            gfx_target: 90010,
            ..base
        },
        sysfs::KfdQueueProperties {
            gfx_target: 90403,
            ..base
        },
        sysfs::KfdQueueProperties {
            xcc_count: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            compute_units: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            compute_units: 303,
            ..base
        },
        sysfs::KfdQueueProperties {
            wavefront_size: 32,
            ..base
        },
        sysfs::KfdQueueProperties {
            maximum_wave_count_per_compute_unit: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            shader_engine_count_per_xcc: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            local_data_share_byte_length: 0,
            ..base
        },
        sysfs::KfdQueueProperties {
            local_data_share_byte_length: 131_072,
            ..base
        },
        sysfs::KfdQueueProperties {
            shader_engine_count_per_xcc: u32::MAX,
            ..base
        },
        sysfs::KfdQueueProperties {
            compute_units: u32::MAX - 7,
            ..base
        },
        sysfs::KfdQueueProperties {
            compute_units: 8000,
            xcc_count: 8000,
            ..base
        },
    ] {
        let mut native = legacy_gfx942_node();
        native.queues = queues;
        assert!(!supports_aql(&native), "{queues:?}");
        assert!(
            Request::validate(&native, descriptor(aql(QueueProducerMode::Single))).is_err(),
            "{queues:?}"
        );
        assert!(
            create(
                fixture.vm.clone(),
                &native,
                descriptor(aql(QueueProducerMode::Single)),
                DriverContextLifetime::Process,
            )
            .is_err(),
            "{queues:?}"
        );
    }
    assert_eq!(fixture.state.lock().unwrap().next_handle, 0);
    fixture.assert_released();
    assert!(!supports_aql(&sysfs::KfdNode {
        queues: sysfs::KfdQueueProperties {
            context_size: 0,
            control_stack_size: 0,
            ..native_node().queues
        },
        ..native_node()
    }));
}
