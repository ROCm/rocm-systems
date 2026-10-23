// SPDX-License-Identifier: MIT

//! Native context sizes pinned to libhsakmt's register, wave and LDS rules.

#![allow(clippy::unwrap_used)]

use super::*;

fn properties(gfx_target: u32) -> KfdQueueProperties {
    KfdQueueProperties {
        gfx_target,
        compute_units: 32,
        maximum_wave_count_per_compute_unit: 32,
        local_data_share_bytes_per_compute_unit: 64 * 1024,
        xcc_count: 1,
        ..KfdQueueProperties::default()
    }
}

#[test]
fn rdna_register_storage_matches_libhsakmt() {
    for (target, context_size) in [
        (100_100, 11_157_504),
        (100_300, 11_157_504),
        (110_000, 15_351_808),
        (110_501, 15_351_808),
        (120_001, 15_351_808),
    ] {
        let storage = ComputeStorage::new(properties(target), 4096).unwrap();
        assert_eq!(storage.context_size, context_size, "target {target}");
        assert_eq!(storage.control_stack_size, 16_384);
        assert_eq!(storage.debug_size, 32_768);
        assert_eq!(storage.total_size, u64::from(context_size) + 32_768);
    }
}

#[test]
fn lds_capacity_and_wave_capacity_size_independent_regions() {
    let mut properties = properties(110_501);
    properties.local_data_share_bytes_per_compute_unit = 128 * 1024;
    properties.maximum_wave_count_per_compute_unit = 16;
    let storage = ComputeStorage::new(properties, 4096).unwrap();
    assert_eq!(storage.control_stack_size, 8192);
    assert_eq!(storage.context_size, 17_440_768);
    assert_eq!(storage.debug_size, 16_384);
}

#[test]
fn gfx10_control_stack_obeys_hardware_limit() {
    let mut properties = properties(100_300);
    properties.compute_units = 80;
    properties.maximum_wave_count_per_compute_unit = 40;
    let storage = ComputeStorage::new(properties, 4096).unwrap();
    assert_eq!(storage.control_stack_size, 0x7000);
    assert_eq!(storage.context_size, 27_881_472);
    assert_eq!(storage.debug_size, 102_400);
}

#[test]
fn host_page_size_and_xcc_count_preserve_native_layout() {
    let mut properties = properties(110_501);
    properties.compute_units = 64;
    properties.xcc_count = 2;
    let storage = ComputeStorage::new(properties, 65536).unwrap();
    assert_eq!(storage.control_stack_size, 65_536);
    assert_eq!(storage.context_size, 15_400_960);
    assert_eq!(storage.debug_size, 32_768);
    assert_eq!(storage.xcc_count, 2);
    assert_eq!(storage.total_size, 30_867_456);
}

#[test]
fn driver_sizes_are_authoritative_and_debug_storage_follows_wave_count() {
    let mut properties = properties(120_001);
    properties.context_sizes = Some(QueueContextSizes {
        context_size: 65_536,
        control_stack_size: 4096,
    });
    properties.maximum_wave_count_per_compute_unit = 16;
    let storage = ComputeStorage::new(properties, 4096).unwrap();
    assert_eq!(storage.context_size, 65_536);
    assert_eq!(storage.control_stack_size, 4096);
    assert_eq!(storage.debug_size, 16_384);
    assert_eq!(storage.total_size, 81_920);
}

#[test]
fn debugger_and_allocation_extents_are_rounded_independently() {
    let mut properties = properties(120_001);
    properties.compute_units = 1;
    properties.maximum_wave_count_per_compute_unit = 1;
    let storage = ComputeStorage::new(properties, 4096).unwrap();
    assert_eq!(storage.context_size, 483_328);
    assert_eq!(storage.debug_size, 64);
    assert_eq!(storage.total_size, 487_424);
}

#[test]
fn driver_contexts_cannot_overflow_the_multi_xcc_header_extent() {
    let mut properties = properties(120_001);
    properties.xcc_count = 2;
    properties.context_sizes = Some(QueueContextSizes {
        context_size: 0xffff_f000,
        control_stack_size: 4096,
    });
    assert!(ComputeStorage::new(properties, 4096).is_err());
}

#[test]
fn malformed_driver_sizes_are_rejected_without_derivation() {
    for (context_size, control_stack_size) in [
        (0, 0),
        (65_537, 4096),
        (65_536, 0),
        (65_536, 40),
        (65_536, 69_632),
    ] {
        let mut properties = properties(120_001);
        properties.context_sizes = Some(QueueContextSizes {
            context_size,
            control_stack_size,
        });
        assert!(ComputeStorage::new(properties, 4096).is_err());
    }
}

#[test]
fn unsupported_targets_geometry_and_overflow_are_rejected() {
    assert!(ComputeStorage::new(properties(120_500), 4096).is_err());
    for page_size in [0, 2048, 6144, u64::MAX] {
        assert!(ComputeStorage::new(properties(120_001), page_size).is_err());
    }
    let valid = properties(120_001);
    for properties in [
        KfdQueueProperties {
            compute_units: 0,
            ..valid
        },
        KfdQueueProperties {
            xcc_count: 0,
            ..valid
        },
        KfdQueueProperties {
            xcc_count: 3,
            ..valid
        },
        KfdQueueProperties {
            maximum_wave_count_per_compute_unit: 0,
            ..valid
        },
        KfdQueueProperties {
            local_data_share_bytes_per_compute_unit: 0,
            ..valid
        },
        KfdQueueProperties {
            compute_units: u32::MAX,
            maximum_wave_count_per_compute_unit: u32::MAX,
            ..valid
        },
        KfdQueueProperties {
            compute_units: 32_768,
            ..valid
        },
    ] {
        assert!(ComputeStorage::new(properties, 4096).is_err());
    }
}
