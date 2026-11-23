// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;
use crate::test_support::{write_i64, write_u32};

const VIRTUAL_BASE: u64 = 0x1000;
const ENTRY: usize = 64;

fn kernel(prologue: &[u32]) -> Vec<u8> {
    let mut image = vec![0; 128];
    write_i64(&mut image, 16, i64::try_from(ENTRY).unwrap());
    for (index, word) in prologue.iter().enumerate() {
        write_u32(&mut image, ENTRY + index * 4, *word);
    }
    image
}

#[test]
fn compiler_prologues_do_not_acquire_a_second_workaround() {
    for prologue in [&GFX1250_LEGACY_PREFETCH[..], &GFX1250_NULL_PREFETCH[..]] {
        for replay in [false, true] {
            let mut words = Vec::new();
            if replay {
                words.extend(GFX1250_REPLAY_MODE);
            }
            words.extend(prologue);
            let image = kernel(&words);
            let plan = TrampolinePlan::new(&image, VIRTUAL_BASE, [VIRTUAL_BASE]).unwrap();
            assert!(plan.is_empty());
            assert_eq!(plan.required_bytes(), 0);
        }
    }
}

#[test]
fn entry_fixup_preserves_original_kernel_and_covers_prefetch() {
    let mut image = kernel(&[0xbf81_0000]);
    // The descriptor requests more prefetch than the generated entry stub.
    write_u32(&mut image, 44, 6 << 4);
    let original_entry = image[ENTRY..].to_vec();
    let plan = TrampolinePlan::new(&image, VIRTUAL_BASE, [VIRTUAL_BASE]).unwrap();
    assert!(!plan.is_empty());
    assert_eq!(plan.required_bytes(), 6 * GFX1250_PREFETCH_LINE_BYTES);
    let mut stub = vec![0; plan.required_bytes()];
    plan.write(&mut image, 0x4000, &mut stub, 0x8000).unwrap();
    assert_eq!(read_i64(&image, 16), Some(0x4000));
    assert_eq!(&image[ENTRY..], original_entry.as_slice());
    // The generated jump returns to the original relocated kernel entry.
    assert_eq!(read_u32(&stub, 5 * 4), Some(0x4040));
    assert_eq!(read_u32(&stub, 7 * 4), Some(0));
}

#[test]
fn malformed_descriptors_and_entries_are_rejected_without_indexing() {
    let image = kernel(&[]);
    for length in 0..ENTRY + 4 {
        assert!(TrampolinePlan::new(&image[..length], VIRTUAL_BASE, [VIRTUAL_BASE]).is_err());
    }
    assert!(TrampolinePlan::new(&image, VIRTUAL_BASE, [VIRTUAL_BASE - 1]).is_err());
    let mut image = image;
    write_i64(&mut image, 16, -1);
    assert!(TrampolinePlan::new(&image, VIRTUAL_BASE, [VIRTUAL_BASE]).is_err());
}

#[test]
fn partial_prefetch_does_not_bypass_the_entry_fixup() {
    let mut image = kernel(&GFX1250_NULL_PREFETCH);
    image.truncate(ENTRY + (GFX1250_NULL_PREFETCH.len() - 1) * 4);
    let plan = TrampolinePlan::new(&image, VIRTUAL_BASE, [VIRTUAL_BASE]).unwrap();
    assert!(!plan.is_empty());
}

#[test]
fn rejected_stub_storage_leaves_the_descriptor_unchanged() {
    let image = kernel(&[]);
    let plan = TrampolinePlan::new(&image, VIRTUAL_BASE, [VIRTUAL_BASE]).unwrap();
    for (length, address) in [
        (plan.required_bytes() - 1, 0x8000),
        (plan.required_bytes(), 0x8001),
    ] {
        let mut destination = image.clone();
        let mut stub = vec![0; length];
        assert!(
            plan.write(&mut destination, 0x4000, &mut stub, address)
                .is_err()
        );
        assert_eq!(destination, image);
    }
}
