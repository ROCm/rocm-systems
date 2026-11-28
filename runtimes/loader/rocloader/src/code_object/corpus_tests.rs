// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Relocation coverage for the actual kernels embedded by the HSA consumer.

use std::path::Path;

use super::{CodeObject, FloatRoundingMode, Profile, SymbolKind, SymbolTableOrder};
use crate::elf::LoadPlan;
use crate::loaded_image::CODE_ALIGNMENT;

#[test]
fn embedded_image_and_copy_kernels_materialize_at_distinct_load_addresses() {
    // These are consumer-owned test inputs, not a production dependency on
    // libhsa. Keep relocation at its implementation owner without exporting the
    // private ELF representation merely for a frontend's unit tests.
    let frontend = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../ddi/libhsa/src");
    for relative in ["image/transfer", "memory/compute_copy"] {
        let mut paths = std::fs::read_dir(frontend.join(relative))
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| {
                path.extension()
                    .is_some_and(|extension| extension == "hsaco")
            })
            .collect::<Vec<_>>();
        paths.sort();
        assert!(
            !paths.is_empty(),
            "missing embedded kernel corpus: {relative}"
        );
        for path in paths {
            let bytes = std::fs::read(&path).unwrap();
            let object = CodeObject::parse(&bytes)
                .unwrap_or_else(|error| panic!("{}: {error:?}", path.display()));
            let contract = object.contract();
            assert_eq!(contract.profile, None, "{}", path.display());
            assert_eq!(contract.rounding, None, "{}", path.display());
            for profile in [Profile::Base, Profile::Full] {
                for rounding in [
                    FloatRoundingMode::Default,
                    FloatRoundingMode::Zero,
                    FloatRoundingMode::Near,
                ] {
                    assert!(
                        contract.validate(profile, rounding).is_ok(),
                        "{}: {profile:?}/{rounding:?}",
                        path.display()
                    );
                }
            }
            assert!(
                object
                    .symbols(SymbolTableOrder::DynamicFirst)
                    .unwrap()
                    .iter()
                    .any(|symbol| symbol.kind == SymbolKind::Kernel && symbol.is_definition),
                "no defined kernel in {}",
                path.display()
            );
            let plan = LoadPlan::new(&bytes, object.layout(), CODE_ALIGNMENT).unwrap();
            // No external definitions are supplied: an accidentally introduced
            // dependency must fail here rather than only on a GPU dispatch.
            for address in [0x1_0000, 0x8000_0000_0000] {
                let mut image = vec![0xa5; plan.allocation_size()];
                plan.write_amdgpu_image(address, &mut image, |_: &str| None)
                    .unwrap_or_else(|error| panic!("{} at {address:x}: {error:?}", path.display()));
            }
        }
    }
}
