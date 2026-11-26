// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;
use crate::elf::{apply_amdgpu_relocations, constants::*};
use crate::test_support::*;

#[test]
/// Checks parsing of GFX1201 code object metadata and symbols.
fn parses_gfx1201_code_object_metadata_and_symbols() {
    let bytes = gfx1201_code_object();
    let object = CodeObject::parse(&bytes).unwrap();
    assert_eq!(object.version().ok(), Some(6));
    assert_eq!(object.target().ok(), Some(("gfx1201", 0)));
    let symbols = object.symbols(SymbolTableOrder::RegularFirst);
    assert!(symbols.is_ok());
    let Some(symbols) = symbols.ok() else {
        return;
    };
    assert_eq!(symbols.len(), 2);

    let kernel = &symbols[0];
    assert_eq!(kernel.full_name, "module::kernel.kd");
    assert_eq!(kernel.module_name, "module");
    assert_eq!(kernel.name, "kernel.kd");
    assert_eq!(kernel.kind, SymbolKind::Kernel);
    assert_eq!(kernel.linkage, SymbolLinkage::Program);
    assert!(kernel.is_definition);
    assert_eq!(kernel.kernarg_size, 16);
    assert_eq!(kernel.group_size, 32);
    assert_eq!(kernel.private_size, 4);
    assert!(kernel.dynamic_callstack);
    assert_eq!(kernel.wavefront_size, 32);

    let variable = &symbols[1];
    assert_eq!(variable.name, "global");
    assert_eq!(variable.kind, SymbolKind::Variable);
    assert_eq!(variable.allocation, VariableAllocation::Agent);
    assert_eq!(variable.segment, VariableSegment::Global);
    assert_eq!(variable.alignment, 8);
    assert!(variable.is_const);
}

#[test]
/// Checks rejection of non HSA ELF images.
fn rejects_non_hsa_elf_images() {
    let mut bytes = gfx1201_code_object();
    bytes[7] = 0;
    assert!(matches!(
        CodeObject::parse(&bytes),
        Err(Error::InvalidObject)
    ));
}

#[test]
/// Legacy kernels and short descriptor sections cannot use modern metadata.
fn rejects_legacy_and_truncated_kernel_descriptors() {
    let mut bytes = gfx1201_code_object();
    bytes[8] = 0;
    assert!(matches!(
        CodeObject::parse(&bytes),
        Err(Error::InvalidObject)
    ));
    let mut bytes = gfx1201_code_object();
    // The fixture has more bytes after .text, but those are not descriptor bytes.
    write_u64(&mut bytes, 64 + 192 + 32, 32);
    assert!(
        CodeObject::parse(&bytes)
            .unwrap()
            .symbols(SymbolTableOrder::RegularFirst)
            .is_err()
    );
    let mut bytes = gfx1201_code_object();
    bytes[448 + 24 + 4] = (SYMBOL_BINDING_GLOBAL << 4) | SYMBOL_TYPE_AMDGPU_HSA_KERNEL;
    assert!(
        CodeObject::parse(&bytes)
            .unwrap()
            .symbols(SymbolTableOrder::RegularFirst)
            .is_err()
    );
}

#[test]
/// External kernel descriptors resolve without local definition bytes.
fn resolves_undefined_kernel_descriptor_references() {
    for symbol_type in [0, SYMBOL_TYPE_OBJECT] {
        let mut bytes = gfx1201_code_object();
        let kernel = 448 + 24;
        bytes[kernel + 4] = (SYMBOL_BINDING_GLOBAL << 4) | symbol_type;
        write_u16(&mut bytes, kernel + 6, 0); // SHN_UNDEF
        write_u64(&mut bytes, kernel + 8, 0);
        write_u64(&mut bytes, kernel + 16, 0);

        // Append a relocation and relocate the header table to add its section.
        let mut sections = bytes[64..384].to_vec();
        let relocation = bytes.len();
        bytes.resize(relocation + 24, 0);
        write_u64(&mut bytes, relocation, 0x2000);
        write_u64(
            &mut bytes,
            relocation + 8,
            (1_u64 << 32) | u64::from(RELOCATION_ABSOLUTE_64),
        );
        let header = sections.len();
        sections.resize(header + 64, 0);
        write_u32(&mut sections, header + 4, SECTION_RELA);
        write_u64(&mut sections, header + 24, relocation as u64);
        write_u64(&mut sections, header + 32, 24);
        write_u32(&mut sections, header + 40, 2); // Symbol-table section.
        write_u64(&mut sections, header + 56, 24);
        let section_offset = bytes.len();
        bytes.extend_from_slice(&sections);
        write_u64(&mut bytes, 40, section_offset as u64);
        write_u16(&mut bytes, 60, 6);

        let object = CodeObject::parse(&bytes).unwrap();
        let symbols = object.symbols(SymbolTableOrder::DynamicFirst).unwrap();
        let imported = &symbols[0];
        assert_eq!(imported.full_name, "module::kernel.kd");
        assert_eq!(imported.kind, SymbolKind::Kernel);
        assert!(!imported.is_definition);
        assert_eq!((imported.group_size, imported.private_size), (0, 0));

        let mut image = [0_u8; 8];
        assert_eq!(
            apply_amdgpu_relocations(&bytes, object.layout(), 0x2000, 0x4000, &mut image, |_| {
                None
            }),
            Err(crate::elf::Error::UndefinedSymbol)
        );
        let address = 0x1234_5678_0000;
        assert_eq!(
            apply_amdgpu_relocations(
                &bytes,
                object.layout(),
                0x2000,
                0x4000,
                &mut image,
                |name| { (name == imported.full_name).then_some(address) }
            ),
            Ok(())
        );
        assert_eq!(u64::from_le_bytes(image), address);
    }
}
