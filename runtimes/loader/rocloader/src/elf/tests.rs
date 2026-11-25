// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::constants::{ELF_MACHINE_AMDGPU, ELF_OS_ABI_AMDGPU_HSA, ELF_TYPE_DYN};
use super::*;
use crate::test_support::*;

#[test]
/// Checks rejection of truncated and corrupt ELF tables.
fn rejects_truncated_and_corrupt_elf_tables() {
    let original = gfx1201_code_object();
    // The last section header ends at byte 384. Every shorter prefix is
    // missing either the ELF header or part of its declared section table.
    for length in 0..384 {
        assert!(matches!(
            parse_elf64(&original[..length]),
            Err(Error::InvalidObject)
        ));
    }

    let mut bytes = original.clone();
    write_u64(&mut bytes, 32, u64::MAX);
    write_u16(&mut bytes, 56, 1);
    assert!(matches!(parse_elf64(&bytes), Err(Error::InvalidObject)));

    let mut bytes = original.clone();
    write_u64(&mut bytes, 40, u64::MAX);
    assert!(matches!(parse_elf64(&bytes), Err(Error::InvalidObject)));

    let symbol_section = 64 + 128;
    let string_section = 64 + 64;
    let cases: &[(usize, u64)] = &[
        (symbol_section + 24, u64::MAX),
        (symbol_section + 32, u64::MAX),
        (symbol_section + 56, 0),
        (string_section + 24, u64::MAX),
        (string_section + 32, u64::MAX),
    ];
    for &(offset, value) in cases {
        let mut bytes = original.clone();
        write_u64(&mut bytes, offset, value);
        let layout = parse_elf64(&bytes).unwrap();
        assert!(matches!(
            amdgpu_symbols(&bytes, layout, &[SECTION_SYMTAB]),
            Err(Error::InvalidObject)
        ));
    }

    let mut bytes = original.clone();
    write_u32(&mut bytes, symbol_section + 40, 5);
    let layout = parse_elf64(&bytes).unwrap();
    assert!(matches!(
        amdgpu_symbols(&bytes, layout, &[SECTION_SYMTAB]),
        Err(Error::InvalidObject)
    ));

    let mut bytes = original.clone();
    bytes[385] = 0xff; // invalid UTF-8 in the first symbol name
    let layout = parse_elf64(&bytes).unwrap();
    assert!(matches!(
        amdgpu_symbols(&bytes, layout, &[SECTION_SYMTAB]),
        Err(Error::InvalidObject)
    ));

    let mut bytes = original;
    bytes[409] = b'x'; // remove the final NUL in the string table
    let layout = parse_elf64(&bytes).unwrap();
    assert!(matches!(
        amdgpu_symbols(&bytes, layout, &[SECTION_SYMTAB]),
        Err(Error::InvalidObject)
    ));
}

#[test]
/// Verifies that generated ELF mutations keep parser and relocator bounded.
fn generated_elf_mutations_keep_parser_and_relocator_bounded() {
    let original = gfx1201_code_object();
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    // Deterministic malformed-input sweep, including table offsets,
    // counts, symbol strings and bytes that can become relocation headers.
    for case in 0..4096 {
        let mut bytes = original.clone();
        for _ in 0..=(case % 8) {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let index = usize::try_from(seed % u64::try_from(bytes.len()).unwrap()).unwrap();
            bytes[index] ^= seed.to_le_bytes()[4] | 1;
        }
        let length = if case % 4 == 0 {
            usize::try_from(seed % u64::try_from(bytes.len() + 1).unwrap()).unwrap()
        } else {
            bytes.len()
        };
        let bytes = &bytes[..length];
        if let Ok(layout) = parse_elf64(bytes) {
            let _ = amdgpu_symbols(bytes, layout, &[SECTION_SYMTAB, SECTION_DYNSYM]);
            let mut image = [0_u8; 256];
            let _ = apply_amdgpu_relocations(bytes, layout, 0x100, 0x1000, &mut image, |_| None);
        }
    }
}

#[test]
/// Checks application of relative and defined absolute relocations.
fn applies_relative_and_defined_absolute_relocations() {
    /// Byte extent of one ELF section-header entry.
    const SECTION_ENTRY_SIZE: usize = 64;
    /// Number of sections in the relocation fixture.
    const SECTION_COUNT: usize = 4;
    /// Byte offset of the symbol table in the fixture.
    const SYMBOL_OFFSET: usize = SECTION_ENTRY_SIZE * SECTION_COUNT;
    /// Byte offset of the relocation table in the fixture.
    const RELOCATION_OFFSET: usize = SYMBOL_OFFSET + 48;
    /// Byte offset of the string table in the fixture.
    const STRING_OFFSET: usize = RELOCATION_OFFSET + 48;
    let mut bytes = vec![0_u8; STRING_OFFSET + 1];

    let symbol_section = SECTION_ENTRY_SIZE;
    write_u32(&mut bytes, symbol_section + 4, SECTION_DYNSYM);
    write_u64(&mut bytes, symbol_section + 24, SYMBOL_OFFSET as u64);
    write_u64(&mut bytes, symbol_section + 32, 48);
    write_u32(&mut bytes, symbol_section + 40, 3);
    write_u64(&mut bytes, symbol_section + 56, 24);
    write_u16(&mut bytes, SYMBOL_OFFSET + 24 + 6, 1);
    write_u64(&mut bytes, SYMBOL_OFFSET + 24 + 8, 0x140);

    let string_section = SECTION_ENTRY_SIZE * 3;
    write_u32(&mut bytes, string_section + 4, SECTION_STRTAB);
    write_u64(&mut bytes, string_section + 24, STRING_OFFSET as u64);
    write_u64(&mut bytes, string_section + 32, 1);

    let relocation_section = SECTION_ENTRY_SIZE * 2;
    write_u32(&mut bytes, relocation_section + 4, SECTION_RELA);
    write_u64(
        &mut bytes,
        relocation_section + 24,
        RELOCATION_OFFSET as u64,
    );
    write_u64(&mut bytes, relocation_section + 32, 48);
    write_u32(&mut bytes, relocation_section + 40, 1);
    write_u64(&mut bytes, relocation_section + 56, 24);

    write_u64(&mut bytes, RELOCATION_OFFSET, 0x120);
    write_u64(
        &mut bytes,
        RELOCATION_OFFSET + 8,
        u64::from(RELOCATION_RELATIVE_64),
    );
    write_i64(&mut bytes, RELOCATION_OFFSET + 16, 0x180);
    write_u64(&mut bytes, RELOCATION_OFFSET + 24, 0x128);
    write_u64(
        &mut bytes,
        RELOCATION_OFFSET + 32,
        (1_u64 << 32) | u64::from(RELOCATION_ABSOLUTE_64),
    );
    write_i64(&mut bytes, RELOCATION_OFFSET + 40, 8);

    let mut image = [0_u8; 256];
    let layout = Layout {
        os_abi: ELF_OS_ABI_AMDGPU_HSA,
        abi_version: 0,
        kind: ELF_TYPE_DYN,
        machine: ELF_MACHINE_AMDGPU,
        flags: 0,
        program_offset: 0,
        program_entry_size: 0,
        program_count: 0,
        section_offset: 0,
        section_entry_size: SECTION_ENTRY_SIZE,
        section_count: SECTION_COUNT,
    };
    assert_eq!(
        apply_amdgpu_relocations(&bytes, layout, 0x100, 0x1000, &mut image, |_| None,),
        Ok(())
    );
    assert_eq!(read_u64(&image, 0x20), Some(0x1080));
    assert_eq!(read_u64(&image, 0x28), Some(0x1048));
    // A supplied destination may be below the object's preferred virtual base.
    assert_eq!(
        apply_amdgpu_relocations(&bytes, layout, 0x100, 0x80, &mut image, |_| None),
        Ok(())
    );
    assert_eq!(read_u64(&image, 0x20), Some(0x100));
    assert_eq!(read_u64(&image, 0x28), Some(0xc8));
}
