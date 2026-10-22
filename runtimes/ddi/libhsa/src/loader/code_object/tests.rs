// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;

fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn write_i64(bytes: &mut [u8], offset: usize, value: i64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn gfx1201_code_object() -> Vec<u8> {
    const SECTION_OFFSET: usize = 64;
    const SECTION_COUNT: usize = 5;
    const STRING_OFFSET: usize = SECTION_OFFSET + SECTION_COUNT * 64;
    const SYMBOL_OFFSET: usize = 448;
    const TEXT_OFFSET: usize = 544;
    const DATA_OFFSET: usize = 608;
    const STRINGS: &[u8] = b"\0module::kernel.kd\0global\0";
    let mut bytes = vec![0_u8; DATA_OFFSET + 8];
    bytes[..4].copy_from_slice(b"\x7fELF");
    bytes[4] = 2;
    bytes[5] = 1;
    bytes[7] = ELF_OS_ABI_AMDGPU_HSA;
    bytes[8] = 4;
    write_u16(&mut bytes, 16, ELF_TYPE_DYN);
    write_u16(&mut bytes, 18, ELF_MACHINE_AMDGPU);
    write_u32(&mut bytes, 20, 1);
    write_u64(&mut bytes, 40, SECTION_OFFSET as u64);
    write_u32(&mut bytes, 48, 0x4e); // gfx1201 machine ID
    write_u16(&mut bytes, 52, ELF_HEADER_SIZE as u16);
    write_u16(&mut bytes, 54, 56);
    write_u16(&mut bytes, 58, 64);
    write_u16(&mut bytes, 60, SECTION_COUNT as u16);

    let strings = SECTION_OFFSET + 64;
    write_u32(&mut bytes, strings + 4, SECTION_STRTAB);
    write_u64(&mut bytes, strings + 24, STRING_OFFSET as u64);
    write_u64(&mut bytes, strings + 32, STRINGS.len() as u64);
    bytes[STRING_OFFSET..STRING_OFFSET + STRINGS.len()].copy_from_slice(STRINGS);

    let symbols = SECTION_OFFSET + 128;
    write_u32(&mut bytes, symbols + 4, SECTION_SYMTAB);
    write_u64(&mut bytes, symbols + 24, SYMBOL_OFFSET as u64);
    write_u64(&mut bytes, symbols + 32, 72);
    write_u32(&mut bytes, symbols + 40, 1);
    write_u64(&mut bytes, symbols + 56, 24);

    let text = SECTION_OFFSET + 192;
    write_u32(&mut bytes, text + 4, SECTION_PROGBITS);
    write_u64(&mut bytes, text + 16, 0x1000);
    write_u64(&mut bytes, text + 24, TEXT_OFFSET as u64);
    write_u64(&mut bytes, text + 32, 64);
    write_u64(&mut bytes, text + 48, 64);

    let data = SECTION_OFFSET + 256;
    write_u32(&mut bytes, data + 4, SECTION_PROGBITS);
    write_u64(
        &mut bytes,
        data + 8,
        SECTION_FLAG_WRITE | SECTION_FLAG_AMDGPU_HSA_AGENT,
    );
    write_u64(&mut bytes, data + 16, 0x2000);
    write_u64(&mut bytes, data + 24, DATA_OFFSET as u64);
    write_u64(&mut bytes, data + 32, 8);
    write_u64(&mut bytes, data + 48, 8);

    let kernel = SYMBOL_OFFSET + 24;
    write_u32(&mut bytes, kernel, 1);
    bytes[kernel + 4] = (SYMBOL_BINDING_GLOBAL << 4) | SYMBOL_TYPE_AMDGPU_HSA_KERNEL;
    write_u16(&mut bytes, kernel + 6, 3);
    write_u64(&mut bytes, kernel + 8, 0x1000);
    write_u64(&mut bytes, kernel + 16, 64);

    let variable = SYMBOL_OFFSET + 48;
    write_u32(&mut bytes, variable, 19);
    bytes[variable + 4] = (SYMBOL_BINDING_GLOBAL << 4) | SYMBOL_TYPE_OBJECT;
    write_u16(&mut bytes, variable + 6, 4);
    write_u64(&mut bytes, variable + 8, 0x2000);
    write_u64(&mut bytes, variable + 16, 8);

    write_u32(&mut bytes, TEXT_OFFSET, 32);
    write_u32(&mut bytes, TEXT_OFFSET + 4, 4);
    write_u32(&mut bytes, TEXT_OFFSET + 8, 16);
    write_u16(&mut bytes, TEXT_OFFSET + 56, 1 << 11);
    bytes
}

#[test]
fn parses_gfx1201_code_object_metadata_and_symbols() {
    let bytes = gfx1201_code_object();
    let layout = parse_elf_layout(&bytes);
    assert!(layout.is_ok());
    let Some(layout) = layout.ok() else {
        return;
    };
    let object = LoaderCodeObject::parse(&bytes).unwrap();
    assert_eq!(object.version().ok(), Some(6));
    assert_eq!(object.target().ok(), Some(("gfx1201", 0)));
    let symbols = parse_symbols(&bytes, layout, &[SECTION_SYMTAB]);
    assert!(symbols.is_ok());
    let Some(symbols) = symbols.ok() else {
        return;
    };
    assert_eq!(symbols.len(), 2);

    let kernel = &symbols[0];
    assert_eq!(kernel.full_name, "module::kernel.kd");
    assert_eq!(kernel.module_name, "module");
    assert_eq!(kernel.name, "kernel.kd");
    assert_eq!(kernel.kind, SYMBOL_KIND_KERNEL);
    assert_eq!(kernel.linkage, SYMBOL_LINKAGE_PROGRAM);
    assert!(kernel.is_definition);
    assert_eq!(kernel.kernarg_size, 16);
    assert_eq!(kernel.group_size, 32);
    assert_eq!(kernel.private_size, 4);
    assert!(kernel.dynamic_callstack);
    assert_eq!(kernel.wavefront_size, 32);

    let variable = &symbols[1];
    assert_eq!(variable.name, "global");
    assert_eq!(variable.kind, SYMBOL_KIND_VARIABLE);
    assert_eq!(variable.allocation, VARIABLE_ALLOCATION_AGENT);
    assert_eq!(variable.segment, VARIABLE_SEGMENT_GLOBAL);
    assert_eq!(variable.alignment, 8);
    assert!(variable.is_const);
}

#[test]
fn rejects_non_hsa_elf_images() {
    let mut bytes = gfx1201_code_object();
    bytes[7] = 0;
    assert!(matches!(parse_elf_layout(&bytes), Err(INVALID_CODE_OBJECT)));
}

#[test]
fn rejects_truncated_and_corrupt_elf_tables() {
    let original = gfx1201_code_object();
    // The last section header ends at byte 384. Every shorter prefix is
    // missing either the ELF header or part of its declared section table.
    for length in 0..384 {
        assert!(matches!(
            parse_elf_layout(&original[..length]),
            Err(INVALID_CODE_OBJECT)
        ));
    }

    let mut bytes = original.clone();
    write_u64(&mut bytes, 32, u64::MAX);
    write_u16(&mut bytes, 56, 1);
    assert!(matches!(parse_elf_layout(&bytes), Err(INVALID_CODE_OBJECT)));

    let mut bytes = original.clone();
    write_u64(&mut bytes, 40, u64::MAX);
    assert!(matches!(parse_elf_layout(&bytes), Err(INVALID_CODE_OBJECT)));

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
        let layout = parse_elf_layout(&bytes).unwrap();
        assert!(matches!(
            parse_symbols(&bytes, layout, &[SECTION_SYMTAB]),
            Err(INVALID_CODE_OBJECT)
        ));
    }

    let mut bytes = original.clone();
    write_u32(&mut bytes, symbol_section + 40, 5);
    let layout = parse_elf_layout(&bytes).unwrap();
    assert!(matches!(
        parse_symbols(&bytes, layout, &[SECTION_SYMTAB]),
        Err(INVALID_CODE_OBJECT)
    ));

    let mut bytes = original.clone();
    bytes[385] = 0xff; // invalid UTF-8 in the first symbol name
    let layout = parse_elf_layout(&bytes).unwrap();
    assert!(matches!(
        parse_symbols(&bytes, layout, &[SECTION_SYMTAB]),
        Err(INVALID_CODE_OBJECT)
    ));

    let mut bytes = original;
    bytes[409] = b'x'; // remove the final NUL in the string table
    let layout = parse_elf_layout(&bytes).unwrap();
    assert!(matches!(
        parse_symbols(&bytes, layout, &[SECTION_SYMTAB]),
        Err(INVALID_CODE_OBJECT)
    ));
}

#[test]
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
            bytes[index] ^= (seed >> 32) as u8 | 1;
        }
        let length = if case % 4 == 0 {
            usize::try_from(seed % u64::try_from(bytes.len() + 1).unwrap()).unwrap()
        } else {
            bytes.len()
        };
        let bytes = &bytes[..length];
        if let Ok(object) = LoaderCodeObject::parse(bytes) {
            let layout = object.layout();
            let _ = object.version();
            let _ = parse_symbols(bytes, layout, &[SECTION_SYMTAB, SECTION_DYNSYM]);
            let mut image = [0_u8; 256];
            let _ = apply_relocations(bytes, layout, 0x100, 0x1000, &mut image, |_| None);
        }
    }
}

#[test]
fn applies_relative_and_defined_absolute_relocations() {
    const SECTION_ENTRY_SIZE: usize = 64;
    const SECTION_COUNT: usize = 4;
    const SYMBOL_OFFSET: usize = SECTION_ENTRY_SIZE * SECTION_COUNT;
    const RELOCATION_OFFSET: usize = SYMBOL_OFFSET + 48;
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
    let layout = ElfLayout {
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
        apply_relocations(&bytes, layout, 0x100, 0x1000, &mut image, |_| None,),
        Ok(())
    );
    assert_eq!(read_u64(&image, 0x20), Some(0x1080));
    assert_eq!(read_u64(&image, 0x28), Some(0x1048));
}
