// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host-only ELF fixtures shared by byte and code-object contract tests.

#![allow(clippy::unwrap_used)]

use crate::elf::constants::*;

/// Writes a little-endian 16-bit value into the ELF fixture.
pub(crate) fn write_u16(bytes: &mut [u8], offset: usize, value: u16) {
    bytes[offset..offset + 2].copy_from_slice(&value.to_le_bytes());
}

/// Writes a little-endian 32-bit value into the ELF fixture.
pub(crate) fn write_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Writes a little-endian 64-bit value into the ELF fixture.
pub(crate) fn write_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Writes a little-endian signed 64-bit ELF relocation value.
pub(crate) fn write_i64(bytes: &mut [u8], offset: usize, value: i64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

/// Builds a minimal GFX1201 code-object fixture.
pub(crate) fn gfx1201_code_object() -> Vec<u8> {
    /// Byte offset of the ELF section-header table.
    const SECTION_OFFSET: usize = 64;
    /// Number of ELF section headers in the fixture.
    const SECTION_COUNT: usize = 5;
    /// Byte offset of the ELF string table.
    const STRING_OFFSET: usize = SECTION_OFFSET + SECTION_COUNT * 64;
    /// Byte offset of the ELF symbol table.
    const SYMBOL_OFFSET: usize = 448;
    /// Byte offset of the executable text section.
    const TEXT_OFFSET: usize = 544;
    /// Byte offset of the fixture data section.
    const DATA_OFFSET: usize = 608;
    /// NUL-separated symbol names referenced by the fixture.
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
    write_u16(&mut bytes, 52, u16::try_from(ELF_HEADER_SIZE).unwrap());
    write_u16(&mut bytes, 54, 56);
    write_u16(&mut bytes, 58, 64);
    write_u16(&mut bytes, 60, u16::try_from(SECTION_COUNT).unwrap());

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
    bytes[kernel + 4] = (SYMBOL_BINDING_GLOBAL << 4) | SYMBOL_TYPE_OBJECT;
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
