// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Bounds-checked ELF64 views and AMDGPU load-image construction.
//!
//! This module reads caller-owned bytes and writes a caller-owned image.
//! Code-object policy, allocation, and device publication live in their owners.

#![forbid(unsafe_code)]

/// ELF or relocation failure, independent of any runtime ABI status code.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// An ELF field, table, range, or relocation is malformed or unsupported.
    InvalidObject,
    /// A dynamic relocation names an external symbol the caller cannot resolve.
    UndefinedSymbol,
}

/// Common ELF and AMDGPU code-object constants.
#[allow(missing_docs)]
pub mod constants {
    pub const ELF_HEADER_SIZE: usize = 64;
    pub const ELF_MACHINE_AMDGPU: u16 = 224;
    pub const ELF_OS_ABI_AMDGPU_HSA: u8 = 64;
    pub const ELF_TYPE_EXEC: u16 = 2;
    pub const ELF_TYPE_DYN: u16 = 3;
    pub const PROGRAM_LOAD: u32 = 1;
    pub const SECTION_PROGBITS: u32 = 1;
    pub const SECTION_SYMTAB: u32 = 2;
    pub const SECTION_STRTAB: u32 = 3;
    pub const SECTION_RELA: u32 = 4;
    pub const SECTION_DYNSYM: u32 = 11;
    pub const SECTION_FLAG_WRITE: u64 = 1;
    pub const SECTION_FLAG_AMDGPU_HSA_READONLY: u64 = 0x0020_0000;
    pub const SECTION_FLAG_AMDGPU_HSA_AGENT: u64 = 0x0080_0000;
    pub const SYMBOL_TYPE_OBJECT: u8 = 1;
    pub const SYMBOL_TYPE_COMMON: u8 = 5;
    pub const SYMBOL_TYPE_AMDGPU_HSA_KERNEL: u8 = 10;
    pub const SYMBOL_TYPE_AMDGPU_HSA_INDIRECT_FUNCTION: u8 = 11;
    pub const SYMBOL_BINDING_GLOBAL: u8 = 1;
    pub const SYMBOL_UNDEFINED: u16 = 0;
    pub const SYMBOL_ABSOLUTE: u16 = 0xfff1;
    pub const SYMBOL_COMMON: u16 = 0xfff2;
    pub const RELOCATION_ABSOLUTE_32_LO: u32 = 1;
    pub const RELOCATION_ABSOLUTE_32_HI: u32 = 2;
    pub const RELOCATION_ABSOLUTE_64: u32 = 3;
    pub const RELOCATION_ABSOLUTE_32: u32 = 6;
    pub const RELOCATION_RELATIVE_64: u32 = 13;
}

use constants::{
    ELF_HEADER_SIZE, PROGRAM_LOAD, RELOCATION_ABSOLUTE_32, RELOCATION_ABSOLUTE_32_HI,
    RELOCATION_ABSOLUTE_32_LO, RELOCATION_ABSOLUTE_64, RELOCATION_RELATIVE_64, SECTION_DYNSYM,
    SECTION_RELA, SECTION_STRTAB, SECTION_SYMTAB, SYMBOL_ABSOLUTE, SYMBOL_COMMON,
    SYMBOL_TYPE_COMMON, SYMBOL_UNDEFINED,
};

/// Validated ELF64 little-endian header and table bounds.
#[derive(Clone, Copy, Debug)]
pub struct Layout {
    /// ELF OS ABI.
    pub os_abi: u8,
    /// ELF ABI version byte.
    pub abi_version: u8,
    /// ELF object type.
    pub kind: u16,
    /// ELF machine identifier.
    pub machine: u16,
    /// Processor-specific ELF flags.
    pub flags: u32,
    /// Program-header table byte offset.
    pub program_offset: u64,
    /// Program-header entry size.
    pub program_entry_size: usize,
    /// Number of program headers.
    pub program_count: usize,
    /// Section-header table byte offset.
    pub section_offset: u64,
    /// Section-header entry size.
    pub section_entry_size: usize,
    /// Number of section headers.
    pub section_count: usize,
}

/// Reads one little-endian 16-bit field from a checked byte range.
#[must_use]
pub fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(offset..offset.checked_add(2)?)?.try_into().ok()?,
    ))
}

/// Reads one little-endian 32-bit field from a checked byte range.
#[must_use]
pub fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset.checked_add(4)?)?.try_into().ok()?,
    ))
}

/// Reads one little-endian 64-bit field from a checked byte range.
#[must_use]
pub fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?,
    ))
}

/// Reads one little-endian signed 64-bit field from a checked byte range.
#[must_use]
pub fn read_i64(bytes: &[u8], offset: usize) -> Option<i64> {
    Some(i64::from_le_bytes(
        bytes.get(offset..offset.checked_add(8)?)?.try_into().ok()?,
    ))
}

/// Returns one complete entry of a fixed-width table.
#[must_use]
pub fn table_entry(bytes: &[u8], base: u64, index: usize, size: usize) -> Option<&[u8]> {
    let offset = usize::try_from(base)
        .ok()?
        .checked_add(index.checked_mul(size)?)?;
    bytes.get(offset..offset.checked_add(size)?)
}

/// Returns a UTF-8 string terminated inside the supplied string table.
#[must_use]
pub fn c_string(bytes: &[u8], offset: usize) -> Option<&str> {
    let tail = bytes.get(offset..)?;
    let end = tail.iter().position(|byte| *byte == 0)?;
    std::str::from_utf8(&tail[..end]).ok()
}

/// Parses a little-endian ELF64 header and bounds-checks its fixed-width tables.
///
/// Machine and OS ABI selection belong to the caller.
///
/// # Errors
/// Returns an invalid-object error for an invalid header or out-of-bounds table.
pub fn parse_elf64(bytes: &[u8]) -> Result<Layout, Error> {
    if bytes.len() < ELF_HEADER_SIZE
        || bytes.get(0..4) != Some(b"\x7fELF")
        || bytes[4] != 2
        || bytes[5] != 1
        || read_u32(bytes, 20) != Some(1)
        || read_u16(bytes, 52).is_none_or(|size| usize::from(size) < ELF_HEADER_SIZE)
    {
        return Err(Error::InvalidObject);
    }
    let layout = Layout {
        os_abi: bytes[7],
        abi_version: bytes[8],
        kind: read_u16(bytes, 16).ok_or(Error::InvalidObject)?,
        machine: read_u16(bytes, 18).ok_or(Error::InvalidObject)?,
        flags: read_u32(bytes, 48).ok_or(Error::InvalidObject)?,
        program_offset: read_u64(bytes, 32).ok_or(Error::InvalidObject)?,
        section_offset: read_u64(bytes, 40).ok_or(Error::InvalidObject)?,
        program_entry_size: usize::from(read_u16(bytes, 54).ok_or(Error::InvalidObject)?),
        program_count: usize::from(read_u16(bytes, 56).ok_or(Error::InvalidObject)?),
        section_entry_size: usize::from(read_u16(bytes, 58).ok_or(Error::InvalidObject)?),
        section_count: usize::from(read_u16(bytes, 60).ok_or(Error::InvalidObject)?),
    };
    if (layout.program_count != 0 && layout.program_entry_size < 56)
        || (layout.section_count != 0 && layout.section_entry_size < 64)
        || (layout.program_count != 0
            && table_entry(
                bytes,
                layout.program_offset,
                layout.program_count - 1,
                layout.program_entry_size,
            )
            .is_none())
        || (layout.section_count != 0
            && table_entry(
                bytes,
                layout.section_offset,
                layout.section_count - 1,
                layout.section_entry_size,
            )
            .is_none())
    {
        return Err(Error::InvalidObject);
    }
    Ok(layout)
}

/// Rounds a value down to a nonzero power-of-two alignment.
#[must_use]
pub fn align_down(value: u64, alignment: u64) -> u64 {
    value & !(alignment - 1)
}

/// Rounds a value up to a nonzero power-of-two alignment.
#[must_use]
pub fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| align_down(value, alignment))
}

/// One file-backed load segment, before runtime allocation.
#[derive(Clone, Copy, Debug)]
pub struct LoadSegment {
    /// File byte offset.
    pub file_offset: u64,
    /// Virtual address from the ELF program header.
    pub virtual_address: u64,
    /// File-backed byte extent.
    pub file_size: u64,
    /// Total in-memory byte extent.
    pub memory_size: u64,
}

/// Collects valid load segments without allocating device memory.
///
/// # Errors
/// Returns an invalid-object error for a malformed segment or file range.
pub fn load_segments(bytes: &[u8], layout: Layout) -> Result<Vec<LoadSegment>, Error> {
    let mut loads = Vec::new();
    for index in 0..layout.program_count {
        let header = table_entry(
            bytes,
            layout.program_offset,
            index,
            layout.program_entry_size,
        )
        .ok_or(Error::InvalidObject)?;
        if read_u32(header, 0) != Some(PROGRAM_LOAD) {
            continue;
        }
        let segment = LoadSegment {
            file_offset: read_u64(header, 8).ok_or(Error::InvalidObject)?,
            virtual_address: read_u64(header, 16).ok_or(Error::InvalidObject)?,
            file_size: read_u64(header, 32).ok_or(Error::InvalidObject)?,
            memory_size: read_u64(header, 40).ok_or(Error::InvalidObject)?,
        };
        let file_end = segment
            .file_offset
            .checked_add(segment.file_size)
            .ok_or(Error::InvalidObject)?;
        if segment.file_size > segment.memory_size || file_end > bytes.len() as u64 {
            return Err(Error::InvalidObject);
        }
        segment
            .virtual_address
            .checked_add(segment.memory_size)
            .ok_or(Error::InvalidObject)?;
        loads.push(segment);
    }
    Ok(loads)
}

/// A validated plan for loading one ELF image into a caller-owned allocation.
///
/// The plan borrows its immutable code-object bytes and owns the validated
/// load-segment list. It has no runtime, device, or operating-system owner.
pub struct LoadPlan<'a> {
    /// Immutable source bytes borrowed through relocation.
    bytes: &'a [u8],
    /// Checked ELF header and table bounds.
    layout: Layout,
    /// Loadable segments validated against those source bytes.
    segments: Vec<LoadSegment>,
    /// Lowest aligned ELF virtual address in the loaded image.
    virtual_base: u64,
    /// End of the loaded ELF virtual interval.
    virtual_end: u64,
    /// Aligned destination extent required from the caller.
    allocation_size: usize,
}

impl<'a> LoadPlan<'a> {
    /// Validates load segments and computes the aligned destination extent.
    ///
    /// # Errors
    /// Returns an invalid-object error for malformed segments, an empty load
    /// image, an invalid alignment, or an extent too large for a Rust slice.
    pub fn new(bytes: &'a [u8], layout: Layout, alignment: u64) -> Result<Self, Error> {
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::InvalidObject);
        }
        let segments = load_segments(bytes, layout)?;
        let virtual_base = segments
            .iter()
            .map(|segment| align_down(segment.virtual_address, alignment))
            .min()
            .ok_or(Error::InvalidObject)?;
        let virtual_end = segments
            .iter()
            .try_fold(virtual_base, |end, segment| {
                segment
                    .virtual_address
                    .checked_add(segment.memory_size)
                    .map(|segment_end| end.max(segment_end))
            })
            .ok_or(Error::InvalidObject)?;
        if virtual_end <= virtual_base {
            return Err(Error::InvalidObject);
        }
        let aligned_end = align_up(virtual_end, alignment).ok_or(Error::InvalidObject)?;
        let allocation_size = usize::try_from(aligned_end - virtual_base)
            .ok()
            .filter(|size| isize::try_from(*size).is_ok())
            .ok_or(Error::InvalidObject)?;
        Ok(Self {
            bytes,
            layout,
            segments,
            virtual_base,
            virtual_end,
            allocation_size,
        })
    }

    /// Returns the first virtual byte covered by the destination allocation.
    #[must_use]
    pub fn virtual_base(&self) -> u64 {
        self.virtual_base
    }

    /// Returns the end of the loadable virtual span, before page alignment.
    #[must_use]
    pub fn virtual_end(&self) -> u64 {
        self.virtual_end
    }

    /// Returns the required size of the destination allocation in bytes.
    #[must_use]
    pub fn allocation_size(&self) -> usize {
        self.allocation_size
    }

    /// Returns the validated file-backed segments in program-header order.
    #[must_use]
    pub fn segments(&self) -> &[LoadSegment] {
        &self.segments
    }

    /// Clears a destination, copies its load segments, and applies AMDGPU
    /// dynamic relocations using the supplied device address and resolver.
    ///
    /// The caller must publish the destination only after this method succeeds.
    ///
    /// # Errors
    /// Returns an invalid-object error if the destination has the wrong size
    /// or a relocation is malformed, and an undefined-symbol error when the
    /// resolver cannot find a required external symbol.
    pub fn write_amdgpu_image(
        &self,
        device_base: u64,
        image: &mut [u8],
        resolve_external: impl Fn(&str) -> Option<u64>,
    ) -> Result<(), Error> {
        if image.len() != self.allocation_size {
            return Err(Error::InvalidObject);
        }
        image.fill(0);
        for segment in &self.segments {
            let source = usize::try_from(segment.file_offset).map_err(|_| Error::InvalidObject)?;
            let destination = usize::try_from(
                segment
                    .virtual_address
                    .checked_sub(self.virtual_base)
                    .ok_or(Error::InvalidObject)?,
            )
            .map_err(|_| Error::InvalidObject)?;
            let count = usize::try_from(segment.file_size).map_err(|_| Error::InvalidObject)?;
            let source = self
                .bytes
                .get(source..source.checked_add(count).ok_or(Error::InvalidObject)?)
                .ok_or(Error::InvalidObject)?;
            let destination = image
                .get_mut(destination..destination.checked_add(count).ok_or(Error::InvalidObject)?)
                .ok_or(Error::InvalidObject)?;
            destination.copy_from_slice(source);
        }
        apply_amdgpu_relocations(
            self.bytes,
            self.layout,
            self.virtual_base,
            device_base,
            image,
            resolve_external,
        )
    }
}

/// A checked ELF64 symbol table with its linked string table.
pub struct SymbolTable<'a> {
    /// Code-object bytes containing the symbol records.
    bytes: &'a [u8],
    /// File byte offset of the first record.
    offset: u64,
    /// Byte stride between symbol records.
    entry_size: usize,
    /// Number of symbol entries.
    pub count: usize,
    /// Linked ELF string table containing symbol names.
    strings: &'a [u8],
}

impl<'a> SymbolTable<'a> {
    /// Returns one complete symbol record.
    ///
    /// # Errors
    /// Returns an invalid-object error for an out-of-bounds entry.
    pub fn entry(&self, index: usize) -> Result<&'a [u8], Error> {
        table_entry(self.bytes, self.offset, index, self.entry_size).ok_or(Error::InvalidObject)
    }

    /// Returns the symbol name from the linked string table.
    ///
    /// # Errors
    /// Returns an invalid-object error for a malformed name.
    pub fn name(&self, symbol: &[u8]) -> Result<&'a str, Error> {
        let offset = usize::try_from(read_u32(symbol, 0).ok_or(Error::InvalidObject)?)
            .map_err(|_| Error::InvalidObject)?;
        c_string(self.strings, offset).ok_or(Error::InvalidObject)
    }
}

/// Resolves the symbol and string tables named by one section header.
///
/// # Errors
/// Returns an invalid-object error for an invalid link, stride, or byte range.
pub fn symbol_table<'a>(
    bytes: &'a [u8],
    layout: Layout,
    section: &[u8],
) -> Result<SymbolTable<'a>, Error> {
    let offset = read_u64(section, 24).ok_or(Error::InvalidObject)?;
    let size = read_u64(section, 32).ok_or(Error::InvalidObject)?;
    let strings_index = usize::try_from(read_u32(section, 40).ok_or(Error::InvalidObject)?)
        .map_err(|_| Error::InvalidObject)?;
    if strings_index >= layout.section_count {
        return Err(Error::InvalidObject);
    }
    let entry_size = usize::try_from(read_u64(section, 56).ok_or(Error::InvalidObject)?)
        .map_err(|_| Error::InvalidObject)?;
    if entry_size < 24 || size % entry_size as u64 != 0 {
        return Err(Error::InvalidObject);
    }
    let strings_section = table_entry(
        bytes,
        layout.section_offset,
        strings_index,
        layout.section_entry_size,
    )
    .filter(|section| read_u32(section, 4) == Some(SECTION_STRTAB))
    .ok_or(Error::InvalidObject)?;
    let strings_offset =
        usize::try_from(read_u64(strings_section, 24).ok_or(Error::InvalidObject)?)
            .map_err(|_| Error::InvalidObject)?;
    let strings_size = usize::try_from(read_u64(strings_section, 32).ok_or(Error::InvalidObject)?)
        .map_err(|_| Error::InvalidObject)?;
    let strings = bytes
        .get(
            strings_offset
                ..strings_offset
                    .checked_add(strings_size)
                    .ok_or(Error::InvalidObject)?,
        )
        .ok_or(Error::InvalidObject)?;
    let count = usize::try_from(size).map_err(|_| Error::InvalidObject)? / entry_size;
    if count != 0 && table_entry(bytes, offset, count - 1, entry_size).is_none() {
        return Err(Error::InvalidObject);
    }
    Ok(SymbolTable {
        bytes,
        offset,
        entry_size,
        count,
        strings,
    })
}

/// Symbol fields and optional kernel descriptor from an AMDGPU code object.
///
/// Runtime-specific symbol classification and handle allocation belong to the
/// caller. Names and descriptors borrow the immutable code-object bytes.
pub struct AmdgpuSymbol<'a> {
    /// Full symbol name from the linked string table.
    pub name: &'a str,
    /// ELF symbol type from the low nibble of `st_info`.
    pub symbol_type: u8,
    /// ELF symbol binding from the high nibble of `st_info`.
    pub binding: u8,
    /// Whether this entry names an AMDGPU kernel or kernel descriptor.
    pub is_kernel: bool,
    /// Section index from `st_shndx`.
    pub section_index: u16,
    /// Virtual value from `st_value`.
    pub value: u64,
    /// Size from `st_size`.
    pub size: u64,
    /// Flags of the containing section, or zero without one.
    pub section_flags: u64,
    /// Alignment of the containing section, or zero without one.
    pub alignment: u64,
    /// First 64 bytes of a file-backed kernel descriptor, when present.
    pub kernel_descriptor: Option<&'a [u8]>,
}

/// Reads named symbols from the first available preferred symbol table.
///
/// The caller supplies section types in priority order, for example dynamic
/// symbols before regular symbols for executable loads. Unnamed entries are
/// skipped. Any symbol type is returned for caller classification.
///
/// # Errors
/// Returns an invalid-object error for malformed symbol or string tables.
pub fn amdgpu_symbols<'a>(
    bytes: &'a [u8],
    layout: Layout,
    preferred_tables: &[u32],
) -> Result<Vec<AmdgpuSymbol<'a>>, Error> {
    let symbol_section = preferred_tables.iter().find_map(|preferred| {
        (0..layout.section_count).find_map(|index| {
            let section = table_entry(
                bytes,
                layout.section_offset,
                index,
                layout.section_entry_size,
            )?;
            (read_u32(section, 4) == Some(*preferred)).then_some(section)
        })
    });
    let Some(symbol_section) = symbol_section else {
        return Ok(Vec::new());
    };
    let table = symbol_table(bytes, layout, symbol_section)?;
    let mut symbols = Vec::new();
    for index in 0..table.count {
        let entry = table.entry(index)?;
        if read_u32(entry, 0) == Some(0) {
            continue;
        }
        let name = table.name(entry)?;
        let info = *entry.get(4).ok_or(Error::InvalidObject)?;
        let symbol_type = info & 0xf;
        let section_index = read_u16(entry, 6).ok_or(Error::InvalidObject)?;
        let value = read_u64(entry, 8).ok_or(Error::InvalidObject)?;
        let size = read_u64(entry, 16).ok_or(Error::InvalidObject)?;
        let section = (usize::from(section_index) < layout.section_count)
            .then(|| {
                table_entry(
                    bytes,
                    layout.section_offset,
                    usize::from(section_index),
                    layout.section_entry_size,
                )
            })
            .flatten();
        let section_flags = section
            .and_then(|section| read_u64(section, 8))
            .unwrap_or(0);
        let alignment = section
            .and_then(|section| read_u64(section, 48))
            .unwrap_or(0);
        let is_kernel = symbol_type == constants::SYMBOL_TYPE_AMDGPU_HSA_KERNEL
            || name
                .rsplit_once('.')
                .is_some_and(|(_, extension)| extension.eq_ignore_ascii_case("kd"));
        let kernel_descriptor = if is_kernel
            && section_index != SYMBOL_UNDEFINED
            && section_index != SYMBOL_COMMON
            && symbol_type != SYMBOL_TYPE_COMMON
        {
            section.and_then(|section| {
                if read_u32(section, 4) != Some(constants::SECTION_PROGBITS) {
                    return None;
                }
                let section_address = read_u64(section, 16)?;
                let section_offset = usize::try_from(read_u64(section, 24)?).ok()?;
                let offset = usize::try_from(value.checked_sub(section_address)?).ok()?;
                let section_size = usize::try_from(read_u64(section, 32)?).ok()?;
                if offset.checked_add(64)? > section_size {
                    return None;
                }
                bytes.get(section_offset.checked_add(offset)?..)?.get(..64)
            })
        } else {
            None
        };
        symbols.push(AmdgpuSymbol {
            name,
            symbol_type,
            binding: info >> 4,
            is_kernel,
            section_index,
            value,
            size,
            section_flags,
            alignment,
            kernel_descriptor,
        });
    }
    Ok(symbols)
}

fn add_relocation_base(base: i128, value: u64, addend: i64) -> Option<u64> {
    let result = base + i128::from(value) + i128::from(addend);
    u64::try_from(result).ok()
}

/// Applies supported AMDGPU RELA dynamic relocations to a caller-owned image.
///
/// The image must cover the complete load-segment virtual span beginning at
/// the supplied virtual base and be published only after this call succeeds.
///
/// # Errors
/// Returns an invalid-object error for malformed or unsupported relocations and
/// an undefined-symbol error when the resolver cannot find a required symbol.
#[allow(clippy::too_many_lines)]
pub fn apply_amdgpu_relocations(
    bytes: &[u8],
    layout: Layout,
    virtual_base: u64,
    device_base: u64,
    image: &mut [u8],
    resolve_external: impl Fn(&str) -> Option<u64>,
) -> Result<(), Error> {
    // An application-owned destination may be below the ELF virtual base.
    // Keep the bias signed until the complete relocated address is checked.
    let load_bias = i128::from(device_base) - i128::from(virtual_base);
    for section_index in 0..layout.section_count {
        let section = table_entry(
            bytes,
            layout.section_offset,
            section_index,
            layout.section_entry_size,
        )
        .ok_or(Error::InvalidObject)?;
        if read_u32(section, 4) != Some(SECTION_RELA) {
            continue;
        }
        let target_section = usize::try_from(read_u32(section, 44).ok_or(Error::InvalidObject)?)
            .map_err(|_| Error::InvalidObject)?;
        if target_section != 0 {
            if target_section >= layout.section_count {
                return Err(Error::InvalidObject);
            }
            // Section-targeted relocations have already been applied by the linker.
            continue;
        }
        let relocation_offset = read_u64(section, 24).ok_or(Error::InvalidObject)?;
        let relocation_size = read_u64(section, 32).ok_or(Error::InvalidObject)?;
        let symbol_section_index =
            usize::try_from(read_u32(section, 40).ok_or(Error::InvalidObject)?)
                .map_err(|_| Error::InvalidObject)?;
        if symbol_section_index >= layout.section_count {
            return Err(Error::InvalidObject);
        }
        let relocation_entry_size =
            usize::try_from(read_u64(section, 56).ok_or(Error::InvalidObject)?)
                .map_err(|_| Error::InvalidObject)?;
        if relocation_entry_size < 24 || relocation_size % relocation_entry_size as u64 != 0 {
            return Err(Error::InvalidObject);
        }
        let symbol_section = table_entry(
            bytes,
            layout.section_offset,
            symbol_section_index,
            layout.section_entry_size,
        )
        .ok_or(Error::InvalidObject)?;
        if !matches!(
            read_u32(symbol_section, 4),
            Some(SECTION_SYMTAB | SECTION_DYNSYM)
        ) {
            return Err(Error::InvalidObject);
        }
        let symbols = symbol_table(bytes, layout, symbol_section)?;
        let relocation_count = usize::try_from(relocation_size)
            .map_err(|_| Error::InvalidObject)?
            / relocation_entry_size;

        for relocation_index in 0..relocation_count {
            let relocation = table_entry(
                bytes,
                relocation_offset,
                relocation_index,
                relocation_entry_size,
            )
            .ok_or(Error::InvalidObject)?;
            let target = read_u64(relocation, 0).ok_or(Error::InvalidObject)?;
            let info = read_u64(relocation, 8).ok_or(Error::InvalidObject)?;
            let addend = read_i64(relocation, 16).ok_or(Error::InvalidObject)?;
            let relocation_type =
                u32::try_from(info & u64::from(u32::MAX)).map_err(|_| Error::InvalidObject)?;
            let symbol_index = usize::try_from(info >> 32).map_err(|_| Error::InvalidObject)?;
            let value = match relocation_type {
                RELOCATION_RELATIVE_64 => add_relocation_base(load_bias, 0, addend),
                RELOCATION_ABSOLUTE_32_LO
                | RELOCATION_ABSOLUTE_32_HI
                | RELOCATION_ABSOLUTE_64
                | RELOCATION_ABSOLUTE_32 => {
                    if symbol_index >= symbols.count {
                        return Err(Error::InvalidObject);
                    }
                    let symbol = symbols.entry(symbol_index)?;
                    let section_index = read_u16(symbol, 6).ok_or(Error::InvalidObject)?;
                    let symbol_value = read_u64(symbol, 8).ok_or(Error::InvalidObject)?;
                    let symbol_type = *symbol.get(4).ok_or(Error::InvalidObject)? & 0xf;
                    let (base, value) = if section_index == SYMBOL_UNDEFINED
                        || section_index == SYMBOL_COMMON
                        || symbol_type == SYMBOL_TYPE_COMMON
                    {
                        let name = symbols.name(symbol)?;
                        (0, resolve_external(name).ok_or(Error::UndefinedSymbol)?)
                    } else if section_index == SYMBOL_ABSOLUTE {
                        (0, symbol_value)
                    } else if usize::from(section_index) < layout.section_count {
                        (load_bias, symbol_value)
                    } else {
                        return Err(Error::InvalidObject);
                    };
                    add_relocation_base(base, value, addend)
                }
                _ => return Err(Error::InvalidObject),
            }
            .ok_or(Error::InvalidObject)?;
            let target_offset = usize::try_from(
                target
                    .checked_sub(virtual_base)
                    .ok_or(Error::InvalidObject)?,
            )
            .map_err(|_| Error::InvalidObject)?;
            let width = if relocation_type == RELOCATION_ABSOLUTE_64
                || relocation_type == RELOCATION_RELATIVE_64
            {
                8
            } else {
                4
            };
            let destination = image
                .get_mut(
                    target_offset
                        ..target_offset
                            .checked_add(width)
                            .ok_or(Error::InvalidObject)?,
                )
                .ok_or(Error::InvalidObject)?;
            if width == 8 {
                destination.copy_from_slice(&value.to_le_bytes());
            } else {
                let word = if relocation_type == RELOCATION_ABSOLUTE_32_HI {
                    value >> 32
                } else {
                    value & u64::from(u32::MAX)
                };
                let word = u32::try_from(word).map_err(|_| Error::InvalidObject)?;
                destination.copy_from_slice(&word.to_le_bytes());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
