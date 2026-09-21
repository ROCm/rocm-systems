//! HSA code-object readers, ELF validation, relocation, and executable state.
//!
//! Readers first take an owned snapshot of caller memory or a bounded file
//! range. Parsing uses checked offsets throughout so malformed ELF data cannot
//! escape its snapshot. Loading allocates rocddi memory, copies loadable bytes,
//! applies the supported AMDGPU relocations, and publishes symbols only after
//! the complete object is valid. Executable freeze and destruction retain the
//! HSA-visible ownership and failure semantics around that native backing.
//!
//! The current file-reader path uses Linux descriptors. That operating-system
//! detail is intentionally isolated here and must become a platform adapter
//! before this frontend can be built as a Windows HSA implementation.

use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};
use std::fs::File;
use std::mem::ManuallyDrop;
use std::os::fd::FromRawFd;
use std::os::unix::fs::FileExt;
use std::sync::Arc;
use std::sync::atomic::{Ordering, fence};

use rocddi::memory::{Allocation, DeviceAccess, MemoryKind};

use crate::ffi::*;
use crate::runtime::{Runtime, boundary, initialized_mut, lock, map_error};

const ELF_HEADER_SIZE: usize = 64;
const ELF_MACHINE_AMDGPU: u16 = 224;
const ELF_OS_ABI_AMDGPU_HSA: u8 = 64;
const ELF_TYPE_EXEC: u16 = 2;
const ELF_TYPE_DYN: u16 = 3;
const PROGRAM_LOAD: u32 = 1;
const SECTION_PROGBITS: u32 = 1;
const SECTION_SYMTAB: u32 = 2;
const SECTION_STRTAB: u32 = 3;
const SECTION_RELA: u32 = 4;
const SECTION_DYNSYM: u32 = 11;
const SECTION_FLAG_WRITE: u64 = 1;
const SECTION_FLAG_AMDGPU_HSA_READONLY: u64 = 0x0020_0000;
const SECTION_FLAG_AMDGPU_HSA_AGENT: u64 = 0x0080_0000;
const SYMBOL_TYPE_OBJECT: u8 = 1;
const SYMBOL_TYPE_COMMON: u8 = 5;
const SYMBOL_TYPE_AMDGPU_HSA_KERNEL: u8 = 10;
const SYMBOL_TYPE_AMDGPU_HSA_INDIRECT_FUNCTION: u8 = 11;
const SYMBOL_BINDING_GLOBAL: u8 = 1;
const SYMBOL_UNDEFINED: u16 = 0;
const SYMBOL_ABSOLUTE: u16 = 0xfff1;
const RELOCATION_ABSOLUTE_64: u32 = 3;
const RELOCATION_RELATIVE_64: u32 = 13;
const AMDGPU_MACHINE_MASK: u32 = 0xff;
const AMDGPU_MACHINE_GFX1201: u32 = 0x4e;
const AMDGPU_MACHINE_GFX12_GENERIC: u32 = 0x59;

/// Origin retained for loader queries after bytes have been snapshotted.
#[derive(Clone, Copy)]
enum ReaderStorage {
    Memory,
    File { descriptor: i32, offset: usize },
}

/// Immutable code-object byte snapshot associated with one reader handle.
pub(crate) struct Reader {
    bytes: Arc<[u8]>,
    storage: ReaderStorage,
}

/// Parsed code-object metadata cached independently of any executable load.
pub(crate) struct CodeObject {
    bytes: Arc<[u8]>,
    names: HashMap<String, u64>,
    symbol_handles: Vec<u64>,
    symbols_initialized: bool,
}

/// HSA-visible symbol facts belonging to one parsed code object.
pub(crate) struct CodeSymbol {
    code_object: u64,
    name: String,
    module_name: String,
    kind: u32,
    linkage: u32,
    is_definition: bool,
    allocation: u32,
    segment: u32,
    alignment: u32,
    size: u32,
    is_const: bool,
    kernarg_size: u32,
    kernarg_alignment: u32,
    group_size: u32,
    private_size: u32,
    dynamic_callstack: bool,
    call_convention: u32,
    wavefront_size: u32,
}

/// One contiguous loaded address interval described to loader clients.
struct LoadedSegment {
    storage_offset: usize,
    address: u64,
    size: usize,
}

fn loaded_segments(
    loads: &[(u64, u64, u64, u64)],
    device_base: u64,
    virtual_base: u64,
) -> Result<Vec<LoadedSegment>, Status> {
    let (file_offset, _, _, _) = loads
        .iter()
        .min_by_key(|(_, virtual_address, _, _)| *virtual_address)
        .ok_or(INVALID_CODE_OBJECT)?;
    let virtual_end = loads
        .iter()
        .try_fold(virtual_base, |end, (_, virtual_address, _, memory_size)| {
            virtual_address
                .checked_add(*memory_size)
                .map(|segment_end| end.max(segment_end))
        })
        .ok_or(INVALID_CODE_OBJECT)?;
    let size = usize::try_from(
        virtual_end
            .checked_sub(virtual_base)
            .ok_or(INVALID_CODE_OBJECT)?,
    )
    .map_err(|_| INVALID_CODE_OBJECT)?;
    Ok(vec![LoadedSegment {
        storage_offset: usize::try_from(*file_offset).map_err(|_| INVALID_CODE_OBJECT)?,
        address: device_base,
        size,
    }])
}

/// Native allocation and relocation result for one executable load operation.
pub(crate) struct LoadedObject {
    pub(crate) handle: u64,
    _allocation: Allocation,
    host_base: usize,
    device_base: u64,
    size: u64,
    virtual_base: u64,
    agent: HsaAgent,
    kind: u32,
    code_object: Arc<[u8]>,
    uri: Vec<u8>,
    segments: Vec<LoadedSegment>,
}

/// Mutable HSA executable assembled from loaded objects and symbol definitions.
///
/// Symbol maps are completed before `frozen` is set. Once frozen, no operation
/// may change load addresses or definitions observed through public queries.
pub(crate) struct Executable {
    pub(crate) profile: u32,
    pub(crate) default_rounding: u32,
    pub(crate) frozen: bool,
    pub(crate) program_loaded: bool,
    pub(crate) loaded: Vec<LoadedObject>,
    pub(crate) names: HashMap<(String, u64), u64>,
    pub(crate) symbol_handles: Vec<u64>,
}

/// HSA-visible executable symbol with its resolved runtime address.
pub(crate) struct Symbol {
    pub(crate) executable: u64,
    pub(crate) name: String,
    pub(crate) module_name: String,
    pub(crate) agent: HsaAgent,
    pub(crate) address: u64,
    pub(crate) size: u32,
    pub(crate) kind: u32,
    pub(crate) linkage: u32,
    pub(crate) is_definition: bool,
    pub(crate) allocation: u32,
    pub(crate) segment: u32,
    pub(crate) alignment: u32,
    pub(crate) is_const: bool,
    pub(crate) kernarg_size: u32,
    pub(crate) kernarg_alignment: u32,
    pub(crate) group_size: u32,
    pub(crate) private_size: u32,
    pub(crate) dynamic_callstack: bool,
    pub(crate) call_convention: u32,
}

/// Validated symbol staged during parsing before it receives a public handle.
struct PendingSymbol {
    full_name: String,
    name: String,
    module_name: String,
    value: u64,
    address: u64,
    size: u32,
    kind: u32,
    linkage: u32,
    is_definition: bool,
    allocation: u32,
    segment: u32,
    alignment: u32,
    is_const: bool,
    kernarg_size: u32,
    kernarg_alignment: u32,
    group_size: u32,
    private_size: u32,
    dynamic_callstack: bool,
    call_convention: u32,
    wavefront_size: u32,
}

/// Bounds-checked view of the ELF tables needed by the supported loader path.
#[derive(Clone, Copy)]
struct ElfLayout {
    abi_version: u8,
    flags: u32,
    program_offset: u64,
    program_entry_size: usize,
    program_count: usize,
    section_offset: u64,
    section_entry_size: usize,
    section_count: usize,
}

fn read_u16(bytes: &[u8], offset: usize) -> Option<u16> {
    Some(u16::from_le_bytes(
        bytes.get(offset..offset + 2)?.try_into().ok()?,
    ))
}

fn read_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    Some(u32::from_le_bytes(
        bytes.get(offset..offset + 4)?.try_into().ok()?,
    ))
}

fn read_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    Some(u64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

fn read_i64(bytes: &[u8], offset: usize) -> Option<i64> {
    Some(i64::from_le_bytes(
        bytes.get(offset..offset + 8)?.try_into().ok()?,
    ))
}

fn parse_elf_layout(bytes: &[u8]) -> Result<ElfLayout, Status> {
    if bytes.len() < ELF_HEADER_SIZE
        || bytes.get(0..4) != Some(b"\x7fELF")
        || bytes[4] != 2
        || bytes[5] != 1
        || bytes[7] != ELF_OS_ABI_AMDGPU_HSA
        || !matches!(read_u16(bytes, 16), Some(ELF_TYPE_EXEC | ELF_TYPE_DYN))
        || read_u16(bytes, 18) != Some(ELF_MACHINE_AMDGPU)
        || read_u32(bytes, 20) != Some(1)
        || read_u16(bytes, 52).is_none_or(|size| usize::from(size) < ELF_HEADER_SIZE)
    {
        return Err(INVALID_CODE_OBJECT);
    }
    let layout = ElfLayout {
        abi_version: bytes[8],
        flags: read_u32(bytes, 48).ok_or(INVALID_CODE_OBJECT)?,
        program_offset: read_u64(bytes, 32).ok_or(INVALID_CODE_OBJECT)?,
        section_offset: read_u64(bytes, 40).ok_or(INVALID_CODE_OBJECT)?,
        program_entry_size: usize::from(read_u16(bytes, 54).ok_or(INVALID_CODE_OBJECT)?),
        program_count: usize::from(read_u16(bytes, 56).ok_or(INVALID_CODE_OBJECT)?),
        section_entry_size: usize::from(read_u16(bytes, 58).ok_or(INVALID_CODE_OBJECT)?),
        section_count: usize::from(read_u16(bytes, 60).ok_or(INVALID_CODE_OBJECT)?),
    };
    if (layout.program_count != 0 && layout.program_entry_size < 56)
        || (layout.section_count != 0 && layout.section_entry_size < 64)
    {
        return Err(INVALID_CODE_OBJECT);
    }
    if layout.program_count != 0
        && table_entry(
            bytes,
            layout.program_offset,
            layout.program_count - 1,
            layout.program_entry_size,
        )
        .is_none()
    {
        return Err(INVALID_CODE_OBJECT);
    }
    if layout.section_count != 0
        && table_entry(
            bytes,
            layout.section_offset,
            layout.section_count - 1,
            layout.section_entry_size,
        )
        .is_none()
    {
        return Err(INVALID_CODE_OBJECT);
    }
    Ok(layout)
}

fn code_object_version(layout: ElfLayout) -> Result<u32, Status> {
    match layout.abi_version {
        0 => Ok(2),
        1 => Ok(3),
        2 => Ok(4),
        3 => Ok(5),
        4 => Ok(6),
        _ => Err(INVALID_CODE_OBJECT),
    }
}

fn code_object_target(machine: u32) -> Option<(&'static str, u64)> {
    match machine & AMDGPU_MACHINE_MASK {
        AMDGPU_MACHINE_GFX1201 => Some(("gfx1201", 0)),
        AMDGPU_MACHINE_GFX12_GENERIC => Some(("gfx12-generic", 1)),
        _ => None,
    }
}

fn split_symbol_name(full_name: &str) -> (&str, &str) {
    full_name.rfind("::").map_or(("", full_name), |separator| {
        (&full_name[..separator], &full_name[separator + 2..])
    })
}

fn variable_metadata(section_flags: u64) -> (u32, bool) {
    let segment = if section_flags & SECTION_FLAG_AMDGPU_HSA_READONLY != 0 {
        VARIABLE_SEGMENT_READONLY
    } else {
        VARIABLE_SEGMENT_GLOBAL
    };
    (segment, section_flags & SECTION_FLAG_WRITE != 0)
}

fn parse_symbols(
    bytes: &[u8],
    layout: ElfLayout,
    preferred_tables: &[u32],
) -> Result<Vec<PendingSymbol>, Status> {
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
    let symbol_offset = read_u64(symbol_section, 24).ok_or(INVALID_CODE_OBJECT)?;
    let symbol_size = read_u64(symbol_section, 32).ok_or(INVALID_CODE_OBJECT)?;
    let strings_index = usize::try_from(read_u32(symbol_section, 40).ok_or(INVALID_CODE_OBJECT)?)
        .map_err(|_| INVALID_CODE_OBJECT)?;
    let entry_size = usize::try_from(read_u64(symbol_section, 56).ok_or(INVALID_CODE_OBJECT)?)
        .map_err(|_| INVALID_CODE_OBJECT)?;
    if entry_size < 24 || symbol_size % entry_size as u64 != 0 {
        return Err(INVALID_CODE_OBJECT);
    }
    let strings = table_entry(
        bytes,
        layout.section_offset,
        strings_index,
        layout.section_entry_size,
    )
    .filter(|section| read_u32(section, 4) == Some(SECTION_STRTAB))
    .ok_or(INVALID_CODE_OBJECT)?;
    let strings_offset = usize::try_from(read_u64(strings, 24).ok_or(INVALID_CODE_OBJECT)?)
        .map_err(|_| INVALID_CODE_OBJECT)?;
    let strings_size = usize::try_from(read_u64(strings, 32).ok_or(INVALID_CODE_OBJECT)?)
        .map_err(|_| INVALID_CODE_OBJECT)?;
    let string_table = bytes
        .get(
            strings_offset
                ..strings_offset
                    .checked_add(strings_size)
                    .ok_or(INVALID_CODE_OBJECT)?,
        )
        .ok_or(INVALID_CODE_OBJECT)?;
    let count = usize::try_from(symbol_size).map_err(|_| INVALID_CODE_OBJECT)? / entry_size;
    let wavefront_size = if matches!(
        layout.flags & AMDGPU_MACHINE_MASK,
        AMDGPU_MACHINE_GFX1201 | AMDGPU_MACHINE_GFX12_GENERIC
    ) {
        32
    } else {
        64
    };
    let mut symbols = Vec::new();
    for symbol_index in 0..count {
        let symbol = table_entry(bytes, symbol_offset, symbol_index, entry_size)
            .ok_or(INVALID_CODE_OBJECT)?;
        let name_offset = usize::try_from(read_u32(symbol, 0).ok_or(INVALID_CODE_OBJECT)?)
            .map_err(|_| INVALID_CODE_OBJECT)?;
        if name_offset == 0 {
            continue;
        }
        let full_name = c_string(string_table, name_offset)
            .ok_or(INVALID_CODE_OBJECT)?
            .to_owned();
        let symbol_info = *symbol.get(4).ok_or(INVALID_CODE_OBJECT)?;
        let symbol_type = symbol_info & 0xf;
        let kind = if symbol_type == SYMBOL_TYPE_AMDGPU_HSA_KERNEL
            || std::path::Path::new(&full_name)
                .extension()
                .is_some_and(|extension| extension.eq_ignore_ascii_case("kd"))
        {
            SYMBOL_KIND_KERNEL
        } else if symbol_type == SYMBOL_TYPE_AMDGPU_HSA_INDIRECT_FUNCTION {
            SYMBOL_KIND_INDIRECT_FUNCTION
        } else if matches!(symbol_type, SYMBOL_TYPE_OBJECT | SYMBOL_TYPE_COMMON) {
            SYMBOL_KIND_VARIABLE
        } else {
            continue;
        };
        let section_index = read_u16(symbol, 6).ok_or(INVALID_CODE_OBJECT)?;
        let value = read_u64(symbol, 8).ok_or(INVALID_CODE_OBJECT)?;
        let size =
            u32::try_from(read_u64(symbol, 16).ok_or(INVALID_CODE_OBJECT)?).unwrap_or(u32::MAX);
        let is_definition = section_index != SYMBOL_UNDEFINED && symbol_type != SYMBOL_TYPE_COMMON;
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
        let section_flags = section.and_then(|entry| read_u64(entry, 8)).unwrap_or(0);
        let alignment = section
            .and_then(|entry| read_u64(entry, 48))
            .and_then(|alignment| u32::try_from(alignment).ok())
            .unwrap_or(0);
        let (segment, is_const) = variable_metadata(section_flags);
        let descriptor = if kind == SYMBOL_KIND_KERNEL && is_definition {
            section.and_then(|entry| {
                if read_u32(entry, 4) != Some(SECTION_PROGBITS) {
                    return None;
                }
                let section_address = read_u64(entry, 16)?;
                let section_offset = usize::try_from(read_u64(entry, 24)?).ok()?;
                let offset = usize::try_from(value.checked_sub(section_address)?).ok()?;
                bytes.get(section_offset.checked_add(offset)?..)?.get(..64)
            })
        } else {
            None
        };
        let (group_size, private_size, kernarg_size, dynamic_callstack) =
            descriptor.map_or((0, 0, 0, false), |descriptor| {
                (
                    read_u32(descriptor, 0).unwrap_or(0),
                    read_u32(descriptor, 4).unwrap_or(0),
                    read_u32(descriptor, 8).unwrap_or(0),
                    read_u16(descriptor, 56).is_some_and(|flags| flags & (1 << 11) != 0),
                )
            });
        let (module_name, name) = split_symbol_name(&full_name);
        symbols.push(PendingSymbol {
            full_name: full_name.clone(),
            name: name.to_owned(),
            module_name: module_name.to_owned(),
            value,
            address: 0,
            size,
            kind,
            linkage: if symbol_info >> 4 == SYMBOL_BINDING_GLOBAL {
                SYMBOL_LINKAGE_PROGRAM
            } else {
                SYMBOL_LINKAGE_MODULE
            },
            is_definition,
            allocation: if section_flags & SECTION_FLAG_AMDGPU_HSA_AGENT != 0 {
                VARIABLE_ALLOCATION_AGENT
            } else {
                VARIABLE_ALLOCATION_PROGRAM
            },
            segment,
            alignment,
            is_const,
            kernarg_size,
            kernarg_alignment: 16,
            group_size,
            private_size,
            dynamic_callstack,
            call_convention: 0,
            wavefront_size,
        });
    }
    Ok(symbols)
}

fn align_down(value: u64, alignment: u64) -> u64 {
    value & !(alignment - 1)
}

fn align_up(value: u64, alignment: u64) -> Option<u64> {
    value
        .checked_add(alignment - 1)
        .map(|value| align_down(value, alignment))
}

fn table_entry(bytes: &[u8], base: u64, index: usize, size: usize) -> Option<&[u8]> {
    let offset = usize::try_from(base)
        .ok()?
        .checked_add(index.checked_mul(size)?)?;
    bytes.get(offset..offset.checked_add(size)?)
}

fn c_string(bytes: &[u8], offset: usize) -> Option<&str> {
    let tail = bytes.get(offset..)?;
    let end = tail.iter().position(|byte| *byte == 0)?;
    std::str::from_utf8(&tail[..end]).ok()
}

fn add_relocation_base(base: u64, value: u64, addend: i64) -> Option<u64> {
    let result = i128::from(base) + i128::from(value) + i128::from(addend);
    u64::try_from(result).ok()
}

fn apply_relocations(
    bytes: &[u8],
    section_offset: u64,
    section_entry_size: usize,
    section_count: usize,
    virtual_base: u64,
    device_base: u64,
    image: &mut [u8],
) -> Result<(), Status> {
    let load_bias = device_base
        .checked_sub(virtual_base)
        .ok_or(INVALID_CODE_OBJECT)?;
    for section_index in 0..section_count {
        let section = table_entry(bytes, section_offset, section_index, section_entry_size)
            .ok_or(INVALID_CODE_OBJECT)?;
        if read_u32(section, 4) != Some(SECTION_RELA) {
            continue;
        }
        let relocation_offset = read_u64(section, 24).ok_or(INVALID_CODE_OBJECT)?;
        let relocation_size = read_u64(section, 32).ok_or(INVALID_CODE_OBJECT)?;
        let symbol_section_index =
            usize::try_from(read_u32(section, 40).ok_or(INVALID_CODE_OBJECT)?)
                .map_err(|_| INVALID_CODE_OBJECT)?;
        let relocation_entry_size =
            usize::try_from(read_u64(section, 56).ok_or(INVALID_CODE_OBJECT)?)
                .map_err(|_| INVALID_CODE_OBJECT)?;
        if relocation_entry_size < 24 || relocation_size % relocation_entry_size as u64 != 0 {
            return Err(INVALID_CODE_OBJECT);
        }

        let symbol_section = table_entry(
            bytes,
            section_offset,
            symbol_section_index,
            section_entry_size,
        )
        .ok_or(INVALID_CODE_OBJECT)?;
        if !matches!(
            read_u32(symbol_section, 4),
            Some(SECTION_SYMTAB | SECTION_DYNSYM)
        ) {
            return Err(INVALID_CODE_OBJECT);
        }
        let symbol_offset = read_u64(symbol_section, 24).ok_or(INVALID_CODE_OBJECT)?;
        let symbol_size = read_u64(symbol_section, 32).ok_or(INVALID_CODE_OBJECT)?;
        let symbol_entry_size =
            usize::try_from(read_u64(symbol_section, 56).ok_or(INVALID_CODE_OBJECT)?)
                .map_err(|_| INVALID_CODE_OBJECT)?;
        if symbol_entry_size < 24 || symbol_size % symbol_entry_size as u64 != 0 {
            return Err(INVALID_CODE_OBJECT);
        }
        let symbol_count =
            usize::try_from(symbol_size).map_err(|_| INVALID_CODE_OBJECT)? / symbol_entry_size;
        let relocation_count = usize::try_from(relocation_size).map_err(|_| INVALID_CODE_OBJECT)?
            / relocation_entry_size;

        for relocation_index in 0..relocation_count {
            let relocation = table_entry(
                bytes,
                relocation_offset,
                relocation_index,
                relocation_entry_size,
            )
            .ok_or(INVALID_CODE_OBJECT)?;
            let target = read_u64(relocation, 0).ok_or(INVALID_CODE_OBJECT)?;
            let info = read_u64(relocation, 8).ok_or(INVALID_CODE_OBJECT)?;
            let addend = read_i64(relocation, 16).ok_or(INVALID_CODE_OBJECT)?;
            let relocation_type = info as u32;
            let symbol_index = usize::try_from(info >> 32).map_err(|_| INVALID_CODE_OBJECT)?;
            let value = match relocation_type {
                RELOCATION_RELATIVE_64 => add_relocation_base(load_bias, 0, addend),
                RELOCATION_ABSOLUTE_64 => {
                    if symbol_index >= symbol_count {
                        return Err(INVALID_CODE_OBJECT);
                    }
                    let symbol = table_entry(bytes, symbol_offset, symbol_index, symbol_entry_size)
                        .ok_or(INVALID_CODE_OBJECT)?;
                    let section_index = read_u16(symbol, 6).ok_or(INVALID_CODE_OBJECT)?;
                    let symbol_value = read_u64(symbol, 8).ok_or(INVALID_CODE_OBJECT)?;
                    if section_index == SYMBOL_UNDEFINED {
                        return Err(INVALID_CODE_OBJECT);
                    }
                    add_relocation_base(
                        if section_index == SYMBOL_ABSOLUTE {
                            0
                        } else {
                            load_bias
                        },
                        symbol_value,
                        addend,
                    )
                }
                _ => return Err(INVALID_CODE_OBJECT),
            }
            .ok_or(INVALID_CODE_OBJECT)?;
            let target_offset = usize::try_from(
                target
                    .checked_sub(virtual_base)
                    .ok_or(INVALID_CODE_OBJECT)?,
            )
            .map_err(|_| INVALID_CODE_OBJECT)?;
            let destination = image
                .get_mut(target_offset..target_offset.checked_add(8).ok_or(INVALID_CODE_OBJECT)?)
                .ok_or(INVALID_CODE_OBJECT)?;
            destination.copy_from_slice(&value.to_le_bytes());
        }
    }
    Ok(())
}

fn parse_and_load(
    runtime: &Runtime,
    agent: HsaAgent,
    code_object: Arc<[u8]>,
    storage: ReaderStorage,
) -> Result<(LoadedObject, Vec<PendingSymbol>), Status> {
    let bytes = code_object.as_ref();
    let layout = parse_elf_layout(bytes)?;
    let _version = code_object_version(layout)?;
    let (gpu_index, memory_kind, object_kind) = if agent.handle == 0 {
        if runtime.gpus.is_empty() {
            return Err(OUT_OF_RESOURCES);
        }
        (0, MemoryKind::System, LOADER_OBJECT_KIND_PROGRAM)
    } else {
        let Some(gpu_index) = runtime.gpu_index(agent) else {
            return Err(INVALID_AGENT);
        };
        (
            gpu_index,
            MemoryKind::DeviceLocal {
                host_visible: true,
                coherent: false,
                uncached: false,
                contiguous: false,
            },
            LOADER_OBJECT_KIND_AGENT,
        )
    };
    let mut loads = Vec::new();
    let mut virtual_base = u64::MAX;
    let mut virtual_end = 0_u64;
    for index in 0..layout.program_count {
        let header = table_entry(
            bytes,
            layout.program_offset,
            index,
            layout.program_entry_size,
        )
        .ok_or(INVALID_CODE_OBJECT)?;
        if read_u32(header, 0) != Some(PROGRAM_LOAD) {
            continue;
        }
        let file_offset = read_u64(header, 8).ok_or(INVALID_CODE_OBJECT)?;
        let virtual_address = read_u64(header, 16).ok_or(INVALID_CODE_OBJECT)?;
        let file_size = read_u64(header, 32).ok_or(INVALID_CODE_OBJECT)?;
        let memory_size = read_u64(header, 40).ok_or(INVALID_CODE_OBJECT)?;
        if file_size > memory_size
            || usize::try_from(file_offset)
                .ok()
                .and_then(|offset| offset.checked_add(usize::try_from(file_size).ok()?))
                .is_none_or(|end| end > bytes.len())
        {
            return Err(INVALID_CODE_OBJECT);
        }
        virtual_base = virtual_base.min(align_down(virtual_address, 4096));
        virtual_end = virtual_end.max(
            virtual_address
                .checked_add(memory_size)
                .ok_or(INVALID_CODE_OBJECT)?,
        );
        loads.push((file_offset, virtual_address, file_size, memory_size));
    }
    if loads.is_empty() || virtual_end <= virtual_base {
        return Err(INVALID_CODE_OBJECT);
    }
    let allocation_end = align_up(virtual_end, 4096).ok_or(INVALID_CODE_OBJECT)?;
    let allocation_size = allocation_end - virtual_base;
    let allocation = runtime.gpus[gpu_index]
        .device
        .allocate(
            memory_kind,
            allocation_size,
            4096,
            DeviceAccess::READ | DeviceAccess::WRITE | DeviceAccess::EXECUTE,
        )
        .map_err(map_error)?;
    let info = allocation.info();
    let host_base = info.host_address.ok_or(OUT_OF_RESOURCES)?;
    // SAFETY: The allocation is live, host-visible, and allocation_size bytes.
    let allocation_size_usize =
        usize::try_from(allocation_size).map_err(|_| INVALID_CODE_OBJECT)?;
    unsafe { std::ptr::write_bytes(host_base as *mut u8, 0, allocation_size_usize) };
    for &(file_offset, virtual_address, file_size, _) in &loads {
        let source = usize::try_from(file_offset).map_err(|_| INVALID_CODE_OBJECT)?;
        let destination =
            usize::try_from(virtual_address - virtual_base).map_err(|_| INVALID_CODE_OBJECT)?;
        let count = usize::try_from(file_size).map_err(|_| INVALID_CODE_OBJECT)?;
        // SAFETY: Both ranges were bounds-checked against the source and the
        // complete mapped PT_LOAD virtual span.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr().add(source),
                (host_base as *mut u8).add(destination),
                count,
            );
        }
    }
    // SAFETY: The allocation spans the complete page-aligned PT_LOAD range.
    let image =
        unsafe { std::slice::from_raw_parts_mut(host_base as *mut u8, allocation_size_usize) };
    apply_relocations(
        bytes,
        layout.section_offset,
        layout.section_entry_size,
        layout.section_count,
        virtual_base,
        info.device_address,
        image,
    )?;
    // The executable is written through a potentially write-combined CPU
    // mapping. Make both segment data and relocations visible before loading
    // can be observed complete by a submitting thread.
    fence(Ordering::SeqCst);

    let mut pending = parse_symbols(bytes, layout, &[SECTION_DYNSYM, SECTION_SYMTAB])?;
    pending.retain_mut(|symbol| {
        if !symbol.is_definition || symbol.value < virtual_base || symbol.value >= virtual_end {
            return false;
        }
        let Some(address) = info.device_address.checked_add(symbol.value - virtual_base) else {
            return false;
        };
        symbol.address = address;
        true
    });
    let segments = loaded_segments(&loads, info.device_address, virtual_base)?;
    let loaded_size = u64::try_from(segments[0].size).map_err(|_| INVALID_CODE_OBJECT)?;
    Ok((
        LoadedObject {
            handle: 0,
            _allocation: allocation,
            host_base,
            device_base: info.device_address,
            size: loaded_size,
            virtual_base,
            agent,
            kind: object_kind,
            uri: match storage {
                ReaderStorage::Memory => format!(
                    "memory://{}#offset=0x{:x}&size={}",
                    std::process::id(),
                    code_object.as_ptr() as usize,
                    code_object.len()
                )
                .into_bytes(),
                ReaderStorage::File { descriptor, offset } => format!(
                    "file:///proc/self/fd/{descriptor}#offset={offset}&size={}",
                    code_object.len()
                )
                .into_bytes(),
            },
            segments,
            code_object,
        },
        pending,
    ))
}

fn code_object_isa(runtime: &Runtime, layout: ElfLayout) -> Result<HsaIsa, Status> {
    let Some((target, variant)) = code_object_target(layout.flags) else {
        return Err(INVALID_ISA_NAME);
    };
    for (index, gpu) in runtime.gpus.iter().enumerate() {
        let Some(name) = crate::isa_name(
            gpu.info.gfx_major,
            gpu.info.gfx_minor,
            gpu.info.gfx_stepping,
            variant,
        ) else {
            continue;
        };
        if name.strip_prefix("amdgcn-amd-amdhsa--") == Some(target) {
            return Ok(crate::isa_handle(index, variant));
        }
    }
    Err(INVALID_ISA_NAME)
}

fn ensure_code_object_symbols(runtime: &mut Runtime, handle: u64) -> Result<(), Status> {
    let Some(code_object) = runtime.code_objects.get(&handle) else {
        return Err(INVALID_CODE_OBJECT);
    };
    if code_object.symbols_initialized {
        return Ok(());
    }
    let bytes = code_object.bytes.clone();
    let layout = parse_elf_layout(&bytes)?;
    let symbols = parse_symbols(&bytes, layout, &[SECTION_SYMTAB, SECTION_DYNSYM])?;
    let mut names = HashMap::new();
    let mut handles = Vec::with_capacity(symbols.len());
    for symbol in symbols {
        let symbol_handle = runtime.allocate_handle()?;
        names.insert(symbol.full_name, symbol_handle);
        runtime.code_symbols.insert(
            symbol_handle,
            CodeSymbol {
                code_object: handle,
                name: symbol.name,
                module_name: symbol.module_name,
                kind: symbol.kind,
                linkage: symbol.linkage,
                is_definition: symbol.is_definition,
                allocation: symbol.allocation,
                segment: symbol.segment,
                alignment: symbol.alignment,
                size: symbol.size,
                is_const: symbol.is_const,
                kernarg_size: symbol.kernarg_size,
                kernarg_alignment: symbol.kernarg_alignment,
                group_size: symbol.group_size,
                private_size: symbol.private_size,
                dynamic_callstack: symbol.dynamic_callstack,
                call_convention: symbol.call_convention,
                wavefront_size: symbol.wavefront_size,
            },
        );
        handles.push(symbol_handle);
    }
    let Some(code_object) = runtime.code_objects.get_mut(&handle) else {
        for symbol_handle in handles {
            runtime.code_symbols.remove(&symbol_handle);
        }
        return Err(INVALID_CODE_OBJECT);
    };
    code_object.names = names;
    code_object.symbol_handles = handles;
    code_object.symbols_initialized = true;
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_deserialize(
    serialized_code_object: *mut c_void,
    serialized_code_object_size: usize,
    _options: *const c_char,
    code_object: *mut HsaCodeObject,
) -> Status {
    boundary(|| {
        if serialized_code_object.is_null()
            || serialized_code_object_size == 0
            || code_object.is_null()
        {
            return INVALID_ARGUMENT;
        }
        let mut bytes = Vec::new();
        if bytes
            .try_reserve_exact(serialized_code_object_size)
            .is_err()
        {
            return OUT_OF_RESOURCES;
        }
        bytes.resize(serialized_code_object_size, 0);
        // SAFETY: The caller supplies serialized_code_object_size readable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(
                serialized_code_object.cast::<u8>(),
                bytes.as_mut_ptr(),
                serialized_code_object_size,
            );
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.code_objects.insert(
            handle,
            CodeObject {
                bytes: bytes.into(),
                names: HashMap::new(),
                symbol_handles: Vec::new(),
                symbols_initialized: false,
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { code_object.write(HsaCodeObject { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_serialize(
    code_object: HsaCodeObject,
    alloc_callback: CodeObjectAllocCallback,
    callback_data: HsaCallbackData,
    _options: *const c_char,
    serialized_code_object: *mut *mut c_void,
    serialized_code_object_size: *mut usize,
) -> Status {
    boundary(|| {
        let Some(alloc_callback) = alloc_callback else {
            return INVALID_ARGUMENT;
        };
        if serialized_code_object.is_null() || serialized_code_object_size.is_null() {
            return INVALID_ARGUMENT;
        }
        let bytes = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(code_object) = runtime.code_objects.get(&code_object.handle) else {
                return INVALID_CODE_OBJECT;
            };
            code_object.bytes.clone()
        };
        // SAFETY: The callback and output storage are supplied by the caller.
        let status = unsafe { alloc_callback(bytes.len(), callback_data, serialized_code_object) };
        if status != SUCCESS {
            return status;
        }
        // SAFETY: A successful callback must initialize the output pointer.
        let destination = unsafe { serialized_code_object.read() };
        if destination.is_null() {
            return OUT_OF_RESOURCES;
        }
        // SAFETY: The callback allocated at least bytes.len() writable bytes.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), destination.cast::<u8>(), bytes.len());
            serialized_code_object_size.write(bytes.len());
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_destroy(code_object: HsaCodeObject) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(code_object) = runtime.code_objects.remove(&code_object.handle) else {
            return INVALID_CODE_OBJECT;
        };
        for handle in code_object.symbol_handles {
            runtime.code_symbols.remove(&handle);
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_get_info(
    code_object: HsaCodeObject,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(code_object) = runtime.code_objects.get(&code_object.handle) else {
            return INVALID_CODE_OBJECT;
        };
        let layout = match parse_elf_layout(&code_object.bytes) {
            Ok(layout) => layout,
            Err(status) => return status,
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                CODE_OBJECT_INFO_VERSION => {
                    let version = match code_object_version(layout) {
                        Ok(version) => version,
                        Err(status) => return status,
                    };
                    let text = format!("{version}.0");
                    std::ptr::write_bytes(value.cast::<u8>(), 0, 64);
                    std::ptr::copy_nonoverlapping(
                        text.as_ptr(),
                        value.cast::<u8>(),
                        text.len().min(63),
                    );
                }
                CODE_OBJECT_INFO_TYPE => value.cast::<u32>().write(CODE_OBJECT_TYPE_PROGRAM),
                CODE_OBJECT_INFO_ISA => {
                    let isa = match code_object_isa(runtime, layout) {
                        Ok(isa) => isa,
                        Err(status) => return status,
                    };
                    value.cast::<HsaIsa>().write(isa);
                }
                CODE_OBJECT_INFO_MACHINE_MODEL => {
                    value.cast::<u32>().write(MACHINE_MODEL_LARGE);
                }
                CODE_OBJECT_INFO_PROFILE => value.cast::<u32>().write(PROFILE_BASE),
                CODE_OBJECT_INFO_DEFAULT_FLOAT_ROUNDING_MODE => {
                    value.cast::<u32>().write(DEFAULT_FLOAT_ROUNDING_MODE_NEAR);
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

unsafe fn code_object_get_symbol_impl(
    code_object: HsaCodeObject,
    module_name: *const c_char,
    symbol_name: *const c_char,
    symbol: *mut HsaCodeSymbol,
) -> Status {
    boundary(|| {
        if symbol_name.is_null() || symbol.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires NUL-terminated symbol and optional module names.
        let Ok(symbol_name) = (unsafe { CStr::from_ptr(symbol_name) }).to_str() else {
            return INVALID_ARGUMENT;
        };
        let module_name = if module_name.is_null() {
            ""
        } else {
            let Ok(module_name) = (unsafe { CStr::from_ptr(module_name) }).to_str() else {
                return INVALID_ARGUMENT;
            };
            module_name
        };
        let full_name = if module_name.is_empty() {
            symbol_name.to_owned()
        } else {
            format!("{module_name}::{symbol_name}")
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if let Err(status) = ensure_code_object_symbols(runtime, code_object.handle) {
            return status;
        }
        let Some(&handle) = runtime
            .code_objects
            .get(&code_object.handle)
            .and_then(|code_object| code_object.names.get(&full_name))
        else {
            return INVALID_SYMBOL_NAME;
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { symbol.write(HsaCodeSymbol { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_get_symbol(
    code_object: HsaCodeObject,
    symbol_name: *const c_char,
    symbol: *mut HsaCodeSymbol,
) -> Status {
    // SAFETY: This forwards the public arguments without changing their contracts.
    unsafe { code_object_get_symbol_impl(code_object, std::ptr::null(), symbol_name, symbol) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_get_symbol_from_name(
    code_object: HsaCodeObject,
    module_name: *const c_char,
    symbol_name: *const c_char,
    symbol: *mut HsaCodeSymbol,
) -> Status {
    // SAFETY: This forwards the public arguments without changing their contracts.
    unsafe { code_object_get_symbol_impl(code_object, module_name, symbol_name, symbol) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_symbol_get_info(
    code_symbol: HsaCodeSymbol,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(symbol) = runtime.code_symbols.get(&code_symbol.handle) else {
            return INVALID_CODE_SYMBOL;
        };
        if !runtime.code_objects.contains_key(&symbol.code_object) {
            return INVALID_CODE_SYMBOL;
        }
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                SYMBOL_INFO_TYPE => value.cast::<u32>().write(symbol.kind),
                SYMBOL_INFO_NAME_LENGTH => value
                    .cast::<u32>()
                    .write(u32::try_from(symbol.name.len()).unwrap_or(u32::MAX)),
                SYMBOL_INFO_NAME => std::ptr::copy_nonoverlapping(
                    symbol.name.as_ptr(),
                    value.cast::<u8>(),
                    symbol.name.len(),
                ),
                SYMBOL_INFO_MODULE_NAME_LENGTH => value
                    .cast::<u32>()
                    .write(u32::try_from(symbol.module_name.len()).unwrap_or(u32::MAX)),
                SYMBOL_INFO_MODULE_NAME => std::ptr::copy_nonoverlapping(
                    symbol.module_name.as_ptr(),
                    value.cast::<u8>(),
                    symbol.module_name.len(),
                ),
                SYMBOL_INFO_LINKAGE => value.cast::<u32>().write(symbol.linkage),
                SYMBOL_INFO_IS_DEFINITION => value.cast::<bool>().write(symbol.is_definition),
                SYMBOL_INFO_VARIABLE_ALLOCATION => value.cast::<u32>().write(symbol.allocation),
                SYMBOL_INFO_VARIABLE_SEGMENT => value.cast::<u32>().write(symbol.segment),
                SYMBOL_INFO_VARIABLE_ALIGNMENT => value.cast::<u32>().write(symbol.alignment),
                SYMBOL_INFO_VARIABLE_SIZE => value.cast::<u32>().write(symbol.size),
                SYMBOL_INFO_VARIABLE_IS_CONST => value.cast::<bool>().write(symbol.is_const),
                SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.kernarg_size);
                }
                SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT => {
                    value.cast::<u32>().write(symbol.kernarg_alignment);
                }
                SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.group_size);
                }
                SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.private_size);
                }
                SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK => {
                    value.cast::<bool>().write(symbol.dynamic_callstack);
                }
                SYMBOL_INFO_KERNEL_CALL_CONVENTION
                | SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION => {
                    value.cast::<u32>().write(symbol.call_convention);
                }
                SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE => {
                    value.cast::<u32>().write(symbol.wavefront_size);
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_iterate_symbols(
    code_object: HsaCodeObject,
    callback: CodeObjectSymbolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
            let mut guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let runtime = match initialized_mut(&mut guard) {
                Ok(runtime) => runtime,
                Err(status) => return status,
            };
            if let Err(status) = ensure_code_object_symbols(runtime, code_object.handle) {
                return status;
            }
            runtime.code_objects[&code_object.handle]
                .symbol_handles
                .clone()
        };
        for handle in handles {
            // SAFETY: Traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(code_object, HsaCodeSymbol { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_reader_create_from_file(
    file: i32,
    reader: *mut HsaCodeObjectReader,
) -> Status {
    boundary(|| {
        if file < 0 || reader.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        // SAFETY: The descriptor remains owned by the application. ManuallyDrop
        // prevents this temporary File view from closing it.
        let file_view = ManuallyDrop::new(unsafe { File::from_raw_fd(file) });
        let size = match file_view
            .metadata()
            .ok()
            .and_then(|metadata| usize::try_from(metadata.len()).ok())
        {
            Some(0) => return INVALID_CODE_OBJECT,
            Some(size) => size,
            None => return INVALID_FILE,
        };
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(size).is_err() {
            return OUT_OF_RESOURCES;
        }
        bytes.resize(size, 0);
        if file_view.read_exact_at(&mut bytes, 0).is_err() {
            return INVALID_FILE;
        }
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.readers.insert(
            handle,
            Reader {
                bytes: bytes.into(),
                storage: ReaderStorage::File {
                    descriptor: file,
                    offset: 0,
                },
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { reader.write(HsaCodeObjectReader { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_reader_create_from_memory(
    code_object: *const c_void,
    size: usize,
    reader: *mut HsaCodeObjectReader,
) -> Status {
    boundary(|| {
        if code_object.is_null() || size == 0 || reader.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplies size readable bytes for the call.
        let bytes = unsafe { std::slice::from_raw_parts(code_object.cast::<u8>(), size) }.to_vec();
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.readers.insert(
            handle,
            Reader {
                bytes: bytes.into(),
                storage: ReaderStorage::Memory,
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { reader.write(HsaCodeObjectReader { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_code_object_reader_destroy(reader: HsaCodeObjectReader) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.readers.remove(&reader.handle).is_some() {
            SUCCESS
        } else {
            INVALID_CODE_OBJECT_READER
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_create(
    profile: u32,
    state: u32,
    _options: *const c_char,
    executable: *mut HsaExecutable,
) -> Status {
    boundary(|| {
        if executable.is_null()
            || !matches!(profile, PROFILE_BASE | PROFILE_FULL)
            || !matches!(state, EXECUTABLE_STATE_UNFROZEN | EXECUTABLE_STATE_FROZEN)
        {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.executables.insert(
            handle,
            Executable {
                profile,
                default_rounding: DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                frozen: state == EXECUTABLE_STATE_FROZEN,
                program_loaded: false,
                loaded: Vec::new(),
                names: HashMap::new(),
                symbol_handles: Vec::new(),
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { executable.write(HsaExecutable { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_create_alt(
    profile: u32,
    rounding: u32,
    _options: *const c_char,
    executable: *mut HsaExecutable,
) -> Status {
    boundary(|| {
        if executable.is_null()
            || !matches!(profile, PROFILE_BASE | PROFILE_FULL)
            || !matches!(
                rounding,
                DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT
                    | DEFAULT_FLOAT_ROUNDING_MODE_ZERO
                    | DEFAULT_FLOAT_ROUNDING_MODE_NEAR
            )
        {
            return INVALID_ARGUMENT;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.executables.insert(
            handle,
            Executable {
                profile,
                default_rounding: rounding,
                frozen: false,
                program_loaded: false,
                loaded: Vec::new(),
                names: HashMap::new(),
                symbol_handles: Vec::new(),
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { executable.write(HsaExecutable { handle }) };
        SUCCESS
    })
}

fn loaded_symbol_agent(program: bool, agent: HsaAgent) -> HsaAgent {
    if program {
        HsaAgent { handle: 0 }
    } else {
        agent
    }
}

unsafe fn load_executable_code_object(
    runtime: &mut Runtime,
    executable: HsaExecutable,
    agent: HsaAgent,
    bytes: Arc<[u8]>,
    storage: ReaderStorage,
    program: bool,
    loaded: *mut HsaLoadedCodeObject,
) -> Status {
    let Some(executable_record) = runtime.executables.get(&executable.handle) else {
        return INVALID_EXECUTABLE;
    };
    if executable_record.frozen {
        return FROZEN_EXECUTABLE;
    }
    if program && executable_record.program_loaded {
        return INCOMPATIBLE_ARGUMENTS;
    }
    let (mut object, symbols) = match parse_and_load(runtime, agent, bytes, storage) {
        Ok(result) => result,
        Err(status) => return status,
    };
    let loaded_handle = match runtime.allocate_handle() {
        Ok(handle) => handle,
        Err(status) => return status,
    };
    object.handle = loaded_handle;
    let mut handles = Vec::with_capacity(symbols.len());
    for _ in &symbols {
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        handles.push(handle);
    }
    if symbols.iter().any(|symbol| {
        let symbol_agent = loaded_symbol_agent(program, agent);
        runtime
            .executables
            .get(&executable.handle)
            .is_some_and(|record| {
                record
                    .names
                    .contains_key(&(symbol.full_name.clone(), symbol_agent.handle))
            })
    }) {
        return VARIABLE_ALREADY_DEFINED;
    }
    let mut inserted = Vec::with_capacity(symbols.len());
    for (symbol, handle) in symbols.into_iter().zip(handles) {
        let symbol_agent = loaded_symbol_agent(program, agent);
        runtime.symbols.insert(
            handle,
            Symbol {
                executable: executable.handle,
                name: symbol.name,
                module_name: symbol.module_name,
                agent: symbol_agent,
                address: symbol.address,
                size: symbol.size,
                kind: symbol.kind,
                linkage: symbol.linkage,
                is_definition: symbol.is_definition,
                allocation: symbol.allocation,
                segment: symbol.segment,
                alignment: symbol.alignment,
                is_const: symbol.is_const,
                kernarg_size: symbol.kernarg_size,
                kernarg_alignment: symbol.kernarg_alignment,
                group_size: symbol.group_size,
                private_size: symbol.private_size,
                dynamic_callstack: symbol.dynamic_callstack,
                call_convention: symbol.call_convention,
            },
        );
        inserted.push((symbol.full_name, symbol_agent.handle, handle));
    }
    let Some(executable_record) = runtime.executables.get_mut(&executable.handle) else {
        for (_, _, handle) in inserted {
            runtime.symbols.remove(&handle);
        }
        return INVALID_EXECUTABLE;
    };
    for (name, agent_handle, handle) in inserted {
        executable_record.names.insert((name, agent_handle), handle);
        executable_record.symbol_handles.push(handle);
    }
    executable_record.program_loaded |= program;
    executable_record.loaded.push(object);
    if !loaded.is_null() {
        // SAFETY: A non-null optional output points to writable storage.
        unsafe {
            loaded.write(HsaLoadedCodeObject {
                handle: loaded_handle,
            });
        }
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_load_agent_code_object(
    executable: HsaExecutable,
    agent: HsaAgent,
    reader: HsaCodeObjectReader,
    _options: *const c_char,
    loaded: *mut HsaLoadedCodeObject,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(record) = runtime.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        if record.frozen {
            return FROZEN_EXECUTABLE;
        }
        let Some((bytes, storage)) = runtime
            .readers
            .get(&reader.handle)
            .map(|reader| (reader.bytes.clone(), reader.storage))
        else {
            return INVALID_CODE_OBJECT_READER;
        };
        // SAFETY: The optional loaded-code-object output follows the public ABI.
        unsafe {
            load_executable_code_object(runtime, executable, agent, bytes, storage, false, loaded)
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_load_program_code_object(
    executable: HsaExecutable,
    reader: HsaCodeObjectReader,
    _options: *const c_char,
    loaded: *mut HsaLoadedCodeObject,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(record) = runtime.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        if record.frozen {
            return FROZEN_EXECUTABLE;
        }
        let Some((bytes, storage)) = runtime
            .readers
            .get(&reader.handle)
            .map(|reader| (reader.bytes.clone(), reader.storage))
        else {
            return INVALID_CODE_OBJECT_READER;
        };
        // SAFETY: The optional loaded-code-object output follows the public ABI.
        unsafe {
            load_executable_code_object(
                runtime,
                executable,
                HsaAgent { handle: 0 },
                bytes,
                storage,
                true,
                loaded,
            )
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_load_code_object(
    executable: HsaExecutable,
    agent: HsaAgent,
    code_object: HsaCodeObject,
    _options: *const c_char,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(record) = runtime.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        if record.frozen {
            return FROZEN_EXECUTABLE;
        }
        let Some(bytes) = runtime
            .code_objects
            .get(&code_object.handle)
            .map(|code_object| code_object.bytes.clone())
        else {
            return INVALID_CODE_OBJECT;
        };
        // SAFETY: The deprecated entry point does not request a loaded handle.
        unsafe {
            load_executable_code_object(
                runtime,
                executable,
                agent,
                bytes,
                ReaderStorage::Memory,
                false,
                std::ptr::null_mut(),
            )
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_freeze(
    executable: HsaExecutable,
    _options: *const c_char,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(executable) = runtime.executables.get_mut(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        if executable.frozen {
            return FROZEN_EXECUTABLE;
        }
        executable.frozen = true;
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_destroy(executable: HsaExecutable) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let Some(executable) = runtime.executables.remove(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        for handle in executable.symbol_handles {
            runtime.symbols.remove(&handle);
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_validate(
    executable: HsaExecutable,
    result: *mut u32,
) -> Status {
    boundary(|| {
        if result.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if !runtime.executables.contains_key(&executable.handle) {
            return INVALID_EXECUTABLE;
        }
        // Every accepted code object has already passed structural validation
        // and relocation while being loaded.
        unsafe { result.write(0) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_validate_alt(
    executable: HsaExecutable,
    _options: *const c_char,
    result: *mut u32,
) -> Status {
    // SAFETY: Options are intentionally ignored, matching ROCr validation.
    unsafe { hsa_executable_validate(executable, result) }
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_get_info(
    executable: HsaExecutable,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(executable) = guard
            .as_ref()
            .and_then(|runtime| runtime.executables.get(&executable.handle))
        else {
            return if guard.is_some() {
                INVALID_EXECUTABLE
            } else {
                NOT_INITIALIZED
            };
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                EXECUTABLE_INFO_PROFILE => value.cast::<u32>().write(executable.profile),
                EXECUTABLE_INFO_STATE => value.cast::<u32>().write(if executable.frozen {
                    EXECUTABLE_STATE_FROZEN
                } else {
                    EXECUTABLE_STATE_UNFROZEN
                }),
                EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE => {
                    value.cast::<u32>().write(executable.default_rounding);
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_get_symbol_by_name(
    executable: HsaExecutable,
    name: *const c_char,
    agent: *const HsaAgent,
    symbol: *mut HsaExecutableSymbol,
) -> Status {
    boundary(|| {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if name.is_null() || symbol.is_null() {
            return INVALID_ARGUMENT;
        }
        let Some(executable) = runtime.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        // SAFETY: The ABI requires a NUL-terminated symbol name.
        let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
            return INVALID_ARGUMENT;
        };
        let requested_agent = if agent.is_null() {
            HsaAgent { handle: 0 }
        } else {
            // SAFETY: A non-null optional agent points to one HsaAgent.
            unsafe { *agent }
        };
        let Some(&handle) = executable
            .names
            .get(&(name.to_owned(), requested_agent.handle))
        else {
            return INVALID_SYMBOL_NAME;
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { symbol.write(HsaExecutableSymbol { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_symbol_get_info(
    symbol: HsaExecutableSymbol,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(symbol) = guard
            .as_ref()
            .and_then(|runtime| runtime.symbols.get(&symbol.handle))
        else {
            return if guard.is_some() {
                INVALID_EXECUTABLE_SYMBOL
            } else {
                NOT_INITIALIZED
            };
        };
        let frozen = guard
            .as_ref()
            .and_then(|runtime| runtime.executables.get(&symbol.executable))
            .is_some_and(|executable| executable.frozen);
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                SYMBOL_INFO_TYPE => value.cast::<u32>().write(symbol.kind),
                SYMBOL_INFO_NAME_LENGTH => {
                    value
                        .cast::<u32>()
                        .write(u32::try_from(symbol.name.len()).unwrap_or(u32::MAX));
                }
                SYMBOL_INFO_NAME => {
                    std::ptr::copy_nonoverlapping(
                        symbol.name.as_ptr(),
                        value.cast::<u8>(),
                        symbol.name.len(),
                    );
                }
                SYMBOL_INFO_MODULE_NAME_LENGTH => value
                    .cast::<u32>()
                    .write(u32::try_from(symbol.module_name.len()).unwrap_or(u32::MAX)),
                SYMBOL_INFO_MODULE_NAME => std::ptr::copy_nonoverlapping(
                    symbol.module_name.as_ptr(),
                    value.cast::<u8>(),
                    symbol.module_name.len(),
                ),
                SYMBOL_INFO_LINKAGE => value.cast::<u32>().write(symbol.linkage),
                SYMBOL_INFO_IS_DEFINITION => value.cast::<bool>().write(symbol.is_definition),
                SYMBOL_INFO_AGENT => value.cast::<HsaAgent>().write(symbol.agent),
                SYMBOL_INFO_VARIABLE_ALLOCATION => value.cast::<u32>().write(symbol.allocation),
                SYMBOL_INFO_VARIABLE_SEGMENT => value.cast::<u32>().write(symbol.segment),
                SYMBOL_INFO_VARIABLE_ALIGNMENT => value.cast::<u32>().write(symbol.alignment),
                SYMBOL_INFO_VARIABLE_SIZE => value.cast::<u32>().write(symbol.size),
                SYMBOL_INFO_VARIABLE_IS_CONST => value.cast::<bool>().write(symbol.is_const),
                SYMBOL_INFO_VARIABLE_ADDRESS
                | SYMBOL_INFO_KERNEL_OBJECT
                | SYMBOL_INFO_INDIRECT_FUNCTION_OBJECT => {
                    value
                        .cast::<u64>()
                        .write(if frozen { symbol.address } else { 0 });
                }
                SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.kernarg_size);
                }
                SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT => {
                    value.cast::<u32>().write(symbol.kernarg_alignment);
                }
                SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.group_size);
                }
                SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE => {
                    value.cast::<u32>().write(symbol.private_size);
                }
                SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK => {
                    value.cast::<bool>().write(symbol.dynamic_callstack);
                }
                SYMBOL_INFO_KERNEL_CALL_CONVENTION
                | SYMBOL_INFO_INDIRECT_FUNCTION_CALL_CONVENTION => {
                    value.cast::<u32>().write(symbol.call_convention);
                }
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_get_symbol(
    executable: HsaExecutable,
    module_name: *const c_char,
    symbol_name: *const c_char,
    agent: HsaAgent,
    _call_convention: i32,
    symbol: *mut HsaExecutableSymbol,
) -> Status {
    boundary(|| {
        if symbol_name.is_null() || symbol.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires NUL-terminated symbol and optional module names.
        let Ok(symbol_name) = (unsafe { CStr::from_ptr(symbol_name) }).to_str() else {
            return INVALID_ARGUMENT;
        };
        let module_name = if module_name.is_null() {
            ""
        } else {
            let Ok(module_name) = (unsafe { CStr::from_ptr(module_name) }).to_str() else {
                return INVALID_ARGUMENT;
            };
            module_name
        };
        let name = if module_name.is_empty() {
            symbol_name.to_owned()
        } else {
            format!("{module_name}::{symbol_name}")
        };
        if name.is_empty() {
            return INVALID_SYMBOL_NAME;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(executable) = runtime.executables.get(&executable.handle) else {
            return INVALID_EXECUTABLE;
        };
        let handle = executable
            .names
            .get(&(name.clone(), 0))
            .or_else(|| executable.names.get(&(name, agent.handle)));
        let Some(&handle) = handle else {
            return INVALID_SYMBOL_NAME;
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { symbol.write(HsaExecutableSymbol { handle }) };
        SUCCESS
    })
}

fn define_external_variable(
    runtime: &mut Runtime,
    executable: HsaExecutable,
    agent: HsaAgent,
    name: &str,
    address: *mut c_void,
    allocation: u32,
    segment: u32,
) -> Status {
    let Some(record) = runtime.executables.get(&executable.handle) else {
        return INVALID_EXECUTABLE;
    };
    if record.frozen {
        return FROZEN_EXECUTABLE;
    }
    let key = (name.to_owned(), agent.handle);
    if record.names.contains_key(&key) {
        return VARIABLE_ALREADY_DEFINED;
    }
    let handle = match runtime.allocate_handle() {
        Ok(handle) => handle,
        Err(status) => return status,
    };
    runtime.symbols.insert(
        handle,
        Symbol {
            executable: executable.handle,
            name: name.to_owned(),
            module_name: String::new(),
            agent,
            address: address as u64,
            size: 0,
            kind: SYMBOL_KIND_VARIABLE,
            linkage: SYMBOL_LINKAGE_PROGRAM,
            is_definition: true,
            allocation,
            segment,
            alignment: 0,
            is_const: segment == VARIABLE_SEGMENT_READONLY,
            kernarg_size: 0,
            kernarg_alignment: 0,
            group_size: 0,
            private_size: 0,
            dynamic_callstack: false,
            call_convention: 0,
        },
    );
    let Some(record) = runtime.executables.get_mut(&executable.handle) else {
        runtime.symbols.remove(&handle);
        return INVALID_EXECUTABLE;
    };
    record.names.insert(key, handle);
    record.symbol_handles.push(handle);
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_global_variable_define(
    executable: HsaExecutable,
    name: *const c_char,
    address: *mut c_void,
) -> Status {
    boundary(|| {
        if name.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires a NUL-terminated variable name.
        let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
            return INVALID_ARGUMENT;
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        define_external_variable(
            runtime,
            executable,
            HsaAgent { handle: 0 },
            name,
            address,
            VARIABLE_ALLOCATION_PROGRAM,
            VARIABLE_SEGMENT_GLOBAL,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_agent_global_variable_define(
    executable: HsaExecutable,
    agent: HsaAgent,
    name: *const c_char,
    address: *mut c_void,
) -> Status {
    boundary(|| {
        if name.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires a NUL-terminated variable name.
        let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
            return INVALID_ARGUMENT;
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        define_external_variable(
            runtime,
            executable,
            agent,
            name,
            address,
            VARIABLE_ALLOCATION_AGENT,
            VARIABLE_SEGMENT_GLOBAL,
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_readonly_variable_define(
    executable: HsaExecutable,
    agent: HsaAgent,
    name: *const c_char,
    address: *mut c_void,
) -> Status {
    boundary(|| {
        if name.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The ABI requires a NUL-terminated variable name.
        let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
            return INVALID_ARGUMENT;
        };
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime.gpu_index(agent).is_none() {
            return INVALID_AGENT;
        }
        define_external_variable(
            runtime,
            executable,
            agent,
            name,
            address,
            VARIABLE_ALLOCATION_AGENT,
            VARIABLE_SEGMENT_READONLY,
        )
    })
}

fn executable_symbol_handles(
    runtime: &Runtime,
    executable: HsaExecutable,
    agent: Option<HsaAgent>,
) -> Result<Vec<u64>, Status> {
    let Some(executable) = runtime.executables.get(&executable.handle) else {
        return Err(INVALID_EXECUTABLE);
    };
    Ok(executable
        .symbol_handles
        .iter()
        .copied()
        .filter(|handle| {
            runtime.symbols.get(handle).is_some_and(|symbol| {
                agent.map_or(symbol.agent.handle == 0, |agent| symbol.agent == agent)
            })
        })
        .collect())
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_iterate_symbols(
    executable: HsaExecutable,
    callback: ExecutableSymbolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(record) = runtime.executables.get(&executable.handle) else {
                return INVALID_EXECUTABLE;
            };
            record.symbol_handles.clone()
        };
        for handle in handles {
            // SAFETY: Traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(executable, HsaExecutableSymbol { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_iterate_agent_symbols(
    executable: HsaExecutable,
    agent: HsaAgent,
    callback: AgentExecutableSymbolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            if !runtime.is_agent(agent) {
                return INVALID_AGENT;
            }
            match executable_symbol_handles(runtime, executable, Some(agent)) {
                Ok(handles) => handles,
                Err(status) => return status,
            }
        };
        for handle in handles {
            // SAFETY: Traversal callbacks are synchronous and data stays live.
            let status =
                unsafe { callback(executable, agent, HsaExecutableSymbol { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_executable_iterate_program_symbols(
    executable: HsaExecutable,
    callback: ExecutableSymbolCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            match executable_symbol_handles(runtime, executable, None) {
                Ok(handles) => handles,
                Err(status) => return status,
            }
        };
        for handle in handles {
            // SAFETY: Traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(executable, HsaExecutableSymbol { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_loader_query_host_address(
    device_address: *const c_void,
    host_address: *mut *const c_void,
) -> Status {
    boundary(|| {
        if device_address.is_null() || host_address.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let address = device_address as u64;
        for executable in runtime.executables.values() {
            for object in &executable.loaded {
                if address >= object.device_base && address - object.device_base < object.size {
                    let Ok(offset) = usize::try_from(address - object.device_base) else {
                        return INVALID_ARGUMENT;
                    };
                    let Some(host) = object.host_base.checked_add(offset) else {
                        return INVALID_ARGUMENT;
                    };
                    // SAFETY: The caller supplied writable output storage.
                    unsafe { host_address.write(host as *const c_void) };
                    return SUCCESS;
                }
            }
        }
        INVALID_ARGUMENT
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_loader_query_segment_descriptors(
    segment_descriptors: *mut HsaLoaderSegmentDescriptor,
    num_segment_descriptors: *mut usize,
) -> Status {
    boundary(|| {
        if num_segment_descriptors.is_null() {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplied readable count storage.
        let requested = unsafe { num_segment_descriptors.read() };
        if (requested == 0) != segment_descriptors.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let actual = runtime
            .executables
            .values()
            .map(|executable| {
                executable
                    .loaded
                    .iter()
                    .map(|object| object.segments.len())
                    .sum::<usize>()
            })
            .sum::<usize>();
        if requested == 0 {
            // SAFETY: The caller supplied writable count storage.
            unsafe { num_segment_descriptors.write(actual) };
            return SUCCESS;
        }
        if requested != actual {
            return INCOMPATIBLE_ARGUMENTS;
        }
        let mut output_index = 0;
        let mut executable_handles = runtime.executables.keys().copied().collect::<Vec<_>>();
        executable_handles.sort_unstable();
        for executable_handle in executable_handles {
            let executable = &runtime.executables[&executable_handle];
            for object in &executable.loaded {
                for segment in &object.segments {
                    let descriptor = HsaLoaderSegmentDescriptor {
                        agent: object.agent,
                        executable: HsaExecutable {
                            handle: executable_handle,
                        },
                        code_object_storage_type: LOADER_STORAGE_MEMORY,
                        code_object_storage_base: object.code_object.as_ptr().cast(),
                        code_object_storage_size: object.code_object.len(),
                        code_object_storage_offset: segment.storage_offset,
                        segment_base: segment.address as usize as *const c_void,
                        segment_size: segment.size,
                    };
                    // SAFETY: The caller supplied exactly actual writable entries.
                    unsafe { segment_descriptors.add(output_index).write(descriptor) };
                    output_index += 1;
                }
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_loader_query_executable(
    device_address: *const c_void,
    executable: *mut HsaExecutable,
) -> Status {
    boundary(|| {
        if device_address.is_null() || executable.is_null() {
            return INVALID_ARGUMENT;
        }
        let address = device_address as u64;
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some(handle) = runtime.executables.iter().find_map(|(handle, record)| {
            record
                .loaded
                .iter()
                .any(|object| {
                    address >= object.device_base && address - object.device_base < object.size
                })
                .then_some(*handle)
        }) else {
            return INVALID_ARGUMENT;
        };
        // SAFETY: The caller supplied writable output storage.
        unsafe { executable.write(HsaExecutable { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
    executable: HsaExecutable,
    callback: LoadedCodeObjectCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let handles = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            let Some(record) = runtime.executables.get(&executable.handle) else {
                return INVALID_EXECUTABLE;
            };
            record
                .loaded
                .iter()
                .map(|object| HsaLoadedCodeObject {
                    handle: object.handle,
                })
                .collect::<Vec<_>>()
        };
        for loaded in handles {
            // SAFETY: Loader traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(executable, loaded, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_loader_loaded_code_object_get_info(
    loaded_code_object: HsaLoadedCodeObject,
    attribute: u32,
    value: *mut c_void,
) -> Status {
    boundary(|| {
        if value.is_null() {
            return INVALID_ARGUMENT;
        }
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        let Some((executable_handle, object)) =
            runtime.executables.iter().find_map(|(handle, executable)| {
                executable
                    .loaded
                    .iter()
                    .find(|object| object.handle == loaded_code_object.handle)
                    .map(|object| (*handle, object))
            })
        else {
            return INVALID_CODE_OBJECT;
        };
        let load_delta = i128::from(object.device_base) - i128::from(object.virtual_base);
        let Ok(load_delta) = i64::try_from(load_delta) else {
            return INVALID_CODE_OBJECT;
        };
        // SAFETY: Each arm writes the public value type for the attribute.
        unsafe {
            match attribute {
                LOADER_INFO_EXECUTABLE => value.cast::<HsaExecutable>().write(HsaExecutable {
                    handle: executable_handle,
                }),
                LOADER_INFO_KIND => value.cast::<u32>().write(object.kind),
                LOADER_INFO_AGENT => value.cast::<HsaAgent>().write(object.agent),
                LOADER_INFO_STORAGE_TYPE => value.cast::<u32>().write(LOADER_STORAGE_MEMORY),
                LOADER_INFO_STORAGE_MEMORY_BASE => value
                    .cast::<u64>()
                    .write(object.code_object.as_ptr() as usize as u64),
                LOADER_INFO_STORAGE_MEMORY_SIZE => value
                    .cast::<u64>()
                    .write(u64::try_from(object.code_object.len()).unwrap_or(u64::MAX)),
                LOADER_INFO_STORAGE_FILE => return INVALID_ARGUMENT,
                LOADER_INFO_LOAD_DELTA => value.cast::<i64>().write(load_delta),
                LOADER_INFO_LOAD_BASE => value.cast::<u64>().write(object.device_base),
                LOADER_INFO_LOAD_SIZE => value.cast::<u64>().write(object.size),
                LOADER_INFO_URI_LENGTH => value
                    .cast::<u32>()
                    .write(u32::try_from(object.uri.len()).unwrap_or(u32::MAX)),
                LOADER_INFO_URI => std::ptr::copy_nonoverlapping(
                    object.uri.as_ptr(),
                    value.cast::<u8>(),
                    object.uri.len(),
                ),
                _ => return INVALID_ARGUMENT,
            }
        }
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    file: i32,
    offset: usize,
    size: usize,
    reader: *mut HsaCodeObjectReader,
) -> Status {
    boundary(|| {
        if file < 0 {
            return INVALID_FILE;
        }
        if size == 0 {
            return INVALID_CODE_OBJECT;
        }
        if reader.is_null() {
            return INVALID_ARGUMENT;
        }
        let mut bytes = Vec::new();
        if bytes.try_reserve_exact(size).is_err() {
            return OUT_OF_RESOURCES;
        }
        bytes.resize(size, 0);
        // SAFETY: The descriptor remains owned by the application. ManuallyDrop
        // prevents this temporary File view from closing it.
        let file_view = ManuallyDrop::new(unsafe { File::from_raw_fd(file) });
        if file_view.read_exact_at(&mut bytes, offset as u64).is_err() {
            return INVALID_FILE;
        }
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        runtime.readers.insert(
            handle,
            Reader {
                bytes: bytes.into(),
                storage: ReaderStorage::File {
                    descriptor: file,
                    offset,
                },
            },
        );
        // SAFETY: The caller supplied writable output storage.
        unsafe { reader.write(HsaCodeObjectReader { handle }) };
        SUCCESS
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ven_amd_loader_iterate_executables(
    callback: ExecutableCallback,
    data: *mut c_void,
) -> Status {
    boundary(|| {
        let Some(callback) = callback else {
            return INVALID_ARGUMENT;
        };
        let mut handles = {
            let guard = match lock() {
                Ok(guard) => guard,
                Err(status) => return status,
            };
            let Some(runtime) = guard.as_ref() else {
                return NOT_INITIALIZED;
            };
            runtime.executables.keys().copied().collect::<Vec<_>>()
        };
        handles.sort_unstable();
        for handle in handles {
            // SAFETY: Loader traversal callbacks are synchronous and data stays live.
            let status = unsafe { callback(HsaExecutable { handle }, data) };
            if status != SUCCESS {
                return status;
            }
        }
        SUCCESS
    })
}

pub(crate) fn loader_extension_table() -> [usize; 7] {
    [
        hsa_ven_amd_loader_query_host_address as *const () as usize,
        hsa_ven_amd_loader_query_segment_descriptors as *const () as usize,
        hsa_ven_amd_loader_query_executable as *const () as usize,
        hsa_ven_amd_loader_executable_iterate_loaded_code_objects as *const () as usize,
        hsa_ven_amd_loader_loaded_code_object_get_info as *const () as usize,
        hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size as *const ()
            as usize,
        hsa_ven_amd_loader_iterate_executables as *const () as usize,
    ]
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
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
        write_u32(&mut bytes, 48, AMDGPU_MACHINE_GFX1201);
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
    fn loader_extension_table_contains_every_version_1_03_entry_point() {
        assert!(loader_extension_table().into_iter().all(|entry| entry != 0));
    }

    #[test]
    fn file_reader_entry_point_matches_the_public_abi() {
        let _: unsafe extern "C" fn(i32, *mut HsaCodeObjectReader) -> Status =
            hsa_code_object_reader_create_from_file;
    }

    #[test]
    fn legacy_entry_points_match_the_public_abi() {
        let _: unsafe extern "C" fn(
            *mut c_void,
            usize,
            *const c_char,
            *mut HsaCodeObject,
        ) -> Status = hsa_code_object_deserialize;
        let _: unsafe extern "C" fn(
            HsaCodeObject,
            CodeObjectAllocCallback,
            HsaCallbackData,
            *const c_char,
            *mut *mut c_void,
            *mut usize,
        ) -> Status = hsa_code_object_serialize;
        let _: unsafe extern "C" fn(HsaCodeObject) -> Status = hsa_code_object_destroy;
        let _: unsafe extern "C" fn(HsaCodeObject, u32, *mut c_void) -> Status =
            hsa_code_object_get_info;
        let _: unsafe extern "C" fn(HsaCodeObject, *const c_char, *mut HsaCodeSymbol) -> Status =
            hsa_code_object_get_symbol;
        let _: unsafe extern "C" fn(
            HsaCodeObject,
            *const c_char,
            *const c_char,
            *mut HsaCodeSymbol,
        ) -> Status = hsa_code_object_get_symbol_from_name;
        let _: unsafe extern "C" fn(HsaCodeSymbol, u32, *mut c_void) -> Status =
            hsa_code_symbol_get_info;
        let _: unsafe extern "C" fn(
            HsaCodeObject,
            CodeObjectSymbolCallback,
            *mut c_void,
        ) -> Status = hsa_code_object_iterate_symbols;
        let _: unsafe extern "C" fn(u32, u32, *const c_char, *mut HsaExecutable) -> Status =
            hsa_executable_create;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            HsaAgent,
            HsaCodeObject,
            *const c_char,
        ) -> Status = hsa_executable_load_code_object;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            HsaCodeObjectReader,
            *const c_char,
            *mut HsaLoadedCodeObject,
        ) -> Status = hsa_executable_load_program_code_object;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            *const c_char,
            *const c_char,
            HsaAgent,
            i32,
            *mut HsaExecutableSymbol,
        ) -> Status = hsa_executable_get_symbol;
        let _: unsafe extern "C" fn(HsaExecutable, *const c_char, *mut c_void) -> Status =
            hsa_executable_global_variable_define;
        let _: unsafe extern "C" fn(HsaExecutable, HsaAgent, *const c_char, *mut c_void) -> Status =
            hsa_executable_readonly_variable_define;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            ExecutableSymbolCallback,
            *mut c_void,
        ) -> Status = hsa_executable_iterate_symbols;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            HsaAgent,
            AgentExecutableSymbolCallback,
            *mut c_void,
        ) -> Status = hsa_executable_iterate_agent_symbols;
        let _: unsafe extern "C" fn(
            HsaExecutable,
            ExecutableSymbolCallback,
            *mut c_void,
        ) -> Status = hsa_executable_iterate_program_symbols;
    }

    #[test]
    fn public_loader_constants_match_hsa_headers() {
        assert_eq!(INVALID_CODE_SYMBOL, 0x1018);
        assert_eq!(INVALID_EXECUTABLE_SYMBOL, 0x1019);
        assert_eq!(INVALID_CODE_OBJECT_READER, 0x1021);
        assert_eq!(SYMBOL_KIND_VARIABLE, 0);
        assert_eq!(SYMBOL_KIND_KERNEL, 1);
        assert_eq!(SYMBOL_KIND_INDIRECT_FUNCTION, 2);
    }

    #[test]
    fn parses_gfx1201_code_object_metadata_and_symbols() {
        let bytes = gfx1201_code_object();
        let layout = parse_elf_layout(&bytes);
        assert!(layout.is_ok());
        let Some(layout) = layout.ok() else {
            return;
        };
        assert_eq!(code_object_version(layout), Ok(6));
        assert_eq!(code_object_target(layout.flags), Some(("gfx1201", 0)));
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
    fn applies_relative_and_defined_absolute_relocations() {
        const SECTION_ENTRY_SIZE: usize = 64;
        const SECTION_COUNT: usize = 3;
        const SYMBOL_OFFSET: usize = SECTION_ENTRY_SIZE * SECTION_COUNT;
        const RELOCATION_OFFSET: usize = SYMBOL_OFFSET + 48;
        let mut bytes = vec![0_u8; RELOCATION_OFFSET + 48];

        let symbol_section = SECTION_ENTRY_SIZE;
        write_u32(&mut bytes, symbol_section + 4, SECTION_DYNSYM);
        write_u64(&mut bytes, symbol_section + 24, SYMBOL_OFFSET as u64);
        write_u64(&mut bytes, symbol_section + 32, 48);
        write_u64(&mut bytes, symbol_section + 56, 24);
        write_u16(&mut bytes, SYMBOL_OFFSET + 24 + 6, 1);
        write_u64(&mut bytes, SYMBOL_OFFSET + 24 + 8, 0x140);

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
        assert_eq!(
            apply_relocations(
                &bytes,
                0,
                SECTION_ENTRY_SIZE,
                SECTION_COUNT,
                0x100,
                0x1000,
                &mut image,
            ),
            Ok(())
        );
        assert_eq!(read_u64(&image, 0x20), Some(0x1080));
        assert_eq!(read_u64(&image, 0x28), Some(0x1048));
    }

    #[test]
    fn reports_one_combined_segment_for_v2_code_objects() {
        let loads = [
            (0, 0, 0x17c0, 0x17c0),
            (0x1800, 0x2800, 0x6c8, 0x6c8),
            (0x1ec8, 0x3ec8, 0x70, 0x138),
            (0x1f38, 0x4f38, 4, 4),
        ];
        let segments = loaded_segments(&loads, 0x0010_0000, 0);
        assert!(segments.is_ok());
        let Some(segments) = segments.ok() else {
            return;
        };
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].storage_offset, 0);
        assert_eq!(segments[0].address, 0x0010_0000);
        assert_eq!(segments[0].size, 0x4f3c);
    }

    #[test]
    fn scopes_v2_code_object_symbols_to_the_load_agent() {
        let agent = HsaAgent { handle: 7 };
        let symbol_agent = loaded_symbol_agent(false, agent);
        assert_eq!(symbol_agent.handle, agent.handle);
    }

    #[test]
    fn variable_metadata_matches_rocr_section_flag_interpretation() {
        assert_eq!(variable_metadata(0), (VARIABLE_SEGMENT_GLOBAL, false));
        assert_eq!(
            variable_metadata(SECTION_FLAG_WRITE),
            (VARIABLE_SEGMENT_GLOBAL, true)
        );
        assert_eq!(
            variable_metadata(SECTION_FLAG_AMDGPU_HSA_READONLY),
            (VARIABLE_SEGMENT_READONLY, false)
        );
    }

    #[test]
    fn executable_symbol_lookup_checks_initialization_before_arguments() {
        assert!(lock().is_ok_and(|runtime| runtime.is_none()));
        // SAFETY: An uninitialized runtime returns before inspecting the
        // deliberately invalid input pointers.
        assert_eq!(
            unsafe {
                hsa_executable_get_symbol_by_name(
                    HsaExecutable { handle: 0 },
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            },
            NOT_INITIALIZED
        );
    }
}
