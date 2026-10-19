// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! AMD HSA ELF identity, target compatibility, and symbol interpretation.

use rocddi::topology::GpuInfo;
use rocelf::constants::{
    ELF_MACHINE_AMDGPU, ELF_OS_ABI_AMDGPU_HSA, ELF_TYPE_DYN, ELF_TYPE_EXEC, SECTION_DYNSYM,
    SECTION_FLAG_AMDGPU_HSA_AGENT, SECTION_FLAG_AMDGPU_HSA_READONLY, SECTION_FLAG_WRITE,
    SECTION_SYMTAB, SYMBOL_BINDING_GLOBAL, SYMBOL_COMMON, SYMBOL_TYPE_AMDGPU_HSA_INDIRECT_FUNCTION,
    SYMBOL_TYPE_COMMON, SYMBOL_TYPE_OBJECT, SYMBOL_UNDEFINED,
};
use rocelf::{Layout, read_u16, read_u32};

use crate::Error;

const AMDGPU_MACHINE_MASK: u32 = 0xff;
const AMDGPU_MACHINE_GFX1200: u32 = 0x48;
const AMDGPU_MACHINE_GFX1250: u32 = 0x49;
const AMDGPU_MACHINE_GFX1201: u32 = 0x4e;
const AMDGPU_MACHINE_GFX12_GENERIC: u32 = 0x59;
const AMDGPU_MACHINE_GFX12_5_GENERIC: u32 = 0x5b;
const AMDGPU_MACHINE_GFX1250_STRICT: u32 = 0xeb;
const AMDGPU_XNACK_MASK: u32 = 0x300;
const AMDGPU_XNACK_ANY: u32 = 0x100;
const AMDGPU_SRAMECC_MASK: u32 = 0xc00;
const AMDGPU_SRAMECC_ANY: u32 = 0x400;
const AMDGPU_GENERIC_VERSION_SHIFT: u32 = 24;

/// GFX identity and ASIC revision of one activated GPU.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Target {
    /// Major GFX version.
    pub gfx_major: u32,
    /// Minor GFX version.
    pub gfx_minor: u32,
    /// GFX stepping.
    pub gfx_stepping: u32,
    /// ASIC revision used by the gfx1250 compatibility rule.
    pub asic_revision: u32,
}

impl From<&GpuInfo> for Target {
    fn from(info: &GpuInfo) -> Self {
        Self {
            gfx_major: info.gfx_major,
            gfx_minor: info.gfx_minor,
            gfx_stepping: info.gfx_stepping,
            asic_revision: info.asic_revision,
        }
    }
}

/// A validated AMD HSA ELF header and its borrowed source bytes.
pub struct CodeObject<'a> {
    /// Immutable caller-owned ELF bytes.
    bytes: &'a [u8],
    /// Checked ELF header and table bounds.
    layout: Layout,
}

/// Preferred table when both dynamic and regular ELF symbols are present.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolTableOrder {
    /// Prefer dynamic symbols used for executable loading.
    DynamicFirst,
    /// Prefer regular symbols used for code-object inspection.
    RegularFirst,
}

impl SymbolTableOrder {
    fn section_types(self) -> [u32; 2] {
        match self {
            Self::DynamicFirst => [SECTION_DYNSYM, SECTION_SYMTAB],
            Self::RegularFirst => [SECTION_SYMTAB, SECTION_DYNSYM],
        }
    }
}

impl<'a> CodeObject<'a> {
    /// Parses the ELF header and checks the AMD HSA object identity.
    ///
    /// # Errors
    /// Returns `InvalidObject` for a malformed or non-HSA ELF object.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, Error> {
        let layout = parse_code_object_layout(bytes)?;
        Ok(Self { bytes, layout })
    }

    /// Returns the checked ELF header.
    #[must_use]
    pub fn layout(&self) -> Layout {
        self.layout
    }

    /// Returns the AMD HSA code-object version.
    ///
    /// # Errors
    /// Returns `InvalidObject` for an unknown ABI version.
    pub fn version(&self) -> Result<u32, Error> {
        code_object_version(self.layout)
    }

    /// Returns the known machine name and generic-ISA variant.
    ///
    /// # Errors
    /// Returns `InvalidTarget` for an unknown machine ID.
    pub fn target(&self) -> Result<(&'static str, u64), Error> {
        code_object_target(self.layout.flags).ok_or(Error::InvalidTarget)
    }

    /// Interprets symbols from the first available preferred table.
    ///
    /// # Errors
    /// Returns `InvalidObject` for malformed symbol or string data.
    pub fn symbols(&self, order: SymbolTableOrder) -> Result<Vec<Symbol>, Error> {
        parse_symbols(self.bytes, self.layout, order)
    }

    /// Checks whether this object is executable on a selected target.
    ///
    /// # Errors
    /// Returns `InvalidTarget` for unknown machine IDs, `InvalidObject` for
    /// unknown ABI versions, or `IncompatibleTarget` for a known mismatch.
    pub fn validate_target(&self, target: Target) -> Result<(), Error> {
        validate_target(self.layout, target)
    }
}

/// Parses ELF64 and requires the AMDGPU HSA machine, OS ABI, and object kind.
///
/// # Errors
/// Returns `InvalidObject` for a malformed or non-HSA ELF object.
fn parse_code_object_layout(bytes: &[u8]) -> Result<Layout, Error> {
    let layout = rocelf::parse_elf64(bytes)?;
    if layout.os_abi != ELF_OS_ABI_AMDGPU_HSA
        || !matches!(layout.kind, ELF_TYPE_EXEC | ELF_TYPE_DYN)
        || layout.machine != ELF_MACHINE_AMDGPU
    {
        return Err(Error::InvalidObject);
    }
    Ok(layout)
}

/// Returns the AMD HSA code-object version described by an ELF ABI byte.
///
/// # Errors
/// Returns `InvalidObject` for an unknown ABI version.
fn code_object_version(layout: Layout) -> Result<u32, Error> {
    match layout.abi_version {
        0 => Ok(2),
        1 => Ok(3),
        2 => Ok(4),
        3 => Ok(5),
        4 => Ok(6),
        _ => Err(Error::InvalidObject),
    }
}

/// Returns the target name and generic-ISA variant for a known machine ID.
#[must_use]
fn code_object_target(machine: u32) -> Option<(&'static str, u64)> {
    match machine & AMDGPU_MACHINE_MASK {
        AMDGPU_MACHINE_GFX1200 => Some(("gfx1200", 0)),
        AMDGPU_MACHINE_GFX1250 => Some(("gfx1250", 0)),
        AMDGPU_MACHINE_GFX1201 => Some(("gfx1201", 0)),
        AMDGPU_MACHINE_GFX12_GENERIC => Some(("gfx12-generic", 1)),
        AMDGPU_MACHINE_GFX12_5_GENERIC => Some(("gfx12-5-generic", 1)),
        AMDGPU_MACHINE_GFX1250_STRICT => Some(("gfx1250-strict", 0)),
        _ => None,
    }
}

/// Checks code-object flags against the selected gfx12 GPU.
///
/// # Errors
/// Returns `InvalidTarget` for unknown machine IDs, `InvalidObject` for
/// unknown ABI versions, or `IncompatibleTarget` for a known mismatch.
fn validate_target(layout: Layout, target: Target) -> Result<(), Error> {
    let version = code_object_version(layout)?;
    let Some((_, variant)) = code_object_target(layout.flags) else {
        return Err(Error::InvalidTarget);
    };
    let gfx = (target.gfx_major, target.gfx_minor, target.gfx_stepping);
    let matches_gpu = match layout.flags & AMDGPU_MACHINE_MASK {
        AMDGPU_MACHINE_GFX1200 => gfx == (12, 0, 0),
        AMDGPU_MACHINE_GFX1201 => gfx == (12, 0, 1),
        AMDGPU_MACHINE_GFX12_GENERIC => (gfx.0, gfx.1) == (12, 0) && matches!(gfx.2, 0 | 1),
        AMDGPU_MACHINE_GFX1250 => gfx == (12, 5, 0) && target.asic_revision != 0,
        AMDGPU_MACHINE_GFX1250_STRICT => gfx == (12, 5, 0) && target.asic_revision == 0,
        AMDGPU_MACHINE_GFX12_5_GENERIC => gfx == (12, 5, 0),
        _ => false,
    };
    let features_supported = if version == 3 {
        layout.flags & (AMDGPU_XNACK_MASK | AMDGPU_SRAMECC_MASK) == 0
    } else {
        matches!(layout.flags & AMDGPU_XNACK_MASK, 0 | AMDGPU_XNACK_ANY)
            && matches!(layout.flags & AMDGPU_SRAMECC_MASK, 0 | AMDGPU_SRAMECC_ANY)
    };
    if !features_supported
        || (variant == 1 && (version < 6 || layout.flags >> AMDGPU_GENERIC_VERSION_SHIFT == 0))
        || !matches_gpu
    {
        return Err(Error::IncompatibleTarget);
    }
    Ok(())
}

/// Kind of an AMD HSA code-object symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolKind {
    /// Kernel descriptor.
    Kernel,
    /// Variable.
    Variable,
    /// Indirect function.
    IndirectFunction,
}

/// Visibility scope of a code-object symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SymbolLinkage {
    /// Module-local symbol.
    Module,
    /// Program-wide symbol.
    Program,
}

/// Allocation scope of a variable symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariableAllocation {
    /// Agent-owned variable.
    Agent,
    /// Program-owned variable.
    Program,
}

/// Storage segment of a variable symbol.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariableSegment {
    /// Global segment.
    Global,
    /// Read-only segment.
    Readonly,
}

/// Interpreted AMD HSA symbol before or after relocation.
#[derive(Clone, Debug)]
pub struct Symbol {
    /// Original symbol name including its optional module prefix.
    pub full_name: String,
    /// Symbol name without the module prefix.
    pub name: String,
    /// Optional module prefix.
    pub module_name: String,
    /// Original virtual value from the ELF symbol table.
    pub value: u64,
    /// Device address after loading; zero before loading.
    pub address: u64,
    /// Symbol size, saturated to the HSA ABI field width.
    pub size: u32,
    /// Symbol kind.
    pub kind: SymbolKind,
    /// Linkage scope.
    pub linkage: SymbolLinkage,
    /// Whether the code object defines the symbol.
    pub is_definition: bool,
    /// Variable allocation scope.
    pub allocation: VariableAllocation,
    /// Variable storage segment.
    pub segment: VariableSegment,
    /// Alignment, saturated to the HSA ABI field width.
    pub alignment: u32,
    /// HSA const-query value derived from the section write flag as in `ROCr`.
    pub is_const: bool,
    /// Kernel argument segment size.
    pub kernarg_size: u32,
    /// Kernel argument segment alignment.
    pub kernarg_alignment: u32,
    /// Kernel group segment size.
    pub group_size: u32,
    /// Kernel private segment size.
    pub private_size: u32,
    /// Whether the kernel uses a dynamic call stack.
    pub dynamic_callstack: bool,
    /// Indirect call convention when provided.
    pub call_convention: u32,
    /// Kernel wavefront size for the target family.
    pub wavefront_size: u32,
}

impl Symbol {
    /// Creates an externally defined variable for an executable's symbol table.
    #[must_use]
    pub fn external_variable(
        name: &str,
        address: u64,
        allocation: VariableAllocation,
        segment: VariableSegment,
    ) -> Self {
        Self {
            full_name: name.to_owned(),
            name: name.to_owned(),
            module_name: String::new(),
            value: 0,
            address,
            size: 0,
            kind: SymbolKind::Variable,
            linkage: SymbolLinkage::Program,
            is_definition: true,
            allocation,
            segment,
            alignment: 0,
            is_const: segment == VariableSegment::Readonly,
            kernarg_size: 0,
            kernarg_alignment: 0,
            group_size: 0,
            private_size: 0,
            dynamic_callstack: false,
            call_convention: 0,
            wavefront_size: 0,
        }
    }
}

fn split_symbol_name(full_name: &str) -> (&str, &str) {
    full_name.rfind("::").map_or(("", full_name), |separator| {
        (&full_name[..separator], &full_name[separator + 2..])
    })
}

fn variable_metadata(section_flags: u64) -> (VariableSegment, bool) {
    let segment = if section_flags & SECTION_FLAG_AMDGPU_HSA_READONLY != 0 {
        VariableSegment::Readonly
    } else {
        VariableSegment::Global
    };
    (segment, section_flags & SECTION_FLAG_WRITE != 0)
}

/// Interprets AMD HSA symbols from the first available preferred ELF table.
///
/// # Errors
/// Returns `InvalidObject` for malformed symbol or string data.
fn parse_symbols(
    bytes: &[u8],
    layout: Layout,
    order: SymbolTableOrder,
) -> Result<Vec<Symbol>, Error> {
    let entries = rocelf::amdgpu_symbols(bytes, layout, &order.section_types())?;
    let wavefront_size = if matches!(
        layout.flags & AMDGPU_MACHINE_MASK,
        AMDGPU_MACHINE_GFX1200
            | AMDGPU_MACHINE_GFX1201
            | AMDGPU_MACHINE_GFX1250
            | AMDGPU_MACHINE_GFX12_GENERIC
            | AMDGPU_MACHINE_GFX12_5_GENERIC
            | AMDGPU_MACHINE_GFX1250_STRICT
    ) {
        32
    } else {
        64
    };
    let mut symbols = Vec::new();
    for symbol in entries {
        let kind = if symbol.is_kernel {
            SymbolKind::Kernel
        } else if symbol.symbol_type == SYMBOL_TYPE_AMDGPU_HSA_INDIRECT_FUNCTION {
            SymbolKind::IndirectFunction
        } else if matches!(symbol.symbol_type, SYMBOL_TYPE_OBJECT | SYMBOL_TYPE_COMMON) {
            SymbolKind::Variable
        } else {
            continue;
        };
        let is_definition = symbol.section_index != SYMBOL_UNDEFINED
            && symbol.section_index != SYMBOL_COMMON
            && symbol.symbol_type != SYMBOL_TYPE_COMMON;
        let (segment, is_const) = variable_metadata(symbol.section_flags);
        let (group_size, private_size, kernarg_size, dynamic_callstack) = symbol
            .kernel_descriptor
            .map_or((0, 0, 0, false), |descriptor| {
                (
                    read_u32(descriptor, 0).unwrap_or(0),
                    read_u32(descriptor, 4).unwrap_or(0),
                    read_u32(descriptor, 8).unwrap_or(0),
                    read_u16(descriptor, 56).is_some_and(|flags| flags & (1 << 11) != 0),
                )
            });
        let (module_name, name) = split_symbol_name(symbol.name);
        symbols.push(Symbol {
            full_name: symbol.name.to_owned(),
            name: name.to_owned(),
            module_name: module_name.to_owned(),
            value: symbol.value,
            address: 0,
            size: u32::try_from(symbol.size).unwrap_or(u32::MAX),
            kind,
            linkage: if symbol.binding == SYMBOL_BINDING_GLOBAL {
                SymbolLinkage::Program
            } else {
                SymbolLinkage::Module
            },
            is_definition,
            allocation: if symbol.section_flags & SECTION_FLAG_AMDGPU_HSA_AGENT != 0 {
                VariableAllocation::Agent
            } else {
                VariableAllocation::Program
            },
            segment,
            alignment: u32::try_from(symbol.alignment).unwrap_or(0),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn variable_metadata_matches_rocr_section_flag_interpretation() {
        assert_eq!(variable_metadata(0), (VariableSegment::Global, false));
        assert_eq!(
            variable_metadata(SECTION_FLAG_WRITE),
            (VariableSegment::Global, true)
        );
        assert_eq!(
            variable_metadata(SECTION_FLAG_AMDGPU_HSA_READONLY),
            (VariableSegment::Readonly, false)
        );
    }
}
