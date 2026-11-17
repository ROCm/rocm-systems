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
const AMDGPU_MACHINE_GFX908: u32 = 0x30;
const AMDGPU_MACHINE_GFX90A: u32 = 0x3f;
const AMDGPU_MACHINE_GFX942: u32 = 0x4c;
const AMDGPU_MACHINE_GFX950: u32 = 0x4f;
const AMDGPU_MACHINE_GFX9_4_GENERIC: u32 = 0x5f;
const AMDGPU_MACHINE_GFX1200: u32 = 0x48;
const AMDGPU_MACHINE_GFX1250: u32 = 0x49;
const AMDGPU_MACHINE_GFX1201: u32 = 0x4e;
const AMDGPU_MACHINE_GFX12_GENERIC: u32 = 0x59;
const AMDGPU_MACHINE_GFX12_5_GENERIC: u32 = 0x5b;
const AMDGPU_MACHINE_GFX1250_STRICT: u32 = 0xeb;
const AMDGPU_XNACK_MASK: u32 = 0x300;
const AMDGPU_XNACK_ANY: u32 = 0x100;
const AMDGPU_XNACK_OFF: u32 = 0x200;
const AMDGPU_XNACK_ON: u32 = 0x300;
const AMDGPU_SRAMECC_MASK: u32 = 0xc00;
const AMDGPU_SRAMECC_ANY: u32 = 0x400;
const AMDGPU_SRAMECC_OFF: u32 = 0x800;
const AMDGPU_SRAMECC_ON: u32 = 0xc00;
const AMDGPU_GENERIC_VERSION_SHIFT: u32 = 24;

/// Exact ELF target identities from the primary AMDGPU HSA machine registry.
const TARGETS: &[(u32, &str, (u32, u32, u32))] = &[
    (AMDGPU_MACHINE_GFX908, "gfx908", (9, 0, 8)),
    (AMDGPU_MACHINE_GFX90A, "gfx90a", (9, 0, 10)),
    (AMDGPU_MACHINE_GFX942, "gfx942", (9, 4, 2)),
    (AMDGPU_MACHINE_GFX950, "gfx950", (9, 5, 0)),
    (0x33, "gfx1010", (10, 1, 0)),
    (0x34, "gfx1011", (10, 1, 1)),
    (0x35, "gfx1012", (10, 1, 2)),
    (0x42, "gfx1013", (10, 1, 3)),
    (0x36, "gfx1030", (10, 3, 0)),
    (0x37, "gfx1031", (10, 3, 1)),
    (0x38, "gfx1032", (10, 3, 2)),
    (0x39, "gfx1033", (10, 3, 3)),
    (0x3e, "gfx1034", (10, 3, 4)),
    (0x3d, "gfx1035", (10, 3, 5)),
    (0x45, "gfx1036", (10, 3, 6)),
    (0x41, "gfx1100", (11, 0, 0)),
    (0x46, "gfx1101", (11, 0, 1)),
    (0x47, "gfx1102", (11, 0, 2)),
    (0x44, "gfx1103", (11, 0, 3)),
    (0x43, "gfx1150", (11, 5, 0)),
    (0x4a, "gfx1151", (11, 5, 1)),
    (0x55, "gfx1152", (11, 5, 2)),
    (0x58, "gfx1153", (11, 5, 3)),
    (0x5d, "gfx1170", (11, 7, 0)),
    (0x5e, "gfx1171", (11, 7, 1)),
    (AMDGPU_MACHINE_GFX1200, "gfx1200", (12, 0, 0)),
    (AMDGPU_MACHINE_GFX1201, "gfx1201", (12, 0, 1)),
    (AMDGPU_MACHINE_GFX1250, "gfx1250", (12, 5, 0)),
    (0x5a, "gfx1251", (12, 5, 1)),
];

/// Whether modern code-object decoding is implemented for this exact GPU.
/// This does not qualify its native queue, scratch, copy or publication path.
#[must_use]
pub fn supports_target(target: Target) -> bool {
    TARGETS
        .iter()
        .any(|(_, _, gfx)| *gfx == (target.gfx_major, target.gfx_minor, target.gfx_stepping))
}

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
    /// Active process XNACK mode, or `None` when it was not queried.
    pub xnack_enabled: Option<bool>,
    /// SRAM ECC mode from GPU topology, or `None` for targets without it.
    pub sram_ecc_enabled: Option<bool>,
}

impl From<&GpuInfo> for Target {
    fn from(info: &GpuInfo) -> Self {
        Self {
            gfx_major: info.gfx_major,
            gfx_minor: info.gfx_minor,
            gfx_stepping: info.gfx_stepping,
            asic_revision: info.asic_revision,
            xnack_enabled: None,
            sram_ecc_enabled: (info.gfx_major == 9).then_some(info.sram_ecc_enabled),
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

/// HSA profile required by a code object or selected for an executable.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Profile {
    /// Base-profile GPU execution.
    Base,
    /// Full-profile execution.
    Full,
}

/// Default floating-point rounding policy at the HSA executable boundary.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FloatRoundingMode {
    /// Use the implementation's default mode.
    Default,
    /// Round toward zero.
    Zero,
    /// Round to nearest, ties to even.
    Near,
}

/// Execution properties interpreted from the supported code-object format.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CodeObjectContract {
    /// Required HSA profile.
    pub profile: Profile,
    /// Required default floating-point rounding mode.
    pub rounding: FloatRoundingMode,
}

impl CodeObjectContract {
    /// Checks executable policy before allocating or publishing an image.
    ///
    /// # Errors
    /// Returns `IncompatibleTarget` for a profile or rounding mismatch.
    pub fn validate(self, profile: Profile, rounding: FloatRoundingMode) -> Result<(), Error> {
        let rounding = match rounding {
            FloatRoundingMode::Default => FloatRoundingMode::Near,
            mode => mode,
        };
        if self.profile != profile
            || (self.rounding != FloatRoundingMode::Default && self.rounding != rounding)
        {
            return Err(Error::IncompatibleTarget);
        }
        Ok(())
    }
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

    /// Returns the execution contract of a modern AMDGPU HSA code object.
    #[must_use]
    pub fn contract(&self) -> CodeObjectContract {
        // V3 and later use the AMDGPU base-profile ABI. Legacy HSAIL notes and
        // amd_kernel_code_t are rejected by parse_code_object_layout below.
        CodeObjectContract {
            profile: Profile::Base,
            rounding: FloatRoundingMode::Near,
        }
    }

    /// Returns the logical loaded span, independent of device page size.
    ///
    /// # Errors
    /// Returns `InvalidObject` for malformed or unrepresentable load segments.
    pub fn load_size(&self) -> Result<usize, Error> {
        let plan = rocelf::LoadPlan::new(self.bytes, self.layout, super::image::CODE_ALIGNMENT)?;
        usize::try_from(plan.virtual_end() - plan.virtual_base()).map_err(|_| Error::InvalidObject)
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
        // V1/V2 require legacy notes, kernel descriptors and relocations. Do
        // not accept them through the modern decoder and fabricate metadata.
        || !matches!(layout.abi_version, 1..=4)
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
    let machine = machine & AMDGPU_MACHINE_MASK;
    if let Some((_, name, _)) = TARGETS.iter().find(|(id, _, _)| *id == machine) {
        return Some((name, 0));
    }
    match machine {
        AMDGPU_MACHINE_GFX9_4_GENERIC => Some(("gfx9-4-generic", 1)),
        0x52 => Some(("gfx10-1-generic", 1)),
        0x53 => Some(("gfx10-3-generic", 1)),
        0x54 => Some(("gfx11-generic", 1)),
        0x62 => Some(("gfx11-7-generic", 1)),
        AMDGPU_MACHINE_GFX12_GENERIC => Some(("gfx12-generic", 1)),
        AMDGPU_MACHINE_GFX12_5_GENERIC => Some(("gfx12-5-generic", 1)),
        AMDGPU_MACHINE_GFX1250_STRICT => Some(("gfx1250-strict", 0)),
        _ => None,
    }
}

fn feature_matches(value: u32, any: u32, off: u32, on: u32, enabled: Option<bool>) -> bool {
    match value {
        0 => true,
        value if value == any => true,
        value if value == off => enabled != Some(true),
        value if value == on => enabled != Some(false),
        _ => false,
    }
}

/// Checks code-object flags against the selected GPU.
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
    let matches_gpu = supports_target(target)
        && match layout.flags & AMDGPU_MACHINE_MASK {
            AMDGPU_MACHINE_GFX9_4_GENERIC => matches!(gfx, (9, 4, 2) | (9, 5, 0)),
            0x52 => (gfx.0, gfx.1) == (10, 1),
            0x53 => (gfx.0, gfx.1) == (10, 3),
            0x54 => gfx.0 == 11 && matches!(gfx.1, 0 | 5),
            0x62 => (gfx.0, gfx.1) == (11, 7),
            AMDGPU_MACHINE_GFX12_GENERIC => (gfx.0, gfx.1) == (12, 0),
            AMDGPU_MACHINE_GFX1250 => gfx == (12, 5, 0) && target.asic_revision != 0,
            AMDGPU_MACHINE_GFX1250_STRICT => gfx == (12, 5, 0) && target.asic_revision == 0,
            AMDGPU_MACHINE_GFX12_5_GENERIC => (gfx.0, gfx.1) == (12, 5),
            machine => TARGETS
                .iter()
                .any(|(id, _, identity)| *id == machine && *identity == gfx),
        };
    let xnack_supported = target.gfx_major == 9 || (target.gfx_major, target.gfx_minor) == (10, 1);
    let sramecc_supported = target.gfx_major == 9;
    let features_supported = if version == 3 {
        let xnack = layout.flags & 0x100 != 0;
        let sram_ecc = layout.flags & 0x200 != 0;
        (if xnack_supported {
            target.xnack_enabled.is_none_or(|enabled| enabled == xnack)
        } else {
            !xnack
        }) && (if sramecc_supported {
            target
                .sram_ecc_enabled
                .is_none_or(|enabled| enabled == sram_ecc)
        } else {
            !sram_ecc
        }) && layout.flags & 0xc00 == 0
    } else {
        (if xnack_supported {
            feature_matches(
                layout.flags & AMDGPU_XNACK_MASK,
                AMDGPU_XNACK_ANY,
                AMDGPU_XNACK_OFF,
                AMDGPU_XNACK_ON,
                target.xnack_enabled,
            )
        } else {
            matches!(layout.flags & AMDGPU_XNACK_MASK, 0 | AMDGPU_XNACK_ANY)
        }) && (if sramecc_supported {
            feature_matches(
                layout.flags & AMDGPU_SRAMECC_MASK,
                AMDGPU_SRAMECC_ANY,
                AMDGPU_SRAMECC_OFF,
                AMDGPU_SRAMECC_ON,
                target.sram_ecc_enabled,
            )
        } else {
            matches!(layout.flags & AMDGPU_SRAMECC_MASK, 0 | AMDGPU_SRAMECC_ANY)
        })
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
    let gfx_major = TARGETS
        .iter()
        .find(|(id, _, _)| *id == layout.flags & AMDGPU_MACHINE_MASK)
        .map_or_else(
            || match layout.flags & AMDGPU_MACHINE_MASK {
                0x52 | 0x53 => 10,
                0x54 | 0x62 => 11,
                AMDGPU_MACHINE_GFX12_GENERIC
                | AMDGPU_MACHINE_GFX12_5_GENERIC
                | AMDGPU_MACHINE_GFX1250_STRICT => 12,
                _ => 9,
            },
            |(_, _, gfx)| gfx.0,
        );
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
        if kind == SymbolKind::Kernel
            && (symbol.symbol_type == rocelf::constants::SYMBOL_TYPE_AMDGPU_HSA_KERNEL
                || (is_definition && symbol.kernel_descriptor.is_none()))
        {
            return Err(Error::InvalidObject);
        }
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
            wavefront_size: if gfx_major >= 12
                || (gfx_major >= 10
                    && symbol
                        .kernel_descriptor
                        .and_then(|descriptor| read_u16(descriptor, 56))
                        .is_some_and(|flags| flags & (1 << 10) != 0))
            {
                32
            } else {
                64
            },
        });
    }
    Ok(symbols)
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    fn modern_object(machine: u32) -> Vec<u8> {
        let mut bytes = vec![0_u8; 120];
        bytes[..4].copy_from_slice(b"\x7fELF");
        bytes[4] = 2;
        bytes[5] = 1;
        bytes[7] = ELF_OS_ABI_AMDGPU_HSA;
        bytes[8] = 4;
        bytes[16..18].copy_from_slice(&ELF_TYPE_DYN.to_le_bytes());
        bytes[18..20].copy_from_slice(&ELF_MACHINE_AMDGPU.to_le_bytes());
        bytes[20..24].copy_from_slice(&1_u32.to_le_bytes());
        bytes[32..40].copy_from_slice(&64_u64.to_le_bytes());
        bytes[48..52].copy_from_slice(&machine.to_le_bytes());
        bytes[52..54].copy_from_slice(&64_u16.to_le_bytes());
        bytes[54..56].copy_from_slice(&56_u16.to_le_bytes());
        bytes[56..58].copy_from_slice(&1_u16.to_le_bytes());
        bytes[64..68].copy_from_slice(&1_u32.to_le_bytes());
        bytes[80..88].copy_from_slice(&0x180_u64.to_le_bytes());
        bytes[96..104].copy_from_slice(&64_u64.to_le_bytes());
        bytes[104..112].copy_from_slice(&0x101_u64.to_le_bytes());
        bytes
    }

    fn target(gfx_major: u32, gfx_minor: u32, gfx_stepping: u32) -> Target {
        Target {
            gfx_major,
            gfx_minor,
            gfx_stepping,
            asic_revision: 1,
            xnack_enabled: Some(false),
            sram_ecc_enabled: Some(false),
        }
    }

    #[test]
    fn modern_contract_rejects_mismatched_executable_policy() {
        let bytes = modern_object(AMDGPU_MACHINE_GFX1201);
        let contract = CodeObject::parse(&bytes).unwrap().contract();
        assert!(
            contract
                .validate(Profile::Base, FloatRoundingMode::Default)
                .is_ok()
        );
        assert!(
            contract
                .validate(Profile::Base, FloatRoundingMode::Near)
                .is_ok()
        );
        assert!(
            contract
                .validate(Profile::Base, FloatRoundingMode::Zero)
                .is_err()
        );
        assert!(
            contract
                .validate(Profile::Full, FloatRoundingMode::Near)
                .is_err()
        );
    }

    #[test]
    fn legacy_abi_is_rejected_before_modern_descriptor_decoding() {
        let mut bytes = modern_object(AMDGPU_MACHINE_GFX908);
        bytes[8] = 0;
        assert!(matches!(
            CodeObject::parse(&bytes),
            Err(Error::InvalidObject)
        ));
    }

    #[test]
    fn load_size_excludes_native_page_padding() {
        let bytes = modern_object(AMDGPU_MACHINE_GFX1201);
        let object = CodeObject::parse(&bytes).unwrap();
        // Segment [0x180, 0x281) occupies [0x100, 0x281) at code alignment.
        assert_eq!(object.load_size().unwrap(), 0x181);
        let plan = rocelf::LoadPlan::new(&bytes, object.layout(), 256).unwrap();
        assert_eq!(plan.allocation_size(), 0x200);
        let mut image = vec![0xcc; plan.allocation_size()];
        plan.write_amdgpu_image(0x1000, &mut image, |_| None)
            .unwrap();
        assert_eq!(&image[0x80..0xc0], &bytes[..64]);
        assert!(image[0xc0..0x181].iter().all(|byte| *byte == 0));
    }

    #[test]
    fn exact_target_and_generic_compatibility_share_one_manifest() {
        for (machine, gfx) in [
            (0x33, (10, 1, 0)),
            (0x45, (10, 3, 6)),
            (0x58, (11, 5, 3)),
            (0x5e, (11, 7, 1)),
            (0x5a, (12, 5, 1)),
        ] {
            let bytes = modern_object(machine);
            let target = target(gfx.0, gfx.1, gfx.2);
            assert!(supports_target(target));
            assert!(
                CodeObject::parse(&bytes)
                    .unwrap()
                    .validate_target(target)
                    .is_ok()
            );
        }
        let bytes = modern_object(0x54 | (1 << AMDGPU_GENERIC_VERSION_SHIFT));
        let object = CodeObject::parse(&bytes).unwrap();
        assert!(object.validate_target(target(11, 5, 3)).is_ok());
        assert!(object.validate_target(target(11, 7, 1)).is_err());
        assert!(!supports_target(target(11, 5, 4)));
    }

    #[test]
    fn gfx101_xnack_requirements_match_the_active_process_mode() {
        let bytes = modern_object(0x33 | AMDGPU_XNACK_ON);
        let object = CodeObject::parse(&bytes).unwrap();
        let mut active = target(10, 1, 0);
        assert!(object.validate_target(active).is_err());
        active.xnack_enabled = Some(true);
        assert!(object.validate_target(active).is_ok());
        let bytes = modern_object(0x36 | AMDGPU_XNACK_ON);
        assert!(
            CodeObject::parse(&bytes)
                .unwrap()
                .validate_target(target(10, 3, 0))
                .is_err()
        );
    }

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
