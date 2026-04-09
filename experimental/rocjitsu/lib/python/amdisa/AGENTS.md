# amdisa Agent Guidelines

## Overview

`amdisa` is an AMD machine-readable ISA specification parser and C++ code generator. It reads XML ISA specs, derives instruction semantics from mnemonics, and emits C++ headers/sources for instruction decoders, encoding classes, operand types, and execute() bodies across 9 ISA variants (CDNA1–4, RDNA1–4, RDNA3.5).

## Module Responsibilities

| Module | Role |
|--------|------|
| `isa_profile.py` | ISA-specific traits — encoding rules, coherency models, hardware constants, wait-counter policy. All ISA knowledge belongs here. |
| `gpuisa.py` | Core data model — `IsaSpec`, `InstEncoding`, `Instruction`, `Operand`, `MicrocodeField`. ISA-agnostic. |
| `parser.py` | XML spec parser — reads machine-readable XML specs using an `IsaProfile` to handle per-ISA quirks. |
| `semantics.py` | Derives `InstructionSemantics` (semantic class, operation, data type) from mnemonic names and encoding formats. |
| `codegen.py` | C++ code generator — `CodeGenerator` emits all generated files. Uses profile traits for ISA-dependent output. |
| `cross_isa.py` | Cross-ISA deduplication — `CrossIsaAnalyzer` classifies instructions as universal, family-shared, or ISA-exclusive. |
| `xml_schema.py` | XML element/attribute name constants and schema validation. |

## Key Design Rules

- **ISA traits live in `isa_profile.py`**, not in the codegen or parser. When adding support for a new ISA generation or changing ISA-specific behavior (coherency models, wait counters, encoding modifiers, wave size, field renames), update or create a profile subclass.
- **Semantic inference from mnemonics**: there is no separate semantics XML. `semantics.py` derives execution behavior from instruction names and encoding formats. Profile `semantic_overrides` handle exceptions.
- **Parser workarounds**: the XML specs contain known bugs (missing fields, typos, dual-opcode formats). Workarounds are documented in profile class docstrings and parser comments — preserve them.
- **Generated C++ is never hand-edited**. Change the generator or inputs, then regenerate. See the parent `AGENTS.md` for the regeneration workflow.

## Testing

Run unit tests from the rocjitsu root:

```bash
python -m pytest lib/python/amdisa/tests/ -v
```

Tests cover cross-ISA analysis, encoding condition dedup, reserved field synthesis, encoding mask parsing, and ISA profile properties.

## Regeneration

Use `./scripts/regenerate_isa.sh` to regenerate all ISA C++ files. See the parent `AGENTS.md` for details.
