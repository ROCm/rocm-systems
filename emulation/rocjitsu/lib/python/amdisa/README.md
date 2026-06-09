# amdisa — AMD GPU ISA Code Generation Pipeline

Python library for parsing AMD Machine-Readable ISA (MR ISA) XML specifications
and generating C++ source files for the rocjitsu project.

## Modules

| Module | Purpose |
|---|---|
| `parser.py` | Parse MR ISA XML specs into `IsaSpec` objects |
| `gpuisa.py` | Core data structures (`IsaSpec`, `Instruction`, `InstEncoding`, `Operand`) |
| `isa_profile.py` | Per-ISA profile constants and encoding rules |
| `semantics.py` | Derive instruction semantics from mnemonics |
| `cross_isa.py` | Cross-ISA instruction overlap analysis |
| `codegen.py` | Generate C++ decoders, encoders, and instruction execute bodies |
| `legalization.py` | Generate cross-ISA legalization tables (Action classification) |
| `legalization_codegen.py` | Emit C++20 `InstructionLegalization[]` legalization table headers |
| `encoding_translator_codegen.py` | Emit C++20 neutral field structs + decode/encode functions using `machine_insts.h` typed structs |

## Installation

Install in editable mode (from the rocjitsu project root):

```bash
pip install -e lib/python/
```

This installs `amdisa` and its dependency (`cgen`) into your active virtualenv.

## Usage

All commands are run from the rocjitsu project root:

```bash
export MRISA=/path/to/mrisa  # directory containing amdgpu_isa_*.xml files
```

### Regenerate ISA decoders/encoders (all AMDGPU ISAs)

Use the convenience script from the rocjitsu project root:

```bash
bash tmp/gen-amdgpu.sh
```

This regenerates all ISA files (decoders, encoders, instruction classes, shared execute
templates) for all AMDGPU ISAs into `isa/arch/amdgpu/`. See `tmp/gen-amdgpu.sh` for the
underlying `python3 -m amdisa --multi --gen-all --gen-shared-execute` invocation.

### Regenerate DBT files (legalization tables + encoding translators)

Use the convenience script from the rocjitsu project root:

```bash
bash tmp/gen_dbt.sh
```

This generates legalization tables (27 ISA pairs) and encoding translators
(currently cdna4→rdna4) into `code/dbt/generated/`, then runs clang-format.

See `tmp/gen_dbt.sh` for the underlying `python3 -m amdisa` invocations.

## Generated file locations

| Generated files | Location | Generator |
|---|---|---|
| ISA decoders, encoders, instruction bodies | `isa/arch/amdgpu/<isa>/` | `codegen.py` |
| Cross-ISA legalization tables | `code/dbt/generated/` | `legalization_codegen.py` |
| Encoding decode/encode functions | `code/dbt/generated/` | `encoding_translator_codegen.py` |

The encoding translator engine (`code/dbt/encoding_translator.h`) is hand-written
and shared across all ISA pairs. Only the per-pair decode/encode functions and neutral field structs are auto-generated.
