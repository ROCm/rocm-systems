# ISA Code Generation

Instruction decoders, execution bodies, legalization tables, and encoding
translators are generated from the
[Machine-Readable ISA (MR ISA)](https://gpuopen.com/machine-readable-isa/)
XML specification via the `amdisa` Python library in `lib/python/amdisa/`.

## amdisa modules

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
| `encoding_translator_codegen.py` | Emit C++20 neutral field structs + decode/encode functions |

## Installation

Install in editable mode from the rocjitsu project root:

```bash
pip install -e lib/python/
```

This installs `amdisa` and its dependency (`cgen`) into your active
virtualenv.

## MR ISA location

```
rocm-systems/shared/machine-readable-isa/isa/
```

## Generated file locations

| Generated files | Location | Generator |
|---|---|---|
| ISA decoders, encoders, execute bodies | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/<isa>/` | `codegen.py` |
| Shared execute templates | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/` | `codegen.py` |
| Cross-ISA legalization tables | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `legalization_codegen.py` |
| Encoding decode/encode functions | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `encoding_translator_codegen.py` |

Hand-written files (`isa.h`, `insts.h`, `mma_exec.h`, `addr_calc.h/.cpp`)
are not overwritten by the generator.

The encoding translator engine (`code/dbt/encoding_translator.h`) is
hand-written and shared across all ISA pairs. Only the per-pair
decode/encode functions and neutral field structs are auto-generated.

## CLI reference

```
python -m amdisa [--multi NAME:XML ...] [--gen-isas] [--gen-dbt]
                 [--isa-output DIR] [--dbt-output DIR] [isafile]
```

| Option | Description |
|---|---|
| `--multi NAME:XML ...` | Multi-ISA mode: parse all XMLs and generate shared execute templates |
| `--gen-isas` | Generate ISA C++ files (decoders, encodings, execute bodies) |
| `--gen-dbt` | Generate DBT legalization tables and encoding translators |
| `--isa-output DIR` | Output path for generated ISA C++ files |
| `--dbt-output DIR` | Output directory for DBT tables (defaults to `--isa-output`) |

When neither `--gen-isas` nor `--gen-dbt` is specified, both are
generated.

## Regenerating everything

All commands are run from the rocjitsu project root. Set `MRISA` to the
shared MR ISA directory and `GFX1250_MRISA` to the out-of-tree gfx1250
XML directory:

```bash
MRISA=../../shared/machine-readable-isa/isa
GFX1250_MRISA=/path/to/gfx1250-mrisa

python -m amdisa \
  --multi \
    cdna1:$MRISA/amdgpu_isa_cdna1.xml \
    cdna2:$MRISA/amdgpu_isa_cdna2.xml \
    cdna3:$MRISA/amdgpu_isa_cdna3.xml \
    cdna4:$MRISA/amdgpu_isa_cdna4.xml \
    rdna1:$MRISA/amdgpu_isa_rdna1.xml \
    rdna2:$MRISA/amdgpu_isa_rdna2.xml \
    rdna3:$MRISA/amdgpu_isa_rdna3.xml \
    rdna3_5:$MRISA/amdgpu_isa_rdna3_5.xml \
    rdna4:$MRISA/amdgpu_isa_rdna4.xml \
    gfx1250:$GFX1250_MRISA/amdgpu_isa_gfx1250.xml \
  --isa-output lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu \
  --dbt-output lib/rocjitsu/src/rocjitsu/code/dbt/generated

find lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu lib/rocjitsu/src/rocjitsu/code/dbt/generated \
  \( -name '*.h' -o -name '*.cpp' \) -exec clang-format -i {} +
```

## Regenerating ISA files only

```bash
python -m amdisa \
  --multi \
    cdna1:$MRISA/amdgpu_isa_cdna1.xml \
    cdna2:$MRISA/amdgpu_isa_cdna2.xml \
    cdna3:$MRISA/amdgpu_isa_cdna3.xml \
    cdna4:$MRISA/amdgpu_isa_cdna4.xml \
    rdna1:$MRISA/amdgpu_isa_rdna1.xml \
    rdna2:$MRISA/amdgpu_isa_rdna2.xml \
    rdna3:$MRISA/amdgpu_isa_rdna3.xml \
    rdna3_5:$MRISA/amdgpu_isa_rdna3_5.xml \
    rdna4:$MRISA/amdgpu_isa_rdna4.xml \
    gfx1250:$GFX1250_MRISA/amdgpu_isa_gfx1250.xml \
  --gen-isas \
  --isa-output lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu

find lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu -name '*.cpp' -o -name '*.h' \
  | xargs clang-format -i
```

## Regenerating DBT files only

```bash
python -m amdisa \
  --multi \
    cdna1:$MRISA/amdgpu_isa_cdna1.xml \
    cdna2:$MRISA/amdgpu_isa_cdna2.xml \
    cdna3:$MRISA/amdgpu_isa_cdna3.xml \
    cdna4:$MRISA/amdgpu_isa_cdna4.xml \
    rdna1:$MRISA/amdgpu_isa_rdna1.xml \
    rdna2:$MRISA/amdgpu_isa_rdna2.xml \
    rdna3:$MRISA/amdgpu_isa_rdna3.xml \
    rdna3_5:$MRISA/amdgpu_isa_rdna3_5.xml \
    rdna4:$MRISA/amdgpu_isa_rdna4.xml \
    gfx1250:$GFX1250_MRISA/amdgpu_isa_gfx1250.xml \
  --gen-dbt \
  --dbt-output lib/rocjitsu/src/rocjitsu/code/dbt/generated

find lib/rocjitsu/src/rocjitsu/code/dbt/generated -name '*.cpp' -o -name '*.h' \
  | xargs clang-format -i
```

## Workflow

When modifying ISA semantics or adding instruction support:

1. Edit `lib/python/amdisa/codegen/_generator.py` (never the generated
   C++ files)
2. Regenerate with `--multi` as shown above
3. Format the generated files with `clang-format`
4. Stage ALL generated files before committing
