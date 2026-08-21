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

Create or activate a virtual environment, then install in editable mode from
the rocjitsu project root together with pre-commit:

```bash
python -m pip install -e lib/python/ pre-commit
```

The helper deliberately takes both Python and pre-commit from the same active
virtual environment. It prepends this checkout's `lib/python/` to `PYTHONPATH`,
so the generator implementation comes from the same checkout as the helper
while its installed dependencies come from the environment.

## MR ISA location

```
rocm-systems/shared/machine-readable-isa/isa/
```

## Generated file locations

| Generated files | Location | Generator |
|---|---|---|
| ISA decoders, encoders, execute bodies, and `insts.h` | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/<output-directory>/` | `codegen.py` |
| Shared execute templates | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/shared/` | `codegen.py` |
| Cross-ISA legalization tables | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `legalization_codegen.py` |
| Encoding decode/encode functions | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `encoding_translator_codegen.py` |

CDNA5's logical generator key and configuration architecture are `cdna5`. Its
MR ISA name and filename, concrete GPU/runtime target, ELF identity, and public
DBT target identity intentionally remain `gfx1250`. Its filesystem
directories, generated and hand-written C++ namespace (`rocjitsu::cdna5`), and
internal CMake provider targets use `cdna5`.

Hand-written per-ISA files (`isa.h`, `mma_exec.h`, `addr_calc.h/.cpp`) remain
under `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/<output-directory>/` and are
not overwritten by the generator. CDNA5 hand-written files are under
`arch/amdgpu/cdna5/`. Hand-written shared headers remain under
`lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/`.

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

<!-- \NPI new ISA family: add a `<isa>:$MRISA/amdgpu_isa_<isa>.xml` entry to \
     each manual `--multi` invocation below and to the supported-ISA list in \
     scripts/generate-amdisa.sh. -->

## Regenerating everything

The repository helper derives the repository, shared MR ISA, and generated
output directories from its own location. Its argument is the CDNA5 MR ISA XML
with the concrete filename `amdgpu_isa_gfx1250.xml`; the logical generator key
used by the command is `cdna5`.
Activate a Python virtual environment containing the generator dependencies and
`pre-commit`, then run this command from the `rocm-systems` repository root:

```bash
./emulation/rocjitsu/scripts/generate-amdisa.sh \
  /path/to/amdgpu_isa_gfx1250.xml
```

The helper can be invoked from any working directory when given by an
appropriate relative or absolute path. It discovers the active environment
through `VIRTUAL_ENV` or the active Python interpreter, then formats changed
generated files through the repository's pre-commit configuration.

The manual commands below are useful for focused generator development and are
run from the rocjitsu project root. Set `MRISA` to the shared MR ISA directory
and `CDNA5_MRISA` to the out-of-tree CDNA5 MR ISA XML directory:

```bash
MRISA=../../shared/machine-readable-isa/isa
CDNA5_MRISA=/path/to/cdna5-mrisa

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
    cdna5:$CDNA5_MRISA/amdgpu_isa_gfx1250.xml \
  --isa-output lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated \
  --dbt-output lib/rocjitsu/src/rocjitsu/code/dbt/generated

find lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated lib/rocjitsu/src/rocjitsu/code/dbt/generated \
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
    cdna5:$CDNA5_MRISA/amdgpu_isa_gfx1250.xml \
  --gen-isas \
  --isa-output lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated

find lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated \
  \( -name '*.cpp' -o -name '*.h' \) -exec clang-format -i {} +
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
    cdna5:$CDNA5_MRISA/amdgpu_isa_gfx1250.xml \
  --gen-dbt \
  --dbt-output lib/rocjitsu/src/rocjitsu/code/dbt/generated

find lib/rocjitsu/src/rocjitsu/code/dbt/generated \
  \( -name '*.cpp' -o -name '*.h' \) -exec clang-format -i {} +
```

## Workflow

When modifying ISA semantics or adding instruction support:

1. Edit the authoritative Python input: the generator orchestration in
   `lib/python/amdisa/codegen/_generator.py`, execution emitters under
   `lib/python/amdisa/codegen/execute/`, instruction classification in
   `lib/python/amdisa/semantics.py`, or compatibility inventory logic in
   `lib/python/amdisa/parser.py` as appropriate. Never edit generated C++.
2. Regenerate with `scripts/generate-amdisa.sh` or `--multi` as shown above
3. If you regenerated manually, format the generated files with `clang-format`
   (the helper formats changed generated files for you)
4. Stage ALL generated files before committing
