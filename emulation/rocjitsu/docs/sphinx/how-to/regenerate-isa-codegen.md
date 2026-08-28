---
myst:
    html_meta:
        "description": "How to regenerate ISA simulation and DBT source files from AMD Machine-Readable ISA XML using the rocJITsu amdisa Python library."
        "keywords": "rocJITsu, ROCm, ISA, codegen, DBT, amdisa, regenerate, code generation"
---

# Regenerate ISA and DBT source files

rocJITsu generates instruction decoders, execution bodies, legalization tables,
and encoding translators from the
[AMD Machine-Readable ISA (MR ISA)](https://gpuopen.com/machine-readable-isa/)
XML specification using the `amdisa` Python library in `lib/python/amdisa/`.

Run the generator after modifying ISA semantics, adding instruction support, or
pulling updated MR ISA XML files.

## Prerequisites

- Python 3.10 or later
- `amdisa` library installed in editable mode:

  ```bash
  pip install -e lib/python/
  ```

- MR ISA XML files at `../../shared/machine-readable-isa/isa/` relative to
  the rocJITsu project root (in the `rocm-systems` repository)
- `clang-format` on `PATH` for formatting generated output

## Generated file locations

| Output | Location | Generator |
|--------|----------|-----------|
| ISA decoders, encoders, execute bodies | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/<isa>/` | `codegen.py` |
| Shared execute templates | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/` | `codegen.py` |
| Cross-ISA legalization tables | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `legalization_codegen.py` |
| Encoding decode/encode functions | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `encoding_translator_codegen.py` |

Hand-written files (`isa.h`, `insts.h`, `mma_exec.h`, `addr_calc.h/.cpp`) are
not overwritten by the generator.

## CLI reference

```text
python -m amdisa [--multi NAME:XML ...] [--gen-isas] [--gen-dbt]
                 [--isa-output DIR] [--dbt-output DIR] [isafile]
```

| Option | Description |
|--------|-------------|
| `--multi NAME:XML ...` | Multi-ISA mode: parse all XMLs and generate shared execute templates. Each argument is `name:xml_path`. |
| `--gen-isas` | Generate ISA C++ files (decoders, encodings, execute bodies). Enabled by default. |
| `--gen-dbt` | Generate DBT legalization tables and encoding translators. Enabled by default. |
| `--isa-output DIR` | Output path for generated ISA C++ files. |
| `--dbt-output DIR` | Output directory for DBT tables. Defaults to `--isa-output`. |

When neither `--gen-isas` nor `--gen-dbt` is specified, both are generated.

## Regenerate everything

Run all commands from the rocJITsu project root. Set environment variables for
the MR ISA XML directories:

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

find lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu \
     lib/rocjitsu/src/rocjitsu/code/dbt/generated \
  \( -name '*.h' -o -name '*.cpp' \) -exec clang-format -i {} +
```

## Regenerate ISA files only

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

find lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu \
  \( -name '*.h' -o -name '*.cpp' \) -exec clang-format -i {} +
```

## Regenerate DBT files only

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

find lib/rocjitsu/src/rocjitsu/code/dbt/generated \
  \( -name '*.h' -o -name '*.cpp' \) -exec clang-format -i {} +
```

## Workflow for modifying ISA semantics

1. Edit `lib/python/amdisa/codegen/_generator.py`. Never edit the generated C++
   files directly — they are overwritten on the next regeneration run.
2. Regenerate using the `--multi` command above.
3. Format the output with `clang-format` as shown.
4. Stage all generated files before committing.
