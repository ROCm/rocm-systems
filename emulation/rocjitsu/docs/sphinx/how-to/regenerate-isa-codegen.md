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
| ISA decoders, encoders, execute bodies | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/<isa>/` | `codegen.py` |
| Shared execute templates | `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/generated/shared/` | `codegen.py` |
| Cross-ISA legalization tables | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `legalization_codegen.py` |
| Encoding decode/encode functions | `lib/rocjitsu/src/rocjitsu/code/dbt/generated/` | `encoding_translator_codegen.py` |

Hand-written files (`isa.h`, `insts.h`, `mma_exec.h`, `addr_calc.h/.cpp`) are
not overwritten by the generator.

## CLI reference

```text
python -m amdisa [--isa-additions NAME:XML]
                 [--isa-variants NAME:JSON]
                 [--gen-isas] [--gen-dbt]
                 [--isa-output DIR] [--include-root DIR]
                 [--dbt-output DIR] [NAME:]XML ...
```

| Option | Description |
|--------|-------------|
| `[NAME:]XML ...` | Parse one or more ISA XMLs. A recognized name selects its semantic profile. |
| `--isa-additions NAME:XML` | Apply an ISA additions XML file to the named ISA. May be repeated. |
| `--isa-variants NAME:JSON` | Attach one target-feature and legality manifest to the named ISA. |
| `--gen-isas` | Generate ISA C++ files (decoders, encodings, execute bodies). Enabled by default. |
| `--gen-dbt` | Generate DBT legalization tables and encoding translators. Enabled by default. |
| `--isa-output DIR` | Output path for generated ISA C++ files. |
| `--include-root DIR` | Compiler include root used to spell relocatable generated includes. |
| `--dbt-output DIR` | Output directory for DBT tables. Defaults to `--isa-output`. |

When neither `--gen-isas` nor `--gen-dbt` is specified, both are generated.

## Regenerate everything

The repository helper is the authoritative whole-tree workflow. It derives the
checked-in MR ISA inputs and output directories from its own location, verifies
the public CDNA5 gfx1251 extension provenance, applies the CDNA5 additions and
target-variant manifest, regenerates ISA and DBT output, and formats changed
generated files.

Activate a Python virtual environment containing the generator dependencies and
`pre-commit`, then run this command from the `rocm-systems` repository root:

```bash
./emulation/rocjitsu/scripts/generate-amdisa.sh
```

Use the lower-level CLI only for focused generator development. Preserve the
CDNA5 `--isa-additions` and `--isa-variants` inputs shown in the detailed
`docs/codegen.md` documentation when generating CDNA5 output.

## Workflow for modifying ISA semantics

1. Edit `lib/python/amdisa/codegen/_generator.py`. Never edit the generated C++
   files directly — they are overwritten on the next regeneration run.
2. Regenerate with `scripts/generate-amdisa.sh`.
3. Let the helper format changed generated files through `pre-commit`.
4. Stage all generated files before committing.
