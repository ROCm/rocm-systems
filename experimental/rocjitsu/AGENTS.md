# rocjitsu Agent Guidelines

## Generated ISA Files

- Treat files under `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/` as generated output unless the file is one of the documented handwritten exceptions: `isa.h`, `insts.h`, `mfma_exec.h`, `addr_calc.h`, or `addr_calc.cpp`.
- Do not hand-edit generated ISA files for feature work or bug fixes. Change the generator inputs or codegen logic first, then regenerate the ISA files.
- If a requested change appears to require editing generated ISA output directly, call that out and prefer updating the MR ISA XML source or the `amdisa` generator instead.
- To regenerate all ISA C++ files, run `bash scripts/regenerate_isa.sh`. This downloads nothing — it reads the XML specs already present under `third_party/machine-readable-isa/` and invokes `amdisa --multi` with `--gen-all --gen-shared-execute`, then auto-formats the output with clang-format.

## MR ISA XML Source

- `python scripts/download_machine_readable_isa.py --force` extracts the latest MR ISA XMLs under `third_party/machine-readable-isa/`.
- After downloading new XMLs, regenerate the ISA files: `bash scripts/regenerate_isa.sh`.