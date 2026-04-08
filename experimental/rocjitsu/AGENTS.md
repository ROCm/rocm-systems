# rocjitsu Agent Guidelines

## Generated ISA Files

- Treat files under `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/` as generated output unless the file is one of the documented handwritten exceptions: `isa.h`, `insts.h`, `mfma_exec.h`, `addr_calc.h`, or `addr_calc.cpp`.
- Do not hand-edit generated ISA files for feature work or bug fixes. Change the generator inputs or codegen logic first, then regenerate the ISA files.
- If a requested change appears to require editing generated ISA output directly, call that out and prefer updating the MR ISA XML source or the `amdisa` generator instead.
- After regeneration, run `bash scripts/clang_format.sh`.

## CDNA4 XML Source

- The `amdgpu_isa_cdna4.xml` input should be obtained from the GPUOpen Machine-Readable ISA download page: https://gpuopen.com/download/machine-readable-isa/latest/