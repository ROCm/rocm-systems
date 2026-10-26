# Changelog for ROCR Debug Agent library (ROCdebug-agent)

Full documentation for the ROCR Debug Agent library is available at
[rocm.docs.amd.com/rocr_debug_agent](https://rocm.docs.amd.com/projects/rocr_debug_agent/en/latest/).

## ROCR Debug Agent 2.2.1 for ROCm 10.3

- After printing the waves on `SIGQUIT`, the agent resumes them and the
  program continues.  Previously the waves stayed stopped and the program
  hung after the dump.
- The agent can be loaded with `LD_PRELOAD` as well as `HSA_TOOLS_LIB`, and no
  longer links `libhsa-runtime64`.  Without `HSA_TOOLS_LIB`, the code object
  list is read when waves are printed, as with `--lazy`.

## ROCR Debug Agent 2.2.0  for ROCm 10.1

- The `--output` and `--save-code-objects` options now support '%' format
  tokens to produce the output file names.  The `%p`, `%h`, `%t`, `%e`, `%u`,
  `%g` and `%%` tokens are supported.

## ROCR Debug Agent 2.1.0 for ROCm 7.0

### Added
- Added the `-e` and `--precise-alu-exceptions` flags to enable precise
  ALU exceptions reporting on supported configurations.

## ROCR Debug Agent 2.0.4 for ROCm 6.4

### Added
- The associated kernel name is printed for each wave.
