# Changelog for AMD Debugger API (ROCdbgapi)

Full documentation for AMD Debugger API is available at
[rocm.docs.amd.com/rocdbgapi](https://rocm.docs.amd.com/projects/ROCdbgapi/en/latest/index.html).

## rocm-dbgapi-0.78 for ROCm-X

### Added
- Initial support to debug HIP applications on Windows OS.  Windows is
  supported only on the gfx120x architectures (gfx1200 and gfx1201) with no
  support for multi-GPU configurations.

### Changed
- The `amd_dbgapi_read_memory` and `amd_dbgapi_write_memory` methods can return
  the new `AMD_DBGAPI_STATUS_ERROR_MEMORY_UNAVAILABLE` error code.

## rocm-dbgapi-0.77.4 for ROCm-7.0
### Added
- Support for the following architectures:
 - `gfx950`
 - `gfx1150`
 - `gfx1151`

### Removed
- Support for the following architectures:
 - `gfx940`
 - `gfx941`

## rocm-dbgapi-0.77.2 for ROCm-6.4
### Added
- Support for generic code object targets:
 - `gfx9-generic`
 - `gfx9-4-generic`
 - `gfx10-1-generic`
 - `gfx10-3-generic`
 - `gfx11-generic`
 - `gfx12-generic`

## Changed
- The name reported for detected agents is now based on the `amdgpu.ids`
  database provided by `libdrm`.

## rocm-dbgapi-0.77.0 for ROCm-6.3

### Added
- Support for gfx1200 and gfx1201 architectures.
- Support for setting precise ALU exception reporting.

## rocm-dbgapi-0.76.0
### Resolved issues
- Before ROCr ABI version 10, there where configurations where a debugger
  would generate unreliable core dumps.  If such configuration is detected
  by rocm-dbgapi, `amd_dbgapi_process_get_info` now returns
  `AMD_DBGAPI_STATUS_ERROR_RESTRICTION` for the
  `AMDGPU_DBGAPI_PROCESS_INFO_CORE_STATE` request.

## rocm-dbgapi-0.75.0 for ROCm 6.2
### Changed
- Rename (AMD_DBGAPI_EXCEPTION_WAVE,AMD_DBGAPI_WAVE_STOP_REASON)_APERTURE_VIOLATION
  -> (AMD_DBGAPI_EXCEPTION_WAVE,AMD_DBGAPI_WAVE_STOP_REASON)_ADDRESS_ERROR.
  Old names are still available but deprecated.

## rocm-dbgapi-0.74.0
### Added
- Added support to create and open core dumps.

## rocm-dbgapi-0.71.0 for ROCm 6.0
### Added
- Add support for gfx940, gfx941 and gfx942 architectures.

## rocm-dbgapi-0.70.0 for ROCm 5.6
### Changed
- The name reported for each agent is now based on the information stored
  in the [`pci.ids`](https://pci-ids.ucw.cz/) database.
- Return `AMD_DBGAPI_STATUS_ERROR_NOT_AVAILABLE` when querying
  `AMD_DBGAPI_REGISTER_INFO_DWARF` for a valid register which does not have
  an associated DWARF register number.

### Known Issues
- Does not support debugging programs that use cooperative groups or CU masking
  for gfx1100, gfx1101, and gfx11102.  A restriction will be reported when
  attaching to a process that has already created cooperative group queues or
  CU masked queues.  Any attempt by the process to create a cooperative queue
  or CU masked queue when attached will fail.

- On gfx1100, gfx1101 and gfx1102, the library cannot debug a program past a
  `s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)` instruction.  If an exception is
  delivered to a wave in an attached process after the wave has executed this
  instruction, the wave is killed and the exception is not reported in the
  debugger.

## rocm-dbgapi-0.68.0
### Added
- Expose SGPRs mapped under `flat_scratch`/`xnack_mask`/`vcc`.
