# Changelog for rocdxg-amd-smi-lib (WSL)

WSL adaptation of AMD SMI in [librocdxg](https://github.com/ROCm/librocdxg).

For upstream AMD SMI release history, see the [native CHANGELOG](https://github.com/ROCm/amdsmi/blob/develop/CHANGELOG.md) and [AMD SMI documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).

***All information listed below is for the WSL build and subject to change.***

## rocdxg-amd-smi-lib 1.3.0

### Added

- **Separate WSL packaging as `rocdxg-amd-smi-lib` under `/opt/rocm-wsl`**.  
  Installs side-by-side with the native ROCm `amd-smi-lib` package without overwriting it.

- **Automatic Python module registration via `zz-wsl-amdsmi.pth`**.  
  Ensures WSL AMD SMI is discovered before `_rocm_sdk_core` in vLLM and similar Python environments.

- **Environment setup scripts**.  
  - `/opt/rocm-wsl/.env.sh` exports `PYTHONPATH`, `PATH`, and `LD_LIBRARY_PATH`.
  - `/etc/profile.d/rocdxg-amd-smi-lib.sh` loads the environment for new shells.

### Changed

- **Python wrapper loads `libamd_smi.so` from `/opt/rocm-wsl`**.  
  Uses a fixed install-prefix path with versioned `.so` naming instead of searching vendored ROCm SDK libraries.

- **Library version macros moved to CMake compile definitions**.  
  Stops rewriting `include/amd_smi/amdsmi.h` during the build.

- **README restructured** with numbered install steps, deb package option, and links to upstream usage documentation.

### Fixed

- **`amdsmi_get_gpu_fan_speed()`, `amdsmi_get_gpu_fan_rpms()`, and `amdsmi_get_gpu_volt_metric()`**.  
  Check metric support before validating output pointers so unsupported queries return `AMDSMI_STATUS_NOT_SUPPORTED` instead of `AMDSMI_STATUS_INVAL`.

## rocdxg-amd-smi-lib 1.2.0

Initial WSL release shipped with librocdxg 1.2.0.

### Added

- **WSL port of AMD SMI** using the librocdxg shared `DeviceContext` and Windows DDK/DXCore backend.
- **GPU monitoring APIs**, including processor enumeration, ASIC/VRAM/power/temperature/clock, and related telemetry.
- **`amd-smi` CLI and Python interface** for supported WSL GPU setups.
- **Debian packaging** as `amd-smi-lib` installed under `/opt/rocm`.

### Changed

- **Removed upstream-only components** not applicable to WSL: ESMI CPU monitoring (stubbed), NIC subsystem, Go/Rust bindings, and legacy `rocm_smi` code.
- **CLI hardened for WSL**: unsupported subcommands (`event`, `set`, `reset`, `ras`) are skipped instead of crashing; KFD-unavailable paths return `AMDSMI_STATUS_NOT_SUPPORTED`.
- **Build integrated with librocdxg**: shared `VERSION` file, shared headers under `shared/include/`, and component deb targets.

### Known WSL limitations

- ESMI-based CPU monitoring is not available.
- Several upstream CLI subcommands and APIs that depend on KFD sysfs or bare-metal driver features are not supported.
- GPU telemetry may be less complete than on native Linux. See [README.md](README.md) for details.
