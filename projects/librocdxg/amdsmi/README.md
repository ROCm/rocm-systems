# AMD System Management Interface (AMD SMI) Library — WSL

A WSL-focused build of the AMD System Management Interface library for GPU telemetry and monitoring on Windows Subsystem for Linux.
This component provides the AMD SMI C++ library, Python interface, and CLI tool (`amd-smi`) for WSL environments.

The package installs under `/opt/rocm-wsl` as `rocdxg-amd-smi-lib` to avoid conflicting with the native ROCm `amd-smi-lib` package.

> **Note:** This is a WSL adaptation of [upstream AMD SMI](https://github.com/ROCm/amdsmi). Some features (e.g. ESMI CPU monitoring, Go API, Rust wrapper) are not available on this platform.

## Prerequisites

- Windows SDK (see [root README](../README.md#2-install-librocdxg))
- CMake >= 3.20
- GCC >= 11.4
- Python 3.6.8+ (for Python interface and CLI)

## Quickstart

### 1. Install AMD SMI

Choose **one** of the following installation methods.

#### Option A — Build from source

1. Install the Windows SDK from [Microsoft](https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/). In WSL, its headers are exposed under `/mnt/c/Program Files (x86)/Windows Kits/10/Include/<version>/`.

2. Clone the repository (if not already done) and enter the `amdsmi` directory:

   ```bash
   git clone https://github.com/ROCm/librocdxg.git
   cd librocdxg/amdsmi
   ```

3. Build and install:

   ```bash
   # Adjust the SDK version to match your installation
   export win_sdk='/mnt/c/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0'

   cmake -B build -DWIN_SDK="${win_sdk}/shared" .
   cmake --build build
   sudo cmake --install build
   ```

   > **Note:** Ensure you have permission to access the Windows SDK directory from WSL.

#### Option B — Install the pre-built deb package

Download the `rocdxg-amd-smi-lib` package from [GitHub Releases](https://github.com/ROCm/librocdxg/releases) and install it:

```bash
sudo dpkg -i rocdxg-amd-smi-lib_<version>_amd64.deb
```

### 2. Set up the environment

Installation loads `/opt/rocm-wsl/.env.sh` from:

- `/etc/profile.d/rocdxg-amd-smi-lib.sh` for login shells
- `/etc/bash.bashrc` for interactive non-login bash shells (for example `docker exec -it <container> bash`)

This sets `PYTHONPATH`, `PATH`, and `LD_LIBRARY_PATH`. Apply it in **one** of these ways:

- **Current shell:** `source /etc/profile.d/rocdxg-amd-smi-lib.sh`
- **New shell:** open another login or interactive shell; the profile script loads automatically.

> **Note:** Installation also registers a `.pth` file so Python can locate the WSL AMD SMI module and CLI. If ROCm packages are installed after this package, or a new pip/venv environment is created later, the generated `zz-wsl-amdsmi.pth` will not apply to that environment. In that case, activate the target environment and use the commented block in `/opt/rocm-wsl/.env.sh` to regenerate `zz-wsl-amdsmi.pth`.

### 3. Verify the installation

Check the CLI version:

```bash
amd-smi version
```

Check the Python interface:

```bash
python3 -c "import amdsmi; amdsmi.amdsmi_init(); print(amdsmi.amdsmi_get_processor_handles()); amdsmi.amdsmi_shut_down()"
```

## Usage

Refer to the upstream AMD SMI documentation for CLI, Python, and C++ usage:

- [AMD SMI documentation portal](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [AMD SMI CLI reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html)
- [AMD SMI Python library usage](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-py-lib.html)
- [AMD SMI Python API reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html)

Use those pages as the baseline reference for commands and API usage. For the C++ library, refer to `include/amd_smi/amdsmi.h` and the upstream documentation.

Some upstream capabilities are not available in this WSL build; see [Known Limitations on WSL](#known-limitations-on-wsl) below.

## Known Limitations on WSL

- **CLI subcommands:** `event`, `set`, `reset`, and `ras` are not supported in this WSL build.
- **ESMI (CPU monitoring):** ESMI symbols are linked in for Python compatibility, but ESMI-based CPU monitoring does not work on WSL.
- **Go and Rust bindings:** Not included in this build.
- **GPU telemetry:** Metrics such as temperature, clocks, and power may be less complete than on bare-metal Linux. For broader GPU visibility, use Windows tools (Task Manager, AMD Software: Adrenalin Edition).
- **Process and engine accounting:** `amd-smi` does not expose per-process GPU memory usage or compute unit (CU) and SDMA utilization on WSL.
- **Profiling and debugging:** The ROCm profiler and debugger are not supported on WSL.

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
