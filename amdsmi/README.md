# AMD System Management Interface (AMD SMI) Library — WSL

A WSL-focused build of the AMD System Management Interface library for GPU telemetry and monitoring on Windows Subsystem for Linux.

This component provides the AMD SMI C++ library, Python interface, and CLI tool (`amd-smi`) for WSL environments.

> **Note:** This is a WSL adaptation of [upstream AMD SMI](https://github.com/ROCm/amdsmi). Some features (e.g. ESMI CPU monitoring, Go API, Rust wrapper) are not available on this platform.

## Prerequisites

- AMD ROCm installed in WSL
- Windows SDK (see [root README](../README.md#1-install-windows-sdk))
- CMake >= 3.20
- GCC >= 11.4
- Python 3.6.8+ (for Python interface and CLI)

## Building

Build AMD SMI directly from this directory:

```bash
cmake -B build -DWIN_SDK=<path-to-windows-sdk> <path-to-amdsmi-source>
cmake --build build
sudo cmake --install build
```

For example, from the amdsmi directory:
```bash
cmake -B build -DWIN_SDK=/path/to/windows/sdk .
cmake --build build
sudo cmake --install build
```

### Post-install: register the Python module

When installing via `make install` (instead of a `.deb` package), the Python module is copied to `/opt/rocm/share/amd_smi/` but **not** automatically registered with Python. Run one of the following to enable `import amdsmi`:

```bash
# Option 1: pip install (recommended)
sudo python3 -m pip install /opt/rocm/share/amd_smi

# Option 2: set PYTHONPATH (add to ~/.bashrc for persistence)
export PYTHONPATH=/opt/rocm/share/amd_smi:$PYTHONPATH
```

> **Note:** `.deb` package installations handle this automatically via the postinst script.

## Usage

### CLI tool

The `amd-smi` command-line tool provides GPU monitoring:

```bash
amd-smi
amd-smi version
amd-smi static
```

**Note:** The following subcommands are not supported in this WSL build:
- `event` — Event monitoring not available
- `set` — Setting GPU parameters not supported
- `reset` — GPU reset operations not supported
- `ras` — RAS features not available

### Python interface

```python
import amdsmi
amdsmi.amdsmi_init()
devices = amdsmi.amdsmi_get_processor_handles()
for dev in devices:
    print(amdsmi.amdsmi_get_gpu_asic_info(dev))
amdsmi.amdsmi_shut_down()
```

### C++ library

```cpp
#include <amd_smi/amdsmi.h>

amdsmi_init(AMDSMI_INIT_AMD_GPUS);
// ... use AMD SMI APIs ...
amdsmi_shut_down();
```

Refer to `include/amd_smi/amdsmi.h` for the full C++ API.

## Known Limitations on WSL

- **ESMI (CPU monitoring):** ESMI symbols are linked in for Python compatibility, but ESMI-based CPU monitoring does not work on WSL.
- **Go and Rust bindings:** Not included in this build.
- **GPU telemetry:** Metrics such as temperature, clocks, and power may be less complete than on bare-metal Linux. For broader GPU visibility, use Windows tools (Task Manager, AMD Software: Adrenalin Edition).
- **Process and engine accounting:** `amd-smi` does not expose per-process GPU memory usage or compute unit (CU) and SDMA utilization on WSL.
- **Profiling and debugging:** The ROCm profiler and debugger are not supported on WSL.

## Documentation

For upstream AMD SMI documentation, see [rocm.docs.amd.com/projects/amdsmi](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.
