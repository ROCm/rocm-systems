---
myst:
  html_meta:
    "description lang=en": "How to build and run AMD SMI under Windows Subsystem for Linux (WSL) using the experimental WSL/WDDM backend."
    "keywords": "amd-smi, wsl, wsl2, dxg, dxgkrnl, wddm, d3dkmt, windows subsystem for linux, backend, feature flag"
---

# Using AMD SMI under WSL (experimental)

AMD SMI targets native Linux, where it reads GPU data from the `amdgpu` kernel
driver through DRM ioctls and sysfs. Under Windows Subsystem for Linux 2 (WSL2)
those interfaces do not exist: there is no `/sys/class/drm` GPU tree and no
`/dev/kfd`. The GPU is reached instead through the Windows WDDM display driver
using the D3DKMT interface exposed by the `/dev/dxg` device node.

To support this without maintaining a separate build of the tool, AMD SMI has an
optional **WSL backend**. The public API, the CLI, and the Python bindings are
unchanged. When the backend is active, the affected queries are answered from
the WDDM path instead of DRM/sysfs; everything else behaves as before.

```{note}
The WSL backend is experimental and gated behind a build flag that is **off by
default**. A default AMD SMI build and its packages are byte-for-byte the native
tool. When enabled, the backend reads real GPU telemetry through `librocdxg`
(`rocdxg_smi_*` APIs); queries with no WDDM equivalent return
`AMDSMI_STATUS_NOT_SUPPORTED`.
```

## How it works

AMD SMI keeps a single code path per API. The WSL-versus-native decision is made
**once per process**, during `amdsmi_init()`, not scattered through every
function:

```text
amdsmi_get_gpu_* (one dispatcher, backend-agnostic)
        │
        ├─ native  → DRM ioctls / sysfs        (default)
        └─ WSL     → WSLGPUBackend → D3DKMT → /dev/dxg → dxgkrnl → Windows KMD
```

Queries that WDDM cannot serve (for example CPU/HSMP metrics, NIC, xGMI fabric,
ECC, partitions) return `AMDSMI_STATUS_NOT_SUPPORTED` on WSL. This is the same
status the native path returns for unsupported hardware, so existing error
handling continues to work.

## Prerequisites

- Windows 11 (or Windows 10 with WSL2) with a WSL-capable AMD GPU driver.
- A WSL2 Ubuntu distribution.
- The `dxgkrnl` module present in the guest (verify with
  `ls /sys/module/dxgkrnl`).
- `librocdxg.so.1` installed and resolvable via `dlopen` (the WSL backend
  loads it at `amdsmi_init()`; without it, calls return
  `AMDSMI_STATUS_DRIVER_NOT_LOADED`). It ships with the ROCm-on-WSL driver
  package, not a standard Linux ROCm devel package — verify with
  `ldconfig -p | grep librocdxg`.
- `libdxcore.so`, provided by the WSL installation
  (`/usr/lib/wsl/lib/libdxcore.so`) — a transitive dependency of
  `librocdxg`, not something you need to install directly.

## Building with the WSL backend

The backend is compiled in only when you enable the CMake option. It is off by
default:

```bash
cmake -S . -B build -DENABLE_WSL_BACKEND=ON
cmake --build build --target amd_smi -j"$(nproc)"
```

A build without `-DENABLE_WSL_BACKEND=ON` produces the standard native library;
the WSL source is not compiled and the intercept hooks expand to nothing.

Building with the flag on also requires `hsakmt/rocdxg_smi.h` at compile time,
from a `rocr-runtime/libhsakmt` tree built with its `WIN_SDK`/dxg path —
distinct from the runtime `librocdxg.so.1` dependency above. CMake fails with
a `FATAL_ERROR` if that header is not found alongside the checkout.

## Impact on existing scripts

The WSL backend is designed to be a drop-in. For users with existing automation:

- **No API, CLI, or output-schema changes.** Function signatures, CLI
  subcommands, flags, and JSON field names are identical. Scripts that parse
  `amd-smi --json` keep working.
- **Some fields report `N/A` / `NOT_SUPPORTED` under WSL.** Where WDDM has no
  equivalent (CPU, NIC, xGMI, ECC, partitions), the field is unavailable rather
  than wrong. Scripts should already tolerate `N/A` for unsupported hardware;
  the same handling covers WSL.
- **Native installs are unaffected.** If you never build with
  `-DENABLE_WSL_BACKEND=ON`, nothing changes.

## Verifying

```bash
# Confirm you are on WSL
ls /dev/dxg && echo "wsl present"

ls /sys/module/dxgkrnl && echo "dxgkrnl present"

# Confirm amd-smi is answering through the WSL backend
amd-smi static --asic
```

If the backend is active you will see WDDM-sourced values for the supported
fields and `N/A` for the rest.

## Limitations

- Experimental: the supported-query set is a subset of the native API and will
  grow as `librocdxg` exposes more telemetry.
- CPU/ESMI, NIC, switch, fabric, ECC, and partition features are not available
  under WSL; those queries return `AMDSMI_STATUS_NOT_SUPPORTED`.
- PCIe info, VRAM type, subvendor/subsystem IDs, and full VBIOS fields depend on
  the installed `librocdxg` version and may show `N/A` on older drivers.
- **Per-process GPU usage (`amd-smi process`) is not available under WSL** and
  returns `AMDSMI_STATUS_NOT_SUPPORTED`. The WSL process-enumeration ioctl
  reports WSL-guest (Linux) PIDs, but the per-process VRAM query expects a
  real Windows process handle, not a bare PID — this mismatch is unverified
  against hardware, so the query is disabled rather than risk reporting
  incorrect per-process memory usage.
