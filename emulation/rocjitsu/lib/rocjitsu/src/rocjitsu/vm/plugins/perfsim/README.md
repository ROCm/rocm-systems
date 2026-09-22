# Perfsim plugin

## What is it

This optional Linux plugin forwards supported gfx1250 RocJITsu execution events
to a separately supplied Perfsim backend implementing FFM observer API versions
8 through 13.

## How to install

The build below installs the RocJITsu adapter as
`librocjitsu_plugin_perfsim.so`. Its Perfsim backend is not included with
RocJITsu; obtain or build the backend's `libgpucsim_ffm_plugin.so` shared
library separately. The backend must support an observer ABI version from 8
through 13; versions outside that range are rejected during initialization. Its
Linux runtime dependencies must also be compatible with the launch environment.

From the repository root, build and install RocJITsu with the adapter enabled:

```bash
cmake -S emulation/rocjitsu -B build/rocjitsu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCJITSU_ENABLE_PERFSIM_PLUGIN=ON
cmake --build build/rocjitsu
cmake --install build/rocjitsu --prefix /absolute/path/to/install
```

## How to use it

Add the Perfsim plugin to a gfx1250 RocJITsu configuration. RocJITsu loads
`librocjitsu_plugin_perfsim.so` for the `perfsim` entry. Its `library_path`
must be the absolute path to the separately obtained Perfsim backend, not the
RocJITsu adapter:

```json
{
  "require_all_plugins": true,
  "plugins": {
    "perfsim": {
      "library_path": "/absolute/path/to/libgpucsim_ffm_plugin.so"
    }
  }
}
```

The adapter stages ordered events before replaying them to the backend.
`max_staged_bytes` defaults to `268435456` bytes (256 MiB). If the shared
budget is exceeded, the adapter rejects and purges the entire affected
dispatch. The workload can still exit zero, so check the logs for
`skipped dispatch` and verify that the expected profile was written.

For a larger trace, the budget can be raised explicitly:

```json
"perfsim": {
  "library_path": "/absolute/path/to/libgpucsim_ffm_plugin.so",
  "max_staged_bytes": 1073741824
}
```

The 1 GiB value is only an example, not a recommended default. A larger budget
trades host memory for trace coverage, and some known traces exceed even 1 GiB.

Configure Perfsim through its own environment, then launch the workload:

```bash
/absolute/path/to/install/bin/rocjitsu \
  --config /absolute/path/to/gfx1250-config.json -- ./application
```

## Backend ABI

The adapter contains a private, non-installed declaration of only the latest
FFM observer binary prefix that it consumes. The declaration is limited to the
validated little-endian Linux LP64 GCC/Clang ABI. The separately built Perfsim
library must export `ffm_observer_plugin_get_api`, negotiate an API version from
8 through 13, and provide the required lifecycle, instruction, regular-memory,
and tensor-DMA callbacks. The adapter requests versions from newest to oldest
and passes the negotiated version back through `FfmHostApi::api_version`.

The ABI additions consumed here are append-only: v8 backends read the legacy
payload prefixes, while newer backends additionally receive the v9 dispatch
name from dispatch-owned storage, the v12 host logger through the configured
RocJITsu sink, and the v13 tensor-DMA descriptor geometry. Tensor-DMA descriptor
strides are converted from element units to the byte units required by the v13
callback.

These FFM versions do not expose a table size or ABI fingerprint, and `on_init`
returns no status. Qualify the exact Perfsim build with a known dispatch and
require a nonempty report containing that dispatch.

## Real-world example: GPT-OSS kernels

Before either launch, save a complete gfx1250 RocJITsu configuration as
`/absolute/path/to/gfx1250-perfsim-config.json`. It must include the
`plugins.perfsim` block above with an absolute `library_path`; this enables the
Perfsim plugin. A C++ compiler must be available on `PATH` because Triton builds
a helper module on first use.

### Use Rocjitsu

Use a Python environment containing ROCm Torch, Triton, and NumPy to run the
GPT-OSS `_rms_norm_kernel/D1` corpus case. Run it directly with RocJITsu:

```bash
export GPT_OSS=/absolute/path/to/gpt-oss-kernel-harness
export ROCM_PATH=/absolute/path/to/rocm
export ROCJITSU_CONFIG=/absolute/path/to/gfx1250-perfsim-config.json
export PYTHONPATH="$GPT_OSS/pkg_src${PYTHONPATH:+:$PYTHONPATH}"
export GPUCSIM_TARGET=gfx1250
export GPUCSIM_WMMA_ONLY=0
export GPUCSIM_REPORT_PATH="$PWD/perfsim_report.json"

PATH="$ROCM_PATH/bin:$PATH" \
LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
HIP_PATH="$ROCM_PATH" \
/absolute/path/to/install/bin/rocjitsu \
  --config "$ROCJITSU_CONFIG" -- \
  /absolute/path/to/python "$GPT_OSS/roofline/harness/launch.py" \
  "$GPT_OSS/roofline/specs/misc.json" \
  --kernel _rms_norm_kernel --scenario D1 \
  --mode run --out "$PWD/gpt-oss-output"
```

### Use Mirage frontend

Mirage is a command-line frontend for RocJITsu and other GPU emulators. It
creates and manages an emulator session around an unmodified ROCm application.
With Mirage installed, use it as the launcher; RocJITsu remains the emulator:

```bash
ROCJITSU_LIB=/absolute/path/to/install/lib/librocjitsu.so \
PATH="$ROCM_PATH/bin:$PATH" \
LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
HIP_PATH="$ROCM_PATH" \
mirage --config "$ROCJITSU_CONFIG" -- \
  /absolute/path/to/python "$GPT_OSS/roofline/harness/launch.py" \
  "$GPT_OSS/roofline/specs/misc.json" \
  --kernel _rms_norm_kernel --scenario D1 \
  --mode run --out "$PWD/gpt-oss-output"
```

The Perfsim plugin writes the simulation results to `perfsim_report.json`.
