# pFFM adapter

## What is it

This optional Linux plugin forwards supported gfx1250 RocJITsu execution events
to a pFFM FFM-v8 backend.

## How to install

pFFM is not included with RocJITsu. Obtain or build its
`libgpucsim_ffm_plugin.so` shared library separately.

From the repository root, build and install RocJITsu with the adapter enabled:

```bash
cmake -S emulation/rocjitsu -B build/rocjitsu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DROCJITSU_ENABLE_PFFM_PLUGIN=ON
cmake --build build/rocjitsu
cmake --install build/rocjitsu --prefix /absolute/path/to/install
```

## How to use it

Add the plugin to a gfx1250 RocJITsu configuration. `library_path` must be the
absolute path to the separately obtained pFFM shared library:

```json
{
  "require_all_plugins": true,
  "plugins": {
    "pffm": {
      "library_path": "/absolute/path/to/libgpucsim_ffm_plugin.so"
    }
  }
}
```

Configure pFFM through its own environment, then launch the workload:

```bash
/absolute/path/to/install/bin/rocjitsu \
  --config /absolute/path/to/gfx1250-config.json -- ./application
```

## Real-world example: GPT-OSS kernels

Before either launch, save a complete gfx1250 RocJITsu configuration as
`/absolute/path/to/gfx1250-pffm-config.json`. It must include the pFFM
`plugins` block above with an absolute `library_path`; this is what enables pFFM.

Use a Python environment containing ROCm Torch, Triton, and NumPy to run the
GPT-OSS `_rms_norm_kernel/D1` corpus case. Run it directly with RocJITsu:

```bash
export GPT_OSS=/absolute/path/to/gpt-oss-kernel-harness
export ROCM_PATH=/absolute/path/to/rocm
export ROCJITSU_CONFIG=/absolute/path/to/gfx1250-pffm-config.json
export PYTHONPATH="$GPT_OSS/pkg_src${PYTHONPATH:+:$PYTHONPATH}"
export GPUCSIM_TARGET=gfx1250
export GPUCSIM_WMMA_ONLY=0
export GPUCSIM_REPORT_PATH="$PWD/pffm_report.json"

PATH="$ROCM_PATH/bin:$PATH" \
LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
HIP_PATH="$ROCM_PATH" \
/absolute/path/to/install/bin/rocjitsu \
  --config "$ROCJITSU_CONFIG" -- \
  /absolute/path/to/python "$GPT_OSS/harness/launch.py" \
  "$GPT_OSS/specs/misc.json" \
  --kernel _rms_norm_kernel --scenario D1 \
  --mode run --out "$PWD/gpt-oss-output"
```

Mirage is a command-line frontend for RocJITsu and other GPU emulators. It
creates and manages an emulator session around an unmodified ROCm application.
With Mirage installed, use it as the launcher; RocJITsu remains the emulator:

```bash
ROCJITSU_LIB=/absolute/path/to/install/lib/librocjitsu.so \
PATH="$ROCM_PATH/bin:$PATH" \
LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/lib/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
HIP_PATH="$ROCM_PATH" \
mirage --config "$ROCJITSU_CONFIG" -- \
  /absolute/path/to/python "$GPT_OSS/harness/launch.py" \
  "$GPT_OSS/specs/misc.json" \
  --kernel _rms_norm_kernel --scenario D1 \
  --mode run --out "$PWD/gpt-oss-output"
```

pFFM writes the simulation results to `pffm_report.json`.
