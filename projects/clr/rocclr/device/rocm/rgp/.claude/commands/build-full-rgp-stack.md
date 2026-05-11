# Build full RGP stack

Builds the complete chain needed for end-to-end RGP capture:
  rocr-runtime → CLR (amdhip64 + GPUOPEN) → rocprofiler-sdk

Run steps in order. Each step's output feeds the next.

## Step 1 — Build CLR with RGP support (amdhip64 + rocclr)
```bash
cd /c/github-emu/hipamd
cmake --build . --config Release -j 6 --target install 2>&1 | tail -30
```
Installs to `C:/github-emu/install/`:
- `bin/amdhip64.dll` — HIP runtime with UberTrace/RGP capture
- `lib/hsa-runtime64.lib` — HSA runtime (includes AQLprofile, trap handler)
- RGP source files under `device/rocm/rgp/` compiled in with `ROC_GPUOPEN_OCL`

## Step 2 — Build rocprofiler-sdk against installed hsa-runtime64
```bash
mkdir -p /c/github-emu/rocm-systems/projects/rocprofiler-sdk/build
cd /c/github-emu/rocm-systems/projects/rocprofiler-sdk/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/c/github-emu/install \
  -DROCM_PATH=/c/opt/rocm \
  -Dhsa-runtime64_DIR=/c/github-emu/install/lib/cmake/hsa-runtime64 \
  -DROCPROFILER_BUILD_TESTS=OFF \
  -DROCPROFILER_BUILD_SAMPLES=OFF 2>&1 | tail -20
cmake --build . --config Release -j 6 --target install 2>&1 | tail -30
```

## Step 3 — Verify RGP capture path
Confirm `amdhip64.dll` has RGP support compiled in:
```bash
dumpbin /exports /c/github-emu/install/bin/amdhip64.dll | grep -i "capture\|trace\|rgp" || \
  strings /c/github-emu/install/bin/amdhip64.dll | grep -i "UberTrace\|RocUberTrace\|COLoadEvent" | head -10
```

## Key cmake flags that enable RGP in CLR
- `-DROCCLR_ENABLE_GPUOPEN=ON` — compiles `rgp/rocgpuopen.cpp`, `rgp/roctracesession.cpp`, etc.
- `-DROCCLR_ENABLE_PAL=ON` — PAL provides DevDriver + RDF transitively (required for UberTrace)
- `-DROCCLR_ENABLE_HSA=ON` — HSA runtime integration (AQLprofile, dispatch packets)

## What "full RGP support" means in this build
- UberTrace service (`RocUberTraceService`) registered with DevDriver over named pipe
- RGP tool connects → triggers `OnTraceRequested` → SQTT buffers allocated via AQLprofile
- Per-dispatch `PreDispatch`/`PostDispatch` hooks in `rocvirtual.cpp` drive state machine
- On stop: `roctracesession.cpp` writes RDF chunks (AsicInfo, SQTT, COLoadEvent) → `.rgp` file
- `loader_event` populated from `AddElfBinary` + `AddKernelLoadEvent` in `rocprogram.cpp`
