# Reconfigure CMake for CLR build

Deletes CMakeCache.txt and re-runs cmake configuration for the CLR/HIP build.
Use this when cache is stale or new CMake variables need to be set.

```bash
cd /c/github-emu/hipamd && rm -f CMakeCache.txt && cmake ../rocm-systems/projects/clr \
  -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON \
  -DHIP_COMMON_DIR=c:/github-emu/rocm-systems/projects/hip \
  -DHIPCC_BIN_DIR=c:\opt\rocm\bin \
  -DCMAKE_INSTALL_PREFIX=../install \
  -D__HIP_ENABLE_PCH=OFF -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=ON \
  -D__HIP_ENABLE_RTC=ON -DUSE_PROF_API=OFF -DROCR_DLL_LOAD=OFF \
  "-DAMD_COMPUTE_WIN=../../../shared/amdgpu-windows-interop/" \
  -DLIB_SRC_BUILD=ON -DROCCLR_ENABLE_GPUOPEN=ON 2>&1 | tail -20
```

Key flags:
- `-DROCCLR_ENABLE_GPUOPEN=ON` — enables the RGP/UberTrace profiler sources in `rgp/`
- `-DROCCLR_ENABLE_PAL=ON` — PAL build; DevDriver/RDF come from PAL link graph
- `-DROCCLR_ENABLE_HSA=ON` — HSA runtime integration
