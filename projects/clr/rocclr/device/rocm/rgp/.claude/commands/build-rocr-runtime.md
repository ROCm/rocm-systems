# Build HSA Runtime (rocr-runtime) standalone

Builds hsa-runtime64 standalone (outside of the hipamd/CLR build tree).
The hipamd build already compiles hsa-runtime64 as a dependency of rocclr —
use this only when you need to iterate on the runtime independently.

## Existing build tree
The rocr-runtime has a pre-configured build at:
  `projects/rocr-runtime/build/`

```bash
cd /c/github-emu/rocm-systems/projects/rocr-runtime/build
cmake --build . --config Release -j 6 2>&1 | tail -30
```

## Configure from scratch (Windows, with WKMI + trap handler support)
```bash
mkdir -p /c/github-emu/rocm-systems/projects/rocr-runtime/build
cd /c/github-emu/rocm-systems/projects/rocr-runtime/build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/c/github-emu/install \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_C_COMPILER="C:/opt/rocm/lib/llvm/bin/clang-cl.exe" \
  -DCMAKE_CXX_COMPILER="C:/opt/rocm/lib/llvm/bin/clang-cl.exe" \
  -DHSA_USE_RUNTIME_AVX512=OFF 2>&1 | tail -20
cmake --build . --config Release -j 6 2>&1 | tail -30
```

## Key files for RGP/trap handler work
- `runtime/hsa-runtime/core/runtime/trap_handler/trap_handler_gfx12.s` — gfx12/gfx13 trap handler (active dev)
- `runtime/hsa-runtime/core/runtime/trap_handler/trap_handler.s` — pre-gfx12 trap handler
- Assembled into `kCodeTrapHandlerV2_12` / `kCodeTrapHandlerV2_1250` etc. at cmake configure time

## Note
The hipamd CLR build at `C:/github-emu/hipamd` builds hsa-runtime64 as a subproject —
modifying trap handler sources there rebuilds automatically when you run `/build-rocclr`.
