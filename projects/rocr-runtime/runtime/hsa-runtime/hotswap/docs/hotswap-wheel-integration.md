# HotSwap Wheel Integration Plan

## Goal

Build ROCm + PyTorch wheels for gfx942 (MI300X) that include HotSwap
cross-gen support, enabling seamless gfx950→gfx942 kernel retargeting
at load time with zero runtime overhead.

## Architecture

```
TheRock build (cmake)
  ├── compiler/amd-llvm          ← provides LLVM MC libs for hotswap
  ├── core/ROCR-Runtime          ← hotswap rewrite engine + retarget
  │     └── -DROCR_ENABLE_HOTSWAP=ON
  ├── core/hip-clr               ← fat binary + bare ELF cross-gen intercept
  └── packaging → rocm-sdk-core wheel
                   └── includes libhsa-runtime64.so (with hotswap)
                   └── includes libamdhip64.so (with cross-gen)

external-builds/pytorch/
  └── build_prod_wheels.py
       └── installs rocm-sdk-core (with hotswap)
       └── builds torch wheel linked against hotswap-enabled ROCR/HIP
```

## Changes Required

### 1. TheRock core/CMakeLists.txt

Pass `ROCR_ENABLE_HOTSWAP=ON` to ROCR-Runtime subproject:

```cmake
# In the ROCR-Runtime CMAKE_ARGS section:
list(APPEND _ROCR_CMAKE_ARGS "-DROCR_ENABLE_HOTSWAP=ON")
```

### 2. TheRock rocm-systems submodule

Point to the `users/powderluv/rocm-hotswap` branch which has:
- `hotswap/` directory with rewrite engine, rule parser, trampolines
- Modified `loader/executable.cpp` with hotswap hook
- Modified `CMakeLists.txt` with ROCR_ENABLE_HOTSWAP option
- Modified `clr/hipamd/src/hip_fatbin.cpp` with cross-gen intercept
- Modified `hsacore.so.def` with `rocr_hotswap_retarget` export

### 3. Build commands

Full ROCm build:
```bash
cmake -B build -S . -GNinja \
  -DTHEROCK_AMDGPU_FAMILIES=gfx942 \
  -DTHEROCK_ENABLE_ALL=ON

cd build && ninja
```

PyTorch wheel build:
```bash
cd external-builds/pytorch
python pytorch_torch_repo.py checkout
python build_prod_wheels.py build \
  --install-rocm --index-url <local-or-nightly> \
  --output-dir $HOME/tmp/pyout
```

### 4. Usage

Install the wheels:
```bash
pip install rocm torch --index-url <wheel-dir>
```

Run with cross-gen:
```bash
# Enable cross-gen for gfx950→gfx942
export HSA_HOTSWAP_RULES=/dev/null  # empty rules, just enable the engine
export HSA_HOTSWAP_ISA_OVERRIDE=gfx942

# gfx950 kernels automatically retargeted at load time
python my_model.py
```

## Testing

1. Build ROCm SDK on mi300-2 with hotswap enabled
2. Build PyTorch wheel against hotswap-enabled SDK
3. Install both in a test venv
4. Run AITER tests with cross-gen
5. Verify: 17/17 execution tests pass, 1.000x performance geomean
