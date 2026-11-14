# NCCL device IR bindings

`bindings/ir` provides a higher-level IR ABI for language bindings on top of
NCCL device APIs. Its C-linkage functions are compiled into LLVM bitcode or
NVIDIA LTO-IR, letting GPU compilers and DSLs link NCCL communication operations
into their kernels.

[nccl4py's CuTeDSL API](../nccl4py/nccl/core/device/cute/) uses the LLVM bitcode
library, `libnccl_device.bc`, to provide NCCL device operations in CuTeDSL kernels.

## Build

Requires a CUDA Toolkit with NVCC. LLVM bitcode additionally requires Clang
with NVPTX support, `opt`, `llvm-dis`, and `llvm-as`.

From the NCCL repository root, build both formats (adjust CUDA path and GPU
architecture as needed):

```sh
make -j ir.build CUDA_HOME=/usr/local/cuda IR_ARCH=sm_90
```

This also builds native NCCL. Use `ir.llvm_ir` or `ir.ltoir` instead of
`ir.build` to select one format.

Outputs under `BUILDDIR` (`build/` by default):

- `include/nccl_device_ir.h`: device API declarations.
- `lib/libnccl_device.bc`: LLVM bitcode library.
- `lib/libnccl_device.ltoir`: NVIDIA LTO-IR library.

See [nccl_device_ir.h](nccl_device_ir.h) for the available APIs.

## RCCL

RCCL carries the files above from NCCL but does not build them. `src/nccl_device_ir.cu`
stops with `#error` unless it is compiled by Clang CUDA or NVCC, and it uses PTX inline
assembly (`activemask.b32`), so RCCL builds no `ncclIr*` functions and does not install
`nccl_device_ir.h`.

RCCL builds its own bitcode library from `nccl_device_wrapper__impl.h` with HIP Clang. From
the RCCL root:

```sh
cmake -B build -DNCCL_EMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=gfx942 .
```

`EMIT_LLVM_IR` is still accepted for `NCCL_EMIT_LLVM_IR`, and `IR_ARCH` for
`BITCODE_LIB_ARCH` (default `gfx950`). The standalone `Makefile` in this directory takes
the same variables and reuses the hipify-staged headers of an existing RCCL build.

Outputs under the build directory:

- `include/nccl_device_wrapper.h`: C declarations of the exported functions.
- `lib/librccl_device.bc`: LLVM bitcode library.

`librccl_device.bc` exports only the functions declared in `nccl_device_wrapper.h`
(`RCCL_DEVICE_BC_PUBLIC_API` in `CMakeLists.txt` and the `Makefile`), not the `ncclIr*`
interface. Code written against `nccl_device_ir.h`, including nccl4py's CuTeDSL API, does
not link against it.
