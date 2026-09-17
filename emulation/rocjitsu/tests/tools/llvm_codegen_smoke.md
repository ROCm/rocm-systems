# LLVM compiler compatibility smoke tests

These optional tests compile fresh kernels with an external LLVM build, check
the ELF feature flags, and execute 128 checked results through the real
runtime on a simulated GPU. Separate HIP and direct HSA tests cover `gfx90a`,
`gfx950`, `gfx1250`, `gfx1250-strict`, and `gfx12-5-generic`. The strict and
generic objects execute on the ordinary gfx1250 simulator configuration;
target compatibility is handled by the runtime.

Configure an existing RocJITsu build with the LLVM tools to test:

```sh
cmake -S rocm-systems/emulation/rocjitsu -B build \
  -DRJ_LLVM_SMOKE_TOOLS_DIR="$HOME/llvm/main/build/bin"
ninja -C build llvm_codegen_smoke llvm_codegen_hsa_smoke
ctest --test-dir build -R '^LlvmCodegen(Hsa)?Smoke\.' -j4 --output-on-failure
```

Build LLVM completely before running the tests. A version string alone does
not establish that every tool was rebuilt from the current checkout.
The HIP/HSA headers and runtimes come from the build's existing `ROCM_PATH`; LLVM's
`llc`, `llvm-mc`, `ld.lld`, `llvm-readobj`, and `llvm-objdump` come exclusively
from `RJ_LLVM_SMOKE_TOOLS_DIR`. No LLVM development libraries are linked into
RocJITsu. Clear the CMake setting to disable these external-toolchain tests.

The driver compares direct object emission with the assembly/link path and
exercises module flags and assembly target directives, including
default, enabled and disabled XNACK/SRAM-ECC settings. On `gfx90a` and `gfx950`,
explicit modes incompatible with the simulated agent must be rejected; the
other cases must run and produce the expected output. The three gfx1250-family
targets have permanently enabled XNACK, so an explicit OFF module flag must
not turn it off. Their default SRAM-ECC mode must execute. Explicit SRAM-ECC
ON and OFF currently remain unsupported by the HIP/ROCr ISA registries, so
these tests check their rejection; they do not establish support for those
modes. Separate metadata controls check legacy XNACK UNSUPPORTED and ANY
compatibility and explicit XNACK OFF rejection. The compiler-produced objects
are never modified.

To compare source-built runtimes without replacing the SDK libraries, set
`RJ_LLVM_SMOKE_PRELOAD` to their absolute paths, separated by colons:

```sh
RJ_LLVM_SMOKE_PRELOAD="/path/to/libamdhip64.so.7:/path/to/libhsa-runtime64.so.1" \
  ctest --test-dir build -R '^LlvmCodegen(Hsa)?Smoke\.' -j4 --output-on-failure
```

The driver injects these libraries only into the simulator process. Do not
preload HIP globally into the compiler tools: the SDK's COMGR can introduce
LLVM libraries incompatible with the compiler under test. For ROCr-only
experiments, select `^LlvmCodegenHsaSmoke\.` and preload only ROCr.

Each target retains generated IR, assembly, linked code objects, tool output,
and a `results.json` under `build/tests/llvm_codegen_smoke_objects/` or
`build/tests/llvm_codegen_hsa_smoke_objects/`. Failures are reported as
failures, including missing compiler support or runtime rejection of a
supported case; these tests do not silently skip unsupported toolchains.
