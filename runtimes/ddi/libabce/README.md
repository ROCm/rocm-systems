<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# Rust ABCE provider

`libabce` provides the existing ABCE host C API through the shared rocddi
runtime image. Its entry points call Rust planning, packet encoding and ring
publication in `rocddi::device::gpu::abce`. They do not initialize or call HSA,
compile the C ABCE implementation, or open a GPU implicitly.

## C and C++ consumers

The headers retain their default implementation. A CMake consumer selects the
Rust implementation using `ABCE_HEADER_ONLY=OFF` and the ABCE interface target:

```cmake
find_package(ROCmRuntimeABCE 1 REQUIRED CONFIG)
target_link_libraries(my_copy_library PRIVATE rocm_runtime::abce)
```

```sh
cmake -S . -B build -G Ninja -DABCE_HEADER_ONLY=OFF -DCMAKE_PREFIX_PATH=/path/to/runtime/install
```

The target supplies `ABCE_ENABLE_IMPLEMENTATION=0` and the library dependency.
Source keeps the same `#include <abce/abce_host.h>` and `abce_*` calls. Use
`PUBLIC` linkage when your own public headers expose ABCE. Omitting the option
selects the header implementation. Projects using `add_subdirectory` can use
the same targets from `runtimes/api-headers`; external mode requires the full
runtime or a parent-provided `rocm_runtime::abce_shared` target.

Installed packages select the facade using the **consumer's** option. The
concrete `rocm_runtime::abce_headers` and `rocm_runtime::abce_shared` targets
are also available. A headers-only installation rejects external mode with a
package diagnostic. Merely adding an include directory does not apply CMake
usage requirements. Non-CMake host builds can define
`ABCE_ENABLE_IMPLEMENTATION=0` and link `-labce` explicitly.

On Linux, `libabce.so.1` is an alias for the same image as `libhsa-runtime64.so.1`.
It adds the independent `ABCE_1` symbol version and `abce_host_abi_version()`;
there is no second runtime state. Existing AMDF static linkage still works.
HIP kernel translation units must retain the header implementation. The
external host API includes device-descriptor initialization, while device
submission remains inline. The optional HSA adapter remains a separate inline
adapter over the chosen C provider.

## Ownership and native GPU use

C callers still own rings, control callbacks, signals and operand memory.
Mapped plans borrow the provider's per-thread packet scratch until another map
on that thread or thread exit, and may be submitted once. Keep the map/submit
pair on one thread. Callbacks must not throw/unwind or recursively map while
those bytes are borrowed. Legacy fixed-layout builders require zeroed output;
see the public header's exact buffer and alignment requirements. Explicit
`ABCE_HAS_KFD_TOPOLOGY` overrides are limited to header mode; the external
provider decides its own platform support.

Rust callers obtain the service from an activated GPU:

```text
Session::activate -> Device::gpu -> GpuDevice::copy_engine
    -> engines -> unsafe prepare_raw -> PreparedCopy::submit
    -> CopySubmission::poll / wait / is_retired
```

SDMA engines belong to that GPU's VM and resource pool. They are neither new
endpoints nor separately activated devices. Engine views preserve their owner
and distinguish direct hardware indices from kernel logical indices. Discovery
creates no queues; preparation leases queues lazily. GPU memory residence in an
ABCE operand is independent of the GPU selected to execute the copy.

`prepare_raw` copies packet bytes into owned storage. Its unsafe contract
requires callers to retain operand, dependency, timestamp and indirect-pointer
backings/mappings until checked retirement. `poll` reports execution failure
separately from `is_retired`. Dropping pending work or observing a timeout does
not revoke GPU access; native queue, command and synchronization storage is
retained conservatively. Inherited services reject use after fork before
accessing mappings or pool locks.

Native recipes currently cover x86-64 GFX9 direct queues and GFX12.0/GFX12.5
kernel queues already supported by rocddi. Operation support is restricted to
linear/fill, plus GFX12.5 swap/indirect. Native multi-engine publication requires
an explicit backend qualification capability; KFD currently leaves it disabled.
The raw C API retains ABCE's broader packet surface and caller qualification
requirements. Host tests establish packet bytes and lifetime logic, not real
GPU execution, multi-engine independence, or performance.

## Maintaining the ABI

Ordinary builds use checked-in Rust bindings. Regeneration requires Clang and
the workspace Rust formatter; it generates value records, C thunks and ELF
aliases, never implementation algorithms:

```sh
python3 ddi/libabce/tools/generate_abi.py --clang /path/to/clang
python3 ddi/libabce/tools/generate_abi.py --clang /path/to/clang --check
python3 ddi/libabce/tests/abi/check_layout.py
python3 ddi/libabce/tests/abi/check_exports.py target/release/librocddi_runtime.so
```

`abi/exports.txt` owns the host symbol surface. The two `core-function` entries
are exported directly to preserve their addresses in the public policy tables.
The independent C layout probe checks sizes, alignments, field offsets and enum
values. Configure `ABCE_BUILD_TESTS=ON` in a full CMake build to run the same
GPU-free host suites with both providers, plus external ABI regressions.
