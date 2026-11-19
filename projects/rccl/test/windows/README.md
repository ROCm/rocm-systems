# Native Windows development build

This experimental path builds `rccl.dll` and its import library `rccl.lib`.
Shared portability fixes use standard C++ or the existing OS abstraction;
Win32 implementations and Linux-only dependencies remain platform-specific.
The Linux build path remains selected on Linux.

Validation so far covers Windows 11 x64, ROCm 10.2.0, and Radeon 8060S
(`gfx1151`) with the restricted development kernels below. Other SDKs and GPU
families require their own validation; accepting a compiler target is not a
claim of runtime support on that hardware.

The SDK declares `hipHostMallocUncached` on Windows but documents that allocation
mode as Linux-only, so the Windows build leaves it disabled. The existing
initialization check still rejects GPUs such as `gfx950` that require it.

## Prerequisites

- Visual Studio 2022 C++ Build Tools and a complete Windows SDK, initialized
  through Developer PowerShell or an equivalent compiler environment.
- CMake 3.22 or newer and Ninja for the native Windows path.
- Python 3, Bash, Perl, and the SDK's `hipify-perl`. Tools are discovered through
  `PATH`, Git's installation, and the SDK; non-default installations can use
  explicit staging-tool overrides.
- A Windows ROCm SDK with HIP and `rocm-core` CMake packages. Set `HIP_PATH` or
  pass `ROCM_PATH`; the build derives the SDK version and probes header/API
  capabilities instead of assuming a particular SDK release.

From the repository root in Developer PowerShell:

```powershell
$env:ROCM_PATH = $env:HIP_PATH
$env:HIP_PLATFORM = 'amd'
$env:LLVM_PATH = "$env:HIP_PATH\lib\llvm"
$env:HIP_DEVICE_LIB_PATH = "$env:LLVM_PATH\amdgcn\bitcode"
$env:Path = "$env:HIP_PATH\bin;$env:LLVM_PATH\bin;$env:Path"
cmake -S projects/rccl -B build/rccl-windows -G Ninja `
  "-DROCM_PATH=$env:HIP_PATH" -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release `
  '-DONLY_FUNCS=AllReduce RING SIMPLE Sum f32|SendRecv RING SIMPLE Sum i8' `
  -DRCCL_BUILD_SMOKE_TEST=ON
cmake --build build/rccl-windows --parallel 8
```

`gfx1151` is the target used for local validation, not a build default.
Set `GPU_TARGETS` explicitly for the intended hardware; a quoted semicolon
list selects multiple compiler-supported targets. Missing targets produce
an actionable configuration error rather than silently building for the
developer's GPU. `ONLY_FUNCS` is an optional development filter; an empty
filter requests the generator's full set, which has not been validated here.

The toolchain respects explicitly selected compilers. Source staging is a
build dependency: edits, removed sources, transform changes, and relevant tool
changes are incorporated by `cmake --build`, without a manual HIPification
step. Windows uses the ordinary HIP device-link flow. Unsupported Linux-only
build options must not be assumed to work in this Windows path.

Use `RCCL_PERL`, `RCCL_BASH`, and `RCCL_HIPIFY_PERL` CMake cache paths (or
environment variables with those names) when tool discovery needs an override.
The CPU-only staging regression suite is also part of the Linux host guards:
`python cmake/scripts/test_stage_windows.py -v` from `projects/rccl`.

## Smoke test

```powershell
ctest --test-dir build/rccl-windows -R '^rccl_windows_smoke$' --output-on-failure
```

The test is built as ordinary host C++, without hardcoding a GPU architecture
in its compile command. CMake stages matching runtime DLLs beside the test
and prepends the SDK runtime directory to its test environment. This avoids
preferring an older HIP DLL in Windows System32, without modifying the SDK or
system directories.

The executable also accepts `--device ordinal` and `--count elements`; the
defaults are device `0` and `1024` elements. It returns `77` when no HIP GPU is
available, which CTest reports as skipped; other runtime failures remain
failures. It checks communicator initialization, out-of-place float32 SUM
AllReduce, every output element, and resource cleanup. A single rank can use
the one-rank execution path, so this does not validate ring communication,
SendRecv, multiple ranks, or multiple GPUs.

## Remaining work

- Expand and validate collective, datatype, and protocol coverage. The kernel
  filter is for development; it is not a complete RCCL release configuration.
- Validate multiple ranks, multiple GPUs, IPC, and Socket/NetworkDirect traffic.
  The local machine used for validation has one GPU and no usable RDMA adapter.
- Port or validate VMM sharing, fabric discovery, GIN, profiler plugins, RAS,
  Linux NIC telemetry, and optional signal diagnostics. The Windows build uses
  existing Windows stubs or excludes those Linux dependencies where applicable.
- Add Windows CI, complete Linux regressions, and review the separate Windows
  build path, export policy, and remaining option parity
  before promoting this draft to a supported configuration.
