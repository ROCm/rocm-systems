# Native Windows development build

This experimental path builds `rccl.dll` and its import library `rccl.lib`.
It has been validated on Windows 11 x64 with ROCm 10.2.0 and Radeon 8060S
(`gfx1151`), using the restricted kernel selection below. The single-GPU
test checks communicator initialization, an out-of-place float32 SUM
AllReduce over 1,024 elements, and resource cleanup. A single rank can use
the one-rank execution path; this does not validate ring communication.

## Prerequisites

- Visual Studio 2022 C++ Build Tools and a complete Windows SDK.
- CMake and Ninja, available in a Developer PowerShell session.
- Python 3 and Git for Windows, including its Bash and Perl tools. The current
  staging script expects Git under `%ProgramFiles%\Git`.
- A writable extracted ROCm 10.2.0 Windows SDK, with `HIP_PATH` set to its root.
  The current build file targets this SDK layout and version.

From the repository root in Developer PowerShell:

```powershell
$env:ROCM_PATH = $env:HIP_PATH
$env:HIP_PLATFORM = 'amd'
$env:LLVM_PATH = "$env:HIP_PATH\lib\llvm"
$env:HIP_DEVICE_LIB_PATH = "$env:LLVM_PATH\amdgcn\bitcode"
$env:Path = "$env:HIP_PATH\bin;$env:LLVM_PATH\bin;$env:Path"
cmake -S projects/rccl -B build/rccl-windows -G Ninja `
  "-DROCM_PATH=$env:HIP_PATH" -DGPU_TARGETS=gfx1151 -DCMAKE_BUILD_TYPE=Release `
  '-DONLY_FUNCS=AllReduce RING SIMPLE Sum f32|SendRecv RING SIMPLE Sum i8'
cmake --build build/rccl-windows --target rccl --parallel 8
```

Re-run configuration before rebuilding after source changes: source HIPification
currently happens during configuration. Windows uses the ordinary HIP device
link flow, rather than the Linux custom device-linker pipeline.

## Smoke test

The following keeps the test beside the SDK's matching runtime DLLs. This avoids
loading an older `amdhip64_7.dll` from Windows System32. It requires a writable
SDK `bin` directory. No system DLLs are replaced.

```powershell
$build = (Resolve-Path build/rccl-windows).Path
Copy-Item "$build\rccl.dll" "$env:HIP_PATH\bin\rccl.dll" -Force
clang++ -fms-runtime-lib=dll -x hip --hip-link --offload-arch=gfx1151 `
  projects/rccl/test/windows/smoke.cpp -I "$build\include" `
  -x none "$build\rccl.lib" -o "$env:HIP_PATH\bin\rccl-smoke.exe"
$env:NCCL_DEBUG = 'INFO'
& "$env:HIP_PATH\bin\rccl-smoke.exe"
```

Expected result:

```text
RCCL version: 23206
PASS: RCCL single-GPU float32 AllReduce verified (1024 elements).
```

## Remaining work

- Expand and validate collective, datatype, and protocol coverage. The kernel
  filter is for development; it is not a complete RCCL release configuration.
- Validate multiple ranks, multiple GPUs, IPC, and Socket/NetworkDirect traffic.
  The local machine used for validation has one GPU and no usable RDMA adapter.
- Port or validate VMM sharing, fabric discovery, GIN, profiler plugins, RAS,
  Linux NIC telemetry, and optional signal diagnostics. The Windows build uses
  existing Windows stubs or excludes those Linux dependencies where applicable.
- Add Windows CI, test Linux regressions, and review the separate Windows build
  path, SDK/version assumptions, source staging, export policy, and option parity
  before promoting this draft to a supported configuration.
