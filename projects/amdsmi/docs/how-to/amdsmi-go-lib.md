---
myst:
  html_meta:
    "description lang=en": "Get started with the AMD SMI Go interface."
    "keywords": "api, smi, lib, go, golang, system, management, interface, ROCm"
---

# AMD SMI Go interface overview

The read-only Linux Go module provides GPU discovery, identity, telemetry,
current partition metadata, and ECC/RAS queries directly through CGO and
`libamd_smi`. It does not use Python, a CLI subprocess, or the legacy Go shim.
See the [API reference](../reference/amdsmi-go-api.md) and the standalone guide
installed at `share/amd_smi/go/README.md` alongside the matching module sources.

(go_prereqs)=
## Read-only module requirements

| Build requirements | Runtime requirements |
| --- | --- |
| Linux, CGO, Go 1.20+, C compiler | AMD GPU driver and native device permissions for GPU queries |
| AMD SMI 27.1 public development header and matching shared library | Matching `libamd_smi.so`, not `libamdsmi.so` |
| Compilation and mock tests need no GPU/root | Go adds no root requirement |

Other operating systems and `CGO_ENABLED=0` are unsupported. The production
file's `linux && cgo` constraint excludes the binding in those configurations,
which is why Go reports that build constraints exclude all Go files.

Follow the [AMD SMI installation guide](../install/install.md)
for native dependencies. The module path is
`github.com/ROCm/rocm-systems/projects/amdsmi/go`; import its `amdsmi` package:

```go
import "github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi"
```

## Build and consume the module

From the module directory, select explicit matching include and library paths:

```bash
: "${AMDSMI_INCLUDE_DIR:?Set the directory containing amd_smi/amdsmi.h}"
: "${AMDSMI_LIBRARY_DIR:?Set the directory containing the matching libamd_smi.so}"
export CGO_ENABLED=1
export CGO_CFLAGS="-I\"${AMDSMI_INCLUDE_DIR}\""
export CGO_LDFLAGS="-L\"${AMDSMI_LIBRARY_DIR}\""
export LD_LIBRARY_PATH="${AMDSMI_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export GOTOOLCHAIN=local GOWORK=off GOPROXY=off GOSUMDB=off
go build ./...
```

The `amd-smi-lib` package ships source assets under `share/amd_smi/go`, grouped
in the CMake/CPack `dev` component, not a separate development package. That
directory contains no tests, fixtures, native binaries, or Go toolchain.
With the native environment configured, consume checkout or installed sources
from an existing application module:

```bash
: "${AMDSMI_GO_SOURCE:?Set the checkout go directory or installed share/amd_smi/go directory}"
go mod edit -require=github.com/ROCm/rocm-systems/projects/amdsmi/go@v0.0.0
go mod edit "-replace=github.com/ROCm/rocm-systems/projects/amdsmi/go=${AMDSMI_GO_SOURCE}"
go build ./...
```

The local `v0.0.0` requirement is only a replacement key. For remote consumption,
pin an immutable published revision containing this module and pair it with the
native 27.1 release for compilation. Any v0 module tag must have the form
`projects/amdsmi/go/v0.X.Y`; this convention does not assert that a tag exists.
Native major 27 does not determine the Go module version. Use a local replacement
for unpublished source changes. The installed module guide gives the separate
network-enabled acquisition command.

## Lifecycle and data handling

Before acquiring the first package reference, `Init` checks the loaded native
version against the compiled header. A different major or older minor returns
`AMDSMI_STATUS_NOT_SUPPORTED`; a newer minor in the same major is allowed, and
release differences are ignored. Version-query failures propagate without
acquiring a native reference. `GetLibVersion` and `StatusCodeToString` remain
available without initialization, including for incompatible-version diagnostics.

Balance every successful `amdsmi.Init(amdsmi.AMDSMI_INIT_AMD_GPUS)` with
`amdsmi.ShutDown()`. Final package shutdown invalidates handles even on cleanup
failure; rediscover after reinitialization. Calls are serialized within this
package, but other bindings must coordinate their native lifetime and
first-initializer flags. Driver reload,
partition changes, external concurrent lifecycle calls, and hotplug recovery are
not supported while initialized.

Use `errors.Is(err, amdsmi.AMDSMI_STATUS_NOT_SUPPORTED)` or `errors.As` with
`*amdsmi.StatusError` to inspect native failures. Failed queries return zero Go
results, including `nil` slices/maps; success can still include unavailable values
or partial native data. `StatusCodeToString` returns `(string, error)` and propagates
lookup failures without replacing the original status of a query error.
The API reference lists field types, units, and native
limitations. The telemetry example installed at
`share/amd_smi/go/examples/telemetry/main.go` balances shutdown and reports
per-query failures without replacing them with zero readings.

## Shared read-only API shape

The common names, signatures, and fields follow the Host declarations; this module
remains a bare-metal (BM) implementation, not a combined backend.
Common spellings such as `Id`, `Gpu`, `Fw`, and `Clk` are deliberately retained.
BM-only names use conventional Go initialisms such as `RAS` and `KFD`, without
aliases or renames.

`GetProcessorHandleFromIndex(uint32)` selects from the current filtered AMD-GPU
discovery order, which need not match CLI `-g`. Use `GetGpuDeviceBdf` to correlate
devices across tools and BDF lookup when selecting a known GPU. For a full
traversal, call `GetProcessorHandles` once and iterate its result; repeated index
lookups rediscover every device and make an N-device traversal O(N²).

| Contract | Bare-metal behavior |
| --- | --- |
| `Init(AMDSMI_INIT_AMD_GPUS)` | GPU-only initialization; no other initializer flags are exported or supported |
| `GetGpuAsicInfo(handle)` | Returns `AsicInfo`, including `RevID`, `AsicSerial`, `OamID`, `PhysicalAccId`, `ChipRevId`, and `ExternalRevId` |
| `GetGpuDeviceBdf(handle)` / `GetProcessorHandleFromBdf(bdf)` | Packed `Bdf uint64`; accessors retain the 48-bit domain and `String()` formats it |
| `GetProcessorHandleFromIndex(index)` | Checked lookup; an invalid index returns `AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS` |
| `GetFwInfo(handle)` | `FwInfo` has a validated `NumFwInfo` and fixed `FwList` array |
| `GetGpuMemoryPartitionConfig(handle)` | `NpsCaps` capability fields, `NumNumaRanges`, and a fixed `NumaRanges` array |
| `GetGpuAcceleratorPartitionProfile(handle)` | Returns `(AcceleratorPartitionProfile, []uint32, error)`; BM returns one current ID, not one per partition |
| `GetGpuEccEnabled(handle)` | Returns `map[GpuBlock]bool`; unknown/reserved enabled bits remain map keys |

Shared source shape does not imply runtime parity or unit conversion:

| Area | Units or limit |
| --- | --- |
| Power | BM socket power is W and voltages are mV; Host documents uW and V respectively. BM `PowerLimit` and power-cap fields are uW; `DpmCap` is a DPM level index, not MHz |
| Clocks | `ClkInfo` is MHz; BM `GetClockFrequencies` returns Hz. `ClkLocked` and `ClkDeepSleep` are false/unavailable, not measured false; BM `ClkLockedRaw`/`ClkDeepSleepRaw` preserve the native bytes |
| Partitions | Empty resources/ranges can be unpopulated. Unknown `NumPartitions == UINT32_MAX` with zero resources is preserved without allocating that count; current ID can remain zero after a subordinate lookup failure |
| BM-only getters | `GetClockFrequencies`, `GetKFDInfo`, `GetMemoryTotal`, `GetMemoryUsage`, `GetRASBlockState` |
| BM-only fields | `Version.Build`, `StatusError.Op`/`Message`, raw clock bytes, and `NpsCaps.RawMask`; native `errors.Is` support is also retained |

No Host runtime parity is claimed. Consult the API reference before interpreting
unavailable values or moving consumers between backends.

## Repository checks

Use the repository's `tests/go/run_tests.py` runner for mock checks; the
`amdsmi_mock` tag is not a standalone test setup or a production option.
Raw `go test -tags=amdsmi_mock` does not build or link the required fixture.
The installed source directory excludes those tests and fixtures. See the
[Go test design](../conceptual/test-design.md#go-checks) for commands and CI coverage.

## Legacy Go interface

The existing `goamdsmi` API and shim remain unchanged. The new module is additive,
not a source-compatible replacement; it does not include legacy CPU or setter APIs.
Use the new module for new read-only GPU integrations; existing consumers keep
their current API and shim path.
The following instructions apply only to the legacy interface.

```{seealso}
Refer to the [Go library API reference](../reference/amdsmi-go-api.md).
```

### Prerequisites

Before get started, make sure your environment satisfies the following prerequisites.
See the [requirements](#install_reqs) section for more information.

1. Ensure `amdgpu` drivers are installed properly for initialization. CPU APIs
   require the `amd_hsmp` kernel module. See {ref}`install_amdgpu_driver`.

2. Export `LD_LIBRARY_PATH` to the `amdsmi` installation directory.

   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64:
   ```

3. Install Go 1.20+.

   Download Go from [https://go.dev/dl/](https://go.dev/dl/) and follow the
   official installation documentation at [Download and
   install](https://go.dev/doc/install).

   Alternatively, use a third-party utility like update-golang.

   ```bash
   git clone https://github.com/udhos/update-golang
   cd update-golang
   sudo ./update-golang.sh
   source /etc/profile.d/golang_path.sh
   go version
   ```

### Get started

```{note}
``hipcc`` and other compilers will not automatically link in the ``libamd_smi``
dynamic library. To compile code that uses the AMD SMI library API, ensure the
``libamd_smi.so`` can be located by setting the ``LD_LIBRARY_PATH`` environment
variable to the directory containing ``librocm_smi64.so`` (usually
``/opt/rocm/lib``) or by passing the ``-lamd_smi`` flag to the compiler.
```

A Go application using AMD SMI must call `goamdsmi.GO_gpu_init()` to initialize
the AMI SMI library before all other calls. This call initializes the internal
data structures required for subsequent AMD SMI operations.

`goamdsmi.GO_gpu_shutdown()` must be the last call to properly close connection to
driver and make sure that any resources held by AMD SMI are released.

### Usage

For an example on using the AMD SMI Go API, refer to this implementation
[https://github.com/amd/amd_smi_exporter/tree/master](https://github.com/amd/amd_smi_exporter/tree/master).

```{seealso}
Refer to the [Go library API reference](../reference/amdsmi-go-api.md).
```

#### Add AMD SMI library to your project

To include the AMD SMI Go API in your project, update your Makefile or Go module configuration
to fetch the appropriate version of the AMD SMI library.

```shell
# Add to go.mod
go get github.com/ROCm/rocm-systems/projects/amdsmi@develop
```

Then import it:

```go
import "github.com/ROCm/rocm-systems/projects/amdsmi"
```

When using a Makefile, ensure you're fetching the latest AMD SMI repository
with Go API support. See
[https://github.com/amd/amd_smi_exporter/blob/master/src/Makefile](https://github.com/amd/amd_smi_exporter/blob/master/src/Makefile)
for an example implementation.
