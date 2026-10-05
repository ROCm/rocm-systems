# Read-only AMD SMI Go module

Linux GPU discovery, identity, telemetry, current partition metadata, and ECC/RAS
queries through CGO and the AMD SMI native library. No Python/CLI subprocess or
legacy shim dependency is used. Native dependencies are still required.

## Requirements and imports

| Requirement | Value |
| --- | --- |
| Build | Linux, CGO, Go 1.20+, a C compiler |
| Native dependency | AMD SMI 27.1 public development header and matching shared library |
| Runtime GPU queries | Loaded AMD GPU driver and native device permissions; Go adds no root requirement |
| Compilation and mock tests | No GPU or root required |
| Module | `github.com/ROCm/rocm-systems/projects/amdsmi/go` |
| Package import | `github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi` |
| Native library | `libamd_smi` (`-lamd_smi`), not `libamdsmi` |

Go 1.20 is the language/toolchain floor declared by the Go module. The native
AMD SMI 27.1 dependency is separate: `Init` checks the loaded library against
the compiled C header, not against the Go version. Both requirements apply.

Other operating systems and `CGO_ENABLED=0` are unsupported. The production
file requires `linux && cgo`, so Go reports that build constraints exclude all
Go files when either condition is missing.

See the [AMD SMI installation guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html)
for native requirements.

```go
import "github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi"
```

Production bindings remain in the single [amdsmi/amdsmi_interface.go](amdsmi/amdsmi_interface.go)
file to match the required Host source layout.
The common read-only names, signatures, and fields follow the Host declarations.
Spellings such as `Id`, `Gpu`, `Fw`, and `Clk` are retained deliberately for
shared source compatibility. BM-only names use conventional Go initialisms,
including `RAS` and `KFD`; no aliases or renames are introduced.
This is a bare-metal (BM) implementation, not a combined BM/Host backend. Shared
source shape does not imply identical units, field availability, or runtime behavior.

## Build with a custom native prefix

Run from this module directory with explicit header and library paths:

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

The [telemetry example](examples/telemetry/main.go) balances initialization and
shutdown and reports each query failure separately. Building it does not run GPU
queries; executing it does.

## Source pairing and consumption

Pin an immutable published source revision containing this module and pair it
with the matching native 27.1 release for compilation. Any v0 module tag must use
the monorepo form `projects/amdsmi/go/v0.X.Y`; this is a naming convention, not a
claim that a tag exists. Native major 27 is not the Go module version.
Remote acquisition requires a published revision containing the module;
use a local replacement for unpublished source changes.

The `amd-smi-lib` package ships module metadata, production sources, this guide,
the license, and the example under `share/amd_smi/go`, grouped in the CMake/CPack
`dev` component rather than a separate development package. That directory
contains no tests, fixtures, native binaries, or Go toolchain. After configuring
the native environment above, use a local replacement from an existing
application module:

```bash
: "${AMDSMI_GO_SOURCE:?Set the checkout go directory or installed share/amd_smi/go directory}"
go mod edit -require=github.com/ROCm/rocm-systems/projects/amdsmi/go@v0.0.0
go mod edit "-replace=github.com/ROCm/rocm-systems/projects/amdsmi/go=${AMDSMI_GO_SOURCE}"
go build ./...
```

`v0.0.0` is a local replacement key, not a published version. Network-enabled
acquisition is a separate user action, not part of offline verification:

```bash
: "${AMDSMI_GO_REV:?Set an immutable published revision that contains this Go module}"
GOTOOLCHAIN=local GOWORK=off GOPROXY=https://proxy.golang.org,direct GOSUMDB=sum.golang.org \
    go get "github.com/ROCm/rocm-systems/projects/amdsmi/go@${AMDSMI_GO_REV}"
```

## Lifecycle and errors

| Contract | Behavior |
| --- | --- |
| Initialization | Each successful `Init(AMDSMI_INIT_AMD_GPUS)` acquires an AMD-GPU reference; balance it with `ShutDown()` |
| Runtime version | Before the first package reference, `Init` rejects a different native major or an older minor than the compiled header with `AMDSMI_STATUS_NOT_SUPPORTED`; newer minors in the same major are allowed, and release differences are ignored. A version-query failure is propagated; neither failure acquires a native reference |
| Flags | `InitFlags` is `uint64`; only the C-bound `AMDSMI_INIT_AMD_GPUS` is exported and supported. Other values return `AMDSMI_STATUS_INVAL` before calling C |
| Index lookup | `GetProcessorHandleFromIndex(uint32)` uses current filtered GPU discovery order; an out-of-range index returns `AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS` |
| Final shutdown | All package handles expire, even on cleanup error; rediscover after reinitialization |
| Concurrency | Calls are serialized; external lifecycle changes must not run concurrently |
| Other bindings | Coordinate first-initializer flags and the entire native lifetime; later initialization does not change those flags |
| Topology changes | No driver reload, partition change, or hotplug recovery while initialized |
| Uninitialized queries/shutdown | `AMDSMI_STATUS_NOT_INIT`; version and status-string lookup do not require initialization |
| Zero/stale handle while initialized | `AMDSMI_STATUS_INVAL` |
| Query failure | Zero Go results plus `*StatusError`; slices/maps are `nil`; all native numeric statuses survive, including unknown values |
| Error inspection | `errors.Is` matches `Status`; `errors.As` exposes `Code`, symbolic `Name`, and BM diagnostics `Op`/`Message` |
| Status lookup | `StatusCodeToString(Status) (string, error)` propagates native lookup failure; success with a null string returns `AMDSMI_STATUS_UNEXPECTED_DATA`. Formatting a query error never replaces its original status |
| Success | Does not imply every field is available or the GPU is healthy |

Go indices select only the current filtered AMD-GPU list and need not match CLI
`-g` indices. Correlate devices across tools using `GetGpuDeviceBdf`. When visiting
all GPUs, call `GetProcessorHandles` once and iterate the returned handles;
repeated `GetProcessorHandleFromIndex` calls rediscover the whole list and make
an N-device traversal O(N²).

This excerpt assumes a valid `handle` and imports `errors`, `fmt`, and `amdsmi`:

```go
power, err := amdsmi.GetPowerInfo(handle)
if errors.Is(err, amdsmi.AMDSMI_STATUS_NOT_SUPPORTED) {
	fmt.Println("power query is not supported")
} else if err != nil {
	var native *amdsmi.StatusError
	if errors.As(err, &native) {
		fmt.Printf("%s failed with %s (status %d)\n", native.Op, native.Name, native.Code)
	}
} else {
	fmt.Printf("raw power fields: %+v\n", power)
}
```

## Values and native gaps

| Data | Units and limitations |
| --- | --- |
| Temperature | Signed degrees C |
| Memory total/usage | Bytes |
| VRAM | MB, bits, GB/s by member |
| Clocks | `ClkInfo` in MHz; BM `GetClockFrequencies` returns `Frequencies.Values` in Hz. Range-check `Current` against `NumSupported` and `len(Values)` |
| Power | BM `SocketPower`, `CurrentSocketPower`, `AverageSocketPower`, `UbbPower` are W; `GfxVoltage`, `SocVoltage`, `MemVoltage` are mV; `PowerLimit` is uW. Host documents `SocketPower` in uW and voltages in V; no conversion is applied |
| Power caps | `PowerCap`, `DefaultPowerCap`, `MinPowerCap`, `MaxPowerCap` are uW on BM; auxiliary zeros have no validity flag. `DpmCap` is a BM DPM level index, not MHz |
| Clock flags | `ClkLocked` and `ClkDeepSleep` are always false/unavailable on BM, not measured false. `ClkLockedRaw` preserves the unpopulated native byte; `ClkDeepSleepRaw` preserves the narrowed sleep-frequency byte. Neither 255 nor 222 is interpreted as true |
| Activity | Percent; `GfxActivity` can contain 65535 for unavailable data |
| Current partitions | `GetGpuAcceleratorPartitionProfile` returns a profile and exactly one current ID on BM, not one ID per partition. Count/index can be `UINT32_MAX`; the ID can remain zero after a subordinate lookup failure |
| Partition metadata | Empty resources or `NumNumaRanges == 0` can mean unpopulated data. Unknown `NumPartitions == UINT32_MAX` with zero resources is preserved without allocating that count |
| ECC/RAS | `GetGpuEccEnabled` returns known-block Boolean entries plus unknown/reserved enabled bits as keys. Total ECC can omit unavailable blocks while succeeding; `RasFeatureInfo` is not a health verdict |

There is no global unavailable-value conversion. Strings and slices are copied
to Go-owned storage; reserved C fields are not exposed. Constants preserve the
public C enumerator names and values.

## Compatibility

`FwBlock`, `ClkType`, `VramType`, `TemperatureType`, `TemperatureMetric`,
`MemoryPartitionType`, and `AcceleratorPartitionType` use `int32`; `Status` uses
`uint32`, and `GpuBlock` uses `uint64`. Constants remain bound to the BM C header.
`Bdf` packs function/device/bus/domain into 3/5/8/48 bits; every `uint64` value is
representable, including the full 48-bit domain.

### BM extensions and compatibility limits

| BM extension | Contract |
| --- | --- |
| `GetClockFrequencies`, `GetKFDInfo`, `GetMemoryTotal`, `GetMemoryUsage`, `GetRASBlockState` | BM-only getters; shared argument types retain Host spelling |
| `Version.Build` | Native build string |
| `StatusError.Op`, `StatusError.Message`, `StatusError.Unwrap()`, `Status.Error()` | BM diagnostics and native status matching with `errors.Is` |
| `ClkInfo.ClkLockedRaw`, `ClkInfo.ClkDeepSleepRaw` | Preserve native bytes; common Boolean fields remain false/unavailable |
| `NpsCaps.RawMask` | Preserves unknown capability bits |

Common source shape is not full backend equivalence. Units and availability remain
platform-specific; no Host runtime parity is claimed. Index lookup uses Go
enumeration on BM, not a new C API. The
[API reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-go-api.html)
lists complete signatures, fields, and array bounds.

The existing `goamdsmi` API and shim remain unchanged. This additive module is
recommended for new read-only GPU integrations, not as a source-compatible
replacement for existing consumers. CPU, NIC, set/reset, all-profile configuration,
and event APIs are outside this module; existing consumers keep their current path.

## Repository tests

Run from the AMD SMI project root, not the installed module. The Python stdlib
runner builds a controlled native fixture against the real public header.
Use `amdsmi_mock` only through this runner, never in production. Raw
`go test -tags=amdsmi_mock` does not build or link the required fixture library.
Fixture files are not installed. Checks use the local toolchain with downloads disabled.

```bash
python3 -B -m unittest discover -s tests/go -p 'test_*.py' -v
python3 -B tests/go/test_api_contract.py
python3 -B tests/go/run_tests.py
python3 -B tests/go/run_tests.py --race
python3 -B tests/go/run_tests.py --checkptr
python3 -B tests/go/run_tests.py --cgocheck2
python3 -B tests/go/run_tests.py --asan
python3 -B tests/go/run_tests.py --vet
python3 -B tests/go/run_tests.py --build-example
```

The external-package contract test checks common names, types, fields, and
initialization; it does not verify Host runtime behavior.
Python tooling tests cover option parsing, environment isolation, cache safety,
and contract/install guards. Runner smoke tests use real subprocesses for the
default fixture suite, ASAN partition/lifecycle cases, vet, and example builds.
Missing Linux, Go, or GCC prerequisites produce visible local skips; CI provides
the required tools. Contract guards still check partial-enum coverage, explicit
exclusions, and the module declaration.

`--cgocheck2` requires Go 1.21+. `--asan` instruments the Go binary and C fixture
with AddressSanitizer, selects GCC, and cannot be combined with `--race`.
The dedicated repository-root `.github/workflows/amdsmi-go.yml` configures
CPU-only `ubuntu-24.04` jobs for Go 1.20.14 and 1.24.1, with tooling tests, API
contracts, default/race/checkptr fixtures, vet, and example builds. Go 1.24.1
adds explicit cgocheck2 and ASAN runs; this selection does not declare ASAN
unsupported on Go 1.20. Both versions build the native shared library and run
the native and staged checks below without GPU access. The existing
`.github/workflows/amdsmi-build.yml` retains the GPU build/test jobs.

The pinned actionlint pre-commit hook checks both workflows, including Actions
expression contexts, instead of Python substring assertions. These are
configured checks, not evidence that a hosted CI run has passed.

Native checks require a fresh matching build;
only the version test executes native code, without initialization or GPU access.
The example is linked, not run. The staging check uses temporary `DESTDIR`,
verifies installed source contents, and builds an independent local-replacement
consumer against the staged header/library pair:

```bash
: "${AMDSMI_NATIVE_BUILD_DIR:?Set a fresh native build directory}"
: "${AMDSMI_NATIVE_INCLUDE_DIR:?Set the matching public-header include directory}"
: "${AMDSMI_NATIVE_LIBRARY_DIR:?Set the matching native shared-library directory}"
python3 -B tests/go/run_tests.py --native \
    --include-dir "$AMDSMI_NATIVE_INCLUDE_DIR" --library-dir "$AMDSMI_NATIVE_LIBRARY_DIR" \
    --run '^TestNativeVersion$'
python3 -B tests/go/run_tests.py --native \
    --include-dir "$AMDSMI_NATIVE_INCLUDE_DIR" --library-dir "$AMDSMI_NATIVE_LIBRARY_DIR" \
    --vet
python3 -B tests/go/run_tests.py --native \
    --include-dir "$AMDSMI_NATIVE_INCLUDE_DIR" --library-dir "$AMDSMI_NATIVE_LIBRARY_DIR" \
    --build-example
python3 -B tests/go/test_install.py --build-dir "$AMDSMI_NATIVE_BUILD_DIR"
```
