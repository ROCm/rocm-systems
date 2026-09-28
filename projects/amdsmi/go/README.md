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

See the [AMD SMI installation guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html)
for native requirements.

```go
import "github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi"
```

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
with the matching native 27.1 release. Future module tags need the monorepo
prefix `projects/amdsmi/go/`. Native major 27 is not automatically the Go module
version. Remote acquisition requires a published revision containing the module;
use a local replacement for unpublished source changes.

The development component installs module metadata, production sources, this
guide, the license, and the example under `share/amd_smi/go`. That directory
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
| Initialization | Each successful `Init()` acquires an AMD-GPU reference; balance it with `ShutDown()` |
| Final shutdown | All package handles expire, even on cleanup error; rediscover after reinitialization |
| Concurrency | Calls are serialized; external lifecycle changes must not run concurrently |
| Other bindings | Coordinate first-initializer flags and the entire native lifetime; later initialization does not change those flags |
| Topology changes | No driver reload, partition change, or hotplug recovery while initialized |
| Uninitialized queries/shutdown | `AMDSMI_STATUS_NOT_INIT`; version and status-string lookup do not require initialization |
| Zero/stale handle while initialized | `AMDSMI_STATUS_INVAL` |
| Native failure | Zero Go result plus `*Error`; all numeric statuses survive, including unknown values |
| Error inspection | `errors.Is` matches `StatusCode`; `errors.As` exposes `Op`, `Code`, and `Message`; message lookup can fall back to numeric status |
| Success | Does not imply every field is available or the GPU is healthy |

This excerpt assumes a valid `handle` and imports `errors`, `fmt`, and `amdsmi`:

```go
power, err := amdsmi.GetPowerInfo(handle)
if errors.Is(err, amdsmi.AMDSMI_STATUS_NOT_SUPPORTED) {
    fmt.Println("power query is not supported")
} else if err != nil {
    var native *amdsmi.Error
    if errors.As(err, &native) {
        fmt.Printf("%s failed with status %d\n", native.Op, native.Code)
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
| Clocks | `ClockInfo` in MHz; `Frequencies.Hertz` in Hz; range-check `CurrentIndex` before indexing |
| Power | W, mV, or uW by member name; native unavailable values retain each member's width |
| Power caps | Auxiliary zeros have no validity flag; `DPMLevel` is an index, not MHz |
| Clock flags | `LockedRaw` is currently unpopulated; `DeepSleepRaw` is a narrowed native sleep-frequency value, not a Boolean |
| Activity | Percent; `GFXPercent` can contain 65535 for unavailable data |
| Current partitions | Empty resources/NUMA ranges may be unpopulated; partition count/index can be `UINT32_MAX`; current partition ID can remain zero after a subordinate lookup failure |
| ECC/RAS | Total ECC may omit unavailable blocks while succeeding; `RASFeatureInfo` is metadata, not a health verdict |

There is no global unavailable-value conversion. Strings and slices are copied
to Go-owned storage; reserved C fields are not exposed. Constants preserve the
public C enumerator names and values.

## Compatibility

The existing `goamdsmi` API and shim remain unchanged. This additive module is
not their source-compatible replacement. CPU, NIC, set/reset, all-profile
configuration, and event APIs are outside this module.

## Repository tests

Run from the AMD SMI project root, not the installed module. The Python stdlib
runner builds a controlled native fixture against the real public header.
`amdsmi_mock` is test-only, never a production build tag; fixture files are not
installed. Checks use the local toolchain with downloads disabled.

```bash
python3 -B -m unittest discover -s tests/go -p 'test_*.py' -v
python3 -B tests/go/test_api_contract.py
python3 -B tests/go/run_tests.py
python3 -B tests/go/run_tests.py --race
python3 -B tests/go/run_tests.py --checkptr
python3 -B tests/go/run_tests.py --cgocheck2
python3 -B tests/go/run_tests.py --vet
python3 -B tests/go/run_tests.py --build-example
```

`--cgocheck2` requires Go 1.21+. Native checks require a fresh matching build;
only the version test executes native code, without `Init()` or GPU access.
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
