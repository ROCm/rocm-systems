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

Production bindings live in [amdsmi/amdsmi_interface.go](amdsmi/amdsmi_interface.go).
The common read-only names, signatures, and fields follow the Host declarations.
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
| Initialization | Each successful `Init(AMDSMI_INIT_AMD_GPUS)` acquires an AMD-GPU reference; balance it with `ShutDown()` |
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

### Pre-release API migration

Replace earlier pre-release names directly; no compatibility aliases are provided.

| Previous API | Current API |
| --- | --- |
| `Init()` / `Init(InitAMDGPUs)` | `Init(AMDSMI_INIT_AMD_GPUS)` |
| `StatusCode` / `Error` | `Status` / `StatusError` |
| `StatusString(status)` | `StatusCodeToString(status)`; handle the additional `error` result |
| `GetLibraryVersion` | `GetLibVersion` |
| `GetBDF` / `GetGPUDeviceBDF` | `GetGpuDeviceBdf` |
| `GetProcessorHandleFromBDF` | `GetProcessorHandleFromBdf` |
| `GetUUID` | `GetGpuDeviceUuid` |
| `GetASICInfo` / `GetGPUAsicInfo` | `GetGpuAsicInfo` |
| `GetDriverInfo` / `GetBoardInfo` | `GetGpuDriverInfo` / `GetGpuBoardInfo` |
| `GetFirmwareInfo` | `GetFwInfo`; returns `FwInfo`, not a slice |
| `GetVBIOSInfo` | `GetGpuVbiosInfo` |
| `GetActivity` / `GetTemperature` | `GetGpuActivity` / `GetTempMetric` |
| `GetVRAMInfo` | `GetGpuVramInfo` |
| `GetMemoryPartitionConfig` | `GetGpuMemoryPartitionConfig` |
| `GetAcceleratorPartitionProfile` | `GetGpuAcceleratorPartitionProfile`; returns `(AcceleratorPartitionProfile, []uint32, error)` |
| `GetECCEnabled` | `GetGpuEccEnabled`; returns `(map[GpuBlock]bool, error)`, not a mask |
| `GetECCCount` / `GetTotalECCCount` | `GetGpuEccCount` / `GetGpuTotalEccCount` |
| `GetRASFeatureInfo` | `GetGpuRasFeatureInfo` |

| Previous type or fields | Current type or fields |
| --- | --- |
| `BDF` struct | `Bdf uint64`; use `Domain()`, `Bus()`, `Device()`, `Function()`, and `String()` |
| `ASICInfo` | `AsicInfo` |
| ASIC `RevisionID`, `Serial`, `OAMID`, `ComputeUnits` | `RevID`, `AsicSerial`, `OamID`, `NumComputeUnits` |
| ASIC `PhysicalAcceleratorID`, `ChipRevisionID`, `ExternalRevisionID` | `PhysicalAccId`, `ChipRevId`, `ExternalRevId` |
| Driver `Version`, `Date`, `Name`; board `FRUID` | `DriverVersion`, `DriverDate`, `DriverName`; `FruID` |
| `FirmwareBlock`; `FirmwareInfo.ID`, `.Version` | `FwBlock`; `FwInfoList.FwID`, `.FwVersion`. `FwInfo` carries `NumFwInfo` and `FwList` |
| `VBIOSInfo` / `VRAMInfo` / `VRAMType` | `VbiosInfo` / `VramInfo` / `VramType` |
| VRAM `Type`, `Vendor`, `SizeMB`, `BitWidth`, `MaxBandwidthGBPerSecond` | `VramType`, `VramVendor`, `VramSize`, `VramBitWidth`, `VramMaxBandwidth` |
| `ClockType` / `ClockInfo` | `ClkType` / `ClkInfo` |
| Clock `ClockMHz`, `MinClockMHz`, `MaxClockMHz`; `LockedRaw`, `DeepSleepRaw` | `Clk`, `MinClk`, `MaxClk`; BM `ClkLockedRaw`, `ClkDeepSleepRaw`. Common Boolean fields are unavailable on BM |
| Frequencies `CurrentIndex`, `Hertz` | `Current`, `Values`; `NumSupported` carries the validated count |
| `Activity.GFXPercent`, `.UMCPercent`, `.MMPercent` | `EngineUsage.GfxActivity`, `.UmcActivity`, `.MmActivity` |
| Power `SocketPowerWatts`, `CurrentSocketPowerWatts`, `AverageSocketPowerWatts`, `UBBPowerWatts` | `SocketPower`, `CurrentSocketPower`, `AverageSocketPower`, `UbbPower` |
| Power `GFXVoltageMillivolts`, `SOCVoltageMillivolts`, `MemoryVoltageMillivolts`, `PowerLimitMicrowatts` | `GfxVoltage`, `SocVoltage`, `MemVoltage`, `PowerLimit` |
| Caps `PowerCapMicrowatts`, `DefaultPowerCapMicrowatts`, `MinPowerCapMicrowatts`, `MaxPowerCapMicrowatts`, `DPMLevel` | `PowerCap`, `DefaultPowerCap`, `MinPowerCap`, `MaxPowerCap`, `DpmCap` |
| `MemoryCapabilities` | `NpsCaps` Boolean fields and `Supported()`/`String()`; BM `RawMask` retains all native bits |
| `NUMARange`; config `Capabilities`, `NUMARanges` | `NumaRange`; `PartitionCaps`, fixed `NumaRanges` array with `NumNumaRanges` |
| Profile `Type`, `MemoryCapabilities`, `PartitionID` | `ProfileType`, `MemoryCaps`; current ID moves to the separate one-element result slice on BM |
| `GPUBlock`; `ECCCounts.Correctable`, `.Uncorrectable`, `.Deferred` | `GpuBlock`; `ErrorCount.CorrectableCount`, `.UncorrectableCount`, `.DeferredCount` |
| `RASFeatureInfo.EEPROMVersion`, `.ECCCorrectionSchema` | `RasFeatureInfo.RasEepromVersion`, `.EccCorrectionSchemaFlag` |

`FwBlock`, `ClkType`, `VramType`, `TemperatureType`, `TemperatureMetric`,
`MemoryPartitionType`, and `AcceleratorPartitionType` use `int32`; `Status` uses
`uint32`, and `GpuBlock` uses `uint64`. Constants remain bound to the BM C header.
`Bdf` packs function/device/bus/domain into 3/5/8/48 bits; every `uint64` value is
representable, including the full 48-bit domain.

### BM extensions and compatibility limits

| BM extension | Contract |
| --- | --- |
| `GetClockFrequencies`, `GetKFDInfo`, `GetMemoryTotal`, `GetMemoryUsage`, `GetRASBlockState` | BM-only getters; shared argument types use the new names |
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

The external-package contract test checks common names, types, fields, and
initialization; it does not verify Host runtime behavior.
`--cgocheck2` requires Go 1.21+. Native checks require a fresh matching build;
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
