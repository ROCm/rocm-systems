---
myst:
  html_meta:
    "description lang=en": "Explore the AMD SMI Go API."
    "keywords": "api, smi, lib, system, management, interface, ROCm, golang"
---

# AMD SMI Go API reference

## Read-only Go module

Import `github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi`. Requires Linux,
CGO, Go 1.20+, a C compiler, and the AMD SMI 27.1 public header and matching
`libamd_smi` shared library. See the [setup guide](../how-to/amdsmi-go-lib.md)
and the standalone guide installed at `share/amd_smi/go/README.md`. CPU, NIC,
set/reset, all-profile configuration, and event APIs are not included.
Other operating systems and `CGO_ENABLED=0` are unsupported; `linux && cgo`
constraints exclude the production binding and produce Go's build-constraints error.

Common read-only names, signatures, and fields follow the Host declarations.
The implementation remains bare metal (BM); matching source shape does not imply
identical units, availability, or runtime behavior. No Host runtime parity is claimed.
Common spellings such as `Id`, `Gpu`, `Fw`, and `Clk` are retained deliberately;
BM-only names use conventional Go initialisms such as `RAS` and `KFD`. No aliases
or renames are introduced.

Results use Go-owned scalars, strings, arrays, slices, and maps. No public field
exposes a C type or borrowed buffer. Errors return zero Go results, including
`nil` slices/maps; successful fields retain native units and unavailable markers.
Constants bind to the BM C enumerators, not copied Host numeric values. Unknown
numeric values are preserved, and unknown input enums are forwarded to C.

### Lifecycle, discovery, and errors

| Signature | Contract |
| --- | --- |
| `Init(InitFlags) error` | Acquires one AMD-GPU reference; only `AMDSMI_INIT_AMD_GPUS` is accepted |
| `ShutDown() error` | Releases one reference; final package shutdown expires all handles, even on cleanup error |
| `GetProcessorHandles() ([]ProcessorHandle, error)` | Enumerates AMD GPUs across sockets, preserving native order without a fixed device cap |
| `GetProcessorHandleFromIndex(uint32) (ProcessorHandle, error)` | Selects the current filtered discovery index; out-of-range indices return `AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS` |
| `GetProcessorHandleFromBdf(Bdf) (ProcessorHandle, error)` | Native BDF lookup; every packed `uint64` is representable |
| `GetGpuDeviceBdf(ProcessorHandle) (Bdf, error)` | Reads packed PCI domain, bus, device, and function |
| `(Bdf).Function() uint8`, `(Bdf).Device() uint8`, `(Bdf).Bus() uint8` | Extract 3-bit function, 5-bit device, and 8-bit bus |
| `(Bdf).Domain() uint64` | Extract the full 48-bit domain |
| `(Bdf).String() string` | Hexadecimal `domain:bus:device.function`, retaining the full domain |
| `GetLibVersion() (Version, error)` | Native library version; does not require initialization |
| `StatusCodeToString(Status) (string, error)` | Native message lookup; propagates lookup failure. Success with a null string returns `AMDSMI_STATUS_UNEXPECTED_DATA`; no initialization required |
| `(Status).Error() string` | BM extension: numeric status text |
| `(*StatusError).Error() string` | Status and diagnostics |
| `(*StatusError).Unwrap() error` | BM extension: underlying `Status` for `errors.Is` |

| Type | Fields / underlying type | Meaning |
| --- | --- | --- |
| `ProcessorHandle` | Opaque struct, no exported fields | Package-lifetime handle; rediscover after final shutdown |
| `InitFlags` | `uint64` | Only C-bound `AMDSMI_INIT_AMD_GPUS` is exported; zero and other values return `AMDSMI_STATUS_INVAL` |
| `Bdf` | `uint64` | Function bits 0-2, device 3-7, bus 8-15, domain 16-63 |
| `Version` | `Major, Minor, Release uint32`; `Build string` | Common version components; `Build` is a BM extension |
| `Status` | `uint32` | All native `AMDSMI_STATUS_*` values preserved, including unknown numeric values |
| `StatusError` | `Code Status`; `Name string`; `Op, Message string` | Common code and symbolic name for known statuses; `Op`/`Message` are BM diagnostics. Inspect with `errors.As` into `*StatusError` |

Every successful `Init(AMDSMI_INIT_AMD_GPUS)` must be balanced. Calls are serialized, including
status/version lookup. External bindings must coordinate the entire native
lifetime and first-initializer flags; later initialization does not change those
flags. No external concurrent lifecycle, driver reload, partition change, or
hotplug recovery is supported while initialized.

Before the first package reference, `Init` rejects a native major different from
the compiled header or an older minor with `AMDSMI_STATUS_NOT_SUPPORTED`. Newer
minors in the same major are allowed; release differences are ignored. A native
version-query error is propagated. Neither failure acquires a native reference.

Go indices refer only to the current filtered AMD-GPU list and need not match
CLI `-g`. Correlate devices across tools with `GetGpuDeviceBdf`. To visit every
GPU, call `GetProcessorHandles` once and iterate its result instead of repeatedly
calling `GetProcessorHandleFromIndex`, which rediscovers the whole list each time.

Except for version and status-string lookup, queries without a package reference
return `AMDSMI_STATUS_NOT_INIT`; with a reference, zero or stale handles return
`AMDSMI_STATUS_INVAL`. Formatting a query
error never replaces its original status if message lookup fails, and does not
recurse through status lookup. Successful data is not a health assessment.

See [lifecycle and error handling](../how-to/amdsmi-go-lib.md#lifecycle-and-data-handling)
and the {download}`telemetry example <../../go/examples/telemetry/main.go>`, also
installed at `share/amd_smi/go/examples/telemetry/main.go`, for a complete example.

### Identity and firmware

| Signature | Native entry |
| --- | --- |
| `GetGpuDeviceUuid(ProcessorHandle) (string, error)` | `amdsmi_get_gpu_device_uuid` |
| `GetGpuAsicInfo(ProcessorHandle) (AsicInfo, error)` | `amdsmi_get_gpu_asic_info` |
| `GetGpuDriverInfo(ProcessorHandle) (DriverInfo, error)` | `amdsmi_get_gpu_driver_info` |
| `GetGpuBoardInfo(ProcessorHandle) (BoardInfo, error)` | `amdsmi_get_gpu_board_info` |
| `GetFwInfo(ProcessorHandle) (FwInfo, error)` | `amdsmi_get_fw_info` |
| `GetGpuVbiosInfo(ProcessorHandle) (VbiosInfo, error)` | `amdsmi_get_gpu_vbios_info` |

| Type | Fields and Go types | Meaning |
| --- | --- | --- |
| `AsicInfo` | `MarketName, VendorName, AsicSerial string` | Copied identity strings |
| `AsicInfo` | `VendorID, SubvendorID, RevID, OamID, NumComputeUnits, SubsystemID uint32` | Native IDs and compute-unit count |
| `AsicInfo` | `DeviceID, TargetGraphicsVersion, Flags uint64` | Native ID, graphics target, and raw flags |
| `AsicInfo` | `PhysicalAccId, ChipRevId, ExternalRevId uint32` | Native physical/revision IDs; `UINT32_MAX` can mean unavailable |
| `DriverInfo` | `DriverVersion, DriverDate, DriverName string` | Native driver metadata |
| `BoardInfo` | `ModelNumber, ProductSerial, FruID, ProductName, ManufacturerName string` | Native board metadata |
| `FwBlock` | `int32` | C-bound `AMDSMI_FW_ID_*` constants |
| `FwInfoList` | `FwID FwBlock`; `FwVersion uint64` | Firmware identifier and raw version |
| `FwInfo` | `NumFwInfo uint8`; `FwList [AMDSMI_FW_ID__MAX]FwInfoList` | Only the first `NumFwInfo` entries are populated |
| `VbiosInfo` | `Name, BuildDate, PartNumber, Version, BootFirmware string` | Native VBIOS/boot metadata |

Fixed strings are bounded by their C capacity. Firmware counts are checked against
the array capacity and `uint8` width before copying or narrowing. Reserved fields
are omitted.

### Telemetry

| Signature | Native entry | Units / rules |
| --- | --- | --- |
| `GetTempMetric(ProcessorHandle, TemperatureType, TemperatureMetric) (int64, error)` | `amdsmi_get_temp_metric` | Whole degrees C, including negatives, not millidegrees |
| `GetPowerInfo(ProcessorHandle) (PowerInfo, error)` | `amdsmi_get_power_info` | W, mV, uW by member |
| `GetPowerCapInfo(ProcessorHandle, uint32) (PowerCapInfo, error)` | `amdsmi_get_power_cap_info` | Second parameter is the sensor index; partial zeros preserved |
| `GetClockInfo(ProcessorHandle, ClkType) (ClkInfo, error)` | `amdsmi_get_clock_info` | MHz; Boolean fields unavailable on BM |
| `GetClockFrequencies(ProcessorHandle, ClkType) (Frequencies, error)` | `amdsmi_get_clk_freq` | BM extension: Hz; caller must range-check `Current` |
| `GetGpuActivity(ProcessorHandle) (EngineUsage, error)` | `amdsmi_get_gpu_activity` | Percentages, including native unavailable values |
| `GetMemoryTotal(ProcessorHandle, MemoryType) (uint64, error)` | `amdsmi_get_gpu_memory_total` | BM extension: bytes |
| `GetMemoryUsage(ProcessorHandle, MemoryType) (uint64, error)` | `amdsmi_get_gpu_memory_usage` | BM extension: bytes |
| `GetGpuVramInfo(ProcessorHandle) (VramInfo, error)` | `amdsmi_get_gpu_vram_info` | MB, bits, GB/s by member |

| Enum type | Underlying type | Constant family |
| --- | --- | --- |
| `TemperatureType` | `int32` | `AMDSMI_TEMPERATURE_TYPE_*`, including GPU-board/baseboard sensors |
| `TemperatureMetric` | `int32` | `AMDSMI_TEMP_*` |
| `ClkType` | `int32` | `AMDSMI_CLK_TYPE_*` |
| `MemoryType` | `uint32` | BM extension: `AMDSMI_MEM_TYPE_*` |
| `VramType` | `int32` | `AMDSMI_VRAM_TYPE_*` |

| Type | Fields and Go types | Units / limitations |
| --- | --- | --- |
| `PowerInfo` | `SocketPower uint64` | W on BM; Host documents uW |
| `PowerInfo` | `CurrentSocketPower, AverageSocketPower, UbbPower uint32` | W on BM |
| `PowerInfo` | `GfxVoltage, SocVoltage, MemVoltage uint64` | mV on BM; Host documents V |
| `PowerInfo` | `PowerLimit uint32` | uW on BM |
| `PowerCapInfo` | `PowerCap, DefaultPowerCap, MinPowerCap, MaxPowerCap uint64` | uW on BM; auxiliary zeros after partial success have no validity flag |
| `PowerCapInfo` | `DpmCap uint64` | BM DPM level index, not MHz |
| `ClkInfo` | `Clk, MinClk, MaxClk uint32` | MHz; preserve `UINT32_MAX` for unavailable values |
| `ClkInfo` | `ClkLocked, ClkDeepSleep bool` | Always false/unavailable on BM, not measured false |
| `ClkInfo` | `ClkLockedRaw uint8` | BM extension: preserves the currently unpopulated native byte |
| `ClkInfo` | `ClkDeepSleepRaw uint8` | BM extension: preserves the narrowed native sleep-frequency byte, not a Boolean |
| `Frequencies` | `HasDeepSleep bool`; `NumSupported, Current uint32`; `Values []uint64` | BM `GetClockFrequencies` returns Hz; validate `Current` against both `NumSupported` and `len(Values)` |
| `EngineUsage` | `GfxActivity, UmcActivity, MmActivity uint32` | Percent; unavailable data can be 65535, including in `GfxActivity` |
| `VramInfo` | `VramType VramType`; `VramVendor string` | Memory type and vendor |
| `VramInfo` | `VramSize uint64`; `VramBitWidth uint32`; `VramMaxBandwidth uint64` | MB, bits, GB/s respectively |

Power unavailable markers retain each member's width. No unit normalization is
applied. Host units above describe its documented contract, not verified runtime
measurements. `Frequencies` has the same shape used for Host PCI bandwidth, but
its BM getter still returns clock frequencies in Hz; `Current` can be
`UINT32_MAX` or outside the slice.

BM clock bytes do not reliably encode the common Boolean fields. Neither the
unavailable byte 255 nor a narrowed sleep-frequency byte such as 222 becomes
`true`; read the raw extensions when needed. Except for these unavailable
Boolean fields, there is no global conversion of unavailable data into zero,
`nil`, or another unit.

### Current partitions, ECC, and RAS

| Signature | Native entry | Rules |
| --- | --- | --- |
| `GetKFDInfo(ProcessorHandle) (KFDInfo, error)` | `amdsmi_get_gpu_kfd_info` | BM extension: raw KFD/node/current-partition IDs |
| `GetGpuMemoryPartitionConfig(ProcessorHandle) (MemoryPartitionConfig, error)` | `amdsmi_get_gpu_memory_partition_config` | Current mode and capabilities |
| `GetGpuAcceleratorPartitionProfile(ProcessorHandle) (AcceleratorPartitionProfile, []uint32, error)` | `amdsmi_get_gpu_accelerator_partition_profile` | Current profile plus one current ID on BM, not all partition IDs |
| `GetGpuEccEnabled(ProcessorHandle) (map[GpuBlock]bool, error)` | `amdsmi_get_gpu_ecc_enabled` | Known-block Boolean entries plus any unknown/reserved enabled bits as keys |
| `GetGpuEccCount(ProcessorHandle, GpuBlock) (ErrorCount, error)` | `amdsmi_get_gpu_ecc_count` | Per-block counters |
| `GetGpuTotalEccCount(ProcessorHandle) (ErrorCount, error)` | `amdsmi_get_gpu_total_ecc_count` | Native totals can omit unavailable blocks while succeeding |
| `GetRASBlockState(ProcessorHandle, GpuBlock) (RASState, error)` | `amdsmi_get_gpu_ras_block_features_enabled` | BM extension: state enum, not a Boolean |
| `GetGpuRasFeatureInfo(ProcessorHandle) (RasFeatureInfo, error)` | `amdsmi_get_gpu_ras_feature_info` | Populated metadata only, not a health verdict |
| `(NpsCaps).Supported() []MemoryPartitionType` | No native call | Enabled NPS1/NPS2/NPS4/NPS8 modes in that order |
| `(NpsCaps).String() string` | No native call | Formats the supported modes |

| Type | Fields / underlying type | Meaning / limitations |
| --- | --- | --- |
| `KFDInfo` | `KFDID uint64`; `NodeID, CurrentPartitionID uint32` | BM extension: native IDs; unavailable values preserved |
| `MemoryPartitionType` | `int32` | `AMDSMI_MEMORY_PARTITION_*` constants |
| `AcceleratorPartitionType` | `int32` | `AMDSMI_ACCELERATOR_PARTITION_*` constants |
| `NpsCaps` | `Nps1Cap, Nps2Cap, Nps4Cap, Nps8Cap bool`; `RawMask uint32` | Common Boolean capabilities; BM `RawMask` preserves all native bits, not NPS enum values |
| `NumaRange` | `MemoryType VramType`; `Start, End uint64` | Native memory-address range, copied without conversion |
| `MemoryPartitionConfig` | `PartitionCaps NpsCaps`; `Mode MemoryPartitionType`; `NumNumaRanges uint32`; `NumaRanges [AMDSMI_MAX_NUM_NUMA_NODES]NumaRange` | First `NumNumaRanges` entries populated; zero count can mean unavailable metadata, not absence of memory |
| `AcceleratorPartitionProfile` | `ProfileType AcceleratorPartitionType`; `MemoryCaps NpsCaps` | Current profile type and memory capabilities |
| `AcceleratorPartitionProfile` | `NumPartitions, ProfileIndex, NumResources uint32` | Native counts/indices; partition count/index can be `UINT32_MAX` |
| `AcceleratorPartitionProfile` | `Resources [][]uint32` | Per-partition resource indices; current native metadata is empty |
| `GpuBlock` | `uint64` | `AMDSMI_GPU_BLOCK_*` constants, including high/reserved bits |
| `RASState` | `uint32` | BM extension: `AMDSMI_RAS_ERR_STATE_*` constants |
| `ErrorCount` | `CorrectableCount, UncorrectableCount, DeferredCount uint64` | Error counts; native accumulator outputs start at zero |
| `RasFeatureInfo` | `RasEepromVersion, EccCorrectionSchemaFlag uint32` | EEPROM version and correction schema, no health/reboot fields |

`AMDSMI_MAX_NUM_NUMA_NODES` is exported from the native C capacity. Counts are
validated before array access; bounds violations are errors, not truncation.
Unknown `NumPartitions == UINT32_MAX` with zero resources is preserved without
allocating that count.

The profile getter allocates the full native partition-ID output array, but BM
populates only slot 0. Its successful Go IDs slice therefore always has one current
ID regardless of `NumPartitions`; that ID can remain zero after a subordinate
lookup failure. On error the result is a zero profile, `nil` IDs, and an error.
The all-profile configuration getter is not exposed.

`GetGpuEccEnabled` includes known blocks even when disabled. Additional enabled
unknown/reserved bits are individual keys with value `true`; the map is not a
health assessment. Empty resources/ranges and successful partial totals do not
imply a lack of resources or errors. Secondary partitions retain native
supported/unsupported results; no per-XCP metrics or write-producing fallback is added.

## Legacy GPU functions

```{eval-rst}
.. go-api-ref:: ../../goamdsmi.go
   :section: gpu
```

## Legacy CPU functions


```{eval-rst}
.. go-api-ref:: ../../goamdsmi.go
   :section: cpu
```
