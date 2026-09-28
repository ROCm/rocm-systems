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
and [standalone module guide](https://github.com/ROCm/rocm-systems/blob/develop/projects/amdsmi/go/README.md). CPU, NIC, set/reset,
all-profile configuration, and event APIs are not included.

All results use Go-owned scalars, strings, and slices. No public field exposes a
C type or borrowed buffer. Native errors return the zero Go result plus an error;
successful fields retain their native units and unavailable markers. Constants
keep the public C enumerator names and values, including aliases. Unknown
numeric values are preserved, and unknown input enums are forwarded to the native API.

### Lifecycle, discovery, and errors

| Signature | Contract |
| --- | --- |
| `Init() error` | Acquires one AMD-GPU initialization reference |
| `ShutDown() error` | Releases one reference; final package shutdown expires all handles, even on cleanup error |
| `GetProcessorHandles() ([]ProcessorHandle, error)` | Enumerates AMD GPUs across sockets, preserving native order without a fixed device cap |
| `GetProcessorHandleFromBDF(BDF) (ProcessorHandle, error)` | Native BDF lookup; rejects values outside the native bit widths |
| `GetBDF(ProcessorHandle) (BDF, error)` | Reads PCI domain, bus, device, and function |
| `(BDF).String() string` | Hexadecimal `domain:bus:device.function`, retaining the full domain |
| `GetLibraryVersion() (Version, error)` | Native library version; does not require `Init()` |
| `StatusString(StatusCode) string` | Native message or numeric fallback; does not require `Init()` |
| `(StatusCode).Error() string` | Numeric status text |
| `(*Error).Error() string` | Operation, status, and message |
| `(*Error).Unwrap() error` | Underlying `StatusCode` for `errors.Is` |

| Type | Fields / underlying type | Meaning |
| --- | --- | --- |
| `ProcessorHandle` | Opaque struct, no exported fields | Package-lifetime handle; rediscover after final shutdown |
| `BDF` | `Domain uint64`; `Bus, Device, Function uint8` | Domain 48 bits, bus 8, device 5, function 3 |
| `Version` | `Major, Minor, Release uint32`; `Build string` | Native version components and build string |
| `StatusCode` | `uint32` | All `AMDSMI_STATUS_*` values preserved |
| `Error` | `Op string`; `Code StatusCode`; `Message string` | Inspect with `errors.As` into `*Error` |

Every successful `Init()` must be balanced. Calls are serialized, including
status/version lookup. External bindings must coordinate the entire native
lifetime and first-initializer flags; later initialization does not change those
flags. No external concurrent lifecycle, driver reload, partition change, or
hotplug recovery is supported while initialized.

Queries without a package reference return `AMDSMI_STATUS_NOT_INIT`; with a
reference, zero or stale handles return `AMDSMI_STATUS_INVAL`. Status-message
lookup failure never replaces the original numeric status. Successful data is
not a health assessment.

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

### Identity and firmware

| Signature | Native entry |
| --- | --- |
| `GetUUID(ProcessorHandle) (string, error)` | `amdsmi_get_gpu_device_uuid` |
| `GetASICInfo(ProcessorHandle) (ASICInfo, error)` | `amdsmi_get_gpu_asic_info` |
| `GetDriverInfo(ProcessorHandle) (DriverInfo, error)` | `amdsmi_get_gpu_driver_info` |
| `GetBoardInfo(ProcessorHandle) (BoardInfo, error)` | `amdsmi_get_gpu_board_info` |
| `GetFirmwareInfo(ProcessorHandle) ([]FirmwareInfo, error)` | `amdsmi_get_fw_info` |
| `GetVBIOSInfo(ProcessorHandle) (VBIOSInfo, error)` | `amdsmi_get_gpu_vbios_info` |

| Type | Fields and Go types | Meaning |
| --- | --- | --- |
| `ASICInfo` | `MarketName, VendorName, Serial string` | Copied identity strings |
| `ASICInfo` | `VendorID, SubvendorID, RevisionID, OAMID, ComputeUnits, SubsystemID uint32` | Native IDs and compute-unit count |
| `ASICInfo` | `DeviceID, TargetGraphicsVersion, Flags uint64` | Native ID, graphics target, and raw flags |
| `ASICInfo` | `PhysicalAcceleratorID, ChipRevisionID, ExternalRevisionID uint32` | Native physical/revision IDs; `UINT32_MAX` can mean unavailable |
| `DriverInfo` | `Version, Date, Name string` | Native driver metadata |
| `BoardInfo` | `ModelNumber, ProductSerial, FRUID, ProductName, ManufacturerName string` | Native board metadata |
| `FirmwareBlock` | `uint32` | `AMDSMI_FW_ID_*` constants |
| `FirmwareInfo` | `ID FirmwareBlock`; `Version uint64` | Firmware identifier and raw version |
| `VBIOSInfo` | `Name, BuildDate, PartNumber, Version, BootFirmware string` | Native VBIOS/boot metadata |

Fixed strings are bounded by their C capacity. Firmware counts are checked before
copying. Reserved fields are omitted.

### Telemetry

| Signature | Native entry | Units / rules |
| --- | --- | --- |
| `GetTemperature(ProcessorHandle, TemperatureType, TemperatureMetric) (int64, error)` | `amdsmi_get_temp_metric` | Whole degrees C, including negatives, not millidegrees |
| `GetPowerInfo(ProcessorHandle) (PowerInfo, error)` | `amdsmi_get_power_info` | W, mV, uW by member |
| `GetPowerCapInfo(ProcessorHandle, uint32) (PowerCapInfo, error)` | `amdsmi_get_power_cap_info` | Second parameter is the sensor index; partial zeros preserved |
| `GetClockInfo(ProcessorHandle, ClockType) (ClockInfo, error)` | `amdsmi_get_clock_info` | MHz and raw native bytes |
| `GetClockFrequencies(ProcessorHandle, ClockType) (Frequencies, error)` | `amdsmi_get_clk_freq` | Hz; caller must range-check `CurrentIndex` |
| `GetActivity(ProcessorHandle) (Activity, error)` | `amdsmi_get_gpu_activity` | Percentages, including native unavailable values |
| `GetMemoryTotal(ProcessorHandle, MemoryType) (uint64, error)` | `amdsmi_get_gpu_memory_total` | Bytes |
| `GetMemoryUsage(ProcessorHandle, MemoryType) (uint64, error)` | `amdsmi_get_gpu_memory_usage` | Bytes |
| `GetVRAMInfo(ProcessorHandle) (VRAMInfo, error)` | `amdsmi_get_gpu_vram_info` | MB, bits, GB/s by member |

| Enum type | Underlying type | Constant family |
| --- | --- | --- |
| `TemperatureType` | `uint32` | `AMDSMI_TEMPERATURE_TYPE_*`, including GPU-board/baseboard sensors |
| `TemperatureMetric` | `uint32` | `AMDSMI_TEMP_*` |
| `ClockType` | `uint32` | `AMDSMI_CLK_TYPE_*` |
| `MemoryType` | `uint32` | `AMDSMI_MEM_TYPE_*` |
| `VRAMType` | `uint32` | `AMDSMI_VRAM_TYPE_*` |

| Type | Fields and Go types | Units / limitations |
| --- | --- | --- |
| `PowerInfo` | `SocketPowerWatts uint64` | W |
| `PowerInfo` | `CurrentSocketPowerWatts, AverageSocketPowerWatts, UBBPowerWatts uint32` | W |
| `PowerInfo` | `GFXVoltageMillivolts, SOCVoltageMillivolts, MemoryVoltageMillivolts uint64` | mV |
| `PowerInfo` | `PowerLimitMicrowatts uint32` | uW |
| `PowerCapInfo` | `PowerCapMicrowatts, DefaultPowerCapMicrowatts, MinPowerCapMicrowatts, MaxPowerCapMicrowatts uint64` | uW; auxiliary zeros after partial success have no validity flag |
| `PowerCapInfo` | `DPMLevel uint64` | Level index, not MHz |
| `ClockInfo` | `ClockMHz, MinClockMHz, MaxClockMHz uint32` | MHz; preserve `UINT32_MAX` for unavailable values |
| `ClockInfo` | `LockedRaw uint8` | Currently unpopulated, not a measured Boolean |
| `ClockInfo` | `DeepSleepRaw uint8` | Narrowed native sleep-frequency value, not a Boolean |
| `Frequencies` | `HasDeepSleep bool`; `CurrentIndex uint32`; `Hertz []uint64` | Frequencies in Hz; index can be `UINT32_MAX` or outside the slice |
| `Activity` | `GFXPercent, UMCPercent, MMPercent uint32` | Percent; unavailable data can be 65535, including in `GFXPercent` |
| `VRAMInfo` | `Type VRAMType`; `Vendor string` | Memory type and vendor |
| `VRAMInfo` | `SizeMB uint64`; `BitWidth uint32`; `MaxBandwidthGBPerSecond uint64` | MB, bits, GB/s respectively |

Power unavailable markers retain each member's width. There is no global
conversion of unavailable data into zero, `nil`, or another unit.

### Current partitions, ECC, and RAS

| Signature | Native entry | Rules |
| --- | --- | --- |
| `GetKFDInfo(ProcessorHandle) (KFDInfo, error)` | `amdsmi_get_gpu_kfd_info` | Raw KFD/node/current-partition IDs |
| `GetMemoryPartitionConfig(ProcessorHandle) (MemoryPartitionConfig, error)` | `amdsmi_get_gpu_memory_partition_config` | Current mode and capability mask |
| `GetAcceleratorPartitionProfile(ProcessorHandle) (AcceleratorPartitionProfile, error)` | `amdsmi_get_gpu_accelerator_partition_profile` | Current profile only, never the all-profile configuration getter |
| `GetECCEnabled(ProcessorHandle) (GPUBlock, error)` | `amdsmi_get_gpu_ecc_enabled` | Raw 64-bit enabled-block mask |
| `GetECCCount(ProcessorHandle, GPUBlock) (ECCCounts, error)` | `amdsmi_get_gpu_ecc_count` | Per-block counters |
| `GetTotalECCCount(ProcessorHandle) (ECCCounts, error)` | `amdsmi_get_gpu_total_ecc_count` | Native totals can omit unavailable blocks while succeeding |
| `GetRASBlockState(ProcessorHandle, GPUBlock) (RASState, error)` | `amdsmi_get_gpu_ras_block_features_enabled` | State enum, not a Boolean |
| `GetRASFeatureInfo(ProcessorHandle) (RASFeatureInfo, error)` | `amdsmi_get_gpu_ras_feature_info` | Populated metadata only, not a health verdict |

| Type | Fields / underlying type | Meaning / limitations |
| --- | --- | --- |
| `KFDInfo` | `KFDID uint64`; `NodeID, CurrentPartitionID uint32` | Native IDs; unavailable values preserved |
| `MemoryPartitionType` | `uint32` | `AMDSMI_MEMORY_PARTITION_*` constants |
| `AcceleratorPartitionType` | `uint32` | `AMDSMI_ACCELERATOR_PARTITION_*` constants |
| `MemoryCapabilities` | `uint32` | Raw NPS capability bitmask, not NPS enum values |
| `NUMARange` | `MemoryType VRAMType`; `Start, End uint64` | Native memory-address range, copied without conversion |
| `MemoryPartitionConfig` | `Capabilities MemoryCapabilities`; `Mode MemoryPartitionType`; `NUMARanges []NUMARange` | Empty ranges may be unpopulated metadata, not absence of memory |
| `AcceleratorPartitionProfile` | `Type AcceleratorPartitionType`; `MemoryCapabilities MemoryCapabilities` | Current profile type and raw memory capability mask |
| `AcceleratorPartitionProfile` | `NumPartitions, ProfileIndex, NumResources, PartitionID uint32` | Native counts/indices; partition count/index can be `UINT32_MAX`; current ID can stay zero on subordinate lookup failure |
| `AcceleratorPartitionProfile` | `Resources [][]uint32` | Per-partition resource indices; current native metadata is empty |
| `GPUBlock` | `uint64` | `AMDSMI_GPU_BLOCK_*` constants, including high/reserved bits |
| `RASState` | `uint32` | `AMDSMI_RAS_ERR_STATE_*` constants |
| `ECCCounts` | `Correctable, Uncorrectable, Deferred uint64` | Error counts; native accumulator outputs start at zero |
| `RASFeatureInfo` | `EEPROMVersion, ECCCorrectionSchema uint32` | EEPROM version and correction schema, no health/reboot fields |

Counts are checked before copying fixed arrays. Empty resources/NUMA ranges and
successful partial totals do not imply a lack of resources or errors. Secondary
partitions retain native supported/unsupported results; no per-XCP metrics or
write-producing fallback is added.

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
