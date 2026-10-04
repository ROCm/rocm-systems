// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi_test

import (
	"fmt"
	"testing"

	"github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi"
)

var (
	_ func(amdsmi.InitFlags) error                                                                  = amdsmi.Init
	_ func() ([]amdsmi.ProcessorHandle, error)                                                      = amdsmi.GetProcessorHandles
	_ amdsmi.InitFlags                                                                              = amdsmi.AMDSMI_INIT_AMD_GPUS
	_ func() error                                                                                  = amdsmi.ShutDown
	_ func() (amdsmi.Version, error)                                                                = amdsmi.GetLibVersion
	_ func(amdsmi.Status) (string, error)                                                           = amdsmi.StatusCodeToString
	_ error                                                                                         = &amdsmi.StatusError{Code: amdsmi.AMDSMI_STATUS_INVAL, Name: "AMDSMI_STATUS_INVAL"}
	_ func(uint32) (amdsmi.ProcessorHandle, error)                                                  = amdsmi.GetProcessorHandleFromIndex
	_ func(amdsmi.ProcessorHandle) (amdsmi.AsicInfo, error)                                         = amdsmi.GetGpuAsicInfo
	_ func(amdsmi.ProcessorHandle) (amdsmi.Bdf, error)                                              = amdsmi.GetGpuDeviceBdf
	_ func(amdsmi.Bdf) (amdsmi.ProcessorHandle, error)                                              = amdsmi.GetProcessorHandleFromBdf
	_ func(amdsmi.ProcessorHandle) (string, error)                                                  = amdsmi.GetGpuDeviceUuid
	_ func(amdsmi.ProcessorHandle) (amdsmi.DriverInfo, error)                                       = amdsmi.GetGpuDriverInfo
	_ func(amdsmi.ProcessorHandle) (amdsmi.BoardInfo, error)                                        = amdsmi.GetGpuBoardInfo
	_ func(amdsmi.ProcessorHandle) (amdsmi.VbiosInfo, error)                                        = amdsmi.GetGpuVbiosInfo
	_ func(amdsmi.Bdf) uint64                                                                       = amdsmi.Bdf.Domain
	_ func(amdsmi.Bdf) uint8                                                                        = amdsmi.Bdf.Bus
	_ func(amdsmi.Bdf) uint8                                                                        = amdsmi.Bdf.Device
	_ func(amdsmi.Bdf) uint8                                                                        = amdsmi.Bdf.Function
	_ fmt.Stringer                                                                                  = amdsmi.Bdf(0)
	_ func(amdsmi.ProcessorHandle) (amdsmi.FwInfo, error)                                           = amdsmi.GetFwInfo
	_ [amdsmi.AMDSMI_FW_ID__MAX]amdsmi.FwInfoList                                                   = amdsmi.FwInfo{}.FwList
	_ uint8                                                                                         = amdsmi.FwInfo{}.NumFwInfo
	_ amdsmi.FwBlock                                                                                = amdsmi.FwInfoList{}.FwID
	_ uint64                                                                                        = amdsmi.FwInfoList{}.FwVersion
	_ func(amdsmi.ProcessorHandle) (amdsmi.PowerInfo, error)                                        = amdsmi.GetPowerInfo
	_ func(amdsmi.ProcessorHandle, uint32) (amdsmi.PowerCapInfo, error)                             = amdsmi.GetPowerCapInfo
	_ func(amdsmi.ProcessorHandle, amdsmi.ClkType) (amdsmi.ClkInfo, error)                          = amdsmi.GetClockInfo
	_ func(amdsmi.ProcessorHandle, amdsmi.TemperatureType, amdsmi.TemperatureMetric) (int64, error) = amdsmi.GetTempMetric
	_ func(amdsmi.ProcessorHandle) (amdsmi.EngineUsage, error)                                      = amdsmi.GetGpuActivity
	_ func(amdsmi.ProcessorHandle) (amdsmi.VramInfo, error)                                         = amdsmi.GetGpuVramInfo
	_ func(amdsmi.ProcessorHandle) (amdsmi.MemoryPartitionConfig, error)                            = amdsmi.GetGpuMemoryPartitionConfig
	_ func(amdsmi.ProcessorHandle) (amdsmi.AcceleratorPartitionProfile, []uint32, error)            = amdsmi.GetGpuAcceleratorPartitionProfile
	_ [amdsmi.AMDSMI_MAX_NUM_NUMA_NODES]amdsmi.NumaRange                                            = amdsmi.MemoryPartitionConfig{}.NumaRanges
	_ func(amdsmi.NpsCaps) []amdsmi.MemoryPartitionType                                             = amdsmi.NpsCaps.Supported
	_ fmt.Stringer                                                                                  = amdsmi.NpsCaps{}
	_ func(amdsmi.ProcessorHandle) (map[amdsmi.GpuBlock]bool, error)                                = amdsmi.GetGpuEccEnabled
	_ func(amdsmi.ProcessorHandle, amdsmi.GpuBlock) (amdsmi.ErrorCount, error)                      = amdsmi.GetGpuEccCount
	_ func(amdsmi.ProcessorHandle) (amdsmi.ErrorCount, error)                                       = amdsmi.GetGpuTotalEccCount
	_ func(amdsmi.ProcessorHandle) (amdsmi.RasFeatureInfo, error)                                   = amdsmi.GetGpuRasFeatureInfo
	_ fmt.Stringer                                                                                  = amdsmi.FwBlock(0)
	_ fmt.Stringer                                                                                  = amdsmi.VramType(0)
	_ fmt.Stringer                                                                                  = amdsmi.MemoryPartitionType(0)
	_ fmt.Stringer                                                                                  = amdsmi.AcceleratorPartitionType(0)
	_ func(*amdsmi.StatusError) string                                                              = (*amdsmi.StatusError).Error
	_ string                                                                                        = amdsmi.StatusError{}.Name
	_ amdsmi.Status                                                                                 = amdsmi.StatusError{}.Code
	_ uint32                                                                                        = amdsmi.Version{}.Major
	_ uint32                                                                                        = amdsmi.Version{}.Minor
	_ uint32                                                                                        = amdsmi.Version{}.Release
	_ string                                                                                        = amdsmi.AsicInfo{}.MarketName
	_ uint32                                                                                        = amdsmi.AsicInfo{}.VendorID
	_ string                                                                                        = amdsmi.AsicInfo{}.VendorName
	_ uint32                                                                                        = amdsmi.AsicInfo{}.SubvendorID
	_ uint64                                                                                        = amdsmi.AsicInfo{}.DeviceID
	_ uint32                                                                                        = amdsmi.AsicInfo{}.RevID
	_ string                                                                                        = amdsmi.AsicInfo{}.AsicSerial
	_ uint32                                                                                        = amdsmi.AsicInfo{}.OamID
	_ uint32                                                                                        = amdsmi.AsicInfo{}.NumComputeUnits
	_ uint64                                                                                        = amdsmi.AsicInfo{}.TargetGraphicsVersion
	_ uint32                                                                                        = amdsmi.AsicInfo{}.SubsystemID
	_ uint64                                                                                        = amdsmi.AsicInfo{}.Flags
	_ uint32                                                                                        = amdsmi.AsicInfo{}.PhysicalAccId
	_ uint32                                                                                        = amdsmi.AsicInfo{}.ChipRevId
	_ uint32                                                                                        = amdsmi.AsicInfo{}.ExternalRevId
	_ string                                                                                        = amdsmi.DriverInfo{}.DriverVersion
	_ string                                                                                        = amdsmi.DriverInfo{}.DriverDate
	_ string                                                                                        = amdsmi.DriverInfo{}.DriverName
	_ string                                                                                        = amdsmi.BoardInfo{}.ModelNumber
	_ string                                                                                        = amdsmi.BoardInfo{}.ProductSerial
	_ string                                                                                        = amdsmi.BoardInfo{}.FruID
	_ string                                                                                        = amdsmi.BoardInfo{}.ProductName
	_ string                                                                                        = amdsmi.BoardInfo{}.ManufacturerName
	_ string                                                                                        = amdsmi.VbiosInfo{}.Name
	_ string                                                                                        = amdsmi.VbiosInfo{}.BuildDate
	_ string                                                                                        = amdsmi.VbiosInfo{}.PartNumber
	_ string                                                                                        = amdsmi.VbiosInfo{}.Version
	_ string                                                                                        = amdsmi.VbiosInfo{}.BootFirmware
	_ amdsmi.VramType                                                                               = amdsmi.VramInfo{}.VramType
	_ string                                                                                        = amdsmi.VramInfo{}.VramVendor
	_ uint64                                                                                        = amdsmi.VramInfo{}.VramSize
	_ uint32                                                                                        = amdsmi.VramInfo{}.VramBitWidth
	_ uint64                                                                                        = amdsmi.VramInfo{}.VramMaxBandwidth
	_ uint64                                                                                        = amdsmi.PowerInfo{}.SocketPower
	_ uint32                                                                                        = amdsmi.PowerInfo{}.CurrentSocketPower
	_ uint32                                                                                        = amdsmi.PowerInfo{}.AverageSocketPower
	_ uint64                                                                                        = amdsmi.PowerInfo{}.GfxVoltage
	_ uint64                                                                                        = amdsmi.PowerInfo{}.SocVoltage
	_ uint64                                                                                        = amdsmi.PowerInfo{}.MemVoltage
	_ uint32                                                                                        = amdsmi.PowerInfo{}.PowerLimit
	_ uint32                                                                                        = amdsmi.PowerInfo{}.UbbPower
	_ uint64                                                                                        = amdsmi.PowerCapInfo{}.PowerCap
	_ uint64                                                                                        = amdsmi.PowerCapInfo{}.DefaultPowerCap
	_ uint64                                                                                        = amdsmi.PowerCapInfo{}.DpmCap
	_ uint64                                                                                        = amdsmi.PowerCapInfo{}.MinPowerCap
	_ uint64                                                                                        = amdsmi.PowerCapInfo{}.MaxPowerCap
	_ uint32                                                                                        = amdsmi.EngineUsage{}.GfxActivity
	_ uint32                                                                                        = amdsmi.EngineUsage{}.UmcActivity
	_ uint32                                                                                        = amdsmi.EngineUsage{}.MmActivity
	_ uint32                                                                                        = amdsmi.ClkInfo{}.Clk
	_ uint32                                                                                        = amdsmi.ClkInfo{}.MinClk
	_ uint32                                                                                        = amdsmi.ClkInfo{}.MaxClk
	_ bool                                                                                          = amdsmi.ClkInfo{}.ClkLocked
	_ bool                                                                                          = amdsmi.ClkInfo{}.ClkDeepSleep
	_ bool                                                                                          = amdsmi.Frequencies{}.HasDeepSleep
	_ uint32                                                                                        = amdsmi.Frequencies{}.NumSupported
	_ uint32                                                                                        = amdsmi.Frequencies{}.Current
	_ []uint64                                                                                      = amdsmi.Frequencies{}.Values
	_ bool                                                                                          = amdsmi.NpsCaps{}.Nps1Cap
	_ bool                                                                                          = amdsmi.NpsCaps{}.Nps2Cap
	_ bool                                                                                          = amdsmi.NpsCaps{}.Nps4Cap
	_ bool                                                                                          = amdsmi.NpsCaps{}.Nps8Cap
	_ amdsmi.VramType                                                                               = amdsmi.NumaRange{}.MemoryType
	_ uint64                                                                                        = amdsmi.NumaRange{}.Start
	_ uint64                                                                                        = amdsmi.NumaRange{}.End
	_ amdsmi.NpsCaps                                                                                = amdsmi.MemoryPartitionConfig{}.PartitionCaps
	_ amdsmi.MemoryPartitionType                                                                    = amdsmi.MemoryPartitionConfig{}.Mode
	_ uint32                                                                                        = amdsmi.MemoryPartitionConfig{}.NumNumaRanges
	_ amdsmi.AcceleratorPartitionType                                                               = amdsmi.AcceleratorPartitionProfile{}.ProfileType
	_ uint32                                                                                        = amdsmi.AcceleratorPartitionProfile{}.NumPartitions
	_ amdsmi.NpsCaps                                                                                = amdsmi.AcceleratorPartitionProfile{}.MemoryCaps
	_ uint32                                                                                        = amdsmi.AcceleratorPartitionProfile{}.ProfileIndex
	_ uint32                                                                                        = amdsmi.AcceleratorPartitionProfile{}.NumResources
	_ [][]uint32                                                                                    = amdsmi.AcceleratorPartitionProfile{}.Resources
	_ uint64                                                                                        = amdsmi.ErrorCount{}.CorrectableCount
	_ uint64                                                                                        = amdsmi.ErrorCount{}.UncorrectableCount
	_ uint64                                                                                        = amdsmi.ErrorCount{}.DeferredCount
	_ uint32                                                                                        = amdsmi.RasFeatureInfo{}.RasEepromVersion
	_ uint32                                                                                        = amdsmi.RasFeatureInfo{}.EccCorrectionSchemaFlag
)

func underlyingInt32[T ~int32]()   {}
func underlyingUint32[T ~uint32]() {}
func underlyingUint64[T ~uint64]() {}

func TestContractUnderlyingTypes(t *testing.T) {
	underlyingInt32[amdsmi.FwBlock]()
	underlyingInt32[amdsmi.ClkType]()
	underlyingInt32[amdsmi.TemperatureType]()
	underlyingInt32[amdsmi.TemperatureMetric]()
	underlyingInt32[amdsmi.VramType]()
	underlyingInt32[amdsmi.MemoryPartitionType]()
	underlyingInt32[amdsmi.AcceleratorPartitionType]()
	underlyingUint32[amdsmi.Status]()
	underlyingUint64[amdsmi.InitFlags]()
	underlyingUint64[amdsmi.Bdf]()
	underlyingUint64[amdsmi.GpuBlock]()
}

func TestContractEnumStrings(t *testing.T) {
	for _, test := range []struct {
		value fmt.Stringer
		want  string
	}{
		{amdsmi.AMDSMI_FW_ID_SMU, "FW_ID_SMU"},
		{amdsmi.AMDSMI_FW_ID_PLDM_BUNDLE, "FW_ID_PLDM_BUNDLE"},
		{amdsmi.AMDSMI_FW_ID__MAX, "FW_ID__MAX"},
		{amdsmi.FwBlock(-1), "UNKNOWN(-1)"},
		{amdsmi.AMDSMI_VRAM_TYPE_UNKNOWN, "UNKNOWN"},
		{amdsmi.AMDSMI_VRAM_TYPE_HBM3E, "HBM3E"},
		{amdsmi.AMDSMI_VRAM_TYPE_LPDDR5, "LPDDR5"},
		{amdsmi.VramType(-1), "UNKNOWN(-1)"},
		{amdsmi.AMDSMI_MEMORY_PARTITION_NPS8, "NPS8"},
		{amdsmi.MemoryPartitionType(-1), "UNKNOWN(-1)"},
		{amdsmi.AMDSMI_ACCELERATOR_PARTITION_INVALID, "INVALID"},
		{amdsmi.AMDSMI_ACCELERATOR_PARTITION_CPX, "CPX"},
		{amdsmi.AcceleratorPartitionType(-1), "UNKNOWN(-1)"},
	} {
		if got := test.value.String(); got != test.want {
			t.Errorf("%T: got %q, want %q", test.value, got, test.want)
		}
	}
}

func ExampleGetGpuAsicInfo() {
	if err := amdsmi.Init(amdsmi.AMDSMI_INIT_AMD_GPUS); err != nil {
		panic(err)
	}
	defer amdsmi.ShutDown()
	handle, err := amdsmi.GetProcessorHandleFromIndex(0)
	if err != nil {
		panic(err)
	}
	info, err := amdsmi.GetGpuAsicInfo(handle)
	if err != nil {
		panic(err)
	}
	bdf, err := amdsmi.GetGpuDeviceBdf(handle)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: revision %d, serial %s, OAM %d, compute units %d\n",
		bdf.String(), info.RevID, info.AsicSerial, info.OamID, info.NumComputeUnits)
}
