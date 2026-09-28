// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi_test

import (
	"fmt"

	"github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi"
)

var (
	_ func(amdsmi.InitFlags) error                          = amdsmi.Init
	_ func(uint32) (amdsmi.ProcessorHandle, error)          = amdsmi.GetProcessorHandleFromIndex
	_ func(amdsmi.ProcessorHandle) (amdsmi.AsicInfo, error) = amdsmi.GetGPUAsicInfo
	_ func(amdsmi.ProcessorHandle) (amdsmi.BDF, error)      = amdsmi.GetGPUDeviceBDF
)

func ExampleGetGPUAsicInfo() {
	if err := amdsmi.Init(amdsmi.InitAMDGPUs); err != nil {
		panic(err)
	}
	defer amdsmi.ShutDown()
	handle, err := amdsmi.GetProcessorHandleFromIndex(0)
	if err != nil {
		panic(err)
	}
	info, err := amdsmi.GetGPUAsicInfo(handle)
	if err != nil {
		panic(err)
	}
	bdf, err := amdsmi.GetGPUDeviceBDF(handle)
	if err != nil {
		panic(err)
	}
	fmt.Printf("%s: revision %d, serial %s, OAM %d, compute units %d\n",
		bdf.String(), info.RevID, info.AsicSerial, info.OamID, info.NumComputeUnits)
}
