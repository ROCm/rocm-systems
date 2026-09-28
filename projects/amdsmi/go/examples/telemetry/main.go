// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package main

import (
	"errors"
	"fmt"
	"os"

	"github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi"
)

func run() (err error) {
	if err = amdsmi.Init(amdsmi.InitAMDGPUs); err != nil {
		return err
	}
	defer func() { err = errors.Join(err, amdsmi.ShutDown()) }()
	version, versionErr := amdsmi.GetLibraryVersion()
	if versionErr != nil {
		return versionErr
	}
	fmt.Printf("AMD SMI %d.%d.%d (%s)\n", version.Major, version.Minor, version.Release, version.Build)
	handles, err := amdsmi.GetProcessorHandles()
	if err != nil {
		return err
	}
	if len(handles) == 0 {
		fmt.Println("No AMD GPUs visible.")
	}
	for index, handle := range handles {
		label := fmt.Sprintf("GPU %d", index)
		if bdf, queryErr := amdsmi.GetGPUDeviceBDF(handle); queryErr != nil {
			fmt.Fprintf(os.Stderr, "%s BDF: %v\n", label, queryErr)
		} else {
			label = bdf.String()
		}
		if temperature, queryErr := amdsmi.GetTemperature(handle,
			amdsmi.AMDSMI_TEMPERATURE_TYPE_HOTSPOT, amdsmi.AMDSMI_TEMP_CURRENT); queryErr != nil {
			fmt.Fprintf(os.Stderr, "%s temperature: %v\n", label, queryErr)
		} else {
			fmt.Printf("%s hotspot: %d C\n", label, temperature)
		}
		if power, queryErr := amdsmi.GetPowerInfo(handle); queryErr != nil {
			fmt.Fprintf(os.Stderr, "%s power: %v\n", label, queryErr)
		} else if power.SocketPowerWatts == ^uint64(0) {
			fmt.Printf("%s socket power: N/A\n", label)
		} else {
			fmt.Printf("%s socket power: %d W\n", label, power.SocketPowerWatts)
		}
		if used, queryErr := amdsmi.GetMemoryUsage(handle, amdsmi.AMDSMI_MEM_TYPE_VRAM); queryErr != nil {
			fmt.Fprintf(os.Stderr, "%s memory: %v\n", label, queryErr)
		} else {
			fmt.Printf("%s VRAM used: %d bytes\n", label, used)
		}
	}
	return nil
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
