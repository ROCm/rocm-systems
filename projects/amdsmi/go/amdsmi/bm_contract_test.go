// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi_test

import "github.com/ROCm/rocm-systems/projects/amdsmi/go/amdsmi"

var (
	_ func(amdsmi.ProcessorHandle, amdsmi.ClkType) (amdsmi.Frequencies, error) = amdsmi.GetClockFrequencies
	_ func(amdsmi.ProcessorHandle) (amdsmi.KFDInfo, error)                     = amdsmi.GetKFDInfo
	_ func(amdsmi.ProcessorHandle, amdsmi.MemoryType) (uint64, error)          = amdsmi.GetMemoryTotal
	_ func(amdsmi.ProcessorHandle, amdsmi.MemoryType) (uint64, error)          = amdsmi.GetMemoryUsage
	_ func(amdsmi.ProcessorHandle, amdsmi.GpuBlock) (amdsmi.RASState, error)   = amdsmi.GetRASBlockState
	_ func(*amdsmi.StatusError) error                                          = (*amdsmi.StatusError).Unwrap
	_ func(amdsmi.Status) string                                               = amdsmi.Status.Error
	_ string                                                                   = amdsmi.StatusError{}.Op
	_ string                                                                   = amdsmi.StatusError{}.Message
	_ string                                                                   = amdsmi.Version{}.Build
	_ uint8                                                                    = amdsmi.ClkInfo{}.ClkLockedRaw
	_ uint8                                                                    = amdsmi.ClkInfo{}.ClkDeepSleepRaw
	_ uint32                                                                   = amdsmi.NpsCaps{}.RawMask
)
