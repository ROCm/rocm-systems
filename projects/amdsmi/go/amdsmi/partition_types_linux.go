// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

type KFDInfo struct {
	KFDID              uint64
	NodeID             uint32
	CurrentPartitionID uint32
}

type MemoryPartitionType uint32
type AcceleratorPartitionType uint32

// MemoryCapabilities preserves the native NPS capability mask, not NPS enum values.
type MemoryCapabilities uint32

type NUMARange struct {
	MemoryType VRAMType
	Start      uint64
	End        uint64
}

type MemoryPartitionConfig struct {
	Capabilities MemoryCapabilities
	Mode         MemoryPartitionType
	// Empty ranges can mean unpopulated native metadata.
	NUMARanges []NUMARange
}

type AcceleratorPartitionProfile struct {
	Type               AcceleratorPartitionType
	NumPartitions      uint32
	MemoryCapabilities MemoryCapabilities
	ProfileIndex       uint32
	NumResources       uint32
	// Current native profiles leave resource metadata empty.
	Resources   [][]uint32
	PartitionID uint32
}

const (
	maxNUMARanges            int = C.AMDSMI_MAX_NUM_NUMA_NODES
	maxAcceleratorPartitions int = C.AMDSMI_MAX_ACCELERATOR_PARTITIONS
	maxProfileResources      int = C.AMDSMI_MAX_CP_PROFILE_RESOURCES
)

const (
	AMDSMI_MEMORY_PARTITION_UNKNOWN MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_UNKNOWN
	AMDSMI_MEMORY_PARTITION_NPS1    MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_NPS1
	AMDSMI_MEMORY_PARTITION_NPS2    MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_NPS2
	AMDSMI_MEMORY_PARTITION_NPS4    MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_NPS4
	AMDSMI_MEMORY_PARTITION_NPS8    MemoryPartitionType = C.AMDSMI_MEMORY_PARTITION_NPS8
)

const (
	AMDSMI_ACCELERATOR_PARTITION_INVALID AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_INVALID
	AMDSMI_ACCELERATOR_PARTITION_SPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_SPX
	AMDSMI_ACCELERATOR_PARTITION_DPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_DPX
	AMDSMI_ACCELERATOR_PARTITION_TPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_TPX
	AMDSMI_ACCELERATOR_PARTITION_QPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_QPX
	AMDSMI_ACCELERATOR_PARTITION_CPX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_CPX
	AMDSMI_ACCELERATOR_PARTITION_MAX     AcceleratorPartitionType = C.AMDSMI_ACCELERATOR_PARTITION_MAX
)
