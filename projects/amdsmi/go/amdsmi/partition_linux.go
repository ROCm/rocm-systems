// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
static uint32_t go_amdsmi_nps_mask(amdsmi_nps_caps_t value) {
	return value.nps_cap_mask;
}
*/
import "C"

func GetKFDInfo(h ProcessorHandle) (KFDInfo, error) {
	const op = "amdsmi_get_gpu_kfd_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (KFDInfo, error) {
		var out C.amdsmi_kfd_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_kfd_info(p, &out)); err != nil {
			return KFDInfo{}, err
		}
		return KFDInfo{KFDID: uint64(out.kfd_id), NodeID: uint32(out.node_id),
			CurrentPartitionID: uint32(out.current_partition_id)}, nil
	})
}

func GetMemoryPartitionConfig(h ProcessorHandle) (MemoryPartitionConfig, error) {
	const op = "amdsmi_get_gpu_memory_partition_config"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (MemoryPartitionConfig, error) {
		var out C.amdsmi_memory_partition_config_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_memory_partition_config(p, &out)); err != nil {
			return MemoryPartitionConfig{}, err
		}
		n, err := checkedCount(op, uint64(out.num_numa_ranges), len(out.numa_range))
		if err != nil {
			return MemoryPartitionConfig{}, err
		}
		result := MemoryPartitionConfig{
			Capabilities: MemoryCapabilities(C.go_amdsmi_nps_mask(out.partition_caps)),
			Mode:         MemoryPartitionType(out.mp_mode),
			NUMARanges:   make([]NUMARange, n),
		}
		for i := range result.NUMARanges {
			value := out.numa_range[i]
			result.NUMARanges[i] = NUMARange{MemoryType: VRAMType(value.memory_type),
				Start: uint64(value.start), End: uint64(value.end)}
		}
		return result, nil
	})
}

func GetAcceleratorPartitionProfile(h ProcessorHandle) (AcceleratorPartitionProfile, error) {
	const op = "amdsmi_get_gpu_accelerator_partition_profile"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (AcceleratorPartitionProfile, error) {
		var out C.amdsmi_accelerator_partition_profile_t
		var partitionID C.uint32_t
		status := C.amdsmi_get_gpu_accelerator_partition_profile(p, &out, &partitionID)
		if err := nativeErrorLocked(op, status); err != nil {
			return AcceleratorPartitionProfile{}, err
		}
		result := AcceleratorPartitionProfile{
			Type:               AcceleratorPartitionType(out.profile_type),
			NumPartitions:      uint32(out.num_partitions),
			MemoryCapabilities: MemoryCapabilities(C.go_amdsmi_nps_mask(out.memory_caps)),
			ProfileIndex:       uint32(out.profile_index),
			NumResources:       uint32(out.num_resources),
			PartitionID:        uint32(partitionID),
		}
		if out.num_resources == 0 {
			return result, nil
		}
		rows, err := checkedCount(op, uint64(out.num_partitions), len(out.resources))
		if err != nil {
			return AcceleratorPartitionProfile{}, err
		}
		columns, err := checkedCount(op, uint64(out.num_resources), len(out.resources[0]))
		if err != nil {
			return AcceleratorPartitionProfile{}, err
		}
		result.Resources = make([][]uint32, rows)
		for i := range result.Resources {
			result.Resources[i] = make([]uint32, columns)
			for j := range result.Resources[i] {
				result.Resources[i][j] = uint32(out.resources[i][j])
			}
		}
		return result, nil
	})
}
