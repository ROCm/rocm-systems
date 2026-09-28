// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

func GetECCEnabled(h ProcessorHandle) (GPUBlock, error) {
	const op = "amdsmi_get_gpu_ecc_enabled"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (GPUBlock, error) {
		var out C.uint64_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_ecc_enabled(p, &out)); err != nil {
			return 0, err
		}
		return GPUBlock(out), nil
	})
}

func GetECCCount(h ProcessorHandle, block GPUBlock) (ECCCounts, error) {
	const op = "amdsmi_get_gpu_ecc_count"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ECCCounts, error) {
		var out C.amdsmi_error_count_t
		status := C.amdsmi_get_gpu_ecc_count(p, C.amdsmi_gpu_block_t(block), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return ECCCounts{}, err
		}
		return ECCCounts{Correctable: uint64(out.correctable_count),
			Uncorrectable: uint64(out.uncorrectable_count), Deferred: uint64(out.deferred_count)}, nil
	})
}

// GetTotalECCCount preserves native totals, which may omit unavailable blocks.
func GetTotalECCCount(h ProcessorHandle) (ECCCounts, error) {
	const op = "amdsmi_get_gpu_total_ecc_count"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ECCCounts, error) {
		var out C.amdsmi_error_count_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_total_ecc_count(p, &out)); err != nil {
			return ECCCounts{}, err
		}
		return ECCCounts{Correctable: uint64(out.correctable_count),
			Uncorrectable: uint64(out.uncorrectable_count), Deferred: uint64(out.deferred_count)}, nil
	})
}

func GetRASBlockState(h ProcessorHandle, block GPUBlock) (RASState, error) {
	const op = "amdsmi_get_gpu_ras_block_features_enabled"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (RASState, error) {
		var out C.amdsmi_ras_err_state_t
		status := C.amdsmi_get_gpu_ras_block_features_enabled(p, C.amdsmi_gpu_block_t(block), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return RASState(out), nil
	})
}

func GetRASFeatureInfo(h ProcessorHandle) (RASFeatureInfo, error) {
	const op = "amdsmi_get_gpu_ras_feature_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (RASFeatureInfo, error) {
		var out C.amdsmi_ras_feature_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_ras_feature_info(p, &out)); err != nil {
			return RASFeatureInfo{}, err
		}
		return RASFeatureInfo{EEPROMVersion: uint32(out.ras_eeprom_version),
			ECCCorrectionSchema: uint32(out.ecc_correction_schema_flag)}, nil
	})
}
