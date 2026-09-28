// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import "testing"

func TestECCEnabled(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_ecc_enabled", GetECCEnabled,
		AMDSMI_GPU_BLOCK_UMC|AMDSMI_GPU_BLOCK_UCIE_PCS|AMDSMI_GPU_BLOCK_RESERVED)
}

func TestECCCount(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_ecc_count", func(h ProcessorHandle) (ECCCounts, error) {
		return GetECCCount(h, AMDSMI_GPU_BLOCK_UCIE_PCS)
	}, ECCCounts{Correctable: 1 << 40, Uncorrectable: 1 << 41, Deferred: ^uint64(0)})
}

func TestECCTotal(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_total_ecc_count", GetTotalECCCount,
		ECCCounts{Correctable: 1 << 40, Uncorrectable: 1 << 41, Deferred: 7})
}

func TestRASBlock(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_ras_block_features_enabled", func(h ProcessorHandle) (RASState, error) {
		return GetRASBlockState(h, AMDSMI_GPU_BLOCK_UCIE_PCS)
	}, AMDSMI_RAS_ERR_STATE_ENABLED)
}

func TestRASFeature(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_ras_feature_info", GetRASFeatureInfo,
		RASFeatureInfo{EEPROMVersion: 0x102, ECCCorrectionSchema: 0xf})
}
