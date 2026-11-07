// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import "testing"

func TestECCEnabled(t *testing.T) {
	want := make(map[GpuBlock]bool)
	for block := AMDSMI_GPU_BLOCK_FIRST; block <= AMDSMI_GPU_BLOCK_LAST; block <<= 1 {
		want[block] = block == AMDSMI_GPU_BLOCK_UMC || block == AMDSMI_GPU_BLOCK_UCIE_PCS
	}
	want[AMDSMI_GPU_BLOCK_RESERVED] = true
	want[GpuBlock(1)<<62] = true
	checkQuery(t, "amdsmi_get_gpu_ecc_enabled", GetGpuEccEnabled, want)
}

func TestECCEnabledEmpty(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_ecc_enabled", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetGpuEccEnabled(h)
	if err != nil || len(got) == 0 {
		t.Fatalf("empty mask: %v, %v", got, err)
	}
	for block, enabled := range got {
		if enabled || block > AMDSMI_GPU_BLOCK_LAST || block < AMDSMI_GPU_BLOCK_FIRST {
			t.Fatalf("unexpected entry %x: %v", block, enabled)
		}
	}
}

func TestECCEnabledOwnedCopy(t *testing.T) {
	h := fixtureHandle(t)
	first, err := GetGpuEccEnabled(h)
	if err != nil {
		t.Fatal(err)
	}
	first[AMDSMI_GPU_BLOCK_UMC] = false
	second, err := GetGpuEccEnabled(h)
	if err != nil || !second[AMDSMI_GPU_BLOCK_UMC] || first[AMDSMI_GPU_BLOCK_UMC] {
		t.Fatalf("map ownership: %v, %v", second, err)
	}
}

func TestECCCount(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_ecc_count", func(h ProcessorHandle) (ErrorCount, error) {
		return GetGpuEccCount(h, AMDSMI_GPU_BLOCK_UCIE_PCS)
	}, ErrorCount{CorrectableCount: 1 << 40, UncorrectableCount: 1 << 41, DeferredCount: ^uint64(0)})
}

func TestECCTotal(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_total_ecc_count", GetGpuTotalEccCount,
		ErrorCount{CorrectableCount: 1 << 40, UncorrectableCount: 1 << 41, DeferredCount: 7})
}

func TestRASBlock(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_ras_block_features_enabled", func(h ProcessorHandle) (RASState, error) {
		return GetRASBlockState(h, AMDSMI_GPU_BLOCK_UCIE_PCS)
	}, AMDSMI_RAS_ERR_STATE_ENABLED)
}

func TestRASFeature(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_ras_feature_info", GetGpuRasFeatureInfo,
		RasFeatureInfo{RasEepromVersion: 0x102, EccCorrectionSchemaFlag: 0xf})
}
