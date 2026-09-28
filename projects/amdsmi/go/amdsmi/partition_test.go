// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import "testing"

func TestPartitionKFD(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_kfd_info", GetKFDInfo,
		KFDInfo{KFDID: ^uint64(0), NodeID: ^uint32(0), CurrentPartitionID: 7})
}

func TestPartitionMemory(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_memory_partition_config", GetMemoryPartitionConfig,
		MemoryPartitionConfig{Capabilities: 5, Mode: AMDSMI_MEMORY_PARTITION_NPS4,
			NUMARanges: []NUMARange{}})
}

func TestPartitionCurrent(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_accelerator_partition_profile", GetAcceleratorPartitionProfile,
		AcceleratorPartitionProfile{Type: AMDSMI_ACCELERATOR_PARTITION_CPX, NumPartitions: 8,
			MemoryCapabilities: 5, ProfileIndex: 3, NumResources: 0, PartitionID: 7})
}

func TestPartitionIDArray(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_SUCCESS, 5)
	got, err := GetAcceleratorPartitionProfile(h)
	if err != nil || got.PartitionID != 7 {
		t.Fatalf("current partition ID: %+v, %v", got, err)
	}
}

func TestPartitionMemoryRanges(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetMemoryPartitionConfig(h)
	if err != nil || len(got.NUMARanges) != maxNUMARanges {
		t.Fatalf("ranges: %+v, %v", got, err)
	}
	for i, value := range got.NUMARanges {
		want := NUMARange{MemoryType: AMDSMI_VRAM_TYPE_HBM3,
			Start: 1<<40 + uint64(i), End: ^uint64(0) - uint64(i)}
		if value != want {
			t.Fatalf("range %d: got %+v, want %+v", i, value, want)
		}
	}
	mockConfigure("amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_SUCCESS, 0)
	if _, err := GetMemoryPartitionConfig(h); err != nil {
		t.Fatal(err)
	}
	if got.NUMARanges[0].End != ^uint64(0) {
		t.Fatal("later call changed the returned ranges")
	}
}

func TestPartitionResources(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetAcceleratorPartitionProfile(h)
	if err != nil || len(got.Resources) != maxAcceleratorPartitions {
		t.Fatalf("resources: %+v, %v", got, err)
	}
	for i, row := range got.Resources {
		if len(row) != maxProfileResources {
			t.Fatalf("row %d length: %d", i, len(row))
		}
		for j, value := range row {
			want := ^uint32(0) - uint32(i*maxProfileResources+j)
			if value != want {
				t.Fatalf("resource %d,%d: got %d, want %d", i, j, value, want)
			}
		}
	}
	got.Resources[0][0] = 0
	again, err := GetAcceleratorPartitionProfile(h)
	if err != nil || again.Resources[0][0] != ^uint32(0) {
		t.Fatalf("resource ownership: %+v, %v", again, err)
	}
}

func TestPartitionMemoryBounds(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_SUCCESS, 2)
	got, err := GetMemoryPartitionConfig(h)
	assertNativeError(t, err, "amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_UNEXPECTED_SIZE)
	assertZero(t, got)
}

func TestPartitionResourceBounds(t *testing.T) {
	h := fixtureHandle(t)
	for _, mode := range []uint32{2, 3} {
		mockConfigure("amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_SUCCESS, mode)
		got, err := GetAcceleratorPartitionProfile(h)
		assertNativeError(t, err, "amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_UNEXPECTED_SIZE)
		assertZero(t, got)
	}
}

func TestPartitionUnknownCount(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_SUCCESS, 4)
	got, err := GetAcceleratorPartitionProfile(h)
	if err != nil || got.NumPartitions != ^uint32(0) || got.ProfileIndex != ^uint32(0) || len(got.Resources) != 0 {
		t.Fatalf("unknown partition metadata: %+v, %v", got, err)
	}
}
