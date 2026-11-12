// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import (
	"reflect"
	"testing"
)

func TestPartitionCaps(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_SUCCESS, 3)
	got, err := GetGpuMemoryPartitionConfig(h)
	if err != nil || got.PartitionCaps != (NpsCaps{Nps1Cap: true, Nps4Cap: true, RawMask: 0x80000005}) {
		t.Fatalf("capabilities: %+v, %v", got, err)
	}
	if !reflect.DeepEqual(got.PartitionCaps.Supported(), []MemoryPartitionType{AMDSMI_MEMORY_PARTITION_NPS1, AMDSMI_MEMORY_PARTITION_NPS4}) || got.PartitionCaps.String() != "[NPS1 NPS4]" {
		t.Fatalf("capability methods: %v", got.PartitionCaps)
	}
	if (NpsCaps{}).String() != "[]" {
		t.Fatal("empty capabilities")
	}
}

func profileOnly(h ProcessorHandle) (AcceleratorPartitionProfile, error) {
	profile, _, err := GetGpuAcceleratorPartitionProfile(h)
	return profile, err
}

func TestPartitionKFD(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_kfd_info", GetKFDInfo,
		KFDInfo{KFDID: ^uint64(0), NodeID: ^uint32(0), CurrentPartitionID: 7})
}

func TestPartitionMemory(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_memory_partition_config", GetGpuMemoryPartitionConfig,
		MemoryPartitionConfig{PartitionCaps: NpsCaps{Nps1Cap: true, Nps4Cap: true, RawMask: 5}, Mode: AMDSMI_MEMORY_PARTITION_NPS4})
}

func TestPartitionCurrent(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_accelerator_partition_profile", profileOnly,
		AcceleratorPartitionProfile{ProfileType: AMDSMI_ACCELERATOR_PARTITION_CPX, NumPartitions: 8,
			MemoryCaps: NpsCaps{Nps1Cap: true, Nps4Cap: true, RawMask: 5}, ProfileIndex: 3, NumResources: 0})
}

func TestPartitionIDArray(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_SUCCESS, 5)
	got, ids, err := GetGpuAcceleratorPartitionProfile(h)
	if err != nil || !reflect.DeepEqual(ids, []uint32{7}) {
		t.Fatalf("current partition ID: %+v, %v, %v", got, ids, err)
	}
	ids[0] = 99
	_, again, err := GetGpuAcceleratorPartitionProfile(h)
	if err != nil || !reflect.DeepEqual(again, []uint32{7}) || ids[0] != 99 {
		t.Fatalf("ID ownership: %v, %v", again, err)
	}
}

func TestPartitionMemoryRanges(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetGpuMemoryPartitionConfig(h)
	if err != nil || got.NumNumaRanges != uint32(maxNUMARanges) {
		t.Fatalf("ranges: %+v, %v", got, err)
	}
	for i, value := range got.NumaRanges {
		want := NumaRange{MemoryType: AMDSMI_VRAM_TYPE_HBM3,
			Start: 1<<40 + uint64(i), End: ^uint64(0) - uint64(i)}
		if value != want {
			t.Fatalf("range %d: got %+v, want %+v", i, value, want)
		}
	}
	mockConfigure("amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_SUCCESS, 0)
	if _, err := GetGpuMemoryPartitionConfig(h); err != nil {
		t.Fatal(err)
	}
	if got.NumaRanges[0].End != ^uint64(0) {
		t.Fatal("later call changed the returned ranges")
	}
}

func TestPartitionResources(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_SUCCESS, 1)
	got, _, err := GetGpuAcceleratorPartitionProfile(h)
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
	again, _, err := GetGpuAcceleratorPartitionProfile(h)
	if err != nil || again.Resources[0][0] != ^uint32(0) {
		t.Fatalf("resource ownership: %+v, %v", again, err)
	}
}

func TestPartitionMemoryBounds(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_SUCCESS, 2)
	got, err := GetGpuMemoryPartitionConfig(h)
	assertNativeError(t, err, "amdsmi_get_gpu_memory_partition_config", AMDSMI_STATUS_UNEXPECTED_SIZE)
	assertZero(t, got)
}

func TestPartitionResourceBounds(t *testing.T) {
	h := fixtureHandle(t)
	for _, mode := range []uint32{2, 3, 6} {
		mockConfigure("amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_SUCCESS, mode)
		got, ids, err := GetGpuAcceleratorPartitionProfile(h)
		assertNativeError(t, err, "amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_UNEXPECTED_SIZE)
		assertZero(t, got)
		assertZero(t, ids)
	}
}

func TestPartitionUnknownCount(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_accelerator_partition_profile", AMDSMI_STATUS_SUCCESS, 4)
	got, ids, err := GetGpuAcceleratorPartitionProfile(h)
	if err != nil || got.NumPartitions != ^uint32(0) || got.ProfileIndex != ^uint32(0) || len(got.Resources) != 0 || !reflect.DeepEqual(ids, []uint32{7}) {
		t.Fatalf("unknown partition metadata: %+v, %v", got, err)
	}
}

func TestPartitionTupleErrors(t *testing.T) {
	h := fixtureHandle(t)
	const op = "amdsmi_get_gpu_accelerator_partition_profile"
	for _, code := range []Status{AMDSMI_STATUS_IO, AMDSMI_STATUS_MORE_DATA, Status(0x12345678)} {
		mockConfigure(op, code, 0)
		profile, ids, err := GetGpuAcceleratorPartitionProfile(h)
		assertNativeError(t, err, op, code)
		assertZero(t, profile)
		assertZero(t, ids)
	}
	mockConfigure(op, AMDSMI_STATUS_SUCCESS, 0)
	profile, ids, err := GetGpuAcceleratorPartitionProfile(ProcessorHandle{})
	assertNativeError(t, err, op, AMDSMI_STATUS_INVAL)
	assertZero(t, profile)
	assertZero(t, ids)
}
