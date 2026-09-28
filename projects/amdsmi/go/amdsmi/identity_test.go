// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import (
	"reflect"
	"strings"
	"testing"
)

func TestIdentityUUID(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_device_uuid", GetUUID,
		"00000000-0000-0000-0000-000000000001")
}

func TestIdentityUUIDUnterminated(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_device_uuid", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetUUID(h)
	want := strings.Repeat("U", gpuUUIDSize)
	if err != nil || got != want {
		t.Fatalf("want %q, got %q, err=%v", want, got, err)
	}
}

func TestIdentityUUIDBounds(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_device_uuid", AMDSMI_STATUS_SUCCESS, 2)
	got, err := GetUUID(h)
	assertNativeError(t, err, "amdsmi_get_gpu_device_uuid", AMDSMI_STATUS_UNEXPECTED_SIZE)
	assertZero(t, got)
}

func TestIdentityUUIDEmpty(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_device_uuid", AMDSMI_STATUS_SUCCESS, 3)
	got, err := GetUUID(h)
	if err != nil || got != "" {
		t.Fatalf("want empty UUID, got %q, err=%v", got, err)
	}
}

func TestIdentityASIC(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_asic_info", GetGPUAsicInfo, AsicInfo{
		MarketName: "test-gpu", VendorID: 0x1002, VendorName: "AMD",
		SubvendorID: 0x1002, DeviceID: 1<<40 + 0x1234, RevID: ^uint32(0),
		AsicSerial: "asic-serial", OamID: 7, NumComputeUnits: 304,
		TargetGraphicsVersion: 1<<40 + 950, SubsystemID: 0x42, Flags: 1<<63 + 1,
		PhysicalAcceleratorID: 9, ChipRevisionID: 0x91, ExternalRevisionID: 0x92,
	})
}

func TestIdentityASICStringsUnterminated(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_asic_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetGPUAsicInfo(h)
	want := AsicInfo{
		MarketName: strings.Repeat("M", nativeStringCapacity),
		VendorName: strings.Repeat("V", nativeStringCapacity),
		AsicSerial: strings.Repeat("S", nativeStringCapacity),
	}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestIdentityDriver(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_driver_info", GetDriverInfo,
		DriverInfo{Version: "6.16", Date: "20260925", Name: "amdgpu"})
}

func TestIdentityDriverStringsUnterminated(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_driver_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetDriverInfo(h)
	want := DriverInfo{
		Version: strings.Repeat("V", nativeStringCapacity),
		Date:    strings.Repeat("D", nativeStringCapacity),
		Name:    strings.Repeat("N", nativeStringCapacity),
	}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestIdentityBoard(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_board_info", GetBoardInfo,
		BoardInfo{ModelNumber: "model", ProductSerial: "serial", FRUID: "fru",
			ProductName: "board", ManufacturerName: "AMD"})
}

func TestIdentityBoardStringsUnterminated(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_board_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetBoardInfo(h)
	want := BoardInfo{
		ModelNumber:      strings.Repeat("M", nativeStringCapacity),
		ProductSerial:    strings.Repeat("S", nativeStringCapacity),
		FRUID:            strings.Repeat("F", nativeStringCapacity),
		ProductName:      strings.Repeat("P", nativeStringCapacity),
		ManufacturerName: strings.Repeat("A", nativeStringCapacity),
	}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestIdentityFirmware(t *testing.T) {
	checkQuery(t, "amdsmi_get_fw_info", GetFirmwareInfo, []FirmwareInfo{
		{ID: AMDSMI_FW_ID_SMU, Version: 1<<40 + 3},
		{ID: FirmwareBlock(0x1234), Version: ^uint64(0)},
	})
}

func TestIdentityFirmwareEmpty(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_fw_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetFirmwareInfo(h)
	if err != nil || len(got) != 0 {
		t.Fatalf("want empty firmware list, got %#v, err=%v", got, err)
	}
}

func TestIdentityFirmwareExactCapacity(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_fw_info", AMDSMI_STATUS_SUCCESS, 2)
	got, err := GetFirmwareInfo(h)
	if err != nil || len(got) != maxFirmwareEntries {
		t.Fatalf("want %d entries, got %d, err=%v", maxFirmwareEntries, len(got), err)
	}
	for i, entry := range got {
		want := FirmwareInfo{ID: FirmwareBlock(i), Version: uint64(i)}
		if entry != want {
			t.Fatalf("entry %d: want %#v, got %#v", i, want, entry)
		}
	}
}

func TestIdentityFirmwareOverCapacity(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_fw_info", AMDSMI_STATUS_SUCCESS, 3)
	got, err := GetFirmwareInfo(h)
	assertNativeError(t, err, "amdsmi_get_fw_info", AMDSMI_STATUS_UNEXPECTED_SIZE)
	assertZero(t, got)
}

func TestIdentityFirmwareOwnedCopy(t *testing.T) {
	h := fixtureHandle(t)
	want := []FirmwareInfo{
		{ID: AMDSMI_FW_ID_SMU, Version: 1<<40 + 3},
		{ID: FirmwareBlock(0x1234), Version: ^uint64(0)},
	}
	first, err := GetFirmwareInfo(h)
	if err != nil || !reflect.DeepEqual(first, want) {
		t.Fatalf("want %#v, got %#v, err=%v", want, first, err)
	}
	changed := FirmwareInfo{ID: FirmwareBlock(0x5678), Version: 0xdead}
	first[0] = changed
	second, err := GetFirmwareInfo(h)
	if err != nil || !reflect.DeepEqual(second, want) {
		t.Fatalf("want %#v, got %#v, err=%v", want, second, err)
	}
	if first[0] != changed {
		t.Fatal("later query overwrote the first firmware slice")
	}
	second[1] = changed
	if first[1] != want[1] {
		t.Fatal("firmware slice shares backing storage across calls")
	}
}

func TestIdentityFirmwareErrorDiscardsOutput(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_fw_info", AMDSMI_STATUS_MORE_DATA, 4)
	got, err := GetFirmwareInfo(h)
	assertNativeError(t, err, "amdsmi_get_fw_info", AMDSMI_STATUS_MORE_DATA)
	assertZero(t, got)
}

func TestIdentityVBIOS(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_vbios_info", GetVBIOSInfo,
		VBIOSInfo{Name: "vbios", BuildDate: "2026-09-25", PartNumber: "part",
			Version: "1", BootFirmware: "ubl"})
}

func TestIdentityVBIOSStringsUnterminated(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_vbios_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetVBIOSInfo(h)
	want := VBIOSInfo{
		Name:         strings.Repeat("N", nativeStringCapacity),
		BuildDate:    strings.Repeat("B", nativeStringCapacity),
		PartNumber:   strings.Repeat("P", nativeStringCapacity),
		Version:      strings.Repeat("V", nativeStringCapacity),
		BootFirmware: strings.Repeat("F", nativeStringCapacity),
	}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}
