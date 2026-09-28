// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo && amdsmi_mock

package amdsmi

import (
	"reflect"
	"strings"
	"testing"
)

func TestTelemetryTemperature(t *testing.T) {
	checkQuery(t, "amdsmi_get_temp_metric", func(h ProcessorHandle) (int64, error) {
		return GetTemperature(h, AMDSMI_TEMPERATURE_TYPE_HOTSPOT, AMDSMI_TEMP_CURRENT)
	}, int64(-17))
}

func TestTelemetryPower(t *testing.T) {
	checkQuery(t, "amdsmi_get_power_info", GetPowerInfo, PowerInfo{
		SocketPowerWatts: ^uint64(0), CurrentSocketPowerWatts: ^uint32(0),
		AverageSocketPowerWatts: 275, GFXVoltageMillivolts: ^uint64(0),
		SOCVoltageMillivolts: 900, MemoryVoltageMillivolts: 850,
		PowerLimitMicrowatts: 350000000, UBBPowerWatts: ^uint32(0),
	})
}

func TestTelemetryPowerCap(t *testing.T) {
	checkQuery(t, "amdsmi_get_power_cap_info", func(h ProcessorHandle) (PowerCapInfo, error) {
		return GetPowerCapInfo(h, 7)
	}, PowerCapInfo{PowerCapMicrowatts: 300000000, DefaultPowerCapMicrowatts: 325000000,
		DPMLevel: 3, MinPowerCapMicrowatts: 100000000, MaxPowerCapMicrowatts: 400000000})
}

// Auxiliary native queries can fail independently of the primary power-cap query.
func TestTelemetryPowerCapPartial(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_power_cap_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetPowerCapInfo(h, 7)
	want := PowerCapInfo{PowerCapMicrowatts: 300000000}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestTelemetryPowerCapErrorDiscardsOutput(t *testing.T) {
	for _, code := range []StatusCode{AMDSMI_STATUS_IO, AMDSMI_STATUS_MORE_DATA,
		StatusCode(0x12345678)} {
		t.Run(code.Error(), func(t *testing.T) {
			h := fixtureHandle(t)
			mockConfigure("amdsmi_get_power_cap_info", code, 2)
			got, err := GetPowerCapInfo(h, 7)
			assertNativeError(t, err, "amdsmi_get_power_cap_info", code)
			assertZero(t, got)
		})
	}
}

func TestTelemetryClock(t *testing.T) {
	checkQuery(t, "amdsmi_get_clock_info", func(h ProcessorHandle) (ClockInfo, error) {
		return GetClockInfo(h, AMDSMI_CLK_TYPE_GFX)
	}, ClockInfo{ClockMHz: ^uint32(0), MinClockMHz: 500, MaxClockMHz: 2100,
		LockedRaw: 255, DeepSleepRaw: 222})
}

func TestTelemetryFrequencies(t *testing.T) {
	checkQuery(t, "amdsmi_get_clk_freq", func(h ProcessorHandle) (Frequencies, error) {
		return GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	}, Frequencies{HasDeepSleep: true, CurrentIndex: ^uint32(0),
		Hertz: []uint64{500000000, 2400000000}})
}

func TestTelemetryFrequencyBounds(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_clk_freq", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	assertNativeError(t, err, "amdsmi_get_clk_freq", AMDSMI_STATUS_UNEXPECTED_SIZE)
	assertZero(t, got)
}

func TestTelemetryFrequenciesEmpty(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_clk_freq", AMDSMI_STATUS_SUCCESS, 2)
	got, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	if err != nil || len(got.Hertz) != 0 {
		t.Fatalf("want empty frequency list, got %#v, err=%v", got, err)
	}
	if !got.HasDeepSleep || got.CurrentIndex != ^uint32(0) {
		t.Fatalf("empty frequencies lost native metadata: %#v", got)
	}
}

func TestTelemetryFrequenciesExactCapacity(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_clk_freq", AMDSMI_STATUS_SUCCESS, 3)
	got, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	if err != nil || len(got.Hertz) != maxFrequencies {
		t.Fatalf("want %d frequencies, got %d, err=%v", maxFrequencies, len(got.Hertz), err)
	}
	if got.HasDeepSleep || got.CurrentIndex != uint32(maxFrequencies-1) {
		t.Fatalf("exact-capacity frequencies lost native metadata: %#v", got)
	}
	for i, hz := range got.Hertz {
		if want := uint64(i+1) * 1000000; hz != want {
			t.Fatalf("frequency %d: want %d, got %d", i, want, hz)
		}
	}
}

func TestTelemetryFrequenciesOwnedCopy(t *testing.T) {
	h := fixtureHandle(t)
	want := Frequencies{HasDeepSleep: true, CurrentIndex: ^uint32(0),
		Hertz: []uint64{500000000, 2400000000}}
	first, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	if err != nil || !reflect.DeepEqual(first, want) {
		t.Fatalf("want %#v, got %#v, err=%v", want, first, err)
	}
	first.Hertz[0] = 0xdead
	second, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	if err != nil || !reflect.DeepEqual(second, want) {
		t.Fatalf("want %#v, got %#v, err=%v", want, second, err)
	}
	if first.Hertz[0] != 0xdead {
		t.Fatal("later query overwrote the first frequency slice")
	}
	second.Hertz[1] = 0xbeef
	if first.Hertz[1] != want.Hertz[1] {
		t.Fatal("frequency slice shares backing storage across calls")
	}
}

func TestTelemetryFrequenciesErrorDiscardsOutput(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_clk_freq", AMDSMI_STATUS_MORE_DATA, 4)
	got, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	assertNativeError(t, err, "amdsmi_get_clk_freq", AMDSMI_STATUS_MORE_DATA)
	assertZero(t, got)
}

func TestTelemetryActivity(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_activity", GetActivity,
		Activity{GFXPercent: 65535, UMCPercent: 43, MMPercent: 7})
}

func TestTelemetryMemoryTotal(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_memory_total", func(h ProcessorHandle) (uint64, error) {
		return GetMemoryTotal(h, AMDSMI_MEM_TYPE_VRAM)
	}, uint64(1<<40))
}

func TestTelemetryMemoryUsage(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_memory_usage", func(h ProcessorHandle) (uint64, error) {
		return GetMemoryUsage(h, AMDSMI_MEM_TYPE_GTT)
	}, uint64(1<<39+7))
}

func TestTelemetryVRAM(t *testing.T) {
	checkQuery(t, "amdsmi_get_gpu_vram_info", GetVRAMInfo,
		VRAMInfo{Type: AMDSMI_VRAM_TYPE_LPDDR5, Vendor: "vendor", SizeMB: 196608,
			BitWidth: ^uint32(0), MaxBandwidthGBPerSecond: 5300})
}

func TestTelemetryVRAMVendorUnterminated(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_vram_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetVRAMInfo(h)
	want := VRAMInfo{Type: AMDSMI_VRAM_TYPE_LPDDR5,
		Vendor: strings.Repeat("V", nativeStringCapacity), SizeMB: 196608,
		BitWidth: ^uint32(0), MaxBandwidthGBPerSecond: 5300}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestTelemetryVRAMUnknownType(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_vram_info", AMDSMI_STATUS_SUCCESS, 2)
	got, err := GetVRAMInfo(h)
	want := VRAMInfo{Type: VRAMType(9999), Vendor: "vendor", SizeMB: 196608,
		BitWidth: ^uint32(0), MaxBandwidthGBPerSecond: 5300}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}
