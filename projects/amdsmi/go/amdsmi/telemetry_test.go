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
		return GetTempMetric(h, AMDSMI_TEMPERATURE_TYPE_HOTSPOT, AMDSMI_TEMP_CURRENT)
	}, int64(-17))
}

func TestTelemetryPower(t *testing.T) {
	checkQuery(t, "amdsmi_get_power_info", GetPowerInfo, PowerInfo{
		SocketPower: ^uint64(0), CurrentSocketPower: ^uint32(0),
		AverageSocketPower: 275, GfxVoltage: ^uint64(0),
		SocVoltage: 900, MemVoltage: 850,
		PowerLimit: 350000000, UbbPower: ^uint32(0),
	})
}

func TestTelemetryPowerManagementEnabled(t *testing.T) {
	checkQuery(t, "amdsmi_is_gpu_power_management_enabled", IsGpuPowerManagementEnabled, true)
}

func TestTelemetryPowerManagementDisabled(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_is_gpu_power_management_enabled", AMDSMI_STATUS_SUCCESS, 1)
	got, err := IsGpuPowerManagementEnabled(h)
	if err != nil || got {
		t.Fatalf("want disabled power management, got %v, err=%v", got, err)
	}
}

func TestTelemetryPowerManagementErrorDiscardsOutput(t *testing.T) {
	h := fixtureHandle(t)
	for _, code := range []Status{AMDSMI_STATUS_IO, AMDSMI_STATUS_MORE_DATA, Status(0x12345678)} {
		mockConfigure("amdsmi_is_gpu_power_management_enabled", code, 2)
		got, err := IsGpuPowerManagementEnabled(h)
		assertNativeError(t, err, "amdsmi_is_gpu_power_management_enabled", code)
		assertZero(t, got)
	}
}

func TestTelemetryPowerCap(t *testing.T) {
	checkQuery(t, "amdsmi_get_power_cap_info", func(h ProcessorHandle) (PowerCapInfo, error) {
		return GetPowerCapInfo(h, 7)
	}, PowerCapInfo{PowerCap: 300000000, DefaultPowerCap: 325000000,
		DpmCap: 3, MinPowerCap: 100000000, MaxPowerCap: 400000000})
}

// Auxiliary native queries can fail independently of the primary power-cap query.
func TestTelemetryPowerCapPartial(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_power_cap_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetPowerCapInfo(h, 7)
	want := PowerCapInfo{PowerCap: 300000000}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestTelemetryPowerCapErrorDiscardsOutput(t *testing.T) {
	for _, code := range []Status{AMDSMI_STATUS_IO, AMDSMI_STATUS_MORE_DATA,
		Status(0x12345678)} {
		t.Run(code.Error(), func(t *testing.T) {
			h := fixtureHandle(t)
			mockConfigure("amdsmi_get_power_cap_info", code, 2)
			got, err := GetPowerCapInfo(h, 7)
			assertNativeError(t, err, "amdsmi_get_power_cap_info", code)
			assertZero(t, got)
		})
	}
}

func TestTelemetryPcie(t *testing.T) {
	checkQuery(t, "amdsmi_get_pcie_info", GetPcieInfo, PcieInfo{
		Static: PcieStatic{MaxPcieWidth: 16, MaxPcieSpeed: 32000,
			PcieInterfaceVersion: 4, SlotType: AMDSMI_CARD_FORM_FACTOR_OAM, MaxPcieInterfaceVersion: 5},
		Metric: PcieMetric{PcieWidth: 8, PcieSpeed: 16000, PcieBandwidth: 123456,
			PcieReplayCount: 1<<40 + 1, PcieL0ToRecoveryCount: 1<<41 + 2,
			PcieReplayRollOverCount: 1<<42 + 3, PcieNakSentCount: 1<<43 + 4,
			PcieNakReceivedCount: 1<<44 + 5, PcieLcPerfOtherEndRecoveryCount: 12345},
	})
}

func TestTelemetryPcieUnavailable(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_pcie_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetPcieInfo(h)
	want := PcieInfo{
		Static: PcieStatic{MaxPcieWidth: ^uint16(0), MaxPcieSpeed: ^uint32(0),
			PcieInterfaceVersion: ^uint32(0), SlotType: CardFormFactor(9999), MaxPcieInterfaceVersion: ^uint32(0)},
		Metric: PcieMetric{PcieWidth: ^uint16(0), PcieSpeed: ^uint32(0), PcieBandwidth: ^uint32(0),
			PcieReplayCount: ^uint64(0), PcieL0ToRecoveryCount: ^uint64(0),
			PcieReplayRollOverCount: ^uint64(0), PcieNakSentCount: ^uint64(0),
			PcieNakReceivedCount: ^uint64(0), PcieLcPerfOtherEndRecoveryCount: ^uint32(0)},
	}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestTelemetryPcieErrorDiscardsOutput(t *testing.T) {
	h := fixtureHandle(t)
	for _, code := range []Status{AMDSMI_STATUS_IO, AMDSMI_STATUS_MORE_DATA, Status(0x12345678)} {
		mockConfigure("amdsmi_get_pcie_info", code, 2)
		got, err := GetPcieInfo(h)
		assertNativeError(t, err, "amdsmi_get_pcie_info", code)
		assertZero(t, got)
	}
}

func TestTelemetryLinkMetrics(t *testing.T) {
	checkQuery(t, "amdsmi_get_link_metrics", GetLinkMetrics, LinkMetrics{
		NumLinks: 2,
		Links: []LinkMetricInfo{
			{Bdf: Bdf(0xabcde123456789ab), BitRate: 32, MaxBandwidth: 64,
				LinkType: AMDSMI_LINK_TYPE_XGMI, Read: 1<<40 + 11, Write: 1<<41 + 13,
				LinkStatus: AMDSMI_LINK_STATUS_ENABLED},
			{Bdf: Bdf(^uint64(0)), BitRate: ^uint32(0), MaxBandwidth: ^uint32(0),
				LinkType: LinkType(9999), Read: ^uint64(0), Write: ^uint64(0), LinkStatus: LinkStatus(8888)},
		},
	})
}

func TestTelemetryLinkMetricsCounts(t *testing.T) {
	for _, test := range []struct {
		name  string
		mode  uint32
		count int
	}{
		{"Empty", 1, 0}, {"ExactCapacity", 2, 64},
		{"Oversized", 3, -1}, {"Maximum", 4, -1},
	} {
		t.Run(test.name, func(t *testing.T) {
			h := fixtureHandle(t)
			mockConfigure("amdsmi_get_link_metrics", AMDSMI_STATUS_SUCCESS, test.mode)
			got, err := GetLinkMetrics(h)
			if test.count < 0 {
				assertNativeError(t, err, "amdsmi_get_link_metrics", AMDSMI_STATUS_UNEXPECTED_SIZE)
				assertZero(t, got)
				return
			}
			if err != nil || got.NumLinks != uint32(test.count) || len(got.Links) != test.count {
				t.Fatalf("want %d links, got %#v, err=%v", test.count, got, err)
			}
			for index, link := range got.Links {
				if link.Read != uint64(1<<40+11+index) {
					t.Fatalf("link %d: unexpected read count %d", index, link.Read)
				}
			}
		})
	}
}

func TestTelemetryLinkMetricsOwnedCopy(t *testing.T) {
	h := fixtureHandle(t)
	first, err := GetLinkMetrics(h)
	if err != nil || len(first.Links) != 2 {
		t.Fatalf("first query: %#v, err=%v", first, err)
	}
	want := append([]LinkMetricInfo(nil), first.Links...)
	first.Links[0].Read = 123
	second, err := GetLinkMetrics(h)
	if err != nil || !reflect.DeepEqual(second.Links, want) || first.Links[0].Read != 123 {
		t.Fatalf("queries share output: first=%#v, second=%#v, err=%v", first, second, err)
	}
	second.Links[1].Write = 456
	if first.Links[1] != want[1] {
		t.Fatal("link slices share backing storage")
	}
}

func TestTelemetryLinkMetricsErrorDiscardsOutput(t *testing.T) {
	h := fixtureHandle(t)
	for _, code := range []Status{AMDSMI_STATUS_IO, AMDSMI_STATUS_MORE_DATA, Status(0x12345678)} {
		mockConfigure("amdsmi_get_link_metrics", code, 5)
		got, err := GetLinkMetrics(h)
		assertNativeError(t, err, "amdsmi_get_link_metrics", code)
		assertZero(t, got)
	}
}

func TestTelemetrySocPstate(t *testing.T) {
	checkQuery(t, "amdsmi_get_soc_pstate", GetSocPstate, DpmPolicy{
		NumSupported: 2, Current: 1,
		Policies: []DpmPolicyEntry{
			{PolicyID: 3, PolicyDescription: "balanced"},
			{PolicyID: 9, PolicyDescription: "performance"},
		},
	})
}

func TestTelemetrySocPstateCounts(t *testing.T) {
	for _, test := range []struct {
		name    string
		mode    uint32
		count   int
		current uint32
	}{
		{"Empty", 1, 0, ^uint32(0)},
		{"ExactCapacity", 2, 32, 31},
		{"Oversized", 3, -1, 0},
		{"Maximum", 4, -1, 0},
	} {
		t.Run(test.name, func(t *testing.T) {
			handle := fixtureHandle(t)
			mockConfigure("amdsmi_get_soc_pstate", AMDSMI_STATUS_SUCCESS, test.mode)
			got, err := GetSocPstate(handle)
			if test.count < 0 {
				assertNativeError(t, err, "amdsmi_get_soc_pstate", AMDSMI_STATUS_UNEXPECTED_SIZE)
				assertZero(t, got)
				return
			}
			if err != nil || got.NumSupported != uint32(test.count) || len(got.Policies) != test.count || got.Policies == nil || got.Current != test.current {
				t.Fatalf("want %d policies, current %d, got %#v, err=%v", test.count, test.current, got, err)
			}
			for index, policy := range got.Policies {
				want := DpmPolicyEntry{PolicyID: uint32(0x100 + index), PolicyDescription: "balanced"}
				if policy != want {
					t.Fatalf("policy %d: want %#v, got %#v", index, want, policy)
				}
			}
		})
	}
}

func TestTelemetrySocPstateUnavailable(t *testing.T) {
	for _, test := range []struct {
		name    string
		mode    uint32
		current uint32
	}{
		{"UnavailableCurrent", 5, ^uint32(0)},
		{"UnknownCurrent", 6, 0x12345678},
	} {
		t.Run(test.name, func(t *testing.T) {
			handle := fixtureHandle(t)
			mockConfigure("amdsmi_get_soc_pstate", AMDSMI_STATUS_SUCCESS, test.mode)
			got, err := GetSocPstate(handle)
			want := DpmPolicy{NumSupported: 2, Current: test.current,
				Policies: []DpmPolicyEntry{
					{PolicyID: ^uint32(0), PolicyDescription: ""},
					{PolicyID: 9, PolicyDescription: "performance"},
				},
			}
			if err != nil || !reflect.DeepEqual(got, want) {
				t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
			}
		})
	}
}

func TestTelemetrySocPstateBoundedStrings(t *testing.T) {
	handle := fixtureHandle(t)
	mockConfigure("amdsmi_get_soc_pstate", AMDSMI_STATUS_SUCCESS, 7)
	got, err := GetSocPstate(handle)
	want := DpmPolicy{NumSupported: 2, Current: 1,
		Policies: []DpmPolicyEntry{
			{PolicyID: 3, PolicyDescription: strings.Repeat("P", nativeStringCapacity)},
			{PolicyID: ^uint32(0), PolicyDescription: "eco"},
		},
	}
	if err != nil || !reflect.DeepEqual(got, want) {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestTelemetrySocPstateErrorDiscardsOutput(t *testing.T) {
	for _, test := range []struct {
		name string
		mode uint32
		code Status
	}{
		{"Partial", 0, AMDSMI_STATUS_IO},
		{"Oversized", 3, AMDSMI_STATUS_MORE_DATA},
		{"Maximum", 4, Status(0x12345678)},
	} {
		t.Run(test.name, func(t *testing.T) {
			handle := fixtureHandle(t)
			mockConfigure("amdsmi_get_soc_pstate", test.code, test.mode)
			got, err := GetSocPstate(handle)
			assertNativeError(t, err, "amdsmi_get_soc_pstate", test.code)
			assertZero(t, got)
		})
	}
}

func TestTelemetrySocPstateOwnedCopy(t *testing.T) {
	handle := fixtureHandle(t)
	first, err := GetSocPstate(handle)
	if err != nil || len(first.Policies) != 2 {
		t.Fatalf("first query: %#v, err=%v", first, err)
	}
	want := append([]DpmPolicyEntry(nil), first.Policies...)
	first.Policies[0].PolicyID = 123
	second, err := GetSocPstate(handle)
	if err != nil || !reflect.DeepEqual(second.Policies, want) || first.Policies[0].PolicyID != 123 {
		t.Fatalf("queries share output: first=%#v, second=%#v, err=%v", first, second, err)
	}
	second.Policies[1].PolicyDescription = "changed"
	if first.Policies[1] != want[1] {
		t.Fatal("policy slices share backing storage")
	}
	mockConfigure("amdsmi_get_soc_pstate", AMDSMI_STATUS_SUCCESS, 7)
	third, err := GetSocPstate(handle)
	if err != nil || len(third.Policies) != 2 || third.Policies[0].PolicyDescription != strings.Repeat("P", nativeStringCapacity) {
		t.Fatalf("changed native descriptions: %#v, err=%v", third, err)
	}
	if first.Policies[0].PolicyDescription != want[0].PolicyDescription || first.Policies[1] != want[1] {
		t.Fatal("later native query changed retained policy strings")
	}
}

func TestTelemetryClock(t *testing.T) {
	checkQuery(t, "amdsmi_get_clock_info", func(h ProcessorHandle) (ClkInfo, error) {
		return GetClockInfo(h, AMDSMI_CLK_TYPE_GFX)
	}, ClkInfo{Clk: ^uint32(0), MinClk: 500, MaxClk: 2100,
		ClkLocked: false, ClkDeepSleep: false, ClkLockedRaw: 255, ClkDeepSleepRaw: 222})
}

func TestTelemetryClockBooleanBytesRemainUnavailable(t *testing.T) {
	h := fixtureHandle(t)
	for _, mode := range []uint32{1, 2} {
		mockConfigure("amdsmi_get_clock_info", AMDSMI_STATUS_SUCCESS, mode)
		got, err := GetClockInfo(h, AMDSMI_CLK_TYPE_GFX)
		want := ClkInfo{Clk: ^uint32(0), MinClk: 500, MaxClk: 2100,
			ClkLockedRaw: uint8(mode - 1), ClkDeepSleepRaw: uint8(2 - mode)}
		if err != nil || got != want {
			t.Fatalf("mode %d: want %#v, got %#v, err=%v", mode, want, got, err)
		}
	}
}

func TestTelemetryFrequencies(t *testing.T) {
	checkQuery(t, "amdsmi_get_clk_freq", func(h ProcessorHandle) (Frequencies, error) {
		return GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	}, Frequencies{HasDeepSleep: true, NumSupported: 2, Current: ^uint32(0),
		Values: []uint64{500000000, 2400000000}})
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
	if err != nil || len(got.Values) != 0 || got.NumSupported != 0 {
		t.Fatalf("want empty frequency list, got %#v, err=%v", got, err)
	}
	if !got.HasDeepSleep || got.Current != ^uint32(0) {
		t.Fatalf("empty frequencies lost native metadata: %#v", got)
	}
}

func TestTelemetryFrequenciesExactCapacity(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_clk_freq", AMDSMI_STATUS_SUCCESS, 3)
	got, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	if err != nil || len(got.Values) != maxFrequencies || got.NumSupported != uint32(maxFrequencies) {
		t.Fatalf("want %d frequencies, got %d, err=%v", maxFrequencies, len(got.Values), err)
	}
	if got.HasDeepSleep || got.Current != uint32(maxFrequencies-1) {
		t.Fatalf("exact-capacity frequencies lost native metadata: %#v", got)
	}
	for i, hz := range got.Values {
		if want := uint64(i+1) * 1000000; hz != want {
			t.Fatalf("frequency %d: want %d, got %d", i, want, hz)
		}
	}
}

func TestTelemetryFrequenciesOwnedCopy(t *testing.T) {
	h := fixtureHandle(t)
	want := Frequencies{HasDeepSleep: true, NumSupported: 2, Current: ^uint32(0),
		Values: []uint64{500000000, 2400000000}}
	first, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	if err != nil || !reflect.DeepEqual(first, want) {
		t.Fatalf("want %#v, got %#v, err=%v", want, first, err)
	}
	first.Values[0] = 0xdead
	second, err := GetClockFrequencies(h, AMDSMI_CLK_TYPE_MEM)
	if err != nil || !reflect.DeepEqual(second, want) {
		t.Fatalf("want %#v, got %#v, err=%v", want, second, err)
	}
	if first.Values[0] != 0xdead {
		t.Fatal("later query overwrote the first frequency slice")
	}
	second.Values[1] = 0xbeef
	if first.Values[1] != want.Values[1] {
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
	checkQuery(t, "amdsmi_get_gpu_activity", GetGpuActivity,
		EngineUsage{GfxActivity: 65535, UmcActivity: 43, MmActivity: 7})
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
	checkQuery(t, "amdsmi_get_gpu_vram_info", GetGpuVramInfo,
		VramInfo{VramType: AMDSMI_VRAM_TYPE_LPDDR5, VramVendor: "vendor", VramSize: 196608,
			VramBitWidth: ^uint32(0), VramMaxBandwidth: 5300})
}

func TestTelemetryVRAMVendorUnterminated(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_vram_info", AMDSMI_STATUS_SUCCESS, 1)
	got, err := GetGpuVramInfo(h)
	want := VramInfo{VramType: AMDSMI_VRAM_TYPE_LPDDR5,
		VramVendor: strings.Repeat("V", nativeStringCapacity), VramSize: 196608,
		VramBitWidth: ^uint32(0), VramMaxBandwidth: 5300}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}

func TestTelemetryVRAMUnknownType(t *testing.T) {
	h := fixtureHandle(t)
	mockConfigure("amdsmi_get_gpu_vram_info", AMDSMI_STATUS_SUCCESS, 2)
	got, err := GetGpuVramInfo(h)
	want := VramInfo{VramType: VramType(9999), VramVendor: "vendor", VramSize: 196608,
		VramBitWidth: ^uint32(0), VramMaxBandwidth: 5300}
	if err != nil || got != want {
		t.Fatalf("want %#v, got %#v, err=%v", want, got, err)
	}
}
