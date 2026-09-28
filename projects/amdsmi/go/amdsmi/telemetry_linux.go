// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//go:build linux && cgo

package amdsmi

/*
#include <stddef.h>
#include <amd_smi/amdsmi.h>
*/
import "C"

// GetTemperature returns whole degrees Celsius, preserving negative values.
func GetTemperature(h ProcessorHandle, sensor TemperatureType, metric TemperatureMetric) (int64, error) {
	const op = "amdsmi_get_temp_metric"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (int64, error) {
		var out C.int64_t
		status := C.amdsmi_get_temp_metric(p, C.amdsmi_temperature_type_t(sensor),
			C.amdsmi_temperature_metric_t(metric), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return int64(out), nil
	})
}

func GetPowerInfo(h ProcessorHandle) (PowerInfo, error) {
	const op = "amdsmi_get_power_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (PowerInfo, error) {
		var out C.amdsmi_power_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_power_info(p, &out)); err != nil {
			return PowerInfo{}, err
		}
		return PowerInfo{
			SocketPowerWatts:        uint64(out.socket_power),
			CurrentSocketPowerWatts: uint32(out.current_socket_power),
			AverageSocketPowerWatts: uint32(out.average_socket_power),
			GFXVoltageMillivolts:    uint64(out.gfx_voltage),
			SOCVoltageMillivolts:    uint64(out.soc_voltage),
			MemoryVoltageMillivolts: uint64(out.mem_voltage),
			PowerLimitMicrowatts:    uint32(out.power_limit),
			UBBPowerWatts:           uint32(out.ubb_power),
		}, nil
	})
}

func GetPowerCapInfo(h ProcessorHandle, sensorIndex uint32) (PowerCapInfo, error) {
	const op = "amdsmi_get_power_cap_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (PowerCapInfo, error) {
		var out C.amdsmi_power_cap_info_t
		status := C.amdsmi_get_power_cap_info(p, C.uint32_t(sensorIndex), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return PowerCapInfo{}, err
		}
		return PowerCapInfo{
			PowerCapMicrowatts:        uint64(out.power_cap),
			DefaultPowerCapMicrowatts: uint64(out.default_power_cap),
			DPMLevel:                  uint64(out.dpm_cap),
			MinPowerCapMicrowatts:     uint64(out.min_power_cap),
			MaxPowerCapMicrowatts:     uint64(out.max_power_cap),
		}, nil
	})
}

func GetClockInfo(h ProcessorHandle, clock ClockType) (ClockInfo, error) {
	const op = "amdsmi_get_clock_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (ClockInfo, error) {
		var out C.amdsmi_clk_info_t
		status := C.amdsmi_get_clock_info(p, C.amdsmi_clk_type_t(clock), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return ClockInfo{}, err
		}
		return ClockInfo{
			ClockMHz:     uint32(out.clk),
			MinClockMHz:  uint32(out.min_clk),
			MaxClockMHz:  uint32(out.max_clk),
			LockedRaw:    uint8(out.clk_locked),
			DeepSleepRaw: uint8(out.clk_deep_sleep),
		}, nil
	})
}

func GetClockFrequencies(h ProcessorHandle, clock ClockType) (Frequencies, error) {
	const op = "amdsmi_get_clk_freq"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (Frequencies, error) {
		var out C.amdsmi_frequencies_t
		status := C.amdsmi_get_clk_freq(p, C.amdsmi_clk_type_t(clock), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return Frequencies{}, err
		}
		n, err := checkedCount(op, uint64(out.num_supported), len(out.frequency))
		if err != nil {
			return Frequencies{}, err
		}
		result := Frequencies{
			HasDeepSleep: bool(out.has_deep_sleep),
			CurrentIndex: uint32(out.current),
			Hertz:        make([]uint64, n),
		}
		for i := range result.Hertz {
			result.Hertz[i] = uint64(out.frequency[i])
		}
		return result, nil
	})
}

func GetActivity(h ProcessorHandle) (Activity, error) {
	const op = "amdsmi_get_gpu_activity"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (Activity, error) {
		var out C.amdsmi_engine_usage_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_activity(p, &out)); err != nil {
			return Activity{}, err
		}
		return Activity{
			GFXPercent: uint32(out.gfx_activity),
			UMCPercent: uint32(out.umc_activity),
			MMPercent:  uint32(out.mm_activity),
		}, nil
	})
}

// GetMemoryTotal returns bytes.
func GetMemoryTotal(h ProcessorHandle, memory MemoryType) (uint64, error) {
	const op = "amdsmi_get_gpu_memory_total"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (uint64, error) {
		var out C.uint64_t
		status := C.amdsmi_get_gpu_memory_total(p, C.amdsmi_memory_type_t(memory), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return uint64(out), nil
	})
}

// GetMemoryUsage returns bytes.
func GetMemoryUsage(h ProcessorHandle, memory MemoryType) (uint64, error) {
	const op = "amdsmi_get_gpu_memory_usage"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (uint64, error) {
		var out C.uint64_t
		status := C.amdsmi_get_gpu_memory_usage(p, C.amdsmi_memory_type_t(memory), &out)
		if err := nativeErrorLocked(op, status); err != nil {
			return 0, err
		}
		return uint64(out), nil
	})
}

func GetVRAMInfo(h ProcessorHandle) (VRAMInfo, error) {
	const op = "amdsmi_get_gpu_vram_info"
	return withProcessor(h, op, func(p C.amdsmi_processor_handle) (VRAMInfo, error) {
		var out C.amdsmi_vram_info_t
		if err := nativeErrorLocked(op, C.amdsmi_get_gpu_vram_info(p, &out)); err != nil {
			return VRAMInfo{}, err
		}
		return VRAMInfo{
			Type:                    VRAMType(out.vram_type),
			Vendor:                  boundedString(&out.vram_vendor[0], len(out.vram_vendor)),
			SizeMB:                  uint64(out.vram_size),
			BitWidth:                uint32(out.vram_bit_width),
			MaxBandwidthGBPerSecond: uint64(out.vram_max_bandwidth),
		}, nil
	})
}
