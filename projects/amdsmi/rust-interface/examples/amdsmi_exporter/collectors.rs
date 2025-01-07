// Copyright (C) 2024 Advanced Micro Devices. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

use crate::amdsmi_collectors::AmdsmiCollectors;
use crate::amdsmi_metric::*;
use amdsmi::*;
use prometheus_client::registry::Unit;
use std::sync::Arc;
pub fn collectors_create(collectors: &mut AmdsmiCollectors) {
    add_metric_group!(
        collectors,
        "AmdsmiEngineUsageMetrics",
        amdsmi_get_gpu_activity,
        AmdsmiEngineUsageT,
        (
            "AMDSMI_FI_GFX_ACTIVITY",
            "GFX Activity",
            Unit::Ratios,
            gfx_activity
        ),
        (
            "AMDSMI_FI_UMC_ACTIVITY",
            "UMC Activity",
            Unit::Ratios,
            umc_activity
        ),
        (
            "AMDSMI_FI_MM_ACTIVITY",
            "MM Activity",
            Unit::Ratios,
            mm_activity
        )
    );

    add_metric_group!(
        collectors,
        "AmdsmiPowerInfoMetrics",
        amdsmi_get_power_info,
        AmdsmiPowerInfoT,
        (
            "AMDSMI_FI_CURRENT_SOCKET_POWER",
            "Current Socket Power",
            metric_unit_other!("W"),
            current_socket_power
        ),
        (
            "AMDSMI_FI_AVERAGE_SOCKET_POWER",
            "Average Socket Power",
            metric_unit_other!("W"),
            average_socket_power
        ),
        (
            "AMDSMI_FI_GFX_VOLTAGE",
            "GFX Voltage",
            metric_unit_other!("mV"),
            gfx_voltage
        ),
        (
            "AMDSMI_FI_SOC_VOLTAGE",
            "SOC Voltage",
            metric_unit_other!("mV"),
            soc_voltage
        ),
        (
            "AMDSMI_FI_MEM_VOLTAGE",
            "Memory Voltage",
            metric_unit_other!("mV"),
            mem_voltage
        ),
    );

    add_metric_group!(
        collectors,
        "AmdsmiGfxClockMetrics",
        amdsmi_func!(amdsmi_get_clock_info, AmdsmiClkTypeT::AmdsmiClkTypeGfx),
        AmdsmiClkInfoT,
        (
            "AMDSMI_FI_GFX_CLK",
            "GFX Clock",
            metric_unit_other!("MHz"),
            clk
        ),
        (
            "AMDSMI_FI_GFX_MIN_CLK",
            "GFX Min Clock",
            metric_unit_other!("MHz"),
            min_clk
        ),
        (
            "AMDSMI_FI_GFX_MAX_CLK",
            "GFX Max Clock",
            metric_unit_other!("MHz"),
            max_clk
        )
    );

    add_metric_group!(
        collectors,
        "AmdsmiVRAMUsageMetrics",
        amdsmi_get_gpu_vram_usage,
        AmdsmiVramUsageT,
        (
            "AMDSMI_FI_VRAM_TOTAL",
            "VRAM total",
            metric_unit_other!("MB"),
            vram_total
        ),
        (
            "AMDSMI_FI_VRAM_USED",
            "VRAM used",
            metric_unit_other!("MB"),
            vram_used
        ),
    );

    add_metric_group!(
        collectors,
        "AmdsmiMemClockMetrics",
        amdsmi_func!(amdsmi_get_clock_info, AmdsmiClkTypeT::AmdsmiClkTypeMem),
        AmdsmiClkInfoT,
        (
            "AMDSMI_FI_MEM_CLK",
            "Memory Clock",
            metric_unit_other!("MHz"),
            clk
        ),
        (
            "AMDSMI_FI_MEM_MIN_CLK",
            "Memory Min Clock",
            metric_unit_other!("MHz"),
            min_clk
        ),
        (
            "AMDSMI_FI_MEM_MAX_CLK",
            "Memory Max Clock",
            metric_unit_other!("MHz"),
            max_clk
        )
    );

    add_metric!(
        collectors,
        "AMDSMI_FI_TEMPERATURE_EDGE",
        "Current edge temperature",
        Unit::Celsius,
        amdsmi_func!(
            amdsmi_get_temp_metric,
            AmdsmiTemperatureTypeT::AmdsmiTemperatureTypeEdge,
            AmdsmiTemperatureMetricT::AmdsmiTempCurrent
        )
    );
    add_metric!(
        collectors,
        "AMDSMI_FI_TEMPERATURE_HOTSPOT",
        "Current Hotspot temperature",
        Unit::Celsius,
        amdsmi_func!(
            amdsmi_get_temp_metric,
            AmdsmiTemperatureTypeT::AmdsmiTemperatureTypeHotspot,
            AmdsmiTemperatureMetricT::AmdsmiTempCurrent
        )
    );
    add_metric!(
        collectors,
        "AMDSMI_FI_TEMPERATURE_MEM",
        "Current VRAM temperature",
        Unit::Celsius,
        amdsmi_func!(
            amdsmi_get_temp_metric,
            AmdsmiTemperatureTypeT::AmdsmiTemperatureTypeVram,
            AmdsmiTemperatureMetricT::AmdsmiTempCurrent
        )
    );
}
