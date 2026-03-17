---
myst:
  html_meta:
    "description lang=en": "AMD SMI GPU violation monitoring and throttling status."
    "keywords": "gpu, violations, throttling, pviol, tviol, power, thermal, performance, throttle_status, gpu_metrics, prochot, ppt, hbm, mi300x"
---

# GPU Violations

GPU violations monitoring in AMD SMI tracks throttling events caused by power or thermal limits. When your GPU throttles, performance decreases to protect the hardware from damage due to overheating or excessive power draw. AMD SMI provides APIs to monitor violation percentages and identify the specific causes of throttling, enabling system administrators and developers to maintain GPU health and optimize performance.

## Key Concepts

**PVIOL (Power Violation)**: Percentage of time the GPU throttled due to power limits. This occurs when the GPU's power consumption exceeds safe limits, triggering power throttling to stay within the power budget.

**TVIOL (Thermal Violation)**: Percentage of time the GPU throttled due to temperature limits. This happens when GPU temperatures exceed safe operating thresholds, causing the GPU to reduce performance to cool down.

**throttle_status**: Real-time bit flags showing *why* throttling is occurring (PROCHOT, PPT, HBM_THM, VR_THM, SOCKET_THM). Available on Navi/MI1x/MI2x via `amd-smi metric --power` (gpu_metrics v1.3). Shows N/A on MI3x+ systems.

**Violation API**: Historical throttling data as percentages and active/not-active status from `amdsmi_get_violation_status()`. Available on MI3x+ (MI300X and newer) via `amd-smi metric --violation` or `amd-smi monitor --violation`. Returns N/A or max_uint on older ASICs.

**Sample rate**: Violations are sampled every 100ms (fastest rate the driver can update the metric cache).

**gpu_metrics versions**: Navi/MI1x/MI2x use gpu_metrics v1.3 (fixed layout with `indep_throttle_status`). MI3x+ uses gpu_metrics v1.9+ (dynamic layout). The available fields differ between these versions.

## GPU Architecture Support

`throttle_status` and the violation API target different GPU generations and rely on different gpu_metrics versions:

| Feature | Navi / MI1x / MI2x | MI3x+ (MI300X and newer) |
|---------|---------------------|--------------------------|
| gpu_metrics version | v1.3 (fixed layout) | v1.9+ (dynamic layout) |
| `THROTTLE_STATUS` in `metric --power` | THROTTLED / UNTHROTTLED | N/A |
| `metric --violation` / `monitor --violation` | N/A (unsupported) | Full violation details |
| Recommended throttle check | `amd-smi metric --power` | `amd-smi metric --violation` or `amd-smi monitor --violation` |

**Navi/MI1x/MI2x:** Use `amd-smi metric --power` to check `THROTTLE_STATUS`. The violation API (`amdsmi_get_violation_status()`) is not supported on these ASICs and returns N/A or max_uint.

**MI3x+ (MI300X and newer):** Use `amd-smi metric --violation` or `amd-smi monitor --violation` for detailed violation data (PROCHOT, PPT, Socket Thermal, VR Thermal, HBM Thermal). The `THROTTLE_STATUS` field in `metric --power` shows N/A on these ASICs.

## Common Questions

### What's the difference between throttle_status and violation API output in AMD SMI?

`throttle_status` shows **current real-time** throttling state as bit flags (PROCHOT, PPT, HBM_THM, etc.) and is available on **Navi/MI1x/MI2x** (gpu_metrics v1.3). The `violation` API shows **historical data** as percentages—how much time was spent throttled over a sampling period (100ms)—and is available on **MI3x+** (gpu_metrics v1.9+). Use `throttle_status` on older ASICs to see what's happening *right now*, and the violation API on MI3x+ to track detailed trends over time.

### Can you use the violations API to infer the same info provided in throttle_status?

No. The violations API (MI3x+ only) provides time-based percentages (PVIOL%, TVIOL%) showing *how much* throttling occurred. `throttle_status` (Navi/MI1x/MI2x) provides bit flags showing *whether* throttling is happening. These APIs target different GPU generations—see [GPU Architecture Support](#gpu-architecture-support).

### How can I monitor PROCHOT throttling using AMD SMI?

On **MI3x+**: Use `amd-smi metric --violation` or `amd-smi monitor --violation` to check `prochot_violation_activity` and `prochot_violation_status`. On **Navi/MI1x/MI2x**: Use `amdsmi_get_gpu_metrics_info()` and check the `throttle_status` field for `PROCHOT_GFX` bits. PROCHOT indicates emergency thermal throttling when the GPU hits critical temperature limits.

### What are the different types of thermal violations AMD SMI can detect?

AMD SMI detects: **TVIOL** (overall thermal violation %), **PROCHOT** (processor hot signal), **HBM_TVIOL** (high-bandwidth memory thermal), **VR_TVIOL** (voltage regulator thermal), and **SOCKET_THM** (socket-level thermal). These detailed violation types are available on **MI3x+ only** via the violation API. Older ASICs (Navi/MI1x/MI2x) return N/A or max_uint for these fields but provide `throttle_status` bit flags through `metric --power`.

### How do I check for power limit violations on AMD GPUs?

On **MI3x+**: Use `amdsmi_get_violation_status()` to get `per_ppt_pwr` (PVIOL%). Values >0% indicate time spent power-throttled. From CLI: `amd-smi metric --violation` or `amd-smi monitor --violation` displays PVIOL percentage. On **Navi/MI1x/MI2x**: Check `throttle_status` in `amd-smi metric --power` for PPT (package power tracking) throttling.

### What does HBM thermal throttling mean and how to detect it?

HBM (High-Bandwidth Memory) thermal throttling occurs when GPU memory overheats. On **MI3x+**: Detected via `per_hbm_thrm` (HBM_TVIOL%) and `active_hbm_thrm` in the violation API. On **Navi/MI1x/MI2x**: Check the `TEMP_MEM` bit in `throttle_status`. Detailed HBM violation percentages are only available on MI3x+.

### How to interpret violation status codes in AMD SMI?

Violation percentages: **0%** = no throttling (good), **>0%** = time spent throttled (higher = worse), **N/A or max_uint** = feature not supported on this GPU. The API returns both power_violation_pct (PVIOL) and thermal_violation_pct (TVIOL) as percentages.

### What triggers socket power violations in AMD GPUs?

Socket power violations (SOCKET_THM) occur when total platform power exceeds limits, not just GPU power. This is tracked separately from PPT (package power tracking) which monitors GPU-only power limits. Triggered by excessive power draw across the entire socket/system.

### How to monitor GPU hotspot temperature violations?

Monitor hotspot temperature with `amdsmi_get_temp_metric()` and correlate with TVIOL%. High TVIOL% combined with high hotspot temps (>95°C) indicates thermal throttling. Use `amd-smi metric --gpu all --temperature` to track temps alongside violation status.

### What's the relationship between throttling and performance degradation?

Throttling directly reduces GPU clock speeds to stay within power/thermal limits, which decreases performance. Higher violation percentages mean more time throttled = more performance loss. For example, 50% TVIOL means the GPU spent half the sampling period at reduced clocks.

### How to distinguish between thermal and power violations?

**PVIOL (power)** = hitting wattage limits, GPU draws too much power. **TVIOL (thermal)** = hitting temperature limits, GPU too hot. A GPU can have both simultaneously (e.g., 30% PVIOL + 20% TVIOL). On **MI3x+**: Check both percentages via `amdsmi_get_violation_status()` or `amd-smi metric --violation`. On **Navi/MI1x/MI2x**: Use `throttle_status` bit flags in `amd-smi metric --power` to see if PPT (power) or thermal throttling is active.

## Interpreting Violation Results

| Value | Meaning |
|-------|----------|
| 0% | No throttling - GPU operating normally |
| 1-25% | Light throttling - minor performance impact |
| 25-50% | Moderate throttling - noticeable performance loss |
| 50-100% | Heavy throttling - significant performance degradation |
| N/A or max_uint | Feature not supported on this GPU |

## Violation Status Fields

The `amdsmi_violation_status_t` struct (returned by `amdsmi_get_violation_status()`) provides three categories of data for each violation type. These fields are available on **MI3x+ only**; older ASICs return max_uint (unsupported).

| Category | Field prefix | Value type | Description |
|----------|-------------|------------|-------------|
| Accumulated counters | `acc_*` | uint64 | Raw counter incremented while violation is active |
| Violation status | `active_*` | uint8 (1/0) | Whether the violation is currently ACTIVE or NOT ACTIVE |
| Violation activity | `per_*` | uint64 (%) | Percentage of sampling period spent in violation (>0% = throttled) |

### Core violation types

| Violation type | Accumulated | Status | Activity | Description |
|----------------|------------|--------|----------|-------------|
| PROCHOT | `acc_prochot_thrm` | `active_prochot_thrm` | `per_prochot_thrm` | Processor hot — emergency thermal throttling at critical temperature |
| PPT (Power) | `acc_ppt_pwr` | `active_ppt_pwr` | `per_ppt_pwr` | Package Power Tracking — PVIOL; power consumption exceeds limits |
| Socket Thermal | `acc_socket_thrm` | `active_socket_thrm` | `per_socket_thrm` | Socket-level thermal — TVIOL; socket temperature exceeds limits |
| VR Thermal | `acc_vr_thrm` | `active_vr_thrm` | `per_vr_thrm` | Voltage regulator thermal throttling |
| HBM Thermal | `acc_hbm_thrm` | `active_hbm_thrm` | `per_hbm_thrm` | High Bandwidth Memory thermal throttling |

### Per-XCP/XCC violation types (gpu_metrics v1.8+)

These fields are 2D arrays indexed by `[XCP][XCC]` and require gpu_metrics v1.8 or newer:

| Violation type | Accumulated | Status | Activity | Description |
|----------------|------------|--------|----------|-------------|
| GFX Clock Below Host Limit (Power) | `acc_gfx_clk_below_host_limit_pwr` | `active_gfx_clk_below_host_limit_pwr` | `per_gfx_clk_below_host_limit_pwr` | GFX clock limited below host limit due to power |
| GFX Clock Below Host Limit (Thermal) | `acc_gfx_clk_below_host_limit_thm` | `active_gfx_clk_below_host_limit_thm` | `per_gfx_clk_below_host_limit_thm` | GFX clock limited below host limit due to thermal |
| GFX Clock Below Host Limit (Total) | `acc_gfx_clk_below_host_limit_total` | `active_gfx_clk_below_host_limit_total` | `per_gfx_clk_below_host_limit_total` | GFX clock limited below host limit for any reason |
| Low Utilization | `acc_low_utilization` | `active_low_utilization` | `per_low_utilization` | Low GPU utilization detected |

### Metadata fields

| Field | Type | Description |
|-------|------|-------------|
| `reference_timestamp` | uint64 | CPU timestamp in microseconds (µs) |
| `violation_timestamp` | uint64 | Violation time in nanoseconds (bare metal Linux) or milliseconds (host) |
| `acc_counter` | uint64 | Accumulation counter used for percentage calculations |

:::{note}
`max_uint64` (for uint64 fields) or `max_uint8` (for uint8 fields) indicates the feature is unsupported on the current ASIC. The original `acc_gfx_clk_below_host_limit`, `per_gfx_clk_below_host_limit`, and `active_gfx_clk_below_host_limit` fields are deprecated in favor of the per-XCP/XCC v1.8 variants above.
:::

### `throttle_status` bit flags (Navi/MI1x/MI2x)

On older ASICs using gpu_metrics v1.3, the `indep_throttle_status` field provides real-time throttle state as bit flags. The bit definitions come from the kernel driver (`amdgpu_smu.h`):

| Category | Bit range | Key flags |
|----------|-----------|-----------|
| Power throttlers | 0–7 | PPT0, PPT1, SPL, FPPT, SPPT |
| Current throttlers | 16–23 | TDC_GFX, TDC_SOC, TDC_MEM, EDC_CPU, EDC_GFX |
| Temperature throttlers | 32–47 | TEMP_GPU, TEMP_MEM (HBM_THM), TEMP_HOTSPOT, TEMP_SOC (SOCKET_THM), TEMP_VR_GFX (VR_THM), PROCHOT_GFX |

## Design Considerations

This section documents key design decisions made during the implementation:

| Design Consideration(s) | Concern | Options | Final Decision Outcome |
|------------------------|---------|---------|------------------------|
| Provide T/F, Active/Not Active<br/>Prior ASIC support - how do we provide no support?<br/>At what % means throttled?<br/>Is there a way to provide Active/Not Active (regardless of ASIC support)? | MI3x is the only ASIC that will provide the PROCHOT/PPT/SOCKET_THM/VR_THM/HBM_THM details<br/>For Asics that support - what percentage means throttled? | a) PVIOL/TVIOL % = % or N/A<br/>b) PVIOL/TVIOL = T/F/N/A or Active/Not Active/N/A (% > 0 = Active or T; % == 0, Not Active or False; Not Supported = Field is max_uint)<br/>c) Provide Throttle Status as a fallback if (a/b) above is N/A<br/>d) Provide All 3 (a,b,c) through API/CLI | **Decision:** Provide All 3 (a,b,c) through API/CLI<br/>Return not supported, recommend gpu_metrics (throttle_status) to view if the device is in a throttled state<br/>Greater than 0% means throttled |
| Service/Daemon Add | Without adding a service/daemon, user would have to manually query over and over in a limited time frame (limited to AMD SMI's CLI fastest query period - currently 1 second). | a) Add a service/daemon<br/>b) Manually query API | **Decision:** Manually query API |
| PMFW violation accumulator counter's update speed | ms, but Nvidia provides at a ns speed | Request PMFW for MI308/MI325 to increase the counter updates | **Decision:** No - we are limited to 2 samples, once per 100 ms (see sample rate below) |
| Sample rate | What is the fastest rate? | N/A | **Decision:** 100 ms - The fastest rate driver can update the metric cache |

## NVML API Compatibility

AMD SMI provides NVML-compatible violation monitoring. The table below shows the mapping between NVML and AMD SMI concepts:

| NVML Concept | AMD SMI Equivalent | Notes |
|--------------|-------------------|-------|
| `nvmlDeviceGetViolationStatus()` | `amdsmi_get_violation_status()` | Returns violation status and timestamp |
| `PVIOL` (Power Violation) | `power_violation_pct` | Power throttling events tracked as percentage |
| `TVIOL` (Thermal Violation) | `thermal_violation_pct` | Thermal throttling events tracked as percentage |
| Percentage format (%) | Percentage format (%) | Compatible format (0-100%) |
| `Not Supported` indicator | `N/A` or max_uint | Indicates feature unavailability |

## Usage

AMD SMI provides tools to programmatically monitor GPU violations and throttling events.

:::::{tab-set}

::::{tab-item} C/C++

The AMD SMI library provides APIs to query violation status.

```c
#include "amd_smi/amdsmi.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    amdsmi_status_t ret;
    uint32_t socket_count = 0;
    uint32_t gpu_number = 0;

    // Initialize AMD SMI for GPUs
    ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to initialize AMD SMI\n");
        return 1;
    }

    // Get socket count
    ret = amdsmi_get_socket_handles(&socket_count, NULL);
    if (ret != AMDSMI_STATUS_SUCCESS || socket_count == 0) {
        fprintf(stderr, "Failed to get socket count\n");
        amdsmi_shut_down();
        return 1;
    }

    amdsmi_socket_handle *sockets = malloc(socket_count * sizeof(amdsmi_socket_handle));
    if (!sockets) {
        amdsmi_shut_down();
        return 1;
    }
    ret = amdsmi_get_socket_handles(&socket_count, sockets);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        free(sockets);
        amdsmi_shut_down();
        return 1;
    }

    // For each socket, get processor handles and query violation status
    for (uint32_t i = 0; i < socket_count; i++) {
        uint32_t device_count = 0;
        ret = amdsmi_get_processor_handles(sockets[i], &device_count, NULL);
        if (ret != AMDSMI_STATUS_SUCCESS || device_count == 0)
            continue;

        amdsmi_processor_handle *processor_handles =
            malloc(device_count * sizeof(amdsmi_processor_handle));
        if (!processor_handles)
            continue;
        ret = amdsmi_get_processor_handles(sockets[i], &device_count, processor_handles);
        if (ret != AMDSMI_STATUS_SUCCESS) {
            free(processor_handles);
            continue;
        }

        for (uint32_t j = 0; j < device_count; j++) {
            amdsmi_violation_status_t violation = {0};
            ret = amdsmi_get_violation_status(processor_handles[j], &violation);
            if (ret == AMDSMI_STATUS_SUCCESS) {
                // PVIOL: per_ppt_pwr; TVIOL: per_socket_thrm (and other thermal % fields)
                printf("GPU %u - PVIOL (PPT): %lu%%, TVIOL (socket): %lu%%\n",
                       gpu_number, (unsigned long)violation.per_ppt_pwr,
                       (unsigned long)violation.per_socket_thrm);
            }
            gpu_number++;
        }
        free(processor_handles);
    }

    free(sockets);
    amdsmi_shut_down();
    return 0;
}
```

::::

::::{tab-item} Python

See related APIs for violations monitoring:

```python
#!/usr/bin/env python3
import amdsmi

try:
    amdsmi.amdsmi_init()
    processors = amdsmi.amdsmi_get_processor_handles()
    
    for i, processor in enumerate(processors):
        # Get violation percentages
        violation_status = amdsmi.amdsmi_get_violation_status(processor)
        print(f"GPU {i} - PVIOL: {violation_status['power_violation_pct']:.2f}%")
        print(f"GPU {i} - TVIOL: {violation_status['thermal_violation_pct']:.2f}%")
        
        # Get detailed throttle status
        metrics = amdsmi.amdsmi_get_gpu_metrics_info(processor)
        if metrics['throttle_status'] > 0:
            print(f"GPU {i} - Active throttling: 0x{metrics['throttle_status']:x}")
        
finally:
    amdsmi.amdsmi_shutdown()
```

::::

::::{tab-item} amd-smi CLI

Monitor GPU violations using the CLI tool:

```shell
# MI3x+ (MI300X and newer): Check detailed violation status
amd-smi metric --violation

# MI3x+: Monitor violations in real time
amd-smi monitor --violation

# MI3x+: Continuous monitoring (update every 2 seconds)
watch -n 2 'amd-smi monitor --violation'

# Navi/MI1x/MI2x: Check throttle status via power metrics
amd-smi metric --power

# All architectures: Monitor temperatures alongside power
amd-smi metric --gpu all --power --temperature
```

::::

:::::

## Troubleshooting

### High PVIOL (Power Violations)?

- Check power limit settings with `amdsmi_get_power_cap()`
- Verify adequate PSU capacity for your system
- Consider reducing workload intensity or power limits
- Monitor with: `amd-smi metric --gpu all --power`

### High TVIOL (Thermal Violations)?

- Check cooling system (fans, airflow)
- Verify thermal paste application
- Monitor ambient temperature
- Check for dust buildup in coolers
- Use: `amd-smi metric --gpu all --temperature`

### Getting N/A or max_uint values?

- **For violation fields (`metric --violation`) returning N/A:** The violation API is only supported on MI3x+ (MI300X and newer). On older ASICs (Navi/MI1x/MI2x), use `amd-smi metric --power` and check `THROTTLE_STATUS` instead.
- **For `THROTTLE_STATUS` in `metric --power` showing N/A:** This field is available on Navi/MI1x/MI2x (gpu_metrics v1.3) but not on MI3x+. On MI3x+, use `amd-smi metric --violation` or `amd-smi monitor --violation` instead.
- Check your ASIC generation with `amdsmi_get_gpu_asic_info()`

## See Also

**Related AMD SMI APIs:**

- `amdsmi_get_violation_status()` - Get violation percentages
- `amdsmi_get_gpu_metrics_info()` - Get throttle_status and detailed metrics
- `amdsmi_get_temp_metric()` - Monitor GPU temperatures
- `amdsmi_get_power_cap()` - Check power limits
- `amdsmi_get_gpu_activity()` - Monitor GPU utilization
- `amdsmi_get_gpu_asic_info()` - Check ASIC capabilities
- `amdsmi_get_gpu_bdf_id()` - Identify GPU device

**Related Topics:**

- GPU Throttling
- Performance Monitoring
- Power Management
- Thermal Management
- NVML Compatibility

**External Documentation:**

- [AMD SMI API Documentation](https://github.com/ROCm/amdsmi)
- [ROCm Documentation](https://rocm.docs.amd.com/)

## Further Reading

- [AMD SMI GitHub Repository](https://github.com/ROCm/amdsmi)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [GPU Metrics Information](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
