---
myst:
  html_meta:
    "description lang=en": "AMD SMI GPU violation monitoring and throttling status."
    "keywords": "gpu, violations, throttling, pviol, tviol, power, thermal, performance"
---

# GPU Violations

GPU violations monitoring in AMD SMI tracks throttling events caused by power or thermal limits. When your GPU throttles, performance decreases to protect the hardware from damage due to overheating or excessive power draw. AMD SMI provides APIs to monitor violation percentages and identify the specific causes of throttling, enabling system administrators and developers to maintain GPU health and optimize performance.

## Key Concepts

**PVIOL (Power Violation)**: Percentage of time the GPU throttled due to power limits. This occurs when the GPU's power consumption exceeds safe limits, triggering power throttling to stay within the power budget.

**TVIOL (Thermal Violation)**: Percentage of time the GPU throttled due to temperature limits. This happens when GPU temperatures exceed safe operating thresholds, causing the GPU to reduce performance to cool down.

**throttle_status**: Real-time bit flags showing *why* throttling is occurring (PROCHOT, PPT, HBM_THM, VR_THM, SOCKET_THM). This provides detailed information about the specific throttling cause.

**Sample rate**: Violations are sampled every 100ms (fastest rate the driver can update the metric cache).

**MI300X**: Only MI300X and newer ASICs provide full violation details; older ASICs return N/A or max_uint for unsupported features.

## Common Questions

### What's the difference between throttle_status and violation API output in AMD SMI?

`throttle_status` shows **current real-time** throttling state as bit flags (PROCHOT, PPT, HBM_THM, etc.). The `violation` API shows **historical data** as percentages—how much time was spent throttled over a sampling period (100ms). Use `throttle_status` to see what's happening *right now*, and violations to track trends over time.

### Can you use the violations API to infer the same info provided in throttle_status?

No. The violations API provides time-based percentages (PVIOL%, TVIOL%) showing *how much* throttling occurred. `throttle_status` provides detailed bit flags showing *why* throttling is happening (PROCHOT, PPT, SOCKET_THM, VR_THM, HBM_THM). You need both APIs for complete monitoring.

### How can I monitor PROCHOT throttling using AMD SMI?

Use `amdsmi_get_gpu_metrics_info()` and check the `throttle_status` field for PROCHOT bits. From CLI: `amd-smi monitor --violation` shows real-time PROCHOT_TVIOL status. PROCHOT indicates emergency thermal throttling when the GPU hits critical temperature limits.

### What are the different types of thermal violations AMD SMI can detect?

AMD SMI detects: **TVIOL** (overall thermal violation %), **PROCHOT** (processor hot signal), **HBM_TVIOL** (high-bandwidth memory thermal), **VR_TVIOL** (voltage regulator thermal), and **SOCKET_THM** (socket-level thermal). MI300X and newer ASICs support all types; older ASICs show N/A or max_uint for unsupported metrics.

### How do I check for power limit violations on AMD GPUs?

Use `amdsmi_get_violation_status()` to get `power_violation_pct` (PVIOL). Values >0% indicate time spent power-throttled. From CLI: `amd-smi monitor --violation` displays PVIOL percentage. You can also check `throttle_status` for PPT (package power tracking) bits.

### What does HBM thermal throttling mean and how to detect it?

HBM (High-Bandwidth Memory) thermal throttling occurs when GPU memory overheats. Detected via `HBM_THM` bit in `throttle_status` or `HBM_TVIOL` in violation monitoring. Only available on MI300X and newer ASICs—older GPUs return N/A.

### How to interpret violation status codes in AMD SMI?

Violation percentages: **0%** = no throttling (good), **>0%** = time spent throttled (higher = worse), **N/A or max_uint** = feature not supported on this GPU. The API returns both power_violation_pct (PVIOL) and thermal_violation_pct (TVIOL) as percentages.

### What triggers socket power violations in AMD GPUs?

Socket power violations (SOCKET_THM) occur when total platform power exceeds limits, not just GPU power. This is tracked separately from PPT (package power tracking) which monitors GPU-only power limits. Triggered by excessive power draw across the entire socket/system.

### How to monitor GPU hotspot temperature violations?

Monitor hotspot temperature with `amdsmi_get_temp_metric()` and correlate with TVIOL%. High TVIOL% combined with high hotspot temps (>95°C) indicates thermal throttling. Use `amd-smi metric --gpu all --temperature` to track temps alongside violation status.

### What's the relationship between throttling and performance degradation?

Throttling directly reduces GPU clock speeds to stay within power/thermal limits, which decreases performance. Higher violation percentages mean more time throttled = more performance loss. For example, 50% TVIOL means the GPU spent half the sampling period at reduced clocks.

### How to distinguish between thermal and power violations?

**PVIOL (power)** = hitting wattage limits, GPU draws too much power. **TVIOL (thermal)** = hitting temperature limits, GPU too hot. A GPU can have both simultaneously (e.g., 30% PVIOL + 20% TVIOL). Check both percentages in `amdsmi_get_violation_status()`.

## Interpreting Violation Results

| Value | Meaning |
|-------|----------|
| 0% | No throttling - GPU operating normally |
| 1-25% | Light throttling - minor performance impact |
| 25-50% | Moderate throttling - noticeable performance loss |
| 50-100% | Heavy throttling - significant performance degradation |
| N/A or max_uint | Feature not supported on this GPU |

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
#include "amdsmi/amdsmi.h"
#include <stdio.h>

int main() {
    amdsmi_status_t ret;
    uint32_t num_devices = 0;
    
    // Initialize AMD SMI
    ret = amdsmi_init(AMDSMI_INIT_ALL_PROCESSORS);
    if (ret != AMDSMI_STATUS_SUCCESS) {
        fprintf(stderr, "Failed to initialize AMD SMI\n");
        return 1;
    }
    
    // Get device count and processor handles
    ret = amdsmi_get_processor_handles(&num_devices, NULL);
    amdsmi_processor_handle* processors = malloc(num_devices * sizeof(amdsmi_processor_handle));
    ret = amdsmi_get_processor_handles(&num_devices, processors);
    
    // Monitor violations
    for (uint32_t i = 0; i < num_devices; i++) {
        amdsmi_violation_status_t violation;
        ret = amdsmi_get_violation_status(processors[i], &violation);
        if (ret == AMDSMI_STATUS_SUCCESS) {
            printf("GPU %d - PVIOL: %.2f%%, TVIOL: %.2f%%\n", 
                   i, violation.power_violation_pct, violation.thermal_violation_pct);
            
            // Check detailed throttle status
            amdsmi_gpu_metrics_t metrics;
            ret = amdsmi_get_gpu_metrics_info(processors[i], &metrics);
            if (ret == AMDSMI_STATUS_SUCCESS && metrics.throttle_status > 0) {
                printf("GPU %d - Active throttling detected (status: 0x%x)\n", 
                       i, metrics.throttle_status);
            }
        }
    }
    
    free(processors);
    amdsmi_shutdown();
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
# Check violation status
amd-smi metric --gpu all --power --temperature

# Monitor violations specifically
amd-smi monitor --violation

# Continuous monitoring (update every 2 seconds)
watch -n 2 'amd-smi monitor --violation'

# Monitor all metrics including throttling
amd-smi monitor --gpu all
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

- Feature may not be supported on your GPU
- MI300X and newer ASICs have full violation support
- Use `throttle_status` from `gpu_metrics` as fallback
- Check ASIC support with `amdsmi_get_gpu_asic_info()`

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
