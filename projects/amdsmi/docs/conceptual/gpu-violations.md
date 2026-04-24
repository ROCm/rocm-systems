---
myst:
  html_meta:
    "description lang=en": "AMD SMI GPU violation monitoring and throttling status."
    "keywords": "gpu, violations, throttling, pviol, tviol, power, thermal, performance, throttle_status, gpu_metrics, prochot, ppt, hbm, mi300x"
---

# GPU violations

GPU violations monitoring in AMD SMI tracks throttling events caused by power or thermal limits. When your GPU throttles, performance decreases to protect the hardware from damage due to overheating or excessive power draw. AMD SMI provides APIs to monitor violation percentages and identify the specific causes of throttling, enabling system administrators and developers to maintain GPU health and optimize performance.

:::{note}
NVML users: the closest equivalent to `nvmlDeviceGetViolationStatus()` is `amdsmi_get_violation_status()`.
See [GPU monitoring](/doxygen/docBin/html/group__tagGPUMonitor) for more information.

| nvidia-smi command | amd-smi equivalent | Notes |
|--------------------|--------------------|-------|
| `nvidia-smi -q -d PERFORMANCE` | `amd-smi metric --violation` | Instantaneous violation status; MI3x+ only |
| `nvidia-smi dmon -s p` | `amd-smi monitor --violation` | Continuous violation monitoring; MI3x+ only |
| `nvidia-smi -q -d CLOCK` | `amd-smi metric --clock` | Current clock frequencies |
| `nvidia-smi -q -d POWER` | `amd-smi metric --power` | Power usage and throttle status (Navi/MI1x/MI2x) |
:::


## Key concepts

**PVIOL (Power Violation)**: Percentage of time the GPU throttled due to power limits. This occurs when the GPU's power consumption exceeds safe limits, triggering power throttling to stay within the power budget.

**TVIOL (Thermal Violation)**: Percentage of time the GPU throttled due to temperature limits. This happens when GPU temperatures exceed safe operating thresholds, causing the GPU to reduce performance to cool down.

**throttle_status** (uint32_t, gpu_metrics v1.3): Simple THROTTLED / UNTHROTTLED indicator — non-zero means the GPU is throttling for *any* reason. Available on Navi/MI1x/MI2x via `amd-smi metric --power`. Shows N/A on MI3x+ systems. Use this when you only need to know *whether* throttling is occurring.

**indep_throttle_status** (uint64_t, gpu_metrics v1.3): Per-reason bit flags (PROCHOT_GFX, TDC_GFX, TEMP_MEM, TEMP_SOC, TEMP_VR_GFX, etc.) — ASIC-independent encoding that shows *why* throttling is occurring. Available on Navi/MI1x/MI2x. Shows N/A on MI3x+ systems. Use this when you need to identify the specific cause of throttling. (Not used in amd-smi CLI output but available via API)

**Violation API**: Historical throttling data as percentages and active/not-active status from `amdsmi_get_violation_status()`. Available on MI3x+ (MI300X and newer) via `amd-smi metric --violation` or `amd-smi monitor --violation`. Returns N/A or max_uint on older ASICs.

**Sample rate**: Violations are sampled every 100ms — the fastest rate the driver can update the metric cache. Set `AMDSMI_GPU_METRICS_CACHE_MS=0` to disable AMD SMI's internal cache and let the driver control when the cache updates. See [AMD SMI C++ library usage](../how-to/amdsmi-cpp-lib.md) for environment variable details.

**gpu_metrics versions**: Navi/MI1x/MI2x use gpu_metrics v1.3 (fixed layout with `indep_throttle_status`). MI3x+ requires at minimum gpu_metrics v1.8, which introduced per-XCP/XCC violation fields. gpu_metrics v1.9+ adds a fully dynamic layout. The available fields differ between these versions.

## GPU architecture support

`throttle_status` and the violation API target different GPU generations and rely on different gpu_metrics versions:

| Feature | Navi / MI1x / MI2x | MI3x+ (MI300X and newer) |
|---------|---------------------|--------------------------|
| gpu_metrics version | v1.3 (fixed layout) | v1.8+ (per-XCP/XCC fields); v1.9+ (full dynamic layout) |
| `THROTTLE_STATUS` in `metric --power` | THROTTLED / UNTHROTTLED | N/A |
| `metric --violation` / `monitor --violation` | N/A (unsupported) | Full violation details |
| Recommended throttle check | `amd-smi metric --power` | `amd-smi metric --violation` or `amd-smi monitor --violation` |

**Navi/MI1x/MI2x:** Use `amd-smi metric --power` to check `THROTTLE_STATUS`. The violation API (`amdsmi_get_violation_status()`) is not supported on these ASICs and returns N/A or max_uint.

**MI3x+ (MI300X and newer):** Use `amd-smi metric --violation` or `amd-smi monitor --violation` for detailed violation data (PROCHOT, PPT, Socket Thermal, VR Thermal, HBM Thermal). The `THROTTLE_STATUS` field in `metric --power` shows N/A on these ASICs.

## Common questions

### What's the difference between throttle_status and violation API output in AMD SMI?

`throttle_status` shows **current real-time** throttling state as bit flags (PROCHOT, PPT, HBM_THM, etc.) and is available on **Navi/MI1x/MI2x** (gpu_metrics v1.3). The `violation` API shows **historical data** as percentages—how much time was spent throttled over a sampling period (100ms)—and is available on **MI3x+** (gpu_metrics v1.9+). Use `throttle_status` on older ASICs to see what's happening *right now*, and the violation API on MI3x+ to track detailed trends over time.

### Can you use the violations API to infer the same info provided in throttle_status?

No. The violations API (MI3x+ only) provides time-based percentages (PVIOL%, TVIOL%) showing *how much* throttling occurred. `throttle_status` (Navi/MI1x/MI2x) provides bit flags showing *whether* throttling is happening. These APIs target different GPU generations—see [](#gpu-architecture-support).

### How can I monitor PROCHOT throttling using AMD SMI?

On **MI3x+**: Use `amd-smi metric --violation` or `amd-smi monitor --violation` to check `prochot_violation_activity` and `prochot_violation_status`. On **Navi/MI1x/MI2x**: Use `amdsmi_get_gpu_metrics_info()` and check the `indep_throttle_status` field for `PROCHOT_GFX` bits. PROCHOT indicates emergency thermal throttling when the GPU hits critical temperature limits.

### What are the different types of thermal violations AMD SMI can detect?

AMD SMI detects: **TVIOL** (overall thermal violation %), **PROCHOT** (processor hot signal), **HBM_TVIOL** (high-bandwidth memory thermal), **VR_TVIOL** (voltage regulator thermal), and **SOCKET_THM** (socket-level thermal). These detailed violation types are available on **MI3x+ only** via the violation API. Older ASICs (Navi/MI1x/MI2x) return N/A or max_uint for these fields but provide `throttle_status` bit flags through `metric --power`.

### How do I check for power limit violations on AMD GPUs?

On **MI3x+**: Use `amdsmi_get_violation_status()` to get `per_ppt_pwr` (PVIOL%). Values >0% indicate time spent power-throttled. From CLI: `amd-smi metric --violation` or `amd-smi monitor --violation` displays PVIOL percentage. On **Navi/MI1x/MI2x**: Check `throttle_status` in `amd-smi metric --power` for PPT (package power tracking) throttling.

### What does HBM thermal throttling mean and how to detect it?

HBM (High-Bandwidth Memory) thermal throttling occurs when GPU memory overheats. On **MI3x+**: Detected via `per_hbm_thrm` (HBM_TVIOL%) and `active_hbm_thrm` in the violation API. On **Navi/MI1x/MI2x**: Check the `TEMP_MEM` bit in `indep_throttle_status`. Detailed HBM violation percentages are only available on MI3x+.

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

## Interpreting violation results

| Value | Meaning |
|-------|----------|
| 0% | No throttling - GPU operating normally |
| 1-25% | Light throttling - minor performance impact |
| 25-50% | Moderate throttling - noticeable performance loss |
| 50-100% | Heavy throttling - significant performance degradation |
| N/A or max_uint | Feature not supported on this GPU |

## Violation status fields

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

:::{note}
**How `GFXCLK_*` and `LOW_UTIL*` differ from core PVIOL/TVIOL fields:**

- **Scope**: These are per-XCP (Compute Partition) × per-XCC (Compute Complex) 2D arrays, not socket-level aggregates like PVIOL/TVIOL.
- **What they measure**: `GFXCLK_*` tracks when the GFX clock is forced *below a host-set clock limit* due to power (`_pwr`) or thermal (`_thm`) reasons. `LOW_UTIL*` tracks periods of low GPU utilization — a clock reduction cause unrelated to power or thermal limits.
- **Availability**: Require gpu_metrics v1.8 or newer; return max_uint on earlier drivers/ASICs.
:::

### Metadata fields

| Field | Type | Description |
|-------|------|-------------|
| `reference_timestamp` | uint64 | CPU timestamp in microseconds (µs) |
| `violation_timestamp` | uint64 | Violation time in nanoseconds (bare metal Linux) or milliseconds (host) |
| `acc_counter` | uint64 | Accumulation counter used for percentage calculations |

:::{note}
`max_uint64` (for uint64 fields) or `max_uint8` (for uint8 fields) indicates the feature is unsupported on the current ASIC. The original `acc_gfx_clk_below_host_limit`, `per_gfx_clk_below_host_limit`, and `active_gfx_clk_below_host_limit` fields are deprecated in favor of the per-XCP/XCC v1.8 variants above.
:::

### `throttle_status` and `indep_throttle_status` bit flags (Navi/MI1x/MI2x)

Both fields are present in gpu_metrics v1.3 on Navi/MI1x/MI2x ASICs. `throttle_status` (uint32_t) indicates *whether* the GPU is throttling (non-zero = throttled). `indep_throttle_status` (uint64_t) encodes *why* via per-reason bit flags defined in the kernel driver.

The canonical bit definitions are in the `SMU_THROTTLER_*` enum inside
[`drivers/gpu/drm/amd/pm/swsmu/inc/amdgpu_smu.h`](https://github.com/ROCm/amdgpu/blob/master/drivers/gpu/drm/amd/pm/swsmu/inc/amdgpu_smu.h)
in the ROCm AMDGPU kernel driver. AMD SMI passes the raw `uint64_t` value through to the caller without interpreting the bits, so refer to that header for the authoritative bit-position-to-flag mapping. The table below summarizes the key ranges:

| Category | Bit range | Key flags |
|----------|-----------|-----------|
| Power throttlers | 0–7 | PPT0, PPT1, SPL, FPPT, SPPT |
| Current throttlers | 16–23 | TDC_GFX, TDC_SOC, TDC_MEM, EDC_CPU, EDC_GFX |
| Temperature throttlers | 32–47 | TEMP_GPU, TEMP_MEM (HBM_THM), TEMP_HOTSPOT, TEMP_SOC (SOCKET_THM), TEMP_VR_GFX (VR_THM), PROCHOT_GFX |

## Usage

AMD SMI provides tools to programmatically monitor GPU violations and throttling events.

:::::{tab-set}

::::{tab-item} C/C++

The AMD SMI library provides APIs to query violation status.

**Related AMD SMI APIs:**

- `amdsmi_get_violation_status()` - Get violation percentages
- `amdsmi_get_gpu_metrics_info()` - Get throttle_status and detailed metrics
- `amdsmi_get_temp_metric()` - Monitor GPU temperatures
- `amdsmi_get_power_cap_info()` - Check power limits
- `amdsmi_get_gpu_activity()` - Monitor GPU utilization
- `amdsmi_get_gpu_asic_info()` - Check ASIC capabilities
- `amdsmi_get_gpu_bdf_id()` - Identify GPU device

See [`example/amd_smi_drm_example.cc`](../../example/amd_smi_drm_example.cc)
for a complete working example.

```cpp
amdsmi_violation_status_t status = {};
amdsmi_status_t ret = amdsmi_get_violation_status(processor_handle, &status);
if (ret == AMDSMI_STATUS_SUCCESS) {
    // MI3x+: access per_ppt_pwr (PVIOL%), per_socket_thrm (TVIOL%),
    // active_prochot_thrm, active_hbm_thrm, etc.
    // Max uint64/uint8 sentinel values indicate unsupported fields (N/A).
} else if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
    // Navi/MI1x/MI2x: violation API not supported.
    // Use amdsmi_get_gpu_metrics_info() and check throttle_status instead.
}
```

::::

::::{tab-item} Python

See related APIs:

- [](/reference/amdsmi-py-api.md#amdsmi_get_violation_status)
- [](/reference/amdsmi-py-api.md#amdsmi_get_gpu_metrics_info)
- [](/reference/amdsmi-py-api.md#amdsmi_get_temp_metric)
- [](/reference/amdsmi-py-api.md#amdsmi_get_power_cap_info)
- [](/reference/amdsmi-py-api.md#amdsmi_get_gpu_activity)
- [](/reference/amdsmi-py-api.md#amdsmi_get_gpu_asic_info)
- [](/reference/amdsmi-py-api.md#amdsmi_get_gpu_bdf_id)

See
[`example/amd_smi_violation_example.py`](../../example/amd_smi_violation_example.py)
for a complete working example.

```python
import amdsmi

amdsmi.amdsmi_init(amdsmi.AmdSmiInitFlags.INIT_AMD_GPUS)
try:
    for processor in amdsmi.amdsmi_get_processor_handles():
        try:
            v = amdsmi.amdsmi_get_violation_status(processor)
            # MI3x+: access v['per_ppt_pwr'] (PVIOL%), v['per_socket_thrm'] (TVIOL%),
            # v['active_prochot_thrm'], v['active_hbm_thrm'], etc.
            # 'N/A' indicates unsupported fields on this ASIC.
        except amdsmi.AmdSmiLibraryException as e:
            if e.err_code == amdsmi.AmdSmiRetCode.STATUS_NOT_SUPPORTED:
                # Navi/MI1x/MI2x: violation API not supported.
                # Fall back to gpu_metrics throttle_status for a basic
                # THROTTLED / UNTHROTTLED indicator.
                m = amdsmi.amdsmi_get_gpu_metrics_info(processor)
                ts = m.get("throttle_status", "N/A")
                if ts == "N/A":
                    print("throttle_status: N/A")
                elif ts:
                    print("throttle_status: THROTTLED")
                else:
                    print("throttle_status: UNTHROTTLED")
            else:
                raise
finally:
    amdsmi.amdsmi_shut_down()
```

::::

::::{tab-item} amd-smi CLI

Monitor GPU violations using the CLI tool:

```shell
# MI3x+ (MI300X and newer): Check detailed violation status
amd-smi metric --violation

# MI3x+: Monitor violations in real time
amd-smi monitor --violation

# MI3x+: Monitor power, temp, GFX clock, and utilization violations every second
# AMDSMI_GPU_METRICS_CACHE_MS=0 disables the 100ms cache so the driver controls updates
AMDSMI_GPU_METRICS_CACHE_MS=0 amd-smi monitor -ptV --watch 1

# Navi/MI1x/MI2x: Check throttle status via power metrics
# (MI3x+ shows N/A here; use metric --violation or monitor --violation instead)
amd-smi metric --power

# All architectures: Monitor temperatures alongside power
amd-smi metric --gpu all --power --temperature
```

::::

:::::

## Troubleshooting

### High PVIOL (Power Violations)?

- Check power limit settings with `amdsmi_get_power_cap_info()`
- View static power cap details (default, min, max): `amd-smi static --limit`
- Monitor live power consumption: `amd-smi monitor --power`
- Verify adequate PSU capacity for your system
- Consider reducing workload intensity or power limits
- Monitor with: `amd-smi metric --gpu all --power`

:::{note}
`amd-smi static --limit` shows power cap thresholds and thermal shutdown/slowdown limits. If your GPU is hitting these limits, it may throttle to stay within them, causing PVIOL/TVIOL. Adjusting power limits or improving cooling can help reduce Power or thermal related violations.
:::

### High TVIOL (Thermal Violations)?

- Check cooling system (fans, airflow)
- Verify thermal paste application
- Monitor ambient temperature
- Check for dust buildup in coolers
- Use: `amd-smi metric --gpu all --temperature`

### Getting N/A or max_uint values?

- **For violation fields (`metric --violation`) returning N/A:** The violation API is only supported on MI3x+ (MI300X and newer). On older ASICs (Navi/MI1x/MI2x), use `amd-smi metric --power` and check `THROTTLE_STATUS` instead.
- **For `THROTTLE_STATUS` in `metric --power` showing N/A:** This field is available on Navi/MI1x/MI2x (gpu_metrics v1.3) but not on MI3x+. On MI3x+, use `amd-smi metric --violation` or `amd-smi monitor --violation` instead.
- Check your ASIC generation with `amdsmi_get_gpu_asic_info()` or `amd-smi static --asic`

## Adjusting clock limits (MI3x+)

Some MI3x+ variants support adjusting the Graphics clock (SCLK) and memory clock (MCLK) min/max limits, which can help manage power violations by capping clock speeds before the hardware throttles.

```shell
# View available clock limit options
amd-smi set -h

# View current min/max clock ranges (Checks capabilities for your specific model)
amd-smi static --clock

# Set clock limits (--clk-limit / -L): adjust sclk or mclk min/max
# Notes:
#    - Not all MI3x+ models support adjusting clock limits for both SCLK and MCLK; check your model's capabilities with `amd-smi static --clock`
#    - Recommend to set max limits, then adjust min limits.
sudo amd-smi set --clk-limit <CLK_TYPE> <LIM_TYPE> <VALUE>

# Confirm changes took place:
amd-smi metric --clock

# Reset clocks back to their default state
sudo amd-smi reset --clocks
```

Lowering the SCLK maximum reduces peak power draw, which can reduce PVIOL percentage at the cost of peak compute throughput. See `amd-smi set -h` for the full list of supported options for your hardware.

## Further reading

- [GPU Power/Thermal Controls and Monitoring (Linux kernel)](https://docs.kernel.org/gpu/amdgpu/thermal.html#gpu-metrics)
