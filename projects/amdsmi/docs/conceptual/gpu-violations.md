---
myst:
  html_meta:
    "description lang=en": "AMD SMI GPU violation monitoring and throttling status."
    "keywords": "gpu, violations, throttling, pviol, tviol, power, thermal, performance, throttle_status, gpu_metrics, prochot, ppt, hbm, mi300x"
---

# GPU violations

GPU violations monitoring in AMD SMI tracks throttling events caused by power or thermal limits. When your GPU throttles, performance decreases to protect the hardware from damage due to overheating or excessive power draw. AMD SMI provides APIs to monitor violation percentages and identify the specific causes of throttling, enabling system administrators and developers to maintain GPU health and optimize performance.

## GPU architecture support

AMD SMI exposes two distinct throttling APIs -- which one you use depends on your GPU generation:

- On **Instinct MI300 Series and newer data center GPUs** (gpu_metrics v1.6+):

  Use {c:func}`amdsmi_get_violation_status` or the CLI (`amd-smi monitor --violation`, `amd-smi metric --violation`). This API reports throttling as time-based percentages, active status flags, and accumulated counters for each violation type — useful for historical and trend-oriented monitoring. See [Interpreting violations API results](#interpreting-violations-api-results) for details.

  On these GPUs, `throttle_status` in `amd-smi metric --power` reports N/A.

- On **Radeon (Navi) and Instinct MI100/MI200 Series GPUs** (gpu_metrics v1.3):

  Use {c:func}`amdsmi_get_gpu_metrics_info` or `amd-smi metric --power` to check `throttle_status` (throttled/unthrottled) and `indep_throttle_status` (per-reason bit flags such as `PROCHOT_GFX`, `TDC_GFX`, `TEMP_MEM`). These indicate whether throttling is happening right now, not how much over time. See [Interpreting throttle_status results](#interpreting-throttle_status-results) for details.

  On these GPUs, the violations API and `amd-smi [metric/monitor] --violation` return N/A or max_uint.

The two mechanisms aren't interchangeable: the violations API measures *how much* throttling occurred (PVIOL%, TVIOL%); `throttle_status` measures *whether* it's happening now.
Both can help surface the same root causes, but the violations API gives you finer-grained trend data.

:::{note}
MI300 Series and newer use [gpu_metrics](https://docs.kernel.org/gpu/amdgpu/thermal.html#gpu-metrics) v1.8+, which introduced per-XCP and per-XCC violation fields (v1.9 adds a fully dynamic layout). Older GPUs use gpu_metrics v1.3 with a fixed layout.
:::

## Key concepts and common tasks

Understanding how violations are measured, what they mean for GPU performance, and how to act on them helps you diagnose throttling issues effectively.

### Throttling and performance degradation

Throttling directly reduces GPU clock speeds to stay within power and thermal limits, which decreases performance. Higher violation percentages mean more time throttled, which translates to more performance loss.

For example, on Instinct MI300X systems and newer, 50% `TVIOL` means the GPU spent half the sampling period at reduced clocks. See [Checking power and thermal violations](#checking-power-and-thermal-violations) for more information.

### Checking power and thermal violations

- On **Instinct MI300 Series and newer**, check `per_ppt_pwr` (PVIOL) and TVIOL percentages via {c:func}`amdsmi_get_violation_status`. Using the CLI, `amd-smi monitor --violation` and `amd-smi metric --violation` display PVIOL and TVIOL percentages. See [Usage](#usage) for more information.

  - `PVIOL`, representing power violations, is the percentage of time the GPU throttled due to power (wattage) limits. This occurs when the GPU's power consumption exceeds safe limits, triggering power throttling to stay within the power budget.

  - `TVIOL`, representing thermal violation, is the percentage of time the GPU throttled due to temperature limits. This happens when GPU temperatures exceed safe operating thresholds, causing the GPU to reduce performance to cool down.

  Values greater than 0% indicate time spent throttled. A GPU can experience both simultaneously, for example 30% `PVIOL` and 20% `TVIOL` at the same time. See [Interpreting violations API results](#interpreting-violations-api-results) for details.

- On **Radeon (Navi) and Instinct MI100/MI200 Series**, look at `throttle_status` bit flags in `amd-smi metric --power` to see if PPT (power) or thermal throttling is active. See [Interpreting throttle_status results](#throttle-status-bit-flags) for more information.

### Monitoring PROCHOT (processor hot) throttling

- On **Instinct MI300 Series and newer**, use `amd-smi metric --violation` or `amd-smi monitor --violation` to check `PROCHOT_VIOLATION_ACTIVITY` and `PROCHOT_VIOLATION_STATUS`.

- On **Radeon (Navi) and Instinct MI100/MI200 Series**, use {c:func}`amdsmi_get_gpu_metrics_info` and check the `indep_throttle_status` field for `PROCHOT_GFX` bits. PROCHOT indicates emergency thermal throttling when the GPU hits critical temperature limits.

### Monitoring GPU hotspot temperature violations

- On **Instinct MI300 Series and newer**, monitor hotspot temperature with {c:func}`amdsmi_get_temp_metric` and correlate with TVIOL%. High TVIOL% combined with high hotspot temps (>95°C) indicates thermal throttling. Use `amd-smi metric --gpu all --temperature` to track temperatures alongside violation status. See [Usage](#usage) for more information.

### Detecting HBM (high bandwidth memory) thermal throttling

HBM (High-Bandwidth Memory) thermal throttling occurs when GPU memory overheats.

- On **Instinct MI300 Series and newer**, this is detected via `per_hbm_thrm` (HBM_TVIOL%) and `active_hbm_thrm` in the violations API. See [Violation status fields](#violation-status-fields) for more information.

- On **Radeon (Navi) and Instinct MI100/MI200 Series**, check the `TEMP_MEM` bit in `indep_throttle_status`. Detailed HBM violation percentages are only available on MI3x+. See [Interpreting throttle_status results](#throttle-status-bit-flags) for more information.

### Adjusting clock limits

Some Instinct MI300X GPUs and newer variants support adjusting the graphics clock (SCLK) and memory clock (MCLK) min/max limits, which can help manage power violations by capping clock speeds before the hardware throttles.

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

### Adjusting the sample rate

Violations are sampled every 100ms — the fastest rate the driver can update the metric cache. Set `AMDSMI_GPU_METRICS_CACHE_MS=0` to disable AMD SMI's internal cache and let the driver control when the cache updates. See [AMD SMI C/C++ library usage](../how-to/amdsmi-cpp-lib.md) for environment variable details.

## Usage

AMD SMI provides tools to programmatically monitor GPU violations and throttling events.

:::{dropdown} Tip for NVML users
The closest equivalent to `nvmlDeviceGetViolationStatus()` is {c:func}`amdsmi_get_violation_status`.

| nvidia-smi command | amd-smi equivalent | Notes |
|--------------------|--------------------|-------|
| `nvidia-smi -q -d PERFORMANCE` | `amd-smi metric --violation` | Instantaneous violation status; MI3x+ only |
| `nvidia-smi dmon -s p` | `amd-smi monitor --violation` | Continuous violation monitoring; MI3x+ only |
| `nvidia-smi -q -d CLOCK` | `amd-smi metric --clock` | Current clock frequencies |
| `nvidia-smi -q -d POWER` | `amd-smi metric --power` | Power usage and throttle status (Navi/MI1x/MI2x) |
:::

:::::{tab-set}
::::{tab-item} C/C++

The AMD SMI library provides APIs to query violation status and related functionalities.

See {ref}`GPU monitoring <tagGPUMonitor>` APIs:

- {c:func}`amdsmi_get_violation_status` - Get violation percentages
- {c:func}`amdsmi_get_temp_metric` - Monitor GPU temperatures
- {c:func}`amdsmi_get_gpu_activity` - Monitor GPU utilization

See {ref}`Clock, power, and performance queries <tagClkPowerPerfQuery>`:

- {c:func}`amdsmi_get_gpu_metrics_info` - Get throttle_status and detailed metrics

See {ref}`PCIe queries <tagPCIeQuery>`:

- {c:func}`amdsmi_get_gpu_asic_info` - Check ASIC capabilities

See {ref}`ASIC and board static information <tagAsicBoardInfo>` APIs:

- {c:func}`amdsmi_get_power_cap_info` - Check power limits
- {c:func}`amdsmi_get_gpu_bdf_id` - Identify GPU device

See [`example/amd_smi_drm_example.cc`](../../example/amd_smi_drm_example.cc) for a complete working example.

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

- [`amdsmi_get_violation_status()`](/reference/amdsmi-py-api.md#amdsmi_get_violation_status)
- [`amdsmi_get_gpu_metrics_info()`](/reference/amdsmi-py-api.md#amdsmi_get_gpu_metrics_info)
- [`amdsmi_get_temp_metric()`](/reference/amdsmi-py-api.md#amdsmi_get_temp_metric)
- [`amdsmi_get_power_cap_info()`](/reference/amdsmi-py-api.md#amdsmi_get_power_cap_info)
- [`amdsmi_get_gpu_activity()`](/reference/amdsmi-py-api.md#amdsmi_get_gpu_activity)
- [`amdsmi_get_gpu_asic_info()`](/reference/amdsmi-py-api.md#amdsmi_get_gpu_asic_info)
- [`amdsmi_get_gpu_bdf_id()`](/reference/amdsmi-py-api.md#amdsmi_get_gpu_bdf_id)

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

## Interpreting violations API results

:::{admonition} GPU architecture support
The violations API is supported on [Instinct MI300 Series and newer data center GPUs](#gpu-architecture-support).
:::

The violations API returns both power violations (PVIOL) and thermal violations (TVIOL) as percentages over time. You can generally interpret these percentages using the following heuristic:

| Value | Meaning |
|-------|----------|
| 0% | No throttling - GPU operating normally |
| 1-25% | Light throttling - minor performance impact |
| 25-50% | Moderate throttling - noticeable performance loss |
| 50-100% | Heavy throttling - significant performance degradation |
| N/A or max_uint | Feature not supported on this GPU |

### Violation status fields

The {c:struct}`amdsmi_violation_status_t` struct (returned by {c:func}`amdsmi_get_violation_status`) provides three categories of data for each violation type.  See {c:struct}`amdsmi_violation_status_t` for details.

| Category | Field prefix | Value type | Description |
|----------|-------------|------------|-------------|
| Accumulated counters | `acc_*` | uint64 | Raw counter incremented while violation is active |
| Violation status | `active_*` | uint8 (1/0) | Whether the violation is currently ACTIVE or NOT ACTIVE |
| Violation activity | `per_*` | uint64 (%) | Percentage of sampling period spent in violation (>0% = throttled) |

#### Core violation types

Each core violation type maps to a specific hardware protection mechanism. The fields below cover the most common causes of GPU throttling.

| Violation type | Accumulated | Status | Activity | Description |
|----------------|------------|--------|----------|-------------|
| PROCHOT | `acc_prochot_thrm` | `active_prochot_thrm` | `per_prochot_thrm` | Processor hot — emergency thermal throttling at critical temperature |
| PPT (Power) | `acc_ppt_pwr` | `active_ppt_pwr` | `per_ppt_pwr` | Package Power Tracking — PVIOL; power consumption exceeds limits |
| Socket Thermal | `acc_socket_thrm` | `active_socket_thrm` | `per_socket_thrm` | Socket-level thermal — TVIOL; socket temperature exceeds limits |
| VR Thermal | `acc_vr_thrm` | `active_vr_thrm` | `per_vr_thrm` | Voltage regulator thermal throttling |
| HBM Thermal | `acc_hbm_thrm` | `active_hbm_thrm` | `per_hbm_thrm` | High Bandwidth Memory thermal throttling |

#### Per-XCP per-XCC violation types

These fields are 2D arrays indexed by `[XCP][XCC]` and require gpu_metrics v1.8 or newer:

| Violation type | Accumulated | Status | Activity | Description |
|----------------|------------|--------|----------|-------------|
| GFX Clock Below Host Limit (Power) | `acc_gfx_clk_below_host_limit_pwr` | `active_gfx_clk_below_host_limit_pwr` | `per_gfx_clk_below_host_limit_pwr` | GFX clock limited below host limit due to power |
| GFX Clock Below Host Limit (Thermal) | `acc_gfx_clk_below_host_limit_thm` | `active_gfx_clk_below_host_limit_thm` | `per_gfx_clk_below_host_limit_thm` | GFX clock limited below host limit due to thermal |
| GFX Clock Below Host Limit (Total) | `acc_gfx_clk_below_host_limit_total` | `active_gfx_clk_below_host_limit_total` | `per_gfx_clk_below_host_limit_total` | GFX clock limited below host limit for any reason |
| Low Utilization | `acc_low_utilization` | `active_low_utilization` | `per_low_utilization` | Low GPU utilization detected |

:::{note}
How `GFXCLK_*` and `LOW_UTIL*` differ from core PVIOL and TVIOL fields:

- **Scope**: These are per-XCP (Compute Partition) × per-XCC (Compute Complex) 2D arrays, not socket-level aggregates like PVIOL/TVIOL.
- **What they measure**: `GFXCLK_*` tracks when the GFX clock is forced *below a host-set clock limit* due to power (`_pwr`) or thermal (`_thm`) reasons. `LOW_UTIL*` tracks periods of low GPU utilization — a clock reduction cause unrelated to power or thermal limits.
- **Availability**: Require gpu_metrics v1.8 or newer; returns max_uint on earlier drivers/ASICs.
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

(throttle-status-bit-flags)=
## Interpreting throttle_status results

:::{admonition} GPU architecture support
`throttle_status` and `indep_throttle_status` apply to [Radeon (Navi) and Instinct MI100/MI200 Series GPUs](#gpu-architecture-support).
:::

- `throttle_status` (uint32_t) indicates *whether* the GPU is throttling (non-zero means throttled).
- `indep_throttle_status` (uint64_t) encodes *why* via per-reason bit flags defined in the kernel driver.

The canonical bit definitions are in the `SMU_THROTTLER_*` enum in
[`drivers/gpu/drm/amd/pm/swsmu/inc/amdgpu_smu.h`](https://github.com/ROCm/amdgpu/blob/master/drivers/gpu/drm/amd/pm/swsmu/inc/amdgpu_smu.h)
in the ROCm AMDGPU kernel driver. AMD SMI passes the raw `uint64_t` value through to the caller without interpreting the bits, so refer to that header for the authoritative bit-position-to-flag mapping. The following table summarizes the key ranges:

| Category | Bit range | Key flags |
|----------|-----------|-----------|
| Power throttlers | 0–7 | PPT0, PPT1, SPL, FPPT, SPPT |
| Current throttlers | 16–23 | TDC_GFX, TDC_SOC, TDC_MEM, EDC_CPU, EDC_GFX |
| Temperature throttlers | 32–47 | TEMP_GPU, TEMP_MEM (HBM_THM), TEMP_HOTSPOT, TEMP_SOC (SOCKET_THM), TEMP_VR_GFX (VR_THM), PROCHOT_GFX |

## Troubleshooting

### High PVIOL (Power Violations)?

- Check power limit settings with {c:func}`amdsmi_get_power_cap_info`
- View static power cap details (default, min, max): `amd-smi static --limit`
- Monitor live power consumption: `amd-smi monitor --power`
- Verify adequate PSU capacity for your system
- Consider reducing workload intensity or power limits
- Monitor with: `amd-smi metric --gpu all --power`

:::{note}
`amd-smi static --limit` shows power cap thresholds and thermal shutdown/slowdown limits. If your GPU is hitting these limits, it may throttle to stay within them, causing PVIOL/TVIOL. Adjusting power limits or improving cooling can help reduce power or thermal related violations.
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
- **For gpu_metrics-sourced fields (violations, `SOCKET_POWER`, engine usage, etc.) returning N/A:** `gpu_metrics` is a versioned structure supplied by the amdgpu driver, and AMD SMI needs explicit support for each version's layout. When the driver is newer than your AMD SMI (or ROCm) release, it can report a `gpu_metrics` version AMD SMI doesn't recognize yet, so these fields read N/A. Upgrade AMD SMI or ROCm to a release paired with your driver's `gpu_metrics` version to resolve this. (ROCm 7.13 added support for the dynamic `gpu_metrics` layout introduced in v1.9, which handles current and future versions, so releases from 7.13 onward are no longer affected by this mismatch.)
- Check your ASIC generation with {c:func}`amdsmi_get_gpu_asic_info` or
  `amd-smi static --asic`

See {ref}`About N/A values <cli-output-na>` for more information.

## Further reading

- [GPU Power/Thermal Controls and Monitoring (Linux kernel)](https://docs.kernel.org/gpu/amdgpu/thermal.html#gpu-metrics)
