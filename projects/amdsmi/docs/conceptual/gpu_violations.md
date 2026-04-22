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

**Sample rate**: Violations are sampled every 100ms — the fastest rate the driver can update the metric cache. Set `AMDSMI_GPU_METRICS_CACHE_MS=0` to disable AMD SMI's internal cache and let the driver control when the cache updates. See [AMD SMI C++ library usage](../how-to/amdsmi-cpp-lib.md) for environment variable details.

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

AMD SMI provides NVML-compatible violation monitoring. The tables below map NVML concepts to AMD SMI equivalents.

### Instantaneous clock event reasons

Equivalent to `nvmlDeviceGetCurrentClocksEventReasons()` / `nvidia-smi -q -d PERFORMANCE`:

| NVML Flag | AMD SMI Equivalent | Notes |
|-----------|-------------------|-------|
| `nvmlClocksEventReasonGpuIdle` | `per_low_utilization` (per-XCP/XCC) | GPU idle / low utilization; v1.8+ only |
| `nvmlClocksEventReasonSwPowerCap` | `active_ppt_pwr` / `per_ppt_pwr` | SW power cap throttling (PVIOL) |
| `nvmlClocksThrottleReasonHwSlowdown` | `active_prochot_thrm` / `active_socket_thrm` | HW thermal or power brake slowdown |
| `nvmlClocksThrottleReasonHwThermalSlowdown` | `active_socket_thrm` / `active_hbm_thrm` / `active_vr_thrm` | HW thermal slowdown (TVIOL) |
| `nvmlClocksThrottleReasonHwPowerBrakeSlowdown` | `active_ppt_pwr` | HW power brake (PVIOL) |
| `nvmlClocksEventReasonSwThermalSlowdown` | `active_prochot_thrm` | SW/firmware thermal (PROCHOT) |
| `nvmlClocksEventReasonSyncBoost` | No direct equivalent | Not tracked by AMD SMI |
| `nvidia-smi -q -d PERFORMANCE` (CLI) | `amd-smi metric --violation` or `amd-smi monitor --violation` | MI3x+ only |

### Cumulative violation durations

Equivalent to `nvmlDeviceGetFieldValues()` with `NVML_FI_DEV_PERF_POLICY_*` / `NVML_FI_DEV_CLOCKS_EVENT_REASON_*`:

| NVML Field | AMD SMI Equivalent | Notes |
|-----------|-------------------|-------|
| `NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_POWER_CAP` | `acc_ppt_pwr` / `per_ppt_pwr` | SW power cap duration (PVIOL) |
| `NVML_FI_DEV_CLOCKS_EVENT_REASON_SW_THERM_SLOWDOWN` | `acc_prochot_thrm` / `per_prochot_thrm` | SW thermal (PROCHOT) duration |
| `NVML_FI_DEV_CLOCKS_EVENT_REASON_SYNC_BOOST` | No direct equivalent | Not tracked by AMD SMI |
| `NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT` | `acc_ppt_pwr` / `per_ppt_pwr` | Board power limit (maps to PPT/PVIOL) |
| `NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION` | `acc_low_utilization` / `per_low_utilization` | Low utilization; gpu_metrics v1.8+ only |
| `NVML_FI_DEV_PERF_POLICY_RELIABILITY` | No direct equivalent | Not tracked by AMD SMI |
| `NVML_FI_DEV_PERF_POLICY_TOTAL_BASE_CLOCKS` | `acc_gfx_clk_below_host_limit_total` / `per_gfx_clk_below_host_limit_total` | GFX clock below host limit; gpu_metrics v1.8+ only |

### Compatibility summary

| NVML Concept | AMD SMI Equivalent | Notes |
|--------------|-------------------|-------|
| `nvmlDeviceGetViolationStatus()` | `amdsmi_get_violation_status()` | Returns violation status and timestamp |
| `PVIOL` (Power Violation) | `per_ppt_pwr` | Power throttling percentage |
| `TVIOL` (Thermal Violation) | `per_socket_thrm` | Thermal throttling percentage |
| Percentage format (%) | Percentage format (%) | Compatible format (0–100%) |
| `Not Supported` indicator | `N/A` or max_uint | Indicates feature unavailability |

## Usage

AMD SMI provides tools to programmatically monitor GPU violations and throttling events.

:::::{tab-set}

::::{tab-item} C/C++

The AMD SMI library provides APIs to query violation status. See `example/amd_smi_drm_example.cc` for the complete integration example.

```cpp
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"

#define CHK_AMDSMI_RET(RET)                                                                \
  {                                                                                        \
    if (RET != AMDSMI_STATUS_SUCCESS) {                                                    \
      const char* err_str;                                                                 \
      std::cout << "AMDSMI call returned " << RET << " at line " << __LINE__ << std::endl; \
      amdsmi_status_code_to_string(RET, &err_str);                                         \
      std::cout << err_str << std::endl;                                                   \
      return RET;                                                                          \
    }                                                                                      \
  }

int main() {
  amdsmi_status_t ret;
  uint32_t gpu_number = 0;

  ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  CHK_AMDSMI_RET(ret)

  uint32_t socket_count = 0;
  ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  CHK_AMDSMI_RET(ret)

  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  CHK_AMDSMI_RET(ret)

  for (uint32_t i = 0; i < socket_count; i++) {
    uint32_t device_count = 0;
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);
    CHK_AMDSMI_RET(ret)

    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, processor_handles.data());
    CHK_AMDSMI_RET(ret)

    for (uint32_t device_index = 0; device_index < device_count; device_index++) {

      // -------------------------------------------------------------------
      // Violation Status (MI3x+ only; older ASICs may return NOT_SUPPORTED)
      // Falls back to throttle_status via gpu_metrics on older ASICs.
      // -------------------------------------------------------------------
      std::cout << "\n    Output of amdsmi_get_violation_status (GPU " << gpu_number << "):\n";
      amdsmi_violation_status_t violation_status = {};
      ret = amdsmi_get_violation_status(processor_handles[device_index], &violation_status);
      if (ret == AMDSMI_STATUS_SUCCESS) {
        constexpr uint64_t kU64Max = std::numeric_limits<uint64_t>::max();
        constexpr uint8_t kU8Max = std::numeric_limits<uint8_t>::max();

        auto u64_str = [kU64Max](uint64_t v) -> std::string {
          return (v == kU64Max) ? "N/A" : std::to_string(v);
        };
        // Matches CLI: active flags shown as ACTIVE / NOT ACTIVE / N/A
        auto active_str = [kU8Max](uint8_t v) -> std::string {
          if (v == kU8Max) return "N/A";
          return v ? "ACTIVE" : "NOT ACTIVE";
        };

        std::cout << "\treference_timestamp (us since epoch): "
                  << u64_str(violation_status.reference_timestamp) << "\n";
        std::cout << "\tviolation_timestamp (ns):             "
                  << u64_str(violation_status.violation_timestamp) << "\n";

        // Accumulated counters — names match CLI output
        std::cout << "\t-- Accumulated --\n";
        std::cout << "\tACCUMULATION_COUNTER:        " << u64_str(violation_status.acc_counter) << "\n";
        std::cout << "\tPROCHOT_ACCUMULATED:        " << u64_str(violation_status.acc_prochot_thrm) << "\n";
        std::cout << "\tPPT_ACCUMULATED:            " << u64_str(violation_status.acc_ppt_pwr) << "\n";
        std::cout << "\tSOCKET_THERMAL_ACCUMULATED: " << u64_str(violation_status.acc_socket_thrm) << "\n";
        std::cout << "\tVR_THERMAL_ACCUMULATED:     " << u64_str(violation_status.acc_vr_thrm) << "\n";
        std::cout << "\tHBM_THERMAL_ACCUMULATED:    " << u64_str(violation_status.acc_hbm_thrm) << "\n";

        // Violation status (active flags) — names match CLI output
        std::cout << "\t-- Violation Status --\n";
        std::cout << "\tPROCHOT_VIOLATION_STATUS:        "
                  << active_str(violation_status.active_prochot_thrm) << "\n";
        std::cout << "\tPPT_VIOLATION_STATUS:            "
                  << active_str(violation_status.active_ppt_pwr) << "\n";
        std::cout << "\tSOCKET_THERMAL_VIOLATION_STATUS: "
                  << active_str(violation_status.active_socket_thrm) << "\n";
        std::cout << "\tVR_THERMAL_VIOLATION_STATUS:     "
                  << active_str(violation_status.active_vr_thrm) << "\n";
        std::cout << "\tHBM_THERMAL_VIOLATION_STATUS:    "
                  << active_str(violation_status.active_hbm_thrm) << "\n";

        // Violation activity (%) — names match CLI output
        std::cout << "\t-- Violation Activity --\n";
        std::cout << "\tPROCHOT_VIOLATION_ACTIVITY:        "
                  << u64_str(violation_status.per_prochot_thrm) << " %\n";
        std::cout << "\tPPT_VIOLATION_ACTIVITY:            "
                  << u64_str(violation_status.per_ppt_pwr) << " %\n";
        std::cout << "\tSOCKET_THERMAL_VIOLATION_ACTIVITY: "
                  << u64_str(violation_status.per_socket_thrm) << " %\n";
        std::cout << "\tVR_THERMAL_VIOLATION_ACTIVITY:     "
                  << u64_str(violation_status.per_vr_thrm) << " %\n";
        std::cout << "\tHBM_THERMAL_VIOLATION_ACTIVITY:    "
                  << u64_str(violation_status.per_hbm_thrm) << " %\n";

        // GPU metrics 1.8 per-XCP/XCC arrays.
        // XCPs/XCCs where all fields are sentinel (unsupported) are skipped.
        std::cout << "\t-- Per-XCP/XCC (GPU metrics 1.8, N/A = unsupported) --\n";

        // Build "[xcc0, xcc1, ...]" string for a row, skipping trailing N/As.
        auto xcc_u64_row = [&](const uint64_t* row) -> std::string {
          int last = -1;
          for (int xcc = 0; xcc < static_cast<int>(AMDSMI_MAX_NUM_XCC); ++xcc)
            if (row[xcc] != kU64Max) last = xcc;
          if (last < 0) return "N/A";
          std::string s = "[";
          for (int xcc = 0; xcc <= last; ++xcc) {
            if (xcc) s += ", ";
            s += u64_str(row[xcc]);
          }
          return s + "]";
        };
        auto xcc_active_row = [&](const uint8_t* row) -> std::string {
          int last = -1;
          for (int xcc = 0; xcc < static_cast<int>(AMDSMI_MAX_NUM_XCC); ++xcc)
            if (row[xcc] != kU8Max) last = xcc;
          if (last < 0) return "N/A";
          std::string s = "[";
          for (int xcc = 0; xcc <= last; ++xcc) {
            if (xcc) s += ", ";
            s += active_str(row[xcc]);
          }
          return s + "]";
        };
        auto xcc_pct_row = [&](const uint64_t* row) -> std::string {
          int last = -1;
          for (int xcc = 0; xcc < static_cast<int>(AMDSMI_MAX_NUM_XCC); ++xcc)
            if (row[xcc] != kU64Max) last = xcc;
          if (last < 0) return "N/A";
          std::string s = "[";
          for (int xcc = 0; xcc <= last; ++xcc) {
            if (xcc) s += ", ";
            s += u64_str(row[xcc]) + " %";
          }
          return s + "]";
        };

        for (uint32_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp) {
          bool any_valid = false;
          for (uint32_t xcc = 0; xcc < AMDSMI_MAX_NUM_XCC; ++xcc) {
            if (violation_status.acc_gfx_clk_below_host_limit_total[xcp][xcc] != kU64Max ||
                violation_status.acc_gfx_clk_below_host_limit_pwr[xcp][xcc] != kU64Max ||
                violation_status.acc_gfx_clk_below_host_limit_thm[xcp][xcc] != kU64Max ||
                violation_status.acc_low_utilization[xcp][xcc] != kU64Max) {
              any_valid = true;
              break;
            }
          }
          if (!any_valid) continue;

          std::cout << "\tXCP[" << xcp << "]:\n";
          std::cout << "\t  -- Accumulated --\n";
          std::cout << "\t  GFX_CLK_BELOW_HOST_LIMIT_POWER_ACCUMULATED:    "
                    << xcc_u64_row(violation_status.acc_gfx_clk_below_host_limit_pwr[xcp]) << "\n";
          std::cout << "\t  GFX_CLK_BELOW_HOST_LIMIT_THERMAL_ACCUMULATED:  "
                    << xcc_u64_row(violation_status.acc_gfx_clk_below_host_limit_thm[xcp]) << "\n";
          std::cout << "\t  TOTAL_GFX_CLK_BELOW_HOST_LIMIT_ACCUMULATED:    "
                    << xcc_u64_row(violation_status.acc_gfx_clk_below_host_limit_total[xcp]) << "\n";
          std::cout << "\t  LOW_UTILIZATION_ACCUMULATED:                   "
                    << xcc_u64_row(violation_status.acc_low_utilization[xcp]) << "\n";
          std::cout << "\t  -- Violation Status --\n";
          std::cout << "\t  GFX_CLK_BELOW_HOST_LIMIT_POWER_VIOLATION_STATUS:    "
                    << xcc_active_row(violation_status.active_gfx_clk_below_host_limit_pwr[xcp]) << "\n";
          std::cout << "\t  GFX_CLK_BELOW_HOST_LIMIT_THERMAL_VIOLATION_STATUS:  "
                    << xcc_active_row(violation_status.active_gfx_clk_below_host_limit_thm[xcp]) << "\n";
          std::cout << "\t  TOTAL_GFX_CLK_BELOW_HOST_LIMIT_VIOLATION_STATUS:    "
                    << xcc_active_row(violation_status.active_gfx_clk_below_host_limit_total[xcp]) << "\n";
          std::cout << "\t  LOW_UTILIZATION_VIOLATION_STATUS:                   "
                    << xcc_active_row(violation_status.active_low_utilization[xcp]) << "\n";
          std::cout << "\t  -- Violation Activity --\n";
          std::cout << "\t  GFX_CLK_BELOW_HOST_LIMIT_POWER_VIOLATION_ACTIVITY:    "
                    << xcc_pct_row(violation_status.per_gfx_clk_below_host_limit_pwr[xcp]) << "\n";
          std::cout << "\t  GFX_CLK_BELOW_HOST_LIMIT_THERMAL_VIOLATION_ACTIVITY:  "
                    << xcc_pct_row(violation_status.per_gfx_clk_below_host_limit_thm[xcp]) << "\n";
          std::cout << "\t  TOTAL_GFX_CLK_BELOW_HOST_LIMIT_VIOLATION_ACTIVITY:    "
                    << xcc_pct_row(violation_status.per_gfx_clk_below_host_limit_total[xcp]) << "\n";
          std::cout << "\t  LOW_UTILIZATION_VIOLATION_ACTIVITY:                   "
                    << xcc_pct_row(violation_status.per_low_utilization[xcp]) << "\n";
        }
      } else if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
        // Navi/MI1x/MI2x: violation API not supported — fall back to gpu_metrics throttle_status
        std::cout << "\tViolation API not supported on this ASIC. "
                     "Falling back to gpu_metrics throttle_status.\n";
        amdsmi_gpu_metrics_t smu = {};
        amdsmi_status_t metrics_ret =
            amdsmi_get_gpu_metrics_info(processor_handles[device_index], &smu);
        if (metrics_ret == AMDSMI_STATUS_SUCCESS) {
          // throttle_status: same field the CLI uses for amd-smi metric --power
          constexpr uint32_t kU32Max = std::numeric_limits<uint32_t>::max();
          if (smu.throttle_status == kU32Max) {
            std::cout << "\tTHROTTLE_STATUS: N/A\n";
          } else {
            std::cout << "\tTHROTTLE_STATUS: "
                      << (smu.throttle_status ? "THROTTLED" : "UNTHROTTLED") << "\n";
          }
        }
      } else {
        const char* err_str = nullptr;
        amdsmi_status_code_to_string(ret, &err_str);
        std::cout << "\tamdsmi_get_violation_status failed: "
                  << (err_str ? err_str : "unknown error") << "\n";
      }
      std::cout << "\n-------------------------------------------------------------\n\n";
      gpu_number++;
    }
  }

  ret = amdsmi_shut_down();
  CHK_AMDSMI_RET(ret)
  return 0;
}
```

::::

::::{tab-item} Python

See `example/amd_smi_violation_example.py` for the complete standalone example.

```python
#!/usr/bin/env python3
"""Standalone violation-status verification script.

Calls amdsmi_get_violation_status() for every GPU and prints all fields.
On older ASICs (Navi/MI1x/MI2x) where the violation API is not supported,
falls back to throttle_status via amdsmi_get_gpu_metrics_info().
N/A means the field is unsupported on this ASIC (max_uint sentinel).
MI3x+ (MI300X and newer) is required for full violation data.
"""

import amdsmi


def main() -> None:
    amdsmi.amdsmi_init()
    try:
        processors = amdsmi.amdsmi_get_processor_handles()
        if not processors:
            print("No processors found.")
            return

        for i, processor in enumerate(processors):
            print(f"\n{'='*60}")
            print(f"GPU {i} violation status")
            print(f"{'='*60}")

            try:
                v = amdsmi.amdsmi_get_violation_status(processor)
            except amdsmi.AmdSmiException as exc:
                # Navi/MI1x/MI2x: violation API not supported — fall back to gpu_metrics
                print(f"  amdsmi_get_violation_status failed: {exc}")
                print("  Falling back to throttle_status via amdsmi_get_gpu_metrics_info()...")
                try:
                    m = amdsmi.amdsmi_get_gpu_metrics_info(processor)
                    # throttle_status: same field the CLI uses for amd-smi metric --power
                    ts = m.get("throttle_status", "N/A")
                    if ts is True:
                        print("  throttle_status: THROTTLED")
                    elif ts is False:
                        print("  throttle_status: UNTHROTTLED")
                    else:
                        print("  throttle_status: N/A")
                except amdsmi.AmdSmiException as metrics_exc:
                    print(f"  amdsmi_get_gpu_metrics_info also failed: {metrics_exc}")
                continue

            # -- Metadata --
            print(f"  reference_timestamp (us since epoch): {v['reference_timestamp']}")
            print(f"  violation_timestamp (ns):             {v['violation_timestamp']}")
            print(f"  acc_counter:                          {v['acc_counter']}")

            # -- Accumulated counters --
            print("\n  -- Accumulated counters --")
            print(f"  acc_prochot_thrm:  {v['acc_prochot_thrm']}")
            print(f"  acc_ppt_pwr:       {v['acc_ppt_pwr']}")       # PVIOL
            print(f"  acc_socket_thrm:   {v['acc_socket_thrm']}")   # TVIOL
            print(f"  acc_vr_thrm:       {v['acc_vr_thrm']}")
            print(f"  acc_hbm_thrm:      {v['acc_hbm_thrm']}")

            # -- Violation % (>0% = throttled) --
            print("\n  -- Violation % (>0% = throttled) --")
            print(f"  per_prochot_thrm (%): {v['per_prochot_thrm']}")
            print(f"  per_ppt_pwr (%):      {v['per_ppt_pwr']}")    # PVIOL
            print(f"  per_socket_thrm (%):  {v['per_socket_thrm']}")  # TVIOL
            print(f"  per_vr_thrm (%):      {v['per_vr_thrm']}")
            print(f"  per_hbm_thrm (%):     {v['per_hbm_thrm']}")

            # -- Active flags --
            print("\n  -- Active flags (True=active, False=not active, N/A=unsupported) --")
            print(f"  active_prochot_thrm: {v['active_prochot_thrm']}")
            print(f"  active_ppt_pwr:      {v['active_ppt_pwr']}")
            print(f"  active_socket_thrm:  {v['active_socket_thrm']}")
            print(f"  active_vr_thrm:      {v['active_vr_thrm']}")
            print(f"  active_hbm_thrm:     {v['active_hbm_thrm']}")

            # -- GPU metrics v1.8 per-XCP/XCC 2D arrays --
            # Each field is a list-of-lists indexed [xcp][xcc].
            # Skip fields where every entry is "N/A" to keep output clean on pre-v1.8 drivers.
            xcp_fields = [
                "acc_gfx_clk_below_host_limit_pwr",
                "acc_gfx_clk_below_host_limit_thm",
                "acc_low_utilization",
                "acc_gfx_clk_below_host_limit_total",
                "per_gfx_clk_below_host_limit_pwr",
                "per_gfx_clk_below_host_limit_thm",
                "per_low_utilization",
                "per_gfx_clk_below_host_limit_total",
                "active_gfx_clk_below_host_limit_pwr",
                "active_gfx_clk_below_host_limit_thm",
                "active_low_utilization",
                "active_gfx_clk_below_host_limit_total",
            ]

            any_xcp_printed = False
            for field in xcp_fields:
                rows = v[field]
                if all(val == "N/A" for row in rows for val in row):
                    continue
                if not any_xcp_printed:
                    print("\n  -- GPU metrics v1.8 per-XCP/XCC (N/A = unsupported) --")
                    any_xcp_printed = True
                print(f"  {field}:")
                for xcp_idx, row in enumerate(rows):
                    # Only print XCC rows that have at least one non-N/A value
                    if all(val == "N/A" for val in row):
                        continue
                    print(f"    XCP[{xcp_idx}]: {row}")

            if not any_xcp_printed:
                print("\n  -- GPU metrics v1.8 per-XCP/XCC: all N/A (pre-v1.8 driver) --")

    finally:
        amdsmi.amdsmi_shut_down()


if __name__ == "__main__":
    main()
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

- Check power limit settings with `amdsmi_get_power_cap()`
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

## Adjusting Clock Limits (MI3x+)

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
