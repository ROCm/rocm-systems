---
myst:
  html_meta:
    "description lang=en": "AMD SMI conceptual guide for GPU performance levels and performance determinism."
    "keywords": "system, management, interface, performance, determinism, clock, frequency, gfxclk, benchmark, perf-level, PowerPlay, amd-smi"
---

# Performance levels and performance determinism

AMD GPUs expose power management controls through the AMD SMI library and CLI
tool. This document covers **performance levels** (`--perf-level`) and
**performance determinism** (`--perf-determinism`): what they do, how they work
in the driver stack, and which GPUs support them.

## Performance levels

### What is a performance level?

A performance level controls the PowerPlay dynamic power management (DPM) policy
on the GPU. PowerPlay automatically adjusts clock frequencies (GFXCLK/SCLK and
UCLK/MCLK) based on workload demand. Setting a performance level overrides this
automatic behavior with a specific policy.

The request flows through the stack as:

1. **AMD SMI** (user space) calls `amdsmi_set_gpu_perf_level()`.
2. The **amdgpu kernel driver** writes the level to the
   `power_dpm_force_performance_level` sysfs file.
3. The driver sends a **PMFW** (Power Management Firmware) message to the SMU
   (System Management Unit).
4. **PMFW** enforces the clock and power behavior on the hardware.

### Available levels

| Level | Enum value | Description |
| :--- | :--- | :--- |
| `AUTO` | `AMDSMI_DEV_PERF_LEVEL_AUTO` | Default. PowerPlay dynamically adjusts clocks based on workload demand. |
| `LOW` | `AMDSMI_DEV_PERF_LEVEL_LOW` | Forces clocks to minimum regardless of workload. |
| `HIGH` | `AMDSMI_DEV_PERF_LEVEL_HIGH` | Forces clocks to maximum regardless of workload. |
| `MANUAL` | `AMDSMI_DEV_PERF_LEVEL_MANUAL` | Enables manual control of GFXCLK (SCLK) frequency ranges. Required before setting clock levels or overdrive. |
| `STABLE_STD` | `AMDSMI_DEV_PERF_LEVEL_STABLE_STD` | Stable power state with standard profiling clocks. |
| `STABLE_PEAK` | `AMDSMI_DEV_PERF_LEVEL_STABLE_PEAK` | Stable power state with peak clocks. |
| `STABLE_MIN_MCLK` | `AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_MCLK` | Stable power state with minimum memory clock. |
| `STABLE_MIN_SCLK` | `AMDSMI_DEV_PERF_LEVEL_STABLE_MIN_SCLK` | Stable power state with minimum system clock. |
| `DETERMINISM` | `AMDSMI_DEV_PERF_LEVEL_DETERMINISM` | Performance determinism state. Set automatically when determinism mode is enabled. |

```{note}
`MANUAL` mode is a prerequisite for setting individual clock frequency levels
(`amd-smi set --clk-level`) or clock ranges (`amd-smi set --clk-limit`).
Setting the level back to `AUTO` restores dynamic power management.
```

### How the driver handles performance levels

The kernel driver (for example,
[smu_v13_0.c](https://github.com/ROCm/amdgpu/blob/master/drivers/gpu/drm/amd/pm/swsmu/smu13/smu_v13_0.c#L1613))
translates each level into PMFW messages. The sysfs strings written by the
driver are: `auto`, `low`, `high`, `manual`, `profile_standard`,
`profile_peak`, `profile_min_mclk`, `profile_min_sclk`, and
`perf_determinism`.

Since AMD SMI supports multiple ASICs, a given ASIC may only support a subset of
these levels. If the ASIC firmware does not support the requested level, the call
returns an error.

## Performance determinism

### What is performance determinism?

Performance determinism is a specialized mode that delivers **consistent,
predictable performance** across multiple GPUs. It enforces a user-defined
SoftMax limit on the GFXCLK frequency, preventing the GFXCLK PLL (Phase-Locked
Loop) from stretching when running the same workload on different GPUs.

### Why it matters

In multi-GPU workloads (for example, distributed LLM inference or training),
individual GPUs can exhibit clock frequency variations due to silicon variation,
thermal conditions, and voltage/frequency curve differences. This causes:

- **Synchronization stalls** -- faster GPUs idle while waiting for slower ones.
- **Unpredictable latency** -- job completion time is driven by the slowest GPU.
- **Benchmarking inaccuracy** -- results vary between runs.

Performance determinism eliminates this by capping all GPUs at the same maximum
GFXCLK frequency.

### How it works

When you enable performance determinism via
`amdsmi_set_gpu_perf_determinism_mode()` or `amd-smi set --perf-determinism
<SCLKMAX>`:

1. The GPU performance level is set to `DETERMINISM`.
2. A SoftMax limit is written to the GPU's overdrive/voltage control interface,
   setting the maximum GFXCLK frequency to the user-specified value (in MHz).
3. The change is committed to hardware.

Internally the driver performs:

```text
1. Writes "perf_determinism" to power_dpm_force_performance_level
2. Writes "s 1 <clkvalue>" to pp_od_clk_voltage (sets max SCLK)
3. Writes "c" to pp_od_clk_voltage (commits the change)
```

The GPU will not exceed the specified frequency. The clock can still scale down
under light loads.

## GPU support

```{important}
Support for performance levels and determinism is **ASIC-specific** and depends
on PMFW capabilities. AMD SMI forwards requests to the kernel driver, which
sends them to PMFW. If the firmware does not support a feature, the operation
returns an error.

```

| Feature | MI200 | MI300 | MI350 | RDNA 3 | RDNA 4 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `--perf-level AUTO` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `--perf-level LOW` | ✅ | ✅ | ✅ | ✅ | ❓ |
| `--perf-level HIGH` | ✅ | ✅ | ✅ | ✅ | ❓ |
| `--perf-level MANUAL` | ✅ | ✅ | ✅ | ✅ | ❓ |
| `--perf-level STABLE_*` | ✅ | ✅ | ❓ | ✅ | ❓ |
| `--perf-determinism` | ✅ | ⚠️ | ❌ | ✅ | ❓ |

**Legend:** ✅ Supported  ⚠️ Limited / ASIC-dependent  ❌ Not supported  ❓ Unconfirmed

**Notes on specific GPU families:**

- **MI200 (Aldebaran)** -- Full support for all performance levels and
  determinism.
- **MI300 (Aqua Vanjaram)** -- Performance determinism support depends on the
  PMFW version. Verify with `amd-smi metric --perf-level` after setting.
- **MI350** -- Performance determinism is **defeatured**. CSC (Clock Stretching
  Compensation) is intended to address the variability use case that determinism
  previously covered.
- **RDNA 3 (Navi 3x)** -- Full support for performance levels and determinism.
- **RDNA 4 (Navi 4x)** -- Depends on PMFW capabilities for the specific ASIC.

## From concept to action

:::::{tab-set}
::::{tab-item} C/C++

- `amdsmi_get_gpu_perf_level()` -- Query the current DPM performance level.
- `amdsmi_set_gpu_perf_level()` -- Set the PowerPlay performance level.
- `amdsmi_set_gpu_perf_determinism_mode()` -- Enable performance determinism
  with a GFXCLK SoftMax (in MHz).

See [Clock, Power, and Performance Control](/doxygen/docBin/html/group__tagClkPowerPerfControl)
for the full API reference.
::::

::::{tab-item} Python

- [](/reference/amdsmi-py-api.md#amdsmi_get_gpu_perf_level) -- Query current
  performance level.
- [](/reference/amdsmi-py-api.md#amdsmi_set_gpu_perf_level) -- Set performance
  level.
- [](/reference/amdsmi-py-api.md#amdsmi_set_gpu_perf_determinism_mode) -- Enable
  performance determinism.
::::

::::{tab-item} amd-smi CLI
See [`amd-smi set --help`](/how-to/amdsmi-cli-tool.md#amd-smi-set) and
[`amd-smi reset --help`](/how-to/amdsmi-cli-tool.md#amd-smi-reset) for all
options.

```shell
# Query current performance level
amd-smi metric --perf-level

# Set performance level
amd-smi set --perf-level HIGH

# Enable performance determinism with GFXCLK capped at 1900 MHz
amd-smi set --perf-determinism 1900

# Disable performance determinism
amd-smi reset --perf-determinism

# Reset to automatic
amd-smi set --perf-level AUTO
```

::::
:::::

## Further reading

- [GPU sysfs Power State Interfaces (Linux kernel)](https://docs.kernel.org/gpu/amdgpu/thermal.html#gpu-sysfs-power-state-interfaces)
- [AMDGPU Documentation](https://docs.kernel.org/gpu/amdgpu/index.html)
- [AMDGPU driver source -- SMU v13 perf level handling](https://github.com/ROCm/amdgpu/blob/master/drivers/gpu/drm/amd/pm/swsmu/smu13/smu_v13_0.c)
