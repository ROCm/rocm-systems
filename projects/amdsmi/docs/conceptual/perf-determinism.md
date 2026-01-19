---
myst:
  html_meta:
    "description lang=en": "AMD SMI guide for setting performance determinism."
    "keywords": "system, management, interface, performance, determinism, clock, frequency, gfxclk, benchmark"
---

# Performance determinism

Performance determinism mode enables consistent GPU performance by enforcing a fixed maximum GFXCLK
(graphics clock) frequency. This prevents clock frequency variations when running the same workload
across different GPUs, making it useful for benchmarking and performance analysis.

## How it works

When enabled, performance determinism sets a user-defined SoftMax limit for the GFXCLK frequency.
This prevents the GFXCLK PLL from stretching during workload execution, minimizing performance
variation. The GPU performance level changes to `DETERMINISM` when this mode is active.

## Supported hardware

Performance determinism is supported on AMD Instinct MI200 series and AMD Radeon RX 7000 series
(RDNA 3) GPUs. Support may vary by ASIC -- MI300 series GPUs may not support this feature.

## From concept to action

AMD SMI provides tools to enable and manage performance determinism mode.

:::::{tab-set}
::::{tab-item} C/C++
The AMD SMI library provides APIs to enable performance determinism and query the current
performance level.

See [Clock, Power, and Performance Control](/doxygen/docBin/html/group__tagClkPowerPerfControl) for
available APIs.
::::

::::{tab-item} Python
See related APIs:

- [](/reference/amdsmi-py-api.md#amdsmi_set_gpu_perf_determinism_mode)
- [](/reference/amdsmi-py-api.md#amdsmi_get_gpu_perf_level)
- [](/reference/amdsmi-py-api.md#amdsmi_set_gpu_perf_level)
::::

::::{tab-item} amd-smi CLI
See [`amd-smi set --help`](/how-to/amdsmi-cli-tool.md#amd-smi-set) and
[`amd-smi reset --help`](/how-to/amdsmi-cli-tool.md#amd-smi-reset) for details and available options.

```shell
# Enable performance determinism with GFXCLK set to 1900 MHz
amd-smi set --perf-determinism 1900

# Disable performance determinism
amd-smi reset --perf-determinism

# Query current performance level
amd-smi metric --perf-level
```
::::
:::::

## Further reading

- [PowerPlay (Linux kernel)](https://docs.kernel.org/gpu/amdgpu/driver-misc.html#powerplay)
- [AMDGPU Documentation](https://docs.kernel.org/gpu/amdgpu/index.html)
