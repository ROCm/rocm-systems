---
myst:
    html_meta:
        "description": "ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software."
        "keywords": "ROCprofiler-SDK API reference, ROCprofiler-SDK PC sampling, Program counter sampling, PC sampling"
---

# ROCprofiler-SDK PC sampling method

Program Counter (PC) sampling is a profiling method that uses statistical approximation of the kernel execution by sampling GPU program counters. Furthermore, this method periodically chooses an active wave in a round robin manner and snapshots its PC. This process takes place on every compute unit simultaneously, making it device-wide PC sampling. The outcome is the histogram of samples, explaining how many times each kernel instruction was sampled.

:::{note}
Risk acknowledgment:

The PC sampling feature is under development and might not be completely stable. Use this beta feature cautiously. It may affect your system's stability and performance. Proceed at your own risk.

By activating this feature through `ROCPROFILER_PC_SAMPLING_BETA_ENABLED` environment variable, you acknowledge and accept the following potential risks:

- Hardware freeze: This beta feature could cause your hardware to freeze unexpectedly.
- Need for cold restart: In the event of a hardware freeze, you might need to perform a cold restart (turning the hardware off and on) to restore normal operations.
:::
