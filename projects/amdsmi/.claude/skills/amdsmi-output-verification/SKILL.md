---
name: amdsmi-output-verification
description: A checklist for verifying whether AMD SMI is reporting GPU information correctly.
---

# AMD SMI Output Correctness Verification

Use this checklist whenever AMD SMI appears to report incorrect GPU information. The goal is to distinguish a real AMD SMI defect from a driver, firmware, environment, documentation, or expectation mismatch.

## Test Metadata
When running a test, include the following metadata to help with triage and debugging:
- Date:
- Host:
- OS:
- Kernel:
- ROCm version:
- AMD SMI version:
- AMDGPU driver version:
- Bare metal / container / VM:
- GPU model:
- GPU count:

## Commands To Capture

Capture raw output, without editing values.

```sh
amd-smi version
amd-smi list
amd-smi static
amd-smi metric
amd-smi monitor
rocminfo
rocm-smi
lspci -nn
lspci -vv
dmesg
```

Also capture relevant sysfs values:

```sh
find /sys/class/drm -maxdepth 4 -type f
```

For each tested GPU, inspect the matching card path under:

```sh
/sys/class/drm/card*/device
```

Useful files often include:

- `vendor`
- `device`
- `subsystem_vendor`
- `subsystem_device`
- `revision`
- `product_name`
- `product_number`
- `serial_number`
- `mem_info_vram_total`
- `mem_info_vram_used`
- `gpu_busy_percent`
- `pp_dpm_sclk`
- `pp_dpm_mclk`
- `pp_power_profile_mode`

Availability varies by GPU, driver, kernel, and permission level.

## Fields To Compare

| Field | AMD SMI value | Baseline value | Baseline source | Result | Notes |
| --- | --- | --- | --- | --- | --- |
| GPU name | | | | | |
| ASIC name | | | | | |
| Vendor ID | | | | | |
| Device ID | | | | | |
| Subsystem ID | | | | | |
| Revision ID | | | | | |
| PCI BDF | | | | | |
| VRAM total | | | | | |
| Memory type | | | | | |
| VBIOS version | | | | | |
| Firmware versions | | | | | |
| Driver version | | | | | |
| ROCm version | | | | | |
| Temperature | | | | | |
| Power | | | | | |
| Clocks | | | | | |
| Utilization | | | | | |
| ECC/RAS support | | | | | |
| Process reporting | | | | | |

## Result Labels

- Correct: AMD SMI matches trusted baseline.
- Missing: AMD SMI does not expose a value that appears available elsewhere.
- Unsupported: AMD SMI explicitly reports the field is unsupported.
- Incorrect: AMD SMI reports a value that conflicts with trusted baseline.
- Misleading: AMD SMI value is technically present but ambiguous, mislabeled, or
  likely to confuse users.
- Environment-dependent: result changes depending on permissions, container,
  VM, kernel, or driver state.

## Finding Template

### Incorrect or Misleading Field

- GPU:
- ROCm version:
- AMD SMI version:
- Environment:
- Field:
- AMD SMI command/API:
- AMD SMI value:
- Expected value:
- Baseline source:
- Why baseline is trusted:
- Reproducibility:
- Severity:
- Suggested fix:
