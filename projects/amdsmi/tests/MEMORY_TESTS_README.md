# Memory Management Tests

This document describes the test coverage for AMD SMI memory management features (UMA carveout and TTM configuration).

## Features Tested

### 1. UMA Carveout (VRAM - Dedicated GPU Memory)
- Reading current carveout configuration
- Validating available options
- Setting carveout index (DRY_RUN mode)

### 2. TTM Configuration (GTT - Shared GPU Memory)
- Reading current TTM pages limit
- Setting TTM pages limit (DRY_RUN mode)
- Resetting TTM to system default (DRY_RUN mode)

## DRY_RUN Mode

To enable comprehensive testing of write operations without actually modifying system configuration, all memory management write functions support a **DRY_RUN** mode.

### How DRY_RUN Works

When the `AMDSMI_DRY_RUN` environment variable is set to `1`:

1. **All validation is performed** - The functions validate inputs, check permissions, and verify file existence
2. **No files are modified** - System files (sysfs, /etc/modprobe.d/) are not written
3. **No system commands run** - Commands like `dracut -f` are not executed
4. **Status is printed** - Debug output shows what would have been done
5. **Return codes are accurate** - Functions return the same status codes as real execution

### Enabling DRY_RUN Mode

#### Bash/Shell:
```bash
export AMDSMI_DRY_RUN=1
amd-smi set --vram-carveout 2 --gpu 0  # Simulates setting VRAM carveout to index 2
```

#### Python:
```python
import os
os.environ['AMDSMI_DRY_RUN'] = '1'
amdsmi.amdsmi_set_gpu_uma_carveout(processor_handle, 2)
```

#### C++:
```cpp
setenv("AMDSMI_DRY_RUN", "1", 1);
amdsmi_set_gpu_uma_carveout(processor_handle, 2);
```

### Disabling DRY_RUN Mode

#### Bash/Shell:
```bash
unset AMDSMI_DRY_RUN
```

#### Python:
```python
del os.environ['AMDSMI_DRY_RUN']
```

#### C++:
```cpp
unsetenv("AMDSMI_DRY_RUN");
```

## Test Files

### C++ Functional Tests (GTest)

**Location:** `tests/amd_smi_test/functional/`

**Files:**
- `memory_read_write.h` - Test class definition
- `memory_read_write.cc` - Test implementation

**What's Tested:**
- ✅ Read UMA carveout information
- ✅ Validate carveout options structure
- ✅ Set UMA carveout (DRY_RUN) - current value
- ✅ Set UMA carveout (DRY_RUN) - different valid value
- ✅ Set UMA carveout (DRY_RUN) - invalid value (expect failure)
- ✅ Read TTM information
- ✅ Validate TTM pages value
- ✅ Set TTM pages limit (DRY_RUN) - current value
- ✅ Set TTM pages limit (DRY_RUN) - different value
- ✅ Set TTM pages limit (DRY_RUN) - invalid value (expect failure)
- ✅ Reset TTM pages limit (DRY_RUN)

**Running C++ Tests:**
```bash
cd build/tests/amd_smi_test
./amdsmitst -v 1  # Run all tests
```

### Python Integration Tests

**Location:** `tests/python_unittest/integration_test.py`

**Test Functions:**
- `test_uma_carveout_info()` - Read-only UMA carveout info test
- `test_uma_carveout_set_dry_run()` - UMA carveout write operations (DRY_RUN)
- `test_ttm_info()` - Read-only TTM info test
- `test_ttm_set_dry_run()` - TTM write operations (DRY_RUN)

**What's Tested:**
- ✅ Read UMA carveout for all GPUs
- ✅ Validate dictionary structure
- ✅ Set UMA carveout (DRY_RUN) - various scenarios
- ✅ Read TTM information
- ✅ Set TTM pages limit (DRY_RUN) - various scenarios
- ✅ Reset TTM (DRY_RUN)

**Running Python Integration Tests:**
```bash
# All tests
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -v

# Memory tests only
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "uma\|ttm" -v
```

### Python CLI Tests

**Location:** `tests/python_unittest/cli_unit_test.py`

**Test Function:**
- `test_static_mem_carveout_gtt()` - CLI command testing (display mode only)

**What's Tested:**
- ✅ `amd-smi static --mem-carveout` - Display VRAM carveout configuration
- ✅ `amd-smi static --gtt` - Display GTT configuration
- ✅ `amd-smi static --mem-carveout --gtt` - Display both
- ✅ JSON and CSV output formats
- ✅ Help text

**Note:** Write operations (--vram-carveout in set, --gtt in set/reset) are not tested in automated CLI tests to avoid requiring root permissions and system reboots. Use DRY_RUN mode for testing these operations.

**Running Python CLI Tests:**
```bash
# All CLI tests
sudo /opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -v

# Memory tests only
sudo /opt/rocm/share/amd_smi/tests/python_unittest/cli_unit_test.py -k "mem_carveout_gtt" -v
```

## Manual Testing (Real Write Operations)

For testing actual write operations that modify system configuration:

### Prerequisites
- Root/sudo permissions
- Ability to reboot the system
- Backup of critical data

### Test UMA Carveout

```bash
# View current configuration and available options
amd-smi static --mem-carveout

# Change VRAM carveout to index 2 (example) for GPU 0
sudo amd-smi set --vram-carveout 2 --gpu 0

# Reboot required
sudo reboot

# After reboot, verify
amd-smi static --mem-carveout
```

### Test GTT Configuration

```bash
# View current configuration
amd-smi static --gtt

# Set GTT to 12 GB (example)
sudo amd-smi set --gtt 12

# Reboot required
sudo reboot

# After reboot, verify
amd-smi static --gtt

# Reset to default
sudo amd-smi reset --gtt

# Reboot required
sudo reboot
```

## Test Coverage Summary

| Feature | Read | Write (DRY_RUN) | Write (Real) |
|---------|------|-----------------|--------------|
| UMA Carveout Info | ✅ Automated | ✅ Automated | ⚠️ Manual Only |
| UMA Carveout Set | - | ✅ Automated | ⚠️ Manual Only |
| TTM Info | ✅ Automated | ✅ Automated | ⚠️ Manual Only |
| TTM Set | - | ✅ Automated | ⚠️ Manual Only |
| TTM Reset | - | ✅ Automated | ⚠️ Manual Only |

**Legend:**
- ✅ Automated - Runs in CI/CD and automated test suites
- ⚠️ Manual Only - Requires manual testing due to system modification requirements

## Benefits of DRY_RUN Mode

1. **No Root Required** - Tests can run without elevated permissions
2. **No Reboot Required** - Tests don't require system restart
3. **Safe for CI/CD** - Can run in automated pipelines without affecting test systems
4. **Full Code Coverage** - Exercises all code paths including validation and error handling
5. **Developer Friendly** - Easy to test changes without risk to development system
6. **Fast Feedback** - Get immediate test results without waiting for reboots

## Example DRY_RUN Output

```bash
$ export AMDSMI_DRY_RUN=1

$ amd-smi set --vram-carveout 2 --gpu 0
[DRY_RUN] Would write UMA carveout index 2 to /sys/class/drm/card0/device/uma/carveout
Successfully set VRAM carveout to [2] (4 GB). Reboot required for changes to take effect.

$ amd-smi set --gtt 12
[DRY_RUN] Would write to /etc/modprobe.d/ttm.conf:
[DRY_RUN]   options ttm pages_limit=3145728
[DRY_RUN] Would rebuild initramfs with: dracut -f
Successfully set GTT to 12.00 GB (3145728 pages). Reboot required for changes to take effect.
```
