# How to Python Unit Tests
## Overview
We use Python's default Python unittest testing framework. You can read more about it here [Python unittest v3.8](https://docs.python.org/3.8/library/unittest.html).

## Warning to Users
AMD SMI Python API tests are subject to change. These tests are currently a work in progress and may not work on your system.

## Pre-Requisites Before Running Tests
Follow our install/build guides to ensure the Python API is installed correctly according to [AMD SMI installation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).

***Versions***: Python 3.8+

> **Note:** **`cli_unit_test.py`** must be run with **`sudo`** and with **`/opt/rocm-wsl/bin` on `PATH`** (it invokes `amd-smi`). Example:
> `sudo env PATH="$PATH:/opt/rocm-wsl/bin" /opt/rocm-wsl/share/amd_smi/tests/python_unittest/cli_unit_test.py -v`

## How to Run
### Basic How To
The three test scripts are installed as:

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/cli_unit_test.py
```

**Unittest only (not verbose)**

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py -b
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -b
sudo env PATH="$PATH:/opt/rocm-wsl/bin" /opt/rocm-wsl/share/amd_smi/tests/python_unittest/cli_unit_test.py -b
```

**Unittest verbose**

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py -v
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -v
sudo env PATH="$PATH:/opt/rocm-wsl/bin" /opt/rocm-wsl/share/amd_smi/tests/python_unittest/cli_unit_test.py -v
```

**Unittest filter and verbose**

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py -k "testname" -v
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -k "testname" -v
sudo env PATH="$PATH:/opt/rocm-wsl/bin" /opt/rocm-wsl/share/amd_smi/tests/python_unittest/cli_unit_test.py -k "testname" -v
```

From the `python_unittest` directory you can use `./cli_unit_test.py` (and the same `sudo env PATH=...` prefix) instead of the full path.

## Unittest Run Options
The Unittest Run calls the tests directly. The cache provider will always be used.

options:
  -  -h, --help           show this help message and exit
  -  -v, --verbose        Verbose output
  -  -q, --quiet          Quiet output
  -  -b, --buffer         Buffer stdout and stderr during tests
  -  -k "testname"        Only run tests which match the given substring

### Unittest: not verbose
Runs all tests. Silence print statements to stdout. Lists tests results.
This is also the best way to list all tests available.

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py -b
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -b
sudo env PATH="$PATH:/opt/rocm-wsl/bin" /opt/rocm-wsl/share/amd_smi/tests/python_unittest/cli_unit_test.py -b
```

ex.
<details open>
  <summary>Click for example: <i><b>Unittest: not verbose</i></b></summary>

~~~shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py -b
....s.sss..s...sss..sss.....s.....ss...s.........s.s.....s......s.........s.................
.....ss...s..s.........ssss.s.s.ss...s.sssss.ssss.sssss.sss....s...ss...
----------------------------------------------------------------------
Ran 164 tests in 0.457s

OK (skipped=53)
~~~

</details>

### Unittest: verbose (with print statements)
Helpful to see print outs of Python.

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py -v
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -v
sudo env PATH="$PATH:/opt/rocm-wsl/bin" /opt/rocm-wsl/share/amd_smi/tests/python_unittest/cli_unit_test.py -v
```


ex.
<details open>
  <summary>Click for example: <i><b>Unittest: verbose (with print statements)</i></b></summary>

~~~shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -v
AMD SMI Integration Tests
asic info(gpu=0)
{
    "market_name": "AMD Radeon RX 9070 XT",
    "vendor_id": "0x1002",
    "vendor_name": "Advanced Micro Devices Inc. [AMD/ATI]",
    "subvendor_id": "0x7550",
    "device_id": "0x7550",
    "rev_id": "0xc0",
    "asic_serial": "0xCE5D239FE548AF74",
    "oam_id": 0,
    "num_compute_units": 64,
    "target_graphics_version": "gfx1201",
    "subsystem_id": "0x1002",
    "flags": 0
}
board info(gpu=0)
{
    "model_number": "N/A",
    "product_serial": "N/A",
    "fru_id": "N/A",
    "product_name": "AMD Radeon RX 9070 XT",
    "manufacturer_name": "Advanced Micro Devices, Inc. [AMD/ATI]"
}
test_init (__main__.TestAmdSmiInit.test_init) ... ok
test_accelerator_partition_profile (__main__.TestAmdSmiPythonInterface.test_accelerator_partition_profile) ... ok
test_accelerator_partition_profile_config (__main__.TestAmdSmiPythonInterface.test_accelerator_partition_profile_config) ... ok
test_asic_kfd_info (__main__.TestAmdSmiPythonInterface.test_asic_kfd_info) ... ok
test_bad_page_info (__main__.TestAmdSmiPythonInterface.test_bad_page_info) ... ok
test_bdf_device_id (__main__.TestAmdSmiPythonInterface.test_bdf_device_id) ... ok
test_board_info (__main__.TestAmdSmiPythonInterface.test_board_info) ... ok
test_clock_frequency (__main__.TestAmdSmiPythonInterface.test_clock_frequency) ... ok
test_clock_frequency_DCEF (__main__.TestAmdSmiPythonInterface.test_clock_frequency_DCEF) ... ok
test_clock_info (__main__.TestAmdSmiPythonInterface.test_clock_info) ... ok
test_clock_info_vclk0_dclk0 (__main__.TestAmdSmiPythonInterface.test_clock_info_vclk0_dclk0) ... ok
test_clock_info_vclk1_dclk1 (__main__.TestAmdSmiPythonInterface.test_clock_info_vclk1_dclk1) ... ok
test_driver_info (__main__.TestAmdSmiPythonInterface.test_driver_info) ... ok
test_ecc_count_block (__main__.TestAmdSmiPythonInterface.test_ecc_count_block) ... ok
test_ecc_count_total (__main__.TestAmdSmiPythonInterface.test_ecc_count_total) ... ok
test_fw_info (__main__.TestAmdSmiPythonInterface.test_fw_info) ... ok
test_get_gpu_compute_partition (__main__.TestAmdSmiPythonInterface.test_get_gpu_compute_partition) ... ok
test_get_gpu_revision (__main__.TestAmdSmiPythonInterface.test_get_gpu_revision) ... ok
test_get_violation_status (__main__.TestAmdSmiPythonInterface.test_get_violation_status) ... ok
test_get_vram_info (__main__.TestAmdSmiPythonInterface.test_get_vram_info) ... ok
test_get_xcd_counter (__main__.TestAmdSmiPythonInterface.test_get_xcd_counter) ... ok
test_gpu_activity (__main__.TestAmdSmiPythonInterface.test_gpu_activity) ... ok
test_gpu_cache_info (__main__.TestAmdSmiPythonInterface.test_gpu_cache_info) ... ok
test_gpu_pm_metrics_info (__main__.TestAmdSmiPythonInterface.test_gpu_pm_metrics_info) ... ok
test_gpu_reg_table_info (__main__.TestAmdSmiPythonInterface.test_gpu_reg_table_info) ... ok
test_memory_usage (__main__.TestAmdSmiPythonInterface.test_memory_usage) ... ok
test_nic_bdf_device_id (__main__.TestAmdSmiPythonInterface.test_nic_bdf_device_id) ... skipped 'test_nic_bdf_device_id | Missing amdsmi API(s) in amdsmi_interface.py: amdsmi_get_nic_processor_handles'
test_pcie_info (__main__.TestAmdSmiPythonInterface.test_pcie_info) ... ok
test_power_info (__main__.TestAmdSmiPythonInterface.test_power_info) ... ok
test_process_list (__main__.TestAmdSmiPythonInterface.test_process_list) ... ok
test_processor_type (__main__.TestAmdSmiPythonInterface.test_processor_type) ... ok
test_ras_block_features_enabled (__main__.TestAmdSmiPythonInterface.test_ras_block_features_enabled) ... ok
test_ras_feature_info (__main__.TestAmdSmiPythonInterface.test_ras_feature_info) ... ok
test_socket_info (__main__.TestAmdSmiPythonInterface.test_socket_info) ... ok
test_switch_bdf_device_id (__main__.TestAmdSmiPythonInterface.test_switch_bdf_device_id) ... skipped 'test_switch_bdf_device_id | Missing amdsmi API(s) in amdsmi_interface.py: amdsmi_get_switch_processor_handles, amdsmi_get_device_id'
test_temperature_metric (__main__.TestAmdSmiPythonInterface.test_temperature_metric) ... ok
test_temperature_metric_edge (__main__.TestAmdSmiPythonInterface.test_temperature_metric_edge) ... ok
test_temperature_metric_hbm (__main__.TestAmdSmiPythonInterface.test_temperature_metric_hbm) ... ok
test_temperature_metric_plx (__main__.TestAmdSmiPythonInterface.test_temperature_metric_plx) ... ok
test_utilization_count (__main__.TestAmdSmiPythonInterface.test_utilization_count) ... ok
test_vbios_info (__main__.TestAmdSmiPythonInterface.test_vbios_info) ... ok
test_vendor_name (__main__.TestAmdSmiPythonInterface.test_vendor_name) ... ok
test_walkthrough (__main__.TestAmdSmiPythonInterface.test_walkthrough) ...

#######################################################################
========> test_walkthrough start <========



###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_gpu_asic_info 

  asic_info['market_name'] is: AMD Radeon RX 9070 XT
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x7550
  asic_info['rev_id'] is: 0xc0
  asic_info['subsystem_id'] is: 0x1002
  asic_info['asic_serial'] is: 0xCE5D239FE548AF74
  asic_info['oam_id'] is: 0
  asic_info['target_graphics_version'] is: gfx1201
  asic_info['num_compute_units'] is: 64

###Test amdsmi_get_gpu_kfd_info 

  Not Supported, skipping...




###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_power_info 

  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: N/A
  power_info['gfx_voltage'] is: 34
  power_info['soc_voltage'] is: N/A
  power_info['mem_voltage'] is: N/A
  power_info['power_limit'] is: 0

###Test amdsmi_get_power_cap_info 

  power_info['dpm_cap'] is: 0
  power_info['power_cap'] is: 0

###Test amdsmi_is_gpu_power_management_enabled 

  Not Supported, skipping...




###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_gpu_vbios_info 

  vbios_info['part_number'] is: 113-G2950200-101
  vbios_info['build_date'] is: 2025/03/05 10:49
  vbios_info['name'] is: Navi48 XTX G29502 AIB
  vbios_info['version'] is: 023.008.000.068
  vbios_info['boot_firmware'] is: N/A




###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_gpu_board_info 

  board_info['model_number'] is: N/A
  board_info['product_serial'] is: N/A
  board_info['fru_id'] is: N/A
  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]
  board_info['product_name'] is: AMD Radeon RX 9070 XT




###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_fw_info 

  FW name:           38
  FW version:        29
  FW name:           19
  FW version:        09.10.90.2F
  FW name:           18
  FW version:        09.10.90.2F
  FW name:           1
  FW version:        6834944
  FW name:           9
  FW version:        12484000
  FW name:           4
  FW version:        2870
  FW name:           7
  FW version:        3180
  FW name:           3
  FW version:        2940
  FW name:           30
  FW version:        00.3A.10.14
  FW name:           29
  FW version:        3805204
  FW name:           10
  FW version:        7966358
  FW name:           46
  FW version:        387




###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_gpu_driver_info 

Driver info:  {'driver_name': 'AMD Radeon RX 9070 XT', 'driver_version': '25.30.33.01-260309a-198975C-AMD-Software-Adrenalin-Edition', 'driver_date': '260309'}


========> test_walkthrough end <========
#######################################################################

ok

----------------------------------------------------------------------
Ran 43 tests in 0.169s

OK (skipped=2)
~~~

</details>


### Unittest: filter and verbose
Allow filtering based on common or specific test names.

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py -k "testname" -v
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -k "test_walkthrough" -v
sudo env PATH="$PATH:/opt/rocm-wsl/bin" /opt/rocm-wsl/share/amd_smi/tests/python_unittest/cli_unit_test.py -k "testname" -v
```

ex.
<details open>
  <summary>Click for example: <i><b>Unittest: filter and verbose</b></i></summary>

~~~shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -k "test_asic_kfd_info" -v
AMD SMI Integration Tests
==============================================================
Legend: . = pass, s = skipped, F = fail, E = error
==============================================================
Running tests...

asic info(gpu=0)
{
    "market_name": "AMD Radeon RX 9070 XT",
    ...
}
test_asic_kfd_info (__main__.TestAmdSmiPythonInterface.test_asic_kfd_info) ... 

###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_gpu_asic_info 

  asic_info['market_name'] is: AMD Radeon RX 9070 XT
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x7550
  asic_info['rev_id'] is: 0xc0
  asic_info['subsystem_id'] is: 0x1002
  asic_info['asic_serial'] is: 0xCE5D239FE548AF74
  asic_info['oam_id'] is: 0
  asic_info['target_graphics_version'] is: gfx1201
  asic_info['num_compute_units'] is: 64

###Test amdsmi_get_gpu_kfd_info 

  Not Supported, skipping...


ok

----------------------------------------------------------------------
Ran 1 test in 0.004s

OK
~~~
</details>

## Run Tests
### Example Runs
Please refer to Python's UnitTest documentation for better overview of commands to run.

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/unit_tests.py -v
...(output truncated)...
test_check_res (__main__.TestAmdSmiPythonBDF.test_check_res) ... ok
test_format_bdf (__main__.TestAmdSmiPythonBDF.test_format_bdf) ... ok
test_parse_bdf (__main__.TestAmdSmiPythonBDF.test_parse_bdf) ... ok

----------------------------------------------------------------------
Ran 164 tests in 0.480s

OK (skipped=53)
```

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -k "temperature" -v
AMD SMI Integration Tests
==============================================================
Legend: . = pass, s = skipped, F = fail, E = error
==============================================================
Running tests...

...(asic/board info truncated)...
test_temperature_metric (__main__.TestAmdSmiPythonInterface.test_temperature_metric) ... 

###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_temp_metric 

  Current temperature for HOTSPOT is: 39
  Current temperature for VRAM is: 36

###Test amdsmi_get_temp_metric 

  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 0

###Test amdsmi_get_temp_metric 

  Shutdown (emergency) temperature for HOTSPOT is: 0
  Shutdown (emergency) temperature for VRAM is: 0


ok
test_temperature_metric_edge (__main__.TestAmdSmiPythonInterface.test_temperature_metric_edge) ... 

###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_temp_metric 

  Current temperature for EDGE is: 37
  Limit (critical) temperature for EDGE is: 110
  Shutdown (emergency) temperature for EDGE is: 0


ok
test_temperature_metric_hbm (__main__.TestAmdSmiPythonInterface.test_temperature_metric_hbm) ... 

###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_temp_metric 

  Current temperature for HBM_0 is: 0
  Limit (critical) temperature for HBM_0 is: 0
  Shutdown (emergency) temperature for HBM_0 is: 0
  ...


ok
test_temperature_metric_plx (__main__.TestAmdSmiPythonInterface.test_temperature_metric_plx) ... 

###Test Processor 0, bdf: 0000:03:00.0

###Test amdsmi_get_temp_metric 

  Current temperature for PLX is: 0
  Limit (critical) temperature for PLX is: 0
  Shutdown (emergency) temperature for PLX is: 0


ok

----------------------------------------------------------------------
Ran 4 tests in 0.019s

OK
```

```shell
/opt/rocm-wsl/share/amd_smi/tests/python_unittest/integration_test.py -k "info" -b -v
test_asic_kfd_info (__main__.TestAmdSmiPythonInterface.test_asic_kfd_info) ... ok
test_bad_page_info (__main__.TestAmdSmiPythonInterface.test_bad_page_info) ... ok
test_board_info (__main__.TestAmdSmiPythonInterface.test_board_info) ... ok
test_clock_info (__main__.TestAmdSmiPythonInterface.test_clock_info) ... ok
test_clock_info_vclk0_dclk0 (__main__.TestAmdSmiPythonInterface.test_clock_info_vclk0_dclk0) ... ok
test_clock_info_vclk1_dclk1 (__main__.TestAmdSmiPythonInterface.test_clock_info_vclk1_dclk1) ... ok
test_driver_info (__main__.TestAmdSmiPythonInterface.test_driver_info) ... ok
test_fw_info (__main__.TestAmdSmiPythonInterface.test_fw_info) ... ok
test_get_vram_info (__main__.TestAmdSmiPythonInterface.test_get_vram_info) ... ok
test_gpu_cache_info (__main__.TestAmdSmiPythonInterface.test_gpu_cache_info) ... ok
test_gpu_pm_metrics_info (__main__.TestAmdSmiPythonInterface.test_gpu_pm_metrics_info) ... ok
test_gpu_reg_table_info (__main__.TestAmdSmiPythonInterface.test_gpu_reg_table_info) ... ok
test_pcie_info (__main__.TestAmdSmiPythonInterface.test_pcie_info) ... ok
test_power_info (__main__.TestAmdSmiPythonInterface.test_power_info) ... ok
test_ras_feature_info (__main__.TestAmdSmiPythonInterface.test_ras_feature_info) ... ok
test_socket_info (__main__.TestAmdSmiPythonInterface.test_socket_info) ... ok
test_vbios_info (__main__.TestAmdSmiPythonInterface.test_vbios_info) ... ok

----------------------------------------------------------------------
Ran 17 tests in 0.069s

OK
```
