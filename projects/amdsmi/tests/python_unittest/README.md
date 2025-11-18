# How to Python Unit Tests
## Overview
We use Python's default Python unittest testing framework. You can read more about it here [Python unittest v3.8](https://docs.python.org/3.8/library/unittest.html).

## Warning to Users
AMD SMI Python API tests are subject to change. These tests are currently a work in progress and may not work on your system.

## Pre-Requisites Before Running Tests
Follow our install/build guides to ensure the Python API is installed correctly according to [AMD SMI installation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).

***Versions***: Python 3.8+

## How to Run
### Basic How To
The 2 tests are in this PATH:  
```/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py```  
```/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py```

The recommended method to run the tests:  
<u>Unittest only (not verbose)</u>  
```/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -b -v```  
```/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -b -v```

<u>Unittest verbose</u>  
```/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -v```  
```/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -v```

<u>Unittest filter and verbose</u>  
```/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -k "testname" -v```  
```/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "testname" -v```

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

```/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -b -v```  
```/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -b -v```

ex.
<details open>
  <summary>Click for example: <i><b>Unittest: not verbose</i></b></summary>

~~~shell
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -b -v
test_check_res (__main__.TestAmdSmiPythonBDF) ... ok
test_format_bdf (__main__.TestAmdSmiPythonBDF) ... ok
test_parse_bdf (__main__.TestAmdSmiPythonBDF) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK
~~~

</details>

### Unittest: verbose (with print statements)
Helpful to see print outs of Python.

```/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -v```  
```/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -v```


ex.
<details open>
  <summary>Click for example: <i><b>Unittest: verbose (with print statements)</i></b></summary>

~~~shell
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -v
test_init (__main__.TestAmdSmiInit) ... ok
test_asic_kfd_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_asic_info

  asic_info['market_name'] is: NAVI21
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73bf
  asic_info['rev_id'] is: 0xc1
  asic_info['asic_serial'] is: 0xF8FFEB47A027DE4D
  asic_info['oam_id'] is: N/A
  asic_info['target_graphics_version'] is: gfx1030
  asic_info['num_compute_units'] is: 72

###Test amdsmi_get_gpu_kfd_info

  kfd_info['kfd_id'] is: 16970
  kfd_info['node_id'] is: 1

ok
test_bad_page_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_bad_page_info

**** [ERROR] | Test: test_bad_page_info | Caught AmdSmiLibraryException: Error code:
        2 | AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported
ok
test_bdf_device_id (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_processor_handle_from_bdf


###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-V395TRIO-2OC
  vbios_info['build_date'] is: 2021/03/28 21:35
  vbios_info['version'] is: 020.001.000.060.000000
  vbios_info['name'] is: 113-MSITV395MH.132
  vbios_info['boot_firmware'] is: 01234567

###Test amdsmi_get_gpu_device_uuid

  uuid is: f8ff73bf-0000-1000-80ff-eb47a027de4d

ok
test_board_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_board_info

  board_info['model_number'] is: N/A
  board_info['product_serial'] is: N/A
  board_info['fru_id'] is: N/A
  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]
  board_info['product_name'] is: Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]

ok
test_clock_frequency (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_clk_freq

  SYS clock_frequency['num_supported']: 2
  SYS clock_frequency['current']: 0
  SYS clock_frequency['frequency']: [500000000, 2575000000]
  DF clock_frequency['num_supported']: 3
  DF clock_frequency['current']: 1
  DF clock_frequency['frequency']: [500000000, 666000000, 1941000000]

ok
test_clock_frequency_DCEF (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_clk_freq

  DCEF clock_frequency['num_supported']: 2
  DCEF clock_frequency['current']: 0
  DCEF clock_frequency['frequency']: [417000000, 1200000000]

ok
test_clock_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_clock_info

  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2575
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255

ok
test_clock_info_vclk0_dclk0 (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_clock_info

  Current clock for domain VCLK0 is: 1400
  Max clock for domain VCLK0 is: 1400
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 1225
  Max clock for domain DCLK0 is: 1225
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255

ok
test_clock_info_vclk1_dclk1 (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_clock_info

  Current clock for domain VCLK1 is: 362
  Max clock for domain VCLK1 is: 362
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 316
  Max clock for domain DCLK1 is: 316
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255

ok
test_driver_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_driver_info

Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.9.2', 'driver_date': '2015/01/01 00:00'}

ok
test_ecc_count_block (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_ecc_count

**** [ERROR] | Test: test_ecc_count_block | Caught AmdSmiLibraryException: Error code:
        2 | AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported
ok
test_ecc_count_total (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_total_ecc_count

Number of uncorrectable errors: 0
Number of correctable errors: 0
Number of deferred errors: 0

ok
test_fw_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_fw_info

  FW name:           AMDSMI_FW_ID_CP_CE
  FW version:        37
  FW name:           AMDSMI_FW_ID_CP_PFP
  FW version:        98
  FW name:           AMDSMI_FW_ID_CP_ME
  FW version:        64
  FW name:           AMDSMI_FW_ID_CP_MEC1
  FW version:        118
  FW name:           AMDSMI_FW_ID_CP_MEC2
  FW version:        118
  FW name:           AMDSMI_FW_ID_RLC
  FW version:        96
  FW name:           AMDSMI_FW_ID_SDMA0
  FW version:        83
  FW name:           AMDSMI_FW_ID_SDMA1
  FW version:        83
  FW name:           AMDSMI_FW_ID_VCN
  FW version:        04.11.F0.00
  FW name:           AMDSMI_FW_ID_PSP_SOSDRV
  FW version:        00.21.0E.64
  FW name:           AMDSMI_FW_ID_ASD
  FW version:        553648350
  FW name:           AMDSMI_FW_ID_TA_RAS
  FW version:        1B.00.01.3E
  FW name:           AMDSMI_FW_ID_TA_XGMI
  FW version:        20.00.00.0F
  FW name:           AMDSMI_FW_ID_PM
  FW version:        00.58.90.00
  FW name:           AMDSMI_FW_ID_PLDM_BUNDLE
  FW version:        00.xx.yy.zz


ok
test_gpu_activity (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_activity

  engine_usage['gfx_activity'] is: 9 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 22 %

ok
test_memory_usage (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_memory_usage

  memory_usage for VRAM is: 17104896
  memory_usage for VIS_VRAM is: 17104896
  memory_usage for GTT is: 15065088

ok
test_pcie_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_pcie_info

  pcie_info['pcie_metric']['pcie_width'] is: 16
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 16000 MT/s
  pcie_info['pcie_static']['max_pcie_speed'] is: 16000
  pcie_info['pcie_static']['pcie_interface_version'] is: 4
  pcie_info['pcie_static']['slot_type'] is: CEM
  pcie_info['pcie_metric']['pcie_replay_count'] is: N/A
  pcie_info['pcie_metric']['pcie_bandwidth'] is: N/A
  pcie_info['pcie_metric']['pcie_l0_to_recovery_count'] is: N/A
  pcie_info['pcie_metric']['pcie_replay_roll_over_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_sent_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_received_count'] is: N/A

ok
test_power_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_power_info

  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 39
  power_info['gfx_voltage'] is: 856
  power_info['soc_voltage'] is: 937
  power_info['mem_voltage'] is: 843
  power_info['power_limit'] is: 272000000

###Test amdsmi_get_power_cap_info

  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 272000000

###Test amdsmi_is_gpu_power_management_enabled

  Power management enabled: True

ok
test_process_list (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_process_list

  Process list: []

ok
test_processor_type (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_processor_type

  Processor type is: AMDSMI_PROCESSOR_TYPE_AMD_GPU

ok
test_ras_block_features_enabled (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_ras_block_features_enabled

**** [ERROR] | Test: test_ras_block_features_enabled | Caught AmdSmiLibraryException: Error code:
        7 | AMDSMI_STATUS_API_FAILED - API call failed
ok
test_ras_feature_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_ras_feature_info

**** [ERROR] | Test: test_ras_feature_info | Caught AmdSmiLibraryException: Error code:
        2 | AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported
ok
test_socket_info (__main__.TestAmdSmiPythonInterface) ...

###Test amdsmi_get_socket_handles


###Test Socket 0

###Test amdsmi_get_socket_info

  Socket: 0000:43:00

ok
test_temperature_metric (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_temp_metric

  Current temperature for HOTSPOT is: 35
  Current temperature for VRAM is: 32

###Test amdsmi_get_temp_metric

  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100

###Test amdsmi_get_temp_metric

  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105

ok
test_temperature_metric_edge (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_temp_metric

  Current temperature for EDGE is: 31
  Limit (critical) temperature for EDGE is: 100
  Shutdown (emergency) temperature for EDGE is: 105

ok
test_temperature_metric_hbm (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_temp_metric

**** [ERROR] | Test: test_temperature_metric_hbm | Caught AmdSmiLibraryException: Error code:
        2 | AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported
ok
test_temperature_metric_plx (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_temp_metric

  Current temperature for PLX is: 30
  Limit (critical) temperature for PLX is: 30
  Shutdown (emergency) temperature for PLX is: 30

ok
test_utilization_count (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_utilization_count

  Timestamp: 2570588934628027
  Utilization count for AMDSMI_COARSE_GRAIN_GFX_ACTIVITY is: 4294967295
  Utilization count for AMDSMI_COARSE_GRAIN_MEM_ACTIVITY is: 4294967295
  Utilization count for AMDSMI_COARSE_DECODER_ACTIVITY is: 0

  Timestamp: 2570588935626503
  Utilization count for AMDSMI_FINE_GRAIN_GFX_ACTIVITY is: 4294967295
  Utilization count for AMDSMI_FINE_GRAIN_MEM_ACTIVITY is: 4294967295
  Utilization count for AMDSMI_FINE_DECODER_ACTIVITY is: 0

ok
test_vbios_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-V395TRIO-2OC
  vbios_info['build_date'] is: 2021/03/28 21:35
  vbios_info['name'] is: 113-MSITV395MH.132
  vbios_info['version'] is: 020.001.000.060.000000
  vbios_info['boot_firmware'] is: 01234567

ok
test_vendor_name (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_vendor_name

  Vendor name is: Advanced Micro Devices, Inc. [AMD/ATI]

ok
test_walkthrough (__main__.TestAmdSmiPythonInterface) ...

#######################################################################
========> test_walkthrough start <========



###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_asic_info

  asic_info['market_name'] is: NAVI21
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73bf
  asic_info['rev_id'] is: 0xc1
  asic_info['subsystem_id'] is: 0xc34
  asic_info['asic_serial'] is: 0xF8FFEB47A027DE4D
  asic_info['oam_id'] is: N/A
  asic_info['target_graphics_version'] is: gfx1030
  asic_info['num_compute_units'] is: 72

###Test amdsmi_get_gpu_kfd_info

  kfd_info['kfd_id'] is: 16970
  kfd_info['node_id'] is: 1



###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_power_info

  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 31
  power_info['gfx_voltage'] is: 856
  power_info['soc_voltage'] is: 937
  power_info['mem_voltage'] is: 837
  power_info['power_limit'] is: 272000000

###Test amdsmi_get_power_cap_info

  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 272000000

###Test amdsmi_is_gpu_power_management_enabled

  Power management enabled: True



###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-V395TRIO-2OC
  vbios_info['build_date'] is: 2021/03/28 21:35
  vbios_info['name'] is: 113-MSITV395MH.132
  vbios_info['version'] is: 020.001.000.060.000000
  vbios_info['boot_firmware'] is: 01234567



###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_board_info

  board_info['model_number'] is: N/A
  board_info['product_serial'] is: N/A
  board_info['fru_id'] is: N/A
  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]
  board_info['product_name'] is: Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]



###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_fw_info

  FW name:           AMDSMI_FW_ID_CP_CE
  FW version:        37
  FW name:           AMDSMI_FW_ID_CP_PFP
  FW version:        98
  FW name:           AMDSMI_FW_ID_CP_ME
  FW version:        64
  FW name:           AMDSMI_FW_ID_CP_MEC1
  FW version:        118
  FW name:           AMDSMI_FW_ID_CP_MEC2
  FW version:        118
  FW name:           AMDSMI_FW_ID_RLC
  FW version:        96
  FW name:           AMDSMI_FW_ID_SDMA0
  FW version:        83
  FW name:           AMDSMI_FW_ID_SDMA1
  FW version:        83
  FW name:           AMDSMI_FW_ID_VCN
  FW version:        04.11.F0.00
  FW name:           AMDSMI_FW_ID_PSP_SOSDRV
  FW version:        00.21.0E.64
  FW name:           AMDSMI_FW_ID_ASD
  FW version:        553648350
  FW name:           AMDSMI_FW_ID_TA_RAS
  FW version:        1B.00.01.3E
  FW name:           AMDSMI_FW_ID_TA_XGMI
  FW version:        20.00.00.0F
  FW name:           AMDSMI_FW_ID_PM
  FW version:        00.58.90.00
  FW name:           AMDSMI_FW_ID_PLDM_BUNDLE
  FW version:        00.xx.yy.zz



###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_driver_info

Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.9.2', 'driver_date': '2015/01/01 00:00'}


========> test_walkthrough end <========
#######################################################################

ok

----------------------------------------------------------------------
Ran 31 tests in 0.592s

OK
~~~

</details>


### Unittest: filter and verbose
Allow filtering based on common or specific test names. 

```/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "test_walkthrough" -v```  

ex.
<details open>
  <summary>Click for example: <i><b>Unittest: filter and verbose</b></i></summary>

~~~shell
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "test_asic_kfd_info" -v
test_asic_kfd_info (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_gpu_asic_info

  asic_info['market_name'] is: NAVI21
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73bf
  asic_info['rev_id'] is: 0xc1
  asic_info['asic_serial'] is: 0xF8FFEB47A027DE4D
  asic_info['oam_id'] is: N/A
  asic_info['target_graphics_version'] is: gfx1030
  asic_info['num_compute_units'] is: 72

###Test amdsmi_get_gpu_kfd_info

  kfd_info['kfd_id'] is: 16970
  kfd_info['node_id'] is: 1

ok

----------------------------------------------------------------------
Ran 1 test in 0.453s

OK
~~~
</details>

## Run Tests
### Example Runs
Please refer to Python's UnitTest documentation for better overview of commands to run.

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/unit_tests.py -v
test_check_res (__main__.TestAmdSmiPythonBDF) ... ok
test_format_bdf (__main__.TestAmdSmiPythonBDF) ... ok
test_parse_bdf (__main__.TestAmdSmiPythonBDF) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK
```

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "temperature" -v
test_temperature_metric (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_temp_metric

  Current temperature for HOTSPOT is: 33
  Current temperature for VRAM is: 32

###Test amdsmi_get_temp_metric

  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100

###Test amdsmi_get_temp_metric

  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105

ok
test_temperature_metric_edge (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_temp_metric

  Current temperature for EDGE is: 31
  Limit (critical) temperature for EDGE is: 100
  Shutdown (emergency) temperature for EDGE is: 105

ok
test_temperature_metric_hbm (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_temp_metric

**** [ERROR] | Test: test_temperature_metric_hbm | Caught AmdSmiLibraryException: Error code:
        2 | AMDSMI_STATUS_NOT_SUPPORTED - Feature not supported
ok
test_temperature_metric_plx (__main__.TestAmdSmiPythonInterface) ...

###Test Processor 0, bdf: 0000:43:00.0

###Test amdsmi_get_temp_metric

  Current temperature for PLX is: 30
  Limit (critical) temperature for PLX is: 30
  Shutdown (emergency) temperature for PLX is: 30

ok

----------------------------------------------------------------------
Ran 4 tests in 0.466s

OK
```

```shell
/opt/rocm/share/amd_smi/tests/python_unittest/integration_test.py -k "info" -b -v
test_asic_kfd_info (__main__.TestAmdSmiPythonInterface) ... ok
test_bad_page_info (__main__.TestAmdSmiPythonInterface) ... ok
test_board_info (__main__.TestAmdSmiPythonInterface) ... ok
test_clock_info (__main__.TestAmdSmiPythonInterface) ... ok
test_clock_info_vclk0_dclk0 (__main__.TestAmdSmiPythonInterface) ... ok
test_clock_info_vclk1_dclk1 (__main__.TestAmdSmiPythonInterface) ... ok
test_driver_info (__main__.TestAmdSmiPythonInterface) ... ok
test_fw_info (__main__.TestAmdSmiPythonInterface) ... ok
test_pcie_info (__main__.TestAmdSmiPythonInterface) ... ok
test_power_info (__main__.TestAmdSmiPythonInterface) ... ok
test_ras_feature_info (__main__.TestAmdSmiPythonInterface) ... ok
test_socket_info (__main__.TestAmdSmiPythonInterface) ... ok
test_vbios_info (__main__.TestAmdSmiPythonInterface) ... ok

----------------------------------------------------------------------
Ran 13 tests in 0.506s

OK
```