# How to Python Unit Tests
## Overview
We use Python's default Python unittest testing framework. You can read more about it here [Python unittest v3.8](https://docs.python.org/3.8/library/unittest.html). Alternatively, you can read up on pytest through here [Pytest how-to usage](https://docs.pytest.org/en/latest/index.html).

## Warning to Users
AMD SMI Python API tests are subject to change. These tests are currently a work in progress and may not work on your system.

## Pre-Requisites Before Running Tests
Follow our install/build guides to ensure the Python API is installed correctly according to [AMD SMI installation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).

***Versions***: Python 3.8+

## How to Run
### Basic How To
The 2 tests are in this PATH:  
```/opt/rocm/share/amd_smi/tests/pytest/integration_test.py```  
```/opt/rocm/share/amd_smi/tests/pytest/unit_tests.py```


The recommended method to run the tests:  
<u>Pytest verbose</u>  
```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -s -v```  
```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -s -v```

<u>Pytest only (not verbose)</u>  
```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -v```  
```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -v```

<u>Unittest verbose</u>  
```/opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -v```  
```/opt/rocm/share/amd_smi/tests/pytest/integration_test.py -v```

<u>Unittest only (not verbose)</u>  
```/opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -b -v```  
```/opt/rocm/share/amd_smi/tests/pytest/integration_test.py -b -v```

See sections below for more detailed options with examples.

### Unittest or Pytest Run
The Unittest Run calls the tests directly, assuming pytest is correctly installed in the PATH.  
This is more straightforward and intuitive but with less control for options. For example, the cache provider will always be used.

```/opt/rocm/share/amd_smi/tests/pytest/*```

options:
  -  -h, --help           show this help message and exit
  -  -v, --verbose        Verbose output
  -  -q, --quiet          Quiet output
  -  -b, --buffer         Buffer stdout and stderr during tests
  -  -k "TESTNAME"        Only run tests which match the given substring

The Pytest Run could be more reliable and consistent, especially if pytest is not in the PATH.  
This offers more options and flexibility, such as the option to disable the cache provider, ensuring completely independent runs.

```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/*```

options:
  -  -h, --help           show this help message and exit
  -  --co                 Collect and list tests
  -  -p no:cacheprovider  Disable cache provider
  -  -v, --verbose        Verbose output
  -  -q, --quiet          Quiet output
  -  -s, --capture=no     Disables output capturing, stdout output
  -  -k "TESTNAME"        Only run tests which match the given substring

The complete list of options can be accessed here [Pytest command-line flags](https://docs.pytest.org/en/latest/reference/reference.html#command-line-flags).

## Unittest Run Options
### Unittest Run: Verbose on
Helpful to see print outs of Python.

```/opt/rocm/share/amd_smi/tests/pytest/integration_test.py -v```

```/opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -v```

ex.
<details open>
  <summary>Click for example: <i><b>Unittest run: verbose on</i></b></summary>

~~~shell
/opt/rocm/share/amd_smi/tests/pytest/integration_test.py -v
test_init (__main__.TestAmdSmiInit) ... ok
test_bad_page_info (__main__.TestAmdSmiPythonInterface) ... ###Test amdsmi_get_gpu_bad_page_info

**** [ERROR] | Test: test_bad_page_info | Caught AmdSmiLibraryException
ok
test_bdf_device_id (__main__.TestAmdSmiPythonInterface) ... ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['version'] is: 020.001.000.038.015697

  vbios_info['name'] is: NAVI21 Gaming XL D412

###Test amdsmi_get_gpu_device_uuid

  uuid is: 81ff73bf-0000-1000-80c1-6890a5911040
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['version'] is: 020.001.000.060.016898

  vbios_info['name'] is: NAVI21 D43001 GLXL

###Test amdsmi_get_gpu_device_uuid

  uuid is: 1fff73a3-0000-1000-8075-223e5e64eac1
ok
test_ecc (__main__.TestAmdSmiPythonInterface) ... ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_ras_feature_info

**** [ERROR] | Test: test_ecc | Caught AmdSmiLibraryException
ok
test_gpu_performance (__main__.TestAmdSmiPythonInterface) ... ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_activity
  engine_usage['gfx_activity'] is: 3 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 0 %

###Test amdsmi_get_power_info
  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 8
  power_info['gfx_voltage'] is: 768
  power_info['soc_voltage'] is: 918
  power_info['mem_voltage'] is: 1250
  power_info['power_limit'] is: 203000000
###Test amdsmi_is_gpu_power_management_enabled
  Is power management enabled is: True
###Test amdsmi_get_temp_metric
  Current temperature for EDGE is: 42
  Current temperature for HOTSPOT is: 42
  Current temperature for VRAM is: 38
###Test amdsmi_get_temp_metric
  Limit (critical) temperature for EDGE is: 100
  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100
###Test amdsmi_get_temp_metric
  Shutdown (emergency) temperature for EDGE is: 105
  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105
###Test amdsmi_get_clock_info
  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2475
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255
  Current clock for domain VCLK0 is: 0
  Max clock for domain VCLK0 is: 0
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain VCLK1 is: 0
  Max clock for domain VCLK1 is: 0
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 0
  Max clock for domain DCLK0 is: 0
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 0
  Max clock for domain DCLK1 is: 0
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255
###Test amdsmi_get_pcie_info
  pcie_info['pcie_metric']['pcie_width'] is: 4
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 5000 MT/s
  pcie_info['pcie_static']['max_pcie_speed'] is: 16000
  pcie_info['pcie_static']['pcie_interface_version'] is: 4
  pcie_info['pcie_static']['slot_type'] is: CEM
  pcie_info['pcie_metric']['pcie_replay_count'] is: N/A
  pcie_info['pcie_metric']['pcie_bandwidth'] is: N/A
  pcie_info['pcie_metric']['pcie_l0_to_recovery_count'] is: N/A
  pcie_info['pcie_metric']['pcie_replay_roll_over_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_sent_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_received_count'] is: N/A
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_activity
  engine_usage['gfx_activity'] is: 0 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 0 %

###Test amdsmi_get_power_info
  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 13
  power_info['gfx_voltage'] is: 781
  power_info['soc_voltage'] is: 812
  power_info['mem_voltage'] is: 1250
  power_info['power_limit'] is: 213000000
###Test amdsmi_is_gpu_power_management_enabled
  Is power management enabled is: True
###Test amdsmi_get_temp_metric
  Current temperature for EDGE is: 34
  Current temperature for HOTSPOT is: 38
  Current temperature for VRAM is: 36
###Test amdsmi_get_temp_metric
  Limit (critical) temperature for EDGE is: 109
  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100
###Test amdsmi_get_temp_metric
  Shutdown (emergency) temperature for EDGE is: 114
  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105
###Test amdsmi_get_clock_info
  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2555
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255
  Current clock for domain VCLK0 is: 0
  Max clock for domain VCLK0 is: 0
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain VCLK1 is: 0
  Max clock for domain VCLK1 is: 0
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 0
  Max clock for domain DCLK0 is: 0
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 0
  Max clock for domain DCLK1 is: 0
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255
###Test amdsmi_get_pcie_info
  pcie_info['pcie_metric']['pcie_width'] is: 16
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 8000 MT/s
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
test_walkthrough (__main__.TestAmdSmiPythonInterface) ... ###Test amdsmi_get_processor_handles()
###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = 0
###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_asic_info
  asic_info['market_name'] is: NAVI21
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73bf
  asic_info['rev_id'] is: 0xc3

  asic_info['asic_serial'] is: 0x81C16890A5911040

  asic_info['oam_id'] is: N/A

###Test amdsmi_get_power_cap_info
  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 203000000

###Test amdsmi_get_gpu_vbios_info
  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['name'] is: NAVI21 Gaming XL D412

  vbios_info['version'] is: 020.001.000.038.015697

###Test amdsmi_get_gpu_board_info
  board_info['model_number'] is: N/A

  board_info['product_serial'] is: N/A

  board_info['fru_id'] is: N/A

  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]

  board_info['product_name'] is: Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]

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
FW version:        31.1E.00.8
FW name:           AMDSMI_FW_ID_PSP_SOSDRV
FW version:        21.0E.64
FW name:           AMDSMI_FW_ID_ASD
FW version:        553648340
FW name:           AMDSMI_FW_ID_TA_RAS
FW version:        1B.00.01.3E
FW name:           AMDSMI_FW_ID_TA_XGMI
FW version:        20.00.00.0F
FW name:           AMDSMI_FW_ID_PM
FW version:        58.89.0
###Test amdsmi_get_gpu_driver_info
Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.7.8', 'driver_date': '2015/01/01 00:00'}
###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = 0
###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = 1
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_asic_info
  asic_info['market_name'] is: Navi 21 GL-XL [Radeon PRO W6800]
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73a3
  asic_info['rev_id'] is: 0x00

  asic_info['asic_serial'] is: 0x1F75223E5E64EAC1

  asic_info['oam_id'] is: N/A

###Test amdsmi_get_power_cap_info
  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 213000000

###Test amdsmi_get_gpu_vbios_info
  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['name'] is: NAVI21 D43001 GLXL

  vbios_info['version'] is: 020.001.000.060.016898

###Test amdsmi_get_gpu_board_info
  board_info['model_number'] is: N/A

  board_info['product_serial'] is: N/A

  board_info['fru_id'] is: N/A

  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]

  board_info['product_name'] is: Navi 21 GL-XL [Radeon PRO W6800]

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
FW version:        31.1E.00.8
FW name:           AMDSMI_FW_ID_PSP_SOSDRV
FW version:        21.0E.64
FW name:           AMDSMI_FW_ID_ASD
FW version:        553648340
FW name:           AMDSMI_FW_ID_TA_RAS
FW version:        1B.00.01.3E
FW name:           AMDSMI_FW_ID_TA_XGMI
FW version:        20.00.00.0F
FW name:           AMDSMI_FW_ID_PM
FW version:        58.89.0
###Test amdsmi_get_gpu_driver_info
Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.7.8', 'driver_date': '2015/01/01 00:00'}
###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = 1
ok

----------------------------------------------------------------------
Ran 6 tests in 0.083s

OK

~~~

</details>


### Unittest Run: Verbose on + Filter (or exclude) a test

```/opt/rocm/share/amd_smi/tests/pytest/integration_test.py -k "test_walkthrough" -v```

```/opt/rocm/share/amd_smi/tests/pytest/integration_test.py -k "not test_walkthrough" -v```

ex.
<details open>
  <summary>Click for example: <i><b>Unittest Run: Verbose on + Filter (or exclude) a Test</b></i></summary>

~~~shell
> /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -k "test_bdf_device_id" -v
test_bdf_device_id (__main__.TestAmdSmiPythonInterface) ... ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['version'] is: 020.001.000.038.015697

  vbios_info['name'] is: NAVI21 Gaming XL D412

###Test amdsmi_get_gpu_device_uuid

  uuid is: 81ff73bf-0000-1000-80c1-6890a5911040
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['version'] is: 020.001.000.060.016898

  vbios_info['name'] is: NAVI21 D43001 GLXL

###Test amdsmi_get_gpu_device_uuid

  uuid is: 1fff73a3-0000-1000-8075-223e5e64eac1
ok

----------------------------------------------------------------------
Ran 1 test in 0.012s

OK
~~~
</details>


### Unittest Run: Silence stdout (print statements) and run all tests
 Runs all tests. Silence print statements to stdout. Lists tests results.
 This is also the best way to list all tests available.

```/opt/rocm/share/amd_smi/tests/pytest/integration_test.py -b -v```

```/opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -b -v```

ex.
<details open>
  <summary>Click for example: <i><b>Unittest Run: Silence stdout (print statements) and run all tests</i></b></summary>

~~~shell
/opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -b -v
test_check_res (__main__.TestAmdSmiPythonBDF) ... ok
test_format_bdf (__main__.TestAmdSmiPythonBDF) ... ok
test_parse_bdf (__main__.TestAmdSmiPythonBDF) ... ok

----------------------------------------------------------------------
Ran 3 tests in 0.001s

OK
~~~

</details>

## Pytest Run Options
### Pytest: List tests
```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py --co```

```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/unit_tests.py --co```

ex.
<details open>
  <summary>Click for example: <b><i>Pytest: List tests</i></b></summary>

~~~shell
python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py --co
===================================================== test session starts =====================================================
platform linux -- Python 3.8.10, pytest-8.2.2, pluggy-1.5.0
rootdir: /opt/rocm/share/amd_smi
configfile: pyproject.toml
collected 6 items

<Dir amd_smi>
  <Dir tests>
    <Package pytest>
      <Module integration_test.py>
        <UnitTestCase TestAmdSmiInit>
          <TestCaseFunction test_init>
        <UnitTestCase TestAmdSmiPythonInterface>
          <TestCaseFunction test_bad_page_info>
          <TestCaseFunction test_bdf_device_id>
          <TestCaseFunction test_ecc>
          <TestCaseFunction test_gpu_performance>
          <TestCaseFunction test_walkthrough>

================================================= 6 tests collected in 0.04s ==================================================
~~~
</details>

### Pytest Run: Verbose on
```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -v```

```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -v```

ex.
<details open>
  <summary>Click for example: <b><i>Pytest Run: verbose on</i></b></summary>

~~~shell
 python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -v
===================================================== test session starts =====================================================
platform linux -- Python 3.8.10, pytest-8.2.2, pluggy-1.5.0 -- /usr/bin/python3
rootdir: /opt/rocm/share/amd_smi
configfile: pyproject.toml
collected 3 items

../../opt/rocm/share/amd_smi/tests/pytest/unit_tests.py::TestAmdSmiPythonBDF::test_check_res PASSED                     [ 33%]
../../opt/rocm/share/amd_smi/tests/pytest/unit_tests.py::TestAmdSmiPythonBDF::test_format_bdf PASSED                    [ 66%]
../../opt/rocm/share/amd_smi/tests/pytest/unit_tests.py::TestAmdSmiPythonBDF::test_parse_bdf PASSED                     [100%]

====================================================== 3 passed in 0.04s ======================================================
~~~
</details>

### Pytest Run: Verbose on + stdout (print statements)
```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -s -v```

```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -s -v```

ex.
<details open>
  <summary>Click for example: <b><i> Pytest Run: verbose on + stdout (print statements)</i></b></summary>

~~~shell
python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -s -v
===================================================== test session starts =====================================================
platform linux -- Python 3.8.10, pytest-8.2.2, pluggy-1.5.0 -- /usr/bin/python3
rootdir: /opt/rocm/share/amd_smi
configfile: pyproject.toml
collected 6 items

../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiInit::test_init PASSED
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_bad_page_info ###Test amdsmi_get_gpu_bad_page_info

**** [ERROR] | Test: test_bad_page_info | Caught AmdSmiLibraryException
PASSED
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_bdf_device_id ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['version'] is: 020.001.000.038.015697

  vbios_info['name'] is: NAVI21 Gaming XL D412

###Test amdsmi_get_gpu_device_uuid

  uuid is: 81ff73bf-0000-1000-80c1-6890a5911040
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['version'] is: 020.001.000.060.016898

  vbios_info['name'] is: NAVI21 D43001 GLXL

###Test amdsmi_get_gpu_device_uuid

  uuid is: 1fff73a3-0000-1000-8075-223e5e64eac1
PASSED
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_ecc ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_ras_feature_info

**** [ERROR] | Test: test_ecc | Caught AmdSmiLibraryException
PASSED
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_gpu_performance ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_activity
  engine_usage['gfx_activity'] is: 1 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 0 %

###Test amdsmi_get_power_info
  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 8
  power_info['gfx_voltage'] is: 768
  power_info['soc_voltage'] is: 918
  power_info['mem_voltage'] is: 1250
  power_info['power_limit'] is: 203000000
###Test amdsmi_is_gpu_power_management_enabled
  Is power management enabled is: True
###Test amdsmi_get_temp_metric
  Current temperature for EDGE is: 42
  Current temperature for HOTSPOT is: 43
  Current temperature for VRAM is: 38
###Test amdsmi_get_temp_metric
  Limit (critical) temperature for EDGE is: 100
  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100
###Test amdsmi_get_temp_metric
  Shutdown (emergency) temperature for EDGE is: 105
  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105
###Test amdsmi_get_clock_info
  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2475
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255
  Current clock for domain VCLK0 is: 0
  Max clock for domain VCLK0 is: 0
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain VCLK1 is: 0
  Max clock for domain VCLK1 is: 0
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 0
  Max clock for domain DCLK0 is: 0
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 0
  Max clock for domain DCLK1 is: 0
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255
###Test amdsmi_get_pcie_info
  pcie_info['pcie_metric']['pcie_width'] is: 4
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 5000 MT/s
  pcie_info['pcie_static']['max_pcie_speed'] is: 16000
  pcie_info['pcie_static']['pcie_interface_version'] is: 4
  pcie_info['pcie_static']['slot_type'] is: CEM
  pcie_info['pcie_metric']['pcie_replay_count'] is: N/A
  pcie_info['pcie_metric']['pcie_bandwidth'] is: N/A
  pcie_info['pcie_metric']['pcie_l0_to_recovery_count'] is: N/A
  pcie_info['pcie_metric']['pcie_replay_roll_over_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_sent_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_received_count'] is: N/A
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_activity
  engine_usage['gfx_activity'] is: 0 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 0 %

###Test amdsmi_get_power_info
  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 13
  power_info['gfx_voltage'] is: 787
  power_info['soc_voltage'] is: 806
  power_info['mem_voltage'] is: 1250
  power_info['power_limit'] is: 213000000
###Test amdsmi_is_gpu_power_management_enabled
  Is power management enabled is: True
###Test amdsmi_get_temp_metric
  Current temperature for EDGE is: 34
  Current temperature for HOTSPOT is: 37
  Current temperature for VRAM is: 36
###Test amdsmi_get_temp_metric
  Limit (critical) temperature for EDGE is: 109
  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100
###Test amdsmi_get_temp_metric
  Shutdown (emergency) temperature for EDGE is: 114
  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105
###Test amdsmi_get_clock_info
  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2555
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255
  Current clock for domain VCLK0 is: 0
  Max clock for domain VCLK0 is: 0
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain VCLK1 is: 0
  Max clock for domain VCLK1 is: 0
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 0
  Max clock for domain DCLK0 is: 0
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 0
  Max clock for domain DCLK1 is: 0
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255
###Test amdsmi_get_pcie_info
  pcie_info['pcie_metric']['pcie_width'] is: 16
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 8000 MT/s
  pcie_info['pcie_static']['max_pcie_speed'] is: 16000
  pcie_info['pcie_static']['pcie_interface_version'] is: 4
  pcie_info['pcie_static']['slot_type'] is: CEM
  pcie_info['pcie_metric']['pcie_replay_count'] is: N/A
  pcie_info['pcie_metric']['pcie_bandwidth'] is: N/A
  pcie_info['pcie_metric']['pcie_l0_to_recovery_count'] is: N/A
  pcie_info['pcie_metric']['pcie_replay_roll_over_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_sent_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_received_count'] is: N/A
PASSED
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_walkthrough ###Test amdsmi_get_processor_handles()
###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = 0
###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_asic_info
  asic_info['market_name'] is: NAVI21
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73bf
  asic_info['rev_id'] is: 0xc3

  asic_info['asic_serial'] is: 0x81C16890A5911040

  asic_info['oam_id'] is: N/A

###Test amdsmi_get_power_cap_info
  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 203000000

###Test amdsmi_get_gpu_vbios_info
  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['name'] is: NAVI21 Gaming XL D412

  vbios_info['version'] is: 020.001.000.038.015697

###Test amdsmi_get_gpu_board_info
  board_info['model_number'] is: N/A

  board_info['product_serial'] is: N/A

  board_info['fru_id'] is: N/A

  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]

  board_info['product_name'] is: Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]

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
FW version:        31.1E.00.8
FW name:           AMDSMI_FW_ID_PSP_SOSDRV
FW version:        21.0E.64
FW name:           AMDSMI_FW_ID_ASD
FW version:        553648340
FW name:           AMDSMI_FW_ID_TA_RAS
FW version:        1B.00.01.3E
FW name:           AMDSMI_FW_ID_TA_XGMI
FW version:        20.00.00.0F
FW name:           AMDSMI_FW_ID_PM
FW version:        58.89.0
###Test amdsmi_get_gpu_driver_info
Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.7.8', 'driver_date': '2015/01/01 00:00'}
###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = 0
###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = 1
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_asic_info
  asic_info['market_name'] is: Navi 21 GL-XL [Radeon PRO W6800]
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73a3
  asic_info['rev_id'] is: 0x00

  asic_info['asic_serial'] is: 0x1F75223E5E64EAC1

  asic_info['oam_id'] is: N/A

###Test amdsmi_get_power_cap_info
  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 213000000

###Test amdsmi_get_gpu_vbios_info
  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['name'] is: NAVI21 D43001 GLXL

  vbios_info['version'] is: 020.001.000.060.016898

###Test amdsmi_get_gpu_board_info
  board_info['model_number'] is: N/A

  board_info['product_serial'] is: N/A

  board_info['fru_id'] is: N/A

  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]

  board_info['product_name'] is: Navi 21 GL-XL [Radeon PRO W6800]

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
FW version:        31.1E.00.8
FW name:           AMDSMI_FW_ID_PSP_SOSDRV
FW version:        21.0E.64
FW name:           AMDSMI_FW_ID_ASD
FW version:        553648340
FW name:           AMDSMI_FW_ID_TA_RAS
FW version:        1B.00.01.3E
FW name:           AMDSMI_FW_ID_TA_XGMI
FW version:        20.00.00.0F
FW name:           AMDSMI_FW_ID_PM
FW version:        58.89.0
###Test amdsmi_get_gpu_driver_info
Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.7.8', 'driver_date': '2015/01/01 00:00'}
###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = 1
PASSED

====================================================== 6 passed in 0.13s ======================================================
~~~
</details>

### Pytest Run: Verbose on + Filter (or exclude) a Test
Use [Pytest: List tests](###-Pytest:-List-tests) then either exclude (with "not") or only run the specified test.

```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -k "test_gpu_performance" -v```

```python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -k "not test_gpu_performance" -v```

ex.
<details open>
  <summary>Click for example: <b><i>Pytest Run: Verbose on + Filter (or exclude) a Test</i></b></summary>

~~~shell
python3 -m pytest -p no:cacheprovider /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -k "not test_gpu_performance" -v
===================================================== test session starts =====================================================
platform linux -- Python 3.8.10, pytest-8.2.2, pluggy-1.5.0 -- /usr/bin/python3
rootdir: /opt/rocm/share/amd_smi
configfile: pyproject.toml
collected 6 items / 1 deselected / 5 selected

../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiInit::test_init PASSED                         [ 20%]
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_bad_page_info PASSED     [ 40%]
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_bdf_device_id PASSED     [ 60%]
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_ecc PASSED               [ 80%]
../../opt/rocm/share/amd_smi/tests/pytest/integration_test.py::TestAmdSmiPythonInterface::test_walkthrough PASSED       [100%]

=============================================== 5 passed, 1 deselected in 0.09s ===============================================
~~~
</details>

## Run Tests
### Example Runs
Please refer to Python's UnitTest documentation for better overview of commands to run.

```shell
python3 /opt/rocm/share/amd_smi/tests/pytest/unit_tests.py -v
test_check_res (tests.amd_smi_test.py-test.unit_tests.TestAmdSmiPythonBDF) ... ok
test_format_bdf (tests.amd_smi_test.py-test.unit_tests.TestAmdSmiPythonBDF) ... ok
test_parse_bdf (tests.amd_smi_test.py-test.unit_tests.TestAmdSmiPythonBDF) ... ok
```

```shell
python3 /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -v
test_init (__main__.TestAmdSmiInit) ... ok
test_bad_page_info (__main__.TestAmdSmiPythonInterface) ... ###Test amdsmi_get_gpu_bad_page_info

**** [ERROR] | Test: test_bad_page_info | Caught AmdSmiLibraryException
ok
test_bdf_device_id (__main__.TestAmdSmiPythonInterface) ... ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['version'] is: 020.001.000.038.015697

  vbios_info['name'] is: NAVI21 Gaming XL D412

###Test amdsmi_get_gpu_device_uuid

  uuid is: 81ff73bf-0000-1000-80c1-6890a5911040
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['version'] is: 020.001.000.060.016898

  vbios_info['name'] is: NAVI21 D43001 GLXL

###Test amdsmi_get_gpu_device_uuid

  uuid is: 1fff73a3-0000-1000-8075-223e5e64eac1
ok
test_ecc (__main__.TestAmdSmiPythonInterface) ... ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_ras_feature_info

**** [ERROR] | Test: test_ecc | Caught AmdSmiLibraryException
ok
test_gpu_performance (__main__.TestAmdSmiPythonInterface) ... ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_activity
  engine_usage['gfx_activity'] is: 5 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 0 %

###Test amdsmi_get_power_info
  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 8
  power_info['gfx_voltage'] is: 768
  power_info['soc_voltage'] is: 918
  power_info['mem_voltage'] is: 1250
  power_info['power_limit'] is: 203000000
###Test amdsmi_is_gpu_power_management_enabled
  Is power management enabled is: True
###Test amdsmi_get_temp_metric
  Current temperature for EDGE is: 41
  Current temperature for HOTSPOT is: 42
  Current temperature for VRAM is: 38
###Test amdsmi_get_temp_metric
  Limit (critical) temperature for EDGE is: 100
  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100
###Test amdsmi_get_temp_metric
  Shutdown (emergency) temperature for EDGE is: 105
  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105
###Test amdsmi_get_clock_info
  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2475
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255
  Current clock for domain VCLK0 is: 0
  Max clock for domain VCLK0 is: 0
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain VCLK1 is: 0
  Max clock for domain VCLK1 is: 0
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 0
  Max clock for domain DCLK0 is: 0
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 0
  Max clock for domain DCLK1 is: 0
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255
###Test amdsmi_get_pcie_info
  pcie_info['pcie_metric']['pcie_width'] is: 4
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 5000 MT/s
  pcie_info['pcie_static']['max_pcie_speed'] is: 16000
  pcie_info['pcie_static']['pcie_interface_version'] is: 4
  pcie_info['pcie_static']['slot_type'] is: CEM
  pcie_info['pcie_metric']['pcie_replay_count'] is: N/A
  pcie_info['pcie_metric']['pcie_bandwidth'] is: N/A
  pcie_info['pcie_metric']['pcie_l0_to_recovery_count'] is: N/A
  pcie_info['pcie_metric']['pcie_replay_roll_over_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_sent_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_received_count'] is: N/A
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_activity
  engine_usage['gfx_activity'] is: 0 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 0 %

###Test amdsmi_get_power_info
  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 12
  power_info['gfx_voltage'] is: 787
  power_info['soc_voltage'] is: 806
  power_info['mem_voltage'] is: 1250
  power_info['power_limit'] is: 213000000
###Test amdsmi_is_gpu_power_management_enabled
  Is power management enabled is: True
###Test amdsmi_get_temp_metric
  Current temperature for EDGE is: 33
  Current temperature for HOTSPOT is: 37
  Current temperature for VRAM is: 36
###Test amdsmi_get_temp_metric
  Limit (critical) temperature for EDGE is: 109
  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100
###Test amdsmi_get_temp_metric
  Shutdown (emergency) temperature for EDGE is: 114
  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105
###Test amdsmi_get_clock_info
  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2555
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255
  Current clock for domain VCLK0 is: 0
  Max clock for domain VCLK0 is: 0
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain VCLK1 is: 0
  Max clock for domain VCLK1 is: 0
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 0
  Max clock for domain DCLK0 is: 0
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 0
  Max clock for domain DCLK1 is: 0
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255
###Test amdsmi_get_pcie_info
  pcie_info['pcie_metric']['pcie_width'] is: 16
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 8000 MT/s
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
test_walkthrough (__main__.TestAmdSmiPythonInterface) ... ###Test amdsmi_get_processor_handles()
###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = 0
###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_asic_info
  asic_info['market_name'] is: NAVI21
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73bf
  asic_info['rev_id'] is: 0xc3

  asic_info['asic_serial'] is: 0x81C16890A5911040

  asic_info['oam_id'] is: N/A

###Test amdsmi_get_power_cap_info
  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 203000000

###Test amdsmi_get_gpu_vbios_info
  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['name'] is: NAVI21 Gaming XL D412

  vbios_info['version'] is: 020.001.000.038.015697

###Test amdsmi_get_gpu_board_info
  board_info['model_number'] is: N/A

  board_info['product_serial'] is: N/A

  board_info['fru_id'] is: N/A

  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]

  board_info['product_name'] is: Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]

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
FW version:        31.1E.00.8
FW name:           AMDSMI_FW_ID_PSP_SOSDRV
FW version:        21.0E.64
FW name:           AMDSMI_FW_ID_ASD
FW version:        553648340
FW name:           AMDSMI_FW_ID_TA_RAS
FW version:        1B.00.01.3E
FW name:           AMDSMI_FW_ID_TA_XGMI
FW version:        20.00.00.0F
FW name:           AMDSMI_FW_ID_PM
FW version:        58.89.0
###Test amdsmi_get_gpu_driver_info
Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.7.8', 'driver_date': '2015/01/01 00:00'}
###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = 0
###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = 1
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_asic_info
  asic_info['market_name'] is: Navi 21 GL-XL [Radeon PRO W6800]
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73a3
  asic_info['rev_id'] is: 0x00

  asic_info['asic_serial'] is: 0x1F75223E5E64EAC1

  asic_info['oam_id'] is: N/A

###Test amdsmi_get_power_cap_info
  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 213000000

###Test amdsmi_get_gpu_vbios_info
  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['name'] is: NAVI21 D43001 GLXL

  vbios_info['version'] is: 020.001.000.060.016898

###Test amdsmi_get_gpu_board_info
  board_info['model_number'] is: N/A

  board_info['product_serial'] is: N/A

  board_info['fru_id'] is: N/A

  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]

  board_info['product_name'] is: Navi 21 GL-XL [Radeon PRO W6800]

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
FW version:        31.1E.00.8
FW name:           AMDSMI_FW_ID_PSP_SOSDRV
FW version:        21.0E.64
FW name:           AMDSMI_FW_ID_ASD
FW version:        553648340
FW name:           AMDSMI_FW_ID_TA_RAS
FW version:        1B.00.01.3E
FW name:           AMDSMI_FW_ID_TA_XGMI
FW version:        20.00.00.0F
FW name:           AMDSMI_FW_ID_PM
FW version:        58.89.0
###Test amdsmi_get_gpu_driver_info
Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.7.8', 'driver_date': '2015/01/01 00:00'}
###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = 1
ok

----------------------------------------------------------------------
Ran 6 tests in 0.077s

OK
```

```shell
(Tue Jul-7 12:07:47am)-(CPU 0.3%:0:Net 18)-(charpoag@mlsetools2:/opt/rocm/share/amd_smi/tests/pytest)-(44K:3)
> python3 -m pytest -s -ra -vvv -p no:cacheprovider
==================================== test session starts =====================================
platform linux -- Python 3.8.10, pytest-8.2.2, pluggy-1.5.0 -- /usr/bin/python3
rootdir: /opt/rocm/share/amd_smi
configfile: pyproject.toml
collected 6 items

integration_test.py::TestAmdSmiInit::test_init PASSED
integration_test.py::TestAmdSmiPythonInterface::test_bad_page_info ###Test amdsmi_get_gpu_bad_page_info

**** [ERROR] | Test: test_bad_page_info | Caught AmdSmiLibraryException
PASSED
integration_test.py::TestAmdSmiPythonInterface::test_bdf_device_id ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['version'] is: 020.001.000.038.015697

  vbios_info['name'] is: NAVI21 Gaming XL D412

###Test amdsmi_get_gpu_device_uuid

  uuid is: 81ff73bf-0000-1000-80c1-6890a5911040
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_vbios_info

  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['version'] is: 020.001.000.060.016898

  vbios_info['name'] is: NAVI21 D43001 GLXL

###Test amdsmi_get_gpu_device_uuid

  uuid is: 1fff73a3-0000-1000-8075-223e5e64eac1
PASSED
integration_test.py::TestAmdSmiPythonInterface::test_ecc ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_ras_feature_info

**** [ERROR] | Test: test_ecc | Caught AmdSmiLibraryException
PASSED
integration_test.py::TestAmdSmiPythonInterface::test_gpu_performance ###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_activity
  engine_usage['gfx_activity'] is: 3 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 0 %

###Test amdsmi_get_power_info
  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 8
  power_info['gfx_voltage'] is: 768
  power_info['soc_voltage'] is: 918
  power_info['mem_voltage'] is: 1250
  power_info['power_limit'] is: 203000000
###Test amdsmi_is_gpu_power_management_enabled
  Is power management enabled is: True
###Test amdsmi_get_temp_metric
  Current temperature for EDGE is: 44
  Current temperature for HOTSPOT is: 45
  Current temperature for VRAM is: 40
###Test amdsmi_get_temp_metric
  Limit (critical) temperature for EDGE is: 100
  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100
###Test amdsmi_get_temp_metric
  Shutdown (emergency) temperature for EDGE is: 105
  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105
###Test amdsmi_get_clock_info
  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2475
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255
  Current clock for domain VCLK0 is: 0
  Max clock for domain VCLK0 is: 0
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain VCLK1 is: 0
  Max clock for domain VCLK1 is: 0
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 0
  Max clock for domain DCLK0 is: 0
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 0
  Max clock for domain DCLK1 is: 0
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255
###Test amdsmi_get_pcie_info
  pcie_info['pcie_metric']['pcie_width'] is: 4
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 5000 MT/s
  pcie_info['pcie_static']['max_pcie_speed'] is: 16000
  pcie_info['pcie_static']['pcie_interface_version'] is: 4
  pcie_info['pcie_static']['slot_type'] is: CEM
  pcie_info['pcie_metric']['pcie_replay_count'] is: N/A
  pcie_info['pcie_metric']['pcie_bandwidth'] is: N/A
  pcie_info['pcie_metric']['pcie_l0_to_recovery_count'] is: N/A
  pcie_info['pcie_metric']['pcie_replay_roll_over_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_sent_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_received_count'] is: N/A
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_activity
  engine_usage['gfx_activity'] is: 0 %
  engine_usage['umc_activity'] is: 0 %
  engine_usage['mm_activity'] is: 0 %

###Test amdsmi_get_power_info
  power_info['current_socket_power'] is: N/A
  power_info['average_socket_power'] is: 13
  power_info['gfx_voltage'] is: 781
  power_info['soc_voltage'] is: 806
  power_info['mem_voltage'] is: 1250
  power_info['power_limit'] is: 213000000
###Test amdsmi_is_gpu_power_management_enabled
  Is power management enabled is: True
###Test amdsmi_get_temp_metric
  Current temperature for EDGE is: 36
  Current temperature for HOTSPOT is: 39
  Current temperature for VRAM is: 38
###Test amdsmi_get_temp_metric
  Limit (critical) temperature for EDGE is: 109
  Limit (critical) temperature for HOTSPOT is: 110
  Limit (critical) temperature for VRAM is: 100
###Test amdsmi_get_temp_metric
  Shutdown (emergency) temperature for EDGE is: 114
  Shutdown (emergency) temperature for HOTSPOT is: 115
  Shutdown (emergency) temperature for VRAM is: 105
###Test amdsmi_get_clock_info
  Current clock for domain GFX is: 500
  Max clock for domain GFX is: 2555
  Min clock for domain GFX is: 500
  Is GFX clock locked: 0
  Is GFX clock in deep sleep: 255
  Current clock for domain MEM is: 96
  Max clock for domain MEM is: 1000
  Min clock for domain MEM is: 96
  Is MEM clock in deep sleep: 255
  Current clock for domain VCLK0 is: 0
  Max clock for domain VCLK0 is: 0
  Min clock for domain VCLK0 is: 0
  Is VCLK0 clock in deep sleep: 255
  Current clock for domain VCLK1 is: 0
  Max clock for domain VCLK1 is: 0
  Min clock for domain VCLK1 is: 0
  Is VCLK1 clock in deep sleep: 255
  Current clock for domain DCLK0 is: 0
  Max clock for domain DCLK0 is: 0
  Min clock for domain DCLK0 is: 0
  Is DCLK0 clock in deep sleep: 255
  Current clock for domain DCLK1 is: 0
  Max clock for domain DCLK1 is: 0
  Min clock for domain DCLK1 is: 0
  Is DCLK1 clock in deep sleep: 255
###Test amdsmi_get_pcie_info
  pcie_info['pcie_metric']['pcie_width'] is: 16
  pcie_info['pcie_static']['max_pcie_width'] is: 16
  pcie_info['pcie_metric']['pcie_speed'] is: 8000 MT/s
  pcie_info['pcie_static']['max_pcie_speed'] is: 16000
  pcie_info['pcie_static']['pcie_interface_version'] is: 4
  pcie_info['pcie_static']['slot_type'] is: CEM
  pcie_info['pcie_metric']['pcie_replay_count'] is: N/A
  pcie_info['pcie_metric']['pcie_bandwidth'] is: N/A
  pcie_info['pcie_metric']['pcie_l0_to_recovery_count'] is: N/A
  pcie_info['pcie_metric']['pcie_replay_roll_over_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_sent_count'] is: N/A
  pcie_info['pcie_metric']['pcie_nak_received_count'] is: N/A
PASSED
integration_test.py::TestAmdSmiPythonInterface::test_walkthrough ###Test amdsmi_get_processor_handles()
###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = 0
###Test Processor 0, bdf: 0000:08:00.0

###Test amdsmi_get_gpu_asic_info
  asic_info['market_name'] is: NAVI21
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73bf
  asic_info['rev_id'] is: 0xc3

  asic_info['asic_serial'] is: 0x81C16890A5911040

  asic_info['oam_id'] is: N/A

###Test amdsmi_get_power_cap_info
  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 203000000

###Test amdsmi_get_gpu_vbios_info
  vbios_info['part_number'] is: 113-D41207XL-038
  vbios_info['build_date'] is: 2020/10/06 17:59
  vbios_info['name'] is: NAVI21 Gaming XL D412

  vbios_info['version'] is: 020.001.000.038.015697

###Test amdsmi_get_gpu_board_info
  board_info['model_number'] is: N/A

  board_info['product_serial'] is: N/A

  board_info['fru_id'] is: N/A

  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]

  board_info['product_name'] is: Navi 21 [Radeon RX 6800/6800 XT / 6900 XT]

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
FW version:        31.1E.00.8
FW name:           AMDSMI_FW_ID_PSP_SOSDRV
FW version:        21.0E.64
FW name:           AMDSMI_FW_ID_ASD
FW version:        553648340
FW name:           AMDSMI_FW_ID_TA_RAS
FW version:        1B.00.01.3E
FW name:           AMDSMI_FW_ID_TA_XGMI
FW version:        20.00.00.0F
FW name:           AMDSMI_FW_ID_PM
FW version:        58.89.0
###Test amdsmi_get_gpu_driver_info
Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.7.8', 'driver_date': '2015/01/01 00:00'}
###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = 0
###Test amdsmi_get_gpu_device_bdf() | START walk_through | processor i = 1
###Test Processor 1, bdf: 0000:44:00.0

###Test amdsmi_get_gpu_asic_info
  asic_info['market_name'] is: Navi 21 GL-XL [Radeon PRO W6800]
  asic_info['vendor_id'] is: 0x1002
  asic_info['vendor_name'] is: Advanced Micro Devices Inc. [AMD/ATI]
  asic_info['device_id'] is: 0x73a3
  asic_info['rev_id'] is: 0x00

  asic_info['asic_serial'] is: 0x1F75223E5E64EAC1

  asic_info['oam_id'] is: N/A

###Test amdsmi_get_power_cap_info
  power_info['dpm_cap'] is: 1
  power_info['power_cap'] is: 213000000

###Test amdsmi_get_gpu_vbios_info
  vbios_info['part_number'] is: 113-D4300100-100
  vbios_info['build_date'] is: 2021/04/22 09:34
  vbios_info['name'] is: NAVI21 D43001 GLXL

  vbios_info['version'] is: 020.001.000.060.016898

###Test amdsmi_get_gpu_board_info
  board_info['model_number'] is: N/A

  board_info['product_serial'] is: N/A

  board_info['fru_id'] is: N/A

  board_info['manufacturer_name'] is: Advanced Micro Devices, Inc. [AMD/ATI]

  board_info['product_name'] is: Navi 21 GL-XL [Radeon PRO W6800]

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
FW version:        31.1E.00.8
FW name:           AMDSMI_FW_ID_PSP_SOSDRV
FW version:        21.0E.64
FW name:           AMDSMI_FW_ID_ASD
FW version:        553648340
FW name:           AMDSMI_FW_ID_TA_RAS
FW version:        1B.00.01.3E
FW name:           AMDSMI_FW_ID_TA_XGMI
FW version:        20.00.00.0F
FW name:           AMDSMI_FW_ID_PM
FW version:        58.89.0
###Test amdsmi_get_gpu_driver_info
Driver info:  {'driver_name': 'amdgpu', 'driver_version': '6.7.8', 'driver_date': '2015/01/01 00:00'}
###Test amdsmi_get_gpu_driver_info() | END walk_through | processor i = 1
PASSED

===================================== 6 passed in 0.10s ======================================
```

```shell
$ python3 /opt/rocm/share/amd_smi/tests/pytest/integration_test.py -k "*test_init" -vvv
test_init (__main__.TestAmdSmiInit) ... ok

----------------------------------------------------------------------
Ran 1 test in 0.009s

OK

```

```shell
(Tue Jul-7 12:10:10am)-(CPU 0.3%:0:Net 16)-(charpoag@mlsetools2:/opt/rocm/share/amd_smi/tests/pytest)-(44K:3)
> python3 -m pytest -ra -vvv -p no:cacheprovider
==================================== test session starts =====================================
platform linux -- Python 3.8.10, pytest-8.2.2, pluggy-1.5.0 -- /usr/bin/python3
rootdir: /opt/rocm/share/amd_smi
configfile: pyproject.toml
collected 6 items

integration_test.py::TestAmdSmiInit::test_init PASSED                                  [ 16%]
integration_test.py::TestAmdSmiPythonInterface::test_bad_page_info PASSED              [ 33%]
integration_test.py::TestAmdSmiPythonInterface::test_bdf_device_id PASSED              [ 50%]
integration_test.py::TestAmdSmiPythonInterface::test_ecc PASSED                        [ 66%]
integration_test.py::TestAmdSmiPythonInterface::test_gpu_performance PASSED            [ 83%]
integration_test.py::TestAmdSmiPythonInterface::test_walkthrough PASSED                [100%]

===================================== 6 passed in 0.11s ======================================
```