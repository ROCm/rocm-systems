# API Summary Report
## Overview
The API summary report is generated from reading the amdsmi.h header file and the output from the python and C++ tests.  The python script, api_summary.py, will build a table from the available test log files.

## Pre-Requisites Before Running Summary Report
Run the python and C++ tests prior to running api_summary.py script.  The preferred way to run the tests is as follows:

<u>The python scripts are in the directory /opt/rocm/share/amd_smi/tests/python_unittest</u>
```
sudo unit_tests.py -v > _unit_test.log 2> _unit_test_err.log
sudo integration_test.py -v > _integration_test.log 2> _integration_test_err.log
```

<u>The C++ test is in the directory /opt/rocm/share/amd_smi/tests</u>
```
sudo amdsmitst -v 1 > _amdsmitst.log 2> _amdsmitst_err.log
```

## How to Run Summary Report
### Command Line Options

```
Header File:
  --amdsmi AMDSMI
    Path to header file, default=include/amd_smi/amdsmi.h
Log Files:
  --log_dir LOG_DIR
    Path to where logs exist, default=build
  --c_unit_test C_UNIT_TEST
    Filename for C unit_test output, default=_c_unit_test.log
  --c_integration C_INTEGRATION
    Filename for C integration_test output, default=_c_integration.log
  --py_unit_test PY_UNIT_TEST
    Filename for Python unit_test output, default=_py_unit_test.log
  --py_integration PY_INTEGRATION
    Filename for Python integration_test output, default=_py_integration.log
Output File:
  --output_dir OUTPUT_DIR
    Path to output file, default=.
```

Command line examples:
<details close>
  <summary>Click for example: <i><b>From amdsmi root directory with test ouput in build directory</i></b></summary>

~~~shell
api_summary.py
~~~
<i><b>With specifying summary output</i></b>
~~~shell
api_summary.py --output summary_dir
~~~
</details>

<details close>
  <summary>Click for example: <i><b>From amdsmi root directory with test ouput in current directory</i></b></summary>

~~~shell
api_summary.py --log_dir .
~~~
</details>

<details close>
  <summary>Click for example: <i><b>All input and output in current directory</i></b></summary>

~~~shell
api_summary.py --amdsmi ./amdsmi.h --log_dir . --output_dir .
~~~
</details>

<br> Output Files:
```
  api_summary.csv
  api_summary_table.txt
  api_summary_support.txt
```

<details close>
  <summary>Click for example: <i><b>api_summary.csv</i></b></summary>

~~~shell
API, Tested, c_unit_test, c_integration, py_unit_test, py_integration
amdsmi_init, 2, 0, 0, 1, 1
amdsmi_shut_down, 2, 0, 0, 1, 1
amdsmi_get_socket_handles, 2, 0, 0, 1, 1
amdsmi_get_cpu_handles, 1, 0, 0, 1, 0
amdsmi_get_socket_info, 1, 0, 0, 0, 1
amdsmi_get_processor_info, 1, 0, 0, 1, 0
amdsmi_get_processor_count_from_handles, 1, 0, 0, 1, 0
amdsmi_get_processor_handles_by_type, 1, 0, 0, 0, 1
amdsmi_get_processor_handles, 1, 0, 0, 1, 0
amdsmi_get_node_handle, 0, 0, 0, 0, 0
...
~~~
</details>

<details close>
  <summary>Click for example: <i><b>api_summary_table.txt</i></b></summary>

~~~shell
 API    Test(%)     Unit(%)     Func(%)
  C     64(34.2)    0( 0.0)   64(34.2)
 Py    117(62.6)  101(54.0)   27(14.4)
Total  132(70.6)  101(54.0)   82(43.9)
Num APIs: 187
~~~
</details>

<details close>
  <summary>Click for example: <i><b>api_summary_support.txt</i></b></summary>

~~~shell
API Not Supported: 3
	amdsmi_get_gpu_partition_metrics_info()
	amdsmi_set_gpu_accelerator_partition_profile()
	amdsmi_set_gpu_overdrive_level()
API Supported: 129
	amdsmi_clean_gpu_local_data()
	amdsmi_cpu_apb_disable()
	amdsmi_cpu_apb_enable()
	amdsmi_get_clk_freq()
	amdsmi_get_cpu_cclk_limit()
	amdsmi_get_cpu_core_boostlimit()
  ...
~~~
</details>
