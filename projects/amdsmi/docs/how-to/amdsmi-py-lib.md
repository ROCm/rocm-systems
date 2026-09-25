---
myst:
  html_meta:
    "description lang=en": "Get started with the AMD SMI Python interface."
    "keywords": "api, smi, lib, py, system, management, interface, ROCm"
---

# AMD SMI Python interface overview

The AMD SMI Python interface provides a convenient way to interact with AMD
hardware through a simple and accessible [API](../reference/amdsmi-py-api.md).

```{seealso}
Refer to the [Python library API reference](../reference/amdsmi-py-api.md).
```

## Prerequisites

Before get started, make sure your environment satisfies the following prerequisites.
See the [requirements](#install_reqs) section for more information.

1. Ensure `amdgpu` drivers are installed properly for initialization. CPU APIs
   require the `amd_hsmp` kernel module. See {ref}`install_amdgpu_driver`.

2. Export `LD_LIBRARY_PATH` to the `amdsmi` installation directory.

   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64:
   ```

3. Install Python 3.6.8+.

## Get started

```{note}
``hipcc`` and other compilers will not automatically link in the ``libamd_smi``
dynamic library. To compile code that uses the AMD SMI library API, ensure the
``libamd_smi.so`` can be located by setting the ``LD_LIBRARY_PATH`` environment
variable to the directory containing ``librocm_smi64.so`` (usually
``/opt/rocm/lib``) or by passing the ``-lamd_smi`` flag to the compiler.
```

```{note}
The following environment variables can be set to control internal cache
durations. They must be set **before** the AMDSMI library loads.

| Variable | Description | Default |
|---|---|---|
| ``AMDSMI_GPU_METRICS_CACHE_MS`` | GPU metrics cache duration (ms) | 1 ms (set to 0 to disable) |
| ``AMDSMI_ASIC_INFO_CACHE_MS`` | GPU ASIC info cache duration (ms) | 10000 ms (set to 0 to disable) |
```

You can set these in one of two ways:

1. **In Python** (before the AMDSMI library loads):

   ```python
   import os
   os.environ["AMDSMI_GPU_METRICS_CACHE_MS"] = "200"
   from amdsmi import *
   ```

2. **From the shell** (when invoking Python):

   ```shell
   AMDSMI_GPU_METRICS_CACHE_MS=200 python tools/amdsmi_quick_start.py
   ```

To get started, the `amdsmi` folder should be copied and placed next to
the importing script. Import it as follows:

```python
from amdsmi import *

try:
    amdsmi_init()

    # amdsmi calls ...

except AmdSmiException as e:
    print(e)
finally:
    try:
        amdsmi_shut_down()
    except AmdSmiException as e:
        print(e)
```

(py_lib_fs)=
### Folder structure

File name             | Description
----------------------|-------------------------------------------------
`__init__.py`         | Python package initialization file
`amdsmi_interface.py` | Amdsmi library Python interface
`amdsmi_wrapper.py`   | Python wrapper around amdsmi binary
`amdsmi_exception.py` | Amdsmi [exceptions](#py_exceptions) Python file

(py_usage)=
## Usage

An application using AMD SMI must call `amdsmi_init()` to initialize the AMI SMI
library before all other calls. This call initializes the internal data
structures required for subsequent AMD SMI operations. In the call, a flag can
be passed to indicate if the application is interested in a specific device
type.

`amdsmi_shut_down()` must be the last call to properly close connection to
driver and make sure that any resources held by AMD SMI are released.

```{seealso}
Refer to the [Python library API reference](../reference/amdsmi-py-api.md).
```

(py_version_match)=
### Use the module that ships with the library

The Python module and `libamd_smi.so` are released together. Import the module
from the same installation that provides the library, not a copy from another
release. A module from an older release can fail as soon as it is imported
against a newer library. For example, library major version 27 (ROCm 10.0 and
later) removed `amdsmi_set_gpu_clk_range()` and `amdsmi_gpu_driver_reload()`,
so the `amdsmi` 7.0.2 wheel on PyPI fails at import against it:

```text
AttributeError: .../libamd_smi.so.27: undefined symbol: amdsmi_set_gpu_clk_range
```

PyTorch imports `amdsmi` during `import torch` on ROCm, so this error can
surface there. See [Packaging and install paths](../packaging.md) for where
each installation method places the module and the library.

(py_exceptions)=
## Exceptions

All exceptions are in `amdsmi_exception.py` file.

Exceptions that can be thrown by AMD SMI are:

* `AmdSmiException`: base amdsmi exception class
* `AmdSmiLibraryException`: derives base `AmdSmiException` class and represents errors that can occur in amdsmi-lib.
  When this exception is thrown, `err_code` and `err_info` are set. `err_code` is an integer that corresponds to errors that can occur
  in amdsmi-lib and `err_info` is a string that explains the error that occurred.

   ```python
   try:
       num_of_GPUs = len(amdsmi_get_processor_handles())
   except amdsmi_exception.AmdSmiLibraryException as e:
       print("Unable to get processor handles, error: {} {}".format(str(e.get_error_code()), e.err_info))
   ```


* `OSError`: raised by the first call into the library when `libamd_smi.so`
  could not be loaded. `import amdsmi` still succeeds in that case, so code
  that checks whether AMD SMI is usable should catch `OSError` as well as
  `AmdSmiException`:

   ```python
   try:
       amdsmi_init()
   except (AmdSmiException, OSError) as e:
       print("AMD SMI is not usable: {}".format(e))
   ```

* `AmdSmiParameterException`: Derives base `AmdSmiException` class and
  represents errors related to invalid parameters passed to functions. When this
  exception is thrown, `err_msg` is set and it explains what is the actual and
  expected type of the parameters.

   For example:

   ```python
   try:
       cpu_handles = amdsmi_get_cpu_handles()
       cpu_count = cpu_handles["cpu_count"]
       processor_handles = cpu_handles["processor_handles"]
       if cpu_count == 0:
           print("No CPU sockets on machine")
       else:
           for processor in processor_handles:
               temperature = amdsmi_get_cpu_socket_temperature(processor)
               print(temperature)
   except amdsmi_exception.AmdSmiParameterException as e:
       print("Invalid parameter error: {} {}".format(str(e.get_error_code()), e.err_msg))
   except amdsmi_exception.AmdSmiLibraryException as e:
       print("Unable to get processor handles, error: {} {}".format(str(e.get_error_code()), e.err_info))
   ```


