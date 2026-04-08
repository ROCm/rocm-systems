# AMD SMI Python library for WSL

The AMD SMI Python interface provides access to GPU telemetry and monitoring
data from Python in the WSL-focused AMD SMI build in this repository.

This Python package is part of the [AMD SMI WSL component](../README.md).

## Build and install

Build and install from the `amdsmi` directory.

Use the build instructions in [../README.md](../README.md#building).

If you install with `make install`, the Python module may need to be registered
manually with Python as described in [../README.md](../README.md).

## Basic usage

```python
import amdsmi

amdsmi.amdsmi_init()
devices = amdsmi.amdsmi_get_processor_handles()
for dev in devices:
    print(amdsmi.amdsmi_get_gpu_asic_info(dev))
amdsmi.amdsmi_shut_down()
```

## Online documentation

For the upstream AMD SMI documentation set, see:

- [AMD SMI documentation portal](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [AMD SMI install guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html)
- [AMD SMI Python library usage](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-py-lib.html)
- [AMD SMI Python API reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html)

Use those pages as the baseline reference for API usage, with the
understanding that some upstream capabilities may not apply to the WSL build in
this repository.
