# Change Log for AMD SMI Library

Full documentation for amd_smi_lib is available at [https://docs.amd.com/](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).

## amd_smi_lib for ROCm 6.0.0

### Added

- **Integrated the E-SMI (EPYC-SMI) library**  
You can now query CPU-related information directly through AMD SMI. Metrics include power, energy, performance, and other system details.

- **Added support for gfx942 metrics**  
You can now query MI300 device metrics to get real-time information. Metrics include power, temperature, energy, and performance.

- **Compute and memory partition support**  
Users can now view, set, and reset partitions. The topology display can provide a more in-depth look at the device's current configuration.


### Changed

- **GPU index sorting made consistent with other tools**  
To ensure alignment with other ROCm software tools, GPU index sorting is optimized to use Bus:Device.Function (BDF) rather than the card number.
- **Topology output is now aligned with GPU BDF table**
Earlier versions of the topology output were difficult to read since each GPU was displayed linearly.
Now the information is displayed as a table by each GPU's BDF, which closer resembles rocm-smi output.

### Optimizations

- N/A

### Fixed

- **Fix for driver not initialized**  
If driver module is not loaded, user retrieve error reponse indicating amdgpu module is not loaded.


### Known Issues

- N/A
