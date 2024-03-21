# Change Log for ROCm SMI Library

Full documentation for rocm_smi_lib is available at [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/).

***All information listed below is for reference and subject to change.***

## rocm_smi_lib for ROCm 6.1.0

### Added
- **Added support to set max/min clock level for sclk (`RSMI_CLK_TYPE_SYS`) or mclk (`RSMI_CLK_TYPE_MEM`)**  
Users can now set a maximum or minimum sclk or mclk value through `rsmi_dev_clk_extremum_set()` API provided ASIC support. Alternatively, users can
use our Python CLI tool (`rocm-smi --setextremum max sclk 1500`). See example below.  
       
```shell
$ sudo /opt/rocm/bin/rocm-smi --setextremum max sclk 2100

============================ ROCm System Management Interface ============================

          ******WARNING******

          Operating your AMD GPU outside of official AMD specifications or outside of
          factory settings, including but not limited to the conducting of overclocking,
          over-volting or under-volting (including use of this interface software,
          even if such software has been directly or indirectly provided by AMD or otherwise
          affiliated in any way with AMD), may cause damage to your AMD GPU, system components
          and/or result in system failure, as well as cause other problems.
          DAMAGES CAUSED BY USE OF YOUR AMD GPU OUTSIDE OF OFFICIAL AMD SPECIFICATIONS OR
          OUTSIDE OF FACTORY SETTINGS ARE NOT COVERED UNDER ANY AMD PRODUCT WARRANTY AND
          MAY NOT BE COVERED BY YOUR BOARD OR SYSTEM MANUFACTURER'S WARRANTY.
          Please use this utility with caution.

Do you accept these terms? [y/N] y
================================ Set Valid sclk Extremum =================================
GPU[0]          : Successfully set max sclk to 2100(MHz)
GPU[1]          : Successfully set max sclk to 2100(MHz)
GPU[2]          : Successfully set max sclk to 2100(MHz)
GPU[3]          : Successfully set max sclk to 2100(MHz)
================================== End of ROCm SMI Log ===================================
```

- **Added `rsmi_dev_target_graphics_version_get()`**  
Users can now query through ROCm SMI API (`rsmi_dev_target_graphics_version_get()`) to retreive the target graphics version for a GPU device. Currently, this output is not supplied through our rocm-smi CLI.

### Changed

- **Removed non-unified API headers: Individual GPU metric APIs are no longer supported**  
The individual metric APIs (`rsmi_dev_metrics_*`) were removed in order to keep updates easier for new GPU metric support. By providing a simple API (`rsmi_dev_gpu_metrics_info_get()`) with its reported device metrics, it is worth noting there is a risk for ABI break-age using `rsmi_dev_gpu_metrics_info_get()`. It is vital to understand, that ABI breaks are necessary (in some cases) in order to support newer ASICs and metrics for our customers. We will continue to support `rsmi_dev_gpu_metrics_info_get()` with these considerations and limitations in mind.

- **Depricated rsmi_dev_power_ave_get(),  use newer API rsmi_dev_power_get()**  
As outlined in change below for 6.0.0 (***Added a generic power API: rsmi_dev_power_get***), is now depricated. Please update your ROCm SMI API calls accordingly.

### Optimizations
- N/A


### Fixed
- Fix `--showpids` reporting `[PID] [PROCESS NAME] 1 UNKNOWN UNKNOWN UNKNOWN`  
Output was failing because cu_occupancy debugfs method is not provided on some graphics cards by design. `get_compute_process_info_by_pid` was updated to reflect this and returns with output needed by CLI.
- Fix `rocm-smi --showpower` output was inconsistent on Navi32/31 devices  
Updated to use `rsmi_dev_power_get()` within CLI to provide a consistent device power output. This was caused due to using the now depricated `rsmi_dev_average_power_get()` API.
- Fixed `rocm-smi --setcomputepartition` and `rocm-smi --resetcomputepartition` to notate if device is EBUSY 
- Fixed `rocm-smi --setmemorypartition` and `rocm-smi --resetmemorypartition` read only SYSFS to return RSMI_STATUS_NOT_SUPPORTED  
The  `rsmi_dev_memory_partition_set` API is updated to handle the readonly SYSFS check. Corresponding tests and CLI (`rocm-smi --setmemorypartition` and `rocm-smi --resetmemorypartition`) calls were updated accordingly.
- Fix `rocm-smi --showclkvolt` and `rocm-smi --showvc` displaying 0 for overdrive and voltage curve is not supported

## rocm_smi_lib for ROCm 6.0.0

### Added

- **Added rocm-smi --version**  
The SMI will report two "versions", ROCM-SMI version and other is ROCM-SMI-LIB version.  
  - The ROCM-SMI version is the CLI/tool version number with commit ID appended after `+` sign.  
  - The ROCM-SMI-LIB version is the library package version number.
```
$ rocm-smi --version
ROCM-SMI version: 2.0.0+8e78352
ROCM-SMI-LIB version: 6.0.0
```

- **Added support for gfx941/gfx942 metrics**  
You can now query MI300 device metrics to get real-time information. Metrics include power, temperature, energy, and performance. Users can query through `rsmi_dev_gpu_metrics_info_get()`.


- **Compute and memory partition support**  
Users can now view, set, and reset partitions. The topology display can provide a more in-depth look at the device's current configuration. If your ASIC supports these features, the following commands can help get started:
  - `rocm-smi --showcomputepartition`
  - `rocm-smi --setcomputepartition <SPX, DPX, CPX, TPX, QPX>`
  - `rocm-smi --resetcomputepartition`
  - `rocm-smi --showmemorypartition`
  - `rocm-smi --setmemorypartition <NPS1, NPS2, NPS4, NPS8>`
  - `rocm-smi --resetmemorypartition`

### Changed

- **GPU index sorting made consistent with other tools**  
To ensure alignment with other ROCm software tools, GPU index sorting is optimized to use Bus:Device.Function (BDF) rather than the card number.

- **Increase max BDF ID length**  
To allow for larger BDF data, we have increased the maximum BDF length from 256 to 512 buffer size.

- **Documentation is transitioning to Sphinx**  
Sphinx allows us to generate code documentation easier for our users. Helps us provide centrized HTML documentation at single website location. Here customers can see how to use our software and tools.

- **Added a generic power API: `rsmi_dev_power_get()`**  
Older ASICs provided average socket power, newer ASICs (MI300) provide current socket power. The generic API provides one interface to retreive either of these power readings, allowing backwards compatability.  

- **Added flexible temperature readings (`rocm-smi` and `rocm-smi --showtempgraph`)**  
Older ASICs provided edge temperature, newer ASICs (MI300) provide junction socket power (not edge). The rocm-smi CLI now provides a way to view which type of temperature is read across all sockets.

- **Added deep sleep frequency readings**  
Newer ASICs (MI300) provide ability to know if a clock is in deep sleep. 


### Optimizations

- Add new test to measure api execution time.
- Remove the shared mutex if no process is using it.
- Updated to C++17, gtest-1.14, and cmake 3.14

### Fixed
- Fix memory usage division by 0
- Fix missing firmware blocks (rocm-smi --showfw)  
- Fix rocm-smi --showevents shows wrong gpuID 


## rocm_smi_lib for ROCm 5.5.0

### Optimizations

- Add new test to measure api execution time.
- Remove the shared mutex if no process is using it.

### Added

- ROCm SMI CLI: Add --showtempgraph Feature.

### Changed

- Relying on vendor ID to detect AMDGPU.
- Change pragma message to warning for backward compatibility.

### Fixed

- Fix --showproductname when device's SKU cannot be parsed out of the VBIOS string.
- Fix compile error: ‘memcpy’ was not declared.
- Fix order of CE and UE reporting in ROCm SMI CLI.
- Handle error return value from ReadSysfsStr function.
