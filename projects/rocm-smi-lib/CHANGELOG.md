# Changelog for ROCm SMI Library

Full documentation for rocm_smi_lib is available at [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/).

***All information listed below is for reference and subject to change.***

## rocm_smi_lib for ROCm 6.5.0

### Added

- **Added support for GPU metrics 1.8**.  
  - Added new fields for `rsmi_gpu_metrics_t` including:  
    - Adding the following metrics to allow new calculations for violation status:
    - Per XCP metrics `gfx_below_host_limit_ppt_acc[XCP][MAX_XCC]` - GFX Clock Host limit Package Power Tracking violation counts
    - Per XCP metrics `gfx_below_host_limit_thm_acc[XCP][MAX_XCC]` - GFX Clock Host limit Thermal (TVIOL) violation counts
    - Per XCP metrics `gfx_low_utilization_acc[XCP][MAX_XCC]` - violation counts for how did low utilization caused the GPU to be below application clocks.
    - Per XCP metrics `gfx_below_host_limit_total_acc[XCP][MAX_XCC]`- violation counts for how long GPU was held below application clocks any limiter (see above new violation metrics).
  - Increasing available JPEG engines to 40.  
  Current ASICs may not support all 40. These will be indicated as UINT16_MAX or N/A in CLI.

### Changed

- N/A

### Removed

- **Removed backwards compatibility `rsmi_dev_gpu_metrics_info_get()`'s `jpeg_activity` or `vcn_activity` fields: use `xcp_stats.jpeg_busy` or `xcp_stats.vcn_busy`**  
  - Backwards compability is removed for `jpeg_activity` and `vcn_activity` fields, if the `jpeg_busy` or `vcn_busy` field is available.
    - <i>Reasons for this change</i>:
      - Providing both `vcn_activity`/`jpeg_activity` and XCP (partition) stats `vcn_busy`/`jpeg_busy` caused confusion for users about which field to use. By removing backward compatibility, it is easier to identify the relevant field.
      - The `jpeg_busy` field increased in size (for supported ASICs), making backward compatibility unable to fully copy the structure into `jpeg_activity`.

    See below for comparison of updated CLI outputs:

    Original output:
    ```shell
    $ rocm-smi --showmetrics
    GPU[0]          : vcn_activity (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0]          : jpeg_activity (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[0]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[1]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[2]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[3]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[4]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[5]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[6]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[7]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[0]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[1]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[2]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[3]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[4]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[5]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[6]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[7]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    ```
    New output:
    ```shell
    $ rocm-smi --showmetrics
    GPU[0]          : vcn_activity (%): ['N/A', 'N/A', 'N/A', 'N/A']
    GPU[0]          : jpeg_activity (%): ['N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[0]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[1]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[2]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[3]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[4]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[5]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[6]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[7]   : xcp_stats.jpeg_busy (%): [0, 0, 0, 0, 0, 0, 0, 0, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[0]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[1]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[2]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[3]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[4]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[5]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[6]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    GPU[0] XCP[7]   : xcp_stats.vcn_busy (%): [0, 'N/A', 'N/A', 'N/A']
    ```

### Optimized

- N/A

### Resolved issues

- N/A

### Upcoming changes

- N/A

### Known issues

- N/A

## rocm_smi_lib for ROCm 6.4.1

### Added

- N/A

### Changed

- N/A

### Removed

- N/A

### Optimized

- N/A

### Resolved issues

- **Fixed partition enumeration - now refer to correct DRM Render and Card paths**
    Previously, partitions incorrectly reflected the primary node (1st GPU) and showed the DRM Render Minor as renderD128. Partition nodes mirrored renderD128's information, which was incorrect. See the "<i>Previous Outputs in CPX</i>" example below.

    Device enumeration was updated to correctly map DRM Render Minor paths. See the "<i>Corrected Outputs in CPX</i>" example below.

    These changes impact what information is readable/writable for the partition nodes.

    <b><i>Example: Previous Outputs in CPX</b></i>  
    ```shell
    $ rocm-smi

    ============================================ ROCm System Management Interface ============================================
    ====================================================== Concise Info ======================================================
    Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
                  (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
    ==========================================================================================================================
    0       2     0x74a1,   18421  45.0°C      152.0W    NPS1, CPX, 0        133Mhz  900Mhz  0%   auto  750.0W  0%     0%
    1       3     0x74a1,   48116  45.0°C      152.0W    NPS1, CPX, 1        133Mhz  900Mhz  0%   auto  750.0W  0%     0%
    2       4     0x74a1,   65524  45.0°C      152.0W    NPS1, CPX, 2        138Mhz  900Mhz  0%   auto  750.0W  0%     0%
    3       5     0x74a1,   1013   45.0°C      152.0W    NPS1, CPX, 3        138Mhz  900Mhz  0%   auto  750.0W  0%     0%
    4       6     0x74a1,   30708  45.0°C      152.0W    NPS1, CPX, 4        138Mhz  900Mhz  0%   auto  750.0W  0%     0%
    5       7     0x74a1,   35829  45.0°C      152.0W    NPS1, CPX, 5        153Mhz  900Mhz  0%   auto  750.0W  0%     0%
    6       8     0x74a1,   53237  45.0°C      152.0W    NPS1, CPX, 6        153Mhz  900Mhz  0%   auto  750.0W  0%     0%
    7       9     0x74a1,   13300  45.0°C      152.0W    NPS1, CPX, 7        153Mhz  900Mhz  0%   auto  750.0W  0%     0%
    8       10    0x74a1,   64360  44.0°C      158.0W    NPS1, CPX, 0        144Mhz  900Mhz  0%   auto  750.0W  0%     0%
    ...
    ==========================================================================================================================
    ================================================== End of ROCm SMI Log ===================================================
    ```  
    <b><i>Example: Corrected outputs in CPX</i></b>
    ```shell
    $ rocm-smi

    ============================================ ROCm System Management Interface ============================================
    ====================================================== Concise Info ======================================================
    Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%
                  (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)
    ==========================================================================================================================
    0       2     0x74a1,   18421  44.0°C      151.0W    NPS1, CPX, 0        132Mhz  900Mhz  0%   auto  750.0W  0%     0%
    1       3     N/A,      48116  N/A         N/A       N/A, N/A, 1         N/A     N/A     0%   n/a   N/A     2%     N/A
    2       4     N/A,      65524  N/A         N/A       N/A, N/A, 2         N/A     N/A     0%   n/a   N/A     2%     N/A
    3       5     N/A,      1013   N/A         N/A       N/A, N/A, 3         N/A     N/A     0%   n/a   N/A     2%     N/A
    4       6     N/A,      30708  N/A         N/A       N/A, N/A, 4         N/A     N/A     0%   n/a   N/A     2%     N/A
    5       7     N/A,      35829  N/A         N/A       N/A, N/A, 5         N/A     N/A     0%   n/a   N/A     2%     N/A
    6       8     N/A,      53237  N/A         N/A       N/A, N/A, 6         N/A     N/A     0%   n/a   N/A     2%     N/A
    7       9     N/A,      13300  N/A         N/A       N/A, N/A, 7         N/A     N/A     0%   n/a   N/A     2%     N/A
    8       10    0x74a1,   64360  44.0°C      158.0W    NPS1, CPX, 0        132Mhz  900Mhz  0%   auto  750.0W  0%     0%
    ...
    ==========================================================================================================================
    ================================================== End of ROCm SMI Log ===================================================
    ```

### Upcoming changes

- N/A

### Known issues

- N/A

## rocm_smi_lib for ROCm 6.4

### Added

- **Added support for GPU metrics 1.7 to `rsmi_dev_gpu_metrics_info_get()`**  
Updated `rsmi_dev_gpu_metrics_info_get()` and structure `rsmi_gpu_metrics_t` to include new fields for XGMI Link Status, graphics clocks below host limit (per XCP), and VRAM max bandwidth:  
  - `uint64_t vram_max_bandwidth` - VRAM max bandwidth at max memory clock (GB/s)
  - `uint16_t xgmi_link_status[MAX_NUM_XGMI_LINKS]` - XGMI link statis, 1=Up 0=Down
  - `uint64_t gfx_below_host_limit_acc[MAX_NUM_XCC]` - graphics clocks below host limit (per XCP) accumulators. Used for graphic clk below host limit violation status.

- **Added new GPU metrics 1.7 to `rocm-smi --showmetrics`**  
New metrics added to `rocm-smi --showmetrics`
```shell
$ rocm-smi --showmetrics
  GPU[0]          : vram_max_bandwidth (GB/s): 1555
  GPU[0]          : xgmi_link_status (Up/Down): ['1', '1', '1', '1', '0', '1', '0', '1']
  GPU[0] XCP[0]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[0] XCP[1]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[0] XCP[2]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[0] XCP[3]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[0] XCP[4]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[0] XCP[5]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[0] XCP[6]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[0] XCP[7]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[1]          : vram_max_bandwidth (GB/s): 1555
  GPU[1]          : xgmi_link_status (Up/Down): ['1', '1', '1', '1', '0', '1', '0', '1']
  ...
  GPU[1] XCP[0]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[1] XCP[1]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[1] XCP[2]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[1] XCP[3]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[1] XCP[4]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[1] XCP[5]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[1] XCP[6]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  GPU[1] XCP[7]   : xcp_stats.gfx_below_host_limit_acc (%): ['0', '0', '0', '0', '0', '0', '0', '0']
  ...
```

### Changed

### Removed

### Resolved issues

- **Fixed `rsmi_dev_target_graphics_version_get`, `rocm-smi --showhw`, and `rocm-smi --showprod` not displaying graphics version properly for MI2x, MI1x or Navi 3x ASICs.**  

### Upcoming changes

## rocm_smi_lib for ROCm 6.3

- **Added `rsmi_dev_memory_partition_capabilities_get` which returns driver memory partition capablities.**  
Driver now has the ability to report what the user can set memory partition modes to. User can now see available
memory partition modes upon an invalid argument return from memory partition mode set (`rsmi_dev_memory_partition_set`).


- **Added support for GPU metrics 1.6 to `rsmi_dev_gpu_metrics_info_get()`**  
Updated `rsmi_dev_gpu_metrics_info_get()` and structure `rsmi_gpu_metrics_t` to include new fields for PVIOL / TVIOL,  XCP (Graphics Compute Partitions) stats, and pcie_lc_perf_other_end_recovery:  
  - `uint64_t accumulation_counter` - used for all throttled calculations
  - `uint64_t prochot_residency_acc` - Processor hot accumulator
  - `uint64_t ppt_residency_acc` - Package Power Tracking (PPT) accumulator (used in PVIOL calculations)
  - `uint64_t socket_thm_residency_acc` - Socket thermal accumulator - (used in TVIOL calculations)
  - `uint64_t vr_thm_residency_acc` - Voltage Rail (VR) thermal accumulator
  - `uint64_t hbm_thm_residency_acc` - High Bandwidth Memory (HBM) thermal accumulator 
  - `uint16_t num_partition` - corresponds to the current total number of partitions
  - `struct amdgpu_xcp_metrics_t xcp_stats[MAX_NUM_XCP]` - for each partition associated with current GPU, provides gfx busy & accumulators, jpeg, and decoder (VCN) engine utilizations
    - `uint32_t gfx_busy_inst[MAX_NUM_XCC]` - graphic engine utilization (%)
    - `uint16_t jpeg_busy[MAX_NUM_JPEG_ENGS]` - jpeg engine utilization (%)
    - `uint16_t vcn_busy[MAX_NUM_VCNS]` - decoder (VCN) engine utilization (%)
    - `uint64_t gfx_busy_acc[MAX_NUM_XCC]` - graphic engine utilization accumulated (%)
  - `uint32_t pcie_lc_perf_other_end_recovery` - corresponds to the pcie other end recovery counter

- **Added ability to view raw GPU metrics`rocm-smi --showmetrics`**  
Users can now view GPU metrics from our new `rocm-smi --showmetrics`. Unlike AMD SMI (or other ROCM-SMI interfaces), these values are ***not*** converted into applicable units as users may see in `amd-smi metric`. Units listed display as indicated by the driver, they are not converted (eg. in other AMD SMI/ROCm SMI interfaces which use the data provided). It is important to note, that fields displaying `N/A` data mean this ASIC does not support or backward compatibility was not provided in a newer ASIC's GPU metric structure.   

### Changed

- **Added back in C++ tests for `memorypartition_read_write`**.  
Due to driver adding in all needed features for memory partition write. We have re-enabled memorypartition_read_write.

- **Updated `rsmi_dev_memory_partition_set` to not return until a successful restart of AMD GPU Driver.**  
This change keeps checking for ~ up to 40 seconds for a successful restart of the AMD GPU driver. Additionally, the API call continues to check if memory partition (NPS) SYSFS files are successfully updated to reflect the user's requested memory partition (NPS) mode change. Otherwise, reports an error back to the user. Due to these changes, we have updated ROCm SMI's CLI to reflect the maximum wait of 40 seconds, while memory partition change is in progress.

- **All APIs now have the ability to catch driver reporting invalid arguments.**  
Now ROCm SMI APIs can show RSMI_STATUS_INVALID_ARGS when driver returns EINVAL.

### Removed

- **Removed `--resetcomputepartition`, and  `--resetmemorypartition` options and associated APIs**.
  - This change is part of the partition feature redesign.
  - The related APIs `rsmi_dev_compute_partition_reset()` and `rsmi_dev_memory_partition_reset()`.

### Resolved issues

- **Fixed `rsmi_dev_target_graphics_version_get`, `rocm-smi --showhw`, and `rocm-smi --showprod` not displaying properly for MI2x or Navi 3x ASICs.**  

### Upcoming changes

- **Re-enable C++ tests for `memorypartition_read_write`**.  
  - This change is part of the partition feature redesign.
  - SMI's workflow needs to be adjusted in order to accomidate incoming driver changes to enable
  Dynamic memory partition feature. We plan on re-enabling testing for this feature during ROCm
  6.4.

## rocm_smi_lib for ROCm 6.2.1

### Optimized

- **Improved handling of UnicodeEncodeErrors with non UTF-8 locales**  
Non UTF-8 locales were causing crashing on UTF-8 special characters

### Resolved issues

- **Fixed rsmitstReadWrite.TestComputePartitionReadWrite segfault**  
Segfault was caused due to unhandled start conditions:

1) When setting CPX as a partition mode, there is a DRM node limitation of 64.
This is a known limitation of the driver, if other drivers are using other DRM nodes (ex. using PCIe space, such as ast).  
The number of DRM nodes can be checked via `ls /sys/class/drm`  
Recommended steps for removing unnecessary drivers:  
a. unloading amdgpu - `sudo rmmod amdgpu`  
b. removing unnecessary driver(s) - ex. `sudo rmmod ast`  
c. reload amgpu - `sudo modprobe amdgpu`

2) Since user could start amdgpu in different partition modes (ex. `sudo modprobe amdgpu user_partt_mode=1`).
Test needed to keep track of total number of devices, in order to ensure test comes back to the original configuration.
The test segfault could be seen on all MI3x ASICs, if brought up in a non-SPX configuration upon boot.

## rocm_smi_lib for ROCm 6.2

### Changed

- **Added Partition ID API (`rsmi_dev_partition_id_get(..)`)**  
Previously `rsmi_dev_partition_id_get` could only be retrived by querying through `rsmi_dev_pci_id_get()`
and parsing optional bits in our python CLI/API. We are now making this available directly through API.
As well as added testing, in our compute partitioning tests verifing partition IDs update accordingly. 

### Resolved issues

- **Partition ID CLI output**  
Due to driver changes in KFD, some devices may report bits [31:28] or [2:0]. With the newly added `rsmi_dev_partition_id_get(..)`, we provided this fallback to properly retreive partition ID. We
plan to eventually remove partition ID from the function portion of the BDF (Bus Device Function). See below for PCI ID description.

  - bits [63:32] = domain
  - bits [31:28] or bits [2:0] = partition id 
  - bits [27:16] = reserved
  - bits [15:8]  = Bus
  - bits [7:3] = Device
  - bits [2:0] = Function (partition id maybe in bits [2:0]) <-- Fallback for non SPX modes

## rocm_smi_lib for ROCm 6.1.2

### Added

- **Added Ring Hang event**  
Added `RSMI_EVT_NOTIF_RING_HANG` to the possible events in the `rsmi_evt_notification_type_t` enum.

### Resolved issues

- **Fixed parsing of `pp_od_clk_voltage` within `get_od_clk_volt_info`**  
The parsing of `pp_od_clk_voltage` was not dynamic enough to work with the dropping of voltage curve support on MI series cards.

## rocm_smi_lib for ROCm 6.1.1

### Added

- **Unlock mutex if process is dead**
Added in order to unlock mutex when process is dead. Additional debug output has been added if futher issues are detected.

- **Added Partition ID to rocm-smi CLI**
`rsmi_dev_pci_id_get()` now provides partition ID. See API for better detail. Previously these bits were reserved bits (right before domain) and partition id was within function.
  - bits [63:32] = domain
  - bits [31:28] = partition id
  - bits [27:16] = reserved
  - bits [15: 0] = pci bus/device/function

rocm-smi now provides partition ID in `rocm-smi` and `rocm-smi --showhw`. If device supports partitioning and is in a non-SPX mode (CPX, DPX,TPX,... etc) partition ID will be non-zero. In SPX and non-supported devices will show as 0. See examples provided below.

```shell
 $ rocm-smi

========================================= ROCm System Management Interface =========================================
=================================================== Concise Info ===================================================
Device  Node  IDs              Temp    Power  Partitions          SCLK  MCLK   Fan    Perf    PwrCap  VRAM%  GPU%
              (DID,     GUID)  (Edge)  (Avg)  (Mem, Compute, ID)
====================================================================================================================
0       1     0x73bf,   34495  43.0°C  6.0W   N/A, N/A, 0         0Mhz  96Mhz  0%     manual  150.0W  3%     0%
1       2     0x73a3,   22215  34.0°C  8.0W   N/A, N/A, 0         0Mhz  96Mhz  20.0%  manual  213.0W  0%     0%
====================================================================================================================
=============================================== End of ROCm SMI Log ================================================
```
*Device below is in TPX*
```shell
$ rocm-smi --showhw

================================= ROCm System Management Interface =================================
====================================== Concise Hardware Info =======================================
GPU  NODE  DID     GUID   GFX VER  GFX RAS  SDMA RAS  UMC RAS   VBIOS  BUS           PARTITION ID
0    4     0x74a0  3877   gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  0
1    5     0x74a0  54196  gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  1
2    6     0x74a0  36891  gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  2
3    7     0x74a0  28397  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  0
4    8     0x74a0  45692  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  1
5    9     0x74a0  61907  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  2
6    10    0x74a0  52404  gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  0
7    11    0x74a0  4133   gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  1
8    12    0x74a0  21386  gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  2
9    13    0x74a0  10876  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  0
10   14    0x74a0  63213  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  1
11   15    0x74a0  46402  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  2
====================================================================================================
======================================= End of ROCm SMI Log ========================================
```

- **Added `NODE`, `GUID`, and `GFX Version`**
Changes impact the following rocm-smi CLIs:
  - `rocm-smi`
  - `rocm-smi -i`
  - `rocm-smi --showhw`
  - `rocm-smi --showproduct`

  `NODE` - is the KFD node, since these can both be CPU and GPU devices. This field is invariant between boots.
  `GUID` - also known as GPU ID. GUID is the KFD GPU's ID. This field has a chance to be variant between boots.
  `GFX Version` - this is the device's target graphics version.

See below for a few example outputs.
```shell
$ rocm-smi --showhw

================================= ROCm System Management Interface =================================
====================================== Concise Hardware Info =======================================
GPU  NODE  DID     GUID   GFX VER  GFX RAS  SDMA RAS  UMC RAS   VBIOS  BUS           PARTITION ID
0    4     0x74a0  3877   gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  0
1    5     0x74a0  54196  gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  1
2    6     0x74a0  36891  gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  2
3    7     0x74a0  28397  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  0
4    8     0x74a0  45692  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  1
5    9     0x74a0  61907  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  2
6    10    0x74a0  52404  gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  0
7    11    0x74a0  4133   gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  1
8    12    0x74a0  21386  gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  2
9    13    0x74a0  10876  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  0
10   14    0x74a0  63213  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  1
11   15    0x74a0  46402  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  2
====================================================================================================
======================================= End of ROCm SMI Log ========================================
```
```shell
$ rocm-smi -i

============================ ROCm System Management Interface ============================
=========================================== ID ===========================================
GPU[0]          : Device Name:          Aqua Vanjaram [Instinct MI300A]
GPU[0]          : Device ID:            0x74a0
GPU[0]          : Device Rev:           0x00
GPU[0]          : Subsystem ID:         0x74a0
GPU[0]          : GUID:                 60294
GPU[1]          : Device Name:          Aqua Vanjaram [Instinct MI300A]
GPU[1]          : Device ID:            0x74a0
GPU[1]          : Device Rev:           0x00
GPU[1]          : Subsystem ID:         0x74a0
GPU[1]          : GUID:                 35406
GPU[2]          : Device Name:          Aqua Vanjaram [Instinct MI300A]
GPU[2]          : Device ID:            0x74a0
GPU[2]          : Device Rev:           0x00
GPU[2]          : Subsystem ID:         0x74a0
GPU[2]          : GUID:                 10263
GPU[3]          : Device Name:          Aqua Vanjaram [Instinct MI300A]
GPU[3]          : Device ID:            0x74a0
GPU[3]          : Device Rev:           0x00
GPU[3]          : Subsystem ID:         0x74a0
GPU[3]          : GUID:                 52959
==========================================================================================
================================== End of ROCm SMI Log ===================================
```
```shell
$ rocm-smi --showproduct

============================ ROCm System Management Interface ============================
====================================== Product Info ======================================
GPU[0]          : Card Series:          Aqua Vanjaram [Instinct MI300A]
GPU[0]          : Card Model:           0x74a0
GPU[0]          : Card Vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[0]          : Card SKU:             N/A
GPU[0]          : Subsystem ID:         0x74a0
GPU[0]          : Device Rev:           0x00
GPU[0]          : Node ID:              4
GPU[0]          : GUID:                 60294
GPU[0]          : GFX Version:          gfx942
GPU[1]          : Card Series:          Aqua Vanjaram [Instinct MI300A]
GPU[1]          : Card Model:           0x74a0
GPU[1]          : Card Vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[1]          : Card SKU:             N/A
GPU[1]          : Subsystem ID:         0x74a0
GPU[1]          : Device Rev:           0x00
GPU[1]          : Node ID:              5
GPU[1]          : GUID:                 35406
GPU[1]          : GFX Version:          gfx942
GPU[2]          : Card Series:          Aqua Vanjaram [Instinct MI300A]
GPU[2]          : Card Model:           0x74a0
GPU[2]          : Card Vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[2]          : Card SKU:             N/A
GPU[2]          : Subsystem ID:         0x74a0
GPU[2]          : Device Rev:           0x00
GPU[2]          : Node ID:              6
GPU[2]          : GUID:                 10263
GPU[2]          : GFX Version:          gfx942
GPU[3]          : Card Series:          Aqua Vanjaram [Instinct MI300A]
GPU[3]          : Card Model:           0x74a0
GPU[3]          : Card Vendor:          Advanced Micro Devices, Inc. [AMD/ATI]
GPU[3]          : Card SKU:             N/A
GPU[3]          : Subsystem ID:         0x74a0
GPU[3]          : Device Rev:           0x00
GPU[3]          : Node ID:              7
GPU[3]          : GUID:                 52959
GPU[3]          : GFX Version:          gfx942
==========================================================================================
================================== End of ROCm SMI Log ===================================
```

- **Documentation now includes C++ and Python: tutorials, API guides, and C++ reference pages**
See [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/) once 6.1.1 is released.

### Changed

- **Aligned `rocm-smi` fields display "N/A" instead of "unknown"/"unsupported": `Card ID`, `DID`, `Model`, `SKU`, and `VBIOS`**
Impacts the following commands:
  - `rocm-smi` - see other examples above for 6.1.1
  - `rocm-smi --showhw` - see other examples above for 6.1.1
  - `rocm-smi --showproduct` - see other examples above for 6.1.1
  - `rocm-smi -i` - see other examples above for 6.1.1
  - `rocm-smi --showvbios` - see example below
```shell
$ rocm-smi --showvbios

============================ ROCm System Management Interface ============================
========================================= VBIOS ==========================================
GPU[0]          : VBIOS version: N/A
GPU[1]          : VBIOS version: N/A
GPU[2]          : VBIOS version: N/A
GPU[3]          : VBIOS version: N/A
==========================================================================================
================================== End of ROCm SMI Log ===================================
```
- **Removed stacked id formatting in `rocm-smi`**
 This is to simplify identifiers helpful to users. More identifiers can be found on:
   - `rocm-smi -i`
   - `rocm-smi --showhw`
   - `rocm-smi --showproduct`

  See examples shown above for 6.1.1. Previous output example can be seen below.
 ```shell
 $ rocm-smi

========================================== ROCm System Management Interface ==========================================
==================================================== Concise Info ====================================================
Device  [Model : Revision]    Temp        Power     Partitions      SCLK   MCLK     Fan  Perf    PwrCap  VRAM%  GPU%
        Name (20 chars)       (Junction)  (Socket)  (Mem, Compute)
======================================================================================================================
0       [0x74a0 : 0x00]       40.0°C      102.0W    NPS1, SPX       31Mhz  1300Mhz  0%   manual  550.0W    0%   0%
        Aqua Vanjaram [Insti
======================================================================================================================
================================================ End of ROCm SMI Log =================================================
 ```

### Resolved issues

- **Fixed HIP and ROCm SMI mismatch on GPU bus assignments**
These changes prompted us to to provide better visability for our device nodes and partition IDs (see addition provided above). See examples below for fix overview.
1. MI300a GPU device `Domain:Bus:Device.function` clashes with another AMD USB device
Cause(s):
a. ROCm SMI did not propagate domain consistently (for partitioned devices)
b. AMD GPU driver previously reported partition IDs within function node - causing clash with the other AMD USB device PCIe ID displayed.
2. Domain does not propagate for devices which support partitioning (MI300x/a)
Cause(s):
a. ROCm SMI did not propagate domain consistently (for partitioned devices)
3. Displayed topology will show disordered nodes when compared to HIP
Cause(s):
a. ROCm SMI did not propogate domain consistently (for partitioned devices)

*Device in TPX*
```shell
$ rocm-smi --showhw

================================= ROCm System Management Interface =================================
====================================== Concise Hardware Info =======================================
GPU  NODE  DID     GUID   GFX VER  GFX RAS  SDMA RAS  UMC RAS   VBIOS  BUS           PARTITION ID
0    4     0x74a0  3877   gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  0
1    5     0x74a0  54196  gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  1
2    6     0x74a0  36891  gfx942   ENABLED  ENABLED   DISABLED  N/A    0000:01:00.0  2
3    7     0x74a0  28397  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  0
4    8     0x74a0  45692  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  1
5    9     0x74a0  61907  gfx942   ENABLED  ENABLED   DISABLED  N/A    0001:01:00.0  2
6    10    0x74a0  52404  gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  0
7    11    0x74a0  4133   gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  1
8    12    0x74a0  21386  gfx942   ENABLED  ENABLED   DISABLED  N/A    0002:01:00.0  2
9    13    0x74a0  10876  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  0
10   14    0x74a0  63213  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  1
11   15    0x74a0  46402  gfx942   ENABLED  ENABLED   DISABLED  N/A    0003:01:00.0  2
====================================================================================================
======================================= End of ROCm SMI Log ========================================

$ lspci -D|grep -i "process\|usb"
0000:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300A]
0000:01:00.1 USB controller: Advanced Micro Devices, Inc. [AMD] Device 14df
0001:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300A]
0002:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300A]
0003:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Aqua Vanjaram [Instinct MI300A]
```
```shell
$ rocm-smi ----showtoponuma

======================================= Numa Nodes =======================================
GPU[0]          : (Topology) Numa Node: 0
GPU[0]          : (Topology) Numa Affinity: 0
GPU[1]          : (Topology) Numa Node: 0
GPU[1]          : (Topology) Numa Affinity: 0
GPU[2]          : (Topology) Numa Node: 0
GPU[2]          : (Topology) Numa Affinity: 0
GPU[3]          : (Topology) Numa Node: 1
GPU[3]          : (Topology) Numa Affinity: 1
GPU[4]          : (Topology) Numa Node: 1
GPU[4]          : (Topology) Numa Affinity: 1
GPU[5]          : (Topology) Numa Node: 1
GPU[5]          : (Topology) Numa Affinity: 1
GPU[6]          : (Topology) Numa Node: 2
GPU[6]          : (Topology) Numa Affinity: 2
GPU[7]          : (Topology) Numa Node: 2
GPU[7]          : (Topology) Numa Affinity: 2
GPU[8]          : (Topology) Numa Node: 2
GPU[8]          : (Topology) Numa Affinity: 2
GPU[9]          : (Topology) Numa Node: 3
GPU[9]          : (Topology) Numa Affinity: 3
GPU[10]         : (Topology) Numa Node: 3
GPU[10]         : (Topology) Numa Affinity: 3
GPU[11]         : (Topology) Numa Node: 3
GPU[11]         : (Topology) Numa Affinity: 3
================================== End of ROCm SMI Log ===================================
```
- **Fixed memory leaks**
Caused by not closing directories and creating maps nodes instead of checking using by using .at().
- **Fixed Python rocm_smi API calls**
Fixed initializing calls which reuse rocmsmi.initializeRsmi() bindings.

```shell
Traceback (most recent call last):
  File "/home/charpoag/rocmsmi_pythonapi.py", line 9, in <module>
    rocm_smi.initializeRsmi()
  File "/opt/rocm/libexec/rocm_smi/rocm_smi.py", line 3531, in initializeRsmi
    ret_init = rocmsmi.rsmi_init(0)
NameError: name 'rocmsmi' is not defined
```
- **Fixed rsmi_dev_activity_metric_get gfx/memory activity does not update with GPU activity**
    Checks and forces rereading gpu metrics unconditionally.

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

### Resolved issues

- Fix `--showpids` reporting `[PID] [PROCESS NAME] 1 UNKNOWN UNKNOWN UNKNOWN`
Output was failing because cu_occupancy debugfs method is not provided on some graphics cards by design. `get_compute_process_info_by_pid` was updated to reflect this and returns with output needed by CLI.
- Fix `rocm-smi --showpower` output was inconsistent on Navi32/31 devices
Updated to use `rsmi_dev_power_get()` within CLI to provide a consistent device power output. This was caused due to using the now depricated `rsmi_dev_average_power_get()` API.
- Fixed `rocm-smi --setcomputepartition` and `rocm-smi --resetcomputepartition` to notate if device is EBUSY
- Fixed `rocm-smi --setmemorypartition` and `rocm-smi --resetmemorypartition` read only SYSFS to return RSMI_STATUS_NOT_SUPPORTED
The  `rsmi_dev_memory_partition_set` API is updated to handle the readonly SYSFS check. Corresponding tests and CLI (`rocm-smi --setmemorypartition` and `rocm-smi --resetmemorypartition`) calls were updated accordingly.
- Fix `rocm-smi --showclkvolt` and `rocm-smi --showvc` displaying 0 for overdrive and voltage curve is not supported

### Known issues

- **HIP and ROCm SMI mismatch on GPU bus assignments**
Three separate issues have been identified:
1. MI300a GPU device `Domain:Bus:Device.function` clashes with another AMD USB device
```shell
$ lspci|grep -i "process\|usb"
0000:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Device 74a0
0000:01:00.1 USB controller: Advanced Micro Devices, Inc. [AMD] Device 14df
0001:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Device 74a0
0002:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Device 74a0
0003:01:00.0 Processing accelerators: Advanced Micro Devices, Inc. [AMD/ATI] Device 74a0
```
```shell
$ rocm-smi --showbus

============================ ROCm System Management Interface ============================
======================================= PCI Bus ID =======================================
GPU[0]          : PCI Bus: 0000:01:00.0
GPU[1]          : PCI Bus: 0000:01:00.1
GPU[2]          : PCI Bus: 0000:01:00.2
GPU[3]          : PCI Bus: 0000:01:00.3
...
==========================================================================================
================================== End of ROCm SMI Log ===================================
```
2. Domain does not propagate for devices which support partitioning (MI300x/a)
For example, a device in non-SPX (single partition) - devices will overlap in function device.
```shell
$ rocm-smi --showbus
 ============================ ROCm System Management Interface ============================
======================================= PCI Bus ID =======================================
GPU[0]          : PCI Bus: 0000:01:00.0
GPU[1]          : PCI Bus: 0000:01:00.1
GPU[2]          : PCI Bus: 0000:01:00.1
GPU[3]          : PCI Bus: 0000:01:00.1
GPU[4]          : PCI Bus: 0000:01:00.1
GPU[5]          : PCI Bus: 0000:01:00.2
GPU[6]          : PCI Bus: 0000:01:00.2
GPU[7]          : PCI Bus: 0000:01:00.2
GPU[8]          : PCI Bus: 0000:01:00.2
GPU[9]          : PCI Bus: 0000:01:00.3
GPU[10]         : PCI Bus: 0000:01:00.3
GPU[11]         : PCI Bus: 0000:01:00.3
GPU[12]         : PCI Bus: 0000:01:00.3
GPU[13]         : PCI Bus: 0000:01:00.4
GPU[14]         : PCI Bus: 0000:01:00.4
GPU[15]         : PCI Bus: 0000:01:00.4
GPU[16]         : PCI Bus: 0000:01:00.4
GPU[17]         : PCI Bus: 0000:01:00.5
GPU[18]         : PCI Bus: 0000:01:00.5
GPU[19]         : PCI Bus: 0000:01:00.5
GPU[20]         : PCI Bus: 0000:01:00.5
GPU[21]         : PCI Bus: 0001:01:00.0
GPU[22]         : PCI Bus: 0002:01:00.0
GPU[23]         : PCI Bus: 0003:01:00.0
================================== End of ROCm SMI Log ===================================
```
3. Displayed topology will show disordered nodes when compared to HIP
See rocm-smi output vs transferbench.
```shell
rocm-smi --showtopo option is not displaying the correct information when the MI300 driver is loaded in TPX mode.


============================ ROCm System Management Interface ============================
================================ Weight between two GPUs =================================
get_link_weight_topology, Not supported on the given system
ERROR: GPU[1]   : Cannot read Link Weight: Not supported on this machine

       GPU0         GPU1         GPU2         GPU3         GPU4         GPU5         GPU6         GPU7         GPU8                                                                                                GPU9         GPU10        GPU11
GPU0   0            XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI                                                                                                XGMI         XGMI         XGMI
GPU1   XGMI         0            XXXX         XXXX         XXXX         XGMI         XGMI         XGMI         XGMI                                                                                                XGMI         XGMI         XGMI
GPU2   XGMI         XXXX         0            XXXX         XXXX         XGMI         XGMI         XGMI         XGMI                                                                                                XGMI         XGMI         XGMI
GPU3   XGMI         XXXX         XXXX         0            XXXX         XGMI         XGMI         XGMI         XGMI                                                                                                XGMI         XGMI         XGMI
GPU4   XGMI         XXXX         XXXX         XXXX         0            XGMI         XGMI         XGMI         XGMI                                                                                                XGMI         XGMI         XGMI
GPU5   XGMI         XGMI         XGMI         XGMI         XGMI         0            XXXX         XXXX         XXXX                                                                                                XGMI         XGMI         XGMI
GPU6   XGMI         XGMI         XGMI         XGMI         XGMI         XXXX         0            XXXX         XXXX                                                                                                XGMI         XGMI         XGMI
GPU7   XGMI         XGMI         XGMI         XGMI         XGMI         XXXX         XXXX         0            XXXX                                                                                                XGMI         XGMI         XGMI
GPU8   XGMI         XGMI         XGMI         XGMI         XGMI         XXXX         XXXX         XXXX         0                                                                                                   XGMI         XGMI         XGMI
GPU9   XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI                                                                                                0            XGMI         XGMI
GPU10  XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI                                                                                                XGMI         0            XGMI
GPU11  XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI                                                                                                XGMI         XGMI         0

======================================= Numa Nodes =======================================
GPU[0]          : (Topology) Numa Node: 0
GPU[0]          : (Topology) Numa Affinity: 0
GPU[1]          : (Topology) Numa Node: 0
GPU[1]          : (Topology) Numa Affinity: 0
GPU[2]          : (Topology) Numa Node: 0
GPU[2]          : (Topology) Numa Affinity: 1
GPU[3]          : (Topology) Numa Node: 0
GPU[3]          : (Topology) Numa Affinity: 2
GPU[4]          : (Topology) Numa Node: 0
GPU[4]          : (Topology) Numa Affinity: 3
GPU[5]          : (Topology) Numa Node: 0
GPU[5]          : (Topology) Numa Affinity: 0
GPU[6]          : (Topology) Numa Node: 0
GPU[6]          : (Topology) Numa Affinity: 1
GPU[7]          : (Topology) Numa Node: 0
GPU[7]          : (Topology) Numa Affinity: 2
GPU[8]          : (Topology) Numa Node: 0
GPU[8]          : (Topology) Numa Affinity: 3
GPU[9]          : (Topology) Numa Node: 1
GPU[9]          : (Topology) Numa Affinity: 1
GPU[10]         : (Topology) Numa Node: 2
GPU[10]         : (Topology) Numa Affinity: 2
GPU[11]         : (Topology) Numa Node: 3
GPU[11]         : (Topology) Numa Affinity: 3
================================== End of ROCm SMI Log ===================================
```

```shell
./Transferbench
...
        | GPU 00 | GPU 01 | GPU 02 | GPU 03 | GPU 04 | GPU 05 | GPU 06 | GPU 07 | PCIe Bus ID  | #CUs | Closest NUMA | DMA engines
--------+--------+--------+--------+--------+--------+--------+--------+--------+--------------+------+-------------+------------
 GPU 00 |    -   | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | 0000:0c:00.0 |  304 | 0            |0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
 GPU 01 | XGMI-1 |    -   | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | 0000:22:00.0 |  304 | 0            |0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
 GPU 02 | XGMI-1 | XGMI-1 |    -   | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | 0000:38:00.0 |  304 | 0            |0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
 GPU 03 | XGMI-1 | XGMI-1 | XGMI-1 |    -   | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | 0000:5c:00.0 |  304 | 0            |0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
 GPU 04 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 |    -   | XGMI-1 | XGMI-1 | XGMI-1 | 0000:9f:00.0 |  304 | 1            |0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
 GPU 05 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 |    -   | XGMI-1 | XGMI-1 | 0000:af:00.0 |  304 | 1            |0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
 GPU 06 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 |    -   | XGMI-1 | 0000:bf:00.0 |  304 | 1            |0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
 GPU 07 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 | XGMI-1 |    -   | 0000:df:00.0 |  304 | 1            |0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
...
```

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

### Optimized

- Add new test to measure api execution time.
- Remove the shared mutex if no process is using it.
- Updated to C++17, gtest-1.14, and cmake 3.14

### Resolved issues

- Fix memory usage division by 0
- Fix missing firmware blocks (rocm-smi --showfw)
- Fix rocm-smi --showevents shows wrong gpuID

## rocm_smi_lib for ROCm 5.5.0

### Added

- ROCm SMI CLI: Add --showtempgraph Feature.

### Changed

- Relying on vendor ID to detect AMDGPU.
- Change pragma message to warning for backward compatibility.

### Optimized

- Add new test to measure api execution time.
- Remove the shared mutex if no process is using it.

### Resolved issues

- Fix --showproductname when device's SKU cannot be parsed out of the VBIOS string.
- Fix compile error: ‘memcpy’ was not declared.
- Fix order of CE and UE reporting in ROCm SMI CLI.
- Handle error return value from ReadSysfsStr function.
