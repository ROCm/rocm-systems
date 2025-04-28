# Changelog for AMD SMI Library

Full documentation for amd_smi_lib is available at [https://rocm.docs.amd.com/projects/amdsmi](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).

***All information listed below is for reference and subject to change.***

## amd_smi_lib for ROCm 6.5.0

### Added

- **Added bad page threshold count**.  
  - Added `amdsmi_get_gpu_bad_page_threshold` to Python API and CLI; root/sudo permissions required to display the count.

### Changed

- **The `amd-smi topology` command has been enabled for Guest environments**.  
  - `amd-smi topology` is now availabe in Guest environments. This includes full functionality so users can use the command just as they would in Bare Metal environments.

### Removed

### Optimized

### Resolved issues

### Upcoming changes

### Known issues

- **Added cpu model name for RDC**.  
  - Added new C and Python API `amdsmi_get_cpu_model_name`
  - Not sourced from esmi library.

- **Added support for GPU metrics 1.8**.  
  - Added new fields for `amdsmi_gpu_xcp_metrics_t` including:  
    - Adding the following metrics to allow new calculations for violation status:
    - Per XCP metrics `gfx_below_host_limit_ppt_acc[XCP][MAX_XCC]` - GFX Clock Host limit Package Power Tracking violation counts
    - Per XCP metrics `gfx_below_host_limit_thm_acc[XCP][MAX_XCC]` - GFX Clock Host limit Thermal (TVIOL) violation counts
    - Per XCP metrics `gfx_low_utilization_acc[XCP][MAX_XCC]` - violation counts for how did low utilization caused the GPU to be below application clocks.
    - Per XCP metrics `gfx_below_host_limit_total_acc[XCP][MAX_XCC]`- violation counts for how long GPU was held below application clocks any limiter (see above new violation metrics).
  - Increasing available JPEG engines to 40.  
  Current ASICs may not support all 40. These will be indicated as UINT16_MAX or N/A in CLI.

### Changed

- **Added Power Cap to amd-smi monitor**.  
  - 'amd-smi monitor -p' will display the power cap along with power.

    ```shell
    $ amd-smi monitor -p
    GPU  POWER  PWR_CAP
      0  148 W    750 W
      1  156 W    750 W
      2  153 W    750 W
      ...
    ```

- **Modified VRAM display for `amd-smi monitor -v`**.  
  - Added free VRAM and VRAM percentage.

    ```shell
    $ amd-smi monitor -v
    GPU  VRAM_USED   VRAM_FREE  VRAM_TOTAL    VRAM%
      0     174 MB    16011 MB    16185 MB   0.01 %
      1      78 MB      347 MB      425 MB   0.18 %
      ...
    ```

### Resolved issues

- **Deduplicated GPU IDs when receiving events using the `amd-smi event` command**.  

## amd_smi_lib for ROCm 6.4.1

### Added

- **Added dumping CPER entries from RAS tool `amdsmi_get_gpu_cper_entries()` to Python & C APIs.**  
  - CPER entries consist of `amdsmi_cper_hdr_t`

    ```shell
    typedef struct {
        char                   signature[4];       /* "CPER" */
        uint16_t               revision;
        uint32_t               signature_end;      /* 0xFFFFFFFF */
        uint16_t               sec_cnt;
        amdsmi_cper_sev_t      error_severity;
        //valid_bits_t          valid_bits;
        //uint32_t              valid_mask;    
        amdsmi_cper_valid_bits_t cper_valid_bits; 
        uint32_t                record_length;     /* Total size of CPER Entry */
        amdsmi_cper_timestamp_t timestamp;
        char                    platform_id[16];
        amdsmi_cper_guid_t      partition_id;      /* Reserved */
        char                  creator_id[16];
        amdsmi_cper_guid_t    notify_type;         /* CMC, MCE, can use amdsmi_cper_notifiy_type_t to decode*/
        char                  record_id[8];        /* Unique CPER Entry ID */
        uint32_t              flags;               /* Reserved */
        uint64_t              persistence_info;    /* Reserved */
        uint8_t               reserved[12];        /* Reserved */
    } amdsmi_cper_hdr_t;
    ```

  - Dumping CPER entires is also enabled in the CLI interface via `sudo amd-smi ras --cper`

    ```shell
    $ sudo amd-smi ras --cper
    Dumping CPER file header entries for GPU 0:
    "0": {
       "error_severity": "non_fatal_corrected",
       "notify_type": "CMC",
       "timestamp": "2025/04/08 18:23:44",
       "signature": "CPER",
       "revision": 256,
       "signature_end": "0xffffffff",
       "sec_cnt": 1,
       "record_length": 472,
       "platform_id": "0x1002:0x74A2",
       "creator_id": "amdgpu",
       "record_id": "5:1",
       "flags": 0,
       "persistence_info": 0
       }
    ```
### Changed

- **Changed amd-smi partition --accelerator & `amdsmi_get_gpu_accelerator_partition_profile_config()` detect users running without root/sudo privledges**
     - Updated  `amdsmi_get_gpu_accelerator_partition_profile_config()` to return `AMDSMI_STATUS_NO_PERM` immediately
       if users run without root/sudo permissions.
     - Updated `amd-smi partition --accelerator` to provide a warning for users without root/sudo permissions (see example below, ***output subject to change***).

    ```shell
    $ amd-smi partition --accelerator

    ACCELERATOR_PARTITION_PROFILES:

    ***************************************************************************
    ** WARNING:                                                              **
    ** ACCELERATOR_PARTITION_PROFILES requires sudo/root permissions to run. **
    ** Please run the command with sudo permissions to get accurate results. **
    ***************************************************************************

    GPU_ID  PROFILE_INDEX  MEMORY_PARTITION_CAPS  ACCELERATOR_TYPE  PARTITION_ID     NUM_PARTITIONS  NUM_RESOURCES  RESOURCE_INDEX  RESOURCE_TYPE  RESOURCE_INSTANCES  RESOURCES_SHARED
    N/A     N/A            N/A                    N/A               0                N/A             N/A            N/A             N/A            N/A                 N/A
    N/A     N/A            N/A                    N/A               0                N/A             N/A            N/A             N/A            N/A                 N/A
    N/A     N/A            N/A                    N/A               0                N/A             N/A            N/A             N/A            N/A                 N/A
    N/A     N/A            N/A                    N/A               0                N/A             N/A            N/A             N/A            N/A                 N/A
    N/A     N/A            N/A                    N/A               0                N/A             N/A            N/A             N/A            N/A                 N/A
    N/A     N/A            N/A                    N/A               0                N/A             N/A            N/A             N/A            N/A                 N/A
    N/A     N/A            N/A                    N/A               0                N/A             N/A            N/A             N/A            N/A                 N/A
    N/A     N/A            N/A                    N/A               0                N/A             N/A            N/A             N/A            N/A                 N/A

    ACCELERATOR_PARTITION_RESOURCES:
    RESOURCE_INDEX  RESOURCE_TYPE  RESOURCE_INSTANCES  RESOURCES_SHARED
    N/A             N/A            N/A                 N/A
    N/A             N/A            N/A                 N/A
    N/A             N/A            N/A                 N/A
    N/A             N/A            N/A                 N/A
    N/A             N/A            N/A                 N/A
    N/A             N/A            N/A                 N/A
    N/A             N/A            N/A                 N/A
    N/A             N/A            N/A                 N/A


    Legend:
    * = Current mode
    ```

- **Changed `amd-smi partition --current`, `amd-smi partition --accelerator`, and `amdsmi_get_gpu_accelerator_partition_profile()` to display partition ID for each individual partition**
    - Host will continue to display in the full array format, they do not display the individual partitions as Baremetal/Guest setups.
    - Baremetal and Guest MI3x setups will change to
reflect each individual partition ID, now provided in `partition_id[0]` location (as seen in other amd-smi CLI commands).  
This change was needed for BM/Guest setups due to other related partition outputs seen in (`amd-smi list` and `amd-smi static --partition`) and individual logical partition devices displayed. ***See examples below for reference.***  

    Previous output:

    ```shell
    $ amd-smi partition --current

    CURRENT_PARTITION:
    GPU_ID  MEMORY  ACCELERATOR_TYPE  ACCELERATOR_PROFILE_INDEX  PARTITION_ID
    0       NPS1    CPX               3                          0,1,2,3,4,5,6,7
    1       NPS1    CPX               3                          N/A
    2       NPS1    CPX               3                          N/A
    3       NPS1    CPX               3                          N/A
    4       NPS1    CPX               3                          N/A
    5       NPS1    CPX               3                          N/A
    6       NPS1    CPX               3                          N/A
    7       NPS1    CPX               3                          N/A
    8       NPS1    CPX               3                          0,1,2,3,4,5,6,7
    9       NPS1    CPX               3                          N/A
    10      NPS1    CPX               3                          N/A
    ...
    ```

    New output:

    ```shell
    amd-smi partition --current
    CURRENT_PARTITION:
    GPU_ID  MEMORY  ACCELERATOR_TYPE  ACCELERATOR_PROFILE_INDEX  PARTITION_ID
    0       NPS1    CPX               3                          0
    1       NPS1    CPX               3                          1
    2       NPS1    CPX               3                          2
    3       NPS1    CPX               3                          3
    4       NPS1    CPX               3                          4
    5       NPS1    CPX               3                          5
    6       NPS1    CPX               3                          6
    7       NPS1    CPX               3                          7
    8       NPS1    CPX               3                          0
    9       NPS1    CPX               3                          1
    10      NPS1    CPX               3                          2
    ...
    ```

### Removed

- N/A

### Optimized

- N/A

### Resolved issues

- **Fixed partition enumeration - `amd-smi list -e`, `amdsmi_get_gpu_enumeration_info()`'s  `amdsmi_enumeration_info_t` `drm_card` and `drm_render` fields**  
    Previously, partitions incorrectly reflected the primary node (1st GPU) and showed the DRM Render Minor as renderD128. Partition nodes mirrored renderD128's information, which was incorrect. See the "<i>Previous Outputs in CPX</i>" example below.

    Device enumeration was updated to correctly map DRM Render Minor paths. See the "<i>Corrected Outputs in CPX</i>" example below.

    These changes impact what information is readable/writable for the partition nodes.

    <b><i>Example: Previous Outputs in CPX</b></i>  
    ```shell
    $ amd-smi list -e                                                                    
    GPU: 0
        BDF: 0000:0c:00.0
        UUID: <Redacted>
        KFD_ID: 18421
        NODE_ID: 2
        PARTITION_ID: 0
        RENDER: renderD128
        CARD: card0
        HSA_ID: 2
        HIP_ID: 0
        HIP_UUID: <Redacted>

    GPU: 1
        BDF: 0000:0c:00.1
        UUID: <Redacted>
        KFD_ID: 48116
        NODE_ID: 3
        PARTITION_ID: 1
        RENDER: N/A
        CARD: N/A
        HSA_ID: 3
        HIP_ID: 1
        HIP_UUID: GPU-<Redacted>
    ...
    ```
    ```shell
    $ amd-smi monitor
    GPU  POWER   GPU_T   MEM_T   GFX_CLK   GFX%   MEM%   ENC%   DEC%      VRAM_USAGE
      0  201 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      1  201 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      2  201 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      3  201 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      4  201 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      5  201 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      6  201 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      7  201 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      8  210 W   46 °C   42 °C  2104 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
    ...
    ```
    <b><i>Example: Corrected outputs in CPX</i></b>
    ```shell
    $ amd-smi list -e
    GPU: 0
        BDF: 0000:0c:00.0
        UUID: <Redacted>
        KFD_ID: 18421
        NODE_ID: 2
        PARTITION_ID: 0
        RENDER: renderD128
        CARD: card0
        HSA_ID: 2
        HIP_ID: 0
        HIP_UUID: GPU-<Redacted>

    GPU: 1
        BDF: 0000:0c:00.1
        UUID: <Redacted>
        KFD_ID: 48116
        NODE_ID: 3
        PARTITION_ID: 1
        RENDER: renderD129
        CARD: card1
        HSA_ID: 3
        HIP_ID: 1
        HIP_UUID: GPU-<Redacted>
    ...
    ```  
    ```shell
    $ amd-smi monitor
    GPU  POWER   GPU_T   MEM_T   GFX_CLK   GFX%   MEM%   ENC%   DEC%      VRAM_USAGE
      0  202 W   46 °C   42 °C  2107 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
      1    N/A     N/A     N/A       N/A    N/A    N/A    N/A    N/A    0.5/ 24.0 GB
      2    N/A     N/A     N/A       N/A    N/A    N/A    N/A    N/A    0.5/ 24.0 GB
      3    N/A     N/A     N/A       N/A    N/A    N/A    N/A    N/A    0.5/ 24.0 GB
      4    N/A     N/A     N/A       N/A    N/A    N/A    N/A    N/A    0.5/ 24.0 GB
      5    N/A     N/A     N/A       N/A    N/A    N/A    N/A    N/A    0.5/ 24.0 GB
      6    N/A     N/A     N/A       N/A    N/A    N/A    N/A    N/A    0.5/ 24.0 GB
      7    N/A     N/A     N/A       N/A    N/A    N/A    N/A    N/A    0.5/ 24.0 GB
      8  210 W   46 °C   42 °C  2104 MHz    0 %    0 %    N/A    0 %    0.3/192.0 GB
    ...
    ```

### Upcoming changes

- N/A

### Known issues

- N/A


## amd_smi_lib for ROCm 6.4.0

### Added

- **Added enumeration mapping `amdsmi_get_gpu_enumeration_info()` to Python & C APIs.**  
  - Enumeration mapping consists of `amdsmi_enumeration_info_t`

    ```shell
    typedef struct {
        uint32_t drm_render; // the render node under /sys/class/drm/renderD*
        uint32_t drm_card;   // the graphic card device under /sys/class/drm/card*
        uint32_t hsa_id;     // the HSA enumeration ID
        uint32_t hip_id;     // the HIP enumeration ID
        char hip_uuid[AMDSMI_MAX_STRING_LENGTH];  // the HIP unique identifier
    } amdsmi_enumeration_info_t;
    ```

  - The mapping is also enabled in the CLI interface via `amd-smi list -e`

    ```shell
    $ amd-smi list -e
    GPU: 0
        BDF: 0000:23:00.0
        UUID: XXXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX
        KFD_ID: 45412
        NODE_ID: 1
        PARTITION_ID: 0
        RENDER: renderD128
        CARD: card0
        HSA_ID: 1
        HIP_ID: 0
        HIP_UUID: GPU-XXXXXXXXXXXXXXXX
    ```

- **Added dynamic virtualization mode detection**.  
  - Added new C and Python API `amdsmi_get_gpu_virtualization_mode`
  - Added new C and Python enum `amdsmi_virtualization_mode_t`

- **Added TVIOL_ACTIVE to `amd-smi monitor`**.  
Added temperature violation active or not status to `amd-smi monitor`. TVIOL_ACTIVE will be displayed as below:
  - True if active
  - False if not active
  - N/A if not supported.

  Example CLI output:

    ```shell
    $ amd-smi monitor --viol
    GPU  PVIOL  TVIOL  TVIOL_ACTIVE  PHOT_TVIOL  VR_TVIOL  HBM_TVIOL
    0  100 %    1 %          True         0 %       0 %        0 %
    1  100 %    0 %         False         0 %       0 %        0 %
    2  100 %    0 %         False         0 %       0 %        0 %
    3  100 %    0 %         False         0 %       0 %        0 %
    4  100 %    0 %         False         0 %       0 %        0 %
    5  100 %    3 %          True         0 %       0 %        0 %
    6  100 %    0 %         False         0 %       0 %        0 %
    7  100 %    0 %         False         0 %       0 %        0 %
    ```

- **Added support for GPU metrics 1.7 to `amdsmi_get_gpu_metrics_info()`**.  
Updated `amdsmi_get_gpu_metrics_info()` and structure `amdsmi_gpu_metrics_t` to include new fields for XGMI Link Status, graphics clocks below host limit (per XCP), and VRAM max bandwidth:  
  - `uint64_t vram_max_bandwidth` - VRAM max bandwidth at max memory clock (GB/s)
  - `uint16_t xgmi_link_status[MAX_NUM_XGMI_LINKS]` - XGMI link statis, 1=Up 0=Down
  - `uint64_t gfx_below_host_limit_acc[MAX_NUM_XCC]` - graphics clocks below host limit (per XCP) accumulators. Used for graphic clk below host limit violation status.

- **Added new API `amdsmi_get_gpu_xgmi_link_status()` and CLI `amd-smi xgmi --link-status`**  

    New API is defined as:

    ```C
    typedef enum {
    AMDSMI_XGMI_LINK_DOWN,     //!< The XGMI Link is down
    AMDSMI_XGMI_LINK_UP,       //!< The XGMI Link is up
    AMDSMI_XGMI_LINK_DISABLE,  //!< The XGMI Link is disabled
    } amdsmi_xgmi_link_status_type_t;

    typedef struct {
    uint32_t total_links;     //!< The total links in the status array
    amdsmi_xgmi_link_status_type_t status[AMDSMI_MAX_NUM_XGMI_LINKS];
    uint64_t reserved[7];
    } amdsmi_xgmi_link_status_t;

    amdsmi_status_t amdsmi_get_gpu_xgmi_link_status(amdsmi_processor_handle processor_handle, amdsmi_xgmi_link_status_t *link_status)
    ```

    Example CLI output:

    ```shell
    $ amd-smi xgmi --link-status

    XGMI LINK STATUS:
        bdf           link_status
    GPU0   0000:08:00.0  U  U  U  U  D  U  D  X
    GPU1   0000:44:00.0  U  U  U  U  D  U  D  X
    ...

    * U:Up D:Down X:Disabled
    ```

- **Added fclk and socclk info to `amd-smi metric -c/--clock`**.  
  fclk and socclk information such as min and max clock have been added to the metric command, in line with all the other clocks.  

  ```shell
  amd-smi metric -c -g 1
  ...
          FCLK_0:
            CLK: 2301 MHz
            MIN_CLK: 601 MHz
            MAX_CLK: 2301 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: DISABLED
        SOCCLK_0:
            CLK: 1500 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1500 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: DISABLED
  ```

- **Added new command `amd-smi set -c/--clock-level`**.  
  This new command sets the performance level of the selected clock on the desired GPUs. The command can accept a range of acceptable levels, but will not set the level when a level is beyond the number of frequency levels as show in `amd-smi static -C/--clock`.  

    ```shell
    sudo amd-smi set -c sclk 5 6
    GPU: 0
        CLK_LEVEL: Successfully changed sclk perf level(s) to 5, 6

    GPU: 1
        CLK_LEVEL: level(s) 5, 6 is/are greater than performance levels supported for device
    ```

- **Added new command `amd-smi static -C/--clock`**.  
  This new command displays the clock frequency performance levels for the selected GPUs and clocks.

    ```shell
    amd-smi static --clock all -g 0
    GPU: 0
        CLOCK:
            SYS:
                CURRENT LEVEL: 2
                FREQUENCY_LEVELS:
                    0: 300 MHz
                    1: 904 MHz
                    2: 1165 MHz
                    3: 1360 MHz
                    4: 1440 MHz
                    5: 1544 MHz
                    6: 1627 MHz
                    7: 1720 MHz
                    8: 1800 MHz
            MEM:
                CURRENT LEVEL: 0
                FREQUENCY_LEVELS:
                    0: 167 MHz
            DF:
                CURRENT LEVEL: 0
                FREQUENCY_LEVELS:
                    0: 1400 MHz
            SOC:
                CURRENT LEVEL: 0
                FREQUENCY_LEVELS:
                    0: 302 MHz
            DCEF: N/A
            VCLK0: N/A
            VCLK1: N/A
            DCLK0: N/A
            DCLK1: N/A
    ```

### Changed

- **AMDSMI Library Version number to reflect changes in backwards compatability**.  
  - Removed Year from AMDSMI Library version number.
  - Version changed from 25.2.0.0 (Year.Major.Minor.Patch) to 25.2.0 (Major.Minor.Patch)
  - Removed year in all version references

- **Removed initialization requirements for `amdsmi_get_lib_version()` and added `amdsmi_get_rocm_version()` to the python API & CLI**.  

- **Added `amdsmi_get_power_info_v2()` with `sensor_ind`**.
  - Python API now accepts `sensor_ind` as an optional argument, does not impact previous usage

- **Depricated enum `AMDSMI_NORMAL_STRING_LENGTH` in favor of `AMDSMI_MAX_STRING_LENGTH`**.  

- **Changed to use thread local mutex by default**.  
  - Most sysfs reads do not require cross-process level mutex, and writes to sysfs should be protected by the kernel already.
  - Users can still switch to the old behavior by setting the environment variable `AMDSMI_MUTEX_CROSS_PROCESS=1`.

- **Changed `amdsmi_vram_vendor_type_t` enum names impacting `amdsmi_vram_info_t` structure**.  
This also change impacts usage of the vram_vendor output of `amdsmi_get_gpu_vram_info()`

- **Changed `amdsmi_nps_caps_t` struct impacting `amdsmi_memory_partition_config_t`, `amdsmi_accelerator_partition_t`, `amdsmi_accelerator_partition_profile_config_t`**.  
Functions affected by struct change are:
  - `amdsmi_get_gpu_memory_partition_config()`
  - `amdsmi_get_gpu_accelerator_partition_profile()`
  - `amdsmi_get_gpu_accelerator_partition_profile_config()`

- **Corrected CLI CPU argument name**.  
  - `--cpu-pwr-svi-telemtry-rails` to `--cpu-pwr-svi-telemetry-rails`

- **Added amdgpu driver version and amd_hsmp driver version to `amd-smi version` command**.  
  - The `amd-smi version` command can now also display the amdgpu driver version using the `-g` flag.
  - The amd_hsmp driver version can also be displayed using the `-c` flag.
  - The new default for the `version` command is to display all the version information, including both amdgpu and amd_hsmp driver versions.

    ```shell
    amd-smi version
    AMDSMI Tool: 24.7.1+b446d6c-dirty | AMDSMI Library version: 24.7.2.0 | ROCm version: N/A | amdgpu version: 6.10.10 | amd_hsmp version: 2.2

    amd-smi version -g
    AMDSMI Tool: 24.7.1+b446d6c-dirty | AMDSMI Library version: 24.7.2.0 | ROCm version: N/A | amdgpu version: 6.10.10

    amd-smi version -c
    AMDSMI Tool: 24.7.1+b446d6c-dirty | AMDSMI Library version: 24.7.2.0 | ROCm version: N/A | amd_hsmp version: 2.2
    ```

- **All `amd-smi set` and `amd-smi reset` options are now mutually exclusive**.  
  - Users can only use one set option at a time now.  

- **Python API for `amdsmi_get_energy_count()` will change the name for the `power` field to `energy_accumulator`**.  

- **Added violation status output for Graphics Clock Below Host Limit to our CLI: `amdsmi_get_violation_status()`, `amd-smi metric  --throttle`, and `amd-smi monitor --violation`.**  
  ***Only available for MI300+ ASICs.***  
  Users can retrieve violation status' through either our Python or C++ APIs.  
  Additionally, we have added capability to view these outputs conviently through `amd-smi metric --throttle` and `amd-smi monitor --violation`.  
  Example outputs are listed below (below is for reference, output is subject to change):

    ```shell
    $ amd-smi monitor --violation
    GPU  PVIOL  TVIOL  TVIOL_ACTIVE PHOT_TVIOL  VR_TVIOL  HBM_TVIOL  GFX_CLKVIOL
    0    0 %    0 %           False        0 %       0 %        0 %          0 %
    1    0 %    0 %           False        0 %       0 %        0 %          0 %
    ...
    ```

    ```shell
    $ amd-smi metric --throttle
    GPU: 0
        THROTTLE:
            ACCUMULATION_COUNTER: 11240028
            PROCHOT_ACCUMULATED: 0
            PPT_ACCUMULATED: 0
            SOCKET_THERMAL_ACCUMULATED: 0
            VR_THERMAL_ACCUMULATED: 0
            HBM_THERMAL_ACCUMULATED: 0
            GFX_CLK_BELOW_HOST_LIMIT_ACCUMULATED: N/A
            PROCHOT_VIOLATION_STATUS: NOT ACTIVE
            PPT_VIOLATION_STATUS: NOT ACTIVE
            SOCKET_THERMAL_VIOLATION_STATUS: NOT ACTIVE
            VR_THERMAL_VIOLATION_STATUS: NOT ACTIVE
            HBM_THERMAL_VIOLATION_STATUS: NOT ACTIVE
            GFX_CLK_BELOW_HOST_LIMIT_VIOLATION_STATUS: N/A
            PROCHOT_VIOLATION_ACTIVITY: 0 %
            PPT_VIOLATION_ACTIVITY: 0 %
            SOCKET_THERMAL_VIOLATION_ACTIVITY: 0 %
            VR_THERMAL_VIOLATION_ACTIVITY: 0 %
            HBM_THERMAL_VIOLATION_ACTIVITY: 0 %
            GFX_CLK_BELOW_HOST_LIMIT_VIOLATION_ACTIVITY: 0 %

    GPU: 1
        THROTTLE:
            ACCUMULATION_COUNTER: 11238232
            PROCHOT_ACCUMULATED: 0
            PPT_ACCUMULATED: 0
            SOCKET_THERMAL_ACCUMULATED: 0
            VR_THERMAL_ACCUMULATED: 0
            HBM_THERMAL_ACCUMULATED: 0
            GFX_CLK_BELOW_HOST_LIMIT_ACCUMULATED: 0
            PROCHOT_VIOLATION_STATUS: NOT ACTIVE
            PPT_VIOLATION_STATUS: NOT ACTIVE
            SOCKET_THERMAL_VIOLATION_STATUS: NOT ACTIVE
            VR_THERMAL_VIOLATION_STATUS: NOT ACTIVE
            HBM_THERMAL_VIOLATION_STATUS: NOT ACTIVE
            GFX_CLK_BELOW_HOST_LIMIT_VIOLATION_STATUS: NOT ACTIVE
            PROCHOT_VIOLATION_ACTIVITY: 0 %
            PPT_VIOLATION_ACTIVITY: 0 %
            SOCKET_THERMAL_VIOLATION_ACTIVITY: 0 %
            VR_THERMAL_VIOLATION_ACTIVITY: 0 %
            HBM_THERMAL_VIOLATION_ACTIVITY: 0 %
            GFX_CLK_BELOW_HOST_LIMIT_VIOLATION_ACTIVITY: 0 %
    ...
    ```

- **Updated API `amdsmi_get_violation_status()` structure and CLI `amdsmi_violation_status_t` to include GFX Clk below host limit**  
Updated structure `amdsmi_violation_status_t`:  

    ```C
    typedef struct {
        ...
        uint64_t acc_gfx_clk_below_host_limit;  //!< Current graphic clock below host limit count; Max uint64 means unsupported
        ...
        uint64_t per_gfx_clk_below_host_limit;  //!< Graphics clock below host limit violation % (greater than 0% is a violation); Max uint64 means unsupported
        ...
        uint8_t active_gfx_clk_below_host_limit;  //!< Graphics clock below host limit violation; 1 = active 0 = not active; Max uint8 means unsupported
        ...
    } amdsmi_violation_status_t;
    ```

- **Updated API `amdsmi_get_gpu_vram_info()` structure and CLI `amd-smi static --vram`**  
Updated structure `amdsmi_vram_info_t`:  

    ```C
    typedef struct {
    amdsmi_vram_type_t vram_type;
    amdsmi_vram_vendor_type_t vram_vendor;
    uint64_t vram_size;
    uint32_t vram_bit_width;
    uint64_t vram_max_bandwidth;   //!< The VRAM max bandwidth at current memory clock (GB/s)
    uint64_t reserved[4];
    } amdsmi_vram_info_t;

    amdsmi_status_t amdsmi_get_gpu_vram_info(amdsmi_processor_handle processor_handle, amdsmi_vram_info_t *info)
    ```

  Example CLI output:

    ```shell
    $ amd-smi static --vram
    GPU: 0
        VRAM:
            TYPE: GDDR6
            VENDOR: N/A
            SIZE: 16368 MB
            BIT_WIDTH: 256
            MAX_BANDWIDTH: 1555 GB/s
    GPU: 1
        VRAM:
            TYPE: GDDR6
            VENDOR: N/A
            SIZE: 30704 MB
            BIT_WIDTH: 256
            MAX_BANDWIDTH: 1555 GB/s
    ...
    ```

### Removed

- **Removed `GFX_BUSY_ACC` from `amd-smi metric --usage`**.  
  Displaying `GFX_BUSY_ACC` does not provide helpful outputs for users.  

  Old output:
  
  ```shell
  $ amd-smi metric --usage
    GPU: 0
        USAGE:
            GFX_ACTIVITY: 0 %
            UMC_ACTIVITY: 0 %
            MM_ACTIVITY: N/A
            VCN_ACTIVITY: [0 %, 0 %, 0 %, 0 %]
            JPEG_ACTIVITY: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %]
            GFX_BUSY_INST:
                XCP_0: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %]
            JPEG_BUSY:
                XCP_0: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %]
            VCN_BUSY:
                XCP_0: [0 %, 0 %, 0 %, 0 %]
            GFX_BUSY_ACC:
                XCP_0: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
  ...
  ```

  New Output:

  ```shell
  $ amd-smi metric --usage
  GPU: 0
      USAGE:
          GFX_ACTIVITY: 0 %
          UMC_ACTIVITY: 0 %
          MM_ACTIVITY: N/A
          VCN_ACTIVITY: [0 %, 0 %, 0 %, 0 %]
          JPEG_ACTIVITY: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %]
          GFX_BUSY_INST:
              XCP_0: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %]
          JPEG_BUSY:
              XCP_0: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %]
          VCN_BUSY:
              XCP_0: [0 %, 0 %, 0 %, 0 %]
  ...
  ```

### Optimized

- **Added additional help information to `amd-smi set --help` command**.  
  - sub commands now detail what values are acceptable as input. These include:
    - `amd-smi set --perf-level` with performance levels
    - `amd-smi set --profile` with power profiles
    - `amd-smi set --perf-determinism` with preset GPU frequency limits
    - `amd-smi set --power-cap` with valid power cap values
    - `amd-smi set --soc-pstate` with soc pstate policy ids
    - `amd-smi set --xgmi-plpd` with xgmi per link power down policy ids

- **Modified `amd-smi` CLI to allow case insensitive arguments if the argument does not begin with a single dash**.  
  - With this change `amd-smi version` and `amd-smi VERSION` will now yield the same output.
  - `amd-smi static --bus` and `amd-smi STATIC --BUS` will produce identical results.
  - `amd-smi static -b` and `amd-smi static -B` will still return different results (-b for bus and -B for board).

- **Converted xgmi read and write from KB's to readable units**.  
  - With this change `amd-smi xgmi` will now display the statistics in dynamically selected readable units.
  - Example output CLI output:

    ```shell
    $ amd-smi xgmi
    LINK METRIC TABLE:
        bdf          bit_rate max_bandwidth link_type 0000:05:00.0 0000:26:00.0 0000:46:00.0 0000:65:00.0 0000:85:00.0 0000:a6:00.0 0000:c6:00.0 0000:e5:00.0
    GPU0   0000:05:00.0 32 Gb/s  512 Gb/s      XGMI
    Read                                                N/A          1.123 PB     1.123 PB     1.123 PB     1.123 PB     1.123 PB     1.123 PB     1.123 PB
    Write                                               N/A          229.1 MB     229.1 MB     229.1 MB     229.1 MB     229.1 MB     229.1 MB     229.1 MB
    GPU1   0000:26:00.0 32 Gb/s  512 Gb/s      XGMI
    Read                                                1.123 PB     N/A          1.123 PB     1.123 PB     1.123 PB     1.123 PB     1.123 PB     1.123 PB
    Write                                               229.1 MB     N/A          229.1 MB     229.1 MB     229.1 MB     229.1 MB     229.1 MB     229.1 MB
    GPU2   0000:46:00.0 32 Gb/s  512 Gb/s      XGMI
    Read                                                1.123 PB     1.123 PB     N/A          1.123 PB     1.123 PB     1.123 PB     1.123 PB     1.123 PB
    Write                                               229.1 MB     229.1 MB     N/A          229.1 MB     229.1 MB     229.1 MB     229.1 MB     229.1 MB
    ...
    ```

### Resolved issues

- **Fixed `amd-smi static --partition` for guest systems with MIx ASICs being unable to run**

- **Fixed `amdsmi_get_gpu_asic_info` and `amd-smi static --asic` not displaying graphics version properly for MI2x, MI1x or Navi 3x ASICs.**  

  Before on MI100:

  ```shell
  $ amd-smi static --asic | grep TARGET_GRAPHICS_VERSION
        TARGET_GRAPHICS_VERSION: gfx9008
        TARGET_GRAPHICS_VERSION: gfx9008
  ```

  After on MI100:

  ```shell
  $ amd-smi static --asic | grep TARGET_GRAPHICS_VERSION
        TARGET_GRAPHICS_VERSION: gfx908
        TARGET_GRAPHICS_VERSION: gfx908
  ```

### Upcoming changes

- **Deprication in ROCm 7.0 of the `AMDSMI_LIB_VERSION_YEAR` enum and API fields.**  

- **Deprication in ROCm 7.0 of the `pasid` field within struct `amdsmi_process_info_t`**  

### Known issues

- **AMD SMI only reports 63 GPU devices when setting CPX on all 8 GPUs**  
    When setting CPX as a partition mode, there is a DRM node limitation of 64.  
    This is a known limitation of the Linux kernel, not the driver. Other drivers, such as those using PCIe space (e.g., ast), may be occupying the necessary DRM nodes.  
    The number of DRM nodes used can be checked via `ls /sys/class/drm`  

  - References to kernel changes:  
    - [Updates to number of node](https://cgit.freedesktop.org/drm/libdrm/commit/?id=7130cb163eb860d4a965c6708b64fe87cee881d6)  
    - [Identification of node type](https://cgit.freedesktop.org/drm/libdrm/commit/?id=3bc3cca230c5a064b2f554f26fdec27db0f5ead8)  

    Options are as follows:
    1) ***Workaround - removing other devices using DRM nodes***  

        Recommended steps for removing unnecessary drivers:  
        a. Unload amdgpu - `sudo rmmod amdgpu`  
        b. Remove unnecessary driver(s) - ex. `sudo rmmod ast`  
        c. Reload amgpu - `sudo modprobe amdgpu`  
        d. Confirm `amd-smi list` reports all nodes (this can vary per MI ASIC)

    2) ***Update your OS' kernel***  
    3) ***Building and installing your own kernel***  

- **ModuleNotFoundError: No module named 'more_itertools' issue on Azure Linux 3 and Mariner2.0**  
 With the reintroduction of python3-wheel and python3-setuptools dependencies in the CMake of amdsmi, Azure Linux 3 and Mariner2.0 now require more_itertools to build the Python library successfully. 
  - **Workaround:** Execute `sudo python3 -m pip install more_itertools` before installation to resolve this issue.
## amd_smi_lib for ROCm 6.3.1

### Added

### Changed

- **Changed `amd-smi monitor`: No longer display `ENC_CLOCK`/`DEC_CLOCK` but `VCLOCK` and `DCLOCK`**.  
  Due to fix mentioned in `Resolved Issues`, this change was needed.  
  Reason: Navi products use vclk and dclk for both encode and decode. On MI products, only decode is supported.  
  Before:
  ```shell
  $ amd-smi monitor -n -d
  GPU  ENC_UTIL  ENC_CLOCK  DEC_UTIL  DEC_CLOCK
    0     0.0 %     29 MHz       N/A     22 MHz
    1     0.0 %     29 MHz       N/A     22 MHz
    2     0.0 %     29 MHz       N/A     22 MHz
    3     0.0 %     29 MHz       N/A     22 MHz
    4     0.0 %     29 MHz       N/A     22 MHz
    5     0.0 %     29 MHz       N/A     22 MHz
    6     0.0 %     29 MHz       N/A     22 MHz
    7     0.0 %     29 MHz       N/A     22 MHz
  ```
  After:
  ```shell
  $ amd-smi monitor -n -d
  GPU  ENC_UTIL  DEC_UTIL  VCLOCK  DCLOCK
    0       N/A     0.0 %  29 MHz  22 MHz
    1       N/A     0.0 %  29 MHz  22 MHz
    2       N/A     0.0 %  29 MHz  22 MHz
    3       N/A     0.0 %  29 MHz  22 MHz
    4       N/A     0.0 %  29 MHz  22 MHz
    5       N/A     0.0 %  29 MHz  22 MHz
    6       N/A     0.0 %  29 MHz  22 MHz
    7       N/A     0.0 %  29 MHz  22 MHz
  ```

### Removed

### Optimized

### Resolved issues

- **Fixed `amd-smi monitor`'s encode/decode: `ENC_UTIL`, `DEC_UTIL`, and now associate `VCLOCK`/`DCLOCK` with both**.  
  Navi products use vclk and dclk for both encode and decode. On MI products, only decode is supported. 

  Navi products cannot support displaying ENC_UTIL % at this time.  

  Before:
  ```shell
  $ amd-smi monitor -n -d
  GPU  ENC_UTIL  ENC_CLOCK  DEC_UTIL  DEC_CLOCK
    0     0.0 %     29 MHz       N/A     22 MHz
    1     0.0 %     29 MHz       N/A     22 MHz
    2     0.0 %     29 MHz       N/A     22 MHz
    3     0.0 %     29 MHz       N/A     22 MHz
    4     0.0 %     29 MHz       N/A     22 MHz
    5     0.0 %     29 MHz       N/A     22 MHz
    6     0.0 %     29 MHz       N/A     22 MHz
    7     0.0 %     29 MHz       N/A     22 MHz
  ```
  After:
  ```shell
  $ amd-smi monitor -n -d
  GPU  ENC_UTIL  DEC_UTIL  VCLOCK  DCLOCK
    0       N/A     0.0 %  29 MHz  22 MHz
    1       N/A     0.0 %  29 MHz  22 MHz
    2       N/A     0.0 %  29 MHz  22 MHz
    3       N/A     0.0 %  29 MHz  22 MHz
    4       N/A     0.0 %  29 MHz  22 MHz
    5       N/A     0.0 %  29 MHz  22 MHz
    6       N/A     0.0 %  29 MHz  22 MHz
    7       N/A     0.0 %  29 MHz  22 MHz
  ```

### Upcoming changes

### Known issues

## amd_smi_lib for ROCm 6.3.0

### Added

- **Added support for `amd-smi metric --ecc` & `amd-smi metric --ecc-blocks` on Guest VMs**.  
Guest VMs now support getting current ECC counts and ras information from the Host cards.

- **Added support for GPU metrics 1.6 to `amdsmi_get_gpu_metrics_info()`**.  
Updated `amdsmi_get_gpu_metrics_info()` and structure `amdsmi_gpu_metrics_t` to include new fields for PVIOL / TVIOL,  XCP (Graphics Compute Partitions) stats, and pcie_lc_perf_other_end_recovery:  
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

- **Added new violation status outputs and APIs: `amdsmi_status_t amdsmi_get_violation_status()`, `amd-smi metric  --throttle`, and `amd-smi monitor --violation`**.  
  ***Only available for MI300+ ASICs.***  
  Users can now retrieve violation status' through either our Python or C++ APIs. Additionally, we have
  added capability to view these outputs conviently through `amd-smi metric --throttle` and `amd-smi monitor --violation`.  
  Example outputs are listed below (below is for reference, output is subject to change):

```shell
$ amd-smi metric --throttle
GPU: 0
    THROTTLE:
        ACCUMULATION_COUNTER: 3808991
        PROCHOT_ACCUMULATED: 0
        PPT_ACCUMULATED: 585613
        SOCKET_THERMAL_ACCUMULATED: 2190
        VR_THERMAL_ACCUMULATED: 0
        HBM_THERMAL_ACCUMULATED: 0
        PROCHOT_VIOLATION_STATUS: NOT ACTIVE
        PPT_VIOLATION_STATUS: NOT ACTIVE
        SOCKET_THERMAL_VIOLATION_STATUS: NOT ACTIVE
        VR_THERMAL_VIOLATION_STATUS: NOT ACTIVE
        HBM_THERMAL_VIOLATION_STATUS: NOT ACTIVE
        PROCHOT_VIOLATION_ACTIVITY: 0 %
        PPT_VIOLATION_ACTIVITY: 0 %
        SOCKET_THERMAL_VIOLATION_ACTIVITY: 0 %
        VR_THERMAL_VIOLATION_ACTIVITY: 0 %
        HBM_THERMAL_VIOLATION_ACTIVITY: 0 %



GPU: 1
    THROTTLE:
        ACCUMULATION_COUNTER: 3806335
        PROCHOT_ACCUMULATED: 0
        PPT_ACCUMULATED: 586332
        SOCKET_THERMAL_ACCUMULATED: 18010
        VR_THERMAL_ACCUMULATED: 0
        HBM_THERMAL_ACCUMULATED: 0
        PROCHOT_VIOLATION_STATUS: NOT ACTIVE
        PPT_VIOLATION_STATUS: NOT ACTIVE
        SOCKET_THERMAL_VIOLATION_STATUS: NOT ACTIVE
        VR_THERMAL_VIOLATION_STATUS: NOT ACTIVE
        HBM_THERMAL_VIOLATION_STATUS: NOT ACTIVE
        PROCHOT_VIOLATION_ACTIVITY: 0 %
        PPT_VIOLATION_ACTIVITY: 0 %
        SOCKET_THERMAL_VIOLATION_ACTIVITY: 0 %
        VR_THERMAL_VIOLATION_ACTIVITY: 0 %
        HBM_THERMAL_VIOLATION_ACTIVITY: 0 %

...
```

```shell
$ amd-smi monitor --violation
GPU     PVIOL     TVIOL  PHOT_TVIOL  VR_TVIOL  HBM_TVIOL
  0       0 %       0 %         0 %       0 %        0 %
  1       0 %       0 %         0 %       0 %        0 %
  2       0 %       0 %         0 %       0 %        0 %
  3       0 %       0 %         0 %       0 %        0 %
  4       0 %       0 %         0 %       0 %        0 %
  5       0 %       0 %         0 %       0 %        0 %
  6       0 %       0 %         0 %       0 %        0 %
  7       0 %       0 %         0 %       0 %        0 %
  8       0 %       0 %         0 %       0 %        0 %
  9       0 %       0 %         0 %       0 %        0 %
 10       0 %       0 %         0 %       0 %        0 %
 11       0 %       0 %         0 %       0 %        0 %
 12       0 %       0 %         0 %       0 %        0 %
 13       0 %       0 %         0 %       0 %        0 %
 14       0 %       0 %         0 %       0 %        0 %
 15       0 %       0 %         0 %       0 %        0 %
...
```

- **Added ability to view XCP (Graphics Compute Partition) activity within `amd-smi metric --usage`**.  
  ***Partition specific features are only available on MI300+ ASICs***  
  Users can now retrieve graphic utilization statistic on a per-XCP (per-partition) basis. Here all  XCP activities will be listed,
  but the current XCP is the partition id listed under both `amd-smi list` and `amd-smi static --partition`.  
  Example outputs are listed below (below is for reference, output is subject to change):

```shell
$ amd-smi metric --usage
GPU: 0
    USAGE:
        GFX_ACTIVITY: 0 %
        UMC_ACTIVITY: 0 %
        MM_ACTIVITY: N/A
        VCN_ACTIVITY: [0 %, N/A, N/A, N/A]
        JPEG_ACTIVITY: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A,
            N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
            N/A, N/A, N/A]
        GFX_BUSY_INST:
            XCP_0: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_1: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_2: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_3: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_4: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_5: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_6: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_7: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
        JPEG_BUSY:
            XCP_0: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_1: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_2: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_3: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_4: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_5: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_6: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_7: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
        VCN_BUSY:
            XCP_0: [0 %, N/A, N/A, N/A]
            XCP_1: [0 %, N/A, N/A, N/A]
            XCP_2: [0 %, N/A, N/A, N/A]
            XCP_3: [0 %, N/A, N/A, N/A]
            XCP_4: [0 %, N/A, N/A, N/A]
            XCP_5: [0 %, N/A, N/A, N/A]
            XCP_6: [0 %, N/A, N/A, N/A]
            XCP_7: [0 %, N/A, N/A, N/A]
        GFX_BUSY_ACC:
            XCP_0: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_1: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_2: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_3: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_4: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_5: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_6: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_7: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]

GPU: 1
    USAGE:
        GFX_ACTIVITY: 0 %
        UMC_ACTIVITY: 0 %
        MM_ACTIVITY: N/A
        VCN_ACTIVITY: [0 %, N/A, N/A, N/A]
        JPEG_ACTIVITY: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A,
            N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
            N/A, N/A, N/A]
        GFX_BUSY_INST:
            XCP_0: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_1: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_2: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_3: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_4: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_5: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_6: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_7: [0 %, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
        JPEG_BUSY:
            XCP_0: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_1: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_2: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_3: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_4: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_5: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_6: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
            XCP_7: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A,
                N/A, N/A, N/A]
        VCN_BUSY:
            XCP_0: [0 %, N/A, N/A, N/A]
            XCP_1: [0 %, N/A, N/A, N/A]
            XCP_2: [0 %, N/A, N/A, N/A]
            XCP_3: [0 %, N/A, N/A, N/A]
            XCP_4: [0 %, N/A, N/A, N/A]
            XCP_5: [0 %, N/A, N/A, N/A]
            XCP_6: [0 %, N/A, N/A, N/A]
            XCP_7: [0 %, N/A, N/A, N/A]
        GFX_BUSY_ACC:
            XCP_0: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_1: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_2: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_3: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_4: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_5: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_6: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]
            XCP_7: [N/A, N/A, N/A, N/A, N/A, N/A, N/A, N/A]

...
```

- **Added `LC_PERF_OTHER_END_RECOVERY` CLI output to `amd-smi metric --pcie` and updated `amdsmi_get_pcie_info()` to include this value**.  
  ***Feature is only available on MI300+ ASICs***  
  Users can now retrieve both through `amdsmi_get_pcie_info()` which has an updated structure:

```C
typedef struct {
  ...
  struct pcie_metric_ {
    uint16_t pcie_width;                  //!< current PCIe width
    uint32_t pcie_speed;                  //!< current PCIe speed in MT/s
    uint32_t pcie_bandwidth;              //!< current instantaneous PCIe bandwidth in Mb/s
    uint64_t pcie_replay_count;           //!< total number of the replays issued on the PCIe link
    uint64_t pcie_l0_to_recovery_count;   //!< total number of times the PCIe link transitioned from L0 to the recovery state
    uint64_t pcie_replay_roll_over_count; //!< total number of replay rollovers issued on the PCIe link
    uint64_t pcie_nak_sent_count;         //!< total number of NAKs issued on the PCIe link by the device
    uint64_t pcie_nak_received_count;     //!< total number of NAKs issued on the PCIe link by the receiver
    uint32_t pcie_lc_perf_other_end_recovery_count;  //!< PCIe other end recovery counter
    uint64_t reserved[12];
  } pcie_metric;
  uint64_t reserved[32];
} amdsmi_pcie_info_t;
```

  - Example outputs are listed below (below is for reference, output is subject to change):

```shell
$ amd-smi metric --pcie
GPU: 0
    PCIE:
        WIDTH: 16
        SPEED: 32 GT/s
        BANDWIDTH: 18 Mb/s
        REPLAY_COUNT: 0
        L0_TO_RECOVERY_COUNT: 0
        REPLAY_ROLL_OVER_COUNT: 0
        NAK_SENT_COUNT: 0
        NAK_RECEIVED_COUNT: 0
        CURRENT_BANDWIDTH_SENT: N/A
        CURRENT_BANDWIDTH_RECEIVED: N/A
        MAX_PACKET_SIZE: N/A
        LC_PERF_OTHER_END_RECOVERY: 0

GPU: 1
    PCIE:
        WIDTH: 16
        SPEED: 32 GT/s
        BANDWIDTH: 18 Mb/s
        REPLAY_COUNT: 0
        L0_TO_RECOVERY_COUNT: 0
        REPLAY_ROLL_OVER_COUNT: 0
        NAK_SENT_COUNT: 0
        NAK_RECEIVED_COUNT: 0
        CURRENT_BANDWIDTH_SENT: N/A
        CURRENT_BANDWIDTH_RECEIVED: N/A
        MAX_PACKET_SIZE: N/A
        LC_PERF_OTHER_END_RECOVERY: 0
...
```

- **Added retrieving a set of GPUs that are nearest to a given device at a specific link type level**.  
  - Added `amdsmi_get_link_topology_nearest()` function to amd-smi C and Python Libraries.

- **Added more supported utilization count types to `amdsmi_get_utilization_count()`**.  

- **Added `amd-smi set -L/--clk-limit ...` command**.  
  Equivalent to rocm-smi's '--extremum' command which sets sclk's or mclk's soft minimum or soft maximum clock frequency.

- **Added unittest functionality to test amdsmi API calls in Python**.  

- **Changed the `power` parameter in `amdsmi_get_energy_count()` to `energy_accumulator`**.  
  - Changes propagate forwards into the python interface as well, however we are maintaing backwards compatibility and keeping the `power` field in the python API until ROCm 6.4.

- **Added GPU memory overdrive percentage to `amd-smi metric -o`**.  
  - Added `amdsmi_get_gpu_mem_overdrive_level()` function to amd-smi C and Python Libraries.

- **Added retrieving connection type and P2P capabilities between two GPUs**.  
  - Added `amdsmi_topo_get_p2p_status()` function to amd-smi C and Python Libraries.
  - Added retrieving P2P link capabilities to CLI `amd-smi topology`.

```shell
$ amd-smi topology -h
usage: amd-smi topology [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                        [-g GPU [GPU ...]] [-a] [-w] [-o] [-t] [-b]

If no GPU is specified, returns information for all GPUs on the system.
If no topology argument is provided all topology information will be displayed.

Topology arguments:
  -h, --help               show this help message and exit
  -g, --gpu GPU [GPU ...]  Select a GPU ID, BDF, or UUID from the possible choices:
                           ID: 0 | BDF: 0000:0c:00.0 | UUID: <redacted>
                           ID: 1 | BDF: 0000:22:00.0 | UUID: <redacted>
                           ID: 2 | BDF: 0000:38:00.0 | UUID: <redacted>
                           ID: 3 | BDF: 0000:5c:00.0 | UUID: <redacted>
                           ID: 4 | BDF: 0000:9f:00.0 | UUID: <redacted>
                           ID: 5 | BDF: 0000:af:00.0 | UUID: <redacted>
                           ID: 6 | BDF: 0000:bf:00.0 | UUID: <redacted>
                           ID: 7 | BDF: 0000:df:00.0 | UUID: <redacted>
                             all | Selects all devices

  -a, --access             Displays link accessibility between GPUs
  -w, --weight             Displays relative weight between GPUs
  -o, --hops               Displays the number of hops between GPUs
  -t, --link-type          Displays the link type between GPUs
  -b, --numa-bw            Display max and min bandwidth between nodes
  -c, --coherent           Display cache coherant (or non-coherant) link capability between nodes
  -n, --atomics            Display 32 and 64-bit atomic io link capability between nodes
  -d, --dma                Display P2P direct memory access (DMA) link capability between nodes
  -z, --bi-dir             Display P2P bi-directional link capability between nodes

Command Modifiers:
  --json                   Displays output in JSON format (human readable by default).
  --csv                    Displays output in CSV format (human readable by default).
  --file FILE              Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL         Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```shell
$ amd-smi topology -cndz
CACHE COHERANCY TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 SELF         C            NC           NC           C            C            C            NC
0000:22:00.0 C            SELF         NC           C            C            C            NC           C
0000:38:00.0 NC           NC           SELF         C            C            NC           C            NC
0000:5c:00.0 NC           C            C            SELF         NC           C            NC           NC
0000:9f:00.0 C            C            C            NC           SELF         NC           NC           C
0000:af:00.0 C            C            NC           C            NC           SELF         C            C
0000:bf:00.0 C            NC           C            NC           NC           C            SELF         NC
0000:df:00.0 NC           C            NC           NC           C            C            NC           SELF

ATOMICS TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 SELF         64,32        64,32        64           32           32           N/A          64,32
0000:22:00.0 64,32        SELF         64           32           32           N/A          64,32        64,32
0000:38:00.0 64,32        64           SELF         32           N/A          64,32        64,32        64,32
0000:5c:00.0 64           32           32           SELF         64,32        64,32        64,32        32
0000:9f:00.0 32           32           N/A          64,32        SELF         64,32        32           32
0000:af:00.0 32           N/A          64,32        64,32        64,32        SELF         32           N/A
0000:bf:00.0 N/A          64,32        64,32        64,32        32           32           SELF         64,32
0000:df:00.0 64,32        64,32        64,32        32           32           N/A          64,32        SELF

DMA TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 SELF         T            T            F            F            T            F            T
0000:22:00.0 T            SELF         F            F            T            F            T            T
0000:38:00.0 T            F            SELF         T            F            T            T            T
0000:5c:00.0 F            F            T            SELF         T            T            T            F
0000:9f:00.0 F            T            F            T            SELF         T            F            F
0000:af:00.0 T            F            T            T            T            SELF         F            T
0000:bf:00.0 F            T            T            T            F            F            SELF         F
0000:df:00.0 T            T            T            F            F            T            F            SELF

BI-DIRECTIONAL TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 SELF         T            T            F            F            T            F            T
0000:22:00.0 T            SELF         F            F            T            F            T            T
0000:38:00.0 T            F            SELF         T            F            T            T            T
0000:5c:00.0 F            F            T            SELF         T            T            T            F
0000:9f:00.0 F            T            F            T            SELF         T            F            F
0000:af:00.0 T            F            T            T            T            SELF         F            T
0000:bf:00.0 F            T            T            T            F            F            SELF         F
0000:df:00.0 T            T            T            F            F            T            F            SELF

Legend:
 SELF = Current GPU
 ENABLED / DISABLED = Link is enabled or disabled
 N/A = Not supported
 T/F = True / False
 C/NC = Coherant / Non-Coherant io links
 64,32 = 64 bit and 32 bit atomic support
 <BW from>-<BW to>
```

- **Created new amdsmi_kfd_info_t and added information under `amd-smi list`**.  
  - Due to fixes needed to properly enumerate all logical GPUs in CPX, new device identifiers were added in to a new `amdsmi_kfd_info_t` which gets populated via the API `amdsmi_get_gpu_kfd_info()`.
  - This info has been added to the `amd-smi list`.
  - These new fields are only available for BM/Guest Linux devices at this time.

```C
typedef struct {
  uint64_t kfd_id;  //< 0xFFFFFFFFFFFFFFFF if not supported
  uint32_t node_id;  //< 0xFFFFFFFF if not supported
  uint32_t current_partition_id;  //< 0xFFFFFFFF if not supported
  uint32_t reserved[12];
} amdsmi_kfd_info_t;
```

```shell
$ amd-smi list
GPU: 0
    BDF: 0000:23:00.0
    UUID: <redacted>
    KFD_ID: 45412
    NODE_ID: 1
    PARTITION_ID: 0

GPU: 1
    BDF: 0000:26:00.0
    UUID: <redacted>
    KFD_ID: 59881
    NODE_ID: 2
    PARTITION_ID: 0
```

- **Added Subsystem Device ID to `amd-smi static --asic`**.  
  - No underlying changes to amdsmi_get_gpu_asic_info

```shell
$ amd-smi static --asic
GPU: 0
    ASIC:
        MARKET_NAME: MI308X
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a2
        SUBSYSTEM_ID: 0x74a2
        REV_ID: 0x00
        ASIC_SERIAL: <redacted>
        OAM_ID: 5
        NUM_COMPUTE_UNITS: 20
        TARGET_GRAPHICS_VERSION: gfx942
```

- **Added Target_Graphics_Version to `amd-smi static --asic` and `amdsmi_get_gpu_asic_info()`**.  

```C
typedef struct {
  char  market_name[AMDSMI_256_LENGTH];
  uint32_t vendor_id;   //< Use 32 bit to be compatible with other platform.
  char vendor_name[AMDSMI_MAX_STRING_LENGTH];
  uint32_t subvendor_id;   //< The subsystem vendor id
  uint64_t device_id;   //< The device id of a GPU
  uint32_t rev_id;
  char asic_serial[AMDSMI_NORMAL_STRING_LENGTH];
  uint32_t oam_id;   //< 0xFFFF if not supported
  uint32_t num_of_compute_units;   //< 0xFFFFFFFF if not supported
  uint64_t target_graphics_version;  //< 0xFFFFFFFFFFFFFFFF if not supported
  uint32_t reserved[15];
} amdsmi_asic_info_t;
```

```shell
$ amd-smi static --asic
GPU: 0
    ASIC:
        MARKET_NAME: MI308X
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a2
        SUBSYSTEM_ID: 0x74a2
        REV_ID: 0x00
        ASIC_SERIAL: <redacted>
        OAM_ID: 5
        NUM_COMPUTE_UNITS: 20
        TARGET_GRAPHICS_VERSION: gfx942
```

### Changed

- **Improvement: Users now have the ability to set and reset without providing `-g all` using AMD SMI CLI**.  
Users can now provide set and reset without `-g all`. Previously, users were required to provide:   
`sudo amd-smi set -g all <set arguments>` or `sudo amd-smi reset -g all <set arguments>`  
This update allows users to set or reset without providing `-g all` arguments. Allowing commands:  
`sudo amd-smi set <set arguments>` or `sudo amd-smi reset <set arguments>`  
This action will default to try to set/reset for all AMD GPUs on the user's system.

- **Improvement: `amd-smi set --memory-partition` now includes a warning banner and progress bar**.  
For devices which support dynamically changing memory partitions, we now provide a warning for users. We provide this warning to provide users knowledge that this action requires users to quit any gpu workloads. Also to let them know this process will trigger an AMD GPU driver reload. Since this process takes time to complete, a progress bar has been provided until actions can verified as a successful change. Otherwise, AMD SMI will report any errors to users and what actions can be taken. See example below:  
```shell
$ sudo amd-smi set -M NPS1

          ****** WARNING ******

          Setting Dynamic Memory (NPS) partition modes require users to quit all GPU workloads.
          AMD SMI will then attempt to change memory (NPS) partition mode.
          Upon a successful set, AMD SMI will then initiate an action to restart amdgpu driver.
          This action will change all GPU's in the hive to the requested memory (NPS) partition mode.

          Please use this utility with caution.

Do you accept these terms? [Y/N] y

Updating memory partition for gpu 0: [████████████████████████████████████████] 40/40 secs remain

GPU: 0
    MEMORYPARTITION: Successfully set memory partition to NPS1

GPU: 1
    MEMORYPARTITION: Successfully set memory partition to NPS1

GPU: 2
    MEMORYPARTITION: Successfully set memory partition to NPS1
...
```  

- **Updated `amdsmi_get_gpu_accelerator_partition_profile` to provide driver memory partition capablities**.  
Driver now has the ability to report what the user can set memory partition modes to. User can now see available
memory partition modes upon an invalid argument return from memory partition mode set (`amdsmi_set_gpu_memory_partition`).
This change also updates `amd-smi partition`, `amd-smi partition --memory`, and `amd-smi partition --accelerator` (*see note below)   
***Note: *Subject to change for ROCm 6.4***

- **Updated `amdsmi_set_gpu_memory_partition` to not return until a successful restart of AMD GPU Driver.**  
This change keeps checking for ~ up to 40 seconds for a successful restart of the AMD GPU driver. Additionally, the API call continues to check if memory partition (NPS) SYSFS files are successfully updated to reflect the user's requested memory partition (NPS) mode change. Otherwise, reports an error back to the user. Due to these changes, we have updated AMD SMI's CLI to reflect the maximum wait of 40 seconds, while a memory partition change is in progress.

- **All APIs now have the ability to catch driver reporting invalid arguments.**  
Now AMD SMI APIs can show AMDSMI_STATUS_INVAL when driver returns EINVAL.  
For example, if user tries to set to NPS8, but the memory partition mode is not an available mode to set to. Commonly referred to as `CAPS` (see `amd-smi partition --memory`), provided by `amdsmi_get_gpu_accelerator_partition_profile`(*see note below).  
***Note: *Subject to change for ROCm 6.4***

- **Updated BDF commands to look use KFD SYSFS for BDF: `amdsmi_get_gpu_device_bdf()`**.  
This aligns BDF output with ROCm SMI.
See below for overview as seen from `rsmi_dev_pci_id_get()` now provides partition ID. See API for better detail. Previously these bits were reserved bits (right before domain) and partition id was within function.
  - bits [63:32] = domain
  - bits [31:28] = partition id
  - bits [27:16] = reserved
  - bits [15: 0] = pci bus/device/function

- **Moved python tests directory path install location**.  
  - `/opt/<rocm-path>/share/amd_smi/pytest/..` to `/opt/<rocm-path>/share/amd_smi/tests/python_unittest/..`
  - On amd-smi-lib-tests uninstall, the amd_smi tests folder is removed
  - Removed pytest dependency, our python testing now only depends on the unittest framework.

- **Updated Partition APIs and struct information and added and partition_id to `amd-smi static --partition`**.
  - As part of an overhaul to partition information, some partition information will be made available in the `amdsmi_accelerator_partition_profile_t`.
  - This struct will be filled out by a new API, `amdsmi_get_gpu_accelerator_partition_profile()`.
  - Future data from these APIs wil will eventually get added to `amd-smi partition`.

```C
#define AMDSMI_MAX_ACCELERATOR_PROFILE    32
#define AMDSMI_MAX_CP_PROFILE_RESOURCES   32
#define AMDSMI_MAX_ACCELERATOR_PARTITIONS 8

/**
 * @brief Accelerator Partition. This enum is used to identify
 * various accelerator partitioning settings.
 */
typedef enum {
  AMDSMI_ACCELERATOR_PARTITION_INVALID = 0,
  AMDSMI_ACCELERATOR_PARTITION_SPX,        //!< Single GPU mode (SPX)- All XCCs work
                                       //!< together with shared memory
  AMDSMI_ACCELERATOR_PARTITION_DPX,        //!< Dual GPU mode (DPX)- Half XCCs work
                                       //!< together with shared memory
  AMDSMI_ACCELERATOR_PARTITION_TPX,        //!< Triple GPU mode (TPX)- One-third XCCs
                                       //!< work together with shared memory
  AMDSMI_ACCELERATOR_PARTITION_QPX,        //!< Quad GPU mode (QPX)- Quarter XCCs
                                       //!< work together with shared memory
  AMDSMI_ACCELERATOR_PARTITION_CPX,        //!< Core mode (CPX)- Per-chip XCC with
                                       //!< shared memory
} amdsmi_accelerator_partition_type_t;

/**
 * @brief Possible Memory Partition Modes.
 * This union is used to identify various memory partitioning settings.
 */
typedef union {
    struct {
        uint32_t nps1_cap :1;  // bool 1 = true; 0 = false; Max uint32 means unsupported
        uint32_t nps2_cap :1;  // bool 1 = true; 0 = false; Max uint32 means unsupported
        uint32_t nps4_cap :1;  // bool 1 = true; 0 = false; Max uint32 means unsupported
        uint32_t nps8_cap :1;  // bool 1 = true; 0 = false; Max uint32 means unsupported
        uint32_t reserved :28;
    } amdsmi_nps_flags_t;

    uint32_t nps_cap_mask;
} amdsmi_nps_caps_t;

typedef struct {
  amdsmi_accelerator_partition_type_t  profile_type;   // SPX, DPX, QPX, CPX and so on
  uint32_t num_partitions;  // On MI300X, SPX: 1, DPX: 2, QPX: 4, CPX: 8, length of resources array
  uint32_t profile_index;
  amdsmi_nps_caps_t memory_caps;             // Possible memory partition capabilities
  uint32_t num_resources;                    // length of index_of_resources_profile
  uint32_t resources[AMDSMI_MAX_ACCELERATOR_PARTITIONS][AMDSMI_MAX_CP_PROFILE_RESOURCES];
  uint64_t reserved[6];
} amdsmi_accelerator_partition_profile_t;
```

```shell
$ amd-smi static --partition
GPU: 0
    PARTITION:
        COMPUTE_PARTITION: CPX
        MEMORY_PARTITION: NPS4
        PARTITION_ID: 0
```

### Removed

- **Removed `amd-smi reset --compute-partition` and `... --memory-partition` and associated APIs**.  
  - This change is part of the partition redesign. Reset functionality will be reintroduced in a later update.
  - associated APIs include `amdsmi_reset_gpu_compute_partition()` and `amdsmi_reset_gpu_memory_partition()`

- **Removed usage of _validate_positive in Parser and replaced with _positive_int and _not_negative_int as appropriate**.  
  - This will allow 0 to be a valid input for several options in setting CPUs where appropriate (for example, as a mode or NBIOID)

### Optimized

- **Adjusted ordering of gpu_metrics calls to ensure that pcie_bw values remain stable in `amd-smi metric` & `amd-smi monitor`**.  
  - With this change additional padding was added to PCIE_BW `amd-smi monitor --pcie`

### Resolved issues

- **Fixed `amdsmi_get_gpu_asic_info`'s `target_graphics_version` and `amd-smi --asic` not displaying properly for MI2x or Navi 3x ASICs**.  

- **Fixed `amd-smi reset` commands showing an AttributeError**.  

- **Improved Offline install process & lowered dependency for PyYAML**.  

- **Fixed CPX not showing total number of logical GPUs**.  
  - Updates were made to `amdsmi_init()` and `amdsmi_get_gpu_bdf_id(..)`. In order to display all logical devices, we needed a way to provide order to GPU's enumerated. This was done by adding a partition_id within the BDF optional pci_id bits.
  - Due to driver changes in KFD, some devices may report bits [31:28] or [2:0]. With the newly added `amdsmi_get_gpu_bdf_id(..)`, we provided this fallback to properly retreive partition ID. We
plan to eventually remove partition ID from the function portion of the BDF (Bus Device Function). See below for PCI ID description.

    - bits [63:32] = domain
    - bits [31:28] or bits [2:0] = partition id
    - bits [27:16] = reserved
    - bits [15:8]  = Bus
    - bits [7:3] = Device
    - bits [2:0] = Function (partition id maybe in bits [2:0]) <-- Fallback for non SPX modes

  - Previously in non-SPX modes (ex. CPX/TPX/DPX/etc) some MI3x ASICs would not report all logical GPU devices within AMD SMI.

```shell
$ amd-smi monitor -p -t -v
GPU  POWER  GPU_TEMP  MEM_TEMP  VRAM_USED  VRAM_TOTAL
  0  248 W     55 °C     48 °C     283 MB   196300 MB
  1  247 W     55 °C     48 °C     283 MB   196300 MB
  2  247 W     55 °C     48 °C     283 MB   196300 MB
  3  247 W     55 °C     48 °C     283 MB   196300 MB
  4  221 W     50 °C     42 °C     283 MB   196300 MB
  5  221 W     50 °C     42 °C     283 MB   196300 MB
  6  222 W     50 °C     42 °C     283 MB   196300 MB
  7  221 W     50 °C     42 °C     283 MB   196300 MB
  8  239 W     53 °C     46 °C     283 MB   196300 MB
  9  239 W     53 °C     46 °C     283 MB   196300 MB
 10  239 W     53 °C     46 °C     283 MB   196300 MB
 11  239 W     53 °C     46 °C     283 MB   196300 MB
 12  219 W     51 °C     48 °C     283 MB   196300 MB
 13  219 W     51 °C     48 °C     283 MB   196300 MB
 14  219 W     51 °C     48 °C     283 MB   196300 MB
 15  219 W     51 °C     48 °C     283 MB   196300 MB
 16  222 W     51 °C     47 °C     283 MB   196300 MB
 17  222 W     51 °C     47 °C     283 MB   196300 MB
 18  222 W     51 °C     47 °C     283 MB   196300 MB
 19  222 W     51 °C     48 °C     283 MB   196300 MB
 20  241 W     55 °C     48 °C     283 MB   196300 MB
 21  241 W     55 °C     48 °C     283 MB   196300 MB
 22  241 W     55 °C     48 °C     283 MB   196300 MB
 23  240 W     55 °C     48 °C     283 MB   196300 MB
 24  211 W     51 °C     45 °C     283 MB   196300 MB
 25  211 W     51 °C     45 °C     283 MB   196300 MB
 26  211 W     51 °C     45 °C     283 MB   196300 MB
 27  211 W     51 °C     45 °C     283 MB   196300 MB
 28  227 W     51 °C     49 °C     283 MB   196300 MB
 29  227 W     51 °C     49 °C     283 MB   196300 MB
 30  227 W     51 °C     49 °C     283 MB   196300 MB
 31  227 W     51 °C     49 °C     283 MB   196300 MB
```

- **Fixed incorrect implementation of the Python API `amdsmi_get_gpu_metrics_header_info()`**.  

- **`amdsmitst` TestGpuMetricsRead now prints metric in correct units**.  

### Upcoming changes

- **Python API for `amdsmi_get_energy_count()` will deprecate the `power` field in ROCm 6.4 and use `energy_accumulator` field instead**.  

- **New memory and compute partition APIs incoming for ROCm 6.4**.  
  - These APIs will be updated to fully populate the CLI and allowing compute (accelerator) partitions to be set by profile ID.
  - One API will be provided, to reset both memory and compute (accelerator).
    - There are dependencies regarding available compute partitions when in other memory modes.
    - Driver will be providing these default modes
    - Memory partition resets (for BM) require driver reloads - this will allow us to notify users before taking this action, then change to the default compute partition modes.
  - The following APIs will remain:

```C
amdsmi_status_t
amdsmi_set_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_compute_partition_type_t compute_partition);
amdsmi_status_t
amdsmi_get_gpu_compute_partition(amdsmi_processor_handle processor_handle,
                                  char *compute_partition, uint32_t len);
amdsmi_status_t
amdsmi_get_gpu_memory_partition(amdsmi_processor_handle processor_handle,

                                  char *memory_partition, uint32_t len);
amdsmi_status_t
amdsmi_set_gpu_memory_partition(amdsmi_processor_handle processor_handle,
                                  amdsmi_memory_partition_type_t memory_partition);
```

- **amd-smi set --compute-partition "SPX/DPX/CPX..." will no longer be supported for ROCm 6.4**.  
  - This is due to aligning with Host setups and providing more robust partition information through the APIs outlined above. Furthermore, new APIs which will be available on both BM/Host can set by profile ID. (functionality coming soon!)

- **Added preliminary `amd-smi partition` command**.  
  - The new partition command can be used to display GPU information, including memory and accelerator partition information.
  - The command will be at full functionality once additional partition information from `amdsmi_get_gpu_accelerator_partition_profile()` has been implemented.

## amd_smi_lib for ROCm 6.2.1

### Added

- **Removed `amd-smi metric --ecc` & `amd-smi metric --ecc-blocks` on Guest VMs**.
Guest VMs do not support getting current ECC counts from the Host cards.

- **Added `amd-smi static --ras`on Guest VMs**.
Guest VMs can view enabled/disabled ras features that are on Host cards.

### Resolved issues

- **Fixed TypeError in `amd-smi process -G`**.  

- **Updated CLI error strings to handle empty and invalid GPU/CPU inputs**.  

- **Fixed Guest VM showing passthrough options**.

- **Fixed firmware formatting where leading 0s were missing**.

## amd_smi_lib for ROCm 6.2.0

### Added

- **`amd-smi dmon` is now available as an alias to `amd-smi monitor`**.  

- **Added optional process table under `amd-smi monitor -q`**.  
The monitor subcommand within the CLI Tool now has the `-q` option to enable an optional process table underneath the original monitored output.

```shell
$ amd-smi monitor -q
GPU  POWER  GPU_TEMP  MEM_TEMP  GFX_UTIL  GFX_CLOCK  MEM_UTIL  MEM_CLOCK  ENC_UTIL  ENC_CLOCK  DEC_UTIL  DEC_CLOCK  SINGLE_ECC  DOUBLE_ECC  PCIE_REPLAY  VRAM_USED  VRAM_TOTAL   PCIE_BW
  0  199 W    103 °C     84 °C      99 %   1920 MHz      31 %   1000 MHz       N/A      0 MHz       N/A      0 MHz           0           0            0    1235 MB    16335 MB  N/A Mb/s

PROCESS INFO:
GPU                  NAME      PID  GTT_MEM  CPU_MEM  VRAM_MEM  MEM_USAGE     GFX     ENC
  0                   rvs  1564865    0.0 B    0.0 B    1.1 GB      0.0 B    0 ns    0 ns
```

- **Added Handling to detect VMs with passthrough configurations in CLI Tool**.  
CLI Tool had only allowed a restricted set of options for Virtual Machines with passthrough GPUs. Now we offer an expanded set of functions availble to passthrough configured GPUs.

- **Added Process Isolation and Clear SRAM functionality to the CLI Tool for VMs**.  
VMs now have the ability to set the process isolation and clear the sram from the CLI tool. Using the following commands

```shell
amd-smi set --process-isolation <0 or 1>
amd-smi reset --clean_local_data
```

- **Added macros that were in `amdsmi.h` to the amdsmi Python library `amdsmi_interface.py`**.  
Added macros to reference max size limitations for certain amdsmi functions such as max dpm policies and max fanspeed.

- **Added Ring Hang event**.  
Added `AMDSMI_EVT_NOTIF_RING_HANG` to the possible events in the `amdsmi_evt_notification_type_t` enum.

### Optimized

- **Updated CLI error strings to specify invalid device type queried**

```shell
$ amd-smi static --asic --gpu 123123
Can not find a device: GPU '123123' Error code: -3
```

- **Removed elevated permission requirements for `amdsmi_get_gpu_process_list()`**.  
Previously if a processes with elevated permissions was running amd-smi would required sudo to display all output. Now amd-smi will populate all process data and return N/A for elevated process names instead. However if ran with sudo you will be able to see the name like so:

```shell
$ amd-smi process
GPU: 0
    PROCESS_INFO:
        NAME: N/A
        PID: 1693982
        MEMORY_USAGE:
            GTT_MEM: 0.0 B
            CPU_MEM: 0.0 B
            VRAM_MEM: 10.1 GB
        MEM_USAGE: 0.0 B
        USAGE:
            GFX: 0 ns
            ENC: 0 ns
```

```shell
$ sudo amd-smi process
GPU: 0
    PROCESS_INFO:
        NAME: TransferBench
        PID: 1693982
        MEMORY_USAGE:
            GTT_MEM: 0.0 B
            CPU_MEM: 0.0 B
            VRAM_MEM: 10.1 GB
        MEM_USAGE: 0.0 B
        USAGE:
            GFX: 0 ns
            ENC: 0 ns
```

- **Updated naming for `amdsmi_set_gpu_clear_sram_data()` to `amdsmi_clean_gpu_local_data()`**.  
Changed the naming to be more accurate to what the function was doing. This change also extends to the CLI where we changed the `clear-sram-data` command to `clean_local_data`.

- **Updated `amdsmi_clk_info_t` struct in amdsmi.h and amdsmi_interface.py to align with host/guest**.  
Changed cur_clk to clk, changed sleep_clk to clk_deep_sleep, and added clk_locked value. New struct will be in the following format:

```shell
 typedef struct {
+  uint32_t clk;
   uint32_t min_clk;
   uint32_t max_clk;
+  uint8_t clk_locked;
+  uint8_t clk_deep_sleep;
   uint32_t reserved[4];
 } amdsmi_clk_info_t;
```

- **Multiple structure updates in amdsmi.h and amdsmi_interface.py to align with host/guest**.  
Multiple structures used by APIs were changed for alignment unification:
  - Changed `amdsmi_vram_info_t` `vram_size_mb` field changed to to `vram_size`
  - Updated `amdsmi_vram_type_t` struct updated to include new enums and added `AMDSMI` prefix
  - Updated `amdsmi_status_t` some enums were missing the `AMDSMI_STATUS` prefix
  - Added `AMDSMI_PROCESSOR_TYPE` prefix to `processor_type_t` enums
  - Removed the fields structure definition in favor for an anonymous definition in `amdsmi_bdf_t`

- **Added `AMDSMI` prefix in amdsmi.h and amdsmi_interface.py to align with host/guest**.  
Multiple structures used by APIs were changed for alignment unification. `AMDSMI` prefix was added to the following structures:
  - Added AMDSMI prefix to `amdsmi_container_types_t` enums
  - Added AMDSMI prefix to `amdsmi_clk_type_t` enums
  - Added AMDSMI prefix to `amdsmi_compute_partition_type_t` enums
  - Added AMDSMI prefix to `amdsmi_memory_partition_type_t` enums
  - Added AMDSMI prefix to `amdsmi_clk_type_t` enums
  - Added AMDSMI prefix to `amdsmi_temperature_type_t` enums
  - Added AMDSMI prefix to `amdsmi_fw_block_t` enums

- **Changed dpm_policy references to soc_pstate**.  
The file structure referenced to dpm_policy changed to soc_pstate and we have changed the APIs and CLI tool to be inline with the current structure. `amdsmi_get_dpm_policy()` and `amdsmi_set_dpm_policy()` is no longer valid with the new API being `amdsmi_get_soc_pstate()` and `amdsmi_set_soc_pstate()`. The CLI tool has been changed from `--policy` to `--soc-pstate`

- **Updated `amdsmi_get_gpu_board_info()` product_name to fallback to pciids**.  
Previously on devices without a FRU we would not populate the product name in the `amdsmi_board_info_t` structure, now we will fallback to using the name listed according to the pciids file if available.

- **Updated CLI voltage curve command output**.  
The output for `amd-smi metric --voltage-curve` now splits the frequency and voltage output by curve point or outputs N/A for each curve point if not applicable

```shell
GPU: 0
    VOLTAGE_CURVE:
        POINT_0_FREQUENCY: 872 Mhz
        POINT_0_VOLTAGE: 736 mV
        POINT_1_FREQUENCY: 1354 Mhz
        POINT_1_VOLTAGE: 860 mV
        POINT_2_FREQUENCY: 1837 Mhz
        POINT_2_VOLTAGE: 1186 mV
```

- **Updated `amdsmi_get_gpu_board_info()` now has larger structure sizes for `amdsmi_board_info_t`**.  
Updated sizes that work for retreiving relavant board information across AMD's
ASIC products. This requires users to update any ABIs using this structure.

### Resolved issues

- **Fixed Leftover Mutex deadlock when running multiple instances of the CLI tool**.  
When running `amd-smi reset --gpureset --gpu all` and then running an instance of `amd-smi static` (or any other subcommand that access the GPUs) a mutex would lock and not return requiring either a clear of the mutex in /dev/shm or rebooting the machine.

- **Fixed multiple processes not being registered in `amd-smi process` with json and csv format**.  
Multiple process outputs in the CLI tool were not being registered correctly. The json output did not handle multiple processes and is now in a new valid json format:

```shell
[
    {
        "gpu": 0,
        "process_list": [
            {
                "process_info": {
                    "name": "TransferBench",
                    "pid": 420157,
                    "mem_usage": {
                        "value": 0,
                        "unit": "B"
                    }
                }
            },
            {
                "process_info": {
                    "name": "rvs",
                    "pid": 420315,
                    "mem_usage": {
                        "value": 0,
                        "unit": "B"
                    }
                }
            }
        ]
    }
]
```

- **Removed `throttle-status` from `amd-smi monitor` as it is no longer reliably supported**.  
Throttle status may work for older ASICs, but will be replaced with PVIOL and TVIOL metrics for future ASIC support. It remains a field in the gpu_metrics API and in `amd-smi metric --power`.

- **`amdsmi_get_gpu_board_info()` no longer returns junk char strings**.  
Previously if there was a partial failure to retrieve character strings, we would return
garbage output to users using the API. This fix intends to populate as many values as possible.
Then any failure(s) found along the way, `\0` is provided to `amdsmi_board_info_t`
structures data members which cannot be populated. Ensuring empty char string values.

- **Fixed parsing of `pp_od_clk_voltage` within `amdsmi_get_gpu_od_volt_info`**.  
The parsing of `pp_od_clk_voltage` was not dynamic enough to work with the dropping of voltage curve support on MI series cards. This propagates down to correcting the CLI's output `amd-smi metric --voltage-curve` to N/A if voltage curve is not enabled.

### Known issues

- **`amdsmi_get_gpu_process_isolation` and `amdsmi_clean_gpu_local_data` commands do no currently work and will be supported in a future release**.  

## amd_smi_lib for ROCm 6.1.2

### Added

- **Added process isolation and clean shader APIs and CLI commands**.  
Added APIs CLI and APIs to address LeftoverLocals security issues. Allowing clearing the sram data and setting process isolation on a per GPU basis. New APIs:
  - `amdsmi_get_gpu_process_isolation()`
  - `amdsmi_set_gpu_process_isolation()`
  - `amdsmi_set_gpu_clear_sram_data()`

- **Added `MIN_POWER` to output of `amd-smi static --limit`**.  
This change helps users identify the range to which they can change the power cap of the GPU. The change is added to simplify why a device supports (or does not support) power capping (also known as overdrive). See `amd-smi set -g all --power-cap <value in W>` or `amd-smi reset -g all --power-cap`.

```shell
$ amd-smi static --limit
GPU: 0
    LIMIT:
        MAX_POWER: 203 W
        MIN_POWER: 0 W
        SOCKET_POWER: 203 W
        SLOWDOWN_EDGE_TEMPERATURE: 100 °C
        SLOWDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SLOWDOWN_VRAM_TEMPERATURE: 100 °C
        SHUTDOWN_EDGE_TEMPERATURE: 105 °C
        SHUTDOWN_HOTSPOT_TEMPERATURE: 115 °C
        SHUTDOWN_VRAM_TEMPERATURE: 105 °C

GPU: 1
    LIMIT:
        MAX_POWER: 213 W
        MIN_POWER: 213 W
        SOCKET_POWER: 213 W
        SLOWDOWN_EDGE_TEMPERATURE: 109 °C
        SLOWDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SLOWDOWN_VRAM_TEMPERATURE: 100 °C
        SHUTDOWN_EDGE_TEMPERATURE: 114 °C
        SHUTDOWN_HOTSPOT_TEMPERATURE: 115 °C
        SHUTDOWN_VRAM_TEMPERATURE: 105 °C
```

### Optimized

- **Updated `amd-smi monitor --pcie` output**.  
The source for pcie bandwidth monitor output was a legacy file we no longer support and was causing delays within the monitor command. The output is no longer using TX/RX but instantaneous bandwidth from gpu_metrics instead; updated output:

```shell
$ amd-smi monitor --pcie
GPU   PCIE_BW
  0   26 Mb/s
```

- **`amdsmi_get_power_cap_info` now returns values in uW instead of W**.  
`amdsmi_get_power_cap_info` will return in uW as originally reflected by driver. Previously `amdsmi_get_power_cap_info` returned W values, this conflicts with our sets and modifies values retrieved from driver. We decided to keep the values returned from driver untouched (in original units, uW). Then in CLI we will convert to watts (as previously done - no changes here). Additionally, driver made updates to min power cap displayed for devices when overdrive is disabled which prompted for this change (in this case min_power_cap and max_power_cap are the same).

- **Updated Python Library return types for amdsmi_get_gpu_memory_reserved_pages & amdsmi_get_gpu_bad_page_info**.  
Previously calls were returning "No bad pages found." if no pages were found, now it only returns the list type and can be empty.

- **Updated `amd-smi metric --ecc-blocks` output**.  
The ecc blocks argument was outputing blocks without counters available, updated the filtering show blocks that counters are available for:

``` shell
$ amd-smi metric --ecc-block
GPU: 0
    ECC_BLOCKS:
        UMC:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
        SDMA:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
        GFX:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
        MMHUB:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
        PCIE_BIF:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
        HDP:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
        XGMI_WAFL:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
```

- **Removed `amdsmi_get_gpu_process_info` from Python library**.  
amdsmi_get_gpu_process_info was removed from the C library in an earlier build, but the API was still in the Python interface.

### Resolved issues

- **Fixed `amd-smi metric --power` now provides power output for Navi2x/Navi3x/MI1x**.  
These systems use an older version of gpu_metrics in amdgpu. This fix only updates what CLI outputs.
No change in any of our APIs.

```shell
$ amd-smi metric --power
GPU: 0
    POWER:
        SOCKET_POWER: 11 W
        GFX_VOLTAGE: 768 mV
        SOC_VOLTAGE: 925 mV
        MEM_VOLTAGE: 1250 mV
        POWER_MANAGEMENT: ENABLED
        THROTTLE_STATUS: UNTHROTTLED

GPU: 1
    POWER:
        SOCKET_POWER: 17 W
        GFX_VOLTAGE: 781 mV
        SOC_VOLTAGE: 806 mV
        MEM_VOLTAGE: 1250 mV
        POWER_MANAGEMENT: ENABLED
        THROTTLE_STATUS: UNTHROTTLED
```

- **Fixed `amdsmitstReadWrite.TestPowerCapReadWrite` test for Navi3X, Navi2X, MI100**.  
Updates required `amdsmi_get_power_cap_info` to return in uW as originally reflected by driver. Previously `amdsmi_get_power_cap_info` returned W values, this conflicts with our sets and modifies values retrieved from driver. We decided to keep the values returned from driver untouched (in original units, uW). Then in CLI we will convert to watts (as previously done - no changes here). Additionally, driver made updates to min power cap displayed for devices when overdrive is disabled which prompted for this change (in this case min_power_cap and max_power_cap are the same).

- **Fixed Python interface call amdsmi_get_gpu_memory_reserved_pages & amdsmi_get_gpu_bad_page_info**.  
Previously Python interface calls to populated bad pages resulted in a `ValueError: NULL pointer access`. This fixes the bad-pages subcommand CLI  subcommand as well.

## amd_smi_lib for ROCm 6.1.1

### Changed

- **Updated metrics --clocks**.  
Output for `amd-smi metric --clock` is updated to reflect each engine and bug fixes for the clock lock status and deep sleep status.

``` shell
$ amd-smi metric --clock
GPU: 0
    CLOCK:
        GFX_0:
            CLK: 113 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1800 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: ENABLED
        GFX_1:
            CLK: 113 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1800 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: ENABLED
        GFX_2:
            CLK: 112 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1800 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: ENABLED
        GFX_3:
            CLK: 113 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1800 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: ENABLED
        GFX_4:
            CLK: 113 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1800 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: ENABLED
        GFX_5:
            CLK: 113 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1800 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: ENABLED
        GFX_6:
            CLK: 113 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1800 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: ENABLED
        GFX_7:
            CLK: 113 MHz
            MIN_CLK: 500 MHz
            MAX_CLK: 1800 MHz
            CLK_LOCKED: DISABLED
            DEEP_SLEEP: ENABLED
        MEM_0:
            CLK: 900 MHz
            MIN_CLK: 900 MHz
            MAX_CLK: 1200 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: DISABLED
        VCLK_0:
            CLK: 29 MHz
            MIN_CLK: 914 MHz
            MAX_CLK: 1480 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: ENABLED
        VCLK_1:
            CLK: 29 MHz
            MIN_CLK: 914 MHz
            MAX_CLK: 1480 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: ENABLED
        VCLK_2:
            CLK: 29 MHz
            MIN_CLK: 914 MHz
            MAX_CLK: 1480 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: ENABLED
        VCLK_3:
            CLK: 29 MHz
            MIN_CLK: 914 MHz
            MAX_CLK: 1480 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: ENABLED
        DCLK_0:
            CLK: 22 MHz
            MIN_CLK: 711 MHz
            MAX_CLK: 1233 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: ENABLED
        DCLK_1:
            CLK: 22 MHz
            MIN_CLK: 711 MHz
            MAX_CLK: 1233 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: ENABLED
        DCLK_2:
            CLK: 22 MHz
            MIN_CLK: 711 MHz
            MAX_CLK: 1233 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: ENABLED
        DCLK_3:
            CLK: 22 MHz
            MIN_CLK: 711 MHz
            MAX_CLK: 1233 MHz
            CLK_LOCKED: N/A
            DEEP_SLEEP: ENABLED
```

- **Added deferred ecc counts**.  
Added deferred error correctable counts to `amd-smi metric --ecc --ecc-blocks`

```shell
$ amd-smi metric --ecc --ecc-blocks
GPU: 0
    ECC:
        TOTAL_CORRECTABLE_COUNT: 0
        TOTAL_UNCORRECTABLE_COUNT: 0
        TOTAL_DEFERRED_COUNT: 0
        CACHE_CORRECTABLE_COUNT: 0
        CACHE_UNCORRECTABLE_COUNT: 0
    ECC_BLOCKS:
        UMC:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
        SDMA:
            CORRECTABLE_COUNT: 0
            UNCORRECTABLE_COUNT: 0
            DEFERRED_COUNT: 0
        ...
```

- **Updated `amd-smi topology --json` to align with host/guest**.  
Topology's `--json` output now is changed to align with output host/guest systems. Additionally, users can select/filter specific topology details as desired (refer to `amd-smi topology -h` for full list). See examples shown below.

*Previous format:*

```shell
$ amd-smi topology --json
[
    {
        "gpu": 0,
        "link_accessibility": {
            "gpu_0": "ENABLED",
            "gpu_1": "DISABLED"
        },
        "weight": {
            "gpu_0": 0,
            "gpu_1": 40
        },
        "hops": {
            "gpu_0": 0,
            "gpu_1": 2
        },
        "link_type": {
            "gpu_0": "SELF",
            "gpu_1": "PCIE"
        },
        "numa_bandwidth": {
            "gpu_0": "N/A",
            "gpu_1": "N/A"
        }
    },
    {
        "gpu": 1,
        "link_accessibility": {
            "gpu_0": "DISABLED",
            "gpu_1": "ENABLED"
        },
        "weight": {
            "gpu_0": 40,
            "gpu_1": 0
        },
        "hops": {
            "gpu_0": 2,
            "gpu_1": 0
        },
        "link_type": {
            "gpu_0": "PCIE",
            "gpu_1": "SELF"
        },
        "numa_bandwidth": {
            "gpu_0": "N/A",
            "gpu_1": "N/A"
        }
    }
]
```

*New format:*

```shell
$ amd-smi topology --json
[
    {
        "gpu": 0,
        "bdf": "0000:01:00.0",
        "links": [
            {
                "gpu": 0,
                "bdf": "0000:01:00.0",
                "weight": 0,
                "link_status": "ENABLED",
                "link_type": "SELF",
                "num_hops": 0,
                "bandwidth": "N/A",
            },
            {
                "gpu": 1,
                "bdf": "0001:01:00.0",
                "weight": 15,
                "link_status": "ENABLED",
                "link_type": "XGMI",
                "num_hops": 1,
                "bandwidth": "50000-100000",
            },
        ...
        ]
    },
    ...
]
```

```shell
$ /opt/rocm/bin/amd-smi topology -a -t --json
[
    {
        "gpu": 0,
        "bdf": "0000:08:00.0",
        "links": [
            {
                "gpu": 0,
                "bdf": "0000:08:00.0",
                "link_status": "ENABLED",
                "link_type": "SELF"
            },
            {
                "gpu": 1,
                "bdf": "0000:44:00.0",
                "link_status": "DISABLED",
                "link_type": "PCIE"
            }
        ]
    },
    {
        "gpu": 1,
        "bdf": "0000:44:00.0",
        "links": [
            {
                "gpu": 0,
                "bdf": "0000:08:00.0",
                "link_status": "DISABLED",
                "link_type": "PCIE"
            },
            {
                "gpu": 1,
                "bdf": "0000:44:00.0",
                "link_status": "ENABLED",
                "link_type": "SELF"
            }
        ]
    }
]
```

### Resolved issues

- **Fix for GPU reset error on non-amdgpu cards**.  
Previously our reset could attempting to reset non-amd GPUS- resuting in "Unable to reset non-amd GPU" error. Fix
updates CLI to target only AMD ASICs.

- **Fix for `amd-smi static --pcie` and `amdsmi_get_pcie_info()` Navi32/31 cards**.  
Updated API to include `amdsmi_card_form_factor_t.AMDSMI_CARD_FORM_FACTOR_CEM`. Prevously, this would report "UNKNOWN". This fix
provides the correct board `SLOT_TYPE` associated with these ASICs (and other Navi cards).

- **Fix for `amd-smi process`**.  
Fixed output results when getting processes running on a device.

- **Improved Error handling for `amd-smi process`**.  
Fixed Attribute Error when getting process in csv format

### Known issues

- `amd-smi bad-pages` can results with "ValueError: NULL pointer access" with certain PM FW versions.

## amd_smi_lib for ROCm 6.1.0

### Added

- **Added Monitor Command**.  
Provides users the ability to customize GPU metrics to capture, collect, and observe. Output is provided in a table view. This aligns closer to ROCm SMI `rocm-smi` (no argument), additionally allows uers to customize what data is helpful for their use-case.

```shell
$ amd-smi monitor -h
usage: amd-smi monitor [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                       [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]]
                       [-w INTERVAL] [-W TIME] [-i ITERATIONS] [-p] [-t] [-u] [-m] [-n]
                       [-d] [-s] [-e] [-v] [-r]

Monitor a target device for the specified arguments.
If no arguments are provided, all arguments will be enabled.
Use the watch arguments to run continuously

Monitor Arguments:
  -h, --help                   show this help message and exit
  -g, --gpu GPU [GPU ...]      Select a GPU ID, BDF, or UUID from the possible choices:
                               ID: 0 | BDF: 0000:01:00.0 | UUID: <redacted>
                                 all | Selects all devices
  -U, --cpu CPU [CPU ...]      Select a CPU ID from the possible choices:
                               ID: 0
                                 all | Selects all devices
  -O, --core CORE [CORE ...]   Select a Core ID from the possible choices:
                               ID: 0 - 23
                                 all  | Selects all devices
  -w, --watch INTERVAL         Reprint the command in a loop of INTERVAL seconds
  -W, --watch_time TIME        The total TIME to watch the given command
  -i, --iterations ITERATIONS  Total number of ITERATIONS to loop on the given command
  -p, --power-usage            Monitor power usage in Watts
  -t, --temperature            Monitor temperature in Celsius
  -u, --gfx                    Monitor graphics utilization (%) and clock (MHz)
  -m, --mem                    Monitor memory utilization (%) and clock (MHz)
  -n, --encoder                Monitor encoder utilization (%) and clock (MHz)
  -d, --decoder                Monitor decoder utilization (%) and clock (MHz)
  -s, --throttle-status        Monitor thermal throttle status
  -e, --ecc                    Monitor ECC single bit, ECC double bit, and PCIe replay error counts
  -v, --vram-usage             Monitor memory usage in MB
  -r, --pcie                   Monitor PCIe Tx/Rx in MB/s

Command Modifiers:
  --json                       Displays output in JSON format (human readable by default).
  --csv                        Displays output in CSV format (human readable by default).
  --file FILE                  Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL             Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```shell
$ amd-smi monitor -ptumv
GPU  POWER  GPU_TEMP  MEM_TEMP  GFX_UTIL  GFX_CLOCK  MEM_UTIL  MEM_CLOCK  VRAM_USED  VRAM_TOTAL
  0  171 W     32 °C     33 °C       0 %    114 MHz       0 %    900 MHz     283 MB   196300 MB
  1  175 W     33 °C     34 °C       0 %    113 MHz       0 %    900 MHz     283 MB   196300 MB
  2  177 W     31 °C     33 °C       0 %    113 MHz       0 %    900 MHz     283 MB   196300 MB
  3  172 W     33 °C     32 °C       0 %    113 MHz       0 %    900 MHz     283 MB   196300 MB
  4  178 W     32 °C     32 °C       0 %    113 MHz       0 %    900 MHz     284 MB   196300 MB
  5  176 W     33 °C     35 °C       0 %    113 MHz       0 %    900 MHz     283 MB   196300 MB
  6  176 W     32 °C     32 °C       0 %    113 MHz       0 %    900 MHz     283 MB   196300 MB
  7  175 W     34 °C     32 °C       0 %    113 MHz       0 %    900 MHz     283 MB   196300 MB
```

- **Integrated ESMI Tool**.  
Users can get CPU metrics and telemetry through our API and CLI tools. This information can be seen in `amd-smi static` and `amd-smi metric` commands. Only available for limited target processors. As of ROCm 6.0.2, this is listed as:
  - AMD Zen3 based CPU Family 19h Models 0h-Fh and 30h-3Fh
  - AMD Zen4 based CPU Family 19h Models 10h-1Fh and A0-AFh

  See a few examples listed below.

```shell
$ amd-smi static -U all
CPU: 0
    SMU:
        FW_VERSION: 85.90.0
    INTERFACE_VERSION:
        PROTO VERSION: 6
```

```shell
$ amd-smi metric -O 0 1 2
CORE: 0
    BOOST_LIMIT:
        VALUE: 400 MHz
    CURR_ACTIVE_FREQ_CORE_LIMIT:
        VALUE: 400 MHz
    CORE_ENERGY:
        VALUE: N/A

CORE: 1
    BOOST_LIMIT:
        VALUE: 400 MHz
    CURR_ACTIVE_FREQ_CORE_LIMIT:
        VALUE: 400 MHz
    CORE_ENERGY:
        VALUE: N/A

CORE: 2
    BOOST_LIMIT:
        VALUE: 400 MHz
    CURR_ACTIVE_FREQ_CORE_LIMIT:
        VALUE: 400 MHz
    CORE_ENERGY:
        VALUE: N/A
```

```shell
$ amd-smi metric -U all
CPU: 0
    POWER_METRICS:
        SOCKET POWER: 102675 mW
        SOCKET POWER LIMIT: 550000 mW
        SOCKET MAX POWER LIMIT: 550000 mW
    PROCHOT:
        PROCHOT_STATUS: 0
    FREQ_METRICS:
        FCLKMEMCLK:
            FCLK: 2000 MHz
            MCLK: 1300 MHz
        CCLKFREQLIMIT: 400 MHz
        SOC_CURRENT_ACTIVE_FREQ_LIMIT:
            FREQ: 400 MHz
            FREQ_SRC: [HSMP Agent]
        SOC_FREQ_RANGE:
            MAX_SOCKET_FREQ: 3700 MHz
            MIN_SOCKET_FREQ: 400 MHz
    C0_RESIDENCY:
        RESIDENCY: 4 %
    SVI_TELEMETRY_ALL_RAILS:
        POWER: 102673 mW
    METRIC_VERSION:
        VERSION: 11
    METRICS_TABLE:
        CPU_FAMILY: 25
        CPU_MODEL: 144
        RESPONSE:
            MTBL_ACCUMULATION_COUNTER: 2887162626
            MTBL_MAX_SOCKET_TEMPERATURE: 41.0 °C
            MTBL_MAX_VR_TEMPERATURE: 39.0 °C
            MTBL_MAX_HBM_TEMPERATURE: 40.0 °C
            MTBL_MAX_SOCKET_TEMPERATURE_ACC: 108583340881.125 °C
            MTBL_MAX_VR_TEMPERATURE_ACC: 109472702595.0 °C
            MTBL_MAX_HBM_TEMPERATURE_ACC: 111516663941.0 °C
            MTBL_SOCKET_POWER_LIMIT: 550.0 W
            MTBL_MAX_SOCKET_POWER_LIMIT: 550.0 W
            MTBL_SOCKET_POWER: 102.678 W
            MTBL_TIMESTAMP_RAW: 288731677361880
            MTBL_TIMESTAMP_READABLE: Tue Mar 19 12:32:21 2024
            MTBL_SOCKET_ENERGY_ACC: 166127.84 kJ
            MTBL_CCD_ENERGY_ACC: 3317.837 kJ
            MTBL_XCD_ENERGY_ACC: 21889.147 kJ
            MTBL_AID_ENERGY_ACC: 121932.397 kJ
            MTBL_HBM_ENERGY_ACC: 18994.108 kJ
            MTBL_CCLK_FREQUENCY_LIMIT: 3.7 GHz
            MTBL_GFXCLK_FREQUENCY_LIMIT: 0.0 MHz
            MTBL_FCLK_FREQUENCY: 1999.988 MHz
            MTBL_UCLK_FREQUENCY: 1299.993 MHz
            MTBL_SOCCLK_FREQUENCY: [35.716, 35.715, 35.714, 35.714] MHz
            MTBL_VCLK_FREQUENCY: [0.0, 53.749, 53.749, 53.749] MHz
            MTBL_DCLK_FREQUENCY: [7.143, 44.791, 44.791, 44.791] MHz
            MTBL_LCLK_FREQUENCY: [20.872, 18.75, 35.938, 599.558] MHz
            MTBL_FCLK_FREQUENCY_TABLE: [1200.0, 1600.0, 1900.0, 2000.0] MHz
            MTBL_UCLK_FREQUENCY_TABLE: [900.0, 1100.0, 1200.0, 1300.0] MHz
            MTBL_SOCCLK_FREQUENCY_TABLE: [800.0, 1000.0, 1142.857, 1142.857] MHz
            MTBL_VCLK_FREQUENCY_TABLE: [914.286, 1300.0, 1560.0, 1720.0] MHz
            MTBL_DCLK_FREQUENCY_TABLE: [711.111, 975.0, 1300.0, 1433.333] MHz
            MTBL_LCLK_FREQUENCY_TABLE: [600.0, 844.444, 1150.0, 1150.0] MHz
            MTBL_CCLK_FREQUENCY_ACC: [4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639,
                4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639,
                4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639,
                4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639,
                4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639, 4399751656.639,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] GHz
            MTBL_GFXCLK_FREQUENCY_ACC: [0.0, 0.0, 250534397827.603, 251546257401.82, 250811364089.836,
                249999070486.505, 251622633562.855, 251342375116.05] MHz
            MTBL_GFXCLK_FREQUENCY: [0.0, 0.0, 31.091, 31.414, 31.141, 31.478, 31.32, 31.453]
                MHz
            MTBL_MAX_CCLK_FREQUENCY: 3.7 GHz
            MTBL_MIN_CCLK_FREQUENCY: 0.4 GHz
            MTBL_MAX_GFXCLK_FREQUENCY: 2100.0 MHz
            MTBL_MIN_GFXCLK_FREQUENCY: 500.0 MHz
            MTBL_MAX_LCLK_DPM_RANGE: 2
            MTBL_MIN_LCLK_DPM_RANGE: 0
            MTBL_XGMI_WIDTH: 0.0
            MTBL_XGMI_BITRATE: 0.0 Gbps
            MTBL_XGMI_READ_BANDWIDTH_ACC: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] Gbps
            MTBL_XGMI_WRITE_BANDWIDTH_ACC: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] Gbps
            MTBL_SOCKET_C0_RESIDENCY: 4.329 %
            MTBL_SOCKET_GFX_BUSY: 0.0 %
            MTBL_HBM_BANDWIDTH_UTILIZATION: 0.001 %
            MTBL_SOCKET_C0_RESIDENCY_ACC: 311523106.34
            MTBL_SOCKET_GFX_BUSY_ACC: 84739.281
            MTBL_HBM_BANDWIDTH_ACC: 33231180.073 Gbps
            MTBL_MAX_HBM_BANDWIDTH: 5324.801 Gbps
            MTBL_DRAM_BANDWIDTH_UTILIZATION_ACC: 612843.699
            MTBL_PCIE_BANDWIDTH_ACC: [0.0, 0.0, 0.0, 0.0] Gbps
            MTBL_PROCHOT_RESIDENCY_ACC: 0
            MTBL_PPT_RESIDENCY_ACC: 2887162626
            MTBL_SOCKET_THM_RESIDENCY_ACC: 2887162626
            MTBL_VR_THM_RESIDENCY_ACC: 0
            MTBL_HBM_THM_RESIDENCY_ACC: 2887162626
    SOCKET_ENERGY:
        RESPONSE: N/A
    DDR_BANDWIDTH:
        RESPONSE: N/A
    CPU_TEMP:
        RESPONSE: N/A
```

- **Added support for new metrics: VCN, JPEG engines, and PCIe errors**.  
Using the AMD SMI tool, users can retreive VCN, JPEG engines, and PCIe errors by calling `amd-smi metric -P` or `amd-smi metric --usage`. Depending on device support, `VCN_ACTIVITY` will update for MI3x ASICs (with 4 separate VCN engine activities) for older asics `MM_ACTIVITY` with UVD/VCN engine activity (average of all engines). `JPEG_ACTIVITY` is a new field for MI3x ASICs, where device can support up to 32 JPEG engine activities. See our documentation for more in-depth understanding of these new fields.

```shell
$ amd-smi metric -P
GPU: 0
    PCIE:
        WIDTH: 16
        SPEED: 16 GT/s
        REPLAY_COUNT: 0
        L0_TO_RECOVERY_COUNT: 1
        REPLAY_ROLL_OVER_COUNT: 0
        NAK_SENT_COUNT: 0
        NAK_RECEIVED_COUNT: 0
        CURRENT_BANDWIDTH_SENT: N/A
        CURRENT_BANDWIDTH_RECEIVED: N/A
        MAX_PACKET_SIZE: N/A
```

```shell
$ amd-smi metric --usage
GPU: 0
    USAGE:
        GFX_ACTIVITY: 0 %
        UMC_ACTIVITY: 0 %
        MM_ACTIVITY: N/A
        VCN_ACTIVITY: [0 %, 0 %, 0 %, 0 %]
        JPEG_ACTIVITY: [0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0
            %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %, 0 %,
            0 %, 0 %, 0 %, 0 %]

```

- **Added AMDSMI Tool Version**.  
AMD SMI will report ***three versions***: AMDSMI Tool, AMDSMI Library version, and ROCm version.
The AMDSMI Tool version is the CLI/tool version number with commit ID appended after `+` sign.
The AMDSMI Library version is the library package version number.
The ROCm version is the system's installed ROCm version, if ROCm is not installed it will report N/A.

```shell
$ amd-smi version
AMDSMI Tool: 23.4.2+505b858 | AMDSMI Library version: 24.2.0.0 | ROCm version: 6.1.0
```

- **Added XGMI table**.  
Displays XGMI information for AMD GPU devices in a table format. Only available on supported ASICs (eg. MI300). Here users can view read/write data XGMI or PCIe accumulated data transfer size (in KiloBytes).

```shell
$ amd-smi xgmi
LINK METRIC TABLE:
       bdf          bit_rate max_bandwidth link_type 0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
GPU0   0000:0c:00.0 32 Gb/s  512 Gb/s      XGMI
 Read                                                N/A          2 KB         2 KB         1 KB         2 KB         1 KB         2 KB         2 KB
 Write                                               N/A          1 KB         1 KB         1 KB         1 KB         1 KB         1 KB         1 KB
GPU1   0000:22:00.0 32 Gb/s  512 Gb/s      XGMI
 Read                                                0 KB         N/A          2 KB         2 KB         1 KB         2 KB         1 KB         2 KB
 Write                                               0 KB         N/A          1 KB         1 KB         1 KB         1 KB         1 KB         1 KB
GPU2   0000:38:00.0 32 Gb/s  512 Gb/s      XGMI
 Read                                                0 KB         1 KB         N/A          2 KB         1 KB         2 KB         0 KB         0 KB
 Write                                               0 KB         1 KB         N/A          1 KB         1 KB         1 KB         1 KB         1 KB
GPU3   0000:5c:00.0 32 Gb/s  512 Gb/s      XGMI
 Read                                                0 KB         0 KB         2 KB         N/A          1 KB         0 KB         0 KB         2 KB
 Write                                               0 KB         1 KB         1 KB         N/A          1 KB         1 KB         1 KB         1 KB
GPU4   0000:9f:00.0 32 Gb/s  512 Gb/s      XGMI
 Read                                                0 KB         1 KB         0 KB         0 KB         N/A          2 KB         0 KB         2 KB
 Write                                               0 KB         1 KB         1 KB         1 KB         N/A          1 KB         1 KB         1 KB
GPU5   0000:af:00.0 32 Gb/s  512 Gb/s      XGMI
 Read                                                0 KB         2 KB         0 KB         0 KB         0 KB         N/A          2 KB         0 KB
 Write                                               0 KB         1 KB         1 KB         1 KB         1 KB         N/A          1 KB         1 KB
GPU6   0000:bf:00.0 32 Gb/s  512 Gb/s      XGMI
 Read                                                0 KB         0 KB         0 KB         0 KB         0 KB         0 KB         N/A          0 KB
 Write                                               0 KB         1 KB         1 KB         1 KB         1 KB         1 KB         N/A          1 KB
GPU7   0000:df:00.0 32 Gb/s  512 Gb/s      XGMI
 Read                                                0 KB         0 KB         0 KB         0 KB         0 KB         0 KB         0 KB         N/A
 Write                                               0 KB         1 KB         1 KB         1 KB         1 KB         1 KB         1 KB         N/A

```

- **Added units of measure to JSON output**.  
We added unit of measure to JSON/CSV `amd-smi metric`, `amd-smi static`, and `amd-smi monitor` commands.

Ex.

```shell
amd-smi metric -p --json
[
    {
        "gpu": 0,
        "power": {
            "socket_power": {
                "value": 10,
                "unit": "W"
            },
            "gfx_voltage": {
                "value": 6,
                "unit": "mV"
            },
            "soc_voltage": {
                "value": 918,
                "unit": "mV"
            },
            "mem_voltage": {
                "value": 1250,
                "unit": "mV"
            },
            "power_management": "ENABLED",
            "throttle_status": "UNTHROTTLED"
        }
    }
]
```

### Changed

- **Topology is now left-aligned with BDF of each device listed individual table's row/coloumns**.  
We provided each device's BDF for every table's row/columns, then left aligned data. We want AMD SMI Tool output to be easy to understand and digest for our users. Having users scroll up to find this information made it difficult to follow, especially for devices which have many devices associated with one ASIC.

```shell
$ amd-smi topology
ACCESS TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:22:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:38:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:5c:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:9f:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:af:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:bf:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED
0000:df:00.0 ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED      ENABLED

WEIGHT TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 0            15           15           15           15           15           15           15
0000:22:00.0 15           0            15           15           15           15           15           15
0000:38:00.0 15           15           0            15           15           15           15           15
0000:5c:00.0 15           15           15           0            15           15           15           15
0000:9f:00.0 15           15           15           15           0            15           15           15
0000:af:00.0 15           15           15           15           15           0            15           15
0000:bf:00.0 15           15           15           15           15           15           0            15
0000:df:00.0 15           15           15           15           15           15           15           0

HOPS TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 0            1            1            1            1            1            1            1
0000:22:00.0 1            0            1            1            1            1            1            1
0000:38:00.0 1            1            0            1            1            1            1            1
0000:5c:00.0 1            1            1            0            1            1            1            1
0000:9f:00.0 1            1            1            1            0            1            1            1
0000:af:00.0 1            1            1            1            1            0            1            1
0000:bf:00.0 1            1            1            1            1            1            0            1
0000:df:00.0 1            1            1            1            1            1            1            0

LINK TYPE TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 SELF         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI
0000:22:00.0 XGMI         SELF         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI
0000:38:00.0 XGMI         XGMI         SELF         XGMI         XGMI         XGMI         XGMI         XGMI
0000:5c:00.0 XGMI         XGMI         XGMI         SELF         XGMI         XGMI         XGMI         XGMI
0000:9f:00.0 XGMI         XGMI         XGMI         XGMI         SELF         XGMI         XGMI         XGMI
0000:af:00.0 XGMI         XGMI         XGMI         XGMI         XGMI         SELF         XGMI         XGMI
0000:bf:00.0 XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         SELF         XGMI
0000:df:00.0 XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         XGMI         SELF

NUMA BW TABLE:
             0000:0c:00.0 0000:22:00.0 0000:38:00.0 0000:5c:00.0 0000:9f:00.0 0000:af:00.0 0000:bf:00.0 0000:df:00.0
0000:0c:00.0 N/A          50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000
0000:22:00.0 50000-50000  N/A          50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000
0000:38:00.0 50000-50000  50000-50000  N/A          50000-50000  50000-50000  50000-50000  50000-50000  50000-50000
0000:5c:00.0 50000-50000  50000-50000  50000-50000  N/A          50000-50000  50000-50000  50000-50000  50000-50000
0000:9f:00.0 50000-50000  50000-50000  50000-50000  50000-50000  N/A          50000-50000  50000-50000  50000-50000
0000:af:00.0 50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  N/A          50000-50000  50000-50000
0000:bf:00.0 50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  N/A          50000-50000
0000:df:00.0 50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  50000-50000  N/A
```

### Resolved issues

- **Fix for Navi3X/Navi2X/MI100 `amdsmi_get_gpu_pci_bandwidth()` in frequencies_read tests**.  
Devices which do not report (eg. Navi3X/Navi2X/MI100) we have added checks to confirm these devices return AMDSMI_STATUS_NOT_SUPPORTED. Otherwise, tests now display a return string.
- **Fix for devices which have an older pyyaml installed**.  
Platforms which are identified as having an older pyyaml version or pip, we no manually update both pip and pyyaml as needed. This corrects issues identified below. Fix impacts the following CLI commands:
  - `amd-smi list`
  - `amd-smi static`
  - `amd-smi firmware`
  - `amd-smi metric`
  - `amd-smi topology`

```shell
TypeError: dump_all() got an unexpected keyword argument 'sort_keys'
```

- **Fix for crash when user is not a member of video/render groups**.  
AMD SMI now uses same mutex handler for devices as rocm-smi. This helps avoid crashes when DRM/device data is inaccessable to the logged in user.

## amd_smi_lib for ROCm 6.0.0

### Added

- **Integrated the E-SMI (EPYC-SMI) library**.  
You can now query CPU-related information directly through AMD SMI. Metrics include power, energy, performance, and other system details.

- **Added support for gfx942 metrics**.  
You can now query MI300 device metrics to get real-time information. Metrics include power, temperature, energy, and performance.

- **Compute and memory partition support**.  
Users can now view, set, and reset partitions. The topology display can provide a more in-depth look at the device's current configuration.

### Changed

- **GPU index sorting made consistent with other tools**.  
To ensure alignment with other ROCm software tools, GPU index sorting is optimized to use Bus:Device.Function (BDF) rather than the card number.
- **Topology output is now aligned with GPU BDF table**.  
Earlier versions of the topology output were difficult to read since each GPU was displayed linearly.
Now the information is displayed as a table by each GPU's BDF, which closer resembles rocm-smi output.

### Optimized

- Updated to C++17, gtest-1.14, and cmake 3.14

### Resolved issues

- **Fix for driver not initialized**.  
If driver module is not loaded, user retrieve error reponse indicating amdgpu module is not loaded.

