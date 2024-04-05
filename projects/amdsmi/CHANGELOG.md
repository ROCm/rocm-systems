# Change Log for AMD SMI Library

Full documentation for amd_smi_lib is available at [https://rocm.docs.amd.com/](https://rocm.docs.amd.com/projects/amdsmi/en/latest/).

***All information listed below is for reference and subject to change.***

## amd_smi_lib for ROCm 6.1.1

### Added

- N/A

### Changed

- **Updated metrics --clocks**
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

- **Added deferred ecc counts**
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

- **Updated `amd-smi topology --json` to align with host/guest**  
Topology's `--json` output now is changed to align with output reported bt host/guest systems. Additionally, users can select/filter specific topology details as desired (refer to `amd-smi topology -h` for full list). See examples shown below.

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

### Optimizations

- N/A

### Fixed

- **Fix for GPU reset error on non-amdgpu cards**
Previously our reset could attempting to reset non-amd GPUS- resuting in "Unable to reset non-amd GPU" error. Fix
updates CLI to target only AMD ASICs.

- **Fix for `amd-smi metric --pcie` and `amdsmi_get_pcie_info()`Navi32/31 cards**
Updated API to include `amdsmi_card_form_factor_t.AMDSMI_CARD_FORM_FACTOR_CEM`. Prevously, this would report "UNKNOWN". This fix
provides the correct board `SLOT_TYPE` associated with these ASICs (and other Navi cards).

- **Fix for `amd-smi process`**
Fixed output results when getting processes running on a device.

- **Improved Error handling for `amd-smi process`**
Fixed Attribute Error when getting process in csv format

### Known issues

- `amd-smi bad-pages` can results with "ValueError: NULL pointer access" with certain PM FW versions

## amd_smi_lib for ROCm 6.1.0

### Added

- **Added Monitor Command**
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
                               ID: 0 | BDF: 0000:01:00.0 | UUID: 4eff74a0-0000-1000-802d-1d762a397f73
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

- **Integrated ESMI Tool**
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

- **Added support for new metrics: VCN, JPEG engines, and PCIe errors**
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

- **Added AMDSMI Tool Version**
AMD SMI will report ***three versions***: AMDSMI Tool, AMDSMI Library version, and ROCm version.
The AMDSMI Tool version is the CLI/tool version number with commit ID appended after `+` sign.
The AMDSMI Library version is the library package version number.
The ROCm version is the system's installed ROCm version, if ROCm is not installed it will report N/A.

```shell
$ amd-smi version
AMDSMI Tool: 23.4.2+505b858 | AMDSMI Library version: 24.2.0.0 | ROCm version: 6.1.0
```

- **Added XGMI table**
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

- **Added units of measure to JSON output.**
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

- **Topology is now left-aligned with BDF of each device listed individual table's row/coloumns.**
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

### Optimizations

- N/A

### Fixed

- **Fix for Navi3X/Navi2X/MI100 `amdsmi_get_gpu_pci_bandwidth()` in frequencies_read tests**
Devices which do not report (eg. Navi3X/Navi2X/MI100) we have added checks to confirm these devices return AMDSMI_STATUS_NOT_SUPPORTED. Otherwise, tests now display a return string.
- **Fix for devices which have an older pyyaml installed**
Platforms which are identified as having an older pyyaml version or pip, we no manually update both pip and pyyaml as needed. This corrects issues identified below. Fix impacts the following CLI commands:
  - `amd-smi list`
  - `amd-smi static`
  - `amd-smi firmware`
  - `amd-smi metric`
  - `amd-smi topology`

```shell
TypeError: dump_all() got an unexpected keyword argument 'sort_keys'
```

- **Fix for crash when user is not a member of video/render groups**
AMD SMI now uses same mutex handler for devices as rocm-smi. This helps avoid crashes when DRM/device data is inaccessable to the logged in user.

### Known Issues

- N/A

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

- Updated to C++17, gtest-1.14, and cmake 3.14

### Fixed

- **Fix for driver not initialized**
If driver module is not loaded, user retrieve error reponse indicating amdgpu module is not loaded.

### Known Issues

- N/A
