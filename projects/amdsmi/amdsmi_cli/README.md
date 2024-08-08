# AMD SMI CLI Tool

This tool acts as a command line interface for manipulating
and monitoring the amdgpu kernel, and is intended to replace
and deprecate the existing rocm_smi CLI tool & gpuv-smi tool.
It uses Ctypes to call the amd_smi_lib API.
Recommended: At least one AMD GPU with AMD driver installed

## Install CLI Tool and Python Library

### Requirements

* python 3.6.8+ 64-bit
* amdgpu or amd_hsmp driver must be loaded for amdsmi_init() to pass

### Installation

* [Install amdgpu driver](../README.md#install-amdgpu-using-rocm)
* Optionally install amd_hsmp driver for ESMI CPU functions
* Install amd-smi-lib package through package manager
* amd-smi --help

### Install Example for Ubuntu 22.04

``` bash
apt install amd-smi-lib
amd-smi --help
```

### Optional autocompletion

`amd-smi` cli application supports autocompletion. The package should attempt to install it, if argcomplete is not installed you can enable it by using the following commands:

```bash
python3 -m pip install argcomplete
activate-global-python-argcomplete --user
# restart shell to enable
```

### Manual/Multiple Rocm Instance Python Library Install

In the event there are multiple rocm installations and pyenv is not being used, to use the correct amdsmi version you must uninstall previous versions of amd-smi and install the version you want directly from your rocm instance.

#### Python Library Install Example for Ubuntu 22.04

Remove previous amdsmi installation:

```bash
python3 -m pip list | grep amd
python3 -m pip uninstall amdsmi
```

Then install Python library from your target rocm instance:

``` bash
apt install amd-smi-lib
amd-smi --help
cd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install --user .
```

Now you have the amdsmi python library in your python path:

``` bash
~$ python3
Python 3.8.10 (default, May 26 2023, 14:05:08)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import amdsmi
>>>
```

## Usage

AMD-SMI reports the version and current platform detected when running the command line interface (CLI) without arguments:

``` bash
~$ amd-smi
usage: amd-smi [-h]  ...

AMD System Management Interface | Version: 24.6.3.0 | ROCm version: 6.2.1 | Platform: Linux Baremetal

options:
  -h, --help          show this help message and exit

AMD-SMI Commands:
                      Descriptions:
    version           Display version information
    list              List GPU information
    static            Gets static information about the specified GPU
    firmware (ucode)  Gets firmware information about the specified GPU
    bad-pages         Gets bad page information about the specified GPU
    metric            Gets metric/performance information about the specified GPU
    process           Lists general process information running on the specified GPU
    event             Displays event information for the given GPU
    topology          Displays topology information of the devices
    set               Set options for devices
    reset             Reset options for devices
    monitor (dmon)    Monitor metrics for target devices
    xgmi              Displays xgmi information of the devices
```

More detailed verison information is available from `amd-smi version`

Each command will have detailed information via `amd-smi [command] --help`

## Commands

For convenience, here is the help output for each command

```bash
~$ amd-smi list --help
usage: amd-smi list [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                    [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]]

Lists all the devices on the system and the links between devices.
Lists all the sockets and for each socket, GPUs and/or CPUs associated to
that socket alongside some basic information for each device.
In virtualization environments, it can also list VFs associated to each
GPU with some basic information for each VF.

options:
  -h, --help                  show this help message and exit
  -g, --gpu GPU [GPU ...]     Select a GPU ID, BDF, or UUID from the possible choices:
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                              ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                              ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                all | Selects all devices
  -U, --cpu CPU [CPU ...]     Select a CPU ID from the possible choices:
                              ID: 0
                              ID: 1
                              ID: 2
                              ID: 3
                                all | Selects all devices
  -O, --core CORE [CORE ...]  Select a Core ID from the possible choices:
                              ID: 0 - 95
                                all  | Selects all devices

Command Modifiers:
  --json                      Displays output in JSON format (human readable by default).
  --csv                       Displays output in CSV format (human readable by default).
  --file FILE                 Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL            Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi static --help
usage: amd-smi static [-h] [-g GPU [GPU ...] | -U CPU [CPU ...]] [-a] [-b] [-V] [-d] [-v]
                      [-c] [-B] [-R] [-r] [-p] [-l] [-P] [-x] [-u] [-s] [-i]
                      [--json | --csv] [--file FILE] [--loglevel LEVEL]

If no GPU is specified, returns static information for all GPUs on the system.
If no static argument is provided, all static information will be displayed.

Static Arguments:
  -h, --help               show this help message and exit
  -g, --gpu GPU [GPU ...]  Select a GPU ID, BDF, or UUID from the possible choices:
                           ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                           ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                           ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                           ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                             all | Selects all devices
  -U, --cpu CPU [CPU ...]  Select a CPU ID from the possible choices:
                           ID: 0
                           ID: 1
                           ID: 2
                           ID: 3
                             all | Selects all devices
  -a, --asic               All asic information
  -b, --bus                All bus information
  -V, --vbios              All video bios information (if available)
  -d, --driver             Displays driver version
  -v, --vram               All vram information
  -c, --cache              All cache information
  -B, --board              All board information
  -R, --process-isolation  The process isolation status
  -r, --ras                Displays RAS features information
  -p, --partition          Partition information
  -l, --limit              All limit metric values (i.e. power and thermal limits)
  -P, --policy             The available DPM policy
  -x, --xgmi-plpd          The available XGMI per-link power down policy
  -u, --numa               All numa node information

CPU Arguments:
  -s, --smu                All SMU FW information
  -i, --interface-ver      Displays hsmp interface version

Command Modifiers:
  --json                   Displays output in JSON format (human readable by default).
  --csv                    Displays output in CSV format (human readable by default).
  --file FILE              Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL         Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi firmware --help
usage: amd-smi firmware [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                        [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]] [-f]

If no GPU is specified, return firmware information for all GPUs on the system.

Firmware Arguments:
  -h, --help                   show this help message and exit
  -g, --gpu GPU [GPU ...]      Select a GPU ID, BDF, or UUID from the possible choices:
                               ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                               ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                               ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                               ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                 all | Selects all devices
  -U, --cpu CPU [CPU ...]      Select a CPU ID from the possible choices:
                               ID: 0
                               ID: 1
                               ID: 2
                               ID: 3
                                 all | Selects all devices
  -O, --core CORE [CORE ...]   Select a Core ID from the possible choices:
                               ID: 0 - 95
                                 all  | Selects all devices
  -f, --ucode-list, --fw-list  All FW list information

Command Modifiers:
  --json                       Displays output in JSON format (human readable by default).
  --csv                        Displays output in CSV format (human readable by default).
  --file FILE                  Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL             Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi bad-pages --help
usage: amd-smi bad-pages [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                         [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]] [-p]
                         [-r] [-u]

If no GPU is specified, return bad page information for all GPUs on the system.

Bad Pages Arguments:
  -h, --help                  show this help message and exit
  -g, --gpu GPU [GPU ...]     Select a GPU ID, BDF, or UUID from the possible choices:
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                              ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                              ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                all | Selects all devices
  -U, --cpu CPU [CPU ...]     Select a CPU ID from the possible choices:
                              ID: 0
                              ID: 1
                              ID: 2
                              ID: 3
                                all | Selects all devices
  -O, --core CORE [CORE ...]  Select a Core ID from the possible choices:
                              ID: 0 - 95
                                all  | Selects all devices
  -p, --pending               Displays all pending retired pages
  -r, --retired               Displays retired pages
  -u, --un-res                Displays unreservable pages

Command Modifiers:
  --json                      Displays output in JSON format (human readable by default).
  --csv                       Displays output in CSV format (human readable by default).
  --file FILE                 Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL            Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi metric --help
usage: amd-smi metric [-h] [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]]
                      [-w INTERVAL] [-W TIME] [-i ITERATIONS] [-m] [-u] [-p] [-c] [-t]
                      [-P] [-e] [-k] [-f] [-C] [-o] [-l] [-x] [-E] [--cpu-power-metrics]
                      [--cpu-prochot] [--cpu-freq-metrics] [--cpu-c0-res]
                      [--cpu-lclk-dpm-level NBIOID] [--cpu-pwr-svi-telemtry-rails]
                      [--cpu-io-bandwidth IO_BW LINKID_NAME]
                      [--cpu-xgmi-bandwidth XGMI_BW LINKID_NAME] [--cpu-metrics-ver]
                      [--cpu-metrics-table] [--cpu-socket-energy] [--cpu-ddr-bandwidth]
                      [--cpu-temp] [--cpu-dimm-temp-range-rate DIMM_ADDR]
                      [--cpu-dimm-pow-consumption DIMM_ADDR]
                      [--cpu-dimm-thermal-sensor DIMM_ADDR] [--core-boost-limit]
                      [--core-curr-active-freq-core-limit] [--core-energy]
                      [--json | --csv] [--file FILE] [--loglevel LEVEL]

If no GPU is specified, returns metric information for all GPUs on the system.
If no metric argument is provided all metric information will be displayed.

Metric arguments:
  -h, --help                                show this help message and exit
  -g, --gpu GPU [GPU ...]                   Select a GPU ID, BDF, or UUID from the possible choices:
                                            ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                                            ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                                            ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                                            ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                              all | Selects all devices
  -U, --cpu CPU [CPU ...]                   Select a CPU ID from the possible choices:
                                            ID: 0
                                            ID: 1
                                            ID: 2
                                            ID: 3
                                              all | Selects all devices
  -O, --core CORE [CORE ...]                Select a Core ID from the possible choices:
                                            ID: 0 - 95
                                              all  | Selects all devices
  -w, --watch INTERVAL                      Reprint the command in a loop of INTERVAL seconds
  -W, --watch_time TIME                     The total TIME to watch the given command
  -i, --iterations ITERATIONS               Total number of ITERATIONS to loop on the given command
  -m, --mem-usage                           Memory usage per block
  -u, --usage                               Displays engine usage information
  -p, --power                               Current power usage
  -c, --clock                               Average, max, and current clock frequencies
  -t, --temperature                         Current temperatures
  -P, --pcie                                Current PCIe speed, width, and replay count
  -e, --ecc                                 Total number of ECC errors
  -k, --ecc-blocks                          Number of ECC errors per block
  -f, --fan                                 Current fan speed
  -C, --voltage-curve                       Display voltage curve
  -o, --overdrive                           Current GPU clock overdrive level
  -l, --perf-level                          Current DPM performance level
  -x, --xgmi-err                            XGMI error information since last read
  -E, --energy                              Amount of energy consumed

CPU Arguments:
  --cpu-power-metrics                       CPU power metrics
  --cpu-prochot                             Displays prochot status
  --cpu-freq-metrics                        Displays currentFclkMemclk frequencies and cclk frequency limit
  --cpu-c0-res                              Displays C0 residency
  --cpu-lclk-dpm-level NBIOID               Displays lclk dpm level range. Requires socket ID and NBOID as inputs
  --cpu-pwr-svi-telemtry-rails              Displays svi based telemetry for all rails
  --cpu-io-bandwidth IO_BW LINKID_NAME      Displays current IO bandwidth for the selected CPU.
                                             input parameters are bandwidth type(1) and link ID encodings
                                             i.e. P2, P3, G0 - G7
  --cpu-xgmi-bandwidth XGMI_BW LINKID_NAME  Displays current XGMI bandwidth for the selected CPU
                                             input parameters are bandwidth type(1,2,4) and link ID encodings
                                             i.e. P2, P3, G0 - G7
  --cpu-metrics-ver                         Displays metrics table version
  --cpu-metrics-table                       Displays metric table
  --cpu-socket-energy                       Displays socket energy for the selected CPU socket
  --cpu-ddr-bandwidth                       Displays per socket max ddr bw, current utilized bw,
                                             and current utilized ddr bw in percentage
  --cpu-temp                                Displays cpu socket temperature
  --cpu-dimm-temp-range-rate DIMM_ADDR      Displays dimm temperature range and refresh rate
  --cpu-dimm-pow-consumption DIMM_ADDR      Displays dimm power consumption
  --cpu-dimm-thermal-sensor DIMM_ADDR       Displays dimm thermal sensor

CPU Core Arguments:
  --core-boost-limit                        Get boost limit for the selected cores
  --core-curr-active-freq-core-limit        Get Current CCLK limit set per Core
  --core-energy                             Displays core energy for the selected core

Command Modifiers:
  --json                                    Displays output in JSON format (human readable by default).
  --csv                                     Displays output in CSV format (human readable by default).
  --file FILE                               Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL                          Set the logging level from the possible choices:
                                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi process --help
usage: amd-smi process [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                       [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]]
                       [-w INTERVAL] [-W TIME] [-i ITERATIONS] [-G] [-e] [-p PID]
                       [-n NAME]

If no GPU is specified, returns information for all GPUs on the system.
If no process argument is provided all process information will be displayed.

Process arguments:
  -h, --help                   show this help message and exit
  -g, --gpu GPU [GPU ...]      Select a GPU ID, BDF, or UUID from the possible choices:
                               ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                               ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                               ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                               ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                 all | Selects all devices
  -U, --cpu CPU [CPU ...]      Select a CPU ID from the possible choices:
                               ID: 0
                               ID: 1
                               ID: 2
                               ID: 3
                                 all | Selects all devices
  -O, --core CORE [CORE ...]   Select a Core ID from the possible choices:
                               ID: 0 - 95
                                 all  | Selects all devices
  -w, --watch INTERVAL         Reprint the command in a loop of INTERVAL seconds
  -W, --watch_time TIME        The total TIME to watch the given command
  -i, --iterations ITERATIONS  Total number of ITERATIONS to loop on the given command
  -G, --general                pid, process name, memory usage
  -e, --engine                 All engine usages
  -p, --pid PID                Gets all process information about the specified process based on Process ID
  -n, --name NAME              Gets all process information about the specified process based on Process Name.
                               If multiple processes have the same name information is returned for all of them.

Command Modifiers:
  --json                       Displays output in JSON format (human readable by default).
  --csv                        Displays output in CSV format (human readable by default).
  --file FILE                  Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL             Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi event --help
usage: amd-smi event [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                     [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]]

If no GPU is specified, returns event information for all GPUs on the system.

Event Arguments:
  -h, --help                  show this help message and exit
  -g, --gpu GPU [GPU ...]     Select a GPU ID, BDF, or UUID from the possible choices:
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                              ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                              ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                all | Selects all devices
  -U, --cpu CPU [CPU ...]     Select a CPU ID from the possible choices:
                              ID: 0
                              ID: 1
                              ID: 2
                              ID: 3
                                all | Selects all devices
  -O, --core CORE [CORE ...]  Select a Core ID from the possible choices:
                              ID: 0 - 95
                                all  | Selects all devices

Command Modifiers:
  --json                      Displays output in JSON format (human readable by default).
  --csv                       Displays output in CSV format (human readable by default).
  --file FILE                 Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL            Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi topology --help
usage: amd-smi topology [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                        [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]] [-a]
                        [-w] [-o] [-t] [-b]

If no GPU is specified, returns information for all GPUs on the system.
If no topology argument is provided all topology information will be displayed.

Topology arguments:
  -h, --help                  show this help message and exit
  -g, --gpu GPU [GPU ...]     Select a GPU ID, BDF, or UUID from the possible choices:
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                              ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                              ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                all | Selects all devices
  -U, --cpu CPU [CPU ...]     Select a CPU ID from the possible choices:
                              ID: 0
                              ID: 1
                              ID: 2
                              ID: 3
                                all | Selects all devices
  -O, --core CORE [CORE ...]  Select a Core ID from the possible choices:
                              ID: 0 - 95
                                all  | Selects all devices
  -a, --access                Displays link accessibility between GPUs
  -w, --weight                Displays relative weight between GPUs
  -o, --hops                  Displays the number of hops between GPUs
  -t, --link-type             Displays the link type between GPUs
  -b, --numa-bw               Display max and min bandwidth between nodes

Command Modifiers:
  --json                      Displays output in JSON format (human readable by default).
  --csv                       Displays output in CSV format (human readable by default).
  --file FILE                 Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL            Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi set --help
usage: amd-smi set [-h] (-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]) [-f %]
                   [-l LEVEL] [-P SETPROFILE] [-d SCLKMAX] [-C PARTITION] [-M PARTITION]
                   [-o WATTS] [-p POLICY_ID] [-x POLICY_ID] [-R STATUS]
                   [--cpu-pwr-limit PWR_LIMIT] [--cpu-xgmi-link-width MIN_WIDTH MAX_WIDTH]
                   [--cpu-lclk-dpm-level NBIOID MIN_DPM MAX_DPM] [--cpu-pwr-eff-mode MODE]
                   [--cpu-gmi3-link-width MIN_LW MAX_LW] [--cpu-pcie-link-rate LINK_RATE]
                   [--cpu-df-pstate-range MAX_PSTATE MIN_PSTATE] [--cpu-enable-apb]
                   [--cpu-disable-apb DF_PSTATE] [--soc-boost-limit BOOST_LIMIT]
                   [--core-boost-limit BOOST_LIMIT] [--json | --csv] [--file FILE]
                   [--loglevel LEVEL]

A GPU must be specified to set a configuration.
A set argument must be provided; Multiple set arguments are accepted

Set Arguments:
  -h, --help                                   show this help message and exit
  -g, --gpu GPU [GPU ...]                      Select a GPU ID, BDF, or UUID from the possible choices:
                                               ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                                               ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                                               ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                                               ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                                 all | Selects all devices
  -U, --cpu CPU [CPU ...]                      Select a CPU ID from the possible choices:
                                               ID: 0
                                               ID: 1
                                               ID: 2
                                               ID: 3
                                                 all | Selects all devices
  -O, --core CORE [CORE ...]                   Select a Core ID from the possible choices:
                                               ID: 0 - 95
                                                 all  | Selects all devices
  -f, --fan %                                  Set GPU fan speed (0-255 or 0-100%)
  -l, --perf-level LEVEL                       Set performance level
  -P, --profile SETPROFILE                     Set power profile level (#) or a quoted string of custom profile attributes
  -d, --perf-determinism SCLKMAX               Set GPU clock frequency limit and performance level to determinism to get minimal performance variation
  -C, --compute-partition PARTITION            Set one of the following the compute partition modes:
                                                CPX, SPX, DPX, TPX, QPX
  -M, --memory-partition PARTITION             Set one of the following the memory partition modes:
                                                NPS1, NPS2, NPS4, NPS8
  -o, --power-cap WATTS                        Set power capacity limit
  -p, --dpm-policy POLICY_ID                   Set the GPU DPM policy using policy id
  -x, --xgmi-plpd POLICY_ID                    Set the GPU XGMI per-link power down policy using policy id
  -R, --process-isolation STATUS               Enable or disable the GPU process isolation: 0 for disable and 1 for enable.

CPU Arguments:
  --cpu-pwr-limit PWR_LIMIT                    Set power limit for the given socket. Input parameter is power limit value.
  --cpu-xgmi-link-width MIN_WIDTH MAX_WIDTH    Set max and Min linkwidth. Input parameters are min and max link width values
  --cpu-lclk-dpm-level NBIOID MIN_DPM MAX_DPM  Sets the max and min dpm level on a given NBIO.
                                                Input parameters are die_index, min dpm, max dpm.
  --cpu-pwr-eff-mode MODE                      Sets the power efficency mode policy. Input parameter is mode.
  --cpu-gmi3-link-width MIN_LW MAX_LW          Sets max and min gmi3 link width range
  --cpu-pcie-link-rate LINK_RATE               Sets pcie link rate
  --cpu-df-pstate-range MAX_PSTATE MIN_PSTATE  Sets max and min df-pstates
  --cpu-enable-apb                             Enables the DF p-state performance boost algorithm
  --cpu-disable-apb DF_PSTATE                  Disables the DF p-state performance boost algorithm. Input parameter is DFPstate (0-3)
  --soc-boost-limit BOOST_LIMIT                Sets the boost limit for the given socket. Input parameter is socket BOOST_LIMIT value

CPU Core Arguments:
  --core-boost-limit BOOST_LIMIT               Sets the boost limit for the given core. Input parameter is core BOOST_LIMIT value

Command Modifiers:
  --json                                       Displays output in JSON format (human readable by default).
  --csv                                        Displays output in CSV format (human readable by default).
  --file FILE                                  Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL                             Set the logging level from the possible choices:
                                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi reset --help
usage: amd-smi reset [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                     (-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]) [-G] [-c]
                     [-f] [-p] [-x] [-d] [-C] [-M] [-o] [-l]

A GPU must be specified to reset a configuration.
A reset argument must be provided; Multiple reset arguments are accepted

Reset Arguments:
  -h, --help                  show this help message and exit
  -g, --gpu GPU [GPU ...]     Select a GPU ID, BDF, or UUID from the possible choices:
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                              ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                              ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                all | Selects all devices
  -U, --cpu CPU [CPU ...]     Select a CPU ID from the possible choices:
                              ID: 0
                              ID: 1
                              ID: 2
                              ID: 3
                                all | Selects all devices
  -O, --core CORE [CORE ...]  Select a Core ID from the possible choices:
                              ID: 0 - 95
                                all  | Selects all devices
  -G, --gpureset              Reset the specified GPU
  -c, --clocks                Reset clocks and overdrive to default
  -f, --fans                  Reset fans to automatic (driver) control
  -p, --profile               Reset power profile back to default
  -x, --xgmierr               Reset XGMI error counts
  -d, --perf-determinism      Disable performance determinism
  -C, --compute-partition     Reset compute partitions on the specified GPU
  -M, --memory-partition      Reset memory partitions on the specified GPU
  -o, --power-cap             Reset power capacity limit to max capable
  -l, --run-shader SHADER_NAME  Run the shader on processor. Only CLEANER shader can be used to clean up data in LDS/GPRs

Command Modifiers:
  --json                      Displays output in JSON format (human readable by default).
  --csv                       Displays output in CSV format (human readable by default).
  --file FILE                 Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL            Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi monitor --help
usage: amd-smi monitor [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                       [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]]
                       [-w INTERVAL] [-W TIME] [-i ITERATIONS] [-p] [-t] [-u] [-m] [-n]
                       [-d] [-e] [-v] [-r] [-q]

Monitor a target device for the specified arguments.
If no arguments are provided, all arguments will be enabled.
Use the watch arguments to run continuously

Monitor Arguments:
  -h, --help                   show this help message and exit
  -g, --gpu GPU [GPU ...]      Select a GPU ID, BDF, or UUID from the possible choices:
                               ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                               ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                               ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                               ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                 all | Selects all devices
  -U, --cpu CPU [CPU ...]      Select a CPU ID from the possible choices:
                               ID: 0
                               ID: 1
                               ID: 2
                               ID: 3
                                 all | Selects all devices
  -O, --core CORE [CORE ...]   Select a Core ID from the possible choices:
                               ID: 0 - 95
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
  -e, --ecc                    Monitor ECC single bit, ECC double bit, and PCIe replay error counts
  -v, --vram-usage             Monitor memory usage in MB
  -r, --pcie                   Monitor PCIe bandwidth in Mb/s
  -q, --process                Enable Process information table below monitor output

Command Modifiers:
  --json                       Displays output in JSON format (human readable by default).
  --csv                        Displays output in CSV format (human readable by default).
  --file FILE                  Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL             Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi xgmi --help
usage: amd-smi xgmi [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                    [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]] [-m]

If no GPU is specified, returns information for all GPUs on the system.
If no xgmi argument is provided all xgmi information will be displayed.

XGMI arguments:
  -h, --help                  show this help message and exit
  -g, --gpu GPU [GPU ...]     Select a GPU ID, BDF, or UUID from the possible choices:
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 7eff74a0-0000-1000-808f-7e20764e2714
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b6ff74a0-0000-1000-80ae-7c8cefe1f084
                              ID: 2 | BDF: 0002:01:00.0 | UUID: 36ff74a0-0000-1000-8071-25d815189854
                              ID: 3 | BDF: 0003:01:00.0 | UUID: f4ff74a0-0000-1000-80c4-4c2be5e66537
                                all | Selects all devices
  -U, --cpu CPU [CPU ...]     Select a CPU ID from the possible choices:
                              ID: 0
                              ID: 1
                              ID: 2
                              ID: 3
                                all | Selects all devices
  -O, --core CORE [CORE ...]  Select a Core ID from the possible choices:
                              ID: 0 - 95
                                all  | Selects all devices
  -m, --metric                Metric XGMI information

Command Modifiers:
  --json                      Displays output in JSON format (human readable by default).
  --csv                       Displays output in CSV format (human readable by default).
  --file FILE                 Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL            Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

### Example output from amd-smi static

Here is some example output from the tool:

```bash
~$ amd-smi static
CPU: 0
    SMU:
        FW_VERSION: 85.90.0
    INTERFACE_VERSION:
        PROTO VERSION: 6

CPU: 1
    SMU:
        FW_VERSION: 85.90.0
    INTERFACE_VERSION:
        PROTO VERSION: 6

CPU: 2
    SMU:
        FW_VERSION: 85.90.0
    INTERFACE_VERSION:
        PROTO VERSION: 6

CPU: 3
    SMU:
        FW_VERSION: 85.90.0
    INTERFACE_VERSION:
        PROTO VERSION: 6


GPU: 0
    ASIC:
        MARKET_NAME: MI300A
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a0
        REV_ID: 0x00
        ASIC_SERIAL: 0x7E8F7E20764E2714
        OAM_ID: 0
    BUS:
        BDF: 0000:01:00.0
        MAX_PCIE_WIDTH: 16
        MAX_PCIE_SPEED: 32 GT/s
        PCIE_INTERFACE_VERSION: Gen 5
        SLOT_TYPE: PCIE
    VBIOS:
        NAME: N/A
        BUILD_DATE: N/A
        PART_NUMBER: N/A
        VERSION: N/A
    LIMIT:
        MAX_POWER: 550 W
        MIN_POWER: 0 W
        SOCKET_POWER: 550 W
        SLOWDOWN_EDGE_TEMPERATURE: N/A
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SLOWDOWN_VRAM_TEMPERATURE: 105 °C
        SHUTDOWN_EDGE_TEMPERATURE: N/A
        SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SHUTDOWN_VRAM_TEMPERATURE: 115 °C
    DRIVER:
        NAME: amdgpu
        VERSION: 6.9.0-rc5+
    BOARD:
        MODEL_NUMBER: N/A
        PRODUCT_SERIAL: N/A
        FRU_ID: N/A
        PRODUCT_NAME: Aqua Vanjaram [Instinct MI300A]
        MANUFACTURER_NAME: Advanced Micro Devices, Inc. [AMD/ATI]
    RAS:
        EEPROM_VERSION: 0x0
        PARITY_SCHEMA: DISABLED
        SINGLE_BIT_SCHEMA: DISABLED
        DOUBLE_BIT_SCHEMA: DISABLED
        POISON_SCHEMA: ENABLED
        ECC_BLOCK_STATE:
            UMC: DISABLED
            SDMA: ENABLED
            GFX: ENABLED
            MMHUB: ENABLED
            ATHUB: DISABLED
            PCIE_BIF: DISABLED
            HDP: DISABLED
            XGMI_WAFL: DISABLED
            DF: DISABLED
            SMN: DISABLED
            SEM: DISABLED
            MP0: DISABLED
            MP1: DISABLED
            FUSE: DISABLED
            MCA: DISABLED
            VCN: DISABLED
            JPEG: DISABLED
            IH: DISABLED
            MPIO: DISABLED
    PARTITION:
        COMPUTE_PARTITION: SPX
        MEMORY_PARTITION: NPS1
    SOC_PSTATE:
        NUM_SUPPORTED: 4
        CURRENT_ID: 1
        POLICIES:
            POLICY_ID: 0
            POLICY_DESCRIPTION: pstate_default
            POLICY_ID: 1
            POLICY_DESCRIPTION: soc_pstate_0
            POLICY_ID: 2
            POLICY_DESCRIPTION: soc_pstate_1
            POLICY_ID: 3
            POLICY_DESCRIPTION: soc_pstate_2
    XGMI_PLPD:
        NUM_SUPPORTED: 3
        CURRENT_ID: 1
        PLPDS:
            POLICY_ID: 0
            POLICY_DESCRIPTION: plpd_disallow
            POLICY_ID: 1
            POLICY_DESCRIPTION: plpd_default
            POLICY_ID: 2
            POLICY_DESCRIPTION: plpd_optimized
    PROCESS_ISOLATION: N/A
    NUMA:
        NODE: 0
        AFFINITY: 0
    VRAM:
        TYPE: HBM
        VENDOR: N/A
        SIZE: 64289 MB
    CACHE_INFO:
        CACHE_0:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 32 KB
            CACHE_LEVEL: 1
            MAX_NUM_CU_SHARED: 2
            NUM_CACHE_INSTANCE: 348
        CACHE_1:
            CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
            CACHE_SIZE: 64 KB
            CACHE_LEVEL: 1
            MAX_NUM_CU_SHARED: 2
            NUM_CACHE_INSTANCE: 120
        CACHE_2:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 4096 KB
            CACHE_LEVEL: 2
            MAX_NUM_CU_SHARED: 228
            NUM_CACHE_INSTANCE: 1
        CACHE_3:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 262144 KB
            CACHE_LEVEL: 3
            MAX_NUM_CU_SHARED: 228
            NUM_CACHE_INSTANCE: 1

GPU: 1
    ASIC:
        MARKET_NAME: MI300A
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a0
        REV_ID: 0x00
        ASIC_SERIAL: 0xB6AE7C8CEFE1F084
        OAM_ID: 1
    BUS:
        BDF: 0001:01:00.0
        MAX_PCIE_WIDTH: 16
        MAX_PCIE_SPEED: 32 GT/s
        PCIE_INTERFACE_VERSION: Gen 5
        SLOT_TYPE: PCIE
    VBIOS:
        NAME: N/A
        BUILD_DATE: N/A
        PART_NUMBER: N/A
        VERSION: N/A
    LIMIT:
        MAX_POWER: 550 W
        MIN_POWER: 0 W
        SOCKET_POWER: 550 W
        SLOWDOWN_EDGE_TEMPERATURE: N/A
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SLOWDOWN_VRAM_TEMPERATURE: 105 °C
        SHUTDOWN_EDGE_TEMPERATURE: N/A
        SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SHUTDOWN_VRAM_TEMPERATURE: 115 °C
    DRIVER:
        NAME: amdgpu
        VERSION: 6.9.0-rc5+
    BOARD:
        MODEL_NUMBER: N/A
        PRODUCT_SERIAL: N/A
        FRU_ID: N/A
        PRODUCT_NAME: Aqua Vanjaram [Instinct MI300A]
        MANUFACTURER_NAME: Advanced Micro Devices, Inc. [AMD/ATI]
    RAS:
        EEPROM_VERSION: 0x0
        PARITY_SCHEMA: DISABLED
        SINGLE_BIT_SCHEMA: DISABLED
        DOUBLE_BIT_SCHEMA: DISABLED
        POISON_SCHEMA: ENABLED
        ECC_BLOCK_STATE:
            UMC: DISABLED
            SDMA: ENABLED
            GFX: ENABLED
            MMHUB: ENABLED
            ATHUB: DISABLED
            PCIE_BIF: DISABLED
            HDP: DISABLED
            XGMI_WAFL: DISABLED
            DF: DISABLED
            SMN: DISABLED
            SEM: DISABLED
            MP0: DISABLED
            MP1: DISABLED
            FUSE: DISABLED
            MCA: DISABLED
            VCN: DISABLED
            JPEG: DISABLED
            IH: DISABLED
            MPIO: DISABLED
    PARTITION:
        COMPUTE_PARTITION: SPX
        MEMORY_PARTITION: NPS1
    SOC_PSTATE:
        NUM_SUPPORTED: 4
        CURRENT_ID: 1
        POLICIES:
            POLICY_ID: 0
            POLICY_DESCRIPTION: pstate_default
            POLICY_ID: 1
            POLICY_DESCRIPTION: soc_pstate_0
            POLICY_ID: 2
            POLICY_DESCRIPTION: soc_pstate_1
            POLICY_ID: 3
            POLICY_DESCRIPTION: soc_pstate_2
    XGMI_PLPD:
        NUM_SUPPORTED: 3
        CURRENT_ID: 1
        PLPDS:
            POLICY_ID: 0
            POLICY_DESCRIPTION: plpd_disallow
            POLICY_ID: 1
            POLICY_DESCRIPTION: plpd_default
            POLICY_ID: 2
            POLICY_DESCRIPTION: plpd_optimized
    PROCESS_ISOLATION: N/A
    NUMA:
        NODE: 1
        AFFINITY: 1
    VRAM:
        TYPE: HBM
        VENDOR: N/A
        SIZE: 64289 MB
    CACHE_INFO:
        CACHE_0:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 32 KB
            CACHE_LEVEL: 1
            MAX_NUM_CU_SHARED: 2
            NUM_CACHE_INSTANCE: 348
        CACHE_1:
            CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
            CACHE_SIZE: 64 KB
            CACHE_LEVEL: 1
            MAX_NUM_CU_SHARED: 2
            NUM_CACHE_INSTANCE: 120
        CACHE_2:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 4096 KB
            CACHE_LEVEL: 2
            MAX_NUM_CU_SHARED: 228
            NUM_CACHE_INSTANCE: 1
        CACHE_3:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 262144 KB
            CACHE_LEVEL: 3
            MAX_NUM_CU_SHARED: 228
            NUM_CACHE_INSTANCE: 1

GPU: 2
    ASIC:
        MARKET_NAME: MI300A
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a0
        REV_ID: 0x00
        ASIC_SERIAL: 0x367125D815189854
        OAM_ID: 2
    BUS:
        BDF: 0002:01:00.0
        MAX_PCIE_WIDTH: 16
        MAX_PCIE_SPEED: 32 GT/s
        PCIE_INTERFACE_VERSION: Gen 5
        SLOT_TYPE: PCIE
    VBIOS:
        NAME: N/A
        BUILD_DATE: N/A
        PART_NUMBER: N/A
        VERSION: N/A
    LIMIT:
        MAX_POWER: 550 W
        MIN_POWER: 0 W
        SOCKET_POWER: 550 W
        SLOWDOWN_EDGE_TEMPERATURE: N/A
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SLOWDOWN_VRAM_TEMPERATURE: 105 °C
        SHUTDOWN_EDGE_TEMPERATURE: N/A
        SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SHUTDOWN_VRAM_TEMPERATURE: 115 °C
    DRIVER:
        NAME: amdgpu
        VERSION: 6.9.0-rc5+
    BOARD:
        MODEL_NUMBER: N/A
        PRODUCT_SERIAL: N/A
        FRU_ID: N/A
        PRODUCT_NAME: Aqua Vanjaram [Instinct MI300A]
        MANUFACTURER_NAME: Advanced Micro Devices, Inc. [AMD/ATI]
    RAS:
        EEPROM_VERSION: 0x0
        PARITY_SCHEMA: DISABLED
        SINGLE_BIT_SCHEMA: DISABLED
        DOUBLE_BIT_SCHEMA: DISABLED
        POISON_SCHEMA: ENABLED
        ECC_BLOCK_STATE:
            UMC: DISABLED
            SDMA: ENABLED
            GFX: ENABLED
            MMHUB: ENABLED
            ATHUB: DISABLED
            PCIE_BIF: DISABLED
            HDP: DISABLED
            XGMI_WAFL: DISABLED
            DF: DISABLED
            SMN: DISABLED
            SEM: DISABLED
            MP0: DISABLED
            MP1: DISABLED
            FUSE: DISABLED
            MCA: DISABLED
            VCN: DISABLED
            JPEG: DISABLED
            IH: DISABLED
            MPIO: DISABLED
    PARTITION:
        COMPUTE_PARTITION: SPX
        MEMORY_PARTITION: NPS1
    SOC_PSTATE:
        NUM_SUPPORTED: 4
        CURRENT_ID: 1
        POLICIES:
            POLICY_ID: 0
            POLICY_DESCRIPTION: pstate_default
            POLICY_ID: 1
            POLICY_DESCRIPTION: soc_pstate_0
            POLICY_ID: 2
            POLICY_DESCRIPTION: soc_pstate_1
            POLICY_ID: 3
            POLICY_DESCRIPTION: soc_pstate_2
    XGMI_PLPD:
        NUM_SUPPORTED: 3
        CURRENT_ID: 1
        PLPDS:
            POLICY_ID: 0
            POLICY_DESCRIPTION: plpd_disallow
            POLICY_ID: 1
            POLICY_DESCRIPTION: plpd_default
            POLICY_ID: 2
            POLICY_DESCRIPTION: plpd_optimized
    PROCESS_ISOLATION: N/A
    NUMA:
        NODE: 2
        AFFINITY: 2
    VRAM:
        TYPE: HBM
        VENDOR: N/A
        SIZE: 64289 MB
    CACHE_INFO:
        CACHE_0:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 32 KB
            CACHE_LEVEL: 1
            MAX_NUM_CU_SHARED: 2
            NUM_CACHE_INSTANCE: 348
        CACHE_1:
            CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
            CACHE_SIZE: 64 KB
            CACHE_LEVEL: 1
            MAX_NUM_CU_SHARED: 2
            NUM_CACHE_INSTANCE: 120
        CACHE_2:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 4096 KB
            CACHE_LEVEL: 2
            MAX_NUM_CU_SHARED: 228
            NUM_CACHE_INSTANCE: 1
        CACHE_3:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 262144 KB
            CACHE_LEVEL: 3
            MAX_NUM_CU_SHARED: 228
            NUM_CACHE_INSTANCE: 1

GPU: 3
    ASIC:
        MARKET_NAME: MI300A
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a0
        REV_ID: 0x00
        ASIC_SERIAL: 0xF4C44C2BE5E66537
        OAM_ID: 3
    BUS:
        BDF: 0003:01:00.0
        MAX_PCIE_WIDTH: 16
        MAX_PCIE_SPEED: 32 GT/s
        PCIE_INTERFACE_VERSION: Gen 5
        SLOT_TYPE: PCIE
    VBIOS:
        NAME: N/A
        BUILD_DATE: N/A
        PART_NUMBER: N/A
        VERSION: N/A
    LIMIT:
        MAX_POWER: 550 W
        MIN_POWER: 0 W
        SOCKET_POWER: 550 W
        SLOWDOWN_EDGE_TEMPERATURE: N/A
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SLOWDOWN_VRAM_TEMPERATURE: 105 °C
        SHUTDOWN_EDGE_TEMPERATURE: N/A
        SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SHUTDOWN_VRAM_TEMPERATURE: 115 °C
    DRIVER:
        NAME: amdgpu
        VERSION: 6.9.0-rc5+
    BOARD:
        MODEL_NUMBER: N/A
        PRODUCT_SERIAL: N/A
        FRU_ID: N/A
        PRODUCT_NAME: Aqua Vanjaram [Instinct MI300A]
        MANUFACTURER_NAME: Advanced Micro Devices, Inc. [AMD/ATI]
    RAS:
        EEPROM_VERSION: 0x0
        PARITY_SCHEMA: DISABLED
        SINGLE_BIT_SCHEMA: DISABLED
        DOUBLE_BIT_SCHEMA: DISABLED
        POISON_SCHEMA: ENABLED
        ECC_BLOCK_STATE:
            UMC: DISABLED
            SDMA: ENABLED
            GFX: ENABLED
            MMHUB: ENABLED
            ATHUB: DISABLED
            PCIE_BIF: DISABLED
            HDP: DISABLED
            XGMI_WAFL: DISABLED
            DF: DISABLED
            SMN: DISABLED
            SEM: DISABLED
            MP0: DISABLED
            MP1: DISABLED
            FUSE: DISABLED
            MCA: DISABLED
            VCN: DISABLED
            JPEG: DISABLED
            IH: DISABLED
            MPIO: DISABLED
    PARTITION:
        COMPUTE_PARTITION: SPX
        MEMORY_PARTITION: NPS1
    SOC_PSTATE:
        NUM_SUPPORTED: 4
        CURRENT_ID: 1
        POLICIES:
            POLICY_ID: 0
            POLICY_DESCRIPTION: pstate_default
            POLICY_ID: 1
            POLICY_DESCRIPTION: soc_pstate_0
            POLICY_ID: 2
            POLICY_DESCRIPTION: soc_pstate_1
            POLICY_ID: 3
            POLICY_DESCRIPTION: soc_pstate_2
    XGMI_PLPD:
        NUM_SUPPORTED: 3
        CURRENT_ID: 1
        PLPDS:
            POLICY_ID: 0
            POLICY_DESCRIPTION: plpd_disallow
            POLICY_ID: 1
            POLICY_DESCRIPTION: plpd_default
            POLICY_ID: 2
            POLICY_DESCRIPTION: plpd_optimized
    PROCESS_ISOLATION: N/A
    NUMA:
        NODE: 3
        AFFINITY: 3
    VRAM:
        TYPE: HBM
        VENDOR: N/A
        SIZE: 64289 MB
    CACHE_INFO:
        CACHE_0:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 32 KB
            CACHE_LEVEL: 1
            MAX_NUM_CU_SHARED: 2
            NUM_CACHE_INSTANCE: 348
        CACHE_1:
            CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
            CACHE_SIZE: 64 KB
            CACHE_LEVEL: 1
            MAX_NUM_CU_SHARED: 2
            NUM_CACHE_INSTANCE: 120
        CACHE_2:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 4096 KB
            CACHE_LEVEL: 2
            MAX_NUM_CU_SHARED: 228
            NUM_CACHE_INSTANCE: 1
        CACHE_3:
            CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
            CACHE_SIZE: 262144 KB
            CACHE_LEVEL: 3
            MAX_NUM_CU_SHARED: 228
            NUM_CACHE_INSTANCE: 1

```

## Disclaimer

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2023 Advanced Micro Devices, Inc. All rights reserved.
