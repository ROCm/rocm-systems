# AMD SMI CLI Tool

This tool acts as a command line interface for manipulating
and monitoring the amdgpu kernel, and is intended to replace
and deprecate the existing rocm_smi CLI tool & gpuv-smi tool.
It uses Ctypes to call the amd_smi_lib API.
Recommended: At least one AMD GPU with AMD driver installed

## Install CLI Tool and Python Library

### Requirements

* python 3.6.8+ 64-bit
* amdgpu driver must be loaded for amdsmi_init() to pass

### Installation

* Install amdgpu driver
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

amd-smi will report the version and current platform detected when running the command without arguments:

``` bash
~$ amd-smi
usage: amd-smi [-h]  ...

AMD System Management Interface | Version: 24.2.0.0 | ROCm version: 6.1.0 | Platform: Linux
Baremetal

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
    monitor           Monitor metrics for target devices
```

More detailed verison information is available from `amd-smi version`

Each command will have detailed information via `amd-smi [command] --help`

## Commands

For convenience, here is the help output for each command

``` bash
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
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                              ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                              ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
usage: amd-smi static [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                      [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]] [-a] [-b]
                      [-V] [-d] [-v] [-c] [-B] [-s] [-i] [-r] [-p] [-l] [-u]

If no GPU is specified, returns static information for all GPUs on the system.
If no static argument is provided, all static information will be displayed.

Static Arguments:
  -h, --help                  show this help message and exit
  -g, --gpu GPU [GPU ...]     Select a GPU ID, BDF, or UUID from the possible choices:
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                              ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                              ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
  -a, --asic                  All asic information
  -b, --bus                   All bus information
  -V, --vbios                 All video bios information (if available)
  -d, --driver                Displays driver version
  -v, --vram                  All vram information
  -c, --cache                 All cache information
  -B, --board                 All board information
  -r, --ras                   Displays RAS features information
  -p, --partition             Partition information
  -l, --limit                 All limit metric values (i.e. power and thermal limits)
  -u, --numa                  All numa node information

CPU Option<s>:
  -s, --smu                   All SMU FW information
  -i, --interface_ver         Displays hsmp interface version

Command Modifiers:
  --json                      Displays output in JSON format (human readable by default).
  --csv                       Displays output in CSV format (human readable by default).
  --file FILE                 Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL            Set the logging level from the possible choices:
                                DEBUG, INFO, WARNING, ERROR, CRITICAL
```

``` bash
~$ amd-smi firmware --help
usage: amd-smi firmware [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                        [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]] [-f]

If no GPU is specified, return firmware information for all GPUs on the system.

Firmware Arguments:
  -h, --help                   show this help message and exit
  -g, --gpu GPU [GPU ...]      Select a GPU ID, BDF, or UUID from the possible choices:
                               ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                               ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                               ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                               ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                              ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                              ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
usage: amd-smi metric [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                      [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]]
                      [-w INTERVAL] [-W TIME] [-i ITERATIONS] [-m] [-u] [-p] [-c] [-t]
                      [-e] [-P] [-k] [-f] [-C] [-o] [-l] [-x] [-E] [--cpu_power_metrics]
                      [--cpu_prochot] [--cpu_freq_metrics] [--cpu_c0_res]
                      [--cpu_lclk_dpm_level NBIOID] [--cpu_pwr_svi_telemtry_rails]
                      [--cpu_io_bandwidth IO_BW LINKID_NAME]
                      [--cpu_xgmi_bandwidth XGMI_BW LINKID_NAME] [--cpu_enable_apb]
                      [--cpu_disable_apb DF_PSTATE] [--set_cpu_pow_limit POW_LIMIT]
                      [--set_cpu_xgmi_link_width MIN_WIDTH MAX_WIDTH]
                      [--set_cpu_lclk_dpm_level NBIOID MIN_DPM MAX_DPM]
                      [--core_boost_limit] [--core_curr_active_freq_core_limit]
                      [--set_soc_boost_limit BOOST_LIMIT]
                      [--set_core_boost_limit BOOST_LIMIT] [--cpu_metrics_ver]
                      [--cpu_metrics_table] [--core_energy] [--socket_energy]
                      [--set_cpu_pwr_eff_mode MODE] [--cpu_ddr_bandwidth] [--cpu_temp]
                      [--cpu_dimm_temp_range_rate DIMM_ADDR]
                      [--cpu_dimm_pow_conumption DIMM_ADDR]
                      [--cpu_dimm_thermal_sensor DIMM_ADDR]
                      [--set_cpu_gmi3_link_width MIN_LW MAX_LW]
                      [--set_cpu_pcie_lnk_rate LINK_RATE]
                      [--set_cpu_df_pstate_range MAX_PSTATE MIN_PSTATE]

If no GPU is specified, returns metric information for all GPUs on the system.
If no metric argument is provided all metric information will be displayed.

Metric arguments:
  -h, --help                                       show this help message and exit
  -g, --gpu GPU [GPU ...]                          Select a GPU ID, BDF, or UUID from the possible choices:
                                                   ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                                                   ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                                                   ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                                                   ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
                                                     all | Selects all devices
  -U, --cpu CPU [CPU ...]                          Select a CPU ID from the possible choices:
                                                   ID: 0
                                                   ID: 1
                                                   ID: 2
                                                   ID: 3
                                                     all | Selects all devices
  -O, --core CORE [CORE ...]                       Select a Core ID from the possible choices:
                                                   ID: 0 - 95
                                                     all  | Selects all devices
  -w, --watch INTERVAL                             Reprint the command in a loop of INTERVAL seconds
  -W, --watch_time TIME                            The total TIME to watch the given command
  -i, --iterations ITERATIONS                      Total number of ITERATIONS to loop on the given command
  -m, --mem-usage                                  Memory usage per block
  -u, --usage                                      Displays engine usage information
  -p, --power                                      Current power usage
  -c, --clock                                      Average, max, and current clock frequencies
  -t, --temperature                                Current temperatures
  -e, --ecc                                        Total number of ECC errors
  -P, --pcie                                       Current PCIe speed, width, and replay count
  -k, --ecc-block                                  Number of ECC errors per block
  -f, --fan                                        Current fan speed
  -C, --voltage-curve                              Display voltage curve
  -o, --overdrive                                  Current GPU clock overdrive level
  -l, --perf-level                                 Current DPM performance level
  -x, --xgmi-err                                   XGMI error information since last read
  -E, --energy                                     Amount of energy consumed

CPU Option<s>:
  --cpu_power_metrics                              Cpu power metrics
  --cpu_prochot                                    Displays prochot status
  --cpu_freq_metrics                               Displays currentFclkMemclk frequencies and cclk frequency limit
  --cpu_c0_res                                     Displays C0 residency
  --cpu_lclk_dpm_level NBIOID                      Displays lclk dpm level range. Requires socket ID and nbio id as inputs
  --cpu_pwr_svi_telemtry_rails                     Displays svi based telemetry for all rails
  --cpu_io_bandwidth IO_BW LINKID_NAME             Displays current IO bandwidth for the selected CPU.
                                                    input parameters are bandwidth type(1) and link ID encodings
                                                    i.e. P2, P3, G0 - G7
  --cpu_xgmi_bandwidth XGMI_BW LINKID_NAME         Displays current XGMI bandwidth for the selected CPU
                                                    input parameters are bandwidth type(1,2,4) and link ID encodings
                                                    i.e. P2, P3, G0 - G7
  --cpu_enable_apb                                 Enables the DF p-state performance boost algorithm
  --cpu_disable_apb DF_PSTATE                      Disables the DF p-state performance boost algorithm.
  --core_boost_limit                               Get booslimit for the selected cores
  --core_curr_active_freq_core_limit               Get Current CCLK limit set per Core
  --cpu_metrics_ver                                Displays metrics table version
  --cpu_metrics_table                              Displays metric table
  --core_energy                                    Displays core energy for the selected core
  --socket_energy                                  Displays socket energy for the selected socket
  --cpu_ddr_bandwidth                              Displays per socket max ddr bw, current utilized bw and current utilized ddr bw in percentage
  --cpu_temp                                       Displays cpu socket temperature
  --cpu_dimm_temp_range_rate DIMM_ADDR             Displays dimm temperature range and refresh rate
  --cpu_dimm_pow_conumption DIMM_ADDR              Displays dimm power consumption
  --cpu_dimm_thermal_sensor DIMM_ADDR              Displays dimm thermal sensor

Set Options<s>:
  --set_cpu_pow_limit POW_LIMIT                    Set power limit for the given socket. Input parameter is power limit value.
  --set_cpu_xgmi_link_width MIN_WIDTH MAX_WIDTH    Set max and Min linkwidth. Input parameters are min and max link width values
  --set_cpu_lclk_dpm_level NBIOID MIN_DPM MAX_DPM  Sets the max and min dpm level on a given NBIO. Inpur parameters are die_index, min dpm, max dpm.
  --set_soc_boost_limit BOOST_LIMIT                Sets the boost limit for the given socket. Input parameter is socket limit value
  --set_core_boost_limit BOOST_LIMIT               Sets the boost limit for the given core. Input parameter is core limit value
  --set_cpu_pwr_eff_mode MODE                      Sets the power efficency mode policy. Input parameter is mode.
  --set_cpu_gmi3_link_width MIN_LW MAX_LW          Sets max and min gmi3 link width range
  --set_cpu_pcie_lnk_rate LINK_RATE                Sets pcie link rate
  --set_cpu_df_pstate_range MAX_PSTATE MIN_PSTATE  Sets max and min df-pstates

Command Modifiers:
  --json                                           Displays output in JSON format (human readable by default).
  --csv                                            Displays output in CSV format (human readable by default).
  --file FILE                                      Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL                                 Set the logging level from the possible choices:
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
                               ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                               ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                               ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                               ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                              ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                              ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                              ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                              ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
usage: amd-smi set [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                   (-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]) [-f %]
                   [-l LEVEL] [-P SETPROFILE] [-d SCLKMAX] [-C PARTITION] [-M PARTITION]
                   [-o WATTS]

A GPU must be specified to set a configuration.
A set argument must be provided; Multiple set arguments are accepted

Set Arguments:
  -h, --help                         show this help message and exit
  -g, --gpu GPU [GPU ...]            Select a GPU ID, BDF, or UUID from the possible choices:
                                     ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                                     ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                                     ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                                     ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
                                       all | Selects all devices
  -U, --cpu CPU [CPU ...]            Select a CPU ID from the possible choices:
                                     ID: 0
                                     ID: 1
                                     ID: 2
                                     ID: 3
                                       all | Selects all devices
  -O, --core CORE [CORE ...]         Select a Core ID from the possible choices:
                                     ID: 0 - 95
                                       all  | Selects all devices
  -f, --fan %                        Set GPU fan speed (0-255 or 0-100%)
  -l, --perf-level LEVEL             Set performance level
  -P, --profile SETPROFILE           Set power profile level (#) or a quoted string of custom profile attributes
  -d, --perf-determinism SCLKMAX     Set GPU clock frequency limit and performance level to determinism to get minimal performance variation
  -C, --compute-partition PARTITION  Set one of the following the compute partition modes:
                                        CPX, SPX, DPX, TPX, QPX
  -M, --memory-partition PARTITION   Set one of the following the memory partition modes:
                                        NPS1, NPS2, NPS4, NPS8
  -o, --power-cap WATTS              Set power capacity limit

Command Modifiers:
  --json                             Displays output in JSON format (human readable by default).
  --csv                              Displays output in CSV format (human readable by default).
  --file FILE                        Saves output into a file on the provided path (stdout by default).
  --loglevel LEVEL                   Set the logging level from the possible choices:
                                        DEBUG, INFO, WARNING, ERROR, CRITICAL
```

```bash
~$ amd-smi reset --help
usage: amd-smi reset [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                     (-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]) [-G] [-c]
                     [-f] [-p] [-x] [-d] [-C] [-M] [-o]

A GPU must be specified to reset a configuration.
A reset argument must be provided; Multiple reset arguments are accepted

Reset Arguments:
  -h, --help                  show this help message and exit
  -g, --gpu GPU [GPU ...]     Select a GPU ID, BDF, or UUID from the possible choices:
                              ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                              ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                              ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                              ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
    FW_VERSION: 85:81:0
INTERFACE_VERSION:
    PROTO VERSION: 6

CPU: 1
SMU:
    FW_VERSION: 85:81:0
INTERFACE_VERSION:
    PROTO VERSION: 6

CPU: 2
SMU:
    FW_VERSION: 85:81:0
INTERFACE_VERSION:
    PROTO VERSION: 6

CPU: 3
SMU:
    FW_VERSION: 85:81:0
INTERFACE_VERSION:
    PROTO VERSION: 6


GPU: 0
ASIC:
    MARKET_NAME: MI300A
    VENDOR_ID: 0x1002
    VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
    SUBVENDOR_ID: 0
    DEVICE_ID: 0x74a0
    REV_ID: 0x0
    ASIC_SERIAL: 0x71660a3c71d5f817
    OAM_ID: 0
BUS:
    BDF: 0000:01:00.0
    MAX_PCIE_SPEED: 32 GT/s
    MAX_PCIE_LANES: 16
    PCIE_INTERFACE_VERSION: Gen 5
    SLOT_TYPE: PCIE
VBIOS:
    NAME: N/A
    BUILD_DATE: N/A
    PART_NUMBER: N/A
    VERSION: N/A
BOARD:
    MODEL_NUMBER: N/A
    PRODUCT_SERIAL: N/A
    FRU_ID: N/A
    MANUFACTURER_NAME: N/A
    PRODUCT_NAME: N/A
LIMIT:
    MAX_POWER: 550 W
    CURRENT_POWER: 0 W
    SLOWDOWN_EDGE_TEMPERATURE: N/A
    SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
    SLOWDOWN_VRAM_TEMPERATURE: 95 °C
    SHUTDOWN_EDGE_TEMPERATURE: N/A
    SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
    SHUTDOWN_VRAM_TEMPERATURE: 105 °C
DRIVER:
    DRIVER_NAME: amdgpu
    DRIVER_VERSION: 6.5.2
VRAM:
    VRAM_TYPE: HBM
    VRAM_VENDOR: HYNIX
    VRAM_SIZE_MB: 96432 MB
CACHE:
    CACHE_0:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 32 KB
        CACHE_LEVEL: 1
        MAX_NUM_CU_SHARED: 2
        NUM_CACHE_INSTANCE: 464
    CACHE_1:
        CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
        CACHE_SIZE: 64 KB
        CACHE_LEVEL: 1
        MAX_NUM_CU_SHARED: 2
        NUM_CACHE_INSTANCE: 160
    CACHE_2:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 32768 KB
        CACHE_LEVEL: 2
        MAX_NUM_CU_SHARED: 304
        NUM_CACHE_INSTANCE: 1
    CACHE_3:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 262144 KB
        CACHE_LEVEL: 3
        MAX_NUM_CU_SHARED: 304
        NUM_CACHE_INSTANCE: 1

RAS:
    EEPROM_VERSION: 0x0
    PARITY_SCHEMA: DISABLED
    SINGLE_BIT_SCHEMA: DISABLED
    DOUBLE_BIT_SCHEMA: DISABLED
    POISON_SCHEMA: ENABLED
    ECC_BLOCK_STATE:
        BLOCK: UMC
        STATUS: DISABLED
        BLOCK: SDMA
        STATUS: ENABLED
        BLOCK: GFX
        STATUS: ENABLED
        BLOCK: MMHUB
        STATUS: ENABLED
        BLOCK: ATHUB
        STATUS: DISABLED
        BLOCK: PCIE_BIF
        STATUS: DISABLED
        BLOCK: HDP
        STATUS: DISABLED
        BLOCK: XGMI_WAFL
        STATUS: DISABLED
        BLOCK: DF
        STATUS: DISABLED
        BLOCK: SMN
        STATUS: DISABLED
        BLOCK: SEM
        STATUS: DISABLED
        BLOCK: MP0
        STATUS: DISABLED
        BLOCK: MP1
        STATUS: DISABLED
        BLOCK: FUSE
        STATUS: DISABLED
PARTITION:
    COMPUTE_PARTITION: SPX
    MEMORY_PARTITION: NPS1
NUMA:
    NODE: 0
    AFFINITY: 0

GPU: 1
ASIC:
    MARKET_NAME: MI300A
    VENDOR_ID: 0x1002
    VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
    SUBVENDOR_ID: 0
    DEVICE_ID: 0x74a0
    REV_ID: 0x0
    ASIC_SERIAL: 0xb4b2fa0be8628b1a
    OAM_ID: 1
BUS:
    BDF: 0001:01:00.0
    MAX_PCIE_SPEED: 32 GT/s
    MAX_PCIE_LANES: 16
    PCIE_INTERFACE_VERSION: Gen 5
    SLOT_TYPE: PCIE
VBIOS:
    NAME: N/A
    BUILD_DATE: N/A
    PART_NUMBER: N/A
    VERSION: N/A
BOARD:
    MODEL_NUMBER: N/A
    PRODUCT_SERIAL: N/A
    FRU_ID: N/A
    MANUFACTURER_NAME: N/A
    PRODUCT_NAME: N/A
LIMIT:
    MAX_POWER: 550 W
    CURRENT_POWER: 0 W
    SLOWDOWN_EDGE_TEMPERATURE: N/A
    SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
    SLOWDOWN_VRAM_TEMPERATURE: 95 °C
    SHUTDOWN_EDGE_TEMPERATURE: N/A
    SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
    SHUTDOWN_VRAM_TEMPERATURE: 105 °C
DRIVER:
    DRIVER_NAME: amdgpu
    DRIVER_VERSION: 6.5.2
VRAM:
    VRAM_TYPE: HBM
    VRAM_VENDOR: HYNIX
    VRAM_SIZE_MB: 96432 MB
CACHE:
 CACHE:
    CACHE_0:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 32 KB
        CACHE_LEVEL: 1
        MAX_NUM_CU_SHARED: 2
        NUM_CACHE_INSTANCE: 464
    CACHE_1:
        CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
        CACHE_SIZE: 64 KB
        CACHE_LEVEL: 1
        MAX_NUM_CU_SHARED: 2
        NUM_CACHE_INSTANCE: 160
    CACHE_2:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 32768 KB
        CACHE_LEVEL: 2
        MAX_NUM_CU_SHARED: 304
        NUM_CACHE_INSTANCE: 1
    CACHE_3:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 262144 KB
        CACHE_LEVEL: 3
        MAX_NUM_CU_SHARED: 304
        NUM_CACHE_INSTANCE: 1
RAS:
    EEPROM_VERSION: 0x0
    PARITY_SCHEMA: DISABLED
    SINGLE_BIT_SCHEMA: DISABLED
    DOUBLE_BIT_SCHEMA: DISABLED
    POISON_SCHEMA: ENABLED
    ECC_BLOCK_STATE:
        BLOCK: UMC
        STATUS: DISABLED
        BLOCK: SDMA
        STATUS: ENABLED
        BLOCK: GFX
        STATUS: ENABLED
        BLOCK: MMHUB
        STATUS: ENABLED
        BLOCK: ATHUB
        STATUS: DISABLED
        BLOCK: PCIE_BIF
        STATUS: DISABLED
        BLOCK: HDP
        STATUS: DISABLED
        BLOCK: XGMI_WAFL
        STATUS: DISABLED
        BLOCK: DF
        STATUS: DISABLED
        BLOCK: SMN
        STATUS: DISABLED
        BLOCK: SEM
        STATUS: DISABLED
        BLOCK: MP0
        STATUS: DISABLED
        BLOCK: MP1
        STATUS: DISABLED
        BLOCK: FUSE
        STATUS: DISABLED
PARTITION:
    COMPUTE_PARTITION: SPX
    MEMORY_PARTITION: NPS1
NUMA:
    NODE: 1
    AFFINITY: 1

GPU: 2
ASIC:
    MARKET_NAME: MI300A
    VENDOR_ID: 0x1002
    VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
    SUBVENDOR_ID: 0
    DEVICE_ID: 0x74a0
    REV_ID: 0x0
    ASIC_SERIAL: 0xa9073066a98ba4a6
    OAM_ID: 2
BUS:
    BDF: 0002:01:00.0
    MAX_PCIE_SPEED: 32 GT/s
    MAX_PCIE_LANES: 16
    PCIE_INTERFACE_VERSION: Gen 5
    SLOT_TYPE: PCIE
VBIOS:
    NAME: N/A
    BUILD_DATE: N/A
    PART_NUMBER: N/A
    VERSION: N/A
BOARD:
    MODEL_NUMBER: N/A
    PRODUCT_SERIAL: N/A
    FRU_ID: N/A
    MANUFACTURER_NAME: N/A
    PRODUCT_NAME: N/A
LIMIT:
    MAX_POWER: 550 W
    CURRENT_POWER: 0 W
    SLOWDOWN_EDGE_TEMPERATURE: N/A
    SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
    SLOWDOWN_VRAM_TEMPERATURE: 95 °C
    SHUTDOWN_EDGE_TEMPERATURE: N/A
    SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
    SHUTDOWN_VRAM_TEMPERATURE: 105 °C
DRIVER:
    DRIVER_NAME: amdgpu
    DRIVER_VERSION: 6.5.2
VRAM:
    VRAM_TYPE: HBM
    VRAM_VENDOR: HYNIX
    VRAM_SIZE_MB: 96432 MB
CACHE:
 CACHE:
    CACHE_0:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 32 KB
        CACHE_LEVEL: 1
        MAX_NUM_CU_SHARED: 2
        NUM_CACHE_INSTANCE: 464
    CACHE_1:
        CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
        CACHE_SIZE: 64 KB
        CACHE_LEVEL: 1
        MAX_NUM_CU_SHARED: 2
        NUM_CACHE_INSTANCE: 160
    CACHE_2:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 32768 KB
        CACHE_LEVEL: 2
        MAX_NUM_CU_SHARED: 304
        NUM_CACHE_INSTANCE: 1
    CACHE_3:
        CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
        CACHE_SIZE: 262144 KB
        CACHE_LEVEL: 3
        MAX_NUM_CU_SHARED: 304
        NUM_CACHE_INSTANCE: 1
RAS:
    EEPROM_VERSION: 0x0
    PARITY_SCHEMA: DISABLED
    SINGLE_BIT_SCHEMA: DISABLED
    DOUBLE_BIT_SCHEMA: DISABLED
    POISON_SCHEMA: ENABLED
    ECC_BLOCK_STATE:
        BLOCK: UMC
        STATUS: DISABLED
        BLOCK: SDMA
        STATUS: ENABLED
        BLOCK: GFX
        STATUS: ENABLED
        BLOCK: MMHUB
        STATUS: ENABLED
        BLOCK: ATHUB
        STATUS: DISABLED
        BLOCK: PCIE_BIF
        STATUS: DISABLED
        BLOCK: HDP
        STATUS: DISABLED
        BLOCK: XGMI_WAFL
        STATUS: DISABLED
        BLOCK: DF
        STATUS: DISABLED
        BLOCK: SMN
        STATUS: DISABLED
        BLOCK: SEM
        STATUS: DISABLED
        BLOCK: MP0
        STATUS: DISABLED
        BLOCK: MP1
        STATUS: DISABLED
        BLOCK: FUSE
        STATUS: DISABLED
PARTITION:
    COMPUTE_PARTITION: SPX
    MEMORY_PARTITION: NPS1
NUMA:
    NODE: 2
    AFFINITY: 2

GPU: 3
ASIC:
    MARKET_NAME: MI300A
    VENDOR_ID: 0x1002
    VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
    SUBVENDOR_ID: 0
    DEVICE_ID: 0x74a0
    REV_ID: 0x0
    ASIC_SERIAL: 0x53a0a1ff3830f499
    OAM_ID: 3
BUS:
    BDF: 0003:01:00.0
    MAX_PCIE_SPEED: 32 GT/s
    MAX_PCIE_LANES: 16
    PCIE_INTERFACE_VERSION: Gen 5
    SLOT_TYPE: PCIE
VBIOS:
    NAME: N/A
    BUILD_DATE: N/A
    PART_NUMBER: N/A
    VERSION: N/A
BOARD:
    MODEL_NUMBER: N/A
    PRODUCT_SERIAL: N/A
    FRU_ID: N/A
    MANUFACTURER_NAME: N/A
    PRODUCT_NAME: N/A
LIMIT:
    MAX_POWER: 550 W
    CURRENT_POWER: 0 W
    SLOWDOWN_EDGE_TEMPERATURE: N/A
    SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
    SLOWDOWN_VRAM_TEMPERATURE: 95 °C
    SHUTDOWN_EDGE_TEMPERATURE: N/A
    SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
    SHUTDOWN_VRAM_TEMPERATURE: 105 °C
DRIVER:
    DRIVER_NAME: amdgpu
    DRIVER_VERSION: 6.5.2
VRAM:
    VRAM_TYPE: HBM
    VRAM_VENDOR: HYNIX
    VRAM_SIZE_MB: 96432 MB
CACHE:
    CACHE_0:
CACHE:
    CACHE_0:
        CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
        CACHE_SIZE: 32 KB
        CACHE_LEVEL: 1
        MAX_NUM_CU_SHARED: 2
        NUM_CACHE_INSTANCE: 464
    CACHE_1:
        CACHE_PROPERTIES: INST_CACHE, SIMD_CACHE
        CACHE_SIZE: 64 KB
        CACHE_LEVEL: 1
        MAX_NUM_CU_SHARED: 2
        NUM_CACHE_INSTANCE: 160
    CACHE_2:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 32768 KB
        CACHE_LEVEL: 2
        MAX_NUM_CU_SHARED: 304
        NUM_CACHE_INSTANCE: 1
    CACHE_3:
        CACHE_PROPERTIES: DATA_CACHE, SIMD_CACHE
        CACHE_SIZE: 262144 KB
        CACHE_LEVEL: 3
        MAX_NUM_CU_SHARED: 304
        NUM_CACHE_INSTANCE: 1
RAS:
    EEPROM_VERSION: 0x0
    PARITY_SCHEMA: DISABLED
    SINGLE_BIT_SCHEMA: DISABLED
    DOUBLE_BIT_SCHEMA: DISABLED
    POISON_SCHEMA: ENABLED
    ECC_BLOCK_STATE:
        BLOCK: UMC
        STATUS: DISABLED
        BLOCK: SDMA
        STATUS: ENABLED
        BLOCK: GFX
        STATUS: ENABLED
        BLOCK: MMHUB
        STATUS: ENABLED
        BLOCK: ATHUB
        STATUS: DISABLED
        BLOCK: PCIE_BIF
        STATUS: DISABLED
        BLOCK: HDP
        STATUS: DISABLED
        BLOCK: XGMI_WAFL
        STATUS: DISABLED
        BLOCK: DF
        STATUS: DISABLED
        BLOCK: SMN
        STATUS: DISABLED
        BLOCK: SEM
        STATUS: DISABLED
        BLOCK: MP0
        STATUS: DISABLED
        BLOCK: MP1
        STATUS: DISABLED
        BLOCK: FUSE
        STATUS: DISABLED
PARTITION:
    COMPUTE_PARTITION: SPX
    MEMORY_PARTITION: NPS1
NUMA:
    NODE: 3
    AFFINITY: 3
```

## Disclaimer

The information contained herein is for informational purposes only, and is subject to change without notice. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

Copyright (c) 2014-2023 Advanced Micro Devices, Inc. All rights reserved.
