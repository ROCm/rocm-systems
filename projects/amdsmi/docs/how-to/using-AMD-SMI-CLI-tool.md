# Using AMD SMI Command Line Interface tool

AMD-SMI reports the version and current platform detected when running the command line interface (CLI) without arguments:

``` bash
~$ amd-smi
usage: amd-smi [-h]  ...

AMD System Management Interface | Version: 24.6.0.0 | ROCm version: 6.2.0 | Platform: Linux Baremetal

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
    xgmi              Displays xgmi information of the devices
```

More detailed verison information is available from `amd-smi version`

Each command will have detailed information via `amd-smi [command] --help`

## Commands

For convenience, here is the help output for each command

``` bash
~$ amd-smi list --help
usage: amd-smi list [-h] [--json | --csv] [--file FILE] [--loglevel LEVEL]
                    [-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]]

- Lists all the devices on the system and the links between devices.
- Lists all the sockets and for each socket, GPUs and/or CPUs associated to that socket alongside some basic information for each device.

.. NOTE::

In virtualization environments, it can also list VFs associated to each GPU with some basic information for each VF.

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
usage: amd-smi static [-h] [-g GPU [GPU ...]] [-a] [-b] [-V] [-d] [-v] [-c] [-B] [-r] [-p]
                      [-l] [-P] [-x] [-s] [-u] [--json | --csv] [--file FILE]
                      [--loglevel LEVEL]

- If no GPU is specified, returns static information for all GPUs on the system.
- If no static argument is provided, all static information will be displayed.

Static Arguments:
  -h, --help               show this help message and exit
  -g, --gpu GPU [GPU ...]  Select a GPU ID, BDF, or UUID from the possible choices:
                           ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                           ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                           ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                           ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
  -r, --ras                Displays RAS features information
  -p, --partition          Partition information
  -l, --limit              All limit metric values (i.e. power and thermal limits)
  -s, --process-isolation  The process isolation status
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

- If no GPU is specified, returns metric information for all GPUs on the system.
- If no metric argument is provided all metric information will be displayed.

Metric arguments:
  -h, --help                                show this help message and exit
  -g, --gpu GPU [GPU ...]                   Select a GPU ID, BDF, or UUID from the possible choices:
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

- If no GPU is specified, returns information for all GPUs on the system.
- If no process argument is provided all process information will be displayed.

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

- If no GPU is specified, returns information for all GPUs on the system.
- If no topology argument is provided all topology information will be displayed.

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
usage: amd-smi set [-h] (-g GPU [GPU ...] | -U CPU [CPU ...] | -O CORE [CORE ...]) [-f %]
                   [-l LEVEL] [-P SETPROFILE] [-d SCLKMAX] [-C PARTITION] [-M PARTITION]
                   [-o WATTS] [-p POLICY] [-i STATUS] [--cpu-pwr-limit PWR_LIMIT]
                   [--cpu-xgmi-link-width MIN_WIDTH MAX_WIDTH]
                   [--cpu-lclk-dpm-level NBIOID MIN_DPM MAX_DPM] [--cpu-pwr-eff-mode MODE]
                   [--cpu-gmi3-link-width MIN_LW MAX_LW] [--cpu-pcie-link-rate LINK_RATE]
                   [--cpu-df-pstate-range MAX_PSTATE MIN_PSTATE] [--cpu-enable-apb]
                   [--cpu-disable-apb DF_PSTATE] [--soc-boost-limit BOOST_LIMIT]
                   [--core-boost-limit BOOST_LIMIT] [-c] [--json | --csv] [--file FILE]
                   [--loglevel LEVEL]

- A GPU must be specified to set a configuration.
- A set argument must be provided; Multiple set arguments are accepted

Set Arguments:
  -h, --help                                   show this help message and exit
  -g, --gpu GPU [GPU ...]                      Select a GPU ID, BDF, or UUID from the possible choices:
                                               ID: 0 | BDF: 0000:01:00.0 | UUID: 71ff74a0-0000-1000-8066-0a3c71d5f817
                                               ID: 1 | BDF: 0001:01:00.0 | UUID: b4ff74a0-0000-1000-80b2-fa0be8628b1a
                                               ID: 2 | BDF: 0002:01:00.0 | UUID: a9ff74a0-0000-1000-8007-3066a98ba4a6
                                               ID: 3 | BDF: 0003:01:00.0 | UUID: 53ff74a0-0000-1000-80a0-a1ff3830f499
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
  -i, --process-isolation STATUS               Enable or disable the GPU process isolation: 0 for disable and 1 for enable.
  -c, --clear-sram-data                        Clear the GPU SRAM data

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
                     [-f] [-p] [-x] [-d] [-C] [-M] [-o]

- A GPU must be specified to reset a configuration.
- A reset argument must be provided; Multiple reset arguments are accepted

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
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a0
        REV_ID: 0x0
        ASIC_SERIAL: 0x71660A3C71D5F817
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
        SOCKET_POWER: 550 W
        SLOWDOWN_EDGE_TEMPERATURE: N/A
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SLOWDOWN_VRAM_TEMPERATURE: 95 °C
        SHUTDOWN_EDGE_TEMPERATURE: N/A
        SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SHUTDOWN_VRAM_TEMPERATURE: 105 °C
    DRIVER:
        NAME: amdgpu
        VERSION: 6.7.0
    BOARD:
        MODEL_NUMBER: N/A
        PRODUCT_SERIAL: N/A
        FRU_ID: N/A
        PRODUCT_NAME: N/A
        MANUFACTURER_NAME: N/A
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
    NUMA:
        NODE: 0
        AFFINITY: 0
    VRAM:
        TYPE: HBM
        VENDOR: N/A
        SIZE: 96432 MB
  CACHE_INFO:
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

GPU: 1
    ASIC:
        MARKET_NAME: MI300A
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a0
        REV_ID: 0x0
        ASIC_SERIAL: 0xB4B2FA0BE8628B1A
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
        SOCKET_POWER: 550 W
        SLOWDOWN_EDGE_TEMPERATURE: N/A
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SLOWDOWN_VRAM_TEMPERATURE: 95 °C
        SHUTDOWN_EDGE_TEMPERATURE: N/A
        SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SHUTDOWN_VRAM_TEMPERATURE: 105 °C
    DRIVER:
        NAME: amdgpu
        VERSION: 6.7.0
    BOARD:
        MODEL_NUMBER: N/A
        PRODUCT_SERIAL: N/A
        FRU_ID: N/A
        PRODUCT_NAME: N/A
        MANUFACTURER_NAME: N/A
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
    NUMA:
        NODE: 1
        AFFINITY: 1
    VRAM:
        TYPE: HBM
        VENDOR: N/A
        SIZE: 96432 MB
  CACHE_INFO:
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

GPU: 2
    ASIC:
        MARKET_NAME: MI300A
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a0
        REV_ID: 0x0
        ASIC_SERIAL: 0xA9073066A98BA4A6
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
        SOCKET_POWER: 550 W
        SLOWDOWN_EDGE_TEMPERATURE: N/A
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SLOWDOWN_VRAM_TEMPERATURE: 95 °C
        SHUTDOWN_EDGE_TEMPERATURE: N/A
        SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SHUTDOWN_VRAM_TEMPERATURE: 105 °C
    DRIVER:
        NAME: amdgpu
        VERSION: 6.7.0
    BOARD:
        MODEL_NUMBER: N/A
        PRODUCT_SERIAL: N/A
        FRU_ID: N/A
        PRODUCT_NAME: N/A
        MANUFACTURER_NAME: N/A
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
    NUMA:
        NODE: 2
        AFFINITY: 2
    VRAM:
        TYPE: HBM
        VENDOR: N/A
        SIZE: 96432 MB
  CACHE_INFO:
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

GPU: 3
    ASIC:
        MARKET_NAME: MI300A
        VENDOR_ID: 0x1002
        VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
        SUBVENDOR_ID: 0x1002
        DEVICE_ID: 0x74a0
        REV_ID: 0x0
        ASIC_SERIAL: 0x53A0A1FF3830F499
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
        SOCKET_POWER: 550 W
        SLOWDOWN_EDGE_TEMPERATURE: N/A
        SLOWDOWN_HOTSPOT_TEMPERATURE: 100 °C
        SLOWDOWN_VRAM_TEMPERATURE: 95 °C
        SHUTDOWN_EDGE_TEMPERATURE: N/A
        SHUTDOWN_HOTSPOT_TEMPERATURE: 110 °C
        SHUTDOWN_VRAM_TEMPERATURE: 105 °C
    DRIVER:
        NAME: amdgpu
        VERSION: 6.7.0
    BOARD:
        MODEL_NUMBER: N/A
        PRODUCT_SERIAL: N/A
        FRU_ID: N/A
        PRODUCT_NAME: N/A
        MANUFACTURER_NAME: N/A
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
    NUMA:
        NODE: 3
        AFFINITY: 3
    VRAM:
        TYPE: HBM
        VENDOR: N/A
        SIZE: 96432 MB
  CACHE_INFO:
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

```

