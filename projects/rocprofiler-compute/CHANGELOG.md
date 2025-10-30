# Changelog for ROCm Compute Profiler

Full documentation for ROCm Compute Profiler is available at [https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/).

## Unreleased

### Added

* Add `--list-blocks <arch>` option to general options to list available IP blocks on specified arch (similar to `--list-metrics`), cannot be used with `--block`.

* Added `config_delta/gfx950_diff.yaml` to analysis config yamls to track the revision between a gfx9 architecture against the latest supported architecture gfx950

### Changed

* `-b/--block` accepts block alias(es) (See block aliases using command-line option `--list-blocks <arch>`).

* analysis configs yamls are now managed with the new config management workflow in `tools/config_management/`

* `amdsmi` python API is used instead of `amd-smi` CLI to query GPU specifications.


### Removed
* Removed `database` mode from `rocprofiler-compute`. This is to move our focus from grafana
  and mongodb integration to other visualization methods such as:
  * Analysis DB based Visualizer (upcoming)
  * Plotly server based standalone GUI
  * Commandline based Textual User Interface

### Optimized

### Resolved issues

### Known issues

### Upcoming changes

## ROCm Compute Profiler 3.3.1 for ROCm 7.1.1

### Added

* Improved standalone Roofline plots in profile mode (PDF output) and analyze mode (CLI and GUI visual plots):
  * Fixed the peak MFMA/VALU lines being cut off.
  * Cleaned up the overlapping roofline numeric values by moving them into the side legend.
  * Added AI points chart with respective values, cache level, and compute/memory bound status.
  * Added full kernel names to symbol chart.

* Add support for multi-kernel applications' pc sampling.
  * PC Sampling's outputs' instructions are displayed with the name of the kernel that individual instruction belongs to.
  * Single kernel selection is supported so that the pc samples of selected kernel can be displayed.


### Changed

* Roofline analysis now runs on GPU 0 by default instead of all GPUs.

### Optimized

* Improved Roofline Benchmarking by updating the `flops_benchmark` calculation.

### Resolved issues

* Bugfixes for stability

## ROCm Compute Profiler 3.3.0 for ROCm 7.1.0

### Added
* Live attach/detach feature that allows coupling with a workload process, without controlling its start or end.
  * Use '--attach-pid' to specify the target process ID.
  * Use '--attach-duration-msec' to specify time duration.

* Add `rocpd` choice for `--format-rocprof-output` option in profile mode

* Add `--retain-rocpd-output` option in profile mode to save large raw rocpd databases in workload directory

* Show description of metrics during analysis
  * Use `--include-cols Description` to show the Description column, which is excluded by default from the
  ROCm Compute Profiler CLI output.
* `--set` filtering option in profile mode to enable single-pass counter collection for predefined subsets of metrics.
* `--list-sets` filtering option in profile mode to list the sets available for single pass counter collection

* Add missing counters based on register specification which enables missing metrics
  * Enable SQC_DCACHE_INFLIGHT_LEVEL counter and associated metrics
  * Enable TCP_TCP_LATENCY counter and associated counter for all GPUs except MI300

* Added interactive metric descriptions in TUI analyze mode
  * users can now left click on any metric cell to view detailed descriptions in the dedicated `METRIC DESCRIPTION` tab

* Add support for analysis report output as a sqlite database using ``--output-format db`` analysis mode option

* `Compute Throughput` panel to TUI's `High Level Analysis` category with the following metrics:
  * VALU FLOPs
  * VALU IOPs
  * MFMA FLOPs (F8)
  * MFMA FLOPs (BF16)
  * MFMA FLOPs (F16)
  * MFMA FLOPs (F32)
  * MFMA FLOPs (F64)
  * MFMA FLOPs (F6F4) (in gfx950)
  * MFMA IOPs (Int8)
  * SALU Utilization
  * VALU Utilization
  * MFMA Utilization
  * VMEM Utilization
  * Branch Utilization
  * IPC

* `Memory Throughput` panel to TUI's `High Level Analysis` category with the following metrics:
  * vL1D Cache BW
  * vL1D Cache Utilization
  * Theoretical LDS Bandwidth
  * LDS Utilization
  * L2 Cache BW
  * L2 Cache Utilization
  * L2-Fabric Read BW
  * L2-Fabric Write BW
  * sL1D Cache BW
  * L1I BW
  * Address Processing Unit Busy
  * Data-Return Busy
  * L1I-L2 Bandwidth
  * sL1D-L2 BW

* Roofline support for Debian 12 and Azure Linux 3.0.

### Changed

* On memory chart, long string of numbers are displayed as scientific notation. It also solves the issue of overflow of displaying long number

* Add notice for change in default output format to `rocpd` in a future release
  * This is displayed when `--format-rocprof-output rocpd` is not used in profile mode

* When `--format-rocprof-output rocpd` is used, only pmc_perf.csv will be written to workload directory instead of mulitple csv files.

* Improve analysis block based filtering to accept metric id level filtering
  * This can be used to collect individual metrics from various sections of analysis config

* CLI analysis mode baseline comparison will now only compare common metrics across workloads and will not show Metric ID
  * Remove metrics from analysis configuration files which are explicitly marked as empty or None

* Changed the basic (default) view of TUI from aggregated analysis data to individual kernel analysis data.

* Update `Unit` of the following `Bandwidth` related metrics to `Gbps` instead of `Bytes per Normalization Unit`
  * Theoretical Bandwidth (section 1202)
  * L1I-L2 Bandwidth (section 1303)
  * sL1D-L2 BW (section 1403)
  * Cache BW (section 1603)
  * L1-L2 BW (section 1603)
  * Read BW (section 1702)
  * Write and Atomic BW (section 1702)
  * Bandwidth (section 1703)
  * Atomic/Read/Write Bandwidth (section 1703)
  * Atomic/Read/Write Bandwidth - (HBM/PCIe/Infinity Fabric) (section 1706)

* Add `Utilization` to metric name for the following `Bandwidth` related metrics whose `Unit` is `Percent`
  * Theoretical Bandwidth Utilization (section 1201)
  * L1I-L2 Bandwidth Utilization (section 1301)
  * Bandwidth Utilization (section 1301)
  * Bandwidth Utilization (section 1401)
  * sL1D-L2 BW Utilization (section 1401)
  * Bandwidth Utilization (section 1601)

* Update `System Speed-of-Light` panel to `GPU Speed-of-Light` in TUI with the following metrics:
  * Theoretical LDS Bandwidth
  * vL1D Cache BW
  * L2 Cache BW
  * L2-Fabric Read BW
  * L2-Fabric Write BW
  * Kernel Time
  * Kernel Time (Cycles)
  * SIMD Utilization
  * Clock Rate

* Analysis output:
  * Replace `-o / --output` analyze mode option with `--output-format` and `--output-name`
    * Add ``--output-format`` analysis mode option to select the output format of the analysis report.
    * Add ``--output-name`` analysis mode option to override the default file/folder name.
  * Replace `--save-dfs` analyze mode option with `--output-format csv`

* Command-line options:
  * `--list-metrics` and `--config-dir` options moved to general command-line options.
  * * `--list-metrics` option cannot be used without argument (GPU architecture).
  * `--list-metrics` option do not show number of L2 channels.
  * `--list-available-metrics` profile mode option to display the metrics available for profiling in current GPU.
  * `--list-available-metrics` analyze mode option to display the metrics available for analysis.
  * `--block` option cannot be used with `--list-metrics` and `--list-available-metrics`options.

* Default rocprof interface changed from rocprofv3 to rocprofiler-sdk
  * Use ROCPROF=rocprofv3 to use rocprofv3 interface

* Roofline analysis now runs on GPU 0 by default instead of all GPUs.

### Removed

* Usage of `rocm-smi` in favor of `amd-smi`.
* Hardware IP block-based filtering has been removed in favor of analysis report block-based filtering.
* Removed aggregated analysis view from TUI analyze mode.

### Optimized

* Improved `--time-unit` option in analyze mode to apply time unit conversion across all analysis sections, not just kernel top stats.
* Improved logic to obtain rocprof supported counters which prevents unnecessary warnings.
* Improved post-analysis runtime performance by caching and multi-processing.

### Resolved issues

* Fixed an issue of not detecting the memory clock when using `amd-smi`.
* Fixed standalone GUI crashing.
* Fixed L2 read/write/atomic bandwidths on AMD Instinct MI350 series accelerators.
* Update metric names for better alignment between analysis configuration and documentation
* Fixed an issue where accumulation counters could not be collected on AMD Instinct MI100.
* Fixed an issue of kernel filtering not working in the roofline chart

### Known issues

* MI300A/X L2-Fabric 64B read counter may display negative values - The rocprof-compute metric 17.6.1 (Read 64B) can report negative values due to incorrect calculation when TCC_BUBBLE_sum + TCC_EA0_RDREQ_32B_sum exceeds TCC_EA0_RDREQ_sum.
  * A workaround has been implemented using max(0, calculated_value) to prevent negative display values while the root cause is under investigation.

### Upcoming changes

## ROCm Compute Profiler 3.2.3 for ROCm 7.0.0

### Added

#### CDNA4 (AMD Instinct MI350/MI355) support

* Support for AMD Instinct MI350 series GPUs with the addition of the following counters:
  * VALU co-issue (Two VALUs are issued instructions) efficiency
  * Stream Processor Instruction (SPI) Wave Occupancy
  * Scheduler-Pipe Wave Utilization
  * Scheduler FIFO Full Rate
  * CPC ADC Utilization
  * F6F4 data type metrics
  * Update formula for total FLOPs while taking into account F6F4 ops
  * LDS STORE, LDS LOAD, LDS ATOMIC instruction count metrics
  * LDS STORE, LDS LOAD, LDS ATOMIC bandwidth metrics
  * LDS FIFO full rate
  * Sequencer -> TA ADDR Stall rates
  * Sequencer -> TA CMD Stall rates
  * Sequencer -> TA DATA Stall rates
  * L1 latencies
  * L2 latencies
  * L2 to EA stalls
  * L2 to EA stalls per channel

* Roofline support for AMD Instinct MI350 series architecture.

#### Textual User Interface (TUI) (beta version)

* Text User Interface (TUI) support for analyze mode
  * A command line based user interface to support interactive single-run analysis
  * To launch, use `--tui` option in analyze mode. For example, ``rocprof-compute analyze --tui``.

#### PC Sampling (beta version)

* Stochastic (hardware-based) PC sampling has been enabled for AMD Instinct MI300X series and later accelerators.

* Host-trap PC Sampling has been enabled for AMD Instinct MI200 series and later accelerators.

* Support for sorting of PC sampling by type: offset or count.

* PC Sampling Support on CLI and TUI analysis.

#### Roofline

* Support for Roofline plot on CLI (single run) analysis.

* `FP4` and `FP6` data types have been added for roofline profiling on AMD Instinct MI350 series.

#### rocprofv3 support

* ``rocprofv3`` is supported as the default backend for profiling.
* Support to obtain performance information for all channels for TCC counters.
* Support for profiling on AMD Instinct MI 100 using ``rocprofv3``.
* Deprecation warning for ``rocprofv3`` interface in favor of the ROCprofiler-SDK interface, which directly accesses ``rocprofv3`` C++ tool.

#### Others

* Docker files to package the application and dependencies into a single portable and executable standalone binary file.

* Analysis report based filtering
  * ``-b`` option in profile mode now also accepts metric id(s) for analysis report based filtering.
  * ``-b`` option in profile mode also accepts hardware IP block for filtering; however, this filter support will be deprecated soon.
  * ``--list-metrics`` option added in profile mode to list possible metric id(s), similar to analyze mode.

* Support MEM chart on CLI (single run)

* ``--specs-correction`` option to provide missing system specifications for analysis.

### Changed

* Changed the default ``rocprof`` version to ``rocprofv3``. This is used when environment variable ``ROCPROF`` is not set.
* Changed ``normal_unit`` default to ``per_kernel``.
* Decreased profiling time by not collecting unused counters in post-analysis.
* Updated Dash to >=3.0.0 (for web UI).
* Changed the condition when Roofline PDFs are generated during general profiling and ``--roof-only`` profiling (skip only when ``--no-roof`` option is present).
* Updated Roofline binaries:
  * Rebuild using latest ROCm stack
  * Minimum OS distribution support minimum for roofline feature is now Ubuntu 22.04, RHEL 8, and SLES15 SP6.

### Removed

* Roofline support for Ubuntu 20.04 and SLES below 15.6
* Removed support for AMD Instinct MI50 and MI60.

### Optimized

* ROCm Compute Profiler CLI has been improved to better display the GPU architecture analytics

### Resolved issues

* Fixed kernel name and kernel dispatch filtering when using ``rocprofv3``.
* Fixed an issue of TCC channel counters collection in ``rocprofv3``.
* Fixed peak FLOPS of `F8`, `I8`, `F16`, and `BF16` on AMD Instinct MI300.
* Fixed not detecting memory clock issue when using amd-smi
* Fixed standalone GUI crashing
* Fixed L2 read/write/atomic bandwidths on AMD Instinct MI350 series.

### Known issues

* On AMD Instinct MI100, accumulation counters are not collected, resulting in the following metrics failing to show up in the analysis: Instruction Fetch Latency, Wavefront Occupancy, LDS Latency
  * As a workaround, use the environment variable ``ROCPROF=rocprof``, to use ``rocprof v1`` for profiling on AMD Instinct MI100.

* GPU id filtering is not supported when using ``rocprofv3``.

* Analysis of previously collected workload data will not work due to sysinfo.csv schema change.
  * As a workaround, re-run the profiling operation for the workload and interrupt the process after 10 seconds.
  Followed by copying the ``sysinfo.csv`` file from the new data folder to the old one.
  This assumes your system specification hasn't changed since the creation of the previous workload data.

* Analysis of new workloads might require providing shader/memory clock speed using
``--specs-correction`` operation if amd-smi or rocminfo does not provide clock speeds.

* Memory chart on ROCm Compute Profiler CLI might look corrupted if the CLI width is too narrow.

* Roofline feature is currently not functional on Azure Linux 3.0 and Debian 12.

### Upcoming changes

* ``rocprof v1/v2/v3`` interfaces will be removed in favor of the ROCprofiler-SDK interface, which directly accesses ``rocprofv3`` C++ tool. Using ``rocprof v1/v2/v3`` interfaces will trigger a deprecation warning.
  * To use ROCprofiler-SDK interface, set environment variable `ROCPROF=rocprofiler-sdk` and optionally provide profile mode option ``--rocprofiler-sdk-library-path /path/to/librocprofiler-sdk.so``. Add ``--rocprofiler-sdk-library-path`` runtime option to choose the path to ROCprofiler-SDK library to be used.
* Hardware IP block based filtering using ``-b`` option in profile mode will be removed in favor of analysis report block based filtering using ``-b`` option in profile mode.
* MongoDB database support will be removed, and a deprecation warning has been added to the application interface.
* Usage of ``rocm-smi`` is deprecated in favor of ``amd-smi``, and a deprecation warning has been added to the application interface.

## ROCm Compute Profiler 3.1.1 for ROCm 6.4.2

### Added

* 8-bit floating point (FP8) metrics support for AMD Instinct MI300 GPUs.
* Additional data types for roofline: FP8, FP16, BF16, FP32, FP64, I8, I32, I64 (dependent on the GPU architecture).
* Data type selection option ``--roofline-data-type / -R`` for roofline profiling. The default data type is FP32.

### Changed

* Change dependency from `rocm-smi` to `amd-smi`.

### Resolved issues

* Fixed a crash related to Agent ID caused by the new format of the `rocprofv3` output CSV file.


## ROCm Compute Profiler 3.1.0 for ROCm 6.4.0

### Added

* Roofline support for Ubuntu 24.04
* Experimental support rocprofv3 (not enabled as default)

### Resolved issues

* Fixed PoP of VALU Active Threads
* Workaround broken mclk for old version of rocm-smi

## ROCm Compute Profiler 3.0.0 for ROCm 6.3.0

### Changed

* Renamed Omniperf to ROCm Compute Profiler (#475)

## Omniperf 2.0.1 for ROCm 6.2.1

### Changed

* enable rocprofv1 for MI300 hardware (#391)
* refactoring and updating documemtation (#362, #394, #398, #414, #420)
* branch renaming and workflow updates (#389, #404, #409)
* bug fix for analysis output
* add dependency checks on application launch (#393)
* patch for profiling multi-process/multi-GPU applications (#376, #396)
* packaging updates (#386)
* rename CHANGES to CHANGELOG.md (#410)
* rollback Grafana version in Dockerfile for Angular plugin compatibility (#416)
* enable CI triggers for Azure CI (#426)
* add GPU model distinction for MI300 systems (#423)
* new MAINTAINERS.md guide for omniperf publishing procedures (#402)

### Optimized

* reduced running time of Omniperf when profiling (#384)
* console logging improvements

## Omniperf 2.0.1 for ROCm 6.2.0

### Added

  * new option to force hardware target via `OMNIPERF_ARCH_OVERRIDE` global (#370)
  * CI/CD support for MI300 hardware (#373)
  * support for MI308X hardware (#375)

### Optimized

  * cmake build improvements (#374)

## Omniperf 2.0.0 (17 May 2024)

  * improved logging than spans all modes (#177) (#317) (#335) (#341)
  * overhauled CI/CD that spans all modes (#179)
  * extensible SoC classes to better support adding new hardware configs (#180)
  * --kernel-verbose no longer overwrites kernel names (#193)
  * general cleanup and improved organization of source code (#200) (#210)
  * separate requirement files for docs and testing dependencies (#205) (#262) (#358)
  * add support for MI300 hardware (#231)
  * upgrade Grafana assets and build script to latest release (#235)
  * update minimum ROCm and Python requirements (#277)
  * sort rocprofiler input files prior to profiling (#304)
  * new --quiet option will suppress verbose output and show a progress bar (#308)
  * roofline support for Ubuntu 22.04 (#319)

## Omniperf 1.1.0-PR1 (13 Oct 2023)

  * standardize headers to use 'avg' instead of 'mean'
  * add color code thresholds to standalone gui to match grafana
  * modify kernel name shortener to use cpp_filt (#168)
  * enable stochastic kernel dispatch selection (#183)
  * patch grafana plugin module to address a known issue in the latest version (#186)
  * enhanced communication between analyze mode kernel flags (#187)

## Omniperf 1.0.10 (22 Aug 2023)

  * critical patch for detection of llvm in rocm installs on SLURM systems

## Omniperf 1.0.9 (17 Aug 2023)

  * add units to L2 per-channel panel (#133)
  * new quickstart guide for Grafana setup in docs (#135)
  * more detail on kernel and dispatch filtering in docs (#136, #137)
  * patch manual join utility for ROCm >5.2.x (#139)
  * add % of peak values to low level speed-of-light panels (#140)
  * patch critical bug in Grafana by removing a deprecated plugin (#141)
  * enhancements to KernelName demangeler (#142)
  * general metric updates and enhancements (#144, #155, #159)
  * add min/max/avg breakdown to instruction mix panel (#154)

## Omniperf 1.0.8 (30 May 2023)

  * add `--kernel-names` option to toggle kernelName overlay in standalone roofline plot (#93)
  * remove unused python modules (#96)
  * fix empirical roofline calculation for single dispatch workloads (#97)
  * match color of arithmetic intensity points to corresponding bw lines

  * ux improvements in standalone GUI (#101)
  * enhanced readability for filtering dropdowns in standalone GUI (#102)
  * new logfile to capture rocprofiler output (#106)
  * roofline support for sles15 sp4 and future service packs (#109)
  * adding dockerfiles for all supported Linux distros
  * new examples for `--roof-only` and `--kernel` options added to documentation

  * enable cli analysis in Windows (#110)
  * optional random port number in standalone GUI (#111)
  * limit length of visible kernelName in `--kernel-names` option (#115)
  * adjust metric definitions (#117, #130)
  * manually merge rocprof runs, overriding default rocprofiler implementation (#125)
  * fixed compatibility issues with Python 3.11 (#131)

## Omniperf 1.0.8-PR2 (17 Apr 2023)

  * ux improvements in standalone GUI (#101)
  * enhanced readability for filtering dropdowns in standalone GUI (#102)
  * new logfile to capture rocprofiler output (#106)
  * roofline support for sles15 sp4 and future service packs (#109)
  * adding dockerfiles for all supported Linux distros
  * new examples for `--roof-only` and `--kernel` options added to documentation

## Omniperf 1.0.8-PR1 (13 Mar 2023)

  * add `--kernel-names` option to toggle kernelName overlay in standalone roofline plot (#93)
  * remove unused python modules (#96)
  * fix empirical roofline calculation for single dispatch workloads (#97)
  * match color of arithmetic intensity points to corresponding bw lines

## Omniperf 1.0.7 (21 Feb 2023)

  * update documentation (#52, #64)
  * improved detection of invalid command line arguments (#58, #76)
  * enhancements to standalone roofline (#61)
  * enable Omniperf on systems with X-server (#62)
  * raise minimum version requirement for rocm (#64)
  * enable baseline comparison in CLI analysis (#65)
  * add multi-normalization to new metrics (#68, #81)
  * support alternative profilers (#70)
  * add MI100 configs to override rocprofiler's incomplete default (#75)
  * improve error message when no GPU(s) detected (#85)
  * separate CI tests by Linux distro and add status badges

## Omniperf 1.0.6 (21 Dec 2022)

  * CI update: documentation now published via github action (#22)
  * better error detection for incomplete ROCm installs (#56)

## Omniperf 1.0.5 (13 Dec 2022)

  * store application command-line parameters in profiling output (#27)
  * enable additional normalizations in CLI mode (#30)
  * add missing ubuntu 20.04 roofline binary to packaging (#34)
  * update L1 bandwidth metric calculations (#36)
  * add L1 <-> L2 bandwidth calculation (#37)
  * documentation updates (#38, #41)
  * enhanced subprocess logging to identify critical errors in rocprofiler (#50)
  * maintain git sha in production installs from tarball (#53)

## Omniperf 1.0.4 (11 Nov 2022)

  * update python requirements.txt with minimum versions for numpy and pandas
  * addition of progress bar indicator in web-based GUI (#8)
  * reduced default content for web-based GUI to reduce load times (#9)
  * minor packaging and CI updates
  * variety of documentation updates
  * added an optional argument to vcopy.cpp workload example to specify device id

## Omniperf 1.0.3 (07 Nov 2022)

  * initial Omniperf release
