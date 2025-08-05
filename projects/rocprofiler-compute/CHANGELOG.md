# Changelog for ROCm Compute Profiler

Full documentation for ROCm Compute Profiler is available at [https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/).

## Unreleased

### Added

* Add `rocpd` choice for `--format-rocprof-output` option in profile mode
* Add `--retain-rocpd-output` option in profile mode to save large raw rocpd databases in workload directory
* Show description of metrics during analysis
  * Use `--include-cols Description` to show the Description column, which is excluded by default from the
  ROCm Compute Profiler CLI output.

### Changed

* Add notice for change in default output format to `rocpd` in a future release
  * This is displayed when `--format-rocprof-output rocpd` is not used in profile mode

* When `--format-rocprof-output rocpd` is used, only pmc_perf.csv will be written to workload directory instead of mulitple csv files.

* Improve analysis block based filtering to accept metric id level filtering
  * This can be used to collect individual metrics from various sections of analysis config

* CLI analysis mode baseline comparison will now only compare common metrics across workloads and will not show Metric ID
  * Remove metrics from analysis configuration files which are explicitly marked as empty or None

* Change the basic view of TUI from aggregated analysis data to individual kernel analysis data

### Resolved issues

* Fixed not detecting memory clock issue when using amd-smi
* Fixed standalone GUI crashing
* Fixed L2 read/write/atomic bandwidths on MI350
* Update metric names for better alignment between analysis configuration and documentation

### Known issues

### Optimized

* Improved `--time-unit` option in analyze mode to apply time unit conversion across all analysis sections, not just kernel top stats.

### Removed

* Usage of rocm-smi
* Hardware IP block based filtering has been removed in favor of analysis report block based filtering
* Remove aggregated analysis view from TUI mode


## ROCm Compute Profiler 3.2.1 for ROCm 7.0.0

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

* Roofline support for RHEL 10 OS.

* FP4 and FP6 data types have been added for roofline profiling on AMD Instinct MI350 series.

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

* Interface to ROCprofiler-SDK.
  * Setting the environment variable ``ROCPROF=rocprofiler-sdk`` will use ROCprofiler-SDK C++ library instead of ``rocprofv3`` python script.
  * Add --rocprofiler-sdk-library-path runtime option to choose the path to rocprofiler-sdk library to be used
  * Using rocprof v1 / v2 / v3 interfaces will trigger a deprecation warning to use rocprofiler-sdk interface

* Support MEM chart on CLI (single run)

* Deprecation warning for MongoDB database update mode.

* Deprecation warning for ``rocm-smi``

* ``--specs-correction`` option to provide missing system specifications for analysis.

### Changed

* Changed the default ``rocprof`` version to ``rocprofv3``. This is used when environment variable ``ROCPROF`` is not set.
* Changed ``normal_unit`` default to ``per_kernel``.
* Decreased profiling time by not collecting unused counters in post-analysis.
* Updated Dash to >=3.0.0 (for web UI).
* Changed the condition when Roofline PDFs are generated during general profiling and ``--roof-only`` profiling (skip only when ``--no-roof`` option is present).
* Updated Roofline binaries:
  * Rebuild using latest ROCm stack
  * Minimum OS distribution support minimum for roofline feature is now Ubuntu 22.04, RHEL 9, and SLES15 SP6.

### Optimized

* ROCm Compute Profiler CLI has been improved to better display the GPU architecture analytics

### Resolved issues

* Fixed kernel name and kernel dispatch filtering when using ``rocprofv3``.
* Fixed an issue of TCC channel counters collection in ``rocprofv3``.
* Fixed peak FLOPS of F8, I8, F16, and BF16 on AMD Instinct MI 300.

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

### Removed

* Roofline support for Ubuntu 20.04 and SLES below 15.6
* Removed support for AMD Instinct MI50 and MI60.

### Upcoming changes

* ``rocprof v1/v2/v3`` interfaces will be removed in favor of the ROCprofiler-SDK interface, which directly accesses ``rocprofv3`` C++ tool.
  * To use ROCprofiler-SDK interface, set environment variable `ROCPROF=rocprofiler-sdk` and optionally provide profile mode option ``--rocprofiler-sdk-library-path /path/to/librocprofiler-sdk.so``
* Hardware IP block based filtering using ``-b`` option in profile mode will be removed in favor of analysis report block based filtering using ``-b`` option in profile mode.
* Using rocprof v1 / v2 / v3 interfaces will trigger a deprecation warning to use rocprofiler-sdk interface
* MongoDB database support will be removed.
* Usage of ``rocm-smi`` will be removed in favor of ``amd-smi``.


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
