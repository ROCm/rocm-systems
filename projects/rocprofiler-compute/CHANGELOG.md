# Changelog for ROCm Compute Profiler

Full documentation for ROCm Compute Profiler is available at [https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/).

## Unreleased

### Added

* Support Roofline plot on CLI (single run)

* Stochastic (hardware-based) PC sampling has been enabled for AMD Instinct MI300X series and later accelerators.

* Sorting of PC sampling by type: offset or count.

* Add rocprof-compute Text User Interface (TUI) support for analyze mode
  * A command line based user interface to support interactive single-run analysis
  * launch with `--tui` option in analyze mode. i.e., `rocprof-compute analyze --tui`

* Add support to be able to acquire from rocprofv3 every single channle on each XCD of TCC counters

* Add Docker files to package the application and dependencies into a single portable and executable standalone binary file

* Analysis report based filtering
  * -b option in profile mode now additionally accepts metric id(s) for analysis report based filtering
  * -b option in profile mode also accept hardware IP block for filtering, however, this support will be deprecated soon
  * --list-metrics option added in profile mode to list possible metric id(s), similar to analyze mode

* Datatype selection option for roofline profiling
  * --roofline-data-type / -R option added to specify which datatypes the user wants to capture in the roofline PDF plot outputs
  * Default is FP32, but user can specify as many types as desired to overlay on the same plot output

* Additional datatypes for roofline profiling
  * Now supports FP4, FP6, FP8, FP16, BF16, FP32, FP64, I8, I32, I64 (dependent on gpu architecture)

* Support host-trap PC Sampling on CLI (beta version)

* Support for AMD Instinct MI350 series GPUs with the addition of the following counters:
  * VALU co-issue (Two VALUs are issued instructions) efficiency
  * Stream Processor Instruction (SPI) Wave Occupancy
  * Scheduler-Pipe Wave Utilization
  * Scheduler FIFO Full Rate
  * CPC ADC Utilization
  * F6F4 datatype metrics
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

* Roofline support for RHEL 10

* Roofline support for MI350 series architecture

* Interface to rocprofiler-sdk
  * Setting ROCPROF=rocprofiler-sdk environment variable will use rocprofiler-sdk C++ library instead of rocprofv3 python script
  * Add --rocprofiler-sdk-library-path runtime option to choose the path to rocprofiler-sdk library to be used
  * Using rocprof v1 / v2 / v3 interfaces will trigger a deprecation warning to use rocprofiler-sdk interface

* Support MEM chart on CLI (single run)

* Add deprecation warning for database update mode.

### Changed

* Change the default rocprof version to rocprofv3, this is used when environment variable "ROCPROF" is not set
* Change the rocprof version for unit tests to rocprofv3 on all SoCs except MI100
* Change normal_unit default to per_kernel
* Change dependency from rocm-smi to amd-smi
* Decrease profiling time by not collecting counters not used in post analysis
* Update definition of following metrics for MI 350:
  * VGPR Writes
  * Total FLOPs (consider fp6 and fp4 ops)
* Update Dash to >=3.0.0 (for web UI)

### Resolved issues

* Fixed option specs-correction
* Fixed kernel name and kernel dispatch filtering when using rocprof v3
* Fixed not collecting TCC channel counters in rocprof v3
* Fixed peak FLOPS of F8 I8 F16 and BF16 on MI300

### Known issues

* Profiling on MI 100 will not work unless ROCPROF=rocprofv1 environment variable is explictly provided
* GPU id filtering is not supported when using rocprof v3

* Analysis of previously collected workload data will not work due to sysinfo.csv schema change
  * As a workaround, run the profiling operation again for the workload and interrupt the process after ten seconds.
    Followed by copying the `sysinfo.csv` file from the new data folder to the old one.
    This assumes your system specification hasn't changed since the creation of the previous workload data.

* Analysis of new workloads might require providing shader/memory clock speed using
--specs-correction operation if `amd-smi` or `rocminfo` does not provide clock speeds.

### Removed

* Roofline support for Ubuntu 20.04 and SLES below 15.6

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
