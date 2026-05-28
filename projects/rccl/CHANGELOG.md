# Changelog for RCCL

Full documentation for RCCL is available at [https://rccl.readthedocs.io](https://rccl.readthedocs.io)

## Unreleased - RCCL 2.27.7 for ROCm 7.2.0

### Changed

* RCCL error messages have been made more verbose in several cases. RCCL now prints out fatal error messages by default. Fatal error messages can be suppressed by setting `NCCL_DEBUG=NONE`.
* Disabled `reduceCopyPacks` pipelining for `gfx950`.

## Unreleased - RCCL 2.27.7 for ROCm 7.1.1

### Changed
* Enabling P2P batching with `RCCL_P2P_BATCH_ENABLE=1` is only applicable up to 32 nodes.

### Resolved Issues

* Fixed crash when using the librccl-profiler plugin with the all-to-all collective after the 2.27 update.

## RCCL 2.27.7 for ROCm 7.1.0

### Added
* Added `RCCL_IB_QPS_PER_P2P` to set the number of QPs per connection for P2P operations. When set (≥1), P2P operations (Send/Recv) use `RCCL_IB_QPS_PER_P2P`, while other collective operations continue to use `NCCL_IB_QPS_PER_CONNECTION`. When not set, `NCCL_IB_QPS_PER_CONNECTION` applies to all operations.
* Added `RCCL_FORCE_ENABLE_DMABUF` as a debugging feature if the user wants to explicitly enable DMABUF and forego system/kernel checks.
* Added `RCCL_P2P_BATCH_THRESHOLD` to set the message size limit for batching P2P operations. This mainly affects small message performance for alltoall at a large scale but also applies to alltoallv.
* Added `RCCL_P2P_BATCH_ENABLE` to enable batching P2P operations to receive performance gains for smaller messages up to 4MB for alltoall when the workload requires it. This is to avoid performance dips for larger messages.
* Added `RCCL_CHANNEL_TUNING_ENABLE` to enable channel tuning that overrides RCCL's internal adjustments based on threadThreshold.


### Changed

* The MSCCL++ feature is now disabled by default. The `--disable-mscclpp` build flag is replaced with `--enable-mscclpp` in the `rccl/install.sh` script.
* Compatibility with NCCL 2.27.7.

### Optimized
* Enabled and optimized batched P2P operations to improve small message performance for AllToAll and AllGather.
* Optimized channel count selection to improve efficiency for small to medium message sizes in ReduceScatter.
* Changed code inlining to improve latency for small message sizes for AllReduce, AllGather, and ReduceScatter.

### Known issues
* Symmetric memory kernels are currently disabled due to ongoing CUMEM enablement work.
* When running this version of RCCL using ROCm versions earlier than 6.4.0, the user must set the environment flag `HSA_NO_SCRATCH_RECLAIM=1`.

## RCCL 2.26.6 for ROCm 7.0.0

### Resolved issues

* Resolved an issue when using more than 64 channels when multiple collectives are used in the same `ncclGroup()` call.
* Fixed unit test failures in tests ending with `ManagedMem` and `ManagedMemGraph` suffixes.
* Suboptimal algorithmic switching point for AllReduce on MI300x.
* Fixed the known issue "When splitting a communicator using `ncclCommSplit` in some GPU configurations, MSCCL initialization can cause a segmentation fault." with a design change to use `comm` instead of `rank` for `mscclStatus`. The Global map for `comm` to `mscclStatus` is still not thread safe but should be explicitly handled by mutexes for read writes. This is tested for correctness, but there is a plan to use a thread-safe map data structure in upcoming changes.
* Fixed broken functionality within the LL protocol on gfx950 by disabling inlining of LLGenericOp kernels.

### Added

* Added new GPU target `gfx950`.
* Added support for `unroll=1` in device-code generation to improve performance,
* Set a default of 112 channels for a single node with `8 * gfx950`,
* Enabled LL128 protocol on `gfx950`.
* Added MSCCL support for AllGather multinode gfx942/gfx950 (i.e., 16 and 32 GPUs). To enable, set the environment variable `RCCL_MSCCL_FORCE_ENABLE=1`. Max message size for MSCCL AllGather usage is `12292 * sizeof(datatype) * nGPUs`.
* Thread thresholds for LL/LL128 are selected in Tuning Models for the MI300X. This impacts the number of channels used for AG and RS. Channel tuning model is bypassed if `NCCL_THREAD_THRESHOLDS`, `NCCL_MIN_NCHANNELS', or 'NCCL_MAX_NCHANNELS` are set.
* Multi-node tuning for AllGather, AllReduce, and ReduceScatter that leverages LL/LL64/LL128 protocol to use nontemporal vector load/store for tunable message size ranges.
* LL/LL128 usage ranges for AR, AG, and RS are part of the tuning models, which enable architecture-specific tuning in conjunction with the existing Rome Models scheme in RCCL.
* Two new APIs are exposed as part of an initiative to separate RCCL code. These APIs are `rcclGetAlgoInfo` and `rcclFuncMaxSendRecvCount`. However, user-level invocation requires that RCCL be built with `RCCL_EXPOSE_STATIC` enabled.
* Enabled double-buffering in `reduceCopyPacks` to trigger pipelining, especially to overlap `bf16` arithmetic and bridge the gap between `fp32` performance and `bf16` for both `gfx942` and `gfx950`. Pipelining has been made tunable via `rcclSetPipelining`, similar to algorithms/protocols so that regression is avoided in certain message sizes.
* Added a direct allgather algorithm. This is enabled by default for multi-node if there are 16 nodes or fewer. The message size threshold is 4MB.
* Added `RCCL_OVERRIDE_PROTO` and `RCCL_OVERRIDE_ALGO` to allow direct replacement of protocol and algorithm choices. Unlike `NCCL_PROTO` and `NCCL_ALGO`, which re-run the model across enabled combinations and may not guarantee the intended override, these new options enforce the specified selections explicitly.

### Changed

* Compatibility with NCCL 2.23.4.
* Compatibility with NCCL 2.24.3.
* Compatibility with NCCL 2.25.1.
* Compatibility with NCCL 2.26.6.

### Optimized
* Improved the performance of the `FP8` Sum operation by upcasting to `FP16`.

### Known Issues
* When running this version of RCCL using ROCm versions earlier than 6.4.0, the user must set the environment flag `HSA_NO_SCRATCH_RECLAIM=1`.

## RCCL 2.22.3 for ROCm 6.4.2

### Added

* Added support for the LL128 protocol on gfx942.

## RCCL 2.22.3 for ROCm 6.4.1

### Resolved issues

* Fixed the accuracy issue for MSCCLPP `allreduce7` kernel in graph mode.
* Fixed IntraNet performance.
* Fixed an issue where, in rare circumstances, the application could stop responding due to a proxy thread synchronization issue.

### Known issues

* When splitting a communicator using `ncclCommSplit` in some GPU configurations, MSCCL initialization can cause a segmentation fault.
  The recommended workaround is to disable MSCCL with `export RCCL_MSCCL_ENABLE=0`.
* Within the RCCL-UnitTests test suite, failures occur in tests ending with the `ManagedMem` and `ManagedMemGraph` suffixes. These failures only affect the test results and do not affect the RCCL component itself. This issue will be resolved in the next major release.

## RCCL 2.22.3 for ROCm 6.4.0

### Added

* `RCCL_SOCKET_REUSEADDR` and `RCCL_SOCKET_LINGER` environment parameters.
* Setting `NCCL_DEBUG=TRACE NCCL_DEBUG_SUBSYS=VERBS` will generate traces for fifo and data `ibv_post_sends`.
* Added `--log-trace` flag to enable traces through the install.sh script (e.g. `./install.sh --log-trace`).

### Changed

* Compatibility with NCCL 2.22.3
* Added support for the rail-optimized tree algorithm for the MI300 series. This feature requires the use of all eight GPUs within
  each node. It limits NIC traffic to use only GPUs of the same index across nodes and should not impact performance
  on non-rail-optimized network topologies. The original method of building trees can be enabled by setting the
  environment variable `RCCL_DISABLE_RAIL_TREES=1`.
* Additional debug information about how the trees are built can be logged to the GRAPH logging subsys by setting
  `RCCL_OUTPUT_TREES=1`.
* Added documentation about the NPS4 and CPX partition modes performance benefits on the MI300X.

## RCCL 2.21.5 for ROCm 6.3.1

### Added

### Changed

* Enhanced user documentation

### Resolved issues

* Corrected user help strings in `install.sh`

## RCCL 2.21.5 for ROCm 6.3.0

### Added

* MSCCL++ integration for AllReduce and AllGather on gfx942
* Performance collection to rccl_replayer
* Tuner Plugin example for MI300
* Tuning table for large number of nodes
* Support for amdclang++
* Allow NIC ID remapping using `NCCL_RINGS_REMAP` environment variable

### Changed

* Compatibility with NCCL 2.21.5
* Increased channel count for MI300X multi-node
* Enabled MSCCL for single-process multi-threaded contexts
* Enabled gfx12
* Enabled CPX mode for MI300X
* Enabled tracing with rocprof
* Improved version reporting
* Enabled GDRDMA for Linux kernel 6.4.0+

### Resolved issues

* Fixed model matching with PXN enable

## RCCL 2.20.5 for ROCm 6.2.1
### Fixed
- GDR support flag now set with DMABUF
### Known issues
- On systems running Linux kernel 6.8.0, such as Ubuntu 24.04, Direct Memory Access (DMA) transfers between the GPU and NIC are disabled and impacts multi-node RCCL performance.
  - This issue was reproduced with RCCL 2.20.5 (ROCm 6.2.0 and 6.2.1) on systems with Broadcom Thor-2 NICs and affects other systems with RoCE networks using Linux 6.8.0 or newer.
  - Older RCCL versions are also impacted.
  - This issue will be addressed in a future ROCm release.

## RCCL 2.20.5 for ROCm 6.2.0
### Changed
- Compatibility with NCCL 2.20.5
- Compatibility with NCCL 2.19.4
- Performance tuning for some collective operations on MI300
- Enabled NVTX code in RCCL
- Replaced rccl_bfloat16 with hip_bfloat16
- NPKit updates:
  - Removed warm-up iteration removal by default, need to opt in now
  - Doubled the size of buffers to accommodate for more channels
- Modified rings to be rail-optimized topology friendly
- Replaced ROCmSoftwarePlatform links with ROCm links
### Added
- Support for fp8 and rccl_bfloat8
- Support for using HIP contiguous memory
- Implemented ROC-TX for host-side profiling
- Enabled static build
- Added new rome model
- Added fp16 and fp8 cases to unit tests
- New unit test for main kernel stack size
- New -n option for topo_expl to override # of nodes
- Improved debug messages of memory allocations
### Fixed
- Bug when configuring RCCL for only LL128 protocol
- Scratch memory allocation after API change for MSCCL

## RCCL 2.18.6 for ROCm 6.1.0
### Changed
- Compatibility with NCCL 2.18.6

## RCCL 2.18.3 for ROCm 6.0.0
### Changed
- Compatibility with NCCL 2.18.3

## RCCL 2.17.1-1 for ROCm 5.7.0
### Changed
- Compatibility with NCCL 2.17.1-1
- Performance tuning for some collective operations
### Added
- Minor improvements to MSCCL codepath
- NCCL_NCHANNELS_PER_PEER support
- Improved compilation performance
- Support for gfx94x
### Fixed
- Potential race-condition during ncclSocketClose()

## RCCL 2.16.2 for ROCm 5.6.0
### Changed
- Compatibility with NCCL 2.16.2
### Fixed
- Remove workaround and use indirect function call

## RCCL 2.15.5 for ROCm 5.5.0
### Changed
- Compatibility with NCCL 2.15.5
- Unit test executable renamed to rccl-UnitTests
### Added
- HW-topology aware binary tree implementation
- Experimental support for MSCCL
- New unit tests for hipGraph support
- NPKit integration
### Fixed
- rocm-smi ID conversion
- Support for HIP_VISIBLE_DEVICES for unit tests
- Support for p2p transfers to non (HIP) visible devices
### Removed
- Removed TransferBench from tools.  Exists in standalone repo: https://github.com/ROCm/TransferBench

## RCCL-2.13.4 for ROCm 5.4.0
### Changed
- Compatibility with NCCL 2.13.4
- Improvements to RCCL when running with hipGraphs
- RCCL_ENABLE_HIPGRAPH environment variable is no longer necessary to enable hipGraph support
- Minor latency improvements
### Fixed
- Resolved potential memory access error due to asynchronous memset

## RCCL-2.12.10 for ROCm 5.3.0
### Changed
- Improvements to LL128 algorithms
### Added
- Adding initial hipGraph support via opt-in environment variable RCCL_ENABLE_HIPGRAPH
- Integrating with NPKit (https://github.com/microsoft/NPKit) profiling code

## RCCL-2.12.10 for ROCm 5.2.3
### Added
- Compatibility with NCCL 2.12.10
- Packages for test and benchmark executables on all supported OSes using CPack.
- Adding custom signal handler - opt-in with RCCL_ENABLE_SIGNALHANDLER=1
  - Additional details provided if Binary File Descriptor library (BFD) is pre-installed
- Adding support for reusing ports in NET/IB channels
  - Opt-in with NCCL_IB_SOCK_CLIENT_PORT_REUSE=1 and NCCL_IB_SOCK_SERVER_PORT_REUSE=1
  - When "Call to bind failed : Address already in use" error happens in large-scale AlltoAll
    (e.g., >=64 MI200 nodes), users are suggested to opt-in either one or both of the options
    to resolve the massive port usage issue
  - Avoid using NCCL_IB_SOCK_SERVER_PORT_REUSE when NCCL_NCHANNELS_PER_NET_PEER is tuned >1
### Removed
- Removed experimental clique-based kernels

## RCCL-2.11.4 for ROCm 5.2.0
### Changed
- Unit testing framework rework
- Minor bug fixes
### Known issues
- Managed memory is not currently supported for clique-based kernels

## RCCL-2.11.4 for ROCm 5.1.0
### Added
- Compatibility with NCCL 2.11.4
### Known issues
- Managed memory is not currently supported for clique-based kernels

## RCCL-2.10.3 for ROCm 5.0.0
### Added
- Compatibility with NCCL 2.10.3
### Known issues
- Managed memory is not currently supported for clique-based kernels

## RCCL-2.9.9 for ROCm 4.5.0
### Changed
- Packaging split into a runtime package called rccl and a development package called rccl-devel. The development package depends on runtime. The runtime package suggests the development package for all supported OSes except CentOS 7 to aid in the transition. The suggests feature in packaging is introduced as a deprecated feature and will be removed in a future rocm release.
### Added
- Compatibility with NCCL 2.9.9
### Known issues
- Managed memory is not currently supported for clique-based kernels

## [RCCL-2.8.4 for ROCm 4.3.0]
### Added
- Ability to select the number of channels to use for clique-based all reduce (RCCL_CLIQUE_ALLREDUCE_NCHANNELS).  This can be adjusted to tune for performance when computation kernels are being executed in parallel.
### Optimizations
- Additional tuning for clique-based kernel AllReduce performance (still requires opt in with RCCL_ENABLE_CLIQUE=1)
- Modification of default values for number of channels / byte limits for clique-based all reduce based on device architecture
### Changed
- Replaced RCCL_FORCE_ENABLE_CLIQUE to RCCL_CLIQUE_IGNORE_TOPO
- Clique-based kernels can now be enabled on topologies where all active GPUs are XGMI-connected
- Topologies not normally supported by clique-based kernels require RCCL_CLIQUE_IGNORE_TOPO=1
### Fixed
- Install script '-r' flag invoked alone no longer incorrectly deletes any existing builds.
### Known issues
- Managed memory is not currently supported for clique-based kernels

## [RCCL-2.8.4 for ROCm 4.2.0]
### Added
- Compatibility with NCCL 2.8.4

### Optimizations
- Additional tuning for clique-based kernels
- Enabling GPU direct RDMA read from GPU
- Fixing potential memory leak issue when re-creating multiple communicators within same process
- Improved topology detection
### Known issues
- None

## [RCCL-2.7.8 for ROCm 4.1.0]
### Added
- Experimental support for clique-based kernels (opt in with RCCL_ENABLE_CLIQUE=1)
- Clique-based kernels may offer better performance for smaller input sizes
- Clique-based kernels are currently only enabled for AllReduce under a certain byte limit (controlled via RCCL_CLIQUE_ALLREDUCE_BYTE_LIMIT)
### Optimizations
- Performance improvements for Rome-based systems
### Known issues
- Clique-based kernels are currently experimental and have not been fully tested on all topologies.  By default, clique-based kernels are disabled if the detected topology is not supported (override with RCCL_FORCE_ENABLE_CLIQUE)
- Clique-based kernels may hang if there are differences between environment variables set across ranks.
- Clique-based kernels may fail if the input / output device pointers are not the base device pointers returned by hipMalloc.


## [RCCL-2.7.8 for ROCm 3.9.0]
### Added
- Adding support for alltoallv RCCL kernel
### Optimizations
- Modifications to topology based on XGMI links
### Known issues
- None

## [RCCL-2.7.6 for ROCm 3.8.0]
### Added
- Support for static library builds
### Known issues
- None

## [RCCL-2.7.6 for ROCm 3.7.0]
### Added
- Updated to RCCL API version of 2.7.6
- Added gather, scatter and all-to-all collectives

## [RCCL-2.7.0 for ROCm 3.6.0]
### Added
- Updated to RCCL API version of 2.6.4

## [RCCL-2.7.0 for ROCm 3.5.0]
### Added
- Compatibility with NCCL 2.6
- Network interface improvements with API v3
### Optimizations
- Fixing issues and built time improvements for hip-clang
- Network topology detection
- Improved CPU type detection
- Infiniband adaptive routing support
### Changed
- Switched to hip-clang as default compiler
### Deprecated
- Deprecated hcc build
