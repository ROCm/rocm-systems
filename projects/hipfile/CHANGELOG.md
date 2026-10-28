# Changelog for hipFile

## hipFile 0.6.0 for ROCm 10.2.0

### Added

* MD and LVM software RAID arrays running pass-through personalities whose member devices are all local NVMe are now fastpath capable, and `ais-check` reports them as such.
* `hipFileGetStatsL3()` now populates the `uuid` field of each `hipFilePerGpuStats` entry with the raw 16-byte GPU UUID, zero-filled when the device cannot report one.
* Added a batch API example, `batch-roundtrip`.
* Added reference documentation for the Stats API.

### Changed

* The hipFile CMake options have been renamed from `AIS_*` to `HIPFILE_*` (e.g., `AIS_CXX_STANDARD` is now `HIPFILE_CXX_STANDARD`). The old `AIS_*` names still work but are deprecated and emit a CMake deprecation warning.
* The `HIPFILE_WARN_UNSAFE_BUFFER_OPS` CMake option now defaults to `OFF`.
* Documented that Linux kernels built with `CONFIG_PCI_P2PDMA=y` are required for fastpath.
* `hipFileStreamRegister()` and `hipFileStreamDeregister()` now treat the default stream (a null stream handle or `hipStreamPerThread`) as a no-op that returns success, matching cuFile. Previously, deregistering the default stream returned `hipFileInvalidValue`.
* `hipFileReadAsync()` and `hipFileWriteAsync()` now hand each transfer to an internal thread pool and use stream wait values to keep stream order, so on devices that support stream wait values, transfers on different streams can run at the same time. Queuing a request no longer waits for the stream.
* `hipFileReadAsync()` and `hipFileWriteAsync()` now register a stream implicitly on its first use, with no flags, as cuFile does, so calling `hipFileStreamRegister()` first is optional. Registering a stream that was registered implicitly applies its flags to later requests. Deregister every stream used for asynchronous I/O before destroying it.
* `hipFileReadAsync()` and `hipFileWriteAsync()` on `hipStreamPerThread` now run synchronously, like the null stream.
* `hipFileStreamDeregister()` now synchronizes the stream before freeing its resources, so it waits for all work queued on the stream, including asynchronous I/O in flight.
* hipFile now reserves the device's greatest stream priority for its fallback I/O streams. For a stream at that priority, `hipFileStreamRegister()` returns `hipFileInvalidValue`, and `hipFileReadAsync()` and `hipFileWriteAsync()` return `hipFileAsyncNotSupported`. Use a lower priority for streams used for asynchronous I/O. Devices with a single stream priority level are not affected.

### Fixed

* `hipFileReadAsync()` and `hipFileWriteAsync()` on the null stream now finish the transfer before returning, as documented. Previously, they queued the transfer and returned before it finished.
* Asynchronous fastpath operations now fail over to the fallback backend on `ENODEV` or `EREMOTEIO`, matching the synchronous and batch paths.
* hipFile now returns `hipFileGetNewFDFailed` instead of `hipFileInternalError` when the process or system runs out of file descriptors. API calls that need a new file descriptor, such as `hipFileHandleRegister()`, can return this error.
* `ais-check` no longer reports an LVM parity RAID logical volume (`raid4`, `raid5`, `raid6`) as fastpath capable.
* The `HIPFILE_WARN_UNSAFE_BUFFER_OPS` CMake option now applies the clang `-Wunsafe-buffer-usage` and `-fsafe-buffer-usage-suggestions` flags to C++ sources. Previously, the flags were never passed to the compiler.
* Per-GPU I/O statistics are now attributed to the GPU that owns the I/O buffer instead of the caller's active HIP device. Previously, `hipFileGetStatsL3()` and `ais-stats` could report counters against the wrong GPU when the active device differed from the buffer's device.
* `hipfile.h` now includes `<time.h>`, which declares the `struct timespec` used by `hipFileBatchIOGetStatus()`. Previously, C code that included `<time.h>` after `hipfile.h` got an incompatible pointer type error or warning when passing a `struct timespec *` to `hipFileBatchIOGetStatus()`.

### Removed

* Removed the `HIPFILE_ASYNC_BUFFER_SIZE` environment variable. It sized a per-stream host bounce buffer that asynchronous I/O no longer uses.

### Known issues

* The bandwidth reported by `ais-stats` and the Stats API is derived from the sum of the per-operation latencies instead of the elapsed wall-clock time, so when several threads issue I/O concurrently, the reported bandwidth is understated by roughly the number of concurrent threads. See [issue #11919](https://github.com/ROCm/rocm-systems/issues/11919).
* On devices that don't support stream wait values, asynchronous transfers run one at a time, because the ROCm runtime runs stream host functions on a single thread, and a running transfer also delays host functions queued on other streams.
* A hipFile call that uses the fallback path can hang if the application has blocked one of its own greatest-priority HIP streams until that call returns. Don't block a greatest-priority stream on hipFile I/O.

## hipFile 0.5.0 for ROCm 10.1.0

### Added

* Added a Stats API for querying hipFile I/O statistics. `hipFileGetStatsL1()`, `hipFileGetStatsL2()`, and `hipFileGetStatsL3()` return progressively more detailed counters: basic I/O and operation counts (Level 1), I/O size histograms (Level 2), and per-GPU statistics (Level 3).
* `ais-check` now detects SR-IOV virtual function (VF) GPUs via `amd-smi` and warns when one is present. hipFile's fastpath is only supported on GPU physical functions (PFs); on a VF, I/O falls back to the compatibility path. The check is skipped if `amd-smi` is unavailable.
* `hipFileReadAsync()` and `hipFileWriteAsync()` now support the AIS fastpath backend, enabling asynchronous GPU-direct I/O enqueued on a HIP stream. Transparent async backend failover to the slowpath is not currently supported for async fastpath operations.
* Batch operations now execute on an internal thread pool, enabling batch API support on the AMD backend. Together with async fastpath support, this resolves the 0.3.0 limitation where batch and async API calls were unsupported on the AMD backend.
* Added the `HIPFILE_ASYNC_BUFFER_SIZE` environment variable to control the size of the host bounce buffer used for asynchronous fallback I/O. The default size is 16 MiB; setting it to `0` uses the default.
* `ais-check` now reports whether an LVM logical volume supports fastpath.

### Changed

* The synchronous fallback I/O path now sets the active HIP device to the buffer's GPU before `hipMemcpy` and restores the caller's device afterward, fixing copies that could run against the wrong device context.
* Asynchronous fallback I/O now reuses a single per-stream bounce buffer, splitting large transfers into chunks that fit the buffer, to reduce the memory footprint of asynchronous workloads.

### Fixed

* Corrected CMake ROCm path detection so out-of-tree builds locate the correct ROCm installation.

### Removed

### Known issues

* Asynchronous operations that use the fastpath backend will not retry on the fallback backend. If asynchronous operations have proper alignment, they now run on the fastpath backend. If the fastpath device lookup fails or the P2P DMA transfer is not supported between the devices, the asynchronous operation will now fail.

## hipFile 0.4.0 for ROCm 10.0.0

### Added

* A KFD-based alternative check for P2P DMA support was added to `ais-check`. This inspects the `capability` property under `/sys/class/kfd/kfd/topology/nodes/*/properties`.
* Added support for Logical Volume Manager (LVM) volumes with a maximum of 16 extents
* Added guides for setting up storage targets to the documentation

### Changed

* `ais-check` now lists the AIS-capable file system mounts detected on the system and fails if none are found.
* Fastpath-only tests are now automatically skipped on systems that do not support the AIS fastpath instead of failing. Running ctest in verbose mode (`ctest -V`) will provide the reason why the test was skipped.
* Updated INSTALL.md to point to official install docs
* hipFileRead and hipFileWrite now return -1 with `errno` set to `EINVAL` for negative offsets instead of returning `-hipFileInternalError`

### Removed

### Known issues

## hipFile 0.3.0 for ROCm 7.14.0

### Added

* Examples can be installed on `share/doc/examples/*` by setting `AIS_INSTALL_EXAMPLES` to `ON`.
* An additional check has been added to the Fastpath/AIS backend to ensure the HIP Runtime is initialized, preventing segmentation faults when using the HIP Runtime.
* Added file type and file system validation in Fastpath. Fastpath will only accept I/O targeting block devices or regular files backed by xfs or ext4 with ordered journaling mode. Other file systems can be explicitly allowed via the `HIPFILE_UNSUPPORTED_FILE_SYSTEMS` environment variable.

### Changed

* `hipFileOpStatusError()` has been renamed to `hipFileGetOpErrorString()`.
* The `hipfile-doc` CMake target has been replaced with `doc`. The `AIS_BUILD_DOCS` CMake option must be enabled to build with this target.
* The CMake namespace has changed from `roc::` to `hip::`
* `AIS_BUILD_EXAMPLES` has been renamed to `AIS_INSTALL_EXAMPLES`
* `AIS_USE_SANITIZERS` now also enables the following sanitizers: integer, float-divide-by-zero, local-bounds, vptr, nullability (in addition to address, leak, and undefined). Sanitizers should also now emit usable stack trace info.
* The AIS optimized I/O path now automatically falls back to the POSIX I/O path if a failure occurs and the compatibility mode has not been disabled.
* The default CMake build type has changed from `Debug` to `RelWithDebInfo`

### Removed

* The rocFile library has been completely removed and the code is now a part of hipFile.
* The hipify patch was removed. hipify with hipFile support can be obtained from the main HIPIFY repo at https://github.com/ROCm/HIPIFY. The `amd-develop` branch has hipFile support, but this has not been officially released yet.
* The `AIS_USE_INTEGER_SANITIZER` CMake option has been removed. Use the `AIS_USE_SANITIZERS` option instead.
* Support for GNU sanitizers has been dropped in this release.

### Known issues

* Batch and async API calls are not supported on the AMD backend
* Poor performance with small I/O sizes (<= 64KiB) and many threads/processes
* Poor performance within QEMU virtual machine when PCIe devices are not attached to PCIe root ports
* High memory usage with many processes
* GPU resets encountered with many processes
