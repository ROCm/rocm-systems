# Changelog for hipFile

## (Unreleased) hipFile 0.7.0 

### Added

* Added a Host backend that enables file I/O using host memory. `hipFileBufRegister` now accepts host memory. Only `hipFileRead()` and `hipFileWrite()` accept a host buffer; `hipFileReadAsync()` and `hipFileWriteAsync()` return `hipFileAsyncNotSupported` for one.
* MD and LVM software RAID arrays running pass-through personalities whose member devices are all local NVMe are now fastpath capable, and `ais-check` reports them as such.
* Added the `HIPFILE_DEVICE_STREAM_POOL_SIZE` environment variable, which sets the number of streams in the per-GPU pool that fallback I/O uses for its device copies. The default is 4; setting it to 0 selects the default.
* Added NIXL to the [Frameworks and applications using hipFile](https://rocm.docs.amd.com/projects/hipFile/en/latest/reference/hipFile-integrations.html#frameworks-and-applications-using-hipfile) documentation. NIXL AIS storage backend moves data directly between GPU memory and files on ROCm using `hipFileRead()` and `hipFileWrite()`.

### Changed

* Fallback I/O operations now perform their device copies on a pool of per-GPU non-blocking streams instead of the default stream, so they no longer serialize against other work on the GPU's default (null) stream.
* hipFile now requires CMake 3.25.2 or later, instead of 3.21, to build hipFile, its Python bindings, and the installed examples. Earlier versions couldn't fetch GoogleTest or rocm-cmake, and before 3.22.1, `find_library()` couldn't find system libraries when building with amdclang. Some distributions ship an older CMake, such as Ubuntu 22.04 (3.22.1) and Debian 12 (3.25.1); install a newer one there, for example with `pip install cmake` or from Kitware's APT repository.
* When a failed fastpath I/O is completed by the compatibility path, `hipFileRead()` and `hipFileWrite()` now clear the HIP Runtime's last error.
* The CMake cache variable that selects the include-what-you-use executable when building with `HIPFILE_USE_IWYU` is now `HIPFILE_IWYU_EXE` instead of `IWYU_EXE`. If you set `IWYU_EXE`, set `HIPFILE_IWYU_EXE` instead.
* Building with `HIPFILE_USE_IWYU` no longer runs include-what-you-use on a GoogleTest that the build downloads, so its suggestions now cover only hipFile's own code. Neither include-what-you-use nor clang-tidy runs on a downloaded GoogleTest when set directly through the `CMAKE_C_INCLUDE_WHAT_YOU_USE`, `CMAKE_CXX_INCLUDE_WHAT_YOU_USE`, or `CMAKE_CXX_CLANG_TIDY` cache variables, either.
* After a fastpath I/O fails with `ENOTTY` because the amdgpu kernel driver doesn't support AIS, hipFile now disables the fastpath for the rest of the process, and later I/O goes straight to the fallback path. Previously, every I/O tried the fastpath first, failed with `ENOTTY`, and was then retried on the fallback path. `ais-stats` now counts I/O submitted after the fastpath is disabled under "Fastpath Rejections". If the fallback path is disabled with `HIPFILE_ALLOW_COMPAT_MODE=false`, later `hipFileRead()` and `hipFileWrite()` calls will return `-hipFileInternalError` instead of `-1` with `errno` set to `ENOTTY`.
* `hipFileStreamRegister()` and `hipFileStreamDeregister()` now treat the default stream (a null stream handle or `hipStreamPerThread`) as a no-op that returns success, matching cuFile. Previously, deregistering the default stream returned `hipFileInvalidValue`.
* `hipFileReadAsync()` and `hipFileWriteAsync()` now hand each transfer to an internal thread pool and use stream wait values to keep stream order, so on devices that support stream wait values, transfers on different streams can run at the same time. Queuing a request no longer waits for the stream.
* `hipFileReadAsync()` and `hipFileWriteAsync()` run on any stream, registered or not; calling `hipFileStreamRegister()` first is optional and only an optimization, as in cuFile. An unregistered stream runs the transfer on the blocking path without being registered, so only streams you explicitly registered need to be deregistered before they are destroyed.
* `hipFileReadAsync()` and `hipFileWriteAsync()` on `hipStreamPerThread` now run synchronously, like the null stream.
* `hipFileStreamDeregister()` now synchronizes the stream before freeing its resources, so it waits for all work queued on the stream, including asynchronous I/O in flight. If the stream processed any asynchronous I/O, deregistering it also calls `hipFree` on the stream's signal memory, which synchronizes all streams on the device; do not deregister a stream that has processed I/O while another stream on the same device is blocked.
* hipFile now reserves the device's greatest stream priority for its fallback I/O streams. For a stream at that priority, `hipFileStreamRegister()` returns `hipFileInvalidValue`, and `hipFileReadAsync()` and `hipFileWriteAsync()` return `hipFileAsyncNotSupported`. Use a lower priority for streams used for asynchronous I/O. Devices with a single stream priority level are not affected.
* The `HIPFILE_WARN_UNSAFE_BUFFER_OPS` CMake option now also hardens the C++ standard library, by defining `_GLIBCXX_ASSERTIONS` (libstdc++) and `_LIBCPP_HARDENING_MODE=_LIBCPP_HARDENING_MODE_FAST` (libc++), so that standard library containers and views check their bounds at run time. The option no longer applies to the example programs, which are also built as C++17 when installed.

### Fixed

* `hipFileRead()`, `hipFileWrite()`, and the asynchronous and batch I/O APIs now fall back to the compatibility path when the amdgpu kernel driver doesn't support AIS (for example, the inbox Linux driver). Previously, they failed with `ENOTTY` ("Inappropriate ioctl for device") or an unrelated error code.
* `hipFileRead()` and `hipFileWrite()` now return `-hipFileHipDriverError` when a HIP Runtime call fails. Use `hipGetLastError()` or `hipPeekAtLastError()` to retrieve the `hipError_t`. Previously, they returned the negated `hipError_t` (for example, `-999` for `hipErrorUnknown`), which isn't a documented return value.
* `ais-check` no longer reports an LVM parity RAID logical volume (`raid4`, `raid5`, `raid6`) as fastpath capable.
* Per-GPU I/O statistics are now attributed to the GPU that owns the I/O buffer instead of the caller's active HIP device. Previously, `hipFileGetStatsL3()` and `ais-stats` could report counters against the wrong GPU when the active device differed from the buffer's device.
* `hipfile.h` now includes `<time.h>`, which declares the `struct timespec` used by `hipFileBatchIOGetStatus()`. Previously, C code that included `<time.h>` after `hipfile.h` got an incompatible pointer type error or warning when passing a `struct timespec *` to `hipFileBatchIOGetStatus()`.
* Configuring hipFile with `HIPFILE_CXX_STANDARD=26` and a CMake older than 3.30, which C++26 requires, now fails with an error that names `HIPFILE_CXX_STANDARD` and the CMake version needed. Previously, the error only said that CMake didn't know the `cxx_std_26` compiler feature.
* Building hipFile with GCC 15 or later no longer warns about every line indented with spaces. The build now uses `-Wleading-whitespace=spaces`, which warns about tabs in indentation, instead of `-Wleading-whitespace=tabs`.
* Optimized builds now correctly enable `_FORTIFY_SOURCE=3` for hipFile when compiling with `-O2`, `-O3`, or `-Os` (i.e., the `Release`, `RelWithDebInfo`, and `MinSizeRel` build types). This applies when using Clang, or GCC 12 or later. Previously, the build checked the wrong CMake variable to detect the build type's optimization level, so `_FORTIFY_SOURCE` only activated when `-O2`, `-O3`, or `-Os` appeared directly in `CMAKE_CXX_FLAGS`.  This also meant multi-config generators (e.g., Ninja Multi-Config) were never handled. The check now runs per-configuration, covering all standard optimized build types and multi-config generators. With Clang and glibc 2.40 or later, fortified builds turn off `-Wused-but-marked-unused`, which glibc's fortified `printf`-family functions trigger.
* `hipFileReadAsync()` and `hipFileWriteAsync()` on the null stream now finish the transfer before returning, as documented. Previously, they queued the transfer and returned before it finished.
* The bandwidth reported by `ais-stats` and the Stats API (`hipFileGetStatsL1()`/`hipFileGetStatsL3()`, `read_bw_bytes_per_sec`/`write_bw_bytes_per_sec`) is now derived from the elapsed wall-clock time during which I/O was in flight, rather than the sum of the per-operation latencies. When several threads issue I/O concurrently, their operations overlap in time, as such, the old sum-of-latencies denominator understated the reported bandwidth by roughly the number of concurrent threads. Latency is unchanged. The per-size-bucket bandwidth histogram in the `ais-stats` report remains derived from per-operation latency and reflects per-I/O throughput, not aggregate concurrency. See [issue #11919](https://github.com/ROCm/rocm-systems/issues/11919).
* The exported `hip::hipfile` CMake target no longer requires C++20 of the targets that link it. Previously, it carried hipFile's own C++ standard as a usage requirement, so CMake compiled every C++ target that linked it as at least C++20 (or the standard set by `HIPFILE_CXX_STANDARD`), even if that target asked for an older one. Consumers that relied on getting C++20 this way must now request it themselves, for example with `target_compile_features(<target> PRIVATE cxx_std_20)`.
* `hipfile.h` no longer triggers `-Wundef` when compiled as C++11 or C++14. Previously, it tested `__STDC_VERSION__`, which C++ doesn't define, so those builds failed with `-Wundef -Werror`.

### Removed

* Removed the `HIPFILE_ASYNC_BUFFER_SIZE` environment variable. It sized a per-stream host bounce buffer that asynchronous I/O no longer uses.

### Known issues

* Asynchronous transfers on an unregistered stream, or on any stream on a device that doesn't support stream wait values, run one at a time, because the ROCm runtime runs stream host functions on a single thread, and a running transfer also delays host functions queued on other streams. Register a stream on a wait-value-capable device to avoid this.
* A hipFile call that uses the fallback path can hang if the application has blocked one of its own greatest-priority HIP streams until that call returns. Don't block a greatest-priority stream on hipFile I/O.

## hipFile 0.6.0 for ROCm 10.2.0

### Added

* `hipFileGetStatsL3()` now populates the `uuid` field of each `hipFilePerGpuStats` entry with the raw 16-byte GPU UUID, zero-filled when the device cannot report one.
* Added a batch API example, `batch-roundtrip`.
* Added reference documentation for the Stats API.
* Added a documentation page listing frameworks and applications that use hipFile, including LMCache, fastsafetensors, rocAL, PyTorch, fio, and IOR.

### Changed

* The hipFile CMake options have been renamed from `AIS_*` to `HIPFILE_*` (e.g., `AIS_CXX_STANDARD` is now `HIPFILE_CXX_STANDARD`). The old `AIS_*` names still work but are deprecated and emit a CMake deprecation warning.
* The `HIPFILE_WARN_UNSAFE_BUFFER_OPS` CMake option now defaults to `OFF`.
* Documented that Linux kernels built with `CONFIG_PCI_P2PDMA=y` are required for fastpath.

### Fixed

* On Windows Subsystem for Linux (WSL2), hipFile now sends all I/O to the fallback backend, since the fastpath is not yet implemented on WSL.
* Asynchronous fastpath operations now fail over to the fallback backend on `ENODEV` or `EREMOTEIO`, matching the synchronous and batch paths.
* hipFile now returns `hipFileGetNewFDFailed` instead of `hipFileInternalError` when the process or system runs out of file descriptors. API calls that need a new file descriptor, such as `hipFileHandleRegister()`, can return this error.
* The `HIPFILE_WARN_UNSAFE_BUFFER_OPS` CMake option now applies the clang `-Wunsafe-buffer-usage` and `-fsafe-buffer-usage-suggestions` flags to C++ sources. Previously, the flags were never passed to the compiler.

### Removed

### Known issues

* The bandwidth reported by `ais-stats` and the Stats API is derived from the sum of the per-operation latencies instead of the elapsed wall-clock time, so when several threads issue I/O concurrently, the reported bandwidth is understated by roughly the number of concurrent threads. See [issue #11919](https://github.com/ROCm/rocm-systems/issues/11919).
* Asynchronous fastpath I/O, added in hipFile 0.5.0 for ROCm 10.1.0, performs poorly. The ROCm runtime serializes all stream host functions onto a single thread, so asynchronous fastpath operations cannot run concurrently.

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
