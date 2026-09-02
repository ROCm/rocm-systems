# Changelog for hipFile

## (Unreleased) hipFile 0.5.0

### Added

* `ais-check` now detects SR-IOV virtual function (VF) GPUs via `amd-smi` and warns when one is present. hipFile's fastpath is only supported on GPU physical functions (PFs); on a VF, I/O falls back to the compatibility path. The check is skipped if `amd-smi` is unavailable.

### Changed

* `ais-check` now reports LVM logical volumes as fastpath-capable when their underlying physical volumes are all local NVMe. It walks the device-mapper stack and still marks LVM on any other backing (multipath, dm-crypt, MD RAID, ...) as unsupported.
* The synchronous fallback I/O path now sets the active HIP device to the buffer's GPU before `hipMemcpy` and restores the caller's device afterward, fixing copies that could run against the wrong device context.

### Removed

### Known issues

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
