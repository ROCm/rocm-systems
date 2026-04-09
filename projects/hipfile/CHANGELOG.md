# Changelog for hipFile

## UNRELEASED - hipFile 0.2.0
### Added
* The examples can be installed to `share/doc/examples/*` by setting `AIS_INSTALL_EXAMPLES` (formerly `AIS_BUILD_EXAMPLES`) to ON

### Changed
* Renamed `hipFileOpStatusError()` to `hipFileGetOpErrorString()`
* The `hipfile-doc` CMake target no longer exists (just use `doc`)
* The `doc` CMake target only exists if the `AIS_BUILD_DOCS` option is enabled
* `hipFileRead()`/`hipFileWrite()` will transfer at most 0x7ffff000 (2,147,479,552) bytes returning the number of bytes actually transferred
* The CMake namespace was changed from `roc::` to `hip::`
* `AIS_BUILD_EXAMPLES` has been renamed to `AIS_INSTALL_EXAMPLES`
* `AIS_USE_SANITIZERS` now also enables the following sanitizers: integer, float-divide-by-zero, local-bounds, vptr, nullability (in addition to address, leak, and undefined). Sanitizers should also now emit usable stack trace info.
* The AIS optimized IO path will automatically fallback to the POSIX IO path if a failure occurs and the compatibility mode has not been disabled.
* Added check in the Fastpath/AIS backend to ensure the HIP Runtime is initialized. This avoids causing a segfault in the HIP Runtime.
* The default CMake build type was changed from `Debug` to `RelWithDebInfo`
* Added file type and file system validation in Fastpath. Fastpath will only accept IO targeting block devices or regular files backed by xfs or ext4 with ordered journaling mode. Other file systems can be explicitly allowed via the `HIPFILE_UNSUPPORTED_FILE_SYSTEMS` environment variable.
* The docs directory has been reorganized and `AIS_BUILD_DOCS` will now generate Sphinx output for the web.

### Removed
* The rocFile library has been completely removed and the code is now a part of hipFile.
* The hipify patch was removed. You can get hipify with hipFile support from the main HIPIFY repo at https://github.com/ROCm/HIPIFY. The `amd-develop` branch has hipFile support, but this has not been publicly released yet.
* Dropped the `AIS_USE_INTEGER_SANITIZER` CMake option. This has been rolled into the broader `AIS_USE_SANITIZERS` option.
* Dropped support for GNU sanitizers (probably temporary)

### Limitations
* The batch API calls are not supported w/ an AMD backend
* The async API calls are not supported w/ an AMD backend

### Known issues
