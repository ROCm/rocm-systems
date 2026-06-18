# Changelog

All notable changes to the hipFile Python bindings will be documented in this file.

## [Unreleased]

### Added

- Docstrings for the public Python API.

## [0.2.0.dev1]

### Fixed

- Release the Python GIL during all C function calls to prevent blocking other
  threads during GPU IO operations. This resolves a performance regression
  observed in multi-threaded applications (e.g., LMCache) when switching from
  ctypes-based bindings to the Cython bindings.

### Added

- Complete typing hints for the public API.

## [0.2.0.dev0]

### Added

- Initial release of the hipFile Python bindings.
- Cython-based low-level wrappers for the hipFile C API (`_hipfile.pyx`).
- High-level Pythonic API: `Driver`, `FileHandle`, `Buffer` classes with
  context manager support.
- `hipMalloc` / `hipFree` helpers for GPU memory allocation.
- Error handling via `HipFileException` with hipFile and HIP driver error codes.
- `FileHandleType` and `OpError` enums.
- `get_version()` and `driver_get_properties()` utility functions.
- scikit-build-core based build system.
