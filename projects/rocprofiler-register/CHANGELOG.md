# Changelog for rocprofiler-register

## rocprofiler-register 0.7.0

### Added

  - CHANGELOG added.

### Resolved issues

  - Attach mode could load a stale `librocprofiler-sdk-attach` colocated with rocprofiler-register (e.g., from a ROCm install) instead of the one on `LD_LIBRARY_PATH`:
    - The attach library search now prefers the directory of an already-loaded rocprofiler-sdk, then `LD_LIBRARY_PATH` directories containing both the attach library and rocprofiler-sdk, then the rocprofiler-register directory.
    - On attach, rocprofiler-sdk is loaded from the same directory as the attach library, so the two always match. If that directory has no rocprofiler-sdk, a warning is logged and rocprofiler-sdk is loaded from the rocprofiler-register directory or the default library search path.
