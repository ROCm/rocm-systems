# Changelog for the ROCTx Fortran binding

## Unreleased

### Added

* Generated Fortran bindings for the ROCTx API, as a single `roctx` module: `use roctx`, link
  `roc::roctx_fortran`. A standalone CMake project in `bindings/fortran/roctx`, built against an
  installed `rocprofiler-sdk-roctx`; it does not need HIP. Controlled by `BUILD_FORTRAN_BINDINGS`
  (`ON` on Linux, `OFF` on Windows, where ROCm ships no Fortran compiler) and
  `BUILD_FORTRAN_CLIENTS` (the binding test in `test/`). Found with `find_package(roctx-fortran)`;
  the archive and `.mod` files install per compiler under `<libdir>/fortran/<compiler>` and
  `<includedir>/fortran/<compiler>`, the test under `<bindir>/roctx-fortran`.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
