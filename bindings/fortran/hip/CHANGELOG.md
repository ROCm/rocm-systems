# Changelog for the HIP Fortran binding

## Unreleased

### Added

* Generated Fortran bindings for the HIP runtime API, as a single `hip` module: `use hip`, link
  `hip::hip_fortran`. A standalone CMake project in `bindings/fortran/hip`, built against an
  installed HIP. Controlled by `BUILD_FORTRAN_BINDINGS` (`ON` on Linux, `OFF` on Windows, where
  ROCm ships no Fortran compiler), `BUILD_FORTRAN_CLIENTS` (the binding tests in `test/`) and
  `FORTRAN_ARRAY_INTERFACES` (`assumed-shape` by default; `none` or `assumed-rank`). Found with
  `find_package(hip-fortran)`; the archive and `.mod` files install per compiler under
  `<libdir>/fortran/<compiler>` and `<includedir>/fortran/<compiler>`, the tests under
  `<bindir>/hip-fortran`.
  The archive has one section per procedure: link with `-Wl,--gc-sections` to keep only the wrappers
  you call.
