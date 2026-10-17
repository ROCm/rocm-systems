# Changelog for the rocm-systems Fortran bindings

## Unreleased

### Added
* Generated Fortran bindings for the HIP runtime API, as a single `hip` module: `use hip`, link
  `hip::hip_fortran`. A standalone CMake project, built against an installed HIP. Controlled by
  `BUILD_FORTRAN_BINDINGS` (`ON` on Linux, `OFF` on Windows, where ROCm ships no Fortran
  compiler), `BUILD_FORTRAN_CLIENTS` and `FORTRAN_ARRAY_INTERFACES` (`none`, `assumed-shape`,
  `assumed-rank`). Found with `find_package(hip-fortran)`; installed per compiler, the archive
  under `<libdir>/fortran/<compiler>` and the `.mod` files under `<includedir>/fortran/<compiler>`.
