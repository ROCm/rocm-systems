# Correct the ROCr-fix Clippy verification

The original ad hoc Clippy command reused the runtime build's configured
Cargo environment. Its `CARGO_ENCODED_RUSTFLAGS` includes
`--print=native-static-libs`, which the build needs to discover native link
requirements. Clippy treats a rustc `--print` invocation as compiler
passthrough. Successful exit of those earlier commands therefore did not
establish lint coverage, even with `--deny warnings` after Cargo's separator.

Remove that one build-only argument for Clippy, retaining the selected
compiler, target, offline/frozen dependency settings and remaining codegen
flags. Do not remove it from the ordinary runtime build, which consumes its
output. A reproducible CMake wrapper for the configured build is:

```cmake
include("<build>/runtime-rust-config.cmake")
string(ASCII 31 unit_separator)
string(REPLACE "--print=native-static-libs${unit_separator}" ""
  _runtime_cargo_command "${_runtime_cargo_command}")
execute_process(COMMAND ${_runtime_cargo_command}
  clippy --workspace --all-targets --all-features --frozen
  --profile release -- --deny warnings
  WORKING_DIRECTORY "<runtimes>"
  COMMAND_ERROR_IS_FATAL ANY)
```

The corrected invocation initially exposed documentation and test idiom
warnings in the earlier queue-decoder, polling and imported-VM fixes. Their
cleanup adds missing field/function documentation, uses explicit raw
pointers at FFI test calls, and confines expected test unwraps to the test
module. It also uses checked subtraction for the polling-window boundary.
There is no change to production execution or the original runtime build.

The corrected full workspace command passes after cleanup and the current
core-capture changes. Rustfmt and the full native/libhsa/loader host tests
also pass. Local command and output evidence is in
`target/rocr-test-fixes/clippy.cmake`, `core-clippy-real.log`,
`core-final-clippy.log`, and `core-all-host-tests.log`.
