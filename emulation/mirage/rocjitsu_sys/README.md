# rocjitsu_sys

Low-level Rust bindings and light RAII wrappers for rocjitsu's public C API.

The crate generates two pieces at build time:

- `ffi`: bindgen output for `rocjitsu/rocjitsu.h`
- `fb`: Rust FlatBuffers types for `simulation_config.fbs` and `checkpoint.fbs`

The top-level wrapper types (`Vm`, `Decoder`, `Executable`, and related handle
types) own rocjitsu handles and release them on drop.

## Build Inputs

By default, the build script expects this repository layout:

```text
emulation/
  mirage/rocjitsu_sys/
  rocjitsu/
```

Override discovery with these environment variables when building against an
installed or out-of-tree rocjitsu:

- `ROCJITSU_ROOT`: rocjitsu source root
- `ROCJITSU_INCLUDE_DIR`: directory containing `rocjitsu/rocjitsu.h`
- `ROCJITSU_SCHEMA_DIR`: directory containing the `.fbs` files
- `ROCJITSU_LIB_DIR`: directory containing `librocjitsu.so`
- `FLATC`: path to a `flatc` binary

## Smoke Test

The integration smoke test exercises generated FlatBuffers types, the decoder
wrapper, dynamic linking to `librocjitsu.so`, and VM creation from a rocjitsu JSON
config:

```sh
cargo test -p rocjitsu_sys --test smoke
```
