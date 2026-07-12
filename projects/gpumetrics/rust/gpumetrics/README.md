# gpumetrics (Rust)

Safe, idiomatic Rust bindings for the **gpumetrics** GPU-metrics collector.

This crate wraps the flat C API in [`gpumetrics-sys`] with owned Rust types: a
[`Collector`] that frees its handle on drop, topology / metric metadata as plain
structs, and reads that return a typed [`Value`] plus a [`Status`].

```rust,no_run
use gpumetrics::Collector;

let collector = Collector::new()?;
for gpu in collector.gpus() {
    println!("gpu {}: {} [{}]", gpu.ordinal(), gpu.name(), gpu.bdf());
}
if let Some(entity) = collector.resolve("gpu:0") {
    if let Some(v) = collector.read(&entity, "temp.edge").value() {
        println!("temp.edge = {v}");
    }
}
# Ok::<(), gpumetrics::Status>(())
```

## Building

The C library `libgpumetrics.so` must be built first (see the parent project).
Point the `-sys` build at it:

```bash
GPUMETRICS_LIB_DIR=/path/to/gpumetrics/build/src cargo build
```

Environment overrides (consumed by `gpumetrics-sys/build.rs`):

- `GPUMETRICS_INCLUDE_DIR`: header dir (default `../../include`)
- `GPUMETRICS_LIB_DIR`: dir with `libgpumetrics.so` (default `../../build/src`)

## Example

```bash
GPUMETRICS_LIB_DIR=$PWD/../build/src \
LD_LIBRARY_PATH=$PWD/../build/src \
GPUMETRICS_PLUGIN_PATH=$PWD/../build/plugins/amdsmi \
cargo run --example discover
```

## Testing

Tests run against the hardware-free mock plugins (no GPU needed):

```bash
GPUMETRICS_LIB_DIR=$PWD/../build/src \
LD_LIBRARY_PATH=$PWD/../build/src \
GPUM_TEST_PLUGIN_DIR=$PWD/../build/tests \
cargo test
```

[`gpumetrics-sys`]: https://docs.rs/gpumetrics-sys
