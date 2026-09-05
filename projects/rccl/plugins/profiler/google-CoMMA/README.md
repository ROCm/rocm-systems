# CoMMA profiler plugin

Wrapper around Google's [CoMMA](https://github.com/google/CoMMA) profiler, a Rust
crate. `make` clones the upstream repository into `CoMMA/`, symlinks this directory
in as `third_party/nccl/plugins/profiler`, and runs `cargo build`; no CoMMA source
is vendored here.

## Build

CoMMA needs a Rust toolchain plus `protoc`, and its `bindgen` build script needs a
`libclang`, but bindgen has to be pointed at both the library and
the matching clang resource headers, or it fails to find `stddef.h`.

```bash
# Rust (kept out of $HOME by setting RUSTUP_HOME/CARGO_HOME):
export RUSTUP_HOME=/path/to/rust/rustup CARGO_HOME=/path/to/rust/cargo
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- \
    -y --no-modify-path --default-toolchain stable --profile minimal
export PATH=$CARGO_HOME/bin:$PATH
rustup component add rustfmt          # build logs an error without it, then continues

# protoc, if not already installed (any recent protobuf release works):
export PROTOC=/path/to/protoc/bin/protoc

# libclang from ROCm, plus the resource headers bindgen needs:
export LIBCLANG_PATH=$ROCM_PATH/lib/llvm/lib
export BINDGEN_EXTRA_CLANG_ARGS="-I$(ls -d $ROCM_PATH/lib/llvm/lib/clang/*/include | head -1)"

cd plugins/profiler/google-CoMMA
make
```

Output: `CoMMA/target/debug/libnccl_profiler.so`.
