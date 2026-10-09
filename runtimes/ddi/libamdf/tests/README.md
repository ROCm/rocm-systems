<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# AMDF validation

Run these commands from `runtimes`:

```sh
cargo test --workspace --all-targets --locked
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked
python3 ddi/libamdf/tests/abi/check_layout.py
```

The ABI probe compares independently compiled C and Rust layouts against the
vendored AMDF headers. Those headers and the upstream CTS are pinned to
`hrx-system@bd24215e6a5d1e12570c892356fd5b343ac38a6d`.

On a Linux x86-64 host with an accessible GFX1201 GPU, run the opt-in
[cross-frontend queue fault probes](../../rocddi/tests/gpu/README.md):

```sh
ddi/rocddi/tests/gpu/run_queue_faults.sh /tmp/rocddi-cmake/lib
```

The C sources in `tests/abi` provide native ABI, memory, and queue probes.
The [GPU examples](../examples/README.md) provide build commands and checked
workloads for memory, PM4, AQL, and SDMA. Cargo tests do not execute those
GPU workloads or the upstream CTS. The [API support map](../docs/api-support.md)
lists the implemented requests and their hardware limits.
