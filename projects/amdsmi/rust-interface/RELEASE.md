# Releasing `amdsmi` to crates.io

This checklist covers the manual release flow for the `amdsmi` Rust crate.

## Pre-release checklist

1. Confirm the Rust crate version matches the AMD-SMI C library version:

   ```sh
   cargo_version=$(grep -m1 '^version = ' Cargo.toml | cut -d '"' -f 2)
   header_version=$(awk '/AMDSMI_LIB_VERSION_MAJOR / {maj=$3} /AMDSMI_LIB_VERSION_MINOR / {min=$3} /AMDSMI_LIB_VERSION_RELEASE / {rel=$3} END {print maj "." min "." rel}' ../include/amd_smi/amdsmi.h)
   wrapper_version=$(awk '/AMDSMI_LIB_VERSION_MAJOR/ {maj=$6} /AMDSMI_LIB_VERSION_MINOR/ {min=$6} /AMDSMI_LIB_VERSION_RELEASE/ {rel=$6} END {gsub(\";\", \"\", maj); gsub(\";\", \"\", min); gsub(\";\", \"\", rel); print maj \".\" min \".\" rel}' src/amdsmi_wrapper.rs)
   test "$cargo_version" = "$header_version"
   test "$cargo_version" = "$wrapper_version"
   ```

2. If `../include/amd_smi/amdsmi.h` changed, regenerate the Rust wrapper:

   ```sh
   ../tools/update_rust_wrapper.sh
   ```

3. Ensure AMD-SMI is installed and visible to the build:

   ```sh
   export AMDSMI_LIB_DIR=/opt/rocm/lib
   export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH
   ```

4. Run the pre-publish gate:

   ```sh
   cargo build
   cargo build --release
   cargo build --examples
   cargo clippy --all-targets -- -D warnings
   cargo doc --no-deps
   cargo test --doc -- --show-output --test-threads=1
   DOCS_RS=1 cargo check
   cargo package --no-verify
   cargo publish --dry-run
   ```

   Most API doc examples are marked `no_run` because many AMD-SMI APIs mutate
   GPU state, require elevated permissions, or assume a particular topology.
   `cargo test --doc` is therefore used primarily as a compile-check for the
   documented snippets.

5. Inspect the package contents:

   ```sh
   tar -tzf target/package/amdsmi-*.crate | sort
   du -h target/package/amdsmi-*.crate
   ```

   The archive should include `LICENSE`, `README.md`, `Cargo.toml`,
   `build.rs`, `callbacks.rs`, `src/**`, and `examples/**`. It should not
   include `target/`, `.git/`, or generated build outputs.

## Publishing

Publish from the commit that has merged to `develop`:

```sh
git checkout develop
git pull --ff-only origin develop
cd projects/amdsmi/rust-interface
cargo login
cargo publish
```

Tag the published version:

```sh
git tag amdsmi-rust-v26.4.0
git push origin amdsmi-rust-v26.4.0
```

## Post-publish checks

1. Confirm the crate appears on crates.io:

   ```sh
   cargo search amdsmi --limit 5
   ```

2. Confirm docs.rs renders the crate docs:

   - <https://docs.rs/amdsmi>

3. Smoke-test from a clean directory outside the repository:

   ```sh
   tmpdir=$(mktemp -d)
   cd "$tmpdir"
   cargo init --bin
   cargo add amdsmi@26.4.0
   cat > src/main.rs <<'EOF'
   use amdsmi::*;

   fn main() -> Result<(), Box<dyn std::error::Error>> {
       let _smi = AmdSmiGuard::new(AmdsmiInitFlagsT::AmdsmiInitAmdGpus)?;
       println!("sockets: {}", amdsmi_get_socket_handles()?.len());
       Ok(())
   }
   EOF
   AMDSMI_LIB_DIR=/opt/rocm/lib LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH cargo run
   ```

4. Once an AMD/ROCm crates.io team exists, add it as an owner:

   ```sh
   cargo owner --add github:ROCm:rust-maintainers amdsmi
   ```
