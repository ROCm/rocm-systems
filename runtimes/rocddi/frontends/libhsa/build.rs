use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }

    let map = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/exports_linux.map");
    println!("cargo:rerun-if-changed={}", map.display());
    // Rust 1.85 selects GNU ld here, which rejects its anonymous export map
    // alongside the named ROCR_1 node. LLD accepts both maps and is also used
    // by the pinned Rust toolchain.
    println!("cargo:rustc-cdylib-link-arg=-fuse-ld=lld");
    println!(
        "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
        map.display()
    );
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,libhsa-runtime64.so.1");
}
