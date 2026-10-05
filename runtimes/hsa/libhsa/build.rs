use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=ROCM_RUNTIME_HSA_SONAME");
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
    let soname =
        env::var("ROCM_RUNTIME_HSA_SONAME").unwrap_or_else(|_| "libhsa-runtime64.so.1".to_owned());
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,{soname}");
}
