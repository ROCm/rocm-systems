// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Link configuration for the shared AMDF and HSA Linux library.

use std::env;
use std::path::PathBuf;

fn main() {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return;
    }
    let map = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("hsa/libhsa/src/exports_linux.map");
    println!("cargo:rerun-if-changed={}", map.display());
    println!("cargo:rustc-cdylib-link-arg=-fuse-ld=lld");
    println!(
        "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
        map.display()
    );
    let soname =
        env::var("ROCM_RUNTIME_HSA_SONAME").unwrap_or_else(|_| "libhsa-runtime64.so.1".to_owned());
    println!("cargo:rustc-cdylib-link-arg=-Wl,-soname,{soname}");
    println!("cargo:rustc-cdylib-link-arg=-Wl,-z,nodelete");
    println!("cargo:rustc-cdylib-link-arg=-Wl,--undefined=amdf_query_api");
    println!("cargo:rustc-cdylib-link-arg=-Wl,--undefined=hsa_init");
}
