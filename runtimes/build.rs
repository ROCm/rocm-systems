// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Link configuration for the shared AMDF, HSA and ABCE Linux library.

use std::env;
use std::fmt::Write as _;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return Ok(());
    }
    let map = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ddi/libhsa/src/exports_linux.map");
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
    let abce_exports =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("ddi/libabce/abi/exports.txt");
    println!("cargo:rerun-if-changed={}", abce_exports.display());
    let manifest = std::fs::read_to_string(abce_exports)?;
    let mut script = String::from("ABCE_1 { global:\n");
    for line in manifest.lines().filter(|line| !line.starts_with('#')) {
        if let Some((_, symbol)) = line.split_once(' ') {
            println!("cargo:rustc-cdylib-link-arg=-Wl,--undefined={symbol}");
            writeln!(script, "  {symbol};")?;
        }
    }
    script.push_str("};\n");
    let script_path = PathBuf::from(env::var("OUT_DIR")?).join("abce.map");
    std::fs::write(&script_path, script)?;
    println!(
        "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
        script_path.display()
    );
    Ok(())
}
