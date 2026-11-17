// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Compile the vendored `AddrLib3` source and its narrow C bridge.

use std::env;
use std::path::PathBuf;

const SOURCES: &[&str] = &[
    "src/addrinterface.cpp",
    "src/core/coord.cpp",
    "src/core/addrlib.cpp",
    "src/core/addrlib1.cpp",
    "src/core/addrlib2.cpp",
    "src/core/addrlib3.cpp",
    "src/core/addrobject.cpp",
    "src/core/addrelemlib.cpp",
    "src/gfx9/gfx9addrlib.cpp",
    "src/gfx10/gfx10addrlib.cpp",
    "src/gfx11/gfx11addrlib.cpp",
    "src/gfx12/gfx12addrlib.cpp",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = env::var("CARGO_CFG_TARGET_OS")?;
    if !matches!(target.as_str(), "linux" | "windows") {
        return Err(
            std::io::Error::other("rocaddr3 supports Linux and Windows native toolchains").into(),
        );
    }
    let vendor = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("vendor/addrlib");
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .opt_level(2)
        .pic(true)
        .warnings(false)
        .define("BRAHMA_BUILD", "1")
        .define("AMD_INTERNAL_BUILD", None)
        .define("LITTLEENDIAN_CPU", None);
    if target == "linux" {
        build.define("LINUX", None).define("UNIX_OS", None);
    }
    for directory in [
        "",
        "inc",
        "src",
        "src/core",
        "src/gfx9",
        "src/gfx10",
        "src/gfx11",
        "src/gfx12",
        "src/chip/r800",
        "src/chip/gfx9",
        "src/chip/gfx10",
        "src/chip/gfx11",
        "src/chip/gfx12",
    ] {
        build.include(vendor.join(directory));
    }
    for source in SOURCES {
        build.file(vendor.join(source));
    }
    build.file("src/bridge.cpp");
    println!("cargo:rerun-if-changed={}", vendor.display());
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    // cc selects target compiler/archive flags and C++ runtime linkage. Its
    // standard target-qualified CXX/AR variables also support provisioned SDKs.
    build.try_compile("rocaddr3_native")?;
    Ok(())
}
