// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Compile a selected `AddrLib` source tree and its narrow C bridge.

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
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    println!("cargo:rerun-if-env-changed=ROCADDR_SOURCE_DIR");
    let source = match env::var_os("ROCADDR_SOURCE_DIR") {
        Some(path) if path.is_empty() => {
            return Err(std::io::Error::other("ROCADDR_SOURCE_DIR must not be empty").into());
        }
        Some(path) => manifest.join(path),
        None => manifest.join("vendor/addrlib"),
    };
    // Select one complete input tree. Never mix headers from an installed ROCm
    // with sources from a different revision or silently fall back to the copy.
    for relative in SOURCES.iter().copied().chain([
        "inc/addrinterface.h",
        "inc/addrtypes.h",
        "src/core/addrlib.h",
    ]) {
        let path = source.join(relative);
        if !path.is_file() {
            return Err(std::io::Error::other(format!(
                "incomplete AddrLib source tree: missing {}",
                path.display()
            ))
            .into());
        }
    }
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
        build.include(source.join(directory));
    }
    for relative in SOURCES {
        build.file(source.join(relative));
    }
    build.file("src/bridge.cpp");
    println!("cargo:rerun-if-changed={}", source.display());
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    // cc selects target compiler/archive flags and C++ runtime linkage. Its
    // standard target-qualified CXX/AR variables also support provisioned SDKs.
    build.try_compile("rocaddr3_native")?;
    Ok(())
}
