// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Compile the vendored `AddrLib3` source and its narrow C bridge.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn run(command: &mut Command) -> Result<(), Box<dyn std::error::Error>> {
    let status = command.status()?;
    if !status.success() {
        return Err(std::io::Error::other(format!("native AddrLib build failed: {status}")).into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let target = env::var("CARGO_CFG_TARGET_OS")?;
    if target != "linux" {
        return Err(std::io::Error::other("rocaddr3 requires a Linux C++ toolchain").into());
    }
    let crate_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR")
            .ok_or_else(|| std::io::Error::other("Cargo did not provide the crate directory"))?,
    );
    let output = PathBuf::from(
        env::var_os("OUT_DIR")
            .ok_or_else(|| std::io::Error::other("Cargo did not provide the output directory"))?,
    );
    let vendor = crate_dir.join("vendor/addrlib");
    let include = [
        vendor.clone(),
        vendor.join("inc"),
        vendor.join("src"),
        vendor.join("src/core"),
        vendor.join("src/gfx9"),
        vendor.join("src/gfx10"),
        vendor.join("src/gfx11"),
        vendor.join("src/gfx12"),
        vendor.join("src/chip/r800"),
        vendor.join("src/chip/gfx9"),
        vendor.join("src/chip/gfx10"),
        vendor.join("src/chip/gfx11"),
        vendor.join("src/chip/gfx12"),
    ];
    let compiler = env::var_os("CXX").unwrap_or_else(|| "c++".into());
    let archiver = env::var_os("AR").unwrap_or_else(|| "ar".into());
    println!("cargo:rerun-if-env-changed=CXX");
    println!("cargo:rerun-if-env-changed=AR");
    println!("cargo:rerun-if-changed={}", vendor.display());
    println!("cargo:rerun-if-changed=src/bridge.cpp");
    let mut objects = Vec::new();
    for (index, source) in SOURCES.iter().enumerate() {
        let path = vendor.join(source);
        let object = output.join(format!("addr{index}.o"));
        compile(&compiler, &path, &object, &include)?;
        objects.push(object);
    }
    let bridge = crate_dir.join("src/bridge.cpp");
    let object = output.join("bridge.o");
    compile(&compiler, &bridge, &object, &include)?;
    objects.push(object);
    let archive = output.join("librocaddr3_native.a");
    let mut command = Command::new(archiver);
    command.arg("crs").arg(&archive).args(&objects);
    run(&mut command)?;
    println!("cargo:rustc-link-search=native={}", output.display());
    println!("cargo:rustc-link-lib=static=rocaddr3_native");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    Ok(())
}

fn compile(
    compiler: &std::ffi::OsStr,
    source: &Path,
    object: &Path,
    include: &[PathBuf],
) -> Result<(), Box<dyn std::error::Error>> {
    let mut command = Command::new(compiler);
    command
        .arg("-std=c++17")
        .arg("-fPIC")
        .arg("-O2")
        .arg("-DBRAHMA_BUILD=1")
        .arg("-DAMD_INTERNAL_BUILD")
        .arg("-DLITTLEENDIAN_CPU")
        .arg("-DLINUX")
        .arg("-DUNIX_OS");
    for directory in include {
        command.arg("-I").arg(directory);
    }
    command.arg("-c").arg(source).arg("-o").arg(object);
    run(&mut command)
}
