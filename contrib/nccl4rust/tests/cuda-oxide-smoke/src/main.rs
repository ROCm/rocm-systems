// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CUDA-Oxide compile-and-link coverage for the Rust-facing device API.

use core::ffi::c_void;
use std::error::Error;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use cuda_device::{kernel, thread};
use libnvvm_sys::{LibNvvm, Program};
use nccl_device::{self as nccl, DevComm, Window, sys};
use nvjitlink_sys::{InputType, LibNvJitLink, Linker};

/// Query a device communicator into four consecutive `i32` outputs.
///
/// # Safety
///
/// `comm` must point to a live NCCL device communicator, and `output` must be
/// writable for at least four `i32` values for the duration of the kernel.
#[kernel]
pub unsafe fn nccl4rust_query_smoke(comm: *const sys::ncclDevComm_t, output: *mut i32) {
    if thread::blockIdx_x() == 0 && thread::threadIdx_x() == 0 {
        let comm = unsafe { DevComm::from_raw(comm) };
        unsafe {
            *output.add(0) = nccl::rank(comm);
            *output.add(1) = nccl::size(comm);
            *output.add(2) = nccl::lsa_rank(comm);
            *output.add(3) = nccl::lsa_size(comm);
        }
    }
}

/// One-CTA-per-rank LSA all-reduce building block.
///
/// The host must reserve two LSA barriers, register valid source/destination
/// windows, and launch identical grids on all LSA ranks.
///
/// # Safety
///
/// The communicator and windows must remain live for the launch; offsets and
/// `count` must describe valid `f32` regions; all participating LSA ranks must
/// execute this kernel with matching arguments and barrier reservations.
#[kernel]
pub unsafe fn nccl4rust_lsa_allreduce_f32_smoke(
    comm: *const sys::ncclDevComm_t,
    src_window: *mut c_void,
    src_offset: usize,
    dst_window: *mut c_void,
    dst_offset: usize,
    count: usize,
) {
    let comm = unsafe { DevComm::from_raw(comm) };
    let src = unsafe { Window::from_raw(src_window) };
    let dst = unsafe { Window::from_raw(dst_window) };

    unsafe {
        nccl::lsa_barrier_cta(comm, 0, false);
        nccl::lsa_reduce_sum_copy_f32_cta(comm, src, src_offset, dst, dst_offset, count);
        nccl::lsa_barrier_cta(comm, 1, false);
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let arch = std::env::var("NCCL4RUST_ARCH").unwrap_or_else(|_| "sm_90".to_owned());
    let compute_arch = arch.strip_prefix("sm_").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("NCCL4RUST_ARCH must have the form sm_XX, got {arch:?}"),
        )
    })?;
    let compute_capability = compute_arch.parse::<u32>().map_err(|source| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid NCCL4RUST_ARCH {arch:?}: {source}"),
        )
    })?;
    let link_mode = std::env::var("NCCL4RUST_LINK_MODE").unwrap_or_else(|_| "ptx".to_owned());
    let device_ltoir_path = env_path("NCCL4RUST_DEVICE_LTOIR").unwrap_or_else(|| {
        manifest_dir
            .join("../../build")
            .join(format!("libnccl4rust_device_{arch}.ltoir"))
    });
    let output_path = env_path("NCCL4RUST_CUBIN")
        .unwrap_or_else(|| manifest_dir.join(format!("nccl4rust_cuda_oxide_smoke_{arch}.cubin")));

    let device_ltoir = read_input(&device_ltoir_path, "NCCL device shim LTOIR")?;
    let (rust_input, rust_input_type, rust_input_name) = match link_mode.as_str() {
        "ptx" => {
            let path = env_path("NCCL4RUST_RUST_PTX")
                .unwrap_or_else(|| manifest_dir.join("nccl4rust_cuda_oxide_smoke.ptx"));
            (
                read_input(&path, "Rust PTX")?,
                InputType::Ptx,
                path.file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("nccl4rust_cuda_oxide_smoke.ptx")
                    .to_owned(),
            )
        }
        "nvvm-lto" => {
            if compute_capability < 100 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "this pinned CUDA-Oxide revision emits opaque-pointer NVVM 20 IR, which \
                     requires sm_100 or newer; use NCCL4RUST_LINK_MODE=ptx for pre-Blackwell",
                )
                .into());
            }
            let path = env_path("NCCL4RUST_RUST_NVVM_IR")
                .unwrap_or_else(|| manifest_dir.join("nccl4rust_cuda_oxide_smoke.ll"));
            let rust_ir = read_input(&path, "Rust NVVM IR")?;
            (
                compile_nvvm_ir(&rust_ir, compute_arch)?,
                InputType::Ltoir,
                "nccl4rust_cuda_oxide_smoke.ltoir".to_owned(),
            )
        }
        other => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("NCCL4RUST_LINK_MODE must be `ptx` or `nvvm-lto`, got {other:?}"),
            )
            .into());
        }
    };

    let nvjitlink = LibNvJitLink::load()?;
    let link_arch = format!("-arch={arch}");
    let mut linker = Linker::new(&nvjitlink, &[&link_arch, "-lto"])?;
    linker.add(rust_input_type, &rust_input, &rust_input_name)?;
    linker.add(
        InputType::Ltoir,
        &device_ltoir,
        device_ltoir_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("libnccl4rust_device.ltoir"),
    )?;
    let cubin = linker.finish()?;
    fs::write(&output_path, &cubin).map_err(|source| {
        io::Error::new(
            source.kind(),
            format!("failed to write {}: {source}", output_path.display()),
        )
    })?;

    println!(
        "linked {} bytes of Rust {} and {} bytes of NCCL shim LTOIR into {} ({} bytes)",
        rust_input.len(),
        link_mode,
        device_ltoir.len(),
        output_path.display(),
        cubin.len()
    );
    Ok(())
}

fn compile_nvvm_ir(nvvm_ir: &[u8], compute_arch: &str) -> Result<Vec<u8>, Box<dyn Error>> {
    let libdevice_path = cuda_host::ltoir::find_libdevice()?;
    let libdevice = read_input(&libdevice_path, "CUDA libdevice bitcode")?;
    let nvvm = LibNvvm::load()?;
    let mut program = Program::new(&nvvm)?;
    program.add_module(&libdevice, "libdevice.10.bc")?;
    program.add_module(nvvm_ir, "nccl4rust_cuda_oxide_smoke.ll")?;
    let nvvm_arch = format!("-arch=compute_{compute_arch}");
    Ok(program.compile(&[&nvvm_arch, "-gen-lto"])?)
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name).map(PathBuf::from)
}

fn read_input(path: &Path, description: &str) -> io::Result<Vec<u8>> {
    fs::read(path).map_err(|source| {
        io::Error::new(
            source.kind(),
            format!(
                "failed to read {description} at {}: {source}",
                path.display()
            ),
        )
    })
}
