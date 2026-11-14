// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Target-correct native inputs and transports for the HSA C ABI.

pub(crate) mod cstream;
#[cfg(target_os = "linux")]
mod linux;
pub(crate) mod memory;
pub(crate) mod profiling;
#[cfg(target_os = "linux")]
use linux as native;
#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows as native;

pub(crate) use native::{
    AisOperation, acquire_code_object_file, ais_file_io, bind_process, code_object_file_uri,
    driver_uid, invalid_file, native_cpu_node_ordinal, node_id, process_is_current,
};

#[cfg(all(test, target_os = "linux"))]
pub(crate) use linux::ais::ais_transfer;
#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
pub(crate) use linux::tooling;
