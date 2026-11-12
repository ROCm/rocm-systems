// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Platform operations needed to translate the HSA C ABI.
//!
//! The current adapter is Linux. A new OS supplies this small module with
//! descriptor, IPC, SVM, and event operations backed by its rocddi provider.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::{
    acquire_code_object_file, bind_process, code_object_file_uri, driver_uid, fd, invalid_file,
    memory, native_cpu_node_ordinal, node_id, process_is_current,
};

#[cfg(not(target_os = "linux"))]
compile_error!("the HSA frontend currently requires a platform adapter");

#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
pub(crate) use linux::tooling;
