// SPDX-License-Identifier: MIT

//! Platform operations needed to translate the HSA C ABI.
//!
//! The current adapter is Linux. A new OS supplies this small module with
//! descriptor, IPC, SVM, and event operations backed by its rocddi provider.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::{
    code_object_file_uri, driver_uid, event, fault_matches_endpoint, fd, host, memory, node_id,
};

#[cfg(not(target_os = "linux"))]
compile_error!("the HSA frontend currently requires a platform adapter");
