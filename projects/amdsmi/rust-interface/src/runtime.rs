// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

pub use crate::amdsmi_wrapper::*;
use crate::utils::AmdsmiResult;
use libloading::Library;
use std::path::PathBuf;
use std::sync::OnceLock;

fn library() -> AmdsmiResult<&'static Library> {
    // Keep both the library and resolved function pointers alive through shutdown
    // and reinitialization of the C library. Loading is deferred until a query.
    static LIBRARY: OnceLock<Result<Library, AmdsmiStatusT>> = OnceLock::new();
    LIBRARY
        .get_or_init(|| {
            let path = std::env::var_os("AMDSMI_LIB_DIR")
                .map(|directory| PathBuf::from(directory).join("libamd_smi.so"))
                .unwrap_or_else(|| PathBuf::from("libamd_smi.so"));
            // SAFETY: as with link-time loading, the caller must use a trusted
            // AMD SMI installation whose ABI matches the generated bindings.
            unsafe { Library::new(path) }.map_err(|_| AmdsmiStatusT::AmdsmiStatusFailLoadModule)
        })
        .as_ref()
        .map_err(|status| *status)
}

include!(concat!(env!("OUT_DIR"), "/runtime_bindings.rs"));
