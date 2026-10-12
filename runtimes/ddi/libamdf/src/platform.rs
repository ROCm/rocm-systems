// SPDX-License-Identifier: MIT

//! Platform services used to translate the AMDF C ABI.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "linux")]
pub(crate) use linux::{IntoRawFd, memory, native_identity, render_supported};

#[cfg(not(target_os = "linux"))]
compile_error!("the AMDF frontend currently requires a platform adapter");
