// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA status translation and the C ABI panic boundary.

#[cfg(not(test))]
use super::{ApiLeaseScope, process_admission};
#[cfg(not(test))]
use crate::ffi::NOT_INITIALIZED;
use crate::ffi::{ERROR, INVALID_ARGUMENT, NOT_SUPPORTED, OUT_OF_RESOURCES, RESOURCE_BUSY, Status};

#[allow(clippy::needless_pass_by_value)]
/// Maps a private native error to its HSA status.
pub(crate) fn map_error(error: rocddi::Error) -> Status {
    match error.kind() {
        rocddi::ErrorKind::InvalidArgument => INVALID_ARGUMENT,
        rocddi::ErrorKind::Unsupported => NOT_SUPPORTED,
        rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
        // Busy describes a live native dependency. Only queue-handle
        // validation in the queue adapter may return INVALID_QUEUE.
        rocddi::ErrorKind::Busy => RESOURCE_BUSY,
        _ => ERROR,
    }
}

/// Preserves an opt-in startup diagnostic before native-to-HSA translation.
pub(crate) fn map_startup_error(stage: &'static str, error: rocddi::Error) -> Status {
    if std::env::var_os("ROCDDI_HSA_STARTUP_DIAGNOSTICS").is_some_and(|value| value == "1") {
        use std::io::Write as _;
        // Diagnostic I/O cannot replace the original native failure status.
        let _ = writeln!(
            std::io::stderr().lock(),
            "rocddi HSA startup: stage={stage}; kind={:?}; native={:?}; error={error}",
            error.kind(),
            error.native_status(),
        );
    }
    map_error(error)
}

/// Runs an HSA ABI operation behind the panic-to-status boundary.
pub(crate) fn boundary(operation: impl FnOnce() -> Status) -> Status {
    if !crate::platform::process_is_current() {
        return crate::ffi::NOT_INITIALIZED;
    }
    #[cfg(not(test))]
    {
        let Some(token) = process_admission().enter() else {
            return NOT_INITIALIZED;
        };
        let _scope = ApiLeaseScope::enter(token);
        plain_boundary(operation)
    }
    #[cfg(test)]
    plain_boundary(operation)
}

/// Runs lifecycle and pure ABI operations without requiring HSA admission.
pub(crate) fn plain_boundary(operation: impl FnOnce() -> Status) -> Status {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(operation)).unwrap_or(ERROR)
}
