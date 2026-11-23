// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Opt-in allocation failure context without changing HSA status translation.

use super::{HsaMemoryPool, MemoryKind, SUCCESS, Status, map_error};
use std::io::Write;

/// Stack-only request facts retained through allocation and publication cleanup.
pub(super) struct AllocationDiagnostic {
    /// Public memory-pool handle supplied by the caller.
    pool: HsaMemoryPool,
    /// Original logical allocation size, in bytes.
    requested_bytes: usize,
    /// Public allocation flags supplied by the caller.
    flags: u32,
    /// Page-rounded size when validation has determined it.
    pub(super) rounded_bytes: Option<usize>,
    /// Native placement when pool decoding has determined it.
    pub(super) memory_kind: Option<MemoryKind>,
    /// Last allocation stage entered, including failures without a native error.
    pub(super) phase: &'static str,
    /// Frontend failure whose unpublished allocation is being retired.
    pub(super) publication_status: Option<Status>,
}

impl AllocationDiagnostic {
    /// Starts an observation before the HSA admission boundary.
    pub(super) fn new(pool: HsaMemoryPool, requested_bytes: usize, flags: u32) -> Self {
        Self {
            pool,
            requested_bytes,
            flags,
            rounded_bytes: None,
            memory_kind: None,
            phase: "admission",
            publication_status: None,
        }
    }

    /// Reports the original error before the ordinary consuming conversion.
    pub(super) fn map_error(&self, error: rocddi::Error) -> Status {
        if enabled() {
            self.map_error_to(&mut std::io::stderr().lock(), error)
        } else {
            map_error(error)
        }
    }

    /// Keeps diagnostic write failures from replacing the operation's status.
    fn map_error_to(&self, output: &mut impl Write, error: rocddi::Error) -> Status {
        let native = error.native_status();
        let operation = match &error {
            rocddi::Error::NativeOperation { operation, .. }
            | rocddi::Error::DriverOperation { operation, .. } => Some(*operation),
            _ => None,
        };
        let _ = writeln!(
            output,
            "rocddi HSA memory: {self}; operation={operation:?}; kind={:?}; native={native:?}; errno={:?}; error={error}",
            error.kind(),
            native.and_then(rocddi::NativeStatus::errno_code),
        );
        map_error(error)
    }

    /// Reports every failed exit, including admission, validation and unwinds.
    pub(super) fn finish(&self, status: Status) -> Status {
        // The ABI boundary rejects inherited processes before locks. Its
        // failed result must remain free of environment and stderr locks here.
        if status != SUCCESS && crate::platform::process_is_current() && enabled() {
            let _ = writeln!(
                std::io::stderr().lock(),
                "rocddi HSA memory: {self}; hsa_status={status}",
            );
        }
        status
    }
}

impl std::fmt::Display for AllocationDiagnostic {
    fn fmt(&self, output: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            output,
            "phase={}; pool={:#x}; flags={:#x}; requested_bytes={}; rounded_bytes={:?}; memory_kind={:?}; publication_status={:?}",
            self.phase,
            self.pool.handle,
            self.flags,
            self.requested_bytes,
            self.rounded_bytes,
            self.memory_kind,
            self.publication_status,
        )
    }
}

/// Only the exact value `1` opts into memory diagnostics.
fn enabled() -> bool {
    std::env::var_os("ROCDDI_HSA_MEMORY_DIAGNOSTICS").is_some_and(|value| value == "1")
}

#[cfg(test)]
mod tests;
