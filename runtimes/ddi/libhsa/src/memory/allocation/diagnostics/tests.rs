// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host-only checks of the production diagnostic switch and status preservation.

#![allow(clippy::unwrap_used)]

use super::*;
use crate::ffi::{NOT_INITIALIZED, OUT_OF_RESOURCES, RESOURCE_BUSY};
use rocddi::ErrorKind;
use rocddi::memory::device::HostMappingPolicy;
use std::process::Command;

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod fork;

/// Constructs a native allocation failure without issuing an OS operation.
fn native_error(operation: &'static str, errno: i32) -> rocddi::Error {
    rocddi::Error::NativeOperation {
        kind: ErrorKind::Busy,
        operation,
        source: std::io::Error::from_raw_os_error(errno),
    }
}

#[test]
/// Runs only in a subprocess so the real switch needs no global env mutation.
fn diagnostic_child() {
    if std::env::var_os("ROCDDI_MEMORY_DIAGNOSTIC_TEST_CHILD").is_none() {
        return;
    }
    let mut diagnostic = AllocationDiagnostic::new(HsaMemoryPool { handle: 0x123 }, 4 << 20, 0);
    diagnostic.rounded_bytes = Some(4 << 20);
    diagnostic.memory_kind = Some(MemoryKind::OwnedHost {
        policy: HostMappingPolicy::Coarse,
    });
    diagnostic.phase = "native_allocate";
    let status = diagnostic.map_error(native_error("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", libc::EAGAIN));
    assert_eq!(diagnostic.finish(status), RESOURCE_BUSY);

    diagnostic.phase = "retire_unpublished";
    diagnostic.publication_status = Some(OUT_OF_RESOURCES);
    let status = diagnostic.map_error(native_error("AMDKFD_IOC_FREE_MEMORY_OF_GPU", libc::EBUSY));
    assert_eq!(diagnostic.finish(status), RESOURCE_BUSY);

    let mut pointer = std::ptr::null_mut();
    // SAFETY: The fresh subprocess has no HSA generation. The writable output
    // is valid, and the production helper returns before any native allocation.
    let status = unsafe {
        crate::memory::allocation::memory_pool_allocate(
            HsaMemoryPool { handle: 0x123 },
            4 << 20,
            0,
            &raw mut pointer,
            crate::ffi::INVALID_MEMORY_POOL,
        )
    };
    assert_eq!(status, NOT_INITIALIZED);
    assert!(pointer.is_null());
}

#[test]
/// Exercises exact opt-in, original native errors, cleanup and status-only exits.
fn diagnostic_switch_preserves_error_statuses() {
    for setting in [None, Some("0"), Some("true"), Some("1")] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "memory::allocation::diagnostics::tests::diagnostic_child",
                "--nocapture",
            ])
            .env("ROCDDI_MEMORY_DIAGNOSTIC_TEST_CHILD", "1")
            .env_remove("ROCDDI_HSA_MEMORY_DIAGNOSTICS");
        if let Some(setting) = setting {
            command.env("ROCDDI_HSA_MEMORY_DIAGNOSTICS", setting);
        }
        let output = command.output().unwrap();
        assert!(output.status.success(), "{output:?}");
        let stderr = String::from_utf8(output.stderr).unwrap();
        if setting == Some("1") {
            for expected in [
                "phase=native_allocate; pool=0x123; flags=0x0; requested_bytes=4194304; rounded_bytes=Some(4194304)",
                "memory_kind=Some(OwnedHost { policy: Coarse })",
                "operation=Some(\"AMDKFD_IOC_ALLOC_MEMORY_OF_GPU\"); kind=Busy",
                "domain: Errno, code: 11",
                "errno=Some(11)",
                "phase=retire_unpublished",
                "publication_status=Some(4104)",
                "operation=Some(\"AMDKFD_IOC_FREE_MEMORY_OF_GPU\"); kind=Busy",
                "errno=Some(16)",
                "hsa_status=46",
                "phase=validation",
                "rounded_bytes=None; memory_kind=None; publication_status=None; hsa_status=4107",
            ] {
                assert!(stderr.contains(expected), "missing {expected}: {stderr}");
            }
        } else {
            assert!(stderr.is_empty(), "unexpected diagnostic: {stderr}");
        }
    }
}

/// A failed diagnostic sink must not affect status conversion.
struct BrokenOutput;

impl Write for BrokenOutput {
    fn write(&mut self, _bytes: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
/// Failure of best-effort diagnostic I/O preserves the native Busy result.
fn diagnostic_write_failure_preserves_error_status() {
    let diagnostic = AllocationDiagnostic::new(HsaMemoryPool { handle: 0x123 }, 4 << 20, 0);
    assert_eq!(
        diagnostic.map_error_to(
            &mut BrokenOutput,
            native_error("AMDKFD_IOC_ALLOC_MEMORY_OF_GPU", libc::EAGAIN),
        ),
        RESOURCE_BUSY,
    );
}
