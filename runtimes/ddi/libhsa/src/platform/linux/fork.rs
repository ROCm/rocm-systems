// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! The process witness checked before HSA's lifecycle and tool dispatch.

use crate::ffi::{ERROR, OUT_OF_RESOURCES, Status};
use rocddi::host::linux::ForkWitness;
use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};

// The Linux x86-64 export trampolines read this address before consulting any
// tool slot. A published page is retained until process exit and is zeroed in
// the child by MADV_WIPEONFORK. Zero means HSA has not started in this image.
#[unsafe(no_mangle)]
/// Process image marker read by the Linux export and bridge trampolines.
static hsa_process_witness: AtomicUsize = AtomicUsize::new(0);

/// Reports whether no HSA owner exists or this process still owns it.
pub(crate) fn process_is_current() -> bool {
    let address = hsa_process_witness.load(Ordering::Acquire);
    address == 0 || {
        // SAFETY: Only a retained ForkWitness address is published above.
        unsafe { ForkWitness::retained_is_current(address) }
    }
}

/// Establishes the HSA process before any lifecycle lock or callback.
pub(crate) fn bind_process() -> Result<(), Status> {
    loop {
        let current = hsa_process_witness.load(Ordering::Acquire);
        if current != 0 {
            return process_is_current().then_some(()).ok_or(ERROR);
        }
        let witness = ForkWitness::capture().map_err(|source| {
            if source.kind() == io::ErrorKind::OutOfMemory {
                OUT_OF_RESOURCES
            } else {
                ERROR
            }
        })?;
        let address = witness.address();
        match hsa_process_witness.compare_exchange(0, address, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {
                let retained = witness.retain_until_exit();
                debug_assert_eq!(retained, address);
                return process_is_current().then_some(()).ok_or(ERROR);
            }
            Err(_) => drop(witness),
        }
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::ffi::{NOT_INITIALIZED, SUCCESS};

    #[test]
    fn fork_child_rejects_hsa_before_an_inherited_lifecycle_lock() {
        bind_process().unwrap();
        let guard = crate::runtime::lock().unwrap();
        // SAFETY: The child only checks atomics and calls the guarded HSA ABI
        // entries, then exits without running inherited Rust destructors.
        let child = unsafe { libc::syscall(libc::SYS_fork) };
        assert!(child >= 0);
        if child == 0 {
            // SAFETY: A forked child must never unwind through the test runner.
            unsafe { libc::alarm(3) };
            let rejected = !process_is_current()
                && crate::api::lifecycle::hsa_init() == ERROR
                && crate::api::lifecycle::hsa_shut_down() == NOT_INITIALIZED
                && crate::runtime::boundary(|| SUCCESS) == NOT_INITIALIZED
                && crate::runtime::enter_public_raw_handle().is_none();
            // SAFETY: _exit avoids touching inherited runtime locks.
            unsafe { libc::_exit(i32::from(!rejected)) }
        }
        let mut status = 0;
        let pid = i32::try_from(child).unwrap();
        // SAFETY: This waits only for the child created above.
        assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
        assert_eq!(status, 0);
        drop(guard);
        assert!(process_is_current());
    }
}
