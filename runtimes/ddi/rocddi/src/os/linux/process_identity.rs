// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! A fork-sensitive Linux process marker for native owners.
//!
//! The kernel zeroes one private page in a fork child. An inherited owner's
//! captured PID then differs from the marker before it can touch callbacks,
//! files, or locks. A new session in the child installs the child's PID in
//! that page, leaving all inherited owners invalid. If the kernel cannot
//! provide `MADV_WIPEONFORK`, checks keep using `getpid`.

#![allow(unsafe_code)]

use super::memory::ForkMarkerPage;
use std::io;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

const FALLBACK_TO_GETPID: usize = 1;

// Zero means uninitialized. The one-page mapping remains until process exit,
// so a published pointer stays valid even when sessions and frontends unload.
// One means the kernel lacks the wipe-on-fork contract and checks use getpid.
static MARKER: AtomicUsize = AtomicUsize::new(0);

/// Publishes the getpid fallback unless another thread already chose a marker.
fn install_fallback() -> usize {
    MARKER
        .compare_exchange(0, FALLBACK_TO_GETPID, Ordering::AcqRel, Ordering::Acquire)
        .unwrap_or_else(|winner| winner)
}

/// Returns the published marker page or fallback state, creating one on first
/// use. A losing thread releases only its own unpublished mapping.
fn marker_state() -> usize {
    let state = MARKER.load(Ordering::Acquire);
    if state != 0 {
        return state;
    }
    let Ok(mapping) = ForkMarkerPage::new() else {
        return install_fallback();
    };
    // The marker stays zero until allocation passes the session's existing
    // process check. A concurrent owner check uses getpid while it is zero.
    match MARKER.compare_exchange(0, mapping.address(), Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => mapping.retain_until_exit(),
        Err(winner) => winner,
    }
}

/// Enables cheap checks after an allocation passed its session process check.
/// This preserves inert instance creation and the getpid fallback before the
/// first host allocation or after a fork.
pub(crate) fn prepare_for_hot_checks() {
    let state = marker_state();
    if state == FALLBACK_TO_GETPID {
        return;
    }
    // SAFETY: The published mapping stays live for this process lifetime.
    let marker = unsafe { &*(state as *const AtomicU32) };
    if marker.load(Ordering::Acquire) != 0 {
        return;
    }
    let current = std::process::id();
    marker.store(current, Ordering::Release);
    // A fork from a signal handler between the PID read and store can wipe
    // the page before the stale store. Recheck before returning to the caller.
    let confirmed = std::process::id();
    if confirmed != current {
        marker.store(confirmed, Ordering::Release);
    }
}

/// Rejects inherited native ownership without a syscall on supported Linux.
pub(crate) fn check_process(process: u32) -> io::Result<()> {
    let state = MARKER.load(Ordering::Acquire);
    let current = if state <= FALLBACK_TO_GETPID {
        // Directly constructed low-level owners may precede any session.
        std::process::id()
    } else {
        // SAFETY: A published marker mapping is never unmapped. In a fork
        // child its contents are zero until a new session is constructed.
        let cached = unsafe { &*(state as *const AtomicU32) }.load(Ordering::Acquire);
        if cached == 0 {
            std::process::id()
        } else {
            cached
        }
    };
    if process == current {
        Ok(())
    } else {
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[allow(unsafe_code)]
mod tests {
    use super::*;
    use std::ffi::{c_int, c_long};

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
        fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
        fn _exit(status: c_int) -> !;
    }

    const SYS_FORK: c_long = 57;

    #[test]
    #[allow(clippy::used_underscore_items)]
    fn raw_fork_rejects_inherited_owner_even_after_new_session() {
        let parent = std::process::id();
        prepare_for_hot_checks();
        assert!(check_process(parent).is_ok());
        // SAFETY: The child only reads this atomic marker, calls getpid, and
        // exits without accessing a test-harness lock or allocator.
        let child = unsafe { syscall(SYS_FORK) };
        assert!(child >= 0);
        if child == 0 {
            let rejected_before = check_process(parent).is_err();
            let current = std::process::id();
            let accepted_before = check_process(current).is_ok();
            prepare_for_hot_checks();
            let accepted_after = check_process(current).is_ok();
            let rejected_after = check_process(parent).is_err();
            // SAFETY: _exit does not run inherited Rust destructors.
            unsafe {
                _exit(i32::from(
                    !(rejected_before && accepted_before && accepted_after && rejected_after),
                ))
            }
        }
        let child_pid = c_int::try_from(child).unwrap_or_default();
        assert!(child_pid > 0);
        let mut status = 0;
        // SAFETY: This waits only for the child created by the syscall above.
        assert_eq!(unsafe { waitpid(child_pid, &raw mut status, 0) }, child_pid);
        assert_eq!(status, 0);
        assert!(check_process(parent).is_ok());
    }
}
