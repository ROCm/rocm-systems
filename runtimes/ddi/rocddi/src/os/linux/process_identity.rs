// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! A fork-sensitive Linux process marker for native owners.
//!
//! The kernel zeroes each owner's private witness page in a fork child. A new
//! session captures a fresh page, leaving inherited owners invalid even if
//! their PID is reused in a nested namespace. If the kernel cannot
//! provide `MADV_WIPEONFORK`, checks also compare the PID namespace identity.

#![allow(unsafe_code)]

use super::memory::ForkMarkerPage;
use crate::memory::storage::Shared;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

/// Creating process identity retained by one session and its native owners.
/// The witness is private to this owner graph, so creating a passive session
/// does not initialize process-global KFD state.
#[derive(Clone)]
pub(crate) struct ProcessIdentity(Shared<ProcessIdentityState>);

struct ProcessIdentityState {
    pid: u32,
    witness: Option<FreshForkWitness>,
    /// Namespace identity used only when wipe-on-fork is unavailable.
    namespace: Option<(u64, u64)>,
}

fn pid_namespace() -> io::Result<(u64, u64)> {
    let metadata = std::fs::metadata("/proc/self/ns/pid")?;
    Ok((metadata.dev(), metadata.ino()))
}

impl ProcessIdentity {
    /// Captures a private fork witness without opening a native device.
    pub(crate) fn capture() -> io::Result<Self> {
        let witness = FreshForkWitness::new();
        let namespace = if witness.is_none() {
            Some(pid_namespace()?)
        } else {
            None
        };
        let state = Shared::new(ProcessIdentityState {
            pid: std::process::id(),
            witness,
            namespace,
        })
        .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        Ok(Self(state))
    }

    /// Distinguishes a confirmed fork from an unavailable namespace check.
    /// Process slots replace an inherited lock only on a confirmed mismatch.
    pub(crate) fn matches_current_process(&self) -> io::Result<bool> {
        self.matches_current_process_with_namespace(pid_namespace)
    }

    fn matches_current_process_with_namespace(
        &self,
        namespace: impl FnOnce() -> io::Result<(u64, u64)>,
    ) -> io::Result<bool> {
        if self.0.pid != std::process::id() {
            return Ok(false);
        }
        if let Some(witness) = &self.0.witness {
            return Ok(fork_witness_valid(Some(witness.address())));
        }
        Ok(Some(namespace()?) == self.0.namespace)
    }

    /// Rejects inherited owners before they touch native handles or locks.
    pub(crate) fn check(&self) -> io::Result<()> {
        if self.matches_current_process()? {
            Ok(())
        } else {
            Err(io::Error::from(io::ErrorKind::Unsupported))
        }
    }

    #[cfg(test)]
    #[allow(clippy::expect_used)]
    pub(crate) fn with_pid_for_test(pid: u32) -> Self {
        let mut identity = Self::capture().expect("test process identity");
        Shared::get_mut(&mut identity.0)
            .expect("test identity has one owner")
            .pid = pid;
        identity
    }
}

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

/// A separate wipe-on-fork page that validates one process-owned slot.
/// Unlike the shared hot-path marker, an inherited slot never rewrites this
/// witness after a fork, even if a descendant later reuses the original PID.
pub(crate) struct FreshForkWitness {
    /// Unpublished witness mapping, dropped when slot publication loses a race.
    page: ForkMarkerPage,
}

impl FreshForkWitness {
    /// Creates a witness when the kernel supports wipe-on-fork mappings.
    pub(crate) fn new() -> Option<Self> {
        let page = ForkMarkerPage::new().ok()?;
        // SAFETY: The unpublished page is aligned for an atomic marker and
        // remains owned by `page` until publication or drop.
        unsafe { &*(page.address() as *const AtomicU32) }.store(1, Ordering::Release);
        Some(Self { page })
    }

    /// Returns the address retained in the process slot before publication.
    pub(crate) fn address(&self) -> usize {
        self.page.address()
    }
}

/// Checks that an inherited slot's witness was not wiped by a fork.
/// Missing witnesses use the PID fallback on older kernels.
pub(crate) fn fork_witness_valid(address: Option<usize>) -> bool {
    address.is_none_or(|address| {
        // SAFETY: A published witness mapping remains allocated through exit.
        unsafe { &*(address as *const AtomicU32) }.load(Ordering::Acquire) != 0
    })
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
#[allow(unsafe_code, clippy::unwrap_used)]
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
    fn wiped_slot_witness_rejects_even_a_matching_pid() {
        assert!(fork_witness_valid(None));
        let Some(witness) = FreshForkWitness::new() else {
            return;
        };
        let address = witness.address();
        assert!(fork_witness_valid(Some(address)));
        // SAFETY: This test owns the unpublished witness mapping and changes
        // only its atomic word to model the kernel's fork-time wipe.
        unsafe { &*(address as *const AtomicU32) }.store(0, Ordering::Release);
        assert!(!fork_witness_valid(Some(address)));
    }

    #[test]
    fn captured_owner_stays_invalid_after_a_new_capture() {
        let inherited = ProcessIdentity::capture().unwrap();
        let Some(witness) = &inherited.0.witness else {
            return;
        };
        let address = witness.address();
        assert!(inherited.check().is_ok());
        // SAFETY: This test owns the witness page and changes only its atomic
        // word to model a fork into a namespace with the same visible PID.
        unsafe { &*(address as *const AtomicU32) }.store(0, Ordering::Release);
        assert!(inherited.check().is_err());
        let fresh = ProcessIdentity::capture().unwrap();
        assert!(fresh.check().is_ok());
        assert!(inherited.check().is_err());
    }

    #[test]
    fn namespace_fallback_distinguishes_foreign_from_unavailable() {
        let identity = ProcessIdentity(
            Shared::new(ProcessIdentityState {
                pid: std::process::id(),
                witness: None,
                namespace: Some((11, 13)),
            })
            .unwrap(),
        );
        assert!(
            identity
                .matches_current_process_with_namespace(|| Ok((11, 13)))
                .unwrap()
        );
        assert!(
            !identity
                .matches_current_process_with_namespace(|| Ok((11, 14)))
                .unwrap()
        );
        assert_eq!(
            identity
                .matches_current_process_with_namespace(|| Err(io::ErrorKind::NotFound.into()))
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
    }

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
