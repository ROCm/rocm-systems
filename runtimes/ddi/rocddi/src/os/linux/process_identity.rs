// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! A fork-sensitive Linux process marker for native owners.
//!
//! The kernel zeroes each process witness page in a fork child. A thread
//! reuses its current witness for new owners, but a child replaces its wiped
//! cache without rewriting the witness held by inherited owners. If the
//! kernel cannot provide `MADV_WIPEONFORK`, checks also compare the PID
//! namespace identity.

#![allow(unsafe_code)]

use super::memory::ForkMarkerPage;
use crate::memory::storage::{Allocator, Owned, Shared};
use std::cell::RefCell;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::sync::atomic::{AtomicPtr, AtomicU32, Ordering};

/// Creating process identity retained by one session and its native owners.
/// The witness cache is local to the capturing thread; creating a passive
/// session still initializes no process-global KFD state.
#[derive(Clone)]
pub(crate) struct ProcessIdentity(Shared<ProcessIdentityState>);

thread_local! {
    /// One creating-process epoch reused by owners born on this thread.
    static CURRENT_PROCESS: RefCell<Option<ProcessIdentity>> = const { RefCell::new(None) };
}

// A session records the process image before any native activation. Keep the
// published identity until exit so a child can create passive sessions, but
// cannot acquire its first GPU connection without exec. Atomic publication
// avoids an inherited lock if another thread was binding during fork.
static RUNTIME_IMAGE: AtomicPtr<ProcessIdentity> = AtomicPtr::new(std::ptr::null_mut());

fn allocate_image_owner(
    process: ProcessIdentity,
    allocator: Allocator,
) -> io::Result<*mut ProcessIdentity> {
    allocator
        .with_scope(|| Owned::new(process))
        .map(Owned::into_raw)
        .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))
}

/// Records the first session or native connection attempt in this image.
pub(crate) fn note_runtime_image() -> io::Result<()> {
    if !RUNTIME_IMAGE.load(Ordering::Acquire).is_null() {
        return Ok(());
    }
    // The published owner outlives frontend callback policies and must not
    // abort the process if its metadata allocation fails.
    let candidate = allocate_image_owner(ProcessIdentity::capture()?, Allocator::system())?;
    if RUNTIME_IMAGE
        .compare_exchange(
            std::ptr::null_mut(),
            candidate,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
        .is_err()
    {
        // SAFETY: This unconsumed pointer came from Owned::into_raw above and
        // was never published. Its system allocator releases the losing slot.
        unsafe { drop(Owned::from_raw(candidate)) };
    }
    Ok(())
}

/// Rejects native activation if the image was first used before fork.
pub(crate) fn require_runtime_image() -> io::Result<()> {
    note_runtime_image()?;
    let current = RUNTIME_IMAGE.load(Ordering::Acquire);
    // SAFETY: The published identity is retained for the image lifetime.
    unsafe { &*current }.check()
}

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
    /// Captures the current thread's fork witness without opening a device.
    pub(crate) fn capture() -> io::Result<Self> {
        CURRENT_PROCESS
            .try_with(|slot| {
                if let Some(current) = slot.borrow().as_ref() {
                    match current.matches_current_process() {
                        Ok(true) => return Ok(current.clone()),
                        Ok(false) => {}
                        Err(source) => return Err(source),
                    }
                }
                let fresh = Self::capture_fresh()?;
                slot.replace(Some(fresh.clone()));
                Ok(fresh)
            })
            .unwrap_or_else(|_| Self::capture_fresh())
    }

    /// Creates a new epoch after fork or when this thread has no cache.
    fn capture_fresh() -> io::Result<Self> {
        let witness = FreshForkWitness::new();
        let namespace = if witness.is_none() {
            Some(pid_namespace()?)
        } else {
            None
        };
        // This support epoch outlives an individual AMDF allocator policy.
        let state = Allocator::system()
            .with_scope(|| {
                Shared::new(ProcessIdentityState {
                    pid: std::process::id(),
                    witness,
                    namespace,
                })
            })
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        Ok(Self(state))
    }

    /// Distinguishes a confirmed fork from an unavailable namespace check.
    /// Process slots replace an inherited lock only on a confirmed mismatch.
    pub(crate) fn matches_current_process(&self) -> io::Result<bool> {
        self.matches_current_process_with_fallback(std::process::id, pid_namespace)
    }

    fn matches_current_process_with_fallback(
        &self,
        pid: impl FnOnce() -> u32,
        namespace: impl FnOnce() -> io::Result<(u64, u64)>,
    ) -> io::Result<bool> {
        if let Some(witness) = &self.0.witness {
            return Ok(fork_witness_valid(Some(witness.address())));
        }
        if self.0.pid != pid() {
            return Ok(false);
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
        let mut identity = Self::capture_fresh().expect("test process identity");
        let state = Shared::get_mut(&mut identity.0).expect("test identity has one owner");
        // A valid wipe-on-fork witness is authoritative; use the PID fallback
        // to model a foreign owner without forking in these tests.
        state.witness = None;
        state.pid = pid;
        identity
    }
}

/// A wipe-on-fork page that validates process owners from one capture epoch.
/// An inherited owner never rewrites this witness after a fork, even if a
/// descendant later reuses the original PID.
pub(crate) struct FreshForkWitness {
    /// Unpublished witness mapping, dropped when slot publication loses a race.
    page: ForkMarkerPage,
}

impl FreshForkWitness {
    /// Creates a witness or preserves the Linux failure for its caller.
    pub(crate) fn try_new() -> io::Result<Self> {
        let page = ForkMarkerPage::new()?;
        // SAFETY: The unpublished page is aligned for an atomic marker and
        // remains owned by `page` until publication or drop.
        unsafe { &*(page.address() as *const AtomicU32) }.store(1, Ordering::Release);
        Ok(Self { page })
    }

    /// Creates a witness when the kernel supports wipe-on-fork mappings.
    pub(crate) fn new() -> Option<Self> {
        Self::try_new().ok()
    }

    /// Returns the address retained in the process slot before publication.
    pub(crate) fn address(&self) -> usize {
        self.page.address()
    }

    /// Retains a published witness for the lifetime of this process image.
    pub(crate) fn retain_until_exit(self) -> usize {
        self.page.retain_until_exit()
    }
}

/// Checks that an inherited slot's witness was not wiped by a fork.
/// The caller handles the PID and namespace fallback when no page exists.
pub(crate) fn fork_witness_valid(address: Option<usize>) -> bool {
    address.is_none_or(|address| {
        // SAFETY: A published witness mapping remains allocated through exit.
        unsafe { &*(address as *const AtomicU32) }.load(Ordering::Acquire) != 0
    })
}

#[cfg(all(test, target_arch = "x86_64"))]
#[allow(unsafe_code, clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::ffi::{c_int, c_long, c_void};

    unsafe extern "C" fn deny_allocation(
        _data: *mut c_void,
        _length: u64,
        _alignment: u64,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }

    unsafe extern "C" fn unused_free(_data: *mut c_void, _pointer: *mut c_void) {}

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
        fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
        fn _exit(status: c_int) -> !;
    }

    const SYS_FORK: c_long = 57;

    #[test]
    fn runtime_image_owner_allocation_reports_out_of_memory() {
        // SAFETY: The callback always fails, so no allocation or free can
        // outlive this test; the paired free function satisfies the API.
        let failing = unsafe {
            Allocator::from_callbacks(
                std::ptr::null_mut(),
                Some(deny_allocation),
                Some(unused_free),
            )
            .unwrap()
        };
        let process = ProcessIdentity::capture().unwrap();
        let error = allocate_image_owner(process, failing).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::OutOfMemory);

        // The runtime-image owner deliberately selects the system allocator
        // even while an AMDF callback policy is active on this thread.
        let process = ProcessIdentity::capture().unwrap();
        let candidate = failing
            .with_scope(|| allocate_image_owner(process, Allocator::system()))
            .unwrap();
        // SAFETY: This candidate was never published and came from into_raw.
        unsafe { drop(Owned::from_raw(candidate)) };
    }

    #[test]
    fn passive_parent_image_prevents_child_gpu_activation() {
        note_runtime_image().unwrap();
        // SAFETY: The child reads only the published process marker and exits
        // without entering an inherited lock or running Rust destructors.
        let child = unsafe { syscall(SYS_FORK) };
        assert!(child >= 0);
        if child == 0 {
            let rejected = note_runtime_image().is_ok()
                && require_runtime_image()
                    .err()
                    .is_some_and(|source| source.kind() == io::ErrorKind::Unsupported);
            // SAFETY: _exit leaves inherited state untouched.
            unsafe { _exit(i32::from(!rejected)) }
        }
        let child_pid = c_int::try_from(child).unwrap_or_default();
        assert!(child_pid > 0);
        let mut status = 0;
        // SAFETY: This waits only for the child created above.
        assert_eq!(unsafe { waitpid(child_pid, &raw mut status, 0) }, child_pid);
        assert_eq!(status, 0);
        assert!(require_runtime_image().is_ok());
    }

    #[test]
    fn thread_reuses_witness_without_revalidating_inherited_owners() {
        let inherited = ProcessIdentity::capture().unwrap();
        let same_epoch = ProcessIdentity::capture().unwrap();
        assert!(Shared::ptr_eq(&inherited.0, &same_epoch.0));
        let Some(witness) = &inherited.0.witness else {
            return;
        };
        // Model the kernel's wipe in a fork child. The cache may now publish
        // a fresh epoch, while owners of this page remain invalid.
        let marker = witness.address() as *const AtomicU32;
        // SAFETY: The captured page remains mapped and aligned.
        unsafe { &*marker }.store(0, Ordering::Release);
        let fresh = ProcessIdentity::capture().unwrap();
        assert!(!Shared::ptr_eq(&inherited.0, &fresh.0));
        assert!(inherited.check().is_err());
        assert!(fresh.check().is_ok());
    }

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
    fn witness_path_skips_pid_and_namespace_fallback() {
        let mut identity = ProcessIdentity::capture_fresh().unwrap();
        let Some(witness) = &identity.0.witness else {
            return;
        };
        let address = witness.address();
        Shared::get_mut(&mut identity.0).unwrap().pid = std::process::id().wrapping_add(1);
        let check_without_fallback = |identity: &ProcessIdentity| {
            let pid_called = Cell::new(false);
            let namespace_called = Cell::new(false);
            let matches = identity
                .matches_current_process_with_fallback(
                    || {
                        pid_called.set(true);
                        identity.0.pid
                    },
                    || {
                        namespace_called.set(true);
                        Ok((0, 0))
                    },
                )
                .unwrap();
            assert!(!pid_called.get());
            assert!(!namespace_called.get());
            matches
        };
        assert!(check_without_fallback(&identity));

        // Model the kernel's fork-time wipe; even a reused PID must not make
        // an inherited witness valid or enter the fallback.
        Shared::get_mut(&mut identity.0).unwrap().pid = std::process::id();
        // SAFETY: This test owns the unpublished, aligned witness page.
        unsafe { &*(address as *const AtomicU32) }.store(0, Ordering::Release);
        assert!(!check_without_fallback(&identity));
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
                .matches_current_process_with_fallback(std::process::id, || Ok((11, 13)))
                .unwrap()
        );
        assert!(
            !identity
                .matches_current_process_with_fallback(std::process::id, || Ok((11, 14)))
                .unwrap()
        );
        assert_eq!(
            identity
                .matches_current_process_with_fallback(std::process::id, || {
                    Err(io::ErrorKind::NotFound.into())
                })
                .unwrap_err()
                .kind(),
            io::ErrorKind::NotFound
        );
    }

    #[test]
    #[allow(clippy::used_underscore_items)]
    fn raw_fork_rejects_inherited_owner() {
        let inherited = ProcessIdentity::capture().unwrap();
        if inherited.0.witness.is_none() {
            return;
        }
        assert!(inherited.check().is_ok());
        // SAFETY: The child only reads this owner marker, calls getpid, and
        // exits without accessing a test-harness lock or allocator.
        let child = unsafe { syscall(SYS_FORK) };
        assert!(child >= 0);
        if child == 0 {
            let rejected = inherited.check().is_err();
            // SAFETY: _exit does not run inherited Rust destructors.
            unsafe { _exit(i32::from(!rejected)) }
        }
        let child_pid = c_int::try_from(child).unwrap_or_default();
        assert!(child_pid > 0);
        let mut status = 0;
        // SAFETY: This waits only for the child created by the syscall above.
        assert_eq!(unsafe { waitpid(child_pid, &raw mut status, 0) }, child_pid);
        assert_eq!(status, 0);
        assert!(inherited.check().is_ok());
    }
}
