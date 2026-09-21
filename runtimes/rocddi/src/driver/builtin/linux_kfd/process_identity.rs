//! A fork-sensitive Linux process marker for native owners.
//!
//! The kernel zeroes one private page in a fork child. An inherited owner's
//! captured PID then differs from the marker before it can touch callbacks,
//! files, or locks. A new session in the child installs the child's PID in
//! that page, leaving all inherited owners invalid. If the kernel cannot
//! provide `MADV_WIPEONFORK`, checks keep using `getpid`.

#![allow(unsafe_code)]

use std::ffi::{c_int, c_void};
use std::io;
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

unsafe extern "C" {
    fn getpagesize() -> c_int;
    fn mmap(
        address: *mut c_void,
        length: usize,
        protection: c_int,
        flags: c_int,
        fd: c_int,
        offset: i64,
    ) -> *mut c_void;
    fn madvise(address: *mut c_void, length: usize, advice: c_int) -> c_int;
    fn munmap(address: *mut c_void, length: usize) -> c_int;
}

const PROT_READ_WRITE: c_int = 3;
const MAP_PRIVATE: c_int = 2;
const MAP_ANONYMOUS: c_int = 0x20;
const MADV_WIPEONFORK: c_int = 18;
const FALLBACK_TO_GETPID: usize = 1;

// Zero means uninitialized. The one-page mapping remains until process exit,
// so a published pointer stays valid even when sessions and frontends unload.
// One means the kernel lacks the wipe-on-fork contract and checks use getpid.
static MARKER: AtomicUsize = AtomicUsize::new(0);

fn install_fallback() -> usize {
    MARKER
        .compare_exchange(0, FALLBACK_TO_GETPID, Ordering::AcqRel, Ordering::Acquire)
        .unwrap_or_else(|winner| winner)
}

fn marker_state() -> usize {
    let state = MARKER.load(Ordering::Acquire);
    if state != 0 {
        return state;
    }
    // SAFETY: getpagesize has no pointer arguments or mutable process state.
    let page = unsafe { getpagesize() };
    let Ok(page) = usize::try_from(page) else {
        return install_fallback();
    };
    if page < size_of::<AtomicU32>() || !page.is_power_of_two() {
        return install_fallback();
    }
    // SAFETY: The private anonymous page is owned by this call until the
    // publish CAS succeeds; it is never unmapped after publication.
    let mapping = unsafe {
        mmap(
            ptr::null_mut(),
            page,
            PROT_READ_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if mapping as isize == -1 {
        return install_fallback();
    }
    // SAFETY: mmap returned a writable page-aligned mapping of this length.
    let advised = unsafe { madvise(mapping, page, MADV_WIPEONFORK) };
    if advised != 0 {
        // SAFETY: The mapping has not been published to another thread.
        let _ = unsafe { munmap(mapping, page) };
        return install_fallback();
    }
    // The marker stays zero until allocation passes the session's existing
    // process check. A concurrent owner check uses getpid while it is zero.
    match MARKER.compare_exchange(0, mapping as usize, Ordering::AcqRel, Ordering::Acquire) {
        Ok(_) => mapping as usize,
        Err(winner) => {
            // SAFETY: Only the winning mapping is visible to other threads.
            let _ = unsafe { munmap(mapping, page) };
            winner
        }
    }
}

/// Enables cheap checks after an allocation passed its session process check.
/// This preserves inert instance creation and the getpid fallback before the
/// first host allocation or after a fork.
pub(super) fn prepare_for_hot_checks() {
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
pub(super) fn check_process(process: u32) -> io::Result<()> {
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
    use std::ffi::c_long;

    unsafe extern "C" {
        fn syscall(number: c_long, ...) -> c_long;
        fn waitpid(pid: c_int, status: *mut c_int, options: c_int) -> c_int;
        fn _exit(status: c_int) -> !;
    }

    const SYS_FORK: c_long = 57;

    #[test]
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
