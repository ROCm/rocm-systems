// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Fork-child teardown must not enter a parent-owned allocator callback.

#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(unsafe_code, clippy::unwrap_used)]

use rocddi::memory::storage::Allocator;
use rocddi::session::{DriverContextLifetime, Session};
use std::ffi::c_void;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

struct CallbackContext {
    lock: Mutex<()>,
    frees: AtomicUsize,
}

impl CallbackContext {
    fn allocator(&self) -> Allocator {
        // SAFETY: The context remains live through parent cleanup and the
        // child's immediate _exit.
        unsafe {
            Allocator::from_callbacks(
                std::ptr::from_ref(self).cast_mut().cast(),
                Some(allocate),
                Some(free),
            )
            .unwrap()
        }
    }
}

unsafe extern "C" fn allocate(_data: *mut c_void, length: u64, alignment: u64) -> *mut c_void {
    let (Ok(length), Ok(alignment)) = (usize::try_from(length), usize::try_from(alignment)) else {
        return std::ptr::null_mut();
    };
    let mut pointer = std::ptr::null_mut();
    // SAFETY: The allocator contract supplies a nonzero length and valid
    // power-of-two alignment.
    if unsafe { libc::posix_memalign(&raw mut pointer, alignment, length) } == 0 {
        pointer
    } else {
        std::ptr::null_mut()
    }
}

unsafe extern "C" fn free(data: *mut c_void, pointer: *mut c_void) {
    // SAFETY: Each callback receives the stable context above.
    let context = unsafe { &*data.cast::<CallbackContext>() };
    let _held = context.lock.lock().unwrap();
    context.frees.fetch_add(1, Ordering::Relaxed);
    // SAFETY: The pointer came from posix_memalign in this allocator.
    unsafe { libc::free(pointer) };
}

#[test]
fn inherited_final_session_skips_allocator_callback() {
    let context = CallbackContext {
        lock: Mutex::new(()),
        frees: AtomicUsize::new(0),
    };
    let mut session = context
        .allocator()
        .with_scope(|| Session::new(DriverContextLifetime::Session))
        .unwrap();
    let held = context.lock.lock().unwrap();
    // SAFETY: The child only drops the copied session and exits without
    // running other inherited Rust destructors.
    let child = unsafe { libc::syscall(libc::SYS_fork) };
    assert!(child >= 0);
    if child == 0 {
        // SAFETY: Bound a regression if the copied callback deadlocks.
        unsafe { libc::alarm(3) };
        let inherited = !session.is_current_process();
        drop(session);
        // SAFETY: _exit avoids inherited test-harness cleanup.
        unsafe { libc::_exit(i32::from(!inherited)) }
    }
    let pid = i32::try_from(child).unwrap();
    let mut status = 0;
    // SAFETY: Wait only for the child created above.
    assert_eq!(unsafe { libc::waitpid(pid, &raw mut status, 0) }, pid);
    assert_eq!(status, 0);

    drop(held);
    session.destroy().unwrap();
    let frees_before_drop = context.frees.load(Ordering::Relaxed);
    drop(session);
    assert_eq!(context.frees.load(Ordering::Relaxed), frees_before_drop + 1);
}
