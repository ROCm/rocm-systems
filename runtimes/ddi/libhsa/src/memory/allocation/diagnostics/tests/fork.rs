// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Fork rejection must precede environment and inherited stderr locks.

use super::*;

#[test]
/// Isolates a deliberately inherited stderr lock from the rest of the tests.
fn inherited_stderr_child() {
    if std::env::var_os("ROCDDI_MEMORY_FORK_TEST_CHILD").is_none() {
        return;
    }
    crate::platform::bind_process().unwrap();
    let (held_sender, held_receiver) = std::sync::mpsc::sync_channel(0);
    let (release_sender, release_receiver) = std::sync::mpsc::sync_channel(0);
    let holder = std::thread::spawn(move || {
        let _stderr = std::io::stderr().lock();
        held_sender.send(()).unwrap();
        release_receiver.recv().unwrap();
    });
    held_receiver.recv().unwrap();
    // SAFETY: The child uses only the lock-free process guard and the public
    // allocation entry's inherited-process rejection, then exits without Drop.
    let child = unsafe { libc::syscall(libc::SYS_fork) };
    if child == 0 {
        // SAFETY: Bound this deadlock witness if rejection touches inherited
        // locks; no inherited destructor or test-harness state may run here.
        unsafe { libc::alarm(3) };
        let original_pointer = std::ptr::dangling_mut::<u8>().cast::<std::ffi::c_void>();
        let mut pointer = original_pointer;
        // SAFETY: Output storage is valid. The fork witness must reject the
        // call before lifecycle access or any native allocation operation.
        let status = unsafe {
            crate::api::memory::allocation::hsa_amd_memory_pool_allocate(
                HsaMemoryPool { handle: 0x123 },
                4 << 20,
                0,
                &raw mut pointer,
            )
        };
        let rejected = status == NOT_INITIALIZED
            && pointer == original_pointer
            && !crate::platform::process_is_current();
        // SAFETY: _exit avoids all inherited runtime locks and destructors.
        unsafe { libc::_exit(i32::from(!rejected)) }
    }
    let mut status = -1;
    // SAFETY: A successful fork's returned PID names only this test's child.
    let waited = if child > 0 {
        unsafe { libc::waitpid(i32::try_from(child).unwrap(), &raw mut status, 0) }
    } else {
        -1
    };
    // Release stderr before any assertion can emit a parent-process panic.
    release_sender.send(()).unwrap();
    holder.join().unwrap();
    assert!(child > 0);
    assert_eq!(i64::from(waited), child);
    assert_eq!(
        status, 0,
        "inherited allocation entered a locked diagnostic"
    );
}

#[test]
/// The public entry rejects a fork child silently while another thread owns stderr.
fn inherited_allocation_rejects_before_diagnostic_locks() {
    for setting in [None, Some("0"), Some("1")] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "memory::allocation::diagnostics::tests::fork::inherited_stderr_child",
                "--nocapture",
            ])
            .env("ROCDDI_MEMORY_FORK_TEST_CHILD", "1")
            .env_remove("ROCDDI_HSA_MEMORY_DIAGNOSTICS");
        if let Some(setting) = setting {
            command.env("ROCDDI_HSA_MEMORY_DIAGNOSTICS", setting);
        }
        let output = command.output().unwrap();
        assert!(output.status.success(), "{output:?}");
        assert!(output.stderr.is_empty(), "{output:?}");
    }
}
