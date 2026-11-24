// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! A late caller TLS destructor must receive an error instead of aborting.

#![allow(clippy::unwrap_used)]

use rocddi::session::{DriverContextLifetime, Session};
use std::cell::RefCell;
use std::process::Command;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

struct Cleanup {
    session: Session,
    completed: Arc<AtomicBool>,
}

impl Drop for Cleanup {
    fn drop(&mut self) {
        // Admission's TLS was initialized later and has already been destroyed.
        assert!(self.session.enumerate(&mut |_| Ok(())).is_err());
        // Resource retirement does not need new-operation admission.
        self.session.destroy().unwrap();
        self.completed.store(true, Ordering::Release);
    }
}

thread_local! {
    static CLEANUP: RefCell<Option<Cleanup>> = const { RefCell::new(None) };
}

#[test]
fn session_api_from_late_tls_destructor() {
    const CHILD: &str = "ROCDDI_TEST_LATE_TLS_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let completed = Arc::new(AtomicBool::new(false));
        let observed = completed.clone();
        std::thread::spawn(move || {
            CLEANUP.with(|slot| {
                let session = Session::new(DriverContextLifetime::Session).unwrap();
                *slot.borrow_mut() = Some(Cleanup { session, completed });
                // Passive discovery initializes admission TLS without acquiring
                // GPU execution state. Its native availability is immaterial.
                let _ = slot
                    .borrow()
                    .as_ref()
                    .unwrap()
                    .session
                    .enumerate(&mut |_| Ok(()));
            });
        })
        .join()
        .unwrap();
        assert!(observed.load(Ordering::Acquire));
        return;
    }
    // A regression aborts during TLS destruction, so isolate that outcome from
    // the test runner and report it as a failed child process.
    let output = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "session_api_from_late_tls_destructor",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
