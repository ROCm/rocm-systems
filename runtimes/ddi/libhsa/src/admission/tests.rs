// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

use super::*;
use std::thread;

#[test]
#[allow(clippy::unwrap_used)]
/// Verifies that shutdown waits for inflight calls.
fn shutdown_waits_for_inflight_calls() {
    let inflight = Arc::new(InFlightTracker::new());
    let token = inflight.enter().unwrap();
    let (started_send, started_recv) = std::sync::mpsc::channel();
    let (done_send, done_recv) = std::sync::mpsc::channel();
    let worker_inflight = inflight.clone();
    let worker = thread::spawn(move || {
        worker_inflight.close();
        started_send.send(()).unwrap();
        worker_inflight.wait_idle();
        done_send.send(()).unwrap();
    });
    started_recv.recv_timeout(Duration::from_secs(1)).unwrap();
    assert!(done_recv.recv_timeout(Duration::from_millis(10)).is_err());
    drop(token);
    done_recv.recv_timeout(Duration::from_secs(1)).unwrap();
    worker.join().unwrap();
}
