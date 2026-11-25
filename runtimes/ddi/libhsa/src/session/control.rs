// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA call leases, cancellation, and uncertain session ownership.

use super::publish_stop_request;
use crate::admission::{ApiLeaseScope, InFlightToken, InFlightTracker};
use crate::ffi::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Sticky uncertainty shared by native work in one HSA session.
pub(crate) struct SessionHealth {
    /// Any cleanup result that could leave native users of frontend owners.
    uncertain: AtomicBool,
}

impl SessionHealth {
    /// Starts a session without uncertain native owners.
    pub(crate) fn new() -> Self {
        Self {
            uncertain: AtomicBool::new(false),
        }
    }

    /// Retains the session when native retirement cannot be proved.
    pub(crate) fn quarantine(&self) {
        self.uncertain.store(true, Ordering::Release);
    }

    /// Reports whether any participant retained an uncertain owner.
    pub(crate) fn is_quarantined(&self) -> bool {
        self.uncertain.load(Ordering::Acquire)
    }
}

/// Lifetime authority shared by the session's frontend operations.
pub(crate) struct SessionControl {
    /// Shared stop request observed by those workers.
    pub(crate) stop_workers: Arc<AtomicBool>,
    /// Cancels ordinary waits and native work while cleanup services stay live.
    pub(crate) cancel_work: Arc<AtomicBool>,
    /// Counts native work that must finish before session release.
    pub(crate) inflight: Arc<InFlightTracker>,
    /// Sticky native cleanup uncertainty for this HSA session.
    pub(crate) health: Arc<SessionHealth>,
}

impl SessionControl {
    /// Attaches ordinary and cleanup work to the same HSA admission owner.
    pub(crate) fn new(inflight: Arc<InFlightTracker>) -> Self {
        Self {
            inflight,
            health: Arc::new(SessionHealth::new()),
            stop_workers: Arc::new(AtomicBool::new(false)),
            cancel_work: Arc::new(AtomicBool::new(false)),
        }
    }
    /// Requests all runtime workers to stop before teardown.
    pub(crate) fn request_stop(&self) {
        self.cancel_work.store(true, Ordering::Release);
        publish_stop_request(&self.stop_workers, &self.inflight);
    }

    /// Closes public admission without stopping callbacks needed by tooling.
    pub(crate) fn request_shutdown(&self) {
        self.cancel_work.store(true, Ordering::Release);
        self.inflight.close();
    }

    /// Cleanup stop packets must still complete after ordinary work cancels.
    pub(crate) fn operation_stop(&self) -> Arc<AtomicBool> {
        if ApiLeaseScope::cleanup() {
            self.stop_workers.clone()
        } else {
            self.cancel_work.clone()
        }
    }

    /// Keeps a cleanup task's nested HSA calls admitted when it moves to an
    /// asynchronous worker. Ordinary tasks use cancellation and worker joins.
    pub(crate) fn inherit_cleanup(&self) -> Result<Option<InFlightToken>, Status> {
        if ApiLeaseScope::cleanup() {
            self.inflight.enter().map(Some).ok_or(INVALID_RUNTIME_STATE)
        } else {
            Ok(None)
        }
    }
}
