// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Prepared HSA session retirement and retained native cleanup ownership.

use super::*;
#[cfg(test)]
use crate::events::CallbackScope;
use crate::events::SystemEventShutdown;
use crate::pc_sampling::session::SessionHandle;
use crate::queue_intercept::InterceptQueue;
use std::thread;

/// Callback work detached while HSA storage remains published for `OnUnload`.
pub(crate) struct ToolShutdownWork {
    /// Native operations admitted before the stop request.
    inflight: Arc<InFlightTracker>,
    /// Detached system callbacks and their native event worker.
    system_events: SystemEventShutdown,
    /// Asynchronous memory operation workers.
    memory_workers: Vec<JoinHandle<()>>,
    /// Serialized signal callback worker.
    signal_worker: Option<JoinHandle<()>>,
    /// Per-queue error callback workers.
    queue_workers: Vec<JoinHandle<()>>,
    /// Proxy queues whose interception callbacks must stop.
    intercept_queues: Vec<Arc<InterceptQueue>>,
    /// Sampling sessions whose callbacks must finish before tool unload.
    pc_sampling_sessions: Vec<Arc<SessionHandle>>,
}

impl ToolShutdownWork {
    /// Joins every callback producer, then delivers the final system event.
    pub(crate) fn quiesce(mut self) -> Status {
        self.inflight.wait_idle();
        let mut failed = false;
        for queue in &self.intercept_queues {
            failed |= queue.stop().is_err();
        }
        for worker in self.queue_workers {
            failed |= worker.join().is_err();
        }
        failed |= self.system_events.join_worker().is_err();
        for worker in self.memory_workers {
            failed |= worker.join().is_err();
        }
        if let Some(worker) = self.signal_worker {
            failed |= worker.join().is_err();
        }
        for session in self.pc_sampling_sessions {
            failed |= session.quiesce_for_tool_unload() != SUCCESS;
        }
        if failed {
            return ERROR;
        }
        self.system_events.notify();
        SUCCESS
    }
}

/// Result of retiring one HSA session and its remaining owner.
#[allow(
    clippy::large_enum_variant,
    reason = "boxing HSA state would allocate on an uncertain cleanup path"
)]
pub(crate) enum ShutdownOutcome {
    /// HSA and native owners have been released.
    Complete,
    /// HSA owners have retired, but the native session needs another attempt.
    NativeRetry {
        /// Error reported by this cleanup attempt.
        status: Status,
        /// Session retained for the next shutdown call.
        session: SessionRetirement,
    },
    /// Uncertain HSA cleanup retains the entire HSA session until process exit.
    RetainedSession {
        /// Error reported by this cleanup attempt.
        status: Status,
        /// Complete frontend owner graph retained even if this outcome drops.
        state: RetainOnDrop<HsaSession>,
    },
}

impl ShutdownOutcome {
    /// Returns the HSA status reported for this attempt.
    pub(crate) fn status(&self) -> Status {
        match self {
            Self::Complete => SUCCESS,
            Self::NativeRetry { status, .. } | Self::RetainedSession { status, .. } => *status,
        }
    }

    /// Retries a native session after all HSA objects have retired.
    pub(crate) fn retry(mut retirement: SessionRetirement) -> Self {
        match retirement.try_close() {
            RetirementOutcome::Closed => Self::Complete,
            RetirementOutcome::Busy => Self::NativeRetry {
                status: RESOURCE_BUSY,
                session: retirement,
            },
            RetirementOutcome::Failed(error) => Self::NativeRetry {
                status: map_error(error),
                session: retirement,
            },
        }
    }
}

/// Schedules cleanup after in-flight native work has released its borrows.
pub(crate) fn defer_cleanup<T: Send + 'static>(
    owner: T,
    cleanup: impl FnOnce(T) + Send + 'static,
) -> Result<(), Option<T>> {
    let pending = Arc::new(Mutex::new(Some(owner)));
    let worker_pending = pending.clone();
    if thread::Builder::new()
        .name("rocddi-shutdown".into())
        .spawn(move || {
            let owner = worker_pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .take();
            if let Some(owner) = owner {
                cleanup(owner);
            }
        })
        .is_err()
    {
        return Err(pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take());
    }
    Ok(())
}

/// Publishes a shutdown request without waiting for the worker that called it.
pub(crate) fn publish_stop_request(stop_workers: &AtomicBool, inflight: &InFlightTracker) {
    stop_workers.store(true, Ordering::Release);
    inflight.close();
}

impl HsaSession {
    /// Detaches callback workers under the lifecycle lock. Their joins and
    /// the final event run after that lock is released, before tool `OnUnload`.
    pub(crate) fn begin_tool_shutdown(&mut self) -> ToolShutdownWork {
        self.control.request_stop();
        let (queue_workers, intercept_queues) = self.queue_state.take_callback_workers();
        ToolShutdownWork {
            inflight: self.control.inflight.clone(),
            system_events: self.events.take_shutdown(),
            memory_workers: self.memory_state.take_workers(),
            signal_worker: self.signal_state.take_worker(),
            queue_workers,
            intercept_queues,
            pc_sampling_sessions: self.pc_sampling_state.sessions_for_tool_unload(),
        }
    }

    /// Prepares an isolated test session without public/tool orchestration.
    #[cfg(test)]
    pub(crate) fn stop(self) -> ShutdownOutcome {
        let mut state = RetainOnDrop::new(self);
        if let Err(status) = state.begin_native_retirement() {
            return ShutdownOutcome::RetainedSession { status, state };
        }
        state.logging.stop();
        if state.running_on_worker() || CallbackScope::active() {
            state.control.request_stop();
            return ShutdownOutcome::RetainedSession {
                status: INVALID_RUNTIME_STATE,
                state,
            };
        }
        state.control.request_stop();
        state.control.inflight.wait_idle();
        state.memory_state.join_workers();
        state.signal_state.join_worker();
        let mut system_events = state.events.take_shutdown();
        system_events.notify();
        let _ = system_events.join_worker();
        state.into_inner().retire_prepared()
    }

    /// Retires backing after the lifecycle owner has drained calls, stopped
    /// callbacks and tools, and closed core session admission.
    pub(crate) fn retire_prepared(self) -> ShutdownOutcome {
        // Unwinding cannot release backing whose last native user is uncertain.
        let mut state = RetainOnDrop::new(self);
        if state.control.health.is_quarantined() {
            state.pc_sampling_state.quarantine_all();
            return ShutdownOutcome::RetainedSession {
                status: INVALID_RUNTIME_STATE,
                state,
            };
        }
        if !state.pc_sampling_state.shutdown() {
            return ShutdownOutcome::RetainedSession {
                status: INVALID_RUNTIME_STATE,
                state,
            };
        }
        if !state.queue_state.shutdown() {
            return ShutdownOutcome::RetainedSession {
                status: INVALID_RUNTIME_STATE,
                state,
            };
        }
        // HSA queues are stopped; foreign queues may still execute the shared
        // normal trap. Release only HSA handles, leaving its native owner live.
        state.debugger = None;
        let signal_status = state.signal_state.destroy_events();
        if signal_status != SUCCESS {
            // A failed native event destroy can still retain its KFD context.
            // Do not discard the session that owns the remaining driver state.
            return ShutdownOutcome::RetainedSession {
                status: signal_status,
                state,
            };
        }
        if !state.loader.can_release() {
            return ShutdownOutcome::RetainedSession {
                status: INVALID_RUNTIME_STATE,
                state,
            };
        }
        state.image_state.clear();
        state.loader.clear();
        state.memory_state.clear();
        state.signal_state.clear_storage();
        state.pc_sampling_state.clear_retired();
        state.agents.gpus.clear();
        let Some(session) = state.native_session.take() else {
            return ShutdownOutcome::RetainedSession {
                status: ERROR,
                state,
            };
        };
        let retirement = session.into_retirement();
        drop(state.into_inner());
        ShutdownOutcome::retry(retirement)
    }
}
