// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA session ownership extending a native rocddi session through composition.

use crate::admission::{InFlightTracker, process_admission};
use crate::agent::Agents;
use crate::config::HsaConfig;
use crate::debugger::PendingDebugger;
use crate::events::SystemEvents;
use crate::ffi::*;
use crate::image::ImageState;
use crate::loader::LoaderState;
use crate::logging::{LogOutput, LogWork};
use crate::memory::MemoryState;
use crate::pc_sampling::PcSamplingState;
use crate::queue::QueueState;
use crate::signal::SignalState;
use crate::status::map_error;
use rocddi::memory::storage::RetainOnDrop;
use rocddi::session::{
    DriverContextLifetime, RetirementOutcome, Session as NativeSession, SessionRetirement,
};
use rocddi::topology::TopologyGraphSnapshot;
use std::fmt::Arguments;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

mod control;
pub(crate) use control::{SessionControl, SessionHealth};
mod retirement;
mod startup;
pub(crate) use retirement::{ShutdownOutcome, defer_cleanup, publish_stop_request};
pub(crate) use startup::InitRollback;

/// HSA-visible objects and policy attached to one native session.
///
/// Every integer handle and public pointer accepted by this frontend must map
/// to an owner in this structure. Removing an entry transfers that owner to the
/// entry point performing destruction, allowing callbacks, blocking cleanup,
/// and worker joins to happen after the global mutex is released.
pub(crate) struct HsaSession {
    /// HSA agent views and their activated native attachments.
    pub(crate) agents: Agents,
    /// Immutable policy selected for this HSA session.
    pub(crate) config: HsaConfig,
    /// Public-call lifetime and distinct ordinary/cleanup cancellation.
    pub(crate) control: SessionControl,
    /// Native trap owners released after queues and before GPU devices.
    pub(crate) debugger: Option<PendingDebugger>,
    /// Process logging sink retained by work outside the registry lock.
    pub(crate) logging: Arc<LogOutput>,
    /// Next opaque HSA handle value.
    next_handle: u64,
    /// HSA queue handles and their frontend teardown dependencies.
    pub(crate) queue_state: QueueState,
    /// HSA code-object and executable handles around rocloader owners.
    pub(crate) loader: LoaderState,
    /// HSA image and sampler descriptor handles.
    pub(crate) image_state: ImageState,
    /// HSA memory handles and metadata around rocddi allocation owners.
    pub(crate) memory_state: MemoryState,
    /// HSA signal handles, backing, events, and asynchronous registrations.
    pub(crate) signal_state: SignalState,
    /// HSA PC sampling handles and native cleanup state.
    pub(crate) pc_sampling_state: PcSamplingState,
    /// HSA system callbacks, native event observation, and fault metadata.
    pub(crate) events: SystemEvents,
    /// Native session retained until all GPU and memory owners are gone.
    native_session: Option<NativeSession>,
}

impl HsaSession {
    /// Borrows the native session while this HSA session is published.
    pub(crate) fn native_session(&self) -> Option<&NativeSession> {
        self.native_session.as_ref()
    }

    /// Closes core admission after tool cleanup has finished using services.
    pub(crate) fn begin_native_retirement(&self) -> Result<(), Status> {
        self.native_session
            .as_ref()
            .ok_or(ERROR)?
            .begin_retirement()
            .map_err(map_error)
    }

    /// Checks whether the caller is one of this runtime’s callback workers.
    pub(crate) fn running_on_worker(&self) -> bool {
        self.signal_state.running_on_worker()
            || self.memory_state.running_on_worker()
            || self.events.running_on_worker()
    }

    /// Allocates handle for runtime use.
    pub(crate) fn allocate_handle(&mut self) -> Result<u64, Status> {
        let handle = self.next_handle;
        self.next_handle = self.next_handle.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
        Ok(handle)
    }

    /// Prepares log before publication or execution.
    pub(crate) fn prepare_log(&self, flag: u32, arguments: Arguments<'_>) -> Option<LogWork> {
        self.logging
            .prepare(&self.control.inflight, flag, arguments)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;

#[cfg(test)]
pub(crate) mod testing;
