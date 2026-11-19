// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Queue lifecycle operations shared by C adapters and internal HSA dispatch.

use std::ptr::NonNull;

use super::*;

/// Producer policy of an ordinary HSA AQL queue.
#[derive(Clone, Copy)]
pub(crate) enum AqlQueueKind {
    /// Several host producers reserve packet slots atomically.
    Multi,
    /// One producer publishes packets.
    Single,
    /// Acquisitions share the agent's cooperative queue.
    Cooperative,
}

impl AqlQueueKind {
    /// Validates the public producer-policy value at the C boundary.
    pub(crate) fn from_abi(value: u32) -> Option<Self> {
        match value {
            QUEUE_TYPE_MULTI => Some(Self::Multi),
            QUEUE_TYPE_SINGLE => Some(Self::Single),
            QUEUE_TYPE_COOPERATIVE => Some(Self::Cooperative),
            _ => None,
        }
    }

    /// Supplies the queue header and native creation path's producer policy.
    fn abi(self) -> u32 {
        match self {
            Self::Multi => QUEUE_TYPE_MULTI,
            Self::Single => QUEUE_TYPE_SINGLE,
            Self::Cooperative => QUEUE_TYPE_COOPERATIVE,
        }
    }
}

/// Input to ordinary AQL creation without foreign output pointers.
#[derive(Clone, Copy)]
pub(crate) struct AqlQueueRequest {
    /// GPU whose registry owns the resulting queue.
    pub(crate) agent: HsaAgent,
    /// Requested power-of-two packet capacity.
    pub(crate) size: u32,
    /// Validated producer policy.
    pub(crate) kind: AqlQueueKind,
    /// Optional asynchronous error callback.
    pub(crate) callback: QueueErrorCallback,
    /// Caller-owned callback storage retained through native retirement.
    pub(crate) data: CallbackArg,
    /// Initial private segment requirement used to plan scratch.
    pub(crate) private_segment_size: u32,
}

/// Creates and publishes an ordinary queue in the admitted HSA generation.
///
/// # Safety
/// Callback code and data must remain live until conclusive queue retirement
/// and completion of the last callback, including after an uncertain failure.
pub(crate) unsafe fn create(request: AqlQueueRequest) -> Result<NonNull<HsaQueue>, Status> {
    if request.size == 0 || !request.size.is_power_of_two() {
        return Err(INVALID_ARGUMENT);
    }
    let mut queue = std::ptr::null_mut();
    let mut created_log = None;
    // SAFETY: The local output is writable and the caller retains callback
    // storage. The creation transaction retains all native queue dependencies.
    let status = unsafe {
        create_hardware_queue(
            None,
            false,
            request.agent,
            request.size,
            request.kind.abi(),
            QueuePriority::Normal,
            request.callback,
            request.data,
            request.private_segment_size,
            QueueRingMemory::System,
            QueueRingMemory::System,
            None,
            &raw mut queue,
            &mut created_log,
        )
    };
    let log = created_log.and_then(|(id, public)| {
        lock().ok().and_then(|guard| {
            guard.as_ref().and_then(|runtime| {
                runtime.prepare_log(
                    AMD_LOG_FLAG_INFO,
                    format_args!(
                        "created AQL queue id={id} agent=0x{:x} address=0x{public:x} packets={}",
                        request.agent.handle,
                        request.size.max(64)
                    ),
                )
            })
        })
    });
    if let Some(log) = log {
        log.write();
    }
    if status != SUCCESS {
        return Err(status);
    }
    NonNull::new(queue).ok_or(ERROR)
}

/// Removes a queue from public lookup while its native owner is retired.
///
/// The caller is admitted to the HSA generation. Failed native retirement
/// restores the owner for retry, or retains it and quarantines the generation
/// if its original registry can no longer accept the owner.
pub(crate) fn destroy(queue: NonNull<HsaQueue>) -> Result<(), Status> {
    #[allow(
        clippy::large_enum_variant,
        reason = "queue teardown transfers owners on the stack without allocating"
    )]
    enum RemovedQueue {
        Aql(Queue),
        Sdma(SdmaQueue),
    }

    if CallbackScope::active() {
        return Err(INVALID_RUNTIME_STATE);
    }
    let key = queue.as_ptr() as usize;
    let intercept = {
        let guard = lock()?;
        let runtime = guard.as_ref().ok_or(NOT_INITIALIZED)?;
        runtime
            .queue_state
            .intercept_queues
            .get(&key)
            .cloned()
            .map(|owner| runtime.control.inflight.enter().map(|token| (owner, token)))
    };
    if let Some(intercept) = intercept {
        let (intercept, _in_flight) = intercept.ok_or(INVALID_RUNTIME_STATE)?;
        return match crate::queue_intercept::destroy(&intercept) {
            SUCCESS => Ok(()),
            status => Err(status),
        };
    }
    let (mut record, cooperative_agent, in_flight) = {
        let mut guard = lock()?;
        let runtime = initialized_mut(&mut guard)?;
        let cooperative_agent = if let Some(record) = runtime.queue_state.queues.get_mut(&key) {
            if record.public_handle != key || record.counted_pool_key.is_some() {
                return Err(INVALID_QUEUE);
            }
            if record.cooperative_refs > 1 {
                record.cooperative_refs -= 1;
                return Ok(());
            }
            (record.cooperative_refs == 1).then_some(record.agent.handle)
        } else {
            None
        };
        let in_flight = guard.retain_session()?;
        let runtime = initialized_mut(&mut guard)?;
        if let Some(agent) = cooperative_agent {
            runtime
                .queue_state
                .cooperative_teardown
                .try_reserve(1)
                .map_err(|_| OUT_OF_RESOURCES)?;
            runtime.queue_state.cooperative_teardown.insert(agent);
        }
        let record = if let Some(record) = runtime.queue_state.queues.remove(&key) {
            RemovedQueue::Aql(record)
        } else if let Some(record) = runtime.queue_state.sdma_queues.remove(&key) {
            RemovedQueue::Sdma(record)
        } else {
            let soft = runtime
                .queue_state
                .soft_queues
                .remove(&key)
                .ok_or(INVALID_QUEUE)?;
            runtime
                .signal_state
                .release_async_signal(soft.doorbell_signal);
            return Ok(());
        };
        (record, cooperative_agent, in_flight)
    };
    let result = match &mut record {
        RemovedQueue::Aql(queue) => destroy_runtime_queue(queue),
        RemovedQueue::Sdma(queue) => destroy_runtime_sdma_queue(queue),
    };
    match result {
        Ok(()) => {
            let engine = match record {
                RemovedQueue::Aql(_) => "AQL",
                RemovedQueue::Sdma(_) => "SDMA",
            };
            let log = lock().ok().and_then(|mut guard| {
                in_flight.is_current(&guard).then_some(())?;
                guard.as_mut().and_then(|runtime| {
                    if let Some(agent) = cooperative_agent {
                        runtime.queue_state.cooperative_teardown.remove(&agent);
                    }
                    runtime.prepare_log(
                        AMD_LOG_FLAG_INFO,
                        format_args!("destroyed {engine} queue address=0x{key:x}"),
                    )
                })
            });
            if let Some(log) = log {
                log.write();
            }
            Ok(())
        }
        Err(error) => {
            let status = map_error(error);
            let Ok(mut guard) = lock() else {
                in_flight.quarantine();
                std::mem::forget(record);
                return Err(status);
            };
            if !in_flight.is_current(&guard) {
                in_flight.quarantine();
                std::mem::forget(record);
                return Err(status);
            }
            let Some(runtime) = guard.as_mut() else {
                in_flight.quarantine();
                std::mem::forget(record);
                return Err(status);
            };
            // Native teardown may partially release an ID. Never overwrite a
            // queue created under the same key while native work was unlocked.
            if runtime.queue_state.queues.contains_key(&key)
                || runtime.queue_state.sdma_queues.contains_key(&key)
            {
                in_flight.quarantine();
                std::mem::forget(record);
                return Err(INVALID_RUNTIME_STATE);
            }
            if let Some(agent) = cooperative_agent {
                runtime.queue_state.cooperative_teardown.remove(&agent);
            }
            match record {
                RemovedQueue::Aql(record) => {
                    runtime.queue_state.queues.insert(key, record);
                }
                RemovedQueue::Sdma(record) => {
                    runtime.queue_state.sdma_queues.insert(key, record);
                }
            }
            Err(status)
        }
    }
}
