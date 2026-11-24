// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Bounded, read-only evidence capture at a failed native queue update.

use super::{DeviceEvent, DeviceVm, LossEvent};
use std::fmt;
use std::io::{self, Write};
use std::sync::TryLockError;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// Capture failures are diagnostic evidence, separate from the queue error.
#[derive(Debug)]
enum SnapshotUnavailable {
    Process(io::Error),
    Busy,
    Poisoned,
}

impl fmt::Display for SnapshotUnavailable {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Process(source) => write!(output, "process_check_failed({source})"),
            Self::Busy => output.write_str("observation_lock_busy"),
            Self::Poisoned => output.write_str("observation_lock_poisoned"),
        }
    }
}

/// An event ID is usable only while its exact owner confirms its lifetime.
#[derive(Debug)]
enum PayloadUnavailable {
    Closed,
    DestructionUncertain,
    Read(io::Error),
}

impl fmt::Display for PayloadUnavailable {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => output.write_str("event_closed"),
            Self::DestructionUncertain => output.write_str("event_destruction_uncertain"),
            Self::Read(source) => write!(output, "read_failed({source})"),
        }
    }
}

/// Connection-wide history and independent pending native payloads.
#[derive(Debug)]
struct ExceptionSnapshot {
    /// Existing terminal-loss latch, without a health poll.
    lost_latched: bool,
    /// Latest retained memory record's sequence, if present; no decoded payload.
    retained_memory_sequence: Option<u64>,
    /// Latest retained hardware record's sequence, if present; no decoded payload.
    retained_hardware_sequence: Option<u64>,
    /// Raw native memory words if pending; independent of retained history.
    native_pending_memory_words: Result<Option<[u64; 4]>, PayloadUnavailable>,
    /// Raw native hardware words if pending; independent of the memory read.
    native_pending_hardware_words: Result<Option<[u64; 4]>, PayloadUnavailable>,
}

impl LossEvent {
    fn passive_payload(
        &self,
        event_id: &AtomicU32,
        uncertain: &AtomicBool,
    ) -> Result<Option<[u64; 4]>, PayloadUnavailable> {
        if uncertain.load(Ordering::Acquire) {
            return Err(PayloadUnavailable::DestructionUncertain);
        }
        let id = event_id.load(Ordering::Acquire);
        if id == 0 {
            return Err(PayloadUnavailable::Closed);
        }
        self.kfd
            .exception_payload(id)
            .map_err(PayloadUnavailable::Read)
    }

    fn passive_snapshot(&self) -> Result<ExceptionSnapshot, SnapshotUnavailable> {
        self.kfd
            .check_process()
            .map_err(SnapshotUnavailable::Process)?;
        let records = match self.observed.try_lock() {
            Ok(records) => records,
            Err(TryLockError::WouldBlock) => return Err(SnapshotUnavailable::Busy),
            Err(TryLockError::Poisoned(_)) => return Err(SnapshotUnavailable::Poisoned),
        };
        // Keep the exact owner and observation lock through both timeout-zero
        // reads: normal observers can reset these manual-reset events. KFD may
        // still replace either payload, so the channels are independent samples.
        Ok(ExceptionSnapshot {
            lost_latched: self.lost.load(Ordering::Acquire),
            retained_memory_sequence: records
                .events
                .iter()
                .rev()
                .find(|record| matches!(record.event, DeviceEvent::GpuMemoryFault(_)))
                .map(|record| record.sequence),
            retained_hardware_sequence: records
                .events
                .iter()
                .rev()
                .find(|record| matches!(record.event, DeviceEvent::GpuHardwareException(_)))
                .map(|record| record.sequence),
            native_pending_memory_words: self
                .passive_payload(&self.memory_event_id, &self.memory_destroy_uncertain),
            native_pending_hardware_words: self
                .passive_payload(&self.hardware_event_id, &self.hardware_destroy_uncertain),
        })
    }
}

impl DeviceVm {
    pub(in super::super) fn diagnose_queue_update_failure(
        &self,
        queue_id: u32,
        source: &io::Error,
    ) {
        // Cleanup remains available after admission closes or loss is latched.
        // Reject inherited state before environment access, formatting or locks.
        if self.kfd().check_process().is_err()
            || std::env::var_os("ROCDDI_CODE_CACHE_DIAGNOSTICS").is_none_or(|value| value != "1")
        {
            return;
        }
        let snapshot = self.loss.passive_snapshot();
        // passive_snapshot has released the observation lock before stderr I/O.
        let _ = write_queue_update_diagnostic(
            &mut io::stderr().lock(),
            queue_id,
            self.gpu_id,
            source,
            &snapshot,
        );
    }
}

fn write_queue_update_diagnostic(
    output: &mut impl Write,
    queue_id: u32,
    expected_gpu_id: u32,
    source: &io::Error,
    snapshot: &Result<ExceptionSnapshot, SnapshotUnavailable>,
) -> io::Result<()> {
    write!(
        output,
        "rocddi: boundary=UPDATE_QUEUE-error queue_id={queue_id} \
         expected_gpu_id={expected_gpu_id} update_errno={:?} update_error={source} \
         connection_exception_snapshot=",
        source.raw_os_error(),
    )?;
    match snapshot {
        Err(reason) => write!(output, "unavailable({reason})")?,
        Ok(snapshot) => {
            write!(
                output,
                "lost_latched={} retained_memory_sequence={:?} \
                 retained_hardware_sequence={:?} native_pending_memory_words=",
                snapshot.lost_latched,
                snapshot.retained_memory_sequence,
                snapshot.retained_hardware_sequence,
            )?;
            write_payload(output, &snapshot.native_pending_memory_words)?;
            write!(output, " native_pending_hardware_words=")?;
            write_payload(output, &snapshot.native_pending_hardware_words)?;
        }
    }
    writeln!(output)
}

fn write_payload(
    output: &mut impl Write,
    payload: &Result<Option<[u64; 4]>, PayloadUnavailable>,
) -> io::Result<()> {
    match payload {
        Ok(Some(words)) => write!(output, "{words:?}"),
        Ok(None) => write!(output, "empty"),
        Err(reason) => write!(output, "unavailable({reason})"),
    }
}

#[cfg(test)]
#[path = "../tests/passive_snapshot.rs"]
mod tests;
