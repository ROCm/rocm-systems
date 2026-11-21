// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Pending HSA prefetch destinations layered over completed native locations.

use crate::ffi::{HsaAgent, INVALID_ARGUMENT, OUT_OF_RESOURCES, Status};
use crate::memory::copy_svm::svm_agent_from_location;
use crate::session::HsaSession;
use rocddi::memory::advice::AdviceLocation;

/// Reserves a pending range using the same agent representation as native queries.
pub(crate) fn prepare(
    runtime: &mut HsaSession,
    address: u64,
    size: u64,
    location: AdviceLocation,
) -> Result<PreparedPrefetch, Status> {
    // Native system memory has one location even when HSA exposes several CPUs.
    let agent = svm_agent_from_location(runtime, location)?;
    runtime
        .memory_state
        .pending_prefetches
        .prepare(address, size, agent)
}

/// A disjoint portion of the most recently issued prefetch covering its bytes.
#[derive(Clone, Copy)]
struct PendingRange {
    /// First covered byte.
    start: u64,
    /// First byte after the range.
    end: u64,
    /// Worker whose retirement removes this range.
    operation: u64,
    /// Canonical destination, visible before native migration starts.
    agent: HsaAgent,
}

/// Reserved bookkeeping that does not become visible until worker acceptance.
pub(crate) struct PreparedPrefetch {
    /// Validated range and unique worker identity.
    range: PendingRange,
}

impl PreparedPrefetch {
    /// Identifies the worker independently of later overlapping submissions.
    pub(crate) fn operation(&self) -> u64 {
        self.range.operation
    }
}

/// Disjoint pending ranges, ordered by start, under the HSA registry lock.
#[derive(Default)]
pub(crate) struct PendingPrefetches {
    /// Only the latest issuer remains visible at each address.
    ranges: Vec<PendingRange>,
    /// Monotonic operation identity, never reused while a worker can exist.
    next_operation: u64,
}

impl PendingPrefetches {
    /// Reserves all publication storage without replacing earlier destinations.
    fn prepare(
        &mut self,
        address: u64,
        size: u64,
        agent: HsaAgent,
    ) -> Result<PreparedPrefetch, Status> {
        let end = address.checked_add(size).ok_or(INVALID_ARGUMENT)?;
        if size == 0 || agent.handle == 0 {
            return Err(INVALID_ARGUMENT);
        }
        let operation = self.next_operation.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
        // One insertion can split one old range in two and add the new range.
        self.ranges.try_reserve(2).map_err(|_| OUT_OF_RESOURCES)?;
        self.next_operation = operation;
        Ok(PreparedPrefetch {
            range: PendingRange {
                start: address,
                end,
                operation,
                agent,
            },
        })
    }

    /// Publishes after worker acceptance, under the same lock as preparation.
    /// Superseded portions are removed permanently, so they cannot reappear
    /// after a newer worker completes before an older one.
    #[expect(
        clippy::needless_pass_by_value,
        reason = "publication consumes the preparation so an operation cannot be published twice"
    )]
    pub(crate) fn publish(&mut self, prepared: PreparedPrefetch) {
        let new = prepared.range;
        let mut tail = None;
        self.ranges.retain_mut(|old| {
            if old.end <= new.start || old.start >= new.end {
                return true;
            }
            if old.start < new.start {
                if old.end > new.end {
                    tail = Some(PendingRange {
                        start: new.end,
                        ..*old
                    });
                }
                old.end = new.start;
                true
            } else if old.end > new.end {
                old.start = new.end;
                true
            } else {
                false
            }
        });
        if let Some(tail) = tail {
            self.ranges.push(tail);
        }
        self.ranges.push(new);
        self.ranges.sort_unstable_by_key(|range| range.start);
    }

    /// Removes only the completing worker's portions that were not superseded.
    pub(crate) fn retire(&mut self, operation: u64) {
        self.ranges.retain(|range| range.operation != operation);
    }

    /// Combines pending destinations with native queries for uncovered holes.
    pub(crate) fn query(
        &self,
        address: u64,
        size: u64,
        mut native: impl FnMut(u64, u64) -> Result<HsaAgent, Status>,
    ) -> Result<HsaAgent, Status> {
        let end = address.checked_add(size).ok_or(INVALID_ARGUMENT)?;
        if size == 0 {
            return Err(INVALID_ARGUMENT);
        }
        let mut cursor = address;
        let mut agent = None;
        for range in &self.ranges {
            if range.end <= cursor {
                continue;
            }
            if range.start >= end {
                break;
            }
            if cursor < range.start
                && !merge_agent(&mut agent, native(cursor, range.start - cursor)?)
            {
                return Ok(HsaAgent::default());
            }
            if !merge_agent(&mut agent, range.agent) {
                return Ok(HsaAgent::default());
            }
            cursor = range.end.min(end);
        }
        if cursor < end && !merge_agent(&mut agent, native(cursor, end - cursor)?) {
            return Ok(HsaAgent::default());
        }
        Ok(agent.unwrap_or_default())
    }
}

/// Reduces locations to one non-null agent, rejecting unknown or mixed spans.
fn merge_agent(uniform: &mut Option<HsaAgent>, agent: HsaAgent) -> bool {
    if agent.handle == 0 {
        return false;
    }
    if let Some(previous) = uniform {
        *previous == agent
    } else {
        *uniform = Some(agent);
        true
    }
}

/// Retires query visibility before completion signaling and memory release.
pub(crate) fn retire(operation: u64) {
    if let Ok(mut guard) = crate::lifecycle::lock() {
        if let Some(runtime) = guard.as_mut() {
            runtime.memory_state.pending_prefetches.retire(operation);
        }
    }
    // Shutdown makes the registry inaccessible, joins workers, and then clears
    // its pending ranges together with the retained memory owners.
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::ffi::ERROR;

    /// Publishes one accepted request and returns its worker identity.
    fn issue(state: &mut PendingPrefetches, start: u64, size: u64, agent: u64) -> u64 {
        let prepared = state
            .prepare(start, size, HsaAgent { handle: agent })
            .unwrap();
        let operation = prepared.operation();
        state.publish(prepared);
        operation
    }

    /// Reads a fully pending range; consulting native state would fail the test.
    fn pending_agent(state: &PendingPrefetches, start: u64, size: u64) -> u64 {
        state.query(start, size, |_, _| Err(ERROR)).unwrap().handle
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn cpu_destinations_remain_uniform_across_pending_and_completed_ranges() {
        use crate::agent::Cpu;
        use crate::ffi::CPU_AGENT;
        use crate::memory::copy_svm::svm_location_from_agent;
        use crate::session::testing::session_without_devices;
        use rocddi::session::{DriverContextLifetime, Session};

        let session = Session::new(DriverContextLifetime::Session).unwrap();
        // Selecting the provider requires no discovery or device acquisition.
        let service = session.memory_advice().unwrap();
        let mut runtime = session_without_devices(session);
        runtime.agents.cpus = (0..2)
            .map(|native_node| Cpu {
                node: None,
                native_node,
                name: "CPU".into(),
                compute_units: 1,
                maximum_clock_mhz: 0,
                memory_bytes: 0,
                apu: false,
            })
            .collect();

        for handle in [CPU_AGENT, CPU_AGENT + 1] {
            let location =
                svm_location_from_agent(&runtime, &service, HsaAgent { handle }, false).unwrap();
            assert_eq!(location, AdviceLocation::System);
            let prepared = prepare(&mut runtime, 0x2000, 0x1000, location).unwrap();
            let operation = prepared.operation();
            runtime.memory_state.pending_prefetches.publish(prepared);
            assert_eq!(
                pending_agent(&runtime.memory_state.pending_prefetches, 0x2000, 0x1000),
                CPU_AGENT
            );
            for completed in [false, true] {
                if completed {
                    runtime.memory_state.pending_prefetches.retire(operation);
                }
                for (address, size) in [(0x2000, 0x1000), (0x1000, 0x3000)] {
                    let agent =
                        runtime
                            .memory_state
                            .pending_prefetches
                            .query(address, size, |_, _| {
                                svm_agent_from_location(&runtime, AdviceLocation::System)
                            });
                    assert_eq!(agent.unwrap().handle, CPU_AGENT);
                }
            }
        }
    }

    #[test]
    fn publication_requires_acceptance_and_replaces_only_overlapping_bytes() {
        let mut state = PendingPrefetches::default();
        issue(&mut state, 0x1000, 0x4000, 1);
        {
            let _rejected = state
                .prepare(0x2000, 0x1000, HsaAgent { handle: 2 })
                .unwrap();
        }
        assert_eq!(pending_agent(&state, 0x1000, 0x4000), 1);

        issue(&mut state, 0x2000, 0x1000, 2);
        assert_eq!(pending_agent(&state, 0x1000, 0x1000), 1);
        assert_eq!(pending_agent(&state, 0x2000, 0x1000), 2);
        assert_eq!(pending_agent(&state, 0x3000, 0x2000), 1);
        assert_eq!(pending_agent(&state, 0x1000, 0x4000), 0);
    }

    #[test]
    fn query_reads_only_native_holes_and_reduces_unknown_or_mixed_locations() {
        let mut state = PendingPrefetches::default();
        issue(&mut state, 0x2000, 0x1000, 1);
        issue(&mut state, 0x4000, 0x1000, 1);
        let mut holes = Vec::new();
        let result = state.query(0x1000, 0x5000, |start, size| {
            holes.push((start, size));
            Ok(HsaAgent { handle: 1 })
        });
        assert_eq!(result.unwrap().handle, 1);
        assert_eq!(
            holes,
            [(0x1000, 0x1000), (0x3000, 0x1000), (0x5000, 0x1000)]
        );
        for native_agent in [0, 2] {
            assert_eq!(
                state
                    .query(0x2000, 0x2000, |_, _| Ok(HsaAgent {
                        handle: native_agent
                    }))
                    .unwrap()
                    .handle,
                0
            );
        }
        assert_eq!(
            state.query(0x1000, 0x2000, |_, _| Err(ERROR)).err(),
            Some(ERROR)
        );
        assert_eq!(
            PendingPrefetches::default()
                .query(0x1000, 0x1000, |_, _| Ok(HsaAgent { handle: 2 }))
                .unwrap()
                .handle,
            2
        );
    }

    #[test]
    fn retirement_never_restores_superseded_destinations() {
        for newer_finishes_first in [false, true] {
            let mut state = PendingPrefetches::default();
            let older = issue(&mut state, 0x1000, 0x4000, 1);
            let newer = issue(&mut state, 0x2000, 0x1000, 2);
            if newer_finishes_first {
                state.retire(newer);
                assert_eq!(pending_agent(&state, 0x1000, 0x1000), 1);
                assert_eq!(pending_agent(&state, 0x3000, 0x2000), 1);
                assert_eq!(
                    state
                        .query(0x2000, 0x1000, |_, _| Ok(HsaAgent { handle: 2 }))
                        .unwrap()
                        .handle,
                    2
                );
                state.retire(older);
            } else {
                state.retire(older);
                assert_eq!(pending_agent(&state, 0x2000, 0x1000), 2);
                state.retire(newer);
            }
            let mut holes = Vec::new();
            state
                .query(0x1000, 0x4000, |start, size| {
                    holes.push((start, size));
                    Ok(HsaAgent { handle: 1 })
                })
                .unwrap();
            assert_eq!(holes, [(0x1000, 0x4000)]);
        }
    }

    #[test]
    fn one_request_replaces_multiple_ranges_without_claiming_their_tails() {
        let mut state = PendingPrefetches::default();
        let first = issue(&mut state, 0x1000, 0x2000, 1);
        let second = issue(&mut state, 0x4000, 0x2000, 2);
        let latest = issue(&mut state, 0x2000, 0x3000, 3);
        assert_eq!(pending_agent(&state, 0x1000, 0x1000), 1);
        assert_eq!(pending_agent(&state, 0x2000, 0x3000), 3);
        assert_eq!(pending_agent(&state, 0x5000, 0x1000), 2);
        state.retire(first);
        state.retire(second);
        assert_eq!(pending_agent(&state, 0x2000, 0x3000), 3);
        state.retire(latest);
        assert!(state.ranges.is_empty());
    }

    #[test]
    fn invalid_or_exhausted_preparation_preserves_existing_requests() {
        let mut state = PendingPrefetches::default();
        issue(&mut state, 0x1000, 0x1000, 1);
        for (start, size, agent) in [(0x1000, 0, 1), (u64::MAX, 1, 1), (0x1000, 0x1000, 0)] {
            assert_eq!(
                state.prepare(start, size, HsaAgent { handle: agent }).err(),
                Some(INVALID_ARGUMENT)
            );
        }
        state.next_operation = u64::MAX;
        assert_eq!(
            state.prepare(0x1000, 0x1000, HsaAgent { handle: 2 }).err(),
            Some(OUT_OF_RESOURCES)
        );
        assert_eq!(pending_agent(&state, 0x1000, 0x1000), 1);
    }
}
