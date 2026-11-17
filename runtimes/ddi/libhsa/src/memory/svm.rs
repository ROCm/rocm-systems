// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Managed reservation backing and dependency-ordered SVM discard batches.

use super::*;
use crate::memory::copy_svm::async_copy::batch_slice;
use rocddi::memory::advice::PrefetchCompletion;

/// Validated reservation range retained until its accepted operation retires.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ManagedRange {
    /// Owning reservation's base, used by asynchronous retention.
    reservation: usize,
    /// Page-aligned first byte of this operation.
    address: u64,
    /// Page-aligned byte extent of this operation.
    size: u64,
}

/// Finds a complete managed reservation without manufacturing backing.
fn managed_range(runtime: &HsaState, address: u64, size: u64) -> Result<ManagedRange, Status> {
    let address_host = usize::try_from(address).map_err(|_| INVALID_ARGUMENT)?;
    let size_host = usize::try_from(size).map_err(|_| INVALID_ARGUMENT)?;
    let (&base, reservation) = runtime
        .memory_state
        .vmem_reservations
        .range(..=address_host)
        .next_back()
        .filter(|(_, reservation)| reservation.contains_range(address_host, size_host))
        .ok_or(INVALID_ARGUMENT)?;
    if reservation.description.registered
        || runtime.memory_state.vmem_mappings.values().any(|mapping| {
            mapping.address < address_host + size_host
                && address_host < mapping.address + mapping.size
        })
    {
        return Err(INVALID_ARGUMENT);
    }
    Ok(ManagedRange {
        reservation: base,
        address,
        size,
    })
}

/// Records the union of committed ranges after storage has been reserved.
fn merge_committed(ranges: &mut Vec<(usize, usize)>, address: usize, size: usize) {
    let mut begin = address;
    let mut end = address + size;
    ranges.retain(|&(start, length)| {
        if start <= end && begin <= start + length {
            begin = begin.min(start);
            end = end.max(start + length);
            false
        } else {
            true
        }
    });
    ranges.push((begin, end - begin));
    ranges.sort_unstable_by_key(|&(start, _)| start);
}

/// Finds the committed span containing an address without claiming its holes.
pub(crate) fn committed_range(ranges: &[(usize, usize)], address: usize) -> Option<(usize, usize)> {
    ranges
        .iter()
        .copied()
        .find(|&(start, size)| address >= start && address - start < size)
}

/// Commits only runtime-owned anonymous memory, preserving its reservation.
fn commit_managed(
    runtime: &mut HsaState,
    service: &MemoryAdvice,
    range: ManagedRange,
) -> Result<(), Status> {
    let reservation = runtime
        .memory_state
        .vmem_reservations
        .get_mut(&range.reservation)
        .ok_or(INVALID_ALLOCATION)?;
    reservation
        .svm_ranges
        .try_reserve(1)
        .map_err(|_| OUT_OF_RESOURCES)?;
    // SAFETY: Managed-range validation excludes registered/native mappings.
    // The registry lock and asynchronous borrow retain the reservation and
    // exclude concurrent page-table replacement. mprotect preserves contents.
    unsafe { service.commit_host_range(range.address, range.size) }.map_err(map_error)?;
    merge_committed(
        &mut reservation.svm_ranges,
        range.address as usize,
        range.size as usize,
    );
    Ok(())
}

/// Makes a managed reservation host-backed before KFD consumes its SVM policy.
/// External memory and registered VMEM remain owned by their existing paths.
pub(crate) fn commit_registration(
    runtime: &mut HsaState,
    service: &MemoryAdvice,
    address: u64,
    size: u64,
) -> Result<(), Status> {
    let overlaps = runtime
        .memory_state
        .vmem_reservations
        .range(..address.saturating_add(size) as usize)
        .next_back()
        .is_some_and(|(_, reservation)| {
            reservation.address.info().address + reservation.address.info().size > address
        });
    if !overlaps {
        return Ok(());
    }
    if runtime
        .memory_state
        .vmem_reservations
        .range(..=address as usize)
        .next_back()
        .is_some_and(|(_, reservation)| {
            reservation.contains_range(address as usize, size as usize)
                && reservation.description.registered
        })
        || runtime.memory_state.vmem_mappings.values().any(|mapping| {
            mapping.address < (address + size) as usize
                && (address as usize) < mapping.address + mapping.size
        })
    {
        return Ok(());
    }
    let range = managed_range(runtime, address, size)?;
    if runtime.memory_state.copy_borrowed(range.reservation) {
        return Err(OUT_OF_RESOURCES);
    }
    commit_managed(runtime, service, range)
}

/// Executes discard on one retained range, stopping at the first native error.
fn discard_range(
    service: &MemoryAdvice,
    range: ManagedRange,
    destination: Option<SvmLocation>,
) -> Result<(), Status> {
    if destination.is_none() {
        service
            .prefetch(
                range.address,
                range.size,
                SvmLocation::System,
                PrefetchCompletion::Attempt,
            )
            .map_err(map_error)?;
    }
    // SAFETY: The accepted operation retains this anonymous reservation.
    // HSA dependency completion authorizes discarding prior contents, and the
    // caller must not access them concurrently. Native MMU notifiers coordinate
    // device mappings during the synchronous Linux advice operation.
    unsafe { service.discard_host_range(range.address, range.size, destination.is_some()) }
        .map_err(map_error)?;
    if let Some(destination) = destination {
        service
            .prefetch(
                range.address,
                range.size,
                destination,
                PrefetchCompletion::Attempt,
            )
            .map_err(map_error)?;
    }
    Ok(())
}

/// Validates, retains, and schedules a whole discard operation.
///
/// # Safety
/// The caller supplies the documented readable arrays and keeps their memory
/// operands valid, synchronized, and discardable until completion.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn submit_discard(
    pointers: *mut *mut c_void,
    sizes: *mut usize,
    count: u32,
    destinations: Option<(*const HsaAgent, u32)>,
    dependency_count: u32,
    dependencies: *const HsaSignal,
    completion: HsaSignal,
) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if count == 0
            || pointers.is_null()
            || sizes.is_null()
            || destinations.is_some_and(|(agents, count)| agents.is_null() || count == 0)
        {
            return INVALID_ARGUMENT;
        }
        if !runtime.gpus.first().is_some_and(|gpu| gpu.xnack_enabled) {
            return XNACK_DISABLED;
        }
        if (dependency_count == 0) != dependencies.is_null()
            || destinations.is_some_and(|(_, destination_count)| destination_count != count)
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The C ABI supplies these complete readable arrays. No caller
        // array pointer is retained by the worker.
        let pointers = match unsafe { batch_slice(pointers, count as usize) } {
            Ok(values) => values,
            Err(status) => return status,
        };
        let sizes = match unsafe { batch_slice(sizes, count as usize) } {
            Ok(values) => values,
            Err(status) => return status,
        };
        let agents = match destinations {
            Some((agents, count)) => match unsafe { batch_slice(agents, count as usize) } {
                Ok(values) => values,
                Err(status) => return status,
            },
            None => &[],
        };
        let dependencies = if dependency_count == 0 {
            &[]
        } else {
            match unsafe { batch_slice(dependencies, dependency_count as usize) } {
                Ok(values) => values,
                Err(status) => return status,
            }
        };
        if dependencies
            .iter()
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        if completion.handle != 0 {
            let status = runtime.signal_state.async_completion_status(completion);
            if status != SUCCESS {
                return status;
            }
        }
        let service = match svm_service(runtime, agents.first().copied()) {
            Ok(service) => service,
            Err(status) => return status,
        };
        let mut ranges = Vec::new();
        let mut locations = Vec::new();
        let mut devices = Vec::new();
        let mut handles = Vec::new();
        if ranges.try_reserve_exact(pointers.len()).is_err()
            || locations.try_reserve_exact(agents.len()).is_err()
            || devices.try_reserve_exact(agents.len()).is_err()
            || handles
                .try_reserve_exact(dependencies.len() + usize::from(completion.handle != 0))
                .is_err()
            || runtime.memory_state.workers.try_reserve(1).is_err()
        {
            return OUT_OF_RESOURCES;
        }
        for (&pointer, &size) in pointers.iter().zip(sizes) {
            let (address, size) = match svm_range(pointer, size) {
                Ok(range) => range,
                Err(status) => return status,
            };
            let range = match managed_range(runtime, address, size) {
                Ok(range) => range,
                Err(status) => return status,
            };
            if let Err(error) = service.validate_prefetch(
                address,
                size,
                SvmLocation::System,
                PrefetchCompletion::Attempt,
            ) {
                return map_error(error);
            }
            ranges.push(range);
        }
        for &agent in agents {
            let Some(index) = runtime.gpu_index(agent) else {
                return INVALID_AGENT;
            };
            let location = match svm_location_from_agent(runtime, &service, agent, false) {
                Ok(location) => location,
                Err(status) => return status,
            };
            locations.push(location);
            devices.push(runtime.gpus[index].device.clone());
        }
        handles.extend_from_slice(dependencies);
        if completion.handle != 0 {
            handles.push(completion);
        }
        runtime.memory_state.reap_finished_workers();
        let stop = runtime.operation_stop();
        let cleanup = match runtime.inherit_cleanup() {
            Ok(cleanup) => cleanup,
            Err(status) => return status,
        };
        let health = runtime.health.clone();
        let mut pending = match PendingAsyncRetention::new(
            &mut runtime.signal_state,
            &mut runtime.memory_state,
            Arc::new(handles),
        ) {
            Ok(pending) => pending,
            Err(status) => return status,
        };
        for range in &ranges {
            if let Err(status) = pending.retain_memory(range.reservation) {
                return status;
            }
        }
        let worker_handles = pending.handles.clone();
        let dependency_len = dependencies.len();
        let worker = match thread::Builder::new()
            .name("rocddi-svm-discard".into())
            .spawn(move || {
                let _lease = cleanup.map(crate::runtime::ApiLeaseScope::enter);
                let _devices = devices;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
                    || -> Result<(), Status> {
                        if !wait_async_dependencies(&worker_handles[..dependency_len], &stop) {
                            return Err(ERROR);
                        }
                        for (index, &range) in ranges.iter().enumerate() {
                            if stop.load(Ordering::Acquire) {
                                return Err(ERROR);
                            }
                            {
                                let mut guard = lock()?;
                                let runtime = initialized_mut(&mut guard)?;
                                commit_managed(runtime, &service, range)?;
                            }
                            discard_range(&service, range, locations.get(index).copied())?;
                        }
                        Ok(())
                    },
                ));
                let retain = result.is_err();
                if retain {
                    health.quarantine();
                }
                for range in &ranges {
                    release_async_resources(&[], &[Some(range.reservation)], retain);
                }
                finish_async_signal(completion, matches!(result, Ok(Ok(()))));
                release_async_resources(&worker_handles, &[], false);
            }) {
            Ok(worker) => worker,
            Err(_) => return OUT_OF_RESOURCES,
        };
        pending.transfer_to_worker();
        pending.memory.workers.push(worker);
        SUCCESS
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_ranges_merge_overlaps_without_claiming_holes() {
        let mut ranges = vec![(0x1000, 0x1000), (0x4000, 0x1000)];
        merge_committed(&mut ranges, 0x1800, 0x1800);
        assert_eq!(ranges, [(0x1000, 0x2000), (0x4000, 0x1000)]);
        assert_eq!(committed_range(&ranges, 0x1000), Some((0x1000, 0x2000)));
        assert_eq!(committed_range(&ranges, 0x2fff), Some((0x1000, 0x2000)));
        assert_eq!(committed_range(&ranges, 0x3000), None);
        assert_eq!(committed_range(&ranges, 0x3fff), None);
        assert_eq!(committed_range(&ranges, 0x5000), None);
        merge_committed(&mut ranges, 0x3000, 0x1000);
        assert_eq!(ranges, [(0x1000, 0x4000)]);
    }
}
