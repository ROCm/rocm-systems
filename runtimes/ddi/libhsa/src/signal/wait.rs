// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Signal waits, groups, retained wait registrations, and timeout clocks.

use super::*;

/// Time spent spinning before a blocked wait yields the CPU.
const BLOCKED_SPIN_BUDGET: Duration = Duration::from_micros(200);
/// Cached system clock frequency in hertz for timeout conversion.
static SYSTEM_FREQUENCY_HZ: AtomicU64 = AtomicU64::new(0);

/// Caches the frequency reported by native system clock discovery.
pub(crate) fn set_system_frequency(frequency: u64) {
    SYSTEM_FREQUENCY_HZ.store(frequency, Ordering::Release);
}

/// Returns the system-clock frequency used for timeout conversion.
fn system_frequency() -> u64 {
    let cached = SYSTEM_FREQUENCY_HZ.load(Ordering::Acquire);
    if cached != 0 {
        return cached;
    }
    let retained = lock().ok().and_then(|guard| {
        let generation = guard.active_generation()?;
        let runtime = guard.as_ref()?;
        let gpu = runtime.gpus.first()?;
        let token = runtime.inflight.enter()?;
        Some((gpu.device.clone(), generation, token))
    });
    let Some((device, generation, _call)) = retained else {
        return 0;
    };
    let frequency = crate::runtime::clock_counters(&device)
        .ok()
        .map_or(0, |counters| counters.system_frequency);
    if frequency == 0 {
        return 0;
    }
    // Shutdown can clear the cache while the ioctl runs. Publish only while
    // the runtime generation that started the query is still active.
    let Ok(guard) = lock() else {
        return 0;
    };
    if guard.active_generation() == Some(generation) {
        SYSTEM_FREQUENCY_HZ.store(frequency, Ordering::Release);
        frequency
    } else {
        0
    }
}

/// Checks whether an HSA timeout hint has elapsed.
pub(super) fn timeout_elapsed(elapsed: Duration, timeout_hint: u64, frequency: u64) -> bool {
    timeout_hint != u64::MAX
        && (frequency == 0
            || elapsed.as_nanos().saturating_mul(u128::from(frequency))
                >= u128::from(timeout_hint) * 1_000_000_000)
}

/// Waits for one signal condition with the requested memory ordering.
unsafe fn signal_wait(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    order: Ordering,
) -> SignalValue {
    // SAFETY: Callers must supply a live HSA user signal.
    unsafe {
        with_signal(signal, |signal| {
            // The common satisfied path needs only one atomic load. Read the clock
            // and system frequency only once the caller actually has to wait.
            let observed = signal.value.load(order);
            if condition_met(condition, observed, compare_value) {
                return observed;
            }
            let start = Instant::now();
            let frequency = system_frequency();
            loop {
                let observed = signal.value.load(order);
                if condition_met(condition, observed, compare_value) {
                    return observed;
                }
                if timeout_hint != u64::MAX
                    && timeout_elapsed(start.elapsed(), timeout_hint, frequency)
                {
                    return observed;
                }
                wait_pause(wait_state_hint, &start);
            }
        })
    }
    .unwrap_or(0)
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_wait_scacquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Waits for one signal with the HSA SC-acquire completion contract.
pub unsafe extern "C" fn hsa_signal_wait_scacquire(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: The public entry point preserves the HSA signal contract.
    unsafe {
        signal_wait(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
            Ordering::Acquire,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_wait_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Waits for one signal without acquire synchronization.
pub unsafe extern "C" fn hsa_signal_wait_relaxed(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: The public entry point preserves the HSA signal contract.
    unsafe {
        signal_wait(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
            Ordering::Relaxed,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_wait_acquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Uses the deprecated acquire spelling for a single-signal wait.
pub unsafe extern "C" fn hsa_signal_wait_acquire(
    signal: HsaSignal,
    condition: u32,
    compare_value: SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
) -> SignalValue {
    // SAFETY: This deprecated entry point has the same contract as the
    // sequentially-consistent acquire spelling.
    unsafe {
        hsa_signal_wait_scacquire(
            signal,
            condition,
            compare_value,
            timeout_hint,
            wait_state_hint,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_group_create")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Creates a group that retains the listed signals for waits.
pub unsafe extern "C" fn hsa_signal_group_create(
    signal_count: u32,
    signals: *const HsaSignal,
    consumer_count: u32,
    consumers: *const HsaAgent,
    group: *mut HsaSignalGroup,
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
        if signal_count == 0
            || signals.is_null()
            || group.is_null()
            || (consumer_count != 0 && consumers.is_null())
        {
            return INVALID_ARGUMENT;
        }
        // SAFETY: The caller supplies signal_count readable handles.
        let signals = unsafe { std::slice::from_raw_parts(signals, signal_count as usize) };
        if signals
            .iter()
            .any(|signal| !runtime.signal_state.owns_signal(*signal))
        {
            return INVALID_SIGNAL;
        }
        if consumer_count != 0 {
            // SAFETY: The caller supplies consumer_count readable handles.
            let consumers =
                unsafe { std::slice::from_raw_parts(consumers, consumer_count as usize) };
            if consumers.iter().any(|agent| !runtime.is_agent(*agent)) {
                return INVALID_AGENT;
            }
        }
        let handle = match runtime.allocate_handle() {
            Ok(handle) => handle,
            Err(status) => return status,
        };
        let mut stored = Vec::new();
        if stored.try_reserve_exact(signals.len()).is_err() {
            return OUT_OF_RESOURCES;
        }
        stored.extend_from_slice(signals);
        runtime.signal_state.signal_groups.insert(handle, stored);
        // SAFETY: The caller supplied writable output storage.
        unsafe { group.write(HsaSignalGroup { handle }) };
        SUCCESS
    })
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_group_destroy")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Releases a signal group and its retained signal references.
pub extern "C" fn hsa_signal_group_destroy(group: HsaSignalGroup) -> Status {
    boundary(|| {
        let mut guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let runtime = match initialized_mut(&mut guard) {
            Ok(runtime) => runtime,
            Err(status) => return status,
        };
        if runtime
            .signal_state
            .signal_groups
            .remove(&group.handle)
            .is_some()
        {
            SUCCESS
        } else {
            INVALID_SIGNAL_GROUP
        }
    })
}

/// Waits until any signal in a group satisfies its condition.
unsafe fn signal_group_wait_any(
    group: HsaSignalGroup,
    conditions: *const u32,
    compare_values: *const SignalValue,
    wait_state_hint: u32,
    signal: *mut HsaSignal,
    value: *mut SignalValue,
) -> Status {
    let signals = {
        let guard = match lock() {
            Ok(guard) => guard,
            Err(status) => return status,
        };
        let Some(runtime) = guard.as_ref() else {
            return NOT_INITIALIZED;
        };
        if conditions.is_null() || compare_values.is_null() || signal.is_null() || value.is_null() {
            return INVALID_ARGUMENT;
        }
        let Some(signals) = runtime.signal_state.signal_groups.get(&group.handle) else {
            return INVALID_SIGNAL_GROUP;
        };
        signals.clone()
    };
    // SAFETY: The group retains signals.len() handles and the caller supplies
    // matching condition/value arrays and writable outputs.
    let index = unsafe {
        hsa_amd_signal_wait_any(
            signals.len() as u32,
            signals.as_ptr(),
            conditions,
            compare_values,
            u64::MAX,
            wait_state_hint,
            value,
        )
    };
    let Some(satisfied) = signals.get(index as usize) else {
        return INVALID_ARGUMENT;
    };
    // SAFETY: The caller supplied writable output storage.
    unsafe { signal.write(*satisfied) };
    SUCCESS
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_group_wait_any_relaxed")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Returns when any group member satisfies its condition, without acquire.
pub unsafe extern "C" fn hsa_signal_group_wait_any_relaxed(
    group: HsaSignalGroup,
    conditions: *const u32,
    compare_values: *const SignalValue,
    wait_state_hint: u32,
    signal: *mut HsaSignal,
    value: *mut SignalValue,
) -> Status {
    // SAFETY: The public entry point forwards the complete signal-group contract.
    unsafe {
        signal_group_wait_any(
            group,
            conditions,
            compare_values,
            wait_state_hint,
            signal,
            value,
        )
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_signal_group_wait_any_scacquire")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Returns when any group member satisfies its SC-acquire condition.
pub unsafe extern "C" fn hsa_signal_group_wait_any_scacquire(
    group: HsaSignalGroup,
    conditions: *const u32,
    compare_values: *const SignalValue,
    wait_state_hint: u32,
    signal: *mut HsaSignal,
    value: *mut SignalValue,
) -> Status {
    // SAFETY: The public entry point forwards the complete signal-group contract.
    let status = unsafe {
        signal_group_wait_any(
            group,
            conditions,
            compare_values,
            wait_state_hint,
            signal,
            value,
        )
    };
    if status == SUCCESS {
        fence(Ordering::Acquire);
    }
    status
}

/// Owned copies of caller arrays used by a multi-signal wait.
pub(super) struct WaitInputs {
    /// Signal handles to observe.
    pub(super) signals: Vec<HsaSignal>,
    /// Comparison operation for each signal.
    pub(super) conditions: Vec<u32>,
    /// Comparison value for each signal.
    pub(super) values: Vec<SignalValue>,
}

/// Copies wait input after validating its source and destination.
unsafe fn copy_wait_input<T: Copy>(pointer: *const T, count: usize) -> Option<Vec<T>> {
    if count == 0 {
        return Some(Vec::new());
    }
    if pointer.is_null()
        || count
            .checked_mul(size_of::<T>())
            .is_none_or(|bytes| bytes > isize::MAX as usize)
    {
        return None;
    }
    let mut copied = Vec::new();
    copied.try_reserve_exact(count).ok()?;
    // SAFETY: The caller supplies count aligned, readable elements. The
    // borrow ends before any result pointer can be written by the wait.
    copied.extend_from_slice(unsafe { std::slice::from_raw_parts(pointer, count) });
    Some(copied)
}

/// Copies and validates signal handles and conditions from caller memory.
pub(super) unsafe fn signal_wait_inputs(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
) -> Option<WaitInputs> {
    let count = signal_count as usize;
    // SAFETY: The public ABI supplies readable arrays for the duration of each
    // copy. Own them before a satisfying-value output can alias any input.
    Some(WaitInputs {
        signals: unsafe { copy_wait_input(signals, count) }?,
        conditions: unsafe { copy_wait_input(conditions, count) }?,
        values: unsafe { copy_wait_input(values, count) }?,
    })
}

/// Waits for pause to complete.
fn wait_pause(wait_state_hint: u32, start: &Instant) {
    // A brief active phase avoids scheduler latency for short GPU completions
    // even when the caller allows a blocked wait. Longer blocked waits yield.
    if wait_state_hint == WAIT_STATE_ACTIVE
        || (wait_state_hint == WAIT_STATE_BLOCKED && start.elapsed() < BLOCKED_SPIN_BUDGET)
    {
        std::hint::spin_loop();
    } else {
        thread::yield_now();
    }
}

/// One validated signal; `retained` keeps its slot stable even after destroy.
pub(super) struct WaitEntry {
    /// Public handle whose slot is being observed.
    handle: HsaSignal,
    /// Stable mapped signal address while retained.
    signal: Option<NonNull<AmdSignal>>,
    /// Retirement flag for an imported or asynchronously owned signal.
    pub(super) retired: Option<Arc<AtomicBool>>,
    /// Whether this wait incremented the slot's internal reference count.
    retained: bool,
}

/// Retained signal slots and runtime token held for one blocking wait.
struct WaitRegistration {
    /// Slots released before the token permits shutdown to unmap them.
    entries: Vec<WaitEntry>,
    /// Prevents runtime teardown while wait code reads retained slots.
    _token: InFlightToken,
    /// Shared stop request that interrupts this wait during shutdown.
    stop: Arc<AtomicBool>,
}

impl Drop for WaitRegistration {
    /// Removes retained wait entries before their signal storage can retire.
    fn drop(&mut self) {
        if let Ok(mut guard) = lock() {
            if let Some(runtime) = guard.as_mut() {
                release_wait_entries(runtime, &mut self.entries);
            }
        }
    }
}

/// Releases the retained wait entries reference.
fn release_wait_entries(runtime: &mut HsaState, entries: &mut [WaitEntry]) {
    for entry in entries {
        if entry.retained {
            entry.signal = None;
            entry.retired = None;
            runtime.signal_state.release_async_signal(entry.handle);
            entry.retained = false;
        }
    }
}

/// # Safety
/// The caller must hold the runtime registry lock until every accepted signal
/// is retained, and must keep the returned registration alive through the wait.
unsafe fn prepare_multi_wait(
    runtime: &mut HsaState,
    handles: &[HsaSignal],
) -> Option<WaitRegistration> {
    let token = runtime.inflight.enter()?;
    // SAFETY: The runtime lock protects every accepted handle while its
    // storage reference is acquired, and the references are retained below.
    let mut entries =
        unsafe { collect_wait_signals(handles, |signal| runtime.signal_state.owns_signal(signal)) };
    for index in 0..entries.len() {
        if entries[index].signal.is_none() {
            continue;
        }
        let handle = entries[index].handle;
        if runtime.signal_state.retain_async_signal(handle).is_err() {
            release_wait_entries(runtime, &mut entries);
            return None;
        }
        entries[index].retained = true;
        let Some(record) = runtime
            .signal_state
            .async_signal_refs
            .get(&(handle.handle as usize))
        else {
            release_wait_entries(runtime, &mut entries);
            return None;
        };
        entries[index].retired = Some(record.retired.clone());
    }
    Some(WaitRegistration {
        entries,
        _token: token,
        stop: runtime.stop_workers.clone(),
    })
}

/// Selects live signals before a wait without touching invalid handles.
///
/// # Safety
/// Every handle accepted by `owns` must remain backed while the returned
/// entries are used. The caller must synchronize their eventual release.
pub(super) unsafe fn collect_wait_signals(
    signals: &[HsaSignal],
    mut owns: impl FnMut(HsaSignal) -> bool,
) -> Vec<WaitEntry> {
    signals
        .iter()
        .map(|signal| {
            let reference = if owns(*signal) {
                // SAFETY: The caller guarantees the accepted handle's backing.
                unsafe { signal_ptr(*signal) }
            } else {
                None
            };
            WaitEntry {
                handle: *signal,
                signal: reference,
                retired: None,
                retained: false,
            }
        })
        .collect()
}

/// Opens a wait-any operation while retaining all referenced signals.
pub(super) unsafe fn signal_wait_any_open(
    signals: &[WaitEntry],
    conditions: &[u32],
    values: &[SignalValue],
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_value: *mut SignalValue,
    stop: &AtomicBool,
) -> u32 {
    if signals.iter().all(|entry| entry.signal.is_none()) {
        return u32::MAX;
    }
    let start = std::time::Instant::now();
    let frequency = system_frequency();
    loop {
        if stop.load(Ordering::Acquire) {
            return u32::MAX;
        }
        for (index, entry) in signals.iter().enumerate() {
            let Some(signal) = entry.signal else {
                continue;
            };
            if entry
                .retired
                .as_ref()
                .is_some_and(|retired| retired.load(Ordering::Acquire))
            {
                return u32::MAX;
            }
            // SAFETY: The registration retains this slot until the wait exits.
            let observed = unsafe { signal.as_ref().value.load(Ordering::Relaxed) };
            if condition_met(conditions[index], observed, values[index]) {
                if !satisfying_value.is_null() {
                    // SAFETY: The caller supplied optional writable output storage.
                    unsafe { satisfying_value.write(observed) };
                }
                return u32::try_from(index).unwrap_or(u32::MAX);
            }
        }
        if timeout_hint != u64::MAX && timeout_elapsed(start.elapsed(), timeout_hint, frequency) {
            return u32::MAX;
        }
        wait_pause(wait_state_hint, &start);
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_wait_any")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Returns the first satisfied signal and its original input index.
pub unsafe extern "C" fn hsa_amd_signal_wait_any(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_value: *mut SignalValue,
) -> u32 {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(_) => return u32::MAX,
    };
    let Some(runtime) = guard.as_mut() else {
        return u32::MAX;
    };
    // SAFETY: The caller supplies signal_count readable entries in each array.
    let Some(inputs) = (unsafe { signal_wait_inputs(signal_count, signals, conditions, values) })
    else {
        return u32::MAX;
    };
    // SAFETY: The registry lock protects validation and retention; the returned
    // registration stays alive until this wait returns.
    let registration = match unsafe { prepare_multi_wait(runtime, &inputs.signals) } {
        Some(registration) => registration,
        None => return u32::MAX,
    };
    drop(guard);
    // SAFETY: The caller supplied optional writable output storage.
    unsafe {
        signal_wait_any_open(
            &registration.entries,
            &inputs.conditions,
            &inputs.values,
            timeout_hint,
            wait_state_hint,
            satisfying_value,
            &registration.stop,
        )
    }
}

/// Opens a wait-all operation while retaining all referenced signals.
pub(super) unsafe fn signal_wait_all_open(
    pending: &mut [WaitEntry],
    conditions: &[u32],
    values: &[SignalValue],
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_values: *mut SignalValue,
    stop: &AtomicBool,
) -> u32 {
    if !satisfying_values.is_null() {
        // SAFETY: The caller supplied signal_count writable output entries.
        // ROCr publishes zero for invalid and not-yet-satisfied signals even
        // when the wait ultimately times out.
        unsafe { std::ptr::write_bytes(satisfying_values, 0, pending.len()) };
    }
    let start = std::time::Instant::now();
    let frequency = system_frequency();
    loop {
        if stop.load(Ordering::Acquire) {
            return u32::MAX;
        }
        let mut remaining = false;
        for (index, entry) in pending.iter_mut().enumerate() {
            let Some(signal_ref) = entry.signal else {
                continue;
            };
            if entry
                .retired
                .as_ref()
                .is_some_and(|retired| retired.load(Ordering::Acquire))
            {
                return u32::MAX;
            }
            // SAFETY: The registration retains this slot until the wait exits.
            let observed = unsafe { signal_ref.as_ref().value.load(Ordering::Relaxed) };
            if condition_met(conditions[index], observed, values[index]) {
                if !satisfying_values.is_null() {
                    // SAFETY: The caller supplied signal_count writable entries.
                    unsafe { satisfying_values.add(index).write(observed) };
                }
                entry.signal = None;
            } else {
                remaining = true;
            }
        }
        if !remaining {
            return 0;
        }
        if timeout_hint != u64::MAX && timeout_elapsed(start.elapsed(), timeout_hint, frequency) {
            return u32::MAX;
        }
        wait_pause(wait_state_hint, &start);
    }
}

#[cfg_attr(
    all(target_arch = "x86_64", not(test)),
    unsafe(export_name = "__hsa_impl_hsa_amd_signal_wait_all")
)]
#[cfg_attr(any(not(target_arch = "x86_64"), test), unsafe(no_mangle))]
/// Waits until every signal condition holds or the timeout expires.
pub unsafe extern "C" fn hsa_amd_signal_wait_all(
    signal_count: u32,
    signals: *const HsaSignal,
    conditions: *const u32,
    values: *const SignalValue,
    timeout_hint: u64,
    wait_state_hint: u32,
    satisfying_values: *mut SignalValue,
) -> u32 {
    let mut guard = match lock() {
        Ok(guard) => guard,
        Err(_) => return u32::MAX,
    };
    let Some(runtime) = guard.as_mut() else {
        return u32::MAX;
    };
    // SAFETY: The caller supplies signal_count readable entries in each array.
    let Some(inputs) = (unsafe { signal_wait_inputs(signal_count, signals, conditions, values) })
    else {
        return u32::MAX;
    };
    // SAFETY: The registry lock protects validation and retention; the returned
    // registration stays alive until this wait returns.
    let mut registration = match unsafe { prepare_multi_wait(runtime, &inputs.signals) } {
        Some(registration) => registration,
        None => return u32::MAX,
    };
    drop(guard);
    // SAFETY: The caller supplied optional writable output storage.
    unsafe {
        signal_wait_all_open(
            &mut registration.entries,
            &inputs.conditions,
            &inputs.values,
            timeout_hint,
            wait_state_hint,
            satisfying_values,
            &registration.stop,
        )
    }
}
