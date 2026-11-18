// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Signal stores shared by public adapters and internal HSA operations.

use super::*;

/// Stores a retained signal with the requested host/device ordering.
///
/// # Safety
/// The caller retains signal storage and any queue owning its doorbell, and
/// excludes their destruction until this operation completes.
pub(crate) unsafe fn store(signal: HsaSignal, value: SignalValue, order: Ordering) -> Option<()> {
    // SAFETY: The caller retains the complete signal/queue ownership graph.
    unsafe { with_public_signal(signal, |slot| store_signal_value(slot, value, order)) }
}

/// Stores a signal value according to the locally trusted host signal kind.
///
/// # Safety
/// The caller must retain the signal and any queue owning its doorbell until
/// this store completes.
unsafe fn store_signal_value(signal: &AmdSignal, value: SignalValue, order: Ordering) {
    let address = std::ptr::from_ref(signal) as usize;
    let kind = host_store_kind(address, signal.kind.load(Ordering::Acquire));
    match kind {
        AMD_SIGNAL_KIND_DOORBELL => {
            if order == Ordering::Release {
                fence(Ordering::Release);
            }
            let address = signal.value.load(Ordering::Relaxed) as usize;
            // SAFETY: The caller retains the locally owned queue and its MMIO
            // mapping. Shared IPC signals are excluded by host_store_kind.
            unsafe { rocddi::device::gpu::queue::store_ordered_mmio64(address, value as u64) };
        }
        AMD_SIGNAL_KIND_LEGACY_DOORBELL => {
            signal.value.store(value, order);
            // SAFETY: The HSA caller retains the owning queue through this
            // store, including its stable intercept owner.
            unsafe { crate::queue_intercept::ring(signal.intercept_owner()) };
        }
        _ => {
            signal.value.store(value, order);
            flush_signal_store(signal);
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::signal::{register_ipc_signal, unregister_ipc_signal};

    #[test]
    fn internal_stores_inherit_admission_but_cannot_start_after_shutdown_closes_it() {
        let gate = Arc::new(crate::admission::InFlightTracker::new_process());
        gate.open(73);
        let _test_gate = crate::admission::install_test_public_raw_gate(gate.clone());
        let slot = AmdSignal::user(7);
        let handle = HsaSignal {
            handle: std::ptr::from_ref(&slot) as u64,
        };
        crate::admission::ApiLeaseScope::run(gate.enter().unwrap(), || {
            gate.close();
            // SAFETY: The stack slot remains live for both stores. The inherited
            // generation lease permits descendants to finish after final closure.
            assert_eq!(unsafe { store(handle, 9, Ordering::Release) }, Some(()));
        });
        assert_eq!(unsafe { store(handle, 11, Ordering::Relaxed) }, None);
        assert_eq!(slot.value.load(Ordering::Acquire), 9);
        assert!(gate.wait_idle_for(Duration::from_secs(1)));
    }

    #[test]
    fn peer_modified_ipc_kind_cannot_ring_a_doorbell() {
        let signal = AmdSignal::user(0);
        let address = std::ptr::from_ref(&signal) as usize;
        register_ipc_signal(address).unwrap();

        // An imported peer may modify the shared kind and value at any time.
        // The invalid MMIO address would fault if either store took the
        // doorbell branch; the legacy kind would use an invalid owner.
        signal
            .kind
            .store(AMD_SIGNAL_KIND_DOORBELL, Ordering::Release);
        signal.value.store(1, Ordering::Release);
        // SAFETY: This local signal is retained throughout both stores.
        unsafe { store_signal_value(&signal, 27, Ordering::Relaxed) };
        assert_eq!(signal.value.load(Ordering::Acquire), 27);

        signal
            .kind
            .store(AMD_SIGNAL_KIND_LEGACY_DOORBELL, Ordering::Release);
        // SAFETY: This local signal is retained throughout both stores.
        unsafe { store_signal_value(&signal, 39, Ordering::Release) };
        assert_eq!(signal.value.load(Ordering::Acquire), 39);
        unregister_ipc_signal(address);
    }
}
