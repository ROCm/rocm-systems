//! Linux KFD GPU notification events shared with API frontends.
//!
//! This implementation module is re-exported only through
//! `gpu::event::linux`. Its event identifiers and mailbox slots are Linux KFD
//! transport details, not requirements of the platform-neutral device model.

use crate::Error;
use crate::driver;
use crate::driver::linux_interop::LinuxGpuEventDriver;
use crate::gpu::GpuDevice;
use crate::host_storage::Owned;
use crate::memory::Allocation;
use crate::queue::QueueErrorEvent;

/// One process-level GPU virtual-memory fault reported by the native driver.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent KFD memory-fault cause bits mirror the native event payload"
)]
pub struct GpuMemoryFault {
    /// KFD's per-boot identifier for the faulting GPU.
    pub kfd_gpu_id: u32,
    /// Virtual address reported by KFD.
    pub virtual_address: u64,
    /// The address was not present or required supervisor privilege.
    pub page_not_present: bool,
    /// The access attempted to write a read-only page.
    pub read_only: bool,
    /// The access attempted to execute a non-executable page.
    pub no_execute: bool,
    /// The reported virtual address may be imprecise.
    pub imprecise: bool,
    /// Native memory-exception error classification.
    pub error_type: u32,
}

/// KFD identity and mailbox slot assigned to one interrupt-capable signal.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SignalEventInfo {
    /// Process-local KFD event identifier written by the GPU on notification.
    pub kfd_event_id: u32,
    /// Eight-byte slot within the process signal event page.
    pub event_page_slot_index: u32,
}

/// Owns one process-local KFD signal event.
#[doc(hidden)]
pub struct SignalEvent {
    pub(crate) inner: Owned<driver::NativeSignalEvent>,
    pub(crate) info: SignalEventInfo,
}

impl SignalEvent {
    /// Returns the immutable event identity and mailbox slot.
    #[must_use]
    pub fn info(&self) -> SignalEventInfo {
        self.info
    }

    /// Creates the opaque notification descriptor used by a KFD-backed AQL
    /// queue to report an exception through this event.
    #[must_use]
    pub fn queue_error_event(&self, payload_address: u64) -> QueueErrorEvent {
        QueueErrorEvent {
            payload_address,
            native_event_token: u64::from(self.info.kfd_event_id),
        }
    }

    /// Releases the native signal event while preserving retry state on failure.
    ///
    /// # Errors
    /// Reports the native destruction failure and retains the event identity
    /// when retrying is safe.
    pub fn destroy(&mut self) -> Result<(), Error> {
        driver::PlatformDriver::destroy_kfd_signal_event(&mut self.inner)
    }
}

/// Claims and polls this session's process-level KFD GPU memory-fault event.
/// Once claimed, ordinary device checks leave memory-fault delivery to the
/// caller while continuing to observe terminal hardware loss.
///
/// # Errors
/// Returns a native KFD error, or `DeviceLost` when terminal loss was already
/// observed.
pub fn poll_memory_fault(device: GpuDevice<'_>) -> Result<Option<GpuMemoryFault>, Error> {
    device
        .device
        .driver
        .poll_kfd_memory_fault(&device.device.state)
}

/// Creates an auto-reset KFD signal event. The first event in a process supplies
/// its Linux-managed shared event-page allocation; later events reuse that page.
///
/// # Errors
/// Rejects an invalid page allocation and reports native event creation
/// failures without publishing a partial owner.
pub fn create_signal_event(
    device: GpuDevice<'_>,
    event_page: Option<&Allocation>,
) -> Result<SignalEvent, Error> {
    let inner = device
        .device
        .driver
        .create_kfd_signal_event(&device.device.state, event_page.map(|page| &*page.inner))?;
    let info = inner.info();
    Ok(SignalEvent { inner, info })
}
