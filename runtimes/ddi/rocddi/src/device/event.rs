// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Device notifications shared with API frontends.
//!
//! System events describe observed device failures. Each subscription has its
//! own delivery cursor; the driver observes each native notification once and
//! retains unread records plus the latest details of each exception kind for
//! subscribers that register after an operation observed it.
//! Frontends own callback dispatch, error policy, and any worker threads.

use crate::Error;
use crate::device::Device;
use crate::driver;

/// Driver-independent cause of a GPU virtual-memory fault.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuMemoryFaultCause {
    /// The driver reported no additional cause.
    None,
    /// On-chip SRAM reported an error.
    SramEcc,
    /// Device memory reported an error.
    DramEcc,
    /// The GPU stopped making progress.
    Hang,
    /// The driver reported a cause that has no portable classification.
    Other,
}

/// One process-level GPU virtual-memory fault reported by a driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "independent memory-fault facts come from the driver event payload"
)]
pub struct GpuMemoryFault {
    /// Activated endpoint that reported the fault, when known to this session.
    pub endpoint_id: Option<[u8; 16]>,
    /// Virtual address reported by the driver.
    pub virtual_address: u64,
    /// The address was not present or required supervisor privilege.
    pub page_not_present: bool,
    /// The access attempted to write a read-only page.
    pub read_only: bool,
    /// The access attempted to execute a non-executable page.
    pub no_execute: bool,
    /// The reported virtual address may be imprecise.
    pub imprecise: bool,
    /// Additional fault cause when the driver can classify it.
    pub cause: GpuMemoryFaultCause,
}

/// Scope of a GPU reset reported by the driver.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuResetScope {
    /// The driver reset the entire GPU.
    WholeGpu,
    /// The driver reported another or unknown reset scope.
    Other,
}

/// Driver-independent cause of a GPU hardware exception.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GpuResetCause {
    /// The GPU stopped making progress.
    Hang,
    /// The driver reported an ECC error.
    Ecc,
    /// The driver reported another or unknown cause.
    Other,
}

/// One GPU hardware exception and its reset information.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuHardwareException {
    /// Activated endpoint that reported the exception, when known.
    pub endpoint_id: Option<[u8; 16]>,
    /// Scope of the reset reported by the driver.
    pub scope: GpuResetScope,
    /// Whether the driver reported loss of device memory.
    pub memory_lost: bool,
    /// Cause of the exception reported by the driver.
    pub cause: GpuResetCause,
}

/// A device event retained by the DDI for independent frontend observers.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeviceEvent {
    /// A process-level GPU virtual-memory fault.
    GpuMemoryFault(GpuMemoryFault),
    /// A GPU hardware exception or reset.
    GpuHardwareException(GpuHardwareException),
}

/// One observer of a driver's system event stream.
///
/// A subscription retains the native event owner until it is dropped. Polling
/// never invokes frontend code, and an event is returned at most once to this
/// subscription. Retained events are replayed to late subscribers. If a slow
/// subscriber fills the finite event buffer, native polling waits until that
/// subscriber reads or drops its backlog.
pub struct DeviceEventSubscription {
    inner: driver::EventSubscription,
}

impl DeviceEventSubscription {
    /// Returns the next observed event, or `None` when no new event is ready.
    ///
    /// # Errors
    /// Reports native polling or driver-lifetime failures. `Busy` means a
    /// subscriber has not drained the finite event buffer; retry later. A
    /// later poll can still retrieve an event recorded before a polling failure.
    pub fn poll(&mut self) -> Result<Option<DeviceEvent>, Error> {
        self.inner.poll()
    }
}

/// Subscribes to events on the activated device's driver connection.
///
/// Multiple devices on one connection share native observation, while every
/// returned subscription has an independent cursor. A driver without event
/// support returns `Unsupported`.
///
/// # Errors
/// Reports unsupported drivers and unavailable native event state.
pub fn subscribe(device: &Device) -> Result<DeviceEventSubscription, Error> {
    Ok(DeviceEventSubscription {
        inner: device.driver_state.subscribe_events()?,
    })
}
