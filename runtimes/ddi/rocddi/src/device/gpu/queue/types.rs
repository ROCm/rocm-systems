// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Descriptor fields describe one requested native queue. Engine-specific
//! options stay tagged with their packet format so unrelated fields cannot be
//! mistaken for each other. Each creation requests an independently owned
//! queue; publication policy belongs to the caller.

use std::mem::{align_of, size_of};

/// Scheduling priority requested from the native queue implementation. A
/// backend rejects priorities it cannot implement instead of silently ignoring
/// them. Priority never establishes an execution or memory dependency.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueuePriority {
    /// Lower native scheduling priority than ordinary work.
    Low,
    /// The backend's ordinary queue priority.
    #[default]
    Normal,
    /// Higher native scheduling priority than ordinary work.
    High,
}

/// Host producer discipline for an AQL ring. This is separate from the packet
/// format and grants no automatic locking or reservation service. Callers must
/// follow the corresponding publication protocol when writing the ring.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueueProducerMode {
    /// One producer owns write-index updates and packet publication.
    #[default]
    Single,
    /// Producers coordinate packet reservations and publication explicitly.
    Multiple,
}

/// Placement of a native packet ring. Both choices retain a host mapping for
/// producers; local memory requires CPU-visible VRAM on the selected GPU.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum QueueRingMemory {
    /// System memory owned by the native queue.
    #[default]
    System,
    /// Host-visible local memory owned by the native queue.
    HostVisibleLocal,
}

/// Packet format and its engine-specific creation parameters. PM4 and AQL use
/// compute engines; SDMA uses a copy engine. Producer discipline is an AQL
/// option, not a substitute for this format selection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum QueueParameters {
    /// A native PM4 compute queue. The initial profile is a directly published,
    /// host-only, single-producer queue with no scratch backing.
    Pm4,
    /// An AQL compute queue. Dynamic scratch management, callback policy, and
    /// shared cooperative-queue acquisition are upper-runtime services, not
    /// implied by allocating this native queue and its transport storage.
    Aql {
        /// Producer discipline that the caller will uphold.
        producer_mode: QueueProducerMode,
        /// Native packet-ring placement. The caller still owns packet-store
        /// ordering before advancing the write index and ringing the doorbell.
        ring_memory: QueueRingMemory,
        /// Allocate native global work synchronization for this queue before
        /// publishing its transport. KFD permits only one such queue per
        /// process and device; sharing it is the caller's policy.
        global_work_sync: bool,
        /// GPU-visible signal payload used by the command processor to stop
        /// and report AQL queue errors, or `None` when the frontend does not
        /// service them.
        /// The frontend retains the signal storage through queue destruction.
        inactive_signal: Option<u64>,
        /// Separate exception payload and opaque native event identity used by
        /// the selected backend for queue-error notification.
        error_event: Option<QueueErrorEvent>,
        /// Fixed AQL scratch backing, or `None` for a no-scratch queue.
        scratch: Option<QueueScratch>,
    },
    /// A byte-addressed SDMA copy queue with the backend's ordinary engine
    /// selection. Existing callers need no engine-specific policy.
    Sdma,
    /// A byte-addressed SDMA copy queue routed to a selected native engine.
    SdmaByEngine {
        /// Explicit engine ID or round-robin selection within one device VM.
        selection: SdmaEngineSelection,
        /// Native packet-ring placement. Producers order packet stores before
        /// advancing the write index and ringing the doorbell.
        ring_memory: QueueRingMemory,
    },
}

/// Native SDMA engine selection for a targeted queue request.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SdmaEngineSelection {
    /// Select the next engine in the device's round-robin sequence.
    Any,
    /// Select this engine in the device's general-plus-XGMI ID space.
    Id(u32),
}

/// Fixed scratch backing supplied for the lifetime of one AQL queue. The
/// adapter resolves the public memory attachment to this stable device range;
/// the native backend derives target-specific control fields from it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueScratch {
    /// Device address of the first scratch byte.
    pub device_address: u64,
    /// Bytes available to the queue starting at `device_address`.
    pub byte_length: u64,
    /// Largest private segment per execution lane accepted by the queue.
    pub maximum_private_segment_byte_length: u32,
    /// Execution lanes per wave used to size the private segment.
    pub lanes_per_wave: u32,
    /// Maximum number of simultaneously scratch-backed waves.
    pub maximum_wave_count: u32,
}

/// Native queue exception notification published through the CWSR header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueErrorEvent {
    /// GPU-visible address of the signal payload that receives exception bits.
    pub(crate) payload_address: u64,
    /// Backend-defined event token. Its interpretation is deliberately private
    /// so a platform event identity does not become a queue-format contract.
    /// The width can carry a native 64-bit event or handle identity.
    pub(crate) native_event_token: u64,
}

/// One queue allocation request with common fields and tagged engine
/// parameters. Frontend ownership and callback policy are intentionally absent.
/// Ring capacity is always expressed in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueRequest {
    /// Requested power-of-two packet-ring capacity in bytes. Native limits are
    /// validated before acquisition; no implicit size rounding is performed.
    pub ring_size_bytes: u64,
    /// Packet format and options belonging to its execution engine.
    pub parameters: QueueParameters,
    /// Scheduling priority, with no implied dependency or completion order.
    pub priority: QueuePriority,
    /// Require queue transport mappings usable by the queue's own device.
    pub device_producer: bool,
}

/// Access width reported for an index or queue notification word. The adapter checks these
/// native facts against the negotiated queue format before exposing a mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueAccessWidth {
    /// One aligned 32-bit access.
    Bits32,
    /// One aligned 64-bit access.
    Bits64,
}

/// Existing queue transport addresses, borrowed from a live queue owner. These
/// numeric addresses create no Rust references and convey no ordering promise.
/// Callers must obey the native index, packet-publication, and notification protocol;
/// accessing arbitrary bytes or ringing a malformed queue is caller-invalid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueTransport {
    /// Host mapping of the packet ring, valid until destruction begins.
    pub ring_host_address: usize,
    /// Address from which the selected device fetches ring packets.
    pub ring_device_address: u64,
    /// Usable ring capacity in bytes, independent of native index units.
    pub ring_size_bytes: u64,
    /// Validated host control region for an AQL queue, when the backend has one.
    pub aql_control: Option<AqlControlRegion>,
    /// Resolved engine ID for a targeted SDMA queue, or `None` for other
    /// formats and the backend-selected generic SDMA format.
    pub sdma_engine_id: Option<u32>,
    /// Host mapping of the device-maintained read index.
    pub read_index_host_address: usize,
    /// Queue-device address of the device-maintained read index.
    pub read_index_device_address: u64,
    /// Host mapping of the caller-maintained write index.
    pub write_index_host_address: usize,
    /// Queue-device address of the caller-maintained write index.
    pub write_index_device_address: u64,
    /// Required read-index access width.
    pub read_index_width: QueueAccessWidth,
    /// Required write-index access width.
    pub write_index_width: QueueAccessWidth,
    /// Bytes represented by one index step: PM4 dwords, AQL packets, or SDMA
    /// bytes.
    pub index_unit_bytes: u32,
    /// Whether the native read index wraps at the ring capacity rather than
    /// advancing in the full-width counter domain.
    pub read_index_wraps: bool,
    /// Host address of this queue's notification word. It may be an MMIO
    /// doorbell or a backend-serviced word. Use the ordering and value
    /// convention of the selected queue transport.
    pub doorbell_host_address: usize,
    /// Queue-device notification address when device production was required
    /// at creation, or `None` for a host-only queue.
    pub doorbell_device_address: Option<u64>,
    /// Required doorbell access width, not the size of its mapping.
    pub doorbell_width: QueueAccessWidth,
}

/// Backend-validated AQL control storage borrowed from a live native queue.
/// Its field addresses refer to native control words, not HSA policy objects.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AqlControlRegion {
    /// Host address of the complete native control layout.
    host_base: usize,
    /// Number of accessible bytes in that layout.
    byte_length: usize,
    /// Alignment guaranteed for the control base.
    alignment: usize,
    /// Initial bytes available for a frontend's public queue prefix.
    writable_prefix_bytes: usize,
    /// Host address of the device-maintained read index.
    read_index_host_address: usize,
    /// Host address of the producer-maintained write index.
    write_index_host_address: usize,
    /// Host address of the queue properties word.
    properties_host_address: usize,
}

impl AqlControlRegion {
    /// Validates the backend's aligned control span and named word addresses.
    pub(crate) fn new(
        host_base: usize,
        byte_length: usize,
        alignment: usize,
        writable_prefix_bytes: usize,
        read_index_host_address: usize,
        write_index_host_address: usize,
        properties_host_address: usize,
    ) -> Option<Self> {
        let end = host_base.checked_add(byte_length)?;
        let contains = |address: usize, bytes: usize| {
            address >= host_base
                && address
                    .checked_add(bytes)
                    .is_some_and(|word_end| word_end <= end)
        };
        if host_base == 0
            || alignment == 0
            || !alignment.is_power_of_two()
            || host_base % alignment != 0
            || writable_prefix_bytes > byte_length
            || !contains(read_index_host_address, size_of::<u64>())
            || !contains(write_index_host_address, size_of::<u64>())
            || !contains(properties_host_address, size_of::<u32>())
            || read_index_host_address % align_of::<u64>() != 0
            || write_index_host_address % align_of::<u64>() != 0
            || properties_host_address % align_of::<u32>() != 0
        {
            return None;
        }
        Some(Self {
            host_base,
            byte_length,
            alignment,
            writable_prefix_bytes,
            read_index_host_address,
            write_index_host_address,
            properties_host_address,
        })
    }

    /// Host address of the validated native control layout.
    #[must_use]
    pub fn host_base(self) -> usize {
        self.host_base
    }
    /// Accessible extent of the native control layout.
    #[must_use]
    pub fn byte_length(self) -> usize {
        self.byte_length
    }
    /// Guaranteed alignment of the control base.
    #[must_use]
    pub fn alignment(self) -> usize {
        self.alignment
    }
    /// Bytes available for a frontend public prefix.
    #[must_use]
    pub fn writable_prefix_bytes(self) -> usize {
        self.writable_prefix_bytes
    }
    /// Host address of the native read index.
    #[must_use]
    pub fn read_index_host_address(self) -> usize {
        self.read_index_host_address
    }
    /// Host address of the native write index.
    #[must_use]
    pub fn write_index_host_address(self) -> usize {
        self.write_index_host_address
    }
    /// Host address of the native properties word.
    #[must_use]
    pub fn properties_host_address(self) -> usize {
        self.properties_host_address
    }
}
