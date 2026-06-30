// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::communicator::{Communicator, ManagementCompletion};
use crate::error::{Error, Result, check};
use crate::group::ensure_no_active_group;
use crate::sys;
use std::alloc::Layout;
use std::mem::{MaybeUninit, align_of, size_of};
use std::ptr;

/// The GIN connections requested for a device communicator.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum GinConnectionType {
    None,
    Full,
    Rail,
}

impl GinConnectionType {
    const fn as_raw(self) -> sys::ncclGinConnectionType_t {
        match self {
            Self::None => sys::NCCL_GIN_CONNECTION_NONE,
            Self::Full => sys::NCCL_GIN_CONNECTION_FULL,
            Self::Rail => sys::NCCL_GIN_CONNECTION_RAIL,
        }
    }
}

/// Initialized requirements for `ncclDevCommCreate`.
///
/// `Default` exactly mirrors `NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER` from the
/// selected NCCL headers. Resource/team linked-list builders are intentionally
/// not exposed yet because their output-handle pointers need a separate pinned
/// ownership API.
#[derive(Clone)]
pub struct DeviceCommRequirements {
    raw: sys::ncclDevCommRequirements_t,
}

impl Default for DeviceCommRequirements {
    fn default() -> Self {
        Self {
            raw: sys::ncclDevCommRequirements_t {
                size: size_of::<sys::ncclDevCommRequirements_t>(),
                magic: sys::NCCL_API_MAGIC,
                version: sys::NCCL_VERSION_CODE,
                resourceRequirementsList: ptr::null_mut(),
                teamRequirementsList: ptr::null_mut(),
                lsaMultimem: false,
                barrierCount: 0,
                lsaBarrierCount: 0,
                railGinBarrierCount: 0,
                lsaLLA2ABlockCount: 0,
                lsaLLA2ASlotCount: 0,
                ginForceEnable: false,
                ginContextCount: 4,
                ginSignalCount: 0,
                ginCounterCount: 0,
                ginConnectionType: sys::NCCL_GIN_CONNECTION_NONE,
                ginExclusiveContexts: false,
                ginQueueDepth: 0,
                ginTrafficClass: sys::NCCL_CONFIG_UNDEF_INT,
                worldGinBarrierCount: 0,
                ginStrongSignalsRequired: true,
                ginVaSignalsRequired: true,
            },
        }
    }
}

impl std::fmt::Debug for DeviceCommRequirements {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceCommRequirements")
            .field("lsa_multimem", &self.raw.lsaMultimem)
            .field("barrier_count", &self.raw.barrierCount)
            .field("lsa_barrier_count", &self.raw.lsaBarrierCount)
            .field("rail_gin_barrier_count", &self.raw.railGinBarrierCount)
            .field("lsa_lla2a_block_count", &self.raw.lsaLLA2ABlockCount)
            .field("lsa_lla2a_slot_count", &self.raw.lsaLLA2ASlotCount)
            .field("gin_force_enable", &self.raw.ginForceEnable)
            .field("gin_context_count", &self.raw.ginContextCount)
            .field("gin_signal_count", &self.raw.ginSignalCount)
            .field("gin_counter_count", &self.raw.ginCounterCount)
            .field("gin_connection_type", &self.raw.ginConnectionType)
            .field("gin_exclusive_contexts", &self.raw.ginExclusiveContexts)
            .field("gin_queue_depth", &self.raw.ginQueueDepth)
            .field("gin_traffic_class", &self.raw.ginTrafficClass)
            .field("world_gin_barrier_count", &self.raw.worldGinBarrierCount)
            .field(
                "gin_strong_signals_required",
                &self.raw.ginStrongSignalsRequired,
            )
            .field("gin_va_signals_required", &self.raw.ginVaSignalsRequired)
            .finish()
    }
}

impl DeviceCommRequirements {
    pub fn lsa_multimem(mut self, enabled: bool) -> Self {
        self.raw.lsaMultimem = enabled;
        self
    }

    pub fn barrier_count(mut self, count: i32) -> Self {
        self.raw.barrierCount = count;
        self
    }

    pub fn lsa_barrier_count(mut self, count: i32) -> Self {
        self.raw.lsaBarrierCount = count;
        self
    }

    pub fn rail_gin_barrier_count(mut self, count: i32) -> Self {
        self.raw.railGinBarrierCount = count;
        self
    }

    pub fn lsa_lla2a(mut self, block_count: i32, slot_count: i32) -> Self {
        self.raw.lsaLLA2ABlockCount = block_count;
        self.raw.lsaLLA2ASlotCount = slot_count;
        self
    }

    pub fn gin_force_enable(mut self, enabled: bool) -> Self {
        self.raw.ginForceEnable = enabled;
        self
    }

    pub fn gin_context_count(mut self, count: i32) -> Self {
        self.raw.ginContextCount = count;
        self
    }

    pub fn gin_signal_count(mut self, count: i32) -> Self {
        self.raw.ginSignalCount = count;
        self
    }

    pub fn gin_counter_count(mut self, count: i32) -> Self {
        self.raw.ginCounterCount = count;
        self
    }

    pub fn gin_connection_type(mut self, connection_type: GinConnectionType) -> Self {
        self.raw.ginConnectionType = connection_type.as_raw();
        self
    }

    pub fn gin_exclusive_contexts(mut self, enabled: bool) -> Self {
        self.raw.ginExclusiveContexts = enabled;
        self
    }

    pub fn gin_queue_depth(mut self, depth: i32) -> Self {
        self.raw.ginQueueDepth = depth;
        self
    }

    pub fn gin_traffic_class(mut self, traffic_class: i32) -> Self {
        self.raw.ginTrafficClass = traffic_class;
        self
    }

    pub fn world_gin_barrier_count(mut self, count: i32) -> Self {
        self.raw.worldGinBarrierCount = count;
        self
    }

    pub fn gin_strong_signals_required(mut self, required: bool) -> Self {
        self.raw.ginStrongSignalsRequired = required;
        self
    }

    pub fn gin_va_signals_required(mut self, required: bool) -> Self {
        self.raw.ginVaSignalsRequired = required;
        self
    }
}

/// An owned host image of `ncclDevComm_t` and its NCCL-side resources.
///
/// The device wrapper expects a pointer to a device-resident copy of this
/// image. [`Self::as_bytes`] exposes the exact initialized host representation
/// so a CUDA integration can allocate device storage and copy it there without
/// this crate depending on a particular CUDA runtime wrapper.
pub struct DeviceCommunicator<'comm> {
    communicator: &'comm Communicator,
    raw: Option<Box<sys::ncclDevComm_t>>,
}

impl std::fmt::Debug for DeviceCommunicator<'_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceCommunicator")
            .field("byte_len", &Self::BYTE_LEN)
            .field("copy_alignment", &Self::COPY_ALIGNMENT)
            .finish_non_exhaustive()
    }
}

impl Communicator {
    /// Create the device-side communicator image and associated NCCL
    /// resources.
    pub fn create_device_communicator<'comm>(
        &'comm self,
        requirements: &DeviceCommRequirements,
    ) -> Result<DeviceCommunicator<'comm>> {
        ensure_no_active_group("NCCL device-communicator creation")?;
        let communicator = self.active_raw()?;
        // NCCL's implementation initializes the entire public image (including
        // padding) before returning. Starting zeroed also keeps failure cleanup
        // and byte-copy instrumentation deterministic.
        let mut raw = Box::new(MaybeUninit::<sys::ncclDevComm_t>::zeroed());
        let status =
            unsafe { sys::ncclDevCommCreate(communicator, &requirements.raw, raw.as_mut_ptr()) };
        match self.complete_deferred_output_call(status, "NCCL device-communicator creation") {
            ManagementCompletion::Complete => {}
            ManagementCompletion::SynchronousFailure(error)
            | ManagementCompletion::TerminalFailure(error) => return Err(error),
            ManagementCompletion::Uncertain(error) => {
                // A deferred NCCL task may still target this host image.
                Box::leak(raw);
                return Err(error);
            }
        }
        // Validate the sentinel fields after terminal polling before treating
        // the zero-initialized output image as this call's result.
        let initialized = unsafe { raw.assume_init_ref() };
        if initialized.magic != sys::NCCL_API_MAGIC || initialized.version != sys::NCCL_VERSION_CODE
        {
            return Err(Error::local(format!(
                "NCCL did not initialize the device communicator image (magic={:#x}, version={})",
                initialized.magic, initialized.version
            )));
        }
        let raw_pointer = Box::into_raw(raw).cast::<sys::ncclDevComm_t>();
        let raw = unsafe { Box::from_raw(raw_pointer) };
        Ok(DeviceCommunicator {
            communicator: self,
            raw: Some(raw),
        })
    }
}

impl DeviceCommunicator<'_> {
    /// Number of bytes that must be copied for device use.
    pub const BYTE_LEN: usize = size_of::<sys::ncclDevComm_t>();

    /// Required alignment for a typed device-side `ncclDevComm_t` image.
    pub const COPY_ALIGNMENT: usize = align_of::<sys::ncclDevComm_t>();

    pub const fn copy_layout() -> Layout {
        // Both values come from one concrete Rust type and therefore always
        // form a valid Layout.
        match Layout::from_size_align(Self::BYTE_LEN, Self::COPY_ALIGNMENT) {
            Ok(layout) => layout,
            Err(_) => unreachable!(),
        }
    }

    /// Return the complete host image for a host-to-device copy.
    ///
    /// Copy all bytes, unchanged, to storage satisfying [`Self::copy_layout`].
    /// Do not interpret or patch pointer fields. Every copied image becomes
    /// invalid when this owner is dropped; all kernels using a copy must finish
    /// first. A CUDA HtoD copy and subsequent kernel launch are outside Rust's
    /// lifetime model and therefore remain unsafe at that integration boundary.
    pub fn as_bytes(&self) -> &[u8] {
        let raw = self
            .raw
            .as_ref()
            .expect("live DeviceCommunicator must contain its host image");
        unsafe {
            std::slice::from_raw_parts(ptr::from_ref(raw.as_ref()).cast::<u8>(), Self::BYTE_LEN)
        }
    }

    /// Destroy the NCCL-side device-communicator resources and report errors.
    ///
    /// All kernels using a device-resident copy of [`Self::as_bytes`] must have
    /// completed before this call. The host image is removed before entering
    /// NCCL, so `Drop` cannot issue a second destruction even when NCCL reports
    /// an error. Owners that do not call this method retain the best-effort
    /// `Drop` fallback.
    pub fn destroy(mut self) -> Result<()> {
        self.destroy_once()
    }

    fn destroy_once(&mut self) -> Result<()> {
        let Some(raw) = self.raw.take() else {
            return Ok(());
        };
        check(unsafe { sys::ncclDevCommDestroy(self.communicator.raw(), raw.as_ref()) })
    }
}

impl Drop for DeviceCommunicator<'_> {
    fn drop(&mut self) {
        let _ = self.destroy_once();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_matches_device_requirements_initializer() {
        let requirements = DeviceCommRequirements::default();
        let raw = &requirements.raw;

        assert_eq!(raw.size, size_of::<sys::ncclDevCommRequirements_t>());
        assert_eq!(raw.magic, sys::NCCL_API_MAGIC);
        assert_eq!(raw.version, sys::NCCL_VERSION_CODE);
        assert!(raw.resourceRequirementsList.is_null());
        assert!(raw.teamRequirementsList.is_null());
        assert!(!raw.lsaMultimem);
        assert_eq!(raw.barrierCount, 0);
        assert_eq!(raw.lsaBarrierCount, 0);
        assert_eq!(raw.railGinBarrierCount, 0);
        assert_eq!(raw.lsaLLA2ABlockCount, 0);
        assert_eq!(raw.lsaLLA2ASlotCount, 0);
        assert!(!raw.ginForceEnable);
        assert_eq!(raw.ginContextCount, 4);
        assert_eq!(raw.ginSignalCount, 0);
        assert_eq!(raw.ginCounterCount, 0);
        assert_eq!(raw.ginConnectionType, sys::NCCL_GIN_CONNECTION_NONE);
        assert!(!raw.ginExclusiveContexts);
        assert_eq!(raw.ginQueueDepth, 0);
        assert_eq!(raw.ginTrafficClass, sys::NCCL_CONFIG_UNDEF_INT);
        assert_eq!(raw.worldGinBarrierCount, 0);
        assert!(raw.ginStrongSignalsRequired);
        assert!(raw.ginVaSignalsRequired);
    }

    #[test]
    fn device_copy_contract_uses_the_public_abi_layout() {
        assert_eq!(
            DeviceCommunicator::BYTE_LEN,
            size_of::<sys::ncclDevComm_t>()
        );
        assert_eq!(
            DeviceCommunicator::COPY_ALIGNMENT,
            align_of::<sys::ncclDevComm_t>()
        );
        assert_eq!(
            DeviceCommunicator::copy_layout().size(),
            DeviceCommunicator::BYTE_LEN
        );
    }

    #[test]
    fn requirements_builders_update_only_requested_fields() {
        let requirements = DeviceCommRequirements::default()
            .lsa_multimem(true)
            .barrier_count(3)
            .gin_context_count(8)
            .gin_connection_type(GinConnectionType::Rail)
            .gin_strong_signals_required(false);
        assert!(requirements.raw.lsaMultimem);
        assert_eq!(requirements.raw.barrierCount, 3);
        assert_eq!(requirements.raw.ginContextCount, 8);
        assert_eq!(
            requirements.raw.ginConnectionType,
            sys::NCCL_GIN_CONNECTION_RAIL
        );
        assert!(!requirements.raw.ginStrongSignalsRequired);
        assert!(requirements.raw.ginVaSignalsRequired);
    }

    #[test]
    fn explicit_destroy_is_consuming_and_fallible() {
        let _destroy: fn(DeviceCommunicator<'static>) -> Result<()> = DeviceCommunicator::destroy;
    }
}
