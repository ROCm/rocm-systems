// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process-context host registration through the same SVM ranges as advice.
//!
//! KFD rejects a USERPTR BO once advice has established SVM state at its host
//! address. SVM registrations instead grant `ACCESS_IN_PLACE` at the caller's VA.
//! The process path selects this owner only after a definite no-object USERPTR
//! collision, retaining USERPTR for fresh pages and true I/O-memory mappings.
//! The connection registry serializes grants and revocation, preserving pages
//! and GPUs covered by another registration. Page-global cache/permission
//! policies must agree where live registrations overlap. Explicit advice may
//! subsequently change policy; unregister only revokes this registration's
//! GPUs on uncovered pages and leaves scalar advice and other GPUs untouched.

use super::{DeviceVm, error, native_error, os_memory, uapi};
use crate::memory::device::registered_host_page_cover;
use crate::memory::host::HostBackingLease;
use crate::memory::storage::{Owned, Shared, TryPush, try_vec_with_capacity};
use crate::memory::{AllocationDesc, AllocationInfo, DeviceAccess, HostMappingPolicy};
use crate::{Error, ErrorKind};

/// One registration's page cover and native access grants.
struct Registration {
    id: u64,
    start: u64,
    end: u64,
    flags: u32,
    gpu_ids: Vec<u32>,
}

/// Connection-wide ownership, including registrations whose cleanup failed.
#[derive(Default)]
pub(super) struct Registry {
    registrations: Vec<Registration>,
    next_id: u64,
}

impl Registry {
    fn insert(&mut self, mut registration: Registration) -> Result<u64, Error> {
        if self.registrations.iter().any(|old| {
            old.start < registration.end
                && registration.start < old.end
                && old.flags != registration.flags
        }) {
            return Err(error(
                ErrorKind::Unsupported,
                "overlapping SVM registrations require one page cache and permission policy",
            ));
        }
        let id = self.next_id.checked_add(1).ok_or_else(|| {
            error(
                ErrorKind::ResourceExhausted,
                "SVM registration identity exhausted",
            )
        })?;
        self.registrations.try_reserve(1)?;
        registration.id = id;
        self.registrations.try_push(registration)?;
        self.next_id = id;
        Ok(id)
    }

    /// Recomputes uncovered gaps on retry: a later registration can have
    /// acquired some pages after a previous revocation attempt failed.
    fn remove(&mut self, vm: &DeviceVm, id: u64) -> Result<(), Error> {
        let Some(registration) = self.registrations.iter().find(|entry| entry.id == id) else {
            return Ok(());
        };
        let mut covered = try_vec_with_capacity(self.registrations.len())?;
        for &gpu in &registration.gpu_ids {
            covered.clear();
            for other in &self.registrations {
                if other.id != id && other.gpu_ids.contains(&gpu) {
                    let start = other.start.max(registration.start);
                    let end = other.end.min(registration.end);
                    if start < end {
                        covered.try_push((start, end))?;
                    }
                }
            }
            covered.sort_unstable();
            let mut cursor = registration.start;
            for &(start, end) in &covered {
                if cursor < start {
                    revoke(vm, cursor, start - cursor, gpu)?;
                }
                cursor = cursor.max(end);
            }
            if cursor < registration.end {
                revoke(vm, cursor, registration.end - cursor, gpu)?;
            }
        }
        self.registrations.retain(|entry| entry.id != id);
        Ok(())
    }
}

fn revoke(vm: &DeviceVm, start: u64, size: u64, gpu: u32) -> Result<(), Error> {
    vm.kfd()
        .svm_attributes(
            start,
            size,
            uapi::SVM_OP_SET_ATTR,
            &mut [uapi::SvmAttribute {
                attribute_type: uapi::SVM_ATTR_NO_ACCESS,
                value: gpu,
            }],
        )
        .map_err(|source| native_error("SVM host registration revoke", source))
}

/// Desired replacement retained through partial grant or revocation failure.
struct AccessChange {
    id: u64,
    devices: Vec<Shared<DeviceVm>>,
    granted: bool,
}

/// Caller pages retained until all owned SVM grants have been revoked.
pub(crate) struct KfdRegisteredHost {
    owner: Shared<DeviceVm>,
    devices: Vec<Shared<DeviceVm>>,
    host_address: usize,
    size: u64,
    start: u64,
    native_size: u64,
    flags: u32,
    id: Option<u64>,
    access_change: Option<AccessChange>,
    backing: Option<HostBackingLease>,
    freeing: bool,
}

impl KfdRegisteredHost {
    pub(crate) fn create(
        owner: Shared<DeviceVm>,
        peers: impl ExactSizeIterator<Item = Shared<DeviceVm>>,
        desc: AllocationDesc,
        host_address: usize,
        policy: HostMappingPolicy,
        permissions: DeviceAccess,
        backing: Option<HostBackingLease>,
    ) -> Result<Owned<Self>, Error> {
        if !permissions.contains(DeviceAccess::READ) || policy == HostMappingPolicy::Uncached {
            return Err(error(
                ErrorKind::Unsupported,
                "unsupported SVM host mapping policy",
            ));
        }
        let page = os_memory::page_size()
            .map_err(|source| native_error("SVM host registration page size", source))?;
        let size = usize::try_from(desc.size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "host extent exceeds address width",
            )
        })?;
        let (start, native_size) = registered_host_page_cover(host_address, size, page)
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "invalid SVM host page cover"))?;
        if desc.alignment == 0 || (start as u64) % desc.alignment != 0 {
            return Err(error(
                ErrorKind::Unsupported,
                "SVM host address does not meet alignment",
            ));
        }
        let mut devices = try_vec_with_capacity(peers.len().checked_add(1).ok_or_else(|| {
            error(
                ErrorKind::ResourceExhausted,
                "SVM host peer count overflows",
            )
        })?)?;
        devices.try_push(owner.clone())?;
        for peer in peers {
            if !devices.iter().any(|device| Shared::ptr_eq(device, &peer)) {
                devices.try_push(peer)?;
            }
        }
        validate_devices(&owner, &devices, start as u64, native_size as u64)?;
        let mut flags = uapi::SVM_FLAG_HOST_ACCESS;
        if policy != HostMappingPolicy::Coarse {
            flags |= uapi::SVM_FLAG_COHERENT;
        }
        if policy == HostMappingPolicy::Extended {
            flags |= uapi::SVM_FLAG_EXT_COHERENT;
        }
        if !permissions.contains(DeviceAccess::WRITE) {
            flags |= uapi::SVM_FLAG_GPU_READ_ONLY;
        }
        if permissions.contains(DeviceAccess::EXECUTE) {
            flags |= uapi::SVM_FLAG_GPU_EXECUTE;
        }
        let mut allocation = Owned::new(Self {
            owner,
            devices,
            host_address,
            size: desc.size,
            start: start as u64,
            native_size: native_size as u64,
            flags,
            id: None,
            access_change: None,
            backing,
            freeing: false,
        })?;
        let registration = allocation.registration(&allocation.devices)?;
        let attributes = allocation.attributes(&allocation.devices, true)?;
        let owner = allocation.owner.clone();
        let result = {
            let mut registry = owner.loss.svm_registrations.lock().map_err(|_| {
                error(
                    ErrorKind::Internal,
                    "SVM registration registry lock poisoned",
                )
            })?;
            let id = registry.insert(registration)?;
            allocation.id = Some(id);
            allocation.grant(attributes)
        };
        // Drop runs only after the registry guard is released. Even a failed
        // ioctl may have granted access, so retain the published cleanup owner.
        result?;
        Ok(allocation)
    }

    fn registration(&self, devices: &[Shared<DeviceVm>]) -> Result<Registration, Error> {
        let mut gpu_ids = try_vec_with_capacity(devices.len())?;
        for device in devices {
            gpu_ids.try_push(device.gpu_id)?;
        }
        Ok(Registration {
            id: 0,
            start: self.start,
            end: self.start + self.native_size,
            flags: self.flags,
            gpu_ids,
        })
    }

    fn attributes(
        &self,
        devices: &[Shared<DeviceVm>],
        policy: bool,
    ) -> Result<Vec<uapi::SvmAttribute>, Error> {
        let mut attributes = try_vec_with_capacity(devices.len() + 2)?;
        if policy {
            let mask = uapi::SVM_FLAG_HOST_ACCESS
                | uapi::SVM_FLAG_COHERENT
                | uapi::SVM_FLAG_EXT_COHERENT
                | uapi::SVM_FLAG_GPU_READ_ONLY
                | uapi::SVM_FLAG_GPU_EXECUTE;
            attributes.try_push(uapi::SvmAttribute {
                attribute_type: uapi::SVM_ATTR_SET_FLAGS,
                value: self.flags,
            })?;
            attributes.try_push(uapi::SvmAttribute {
                attribute_type: uapi::SVM_ATTR_CLEAR_FLAGS,
                value: mask & !self.flags,
            })?;
        }
        for device in devices {
            attributes.try_push(uapi::SvmAttribute {
                attribute_type: uapi::SVM_ATTR_ACCESS_IN_PLACE,
                value: device.gpu_id,
            })?;
        }
        Ok(attributes)
    }

    fn grant(&self, mut attributes: Vec<uapi::SvmAttribute>) -> Result<(), Error> {
        if attributes.is_empty() {
            return Ok(());
        }
        self.owner
            .kfd()
            .svm_attributes(
                self.start,
                self.native_size,
                uapi::SVM_OP_SET_ATTR,
                &mut attributes,
            )
            .map_err(|source| native_error("SVM host registration grant", source))
    }

    pub(crate) fn set_access(&mut self, devices: &[&Shared<DeviceVm>]) -> Result<(), Error> {
        self.owner.check()?;
        if self.freeing || self.id.is_none() {
            return Err(error(
                ErrorKind::Unsupported,
                "SVM registration is retiring",
            ));
        }
        let mut desired = try_vec_with_capacity(devices.len())?;
        for device in devices {
            if !desired.iter().any(|old| Shared::ptr_eq(old, device)) {
                desired.try_push((*device).clone())?;
            }
        }
        validate_devices(&self.owner, &desired, self.start, self.native_size)?;
        if let Some(change) = &self.access_change {
            if change.devices.len() != desired.len()
                || !change
                    .devices
                    .iter()
                    .all(|old| desired.iter().any(|device| Shared::ptr_eq(old, device)))
            {
                return Err(error(
                    ErrorKind::Busy,
                    "retry the pending SVM access set first",
                ));
            }
        }
        let registration = self.registration(&desired)?;
        let attributes = self.attributes(&desired, false)?;
        let mut registry = self.owner.loss.svm_registrations.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "SVM registration registry lock poisoned",
            )
        })?;
        if self.access_change.is_none() {
            let id = registry.insert(registration)?;
            self.access_change = Some(AccessChange {
                id,
                devices: desired,
                granted: false,
            });
        }
        if self
            .access_change
            .as_ref()
            .is_some_and(|change| !change.granted)
        {
            self.grant(attributes)?;
            if let Some(change) = self.access_change.as_mut() {
                change.granted = true;
            }
        }
        if let Some(id) = self.id {
            registry.remove(&self.owner, id)?;
        }
        if let Some(change) = self.access_change.take() {
            self.id = Some(change.id);
            self.devices = change.devices;
        }
        Ok(())
    }

    pub(crate) fn cached_info(&self) -> AllocationInfo {
        AllocationInfo {
            device_address: if self.id.is_some() {
                self.host_address as u64
            } else {
                0
            },
            host_address: Some(self.host_address),
            size: self.size,
            native_size: self.native_size,
            physical_backing_id: [0; 2],
        }
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        self.owner.check()?;
        if self.freeing || self.id.is_none() || self.access_change.is_some() {
            return Err(error(
                ErrorKind::Unsupported,
                "SVM registration is unavailable",
            ));
        }
        Ok(())
    }

    pub(crate) fn device_address(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        self.check()?;
        if !self
            .devices
            .iter()
            .any(|mapped| Shared::ptr_eq(mapped, device))
        {
            return Err(error(
                ErrorKind::Unsupported,
                "SVM registration is not mapped to device",
            ));
        }
        device.check()?;
        Ok(self.host_address as u64)
    }

    pub(crate) fn is_owned_by(&self, device: &Shared<DeviceVm>) -> bool {
        self.owner.shares_kfd(device) && self.owner.gpu_id == device.gpu_id
    }

    #[allow(unsafe_code)]
    pub(crate) fn diagnostic_memory(
        &self,
    ) -> Result<crate::device::gpu::diagnostic::DiagnosticMemory<'_>, Error> {
        self.check()?;
        // SAFETY: This owner retains the registered pages and native grants.
        unsafe {
            crate::device::gpu::diagnostic::DiagnosticMemory::borrow(
                self,
                self.host_address as u64,
                Some(self.host_address),
                self.size,
            )
        }
    }

    pub(crate) fn free(&mut self) -> Result<(), Error> {
        self.owner
            .kfd()
            .check_process()
            .map_err(|source| native_error("SVM registration process check", source))?;
        self.freeing = true;
        let mut registry = self.owner.loss.svm_registrations.lock().map_err(|_| {
            error(
                ErrorKind::Internal,
                "SVM registration registry lock poisoned",
            )
        })?;
        if let Some(change) = self.access_change.as_ref() {
            registry.remove(&self.owner, change.id)?;
            self.access_change = None;
        }
        if let Some(id) = self.id {
            registry.remove(&self.owner, id)?;
            self.id = None;
        }
        self.backing = None;
        Ok(())
    }
}

fn validate_devices(
    owner: &Shared<DeviceVm>,
    devices: &[Shared<DeviceVm>],
    start: u64,
    size: u64,
) -> Result<(), Error> {
    owner.check()?;
    for (index, device) in devices.iter().enumerate() {
        device.check()?;
        let (base, limit) = device.address_range();
        if !owner.shares_kfd(device)
            || start < base
            || start + size - 1 > limit
            || devices[..index]
                .iter()
                .any(|old| old.gpu_id == device.gpu_id)
        {
            return Err(error(
                ErrorKind::Unsupported,
                "SVM host range or device set is unavailable",
            ));
        }
    }
    Ok(())
}

impl Drop for KfdRegisteredHost {
    fn drop(&mut self) {
        if let Err(error) = self.free() {
            crate::memory::device::record_rollback(error, 0);
            if let Some(backing) = self.backing.take() {
                std::mem::forget(backing);
            }
            std::mem::forget(self.owner.clone());
            for device in &self.devices {
                std::mem::forget(device.clone());
            }
            if let Some(change) = &self.access_change {
                for device in &change.devices {
                    std::mem::forget(device.clone());
                }
            }
        }
    }
}
