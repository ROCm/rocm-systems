// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Reuse of a live KFD BO's CPU mapping for host registration.
//!
//! KFD cannot create a second USERPTR BO for every BO-backed CPU VMA. The
//! connection records mapped owned BOs and hands out borrow owners only when
//! the requested range, GPU set, permissions, and cache policy are already
//! satisfied. The source cannot start native cleanup until all borrows end.

use super::{DeviceVm, KfdAllocation, error, host_cache_flags, native_error, os_memory, uapi};
use crate::driver::linux_kfd::DeviceState;
use crate::memory::device::registered_host_page_cover;
use crate::memory::host::HostBackingLease;
use crate::memory::storage::{Owned, Shared, TryPush};
use crate::memory::{AllocationInfo, DeviceAccess, HostMappingPolicy};
use crate::{Error, ErrorKind};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// Immutable source facts retained independently of the source allocation.
pub(super) struct HostAliasRecord {
    vm: Shared<DeviceVm>,
    host_address: usize,
    host_size: u64,
    device_address: u64,
    gpu_ids: Vec<u32>,
    native_flags: u32,
    access_permissions: DeviceAccess,
    physical_backing_id: [u64; 2],
    exported_backing_id: OnceLock<[u64; 2]>,
    frontend_backing_id: OnceLock<[u64; 2]>,
    aliases: AtomicUsize,
    accepting: AtomicBool,
}

impl HostAliasRecord {
    fn set_frontend_identity(&self, identity: [u64; 2]) -> Result<(), Error> {
        if identity == [0; 2] {
            return Err(error(
                ErrorKind::InvalidArgument,
                "host alias identity is zero",
            ));
        }
        if self
            .frontend_backing_id
            .get()
            .is_some_and(|known| *known == identity)
        {
            return Ok(());
        }
        self.frontend_backing_id.set(identity).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "host alias identity has already been set",
            )
        })
    }

    pub(super) fn note_export(&self, physical_id: [u64; 2]) {
        if physical_id != [0; 2] {
            let _ = self.exported_backing_id.set(physical_id);
        }
    }

    fn covers(
        &self,
        vm: &Shared<DeviceVm>,
        peers: &[&DeviceState],
        address: usize,
        size: usize,
        policy: HostMappingPolicy,
        permissions: DeviceAccess,
    ) -> bool {
        if !self.accepting.load(Ordering::Acquire) || !self.vm.shares_kfd(vm) {
            return false;
        }
        let Some(offset) = address.checked_sub(self.host_address) else {
            return false;
        };
        let (Ok(offset), Ok(size)) = (u64::try_from(offset), u64::try_from(size)) else {
            return false;
        };
        if self.device_address.checked_add(offset).is_none()
            || offset
                .checked_add(size)
                .is_none_or(|end| end > self.host_size)
        {
            return false;
        }
        if !std::iter::once(vm)
            .chain(peers.iter().map(|peer| &peer.vm))
            .all(|requested| {
                self.vm.shares_kfd(requested) && self.gpu_ids.contains(&requested.gpu_id)
            })
        {
            return false;
        }
        let requested_flags = host_cache_flags(policy);
        permissions.contains(DeviceAccess::READ)
            && self.access_permissions.contains(permissions)
            && self.native_flags & requested_flags == requested_flags
    }
}

/// Borrowed registration of a source BO's already established mapping.
pub(crate) struct KfdHostAlias {
    source: Shared<HostAliasRecord>,
    requested_vms: Vec<Shared<DeviceVm>>,
    origin_gpu_id: u32,
    address: usize,
    size: u64,
    backing: Option<HostBackingLease>,
    active: bool,
}

impl KfdHostAlias {
    pub(crate) fn try_create(
        vm: &Shared<DeviceVm>,
        peers: &[&DeviceState],
        address: usize,
        size: u64,
        policy: HostMappingPolicy,
        permissions: DeviceAccess,
        backing: &mut Option<HostBackingLease>,
    ) -> Result<Option<Owned<Self>>, Error> {
        let page =
            os_memory::page_size().map_err(|source| native_error("native page size", source))?;
        let size_usize = usize::try_from(size).map_err(|_| {
            error(
                ErrorKind::InvalidArgument,
                "host range exceeds address width",
            )
        })?;
        let (_, cover) = registered_host_page_cover(address, size_usize, page)
            .ok_or_else(|| error(ErrorKind::InvalidArgument, "invalid host page cover"))?;
        let _native_size = u64::try_from(cover)
            .map_err(|_| error(ErrorKind::InvalidArgument, "host page cover exceeds u64"))?;
        let mut requested_vms = crate::memory::storage::try_vec_with_capacity(
            peers.len().checked_add(1).ok_or_else(|| {
                error(
                    ErrorKind::ResourceExhausted,
                    "host alias peer count overflows",
                )
            })?,
        )?;
        requested_vms.try_push(vm.clone())?;
        for peer in peers {
            requested_vms.try_push(peer.vm.clone())?;
        }
        for requested in &requested_vms {
            requested.check()?;
        }
        let owner = Owned::<Self>::try_new_uninit()?;
        let registry = vm
            .loss
            .host_aliases
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "KFD host alias registry lock poisoned"))?;
        let Some(source) = registry
            .iter()
            .find(|source| source.covers(vm, peers, address, size_usize, policy, permissions))
            .cloned()
        else {
            return Ok(None);
        };
        source
            .aliases
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |count| {
                count.checked_add(1)
            })
            .map_err(|_| error(ErrorKind::ResourceExhausted, "host alias count overflows"))?;
        drop(registry);
        Ok(Some(owner.write(Self {
            source,
            requested_vms,
            origin_gpu_id: vm.gpu_id,
            address,
            size,
            backing: backing.take(),
            active: true,
        })))
    }

    pub(crate) fn cached_info(&self) -> AllocationInfo {
        let offset = (self.address - self.source.host_address) as u64;
        AllocationInfo {
            device_address: self.source.device_address + offset,
            host_address: Some(self.address),
            size: self.size,
            native_size: self.source.host_size,
            physical_backing_id: self
                .source
                .exported_backing_id
                .get()
                .copied()
                .or_else(|| self.source.frontend_backing_id.get().copied())
                .unwrap_or(self.source.physical_backing_id),
        }
    }

    pub(crate) fn source_offset(&self) -> u64 {
        (self.address - self.source.host_address) as u64
    }

    pub(crate) fn check(&self) -> Result<(), Error> {
        if !self.active {
            return Err(error(ErrorKind::Unsupported, "host alias was released"));
        }
        self.source.vm.check()?;
        for vm in &self.requested_vms {
            vm.check()?;
        }
        Ok(())
    }

    pub(crate) fn device_address(&self, device: &Shared<DeviceVm>) -> Result<u64, Error> {
        self.check()?;
        if !self.source.vm.shares_kfd(device)
            || !self
                .requested_vms
                .iter()
                .any(|requested| requested.gpu_id == device.gpu_id)
        {
            return Err(error(
                ErrorKind::Unsupported,
                "host alias is not mapped to this device",
            ));
        }
        Ok(self.cached_info().device_address)
    }

    pub(crate) fn is_owned_by(&self, device: &Shared<DeviceVm>) -> bool {
        self.source.vm.shares_kfd(device) && self.origin_gpu_id == device.gpu_id
    }

    pub(crate) fn free(&mut self) {
        if self.active {
            self.active = false;
            self.source.aliases.fetch_sub(1, Ordering::Release);
            self.backing = None;
        }
    }
}

impl Drop for KfdHostAlias {
    fn drop(&mut self) {
        self.free();
    }
}

impl KfdAllocation {
    pub(crate) fn set_host_alias_identity(&self, identity: [u64; 2]) -> Result<(), Error> {
        self.vm
            .loss
            .kfd
            .check_process()
            .map_err(|source| native_error("host alias identity process check", source))?;
        let source = self.host_alias_record.as_ref().ok_or_else(|| {
            error(
                ErrorKind::Unsupported,
                "allocation has no host alias record",
            )
        })?;
        let _registry = self
            .vm
            .loss
            .host_aliases
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "KFD host alias registry lock poisoned"))?;
        if source.aliases.load(Ordering::Acquire) != 0 {
            return Err(error(
                ErrorKind::Busy,
                "host alias identity is already observable",
            ));
        }
        source.set_frontend_identity(identity)?;
        let _ = self.host_alias_frontend_id.set(identity);
        Ok(())
    }

    pub(super) fn publish_host_alias(&mut self) -> Result<(), Error> {
        if self.host_address.is_none() {
            return Ok(());
        }
        if self.handle.is_none() || self.native_flags & (uapi::GTT | uapi::VRAM) == 0 {
            return Ok(());
        }
        let reservation = self.reservation.as_ref().ok_or_else(|| {
            error(
                ErrorKind::Internal,
                "mapped host BO lost its VA reservation",
            )
        })?;
        let mut gpu_ids = crate::memory::storage::try_vec_with_capacity(self.gpu_ids.len())?;
        for gpu_id in &self.gpu_ids {
            gpu_ids.try_push(*gpu_id)?;
        }
        let source = Shared::new(HostAliasRecord {
            vm: self.vm.clone(),
            host_address: reservation.address(),
            host_size: reservation.usable_size() as u64,
            device_address: reservation.address() as u64,
            gpu_ids,
            native_flags: self.native_flags,
            access_permissions: self.access_permissions,
            physical_backing_id: self.physical_backing_id,
            exported_backing_id: OnceLock::new(),
            frontend_backing_id: OnceLock::new(),
            aliases: AtomicUsize::new(0),
            accepting: AtomicBool::new(true),
        })?;
        if let Some(identity) = self.host_alias_frontend_id.get() {
            source.set_frontend_identity(*identity)?;
        }
        if let Some(identity) = self.host_alias_exported_id.get() {
            source.note_export(*identity);
        }
        let mut registry = self
            .vm
            .loss
            .host_aliases
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "KFD host alias registry lock poisoned"))?;
        registry.try_push(source.clone())?;
        self.host_alias_record = Some(source);
        Ok(())
    }

    pub(super) fn begin_host_alias_cleanup(&mut self) -> Result<(), Error> {
        let Some(source) = self.host_alias_record.as_ref() else {
            return Ok(());
        };
        let mut registry = self
            .vm
            .loss
            .host_aliases
            .lock()
            .map_err(|_| error(ErrorKind::Internal, "KFD host alias registry lock poisoned"))?;
        if source.aliases.load(Ordering::Acquire) != 0 {
            return Err(error(
                ErrorKind::Busy,
                "BO-backed host registrations are still live",
            ));
        }
        source.accepting.store(false, Ordering::Release);
        registry.retain(|entry| !Shared::ptr_eq(entry, source));
        drop(registry);
        self.host_alias_record = None;
        Ok(())
    }

    pub(super) fn abandon_host_alias(&mut self) {
        let Some(source) = self.host_alias_record.take() else {
            return;
        };
        // An inherited mutex may have been held by another parent thread at
        // fork. The inherited KFD connection is unusable in this process, so
        // leave its registry to process teardown without taking that lock.
        if self.vm.loss.kfd.check_process().is_err() {
            source.accepting.store(false, Ordering::Release);
            return;
        }
        if let Ok(mut registry) = self.vm.loss.host_aliases.lock() {
            source.accepting.store(false, Ordering::Release);
            registry.retain(|entry| !Shared::ptr_eq(entry, &source));
        }
    }
}
