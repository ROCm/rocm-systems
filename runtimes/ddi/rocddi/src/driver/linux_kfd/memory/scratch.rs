// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process VM scratch aperture reservation and range allocation.

use super::{DeviceVm, error, native_error, os_memory, sysfs};
use crate::memory::storage::TryPush;
use crate::{Error, ErrorKind};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

const SCRATCH_BASE_ALIGNMENT: usize = 64 * 1024;

/// One free or allocated interval in the process scratch aperture.
#[derive(Clone, Copy)]
pub(in crate::driver::linux_kfd) struct ScratchRange {
    /// Byte offset within the reserved scratch aperture.
    offset: u64,
    /// Byte extent of this interval.
    pub(in crate::driver::linux_kfd) size: u64,
    /// Whether a queue currently owns this interval.
    pub(in crate::driver::linux_kfd) allocated: bool,
}

/// Lazily reserved scratch aperture shared by queues on one device VM.
///
/// The reservation is retained if programming its KFD base has an ambiguous
/// outcome; reusing that virtual range could otherwise alias live GPU state.
pub(super) struct ScratchPool {
    /// Largest scratch aperture supported by this device.
    capacity: u64,
    /// Process VA reservation retained after native base programming.
    pub(in crate::driver::linux_kfd) reservation: Option<os_memory::Reservation>,
    /// Free and allocated intervals within the reservation.
    pub(in crate::driver::linux_kfd) ranges: Vec<ScratchRange>,
    /// Native base programming may have succeeded despite an error.
    uncertain: bool,
}

impl ScratchPool {
    pub(super) fn new(properties: sysfs::KfdQueueProperties) -> Self {
        let per_xcc =
            crate::topology::GpuInfo::scratch_bytes_per_xcc(properties.gfx_target / 10_000);
        let capacity = per_xcc
            .checked_mul(u64::from(properties.xcc_count))
            .filter(|capacity| *capacity != 0 && usize::try_from(*capacity).is_ok())
            .unwrap_or(0);
        Self {
            capacity,
            reservation: None,
            ranges: Vec::new(),
            uncertain: false,
        }
    }

    pub(super) fn initialize(&mut self, vm: &DeviceVm) -> Result<(), Error> {
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD scratch-base programming outcome is uncertain",
            ));
        }
        if self.reservation.is_some() {
            return Ok(());
        }
        if self.capacity == 0 {
            return Err(error(
                ErrorKind::Unsupported,
                "native target has no scratch aperture capacity",
            ));
        }
        self.ranges.try_reserve_exact(1)?;
        let reservation = os_memory::Reservation::new(
            usize::try_from(self.capacity).map_err(|_| {
                error(
                    ErrorKind::ResourceExhausted,
                    "scratch capacity exceeds the host address width",
                )
            })?,
            SCRATCH_BASE_ALIGNMENT,
            (
                vm.address_range().0,
                vm.address_range().1.min((1_u64 << 48) - 1),
            ),
            false,
        )
        .map_err(|source| native_error("scratch address reservation", source))?;
        let address = reservation.address() as u64;
        let result = vm.loss.kfd.set_scratch_backing_va(vm.gpu_id, address);
        self.reservation = Some(reservation);
        if let Err(source) = result {
            // The ioctl has no output that can prove whether a failing call
            // changed the process base. Retain the VA until process teardown.
            self.uncertain = true;
            return Err(native_error("AMDKFD_IOC_SET_SCRATCH_BACKING_VA", source));
        }
        self.ranges.try_push(ScratchRange {
            offset: 0,
            size: self.capacity,
            allocated: false,
        })?;
        Ok(())
    }

    pub(super) fn allocate(
        &mut self,
        vm: &DeviceVm,
        size: u64,
    ) -> Result<(u64, Arc<AtomicBool>), Error> {
        self.initialize(vm)?;
        let reservation = self
            .reservation
            .as_ref()
            .filter(|range| range.is_live())
            .ok_or_else(|| {
                error(
                    ErrorKind::DriverContract,
                    "scratch aperture ownership is uncertain",
                )
            })?;
        let index = self
            .ranges
            .iter()
            .position(|range| !range.allocated && range.size >= size)
            .ok_or_else(|| {
                error(
                    ErrorKind::ResourceExhausted,
                    "scratch aperture is exhausted",
                )
            })?;
        let range = self.ranges[index];
        if range.size != size {
            self.ranges.try_push(ScratchRange {
                offset: range.offset + size,
                size: range.size - size,
                allocated: false,
            })?;
        }
        self.ranges[index].size = size;
        self.ranges[index].allocated = true;
        let address = (reservation.address() as u64)
            .checked_add(range.offset)
            .ok_or_else(|| error(ErrorKind::Internal, "scratch address overflow"))?;
        Ok((address, reservation.poison_token()))
    }

    pub(super) fn release(&mut self, address: u64, size: u64) -> Result<(), Error> {
        let base = self
            .reservation
            .as_ref()
            .filter(|reservation| reservation.is_live())
            .map(|reservation| reservation.address() as u64)
            .ok_or_else(|| error(ErrorKind::Internal, "scratch aperture is not initialized"))?;
        let offset = address
            .checked_sub(base)
            .ok_or_else(|| error(ErrorKind::Internal, "scratch address precedes its aperture"))?;
        let range = self
            .ranges
            .iter_mut()
            .find(|range| range.allocated && range.offset == offset && range.size == size)
            .ok_or_else(|| error(ErrorKind::Internal, "scratch range is not allocated"))?;
        range.allocated = false;
        loop {
            let mut pair = None;
            for first in 0..self.ranges.len() {
                if self.ranges[first].allocated {
                    continue;
                }
                for second in first + 1..self.ranges.len() {
                    if self.ranges[second].allocated {
                        continue;
                    }
                    let first_end = self.ranges[first]
                        .offset
                        .checked_add(self.ranges[first].size);
                    let second_end = self.ranges[second]
                        .offset
                        .checked_add(self.ranges[second].size);
                    if first_end == Some(self.ranges[second].offset)
                        || second_end == Some(self.ranges[first].offset)
                    {
                        pair = Some((first, second));
                        break;
                    }
                }
                if pair.is_some() {
                    break;
                }
            }
            let Some((first, second)) = pair else {
                break;
            };
            let offset = self.ranges[first].offset.min(self.ranges[second].offset);
            let size = self.ranges[first]
                .size
                .checked_add(self.ranges[second].size)
                .ok_or_else(|| error(ErrorKind::Internal, "scratch range size overflow"))?;
            self.ranges[first] = ScratchRange {
                offset,
                size,
                allocated: false,
            };
            let last = self
                .ranges
                .pop()
                .ok_or_else(|| error(ErrorKind::Internal, "scratch range list is empty"))?;
            if second < self.ranges.len() {
                self.ranges[second] = last;
            }
        }
        Ok(())
    }
}
