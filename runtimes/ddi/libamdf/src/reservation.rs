// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Stable host address reservations and explicit backing replacement.
//!
//! Native page-table ownership lives in rocddi. GPU reservation operations
//! remain unavailable until the native driver can replace subranges and report
//! checked removal; no queue advertises virtual-memory mutation today.

use crate::generated::amdf::*;
use crate::instance::{Device, Instance};
use crate::memory;
use crate::support::*;
use rocddi::memory::host;
use rocddi::memory::storage::Owned;
use std::cell::UnsafeCell;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) struct Reservation {
    pub(crate) instance: *mut Instance,
    /// Mutation and destruction are caller-serialized; queries read only the
    /// immutable fields outside this cell.
    mutation: UnsafeCell<ReservationMutation>,
    pub(crate) info: amdf_memory_reservation_access_info_t,
    pub(crate) children: AtomicU64,
}

/// Native state changed only by caller-serialized reservation operations.
struct ReservationMutation {
    /// Owned host VA reservation until terminal destruction.
    native: Option<host::HostReservation>,
    /// A partial native map failure prevents further replacements.
    poisoned: bool,
}

impl Reservation {
    /// Borrows mutation state after the caller excludes another mutation or
    /// destruction of this reservation. Immutable queries never access it.
    ///
    /// # Safety
    /// The caller serializes reservation mutation and destruction.
    #[allow(
        clippy::mut_from_ref,
        reason = "UnsafeCell holds caller-serialized native state beside lock-free immutable queries"
    )]
    unsafe fn mutation(&self) -> &mut ReservationMutation {
        // SAFETY: The C contract excludes another mutator, while the cell
        // permits this disjoint mutation beside immutable query borrows.
        unsafe { &mut *self.mutation.get() }
    }
}

fn capabilities(page: u64) -> amdf_memory_reservation_capabilities_t {
    amdf_memory_reservation_capabilities_t {
        supported_access: AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        required_access: AMDF_MEMORY_ACCESS_READ,
        address_domain_ordinal: AMDF_ADDRESS_DOMAIN_ORDINAL_NONE,
        maximum_byte_length: (isize::MAX as u64) & !(page - 1),
        reservation_granularity: page,
        maximum_alignment: 1_u64 << (usize::BITS - 2),
        mapping_granularity: page,
        minimum_address: page,
        maximum_address: usize::MAX as u64,
        required_source_flags: AMDF_MEMORY_FLAG_HOST_MAPPING_SOURCE,
        ..Default::default()
    }
}

pub(crate) unsafe extern "C" fn host_capabilities(
    pointer: *mut amdf_instance_t,
    out: *mut amdf_memory_reservation_capabilities_t,
) -> u64 {
    boundary(|| unsafe {
        let instance = object(pointer.cast::<Instance>())?;
        let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES)?;
        if instance.closing || instance.process_id != std::process::id() {
            return Err(PRECONDITION);
        }
        let page = host::host_page_size().map_err(|error| native(&error))?;
        out.publish(capabilities(page));
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn device_capabilities(
    pointer: *mut amdf_device_t,
    _domain: u32,
    out: *mut amdf_memory_reservation_capabilities_t,
) -> u64 {
    boundary(|| unsafe {
        let _device = object(pointer.cast::<Device>())?;
        let _out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES)?;
        Err(UNSUPPORTED)
    })
}

pub(crate) unsafe extern "C" fn create(
    pointer: *mut amdf_instance_t,
    info: *const amdf_memory_reservation_create_info_t,
    out: *mut *mut amdf_memory_reservation_t,
) -> u64 {
    boundary(|| unsafe {
        output_pointer(out)?;
        let instance = object(pointer.cast::<Instance>())?;
        if instance.closing || instance.process_id != std::process::id() {
            return Err(PRECONDITION);
        }
        let info = input(info, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO)?;
        if info.reserved != 0 || info.byte_length == 0 || info.access_count == 0 {
            return Err(INVALID);
        }
        let accesses = array(info.accesses, info.access_count)?;
        if accesses.len() != 1 || !accesses[0].device.is_null() {
            return Err(UNSUPPORTED);
        }
        let access = accesses[0];
        let page = host::host_page_size().map_err(|error| native(&error))?;
        let caps = capabilities(page);
        if access.address_domain_ordinal != AMDF_ADDRESS_DOMAIN_ORDINAL_NONE
            || access.access & AMDF_MEMORY_ACCESS_READ == 0
            || access.access & !caps.supported_access != 0
            || info.byte_length % page != 0
            || info.byte_length > caps.maximum_byte_length
            || (info.minimum_alignment != 0
                && (!info.minimum_alignment.is_power_of_two()
                    || info.minimum_alignment > caps.maximum_alignment))
        {
            return Err(INVALID);
        }
        let granularity = if access.mapping_granularity == 0 {
            page
        } else {
            access.mapping_granularity
        };
        if !granularity.is_power_of_two() || granularity < page {
            return Err(INVALID);
        }
        let slot = instance.owner_slot::<Reservation>()?;
        let native_owner =
            host::HostReservation::reserve(info.byte_length, info.minimum_alignment.max(page))
                .map_err(|error| native(&error))?;
        let address = u64::try_from(native_owner.info().address).map_err(|_| RANGE)?;
        register(&instance.children)?;
        let owner = slot.write(Reservation {
            instance: pointer.cast(),
            mutation: UnsafeCell::new(ReservationMutation {
                native: Some(native_owner),
                poisoned: false,
            }),
            info: amdf_memory_reservation_access_info_t {
                ordinal: 0,
                mapping_access_ordinal: 0,
                address_domain_ordinal: AMDF_ADDRESS_DOMAIN_ORDINAL_NONE,
                access: access.access,
                address_kind: AMDF_MEMORY_ADDRESS_HOST,
                address,
                byte_length: info.byte_length,
                mapping_granularity: granularity,
                ..Default::default()
            },
            children: AtomicU64::new(0),
        });
        out.write(owner.into_raw().cast());
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn query_access_info(
    pointer: *mut amdf_memory_reservation_t,
    ordinal: u32,
    out: *mut amdf_memory_reservation_access_info_t,
) -> u64 {
    boundary(|| unsafe {
        let reservation = object(pointer.cast::<Reservation>())?;
        let out = output(out, AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO)?;
        if ordinal != 0 {
            return Err(RANGE);
        }
        out.publish_from(&raw const reservation.info);
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn map(
    pointer: *mut amdf_memory_reservation_t,
    ordinal: u32,
    info: *const amdf_memory_map_info_t,
    out: *mut *mut amdf_host_mapping_t,
) -> u64 {
    boundary(|| unsafe {
        output_pointer(out)?;
        let reservation = object(pointer.cast::<Reservation>())?;
        let map = input(info, AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO)?;
        if ordinal != 0 {
            return Err(RANGE);
        }
        memory::map_reservation(pointer.cast(), reservation, map, out)
    })
}

fn validate_update(
    reservation: &Reservation,
    update: amdf_memory_mapping_update_t,
) -> Result<(), u64> {
    let granularity = reservation.info.mapping_granularity;
    if update.byte_length == 0
        || update.target_byte_offset % granularity != 0
        || update.target_byte_offset > reservation.info.byte_length
        || update.byte_length > reservation.info.byte_length - update.target_byte_offset
        || (update.byte_length % granularity != 0
            && update.byte_length != reservation.info.byte_length - update.target_byte_offset)
    {
        return Err(INVALID);
    }
    if update.memory.is_null() {
        if update.memory_access_ordinal != 0 || update.access != 0 || update.source_byte_offset != 0
        {
            return Err(INVALID);
        }
        return Ok(());
    }
    if update.memory_access_ordinal != 0
        || update.source_byte_offset % granularity != 0
        || update.access & AMDF_MEMORY_ACCESS_READ == 0
        || update.access & !reservation.info.access != 0
    {
        return Err(INVALID);
    }
    // SAFETY: The caller keeps this source live throughout the operation.
    let (source, length) =
        unsafe { memory::host_mapping_source(update.memory, reservation.instance)? };
    if !source.is_remappable()
        || update.source_byte_offset > length
        || update.byte_length > length - update.source_byte_offset
    {
        return Err(UNSUPPORTED);
    }
    Ok(())
}

pub(crate) unsafe extern "C" fn update(
    pointer: *mut amdf_memory_reservation_t,
    ordinal: u32,
    count: u32,
    updates: *const amdf_memory_mapping_update_t,
) -> u64 {
    boundary(|| unsafe {
        let reservation = object(pointer.cast::<Reservation>())?;
        if ordinal != 0 {
            return Err(RANGE);
        }
        // SAFETY: The caller serializes updates and destruction.
        let mutation = reservation.mutation();
        if mutation.poisoned {
            return Err(PRECONDITION);
        }
        let updates = array(updates, count)?;
        for (index, update) in updates.iter().enumerate() {
            validate_update(reservation, *update)?;
            let end = update.target_byte_offset + update.byte_length;
            if updates[..index].iter().any(|earlier| {
                earlier.target_byte_offset < end
                    && update.target_byte_offset < earlier.target_byte_offset + earlier.byte_length
            }) {
                return Err(INVALID);
            }
        }
        for update in updates {
            let result = if update.memory.is_null() {
                mutation
                    .native
                    .as_mut()
                    .ok_or(INTERNAL)?
                    .clear(update.target_byte_offset, update.byte_length)
            } else {
                let (source, _) = memory::host_mapping_source(update.memory, reservation.instance)?;
                mutation.native.as_mut().ok_or(INTERNAL)?.replace(
                    update.target_byte_offset,
                    update.byte_length,
                    source,
                    update.source_byte_offset,
                    update.access & AMDF_MEMORY_ACCESS_WRITE != 0,
                )
            };
            if let Err(error) = result {
                mutation.poisoned = true;
                return Err(native(&error));
            }
        }
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn detach(
    pointer: *mut amdf_memory_reservation_t,
    ordinal: u32,
) -> u64 {
    boundary(|| unsafe {
        let reservation = object(pointer.cast::<Reservation>())?;
        if ordinal != 0 {
            return Err(RANGE);
        }
        // SAFETY: The caller serializes detach and destruction.
        let mutation = reservation.mutation();
        let result = mutation
            .native
            .as_mut()
            .ok_or(INTERNAL)?
            .clear(0, reservation.info.byte_length);
        if let Err(error) = result {
            mutation.poisoned = true;
            return Err(native(&error));
        }
        Ok(())
    })
}

pub(crate) unsafe extern "C" fn destroy(pointer: *mut amdf_memory_reservation_t) -> u64 {
    boundary(|| unsafe {
        let reservation = object(pointer.cast::<Reservation>())?;
        if reservation.children.load(Ordering::Acquire) != 0 {
            return Err(BUSY);
        }
        // SAFETY: Destruction is exclusive to every reservation use.
        let mut native_owner = reservation.mutation().native.take().ok_or(INTERNAL)?;
        let result = native_owner.free().map_err(|error| native(&error));
        if result.is_err() {
            std::mem::forget(native_owner);
        }
        unregister(&(*reservation.instance).children);
        drop(Owned::from_raw(pointer.cast::<Reservation>()));
        result
    })
}

pub(crate) unsafe extern "C" fn queue_update(
    queue: *mut amdf_kernel_queue_t,
    _reservation: *mut amdf_memory_reservation_t,
    _ordinal: u32,
    _count: u32,
    _updates: *const amdf_memory_mapping_update_t,
    out: *mut u64,
) -> u64 {
    boundary(|| {
        output_pointer(out)?;
        output_pointer(queue)?;
        Err(UNSUPPORTED)
    })
}
