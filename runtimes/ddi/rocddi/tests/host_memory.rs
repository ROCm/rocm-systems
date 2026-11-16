// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! The same host-backing contract runs without a GPU on every selected OS.

#![allow(unsafe_code, clippy::unwrap_used)]
use rocddi::memory::host::{HostAllocation, HostReservation, capabilities, host_page_size};

#[test]
fn replacement_preserves_neighbor_views_and_supports_a_short_tail() {
    let page = host_page_size().unwrap();
    let unit = page * 4;
    let size = unit * 2 + page;
    let mut source = HostAllocation::allocate_remappable(size + unit, page).unwrap();
    let mut target = HostReservation::reserve(size, page, unit).unwrap();
    assert_eq!(target.info().mapping_granularity, unit);
    let original = source.info().host_address as *mut u8;
    let alias = target.info().address as *mut u8;
    // SAFETY: The source owns this writable range and no aliases exist yet.
    unsafe { std::ptr::write_bytes(original, 0x3a, usize::try_from(size + unit).unwrap()) };
    target.replace(0, size, &source, 0, true).unwrap();
    target.replace(unit, unit, &source, 0, false).unwrap();
    // SAFETY: All views are live; the middle view is read-only. Raw accesses
    // avoid forming references over aliased backing. This test is single-threaded.
    unsafe {
        original.write_volatile(0x71);
        original
            .add(usize::try_from(2 * unit).unwrap())
            .write_volatile(0x52);
        assert_eq!(alias.read_volatile(), 0x71);
        assert_eq!(
            alias.add(usize::try_from(unit).unwrap()).read_volatile(),
            0x71
        );
        assert_eq!(
            alias
                .add(usize::try_from(2 * unit).unwrap())
                .read_volatile(),
            0x52
        );
    }
    target.clear(unit, unit).unwrap();
    target.replace(unit, unit, &source, unit, true).unwrap();
    // SAFETY: The middle view is writable again and the source remains live.
    unsafe {
        alias
            .add(usize::try_from(unit).unwrap())
            .write_volatile(0x19);
        assert_eq!(
            original.add(usize::try_from(unit).unwrap()).read_volatile(),
            0x19
        );
        assert_eq!(alias.read_volatile(), 0x71);
    }
    target.clear(0, size).unwrap();
    target.clear(0, size).unwrap();
    target.free().unwrap();
    assert!(target.replace(0, size, &source, 0, true).is_err());
    target.free().unwrap();
    source.free().unwrap();
}

#[test]
fn page_sized_views_accept_non_allocation_granularity_source_offsets() {
    let page = host_page_size().unwrap();
    let mut source = HostAllocation::allocate_remappable(3 * page, page).unwrap();
    let mut target = HostReservation::reserve(2 * page, page, page).unwrap();
    // SAFETY: The source owns at least three writable pages.
    unsafe {
        (source.info().host_address as *mut u8)
            .add(usize::try_from(page).unwrap())
            .write_volatile(0xa4);
    };
    target.replace(0, page, &source, page, false).unwrap();
    // SAFETY: The first target page is a live readable view of that source page.
    assert_eq!(
        unsafe { (target.info().address as *const u8).read_volatile() },
        0xa4
    );
    target.free().unwrap();
    source.free().unwrap();
}

#[test]
fn default_mapping_unit_is_reported_and_invalid_requests_acquire_no_owner() {
    let caps = capabilities().unwrap().reservation.unwrap();
    let page = caps.reservation_granularity;
    let mut target = HostReservation::reserve(page, page, 0).unwrap();
    assert!(target.info().mapping_granularity.is_power_of_two());
    assert!(target.info().mapping_granularity >= caps.mapping_granularity);
    target.free().unwrap();
    for unit in [1, page - 1, page + 1] {
        assert!(HostReservation::reserve(page, page, unit).is_err());
    }
}
