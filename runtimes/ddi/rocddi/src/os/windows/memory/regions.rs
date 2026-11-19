// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Placeholder/view ownership independent of Win32 call encoding.
//!
//! Each view covers exactly one selected replacement unit. Only placeholders
//! may be split. The record set is preallocated before native VA acquisition;
//! mutations never retain backing objects or allocate more metadata.

use std::io;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Placeholder,
    View,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Region {
    offset: usize,
    length: usize,
    kind: Kind,
}

/// Native operations preserve their input region on failure. A failed map
/// leaves a placeholder, and a failed unmap leaves the complete original view.
pub(super) trait Operations {
    fn split(&mut self, offset: usize, prefix: usize) -> io::Result<()>;
    fn map(&mut self, offset: usize, length: usize, source: u64, writable: bool) -> io::Result<()>;
    fn unmap(&mut self, offset: usize, preserve: bool) -> io::Result<()>;
    fn release_placeholder(&mut self, offset: usize) -> io::Result<()>;
}

pub(super) struct Regions {
    records: Vec<Region>,
    size: usize,
    unit: usize,
    poisoned: bool,
    releasing: bool,
}

impl Regions {
    pub(super) fn new(size: usize, unit: usize) -> io::Result<Self> {
        if size == 0 || !unit.is_power_of_two() {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let mut records = Vec::new();
        records
            .try_reserve_exact((size - 1) / unit + 1)
            .map_err(|_| io::Error::from(io::ErrorKind::OutOfMemory))?;
        records.push(Region {
            offset: 0,
            length: size,
            kind: Kind::Placeholder,
        });
        Ok(Self {
            records,
            size,
            unit,
            poisoned: false,
            releasing: false,
        })
    }

    pub(super) fn unit(&self) -> usize {
        self.unit
    }

    pub(super) fn validate(&self, offset: u64, length: u64) -> io::Result<(usize, usize)> {
        use crate::memory::constraints::{CheckedRange, RangeTail};
        if self.releasing
            || CheckedRange::new(
                self.size as u64,
                offset,
                length,
                self.unit as u64,
                RangeTail::EndOfReservation,
            )
            .is_none()
        {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        Ok((
            usize::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?,
            usize::try_from(length).map_err(|_| io::ErrorKind::InvalidInput)?,
        ))
    }

    /// Finds or creates a boundary without changing a mapped view.
    fn split_at(&mut self, at: usize, ops: &mut impl Operations) -> io::Result<usize> {
        let index = self
            .records
            .iter()
            .position(|region| at >= region.offset && at < region.offset + region.length)
            .ok_or(io::ErrorKind::InvalidInput)?;
        self.split_record(index, at, ops)
    }

    fn split_record(
        &mut self,
        index: usize,
        at: usize,
        ops: &mut impl Operations,
    ) -> io::Result<usize> {
        let region = self.records[index];
        if at == region.offset {
            return Ok(index);
        }
        if region.kind != Kind::Placeholder {
            return Err(io::ErrorKind::InvalidInput.into());
        }
        let prefix = at - region.offset;
        ops.split(region.offset, prefix)?;
        self.records[index].length = prefix;
        self.records.insert(
            index + 1,
            Region {
                offset: at,
                length: region.length - prefix,
                kind: Kind::Placeholder,
            },
        );
        Ok(index + 1)
    }

    pub(super) fn replace(
        &mut self,
        offset: usize,
        length: usize,
        source: u64,
        writable: bool,
        ops: &mut impl Operations,
    ) -> io::Result<()> {
        self.validate(offset as u64, length as u64)?;
        if self.poisoned {
            return Err(io::Error::other("reservation is poisoned"));
        }
        source
            .checked_add(length as u64)
            .ok_or(io::ErrorKind::InvalidInput)?;
        let result = self.replace_units(offset, length, source, writable, ops);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }

    fn replace_units(
        &mut self,
        offset: usize,
        length: usize,
        source: u64,
        writable: bool,
        ops: &mut impl Operations,
    ) -> io::Result<()> {
        let end = offset + length;
        let mut at = offset;
        let mut index = self.split_at(at, ops)?;
        while at < end {
            let bytes = self.unit.min(self.size - at);
            let region = self.records[index];
            if region.kind == Kind::View {
                ops.unmap(at, true)?;
                self.records[index].kind = Kind::Placeholder;
            }
            if region.length > bytes {
                self.split_record(index, at + bytes, ops)?;
            }
            ops.map(at, bytes, source + (at - offset) as u64, writable)?;
            self.records[index].kind = Kind::View;
            at += bytes;
            index += 1;
        }
        Ok(())
    }

    /// Full detach remains available after poison; poison is never cleared.
    pub(super) fn clear(
        &mut self,
        offset: usize,
        length: usize,
        ops: &mut impl Operations,
    ) -> io::Result<()> {
        self.validate(offset as u64, length as u64)?;
        if self.poisoned && (offset != 0 || length != self.size) {
            return Err(io::Error::other("reservation is poisoned"));
        }
        let mut first = None;
        for region in &mut self.records {
            if region.kind == Kind::View
                && region.offset >= offset
                && region.offset < offset + length
            {
                match ops.unmap(region.offset, true) {
                    Ok(()) => region.kind = Kind::Placeholder,
                    Err(error) => {
                        if first.is_none() {
                            first = Some(error);
                        }
                    }
                }
            }
        }
        if let Some(error) = first {
            self.poisoned = true;
            Err(error)
        } else {
            Ok(())
        }
    }

    /// Terminal release retries only still-owned regions. Released holes may
    /// immediately belong to somebody else and must never be touched again.
    pub(super) fn free(&mut self, ops: &mut impl Operations) -> io::Result<()> {
        self.releasing = true;
        let mut first = None;
        self.records.retain(|region| {
            let result = match region.kind {
                Kind::View => ops.unmap(region.offset, false),
                Kind::Placeholder => ops.release_placeholder(region.offset),
            };
            match result {
                Ok(()) => false,
                Err(error) => {
                    if first.is_none() {
                        first = Some(error);
                    }
                    true
                }
            }
        });
        first.map_or(Ok(()), Err)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Tracks exact native regions and poisons released holes against reuse.
    struct Native {
        records: BTreeMap<usize, (usize, Kind)>,
        views: BTreeMap<usize, (u64, bool)>,
        calls: usize,
        fail: Option<usize>,
    }
    impl Native {
        fn new(size: usize) -> Self {
            Self {
                records: BTreeMap::from([(0, (size, Kind::Placeholder))]),
                views: BTreeMap::new(),
                calls: 0,
                fail: None,
            }
        }
        fn enter(&mut self) -> io::Result<()> {
            self.calls += 1;
            if self.fail == Some(self.calls) {
                Err(io::Error::other("injected native failure"))
            } else {
                Ok(())
            }
        }
        fn assert_owned(&self, owner: &Regions) {
            assert_eq!(
                self.records,
                owner
                    .records
                    .iter()
                    .map(|r| (r.offset, (r.length, r.kind)))
                    .collect()
            );
            assert!(owner.records.len() <= (owner.size - 1) / owner.unit + 1);
        }
    }
    impl Operations for Native {
        fn split(&mut self, offset: usize, prefix: usize) -> io::Result<()> {
            self.enter()?;
            let (length, kind) = *self.records.get(&offset).unwrap();
            assert_eq!(kind, Kind::Placeholder);
            assert!(prefix > 0 && prefix < length);
            self.records.insert(offset, (prefix, kind));
            self.records
                .insert(offset + prefix, (length - prefix, kind));
            Ok(())
        }
        fn map(
            &mut self,
            offset: usize,
            length: usize,
            source: u64,
            writable: bool,
        ) -> io::Result<()> {
            self.enter()?;
            assert_eq!(
                self.records.insert(offset, (length, Kind::View)),
                Some((length, Kind::Placeholder))
            );
            self.views.insert(offset, (source, writable));
            Ok(())
        }
        fn unmap(&mut self, offset: usize, preserve: bool) -> io::Result<()> {
            self.enter()?;
            let (length, kind) = self.records.remove(&offset).unwrap();
            assert_eq!(kind, Kind::View);
            assert!(self.views.remove(&offset).is_some());
            if preserve {
                self.records.insert(offset, (length, Kind::Placeholder));
            }
            Ok(())
        }
        fn release_placeholder(&mut self, offset: usize) -> io::Result<()> {
            self.enter()?;
            assert_eq!(self.records.remove(&offset).unwrap().1, Kind::Placeholder);
            Ok(())
        }
    }

    #[test]
    fn independent_views_preserve_neighbors_and_short_final_unit() {
        let size = 9 * 4096;
        let mut regions = Regions::new(size, 16384).unwrap();
        let mut native = Native::new(size);
        assert_eq!(regions.unit(), 16384);
        regions.replace(0, size, 4096, true, &mut native).unwrap();
        regions
            .replace(16384, 16384, 0, false, &mut native)
            .unwrap();
        assert_eq!(
            native.views,
            BTreeMap::from([
                (0, (4096, true)),
                (16384, (0, false)),
                (32768, (36864, true))
            ])
        );
        native.assert_owned(&regions);
        assert!(regions.validate(4096, 4096).is_err());
        regions.clear(0, size, &mut native).unwrap();
        regions.clear(0, size, &mut native).unwrap();
        regions.free(&mut native).unwrap();
        regions.free(&mut native).unwrap();
        assert!(native.records.is_empty());
    }

    #[test]
    fn every_failed_replace_prefix_retains_exact_native_ownership() {
        // Replacement of two middle units exercises prefix/suffix splitting,
        // replacement of an existing view, and mapping failure after unmap.
        for fail in 1..=4 {
            let mut regions = Regions::new(16384, 4096).unwrap();
            let mut native = Native::new(16384);
            regions.replace(8192, 4096, 0, true, &mut native).unwrap();
            native.fail = Some(native.calls + fail);
            assert!(
                regions
                    .replace(4096, 8192, 4096, false, &mut native)
                    .is_err()
            );
            native.assert_owned(&regions);
            assert!(regions.replace(0, 4096, 0, true, &mut native).is_err());
            native.fail = None;
            regions.clear(0, 16384, &mut native).unwrap();
            assert!(regions.replace(0, 4096, 0, true, &mut native).is_err());
            regions.free(&mut native).unwrap();
            assert!(native.records.is_empty());
        }
    }

    #[test]
    fn partial_release_never_releases_a_completed_hole_again() {
        for fail in 1..=4 {
            let mut regions = Regions::new(16384, 4096).unwrap();
            let mut native = Native::new(16384);
            regions.replace(4096, 8192, 0, true, &mut native).unwrap();
            native.fail = Some(native.calls + fail);
            assert!(regions.free(&mut native).is_err());
            native.assert_owned(&regions);
            assert_eq!(native.records.len(), 1);
            assert!(regions.clear(0, 16384, &mut native).is_err());
            assert!(regions.replace(0, 4096, 0, true, &mut native).is_err());
            native.fail = None;
            regions.free(&mut native).unwrap();
            assert!(native.records.is_empty());
        }
    }

    #[test]
    fn failed_detach_attempts_other_views_and_keeps_poison_sticky() {
        let mut regions = Regions::new(12288, 4096).unwrap();
        let mut native = Native::new(12288);
        regions.replace(0, 12288, 0, true, &mut native).unwrap();
        native.fail = Some(native.calls + 2);
        assert!(regions.clear(0, 12288, &mut native).is_err());
        assert_eq!(native.views.len(), 1);
        native.assert_owned(&regions);
        native.fail = None;
        regions.clear(0, 12288, &mut native).unwrap();
        assert!(regions.replace(0, 4096, 0, true, &mut native).is_err());
        regions.free(&mut native).unwrap();
    }
}
