// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! GFX1250 kernel-entry compatibility prologues and descriptor fixups.
//!
//! These transformations preserve the current loader policy. Their applicability
//! is a target/compiler contract, not a requirement of the public HSA API.

use crate::Error;
use crate::elf::{read_i64, read_u32};

const GFX1250_TRAMPOLINE_BYTES: usize = 256;
const GFX1250_PREFETCH_LINE_BYTES: usize = 128;
const GFX1250_LEGACY_PREFETCH: [u32; 4] = [0xee17_4000, 0x0004_0000, 0, 0x7e00_0000];
const GFX1250_NULL_PREFETCH: [u32; 5] = [0xbec0_0180, 0x7e00_0000, 0xee17_4040, 0x0004_0000, 0];
const GFX1250_REPLAY_MODE: [u32; 2] = [0xb980_0641, 1];

fn gfx1250_entry_has_prefetch(image: &[u8], entry: usize) -> bool {
    let matches_words = |start: usize, words: &[u32]| {
        words.iter().enumerate().all(|(index, word)| {
            start
                .checked_add(index * 4)
                .and_then(|offset| read_u32(image, offset))
                == Some(*word)
        })
    };
    let start = if matches_words(entry, &GFX1250_REPLAY_MODE) {
        entry.saturating_add(GFX1250_REPLAY_MODE.len() * 4)
    } else {
        entry
    };
    matches_words(start, &GFX1250_NULL_PREFETCH) || matches_words(start, &GFX1250_LEGACY_PREFETCH)
}

/// A checked plan for GFX1250 kernel entry trampolines.
///
/// The caller passes the virtual values of defined kernel descriptors. The
/// plan leaves device allocation to its caller and writes only caller-owned
/// buffers, so it can be used without an HSA runtime or a driver.
pub(crate) struct TrampolinePlan {
    /// Descriptor and entry offsets requiring a trampoline.
    fixups: Vec<(usize, usize)>,
    /// Byte extent needed for every stub and its prefetch allowance.
    required_bytes: usize,
}

impl TrampolinePlan {
    /// Finds descriptors whose kernel entries lack the prefetch prologue.
    ///
    /// # Errors
    /// Returns an invalid-object error for an out-of-range descriptor, entry,
    /// or trampoline extent.
    pub fn new(
        image: &[u8],
        virtual_base: u64,
        descriptors: impl IntoIterator<Item = u64>,
    ) -> Result<Self, Error> {
        let mut fixups = Vec::new();
        let mut max_prefetch_bytes = 0usize;
        for value in descriptors {
            let descriptor = usize::try_from(
                value
                    .checked_sub(virtual_base)
                    .ok_or(Error::InvalidObject)?,
            )
            .map_err(|_| Error::InvalidObject)?;
            if image
                .get(descriptor..descriptor.checked_add(64).ok_or(Error::InvalidObject)?)
                .is_none()
            {
                return Err(Error::InvalidObject);
            }
            let entry_delta = read_i64(image, descriptor + 16).ok_or(Error::InvalidObject)?;
            let entry_vaddr = u64::try_from(i128::from(value) + i128::from(entry_delta))
                .map_err(|_| Error::InvalidObject)?;
            let entry = usize::try_from(
                entry_vaddr
                    .checked_sub(virtual_base)
                    .ok_or(Error::InvalidObject)?,
            )
            .map_err(|_| Error::InvalidObject)?;
            if image
                .get(entry..entry.checked_add(4).ok_or(Error::InvalidObject)?)
                .is_none()
            {
                return Err(Error::InvalidObject);
            }
            if gfx1250_entry_has_prefetch(image, entry) {
                continue;
            }
            let rsrc3 = read_u32(image, descriptor + 44).ok_or(Error::InvalidObject)?;
            max_prefetch_bytes = max_prefetch_bytes.max(
                usize::try_from((rsrc3 >> 4) & 0xff)
                    .map_err(|_| Error::InvalidObject)?
                    .checked_mul(GFX1250_PREFETCH_LINE_BYTES)
                    .ok_or(Error::InvalidObject)?,
            );
            fixups.push((descriptor, entry));
        }
        let required_bytes = if fixups.is_empty() {
            0
        } else {
            fixups
                .len()
                .checked_mul(GFX1250_TRAMPOLINE_BYTES)
                .and_then(|bytes| {
                    bytes.checked_add(max_prefetch_bytes.saturating_sub(GFX1250_TRAMPOLINE_BYTES))
                })
                .ok_or(Error::InvalidObject)?
        };
        Ok(Self {
            fixups,
            required_bytes,
        })
    }

    /// Returns whether this image needs any entry workaround.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.fixups.is_empty()
    }

    /// Returns the required byte extent before caller-specific page alignment.
    #[must_use]
    pub fn required_bytes(&self) -> usize {
        self.required_bytes
    }

    /// Writes executable stubs and updates the caller's kernel descriptors.
    ///
    /// The caller must publish neither image until this method succeeds.
    ///
    /// # Errors
    /// Returns an invalid-object error if either buffer or device address
    /// cannot contain the planned stubs.
    pub fn write(
        &self,
        image: &mut [u8],
        device_base: u64,
        stub: &mut [u8],
        stub_device_base: u64,
    ) -> Result<(), Error> {
        if stub.len() < self.required_bytes
            || stub_device_base % GFX1250_TRAMPOLINE_BYTES as u64 != 0
        {
            return Err(Error::InvalidObject);
        }
        for (index, &(descriptor, entry)) in self.fixups.iter().enumerate() {
            let stub_offset = index
                .checked_mul(GFX1250_TRAMPOLINE_BYTES)
                .ok_or(Error::InvalidObject)?;
            let stub_address = stub_device_base
                .checked_add(stub_offset as u64)
                .ok_or(Error::InvalidObject)?;
            let entry_address = device_base
                .checked_add(entry as u64)
                .ok_or(Error::InvalidObject)?;
            let descriptor_address = device_base
                .checked_add(descriptor as u64)
                .ok_or(Error::InvalidObject)?;
            let delta = i64::try_from(i128::from(stub_address) - i128::from(descriptor_address))
                .map_err(|_| Error::InvalidObject)?;
            let mut words = [0xbf9f_0000_u32; GFX1250_TRAMPOLINE_BYTES / 4];
            words[..4].copy_from_slice(&GFX1250_LEGACY_PREFETCH);
            words[4] = 0xbee4_00ff;
            words[5] = u32::try_from(entry_address & u64::from(u32::MAX))
                .map_err(|_| Error::InvalidObject)?;
            words[6] = 0xbee5_00ff;
            words[7] = (entry_address >> 32) as u32;
            words[8] = 0xbe80_4864;
            let destination = stub
                .get_mut(stub_offset..stub_offset + GFX1250_TRAMPOLINE_BYTES)
                .ok_or(Error::InvalidObject)?;
            for (destination, word) in destination.chunks_exact_mut(4).zip(words) {
                destination.copy_from_slice(&word.to_le_bytes());
            }
            image
                .get_mut(descriptor + 16..descriptor + 24)
                .ok_or(Error::InvalidObject)?
                .copy_from_slice(&delta.to_le_bytes());
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
