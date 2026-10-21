// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Host CPU cache maintenance for device-visible host mappings.
//!
//! The recipe depends on CPU instructions rather than an operating-system
//! handle. A caller qualifies it before advertising cache maintenance and
//! keeps every intersecting cache line mapped through execution.

use crate::{Error, ErrorKind};
use std::io;

/// Preserves the source error while classifying an unavailable CPU recipe.
fn cache_error(operation: &'static str, source: io::Error) -> Error {
    let kind = if source.kind() == io::ErrorKind::Unsupported {
        ErrorKind::Unsupported
    } else {
        ErrorKind::Driver
    };
    Error::NativeOperation {
        kind,
        operation,
        source,
    }
}

/// Returns the cache-line granularity only when the CPU supports this recipe.
pub(crate) fn cache_line_size() -> Result<u32, Error> {
    qualified_line_size().map_err(|source| cache_error("CPU cache recipe", source))
}

/// Applies the qualified recipe to a nonempty host range. A zero line size
/// drains prior writes for a write-combined mapping without touching lines.
///
/// # Safety
/// Every intersecting cache line must remain mapped and synchronized through
/// the call, including bytes outside the logical range.
#[allow(unsafe_code)]
pub(crate) unsafe fn cache_control(
    pointer: usize,
    length: u64,
    line_size: u32,
) -> Result<(), Error> {
    // SAFETY: The caller preserves the full intersecting cache-line cover.
    unsafe { apply_recipe(pointer, length, line_size) }
        .map_err(|source| cache_error("CPU cache control", source))
}

/// Reads the CPU's CLFLUSH capability and exact supported line size.
#[allow(unsafe_code)]
fn qualified_line_size() -> io::Result<u32> {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: CPUID leaf one exists on every x86-64 processor and does not
        // read caller memory. CLFLUSH is used only when its feature bit is set.
        #[allow(
            unused_unsafe,
            reason = "CPUID was unsafe on the supported Rust 1.85 toolchain"
        )]
        let leaf = unsafe { std::arch::x86_64::__cpuid(1) };
        let bytes = ((leaf.ebx >> 8) & 255) * 8;
        if leaf.edx & (1 << 19) == 0 || !bytes.is_power_of_two() {
            return Err(io::Error::from(io::ErrorKind::Unsupported));
        }
        Ok(bytes)
    }
    #[cfg(not(target_arch = "x86_64"))]
    Err(io::Error::from(io::ErrorKind::Unsupported))
}

/// Runs the qualified instruction sequence after validating the range.
///
/// # Safety
/// The complete intersecting cache-line cover must stay mapped and
/// synchronized throughout this call.
#[allow(unsafe_code)]
unsafe fn apply_recipe(pointer: usize, length: u64, line_size: u32) -> io::Result<()> {
    let qualified_line_size = qualified_line_size()?;
    if (line_size != 0 && line_size != qualified_line_size) || pointer == 0 || length == 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidInput));
    }
    let length =
        usize::try_from(length).map_err(|_| io::Error::from(io::ErrorKind::InvalidInput))?;
    let end = pointer
        .checked_add(length - 1)
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidInput))?;
    // A zero line size selects the write-combined mapping recipe. No cache
    // line is touched, but prior buffered writes must drain before return.
    if line_size == 0 {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            std::arch::x86_64::_mm_mfence();
        };
        return Ok(());
    }
    let mask = line_size as usize - 1;
    let line = pointer & !mask;
    let last = end & !mask;
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: The caller keeps the cover mapped. CLFLUSH support and exact
        // line size were checked above. MFENCE orders writes and later reads.
        unsafe { std::arch::x86_64::_mm_mfence() };
        let mut line = line;
        loop {
            unsafe { std::arch::x86_64::_mm_clflush(line as *const u8) };
            if line == last {
                break;
            }
            line += line_size as usize;
        }
        unsafe { std::arch::x86_64::_mm_mfence() };
        Ok(())
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = (line, last);
        Err(io::Error::from(io::ErrorKind::Unsupported))
    }
}
