// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Qualified CPU cache maintenance recipes for host mappings.
//!
//! Topology cache geometry does not qualify an executable recipe. A frontend
//! selects a recipe from its mapping policy, retains it with the mapping, and
//! keeps the complete intersecting cache lines mapped and synchronized while
//! applying it. No session or CPU device is activated.

use crate::{Error, ErrorKind};
use std::io;
use std::num::NonZeroU32;

/// Host mapping policy that selects an implemented CPU maintenance recipe.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CachePolicy {
    /// Write-back pages require line flushes and ordering fences.
    WriteBack,
    /// Write-combined pages require an ordering fence without line flushes.
    WriteCombined,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Recipe {
    WriteBack { line_size_bytes: NonZeroU32 },
    WriteCombined,
}

/// CPU operation qualified for one host mapping's cache policy.
///
/// This value owns no native resource. `apply` checks the current CPU at entry
/// and rejects a different line granularity observed then.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CacheMaintenance {
    recipe: Recipe,
}

impl CacheMaintenance {
    /// Qualifies a recipe for the mapping policy on the current host CPU.
    ///
    /// # Errors
    /// Returns `Unsupported` if the required instruction recipe is unavailable.
    pub fn qualify(policy: CachePolicy) -> Result<Self, Error> {
        let recipe = match policy {
            CachePolicy::WriteBack => {
                let bytes = qualified_line_size()
                    .map_err(|source| cache_error("CPU cache recipe", source))?;
                let line_size_bytes = NonZeroU32::new(bytes).ok_or(Error::Operation {
                    kind: ErrorKind::Unsupported,
                    detail: "CPU cache recipe has no line size",
                })?;
                Recipe::WriteBack { line_size_bytes }
            }
            CachePolicy::WriteCombined => {
                if !cfg!(target_arch = "x86_64") {
                    return Err(Error::Operation {
                        kind: ErrorKind::Unsupported,
                        detail: "write-combined host fence is unavailable",
                    });
                }
                Recipe::WriteCombined
            }
        };
        Ok(Self { recipe })
    }

    /// Returns the selected mapping policy.
    #[must_use]
    pub const fn policy(self) -> CachePolicy {
        match self.recipe {
            Recipe::WriteBack { .. } => CachePolicy::WriteBack,
            Recipe::WriteCombined => CachePolicy::WriteCombined,
        }
    }

    /// Returns line granularity for a write-back recipe; a fence has none.
    #[must_use]
    pub const fn line_size_bytes(self) -> Option<u32> {
        match self.recipe {
            Recipe::WriteBack { line_size_bytes } => Some(line_size_bytes.get()),
            Recipe::WriteCombined => None,
        }
    }

    /// Applies the qualified recipe to a nonempty host mapping range.
    ///
    /// Numeric bounds and the current CPU recipe are checked before execution.
    /// A write-back recipe flushes every intersecting line; a write-combined
    /// recipe orders prior writes without touching cache lines.
    ///
    /// # Errors
    /// Rejects an invalid range or changed/unavailable CPU recipe.
    ///
    /// # Safety
    /// The caller keeps every intersecting cache line mapped and externally
    /// synchronized through the call, including bytes outside the logical
    /// range. The selected policy must describe this mapping.
    #[allow(unsafe_code)]
    pub unsafe fn apply(self, pointer: usize, length: u64) -> Result<(), Error> {
        if Self::qualify(self.policy())? != self {
            return Err(Error::Operation {
                kind: ErrorKind::Unsupported,
                detail: "CPU cache recipe changed after qualification",
            });
        }
        let length = usize::try_from(length).map_err(|_| invalid_range())?;
        if pointer == 0 || length == 0 {
            return Err(invalid_range());
        }
        let end = pointer.checked_add(length - 1).ok_or_else(invalid_range)?;
        match self.recipe {
            Recipe::WriteBack { line_size_bytes } => {
                let line_size = line_size_bytes.get() as usize;
                let mask = line_size - 1;
                let mut line = pointer & !mask;
                let last = end & !mask;
                #[cfg(target_arch = "x86_64")]
                {
                    // SAFETY: The caller retains the full line cover; CPUID
                    // qualified CLFLUSH and its line size again above.
                    unsafe { std::arch::x86_64::_mm_mfence() };
                    loop {
                        unsafe { std::arch::x86_64::_mm_clflush(line as *const u8) };
                        if line == last {
                            break;
                        }
                        line += line_size;
                    }
                    unsafe { std::arch::x86_64::_mm_mfence() };
                    Ok(())
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let _ = (line, last);
                    Err(Error::Operation {
                        kind: ErrorKind::Unsupported,
                        detail: "write-back host cache recipe is unavailable",
                    })
                }
            }
            Recipe::WriteCombined => {
                #[cfg(target_arch = "x86_64")]
                {
                    // SAFETY: MFENCE orders writes and does not touch the range.
                    unsafe { std::arch::x86_64::_mm_mfence() };
                    Ok(())
                }
                #[cfg(not(target_arch = "x86_64"))]
                Err(Error::Operation {
                    kind: ErrorKind::Unsupported,
                    detail: "write-combined host fence is unavailable",
                })
            }
        }
    }
}

fn invalid_range() -> Error {
    Error::Operation {
        kind: ErrorKind::InvalidArgument,
        detail: "CPU cache control range is invalid",
    }
}

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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn qualification_keeps_line_flush_and_fence_recipes_distinct() {
        if cfg!(target_arch = "x86_64") {
            let fence = CacheMaintenance::qualify(CachePolicy::WriteCombined).unwrap();
            assert_eq!(fence.line_size_bytes(), None);
            assert_eq!(fence.policy(), CachePolicy::WriteCombined);
            if let Ok(flush) = CacheMaintenance::qualify(CachePolicy::WriteBack) {
                assert!(flush.line_size_bytes().unwrap().is_power_of_two());
                assert_eq!(flush.policy(), CachePolicy::WriteBack);
            }
        } else {
            assert!(CacheMaintenance::qualify(CachePolicy::WriteCombined).is_err());
        }
    }
}
