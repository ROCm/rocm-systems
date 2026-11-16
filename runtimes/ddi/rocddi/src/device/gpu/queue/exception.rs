// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native AQL queue exception decoding and scratch replacement ownership.

/// Portable meaning of the command processor's AQL exception bits.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AqlQueueException {
    /// Queue stopped for scratch backing sized to this execution wave.
    Scratch {
        /// Execution width needed for the replacement scratch descriptor.
        lanes_per_wave: u32,
    },
    /// Kernel arguments or dispatch dimensions were incompatible.
    IncompatibleArguments,
    /// A dispatch referenced an invalid allocation.
    InvalidAllocation,
    /// The code object could not be executed.
    InvalidCodeObject,
    /// A queue packet faulted while accessing memory.
    MemoryFault,
    /// A packet format or header field was invalid.
    InvalidPacketFormat,
    /// A packet argument was invalid.
    InvalidArgument,
    /// The dispatch exceeded its register budget.
    OutOfRegisters,
    /// An access crossed a protected memory aperture.
    MemoryApertureViolation,
    /// The shader executed an illegal instruction.
    IllegalInstruction,
    /// A device exception was reported without a narrower classification.
    Exception,
    /// No known exception bit was set.
    Unknown,
}

/// Extracts the scratch wave size from a stopped AQL queue's inactive signal.
/// A queue-error signal can include other faults in the same word, so this
/// check is separate from the priority-ordered error classification.
#[must_use]
pub const fn aql_scratch_lanes(bits: u64) -> Option<u32> {
    if bits & 0x401 != 0 {
        Some(if bits & 0x400 != 0 { 32 } else { 64 })
    } else {
        None
    }
}

/// Decodes native CP error bits in priority order.
#[must_use]
pub const fn decode_aql_queue_exception(bits: u64) -> AqlQueueException {
    if bits & 2 != 0 {
        AqlQueueException::IncompatibleArguments
    } else if bits & 4 != 0 {
        AqlQueueException::InvalidAllocation
    } else if bits & 8 != 0 {
        AqlQueueException::InvalidCodeObject
    } else if bits & 16 != 0 {
        AqlQueueException::MemoryFault
    } else if bits & (32 | 256) != 0 {
        AqlQueueException::InvalidPacketFormat
    } else if bits & 64 != 0 {
        AqlQueueException::InvalidArgument
    } else if bits & 128 != 0 {
        AqlQueueException::OutOfRegisters
    } else if bits & 0x2000_0000 != 0 {
        AqlQueueException::MemoryApertureViolation
    } else if bits & 0x4000_0000 != 0 {
        AqlQueueException::IllegalInstruction
    } else if bits & 0x8000_0000 != 0 {
        AqlQueueException::Exception
    } else if let Some(lanes_per_wave) = aql_scratch_lanes(bits) {
        AqlQueueException::Scratch { lanes_per_wave }
    } else {
        AqlQueueException::Unknown
    }
}

/// Stages candidate scratch before an update can publish its address. On
/// failure, both the current and candidate owners remain retained; on success,
/// the displaced owner is returned only after the update completes.
///
/// # Errors
/// Reports reserve or update failure without releasing either backing owner.
pub fn replace_scratch_owner<T, E>(
    current: &mut Option<T>,
    uncertain: &mut Vec<T>,
    candidate: T,
    reserve_error: E,
    update: impl FnOnce() -> Result<(), E>,
) -> Result<Option<T>, E> {
    uncertain.try_reserve(1).map_err(|_| reserve_error)?;
    uncertain.push(candidate);
    update()?;
    Ok(std::mem::replace(current, uncertain.pop()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scratch_decode_preserves_wave_size_and_fault_priority() {
        assert_eq!(
            decode_aql_queue_exception(1),
            AqlQueueException::Scratch { lanes_per_wave: 64 }
        );
        assert_eq!(
            decode_aql_queue_exception(0x400),
            AqlQueueException::Scratch { lanes_per_wave: 32 }
        );
        assert_eq!(aql_scratch_lanes(0x402), Some(32));
        assert_eq!(
            decode_aql_queue_exception(0x402),
            AqlQueueException::IncompatibleArguments
        );
        assert_eq!(decode_aql_queue_exception(0), AqlQueueException::Unknown);
        assert_eq!(
            decode_aql_queue_exception(2 | 16),
            AqlQueueException::IncompatibleArguments
        );
    }
}
