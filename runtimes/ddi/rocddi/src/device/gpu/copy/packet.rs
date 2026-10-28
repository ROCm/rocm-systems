// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! SDMA packet encoders for the qualified GPU target dialects.

use super::{
    COMMAND_BYTES, COPY_PACKET_BYTES, FILL_COMMAND_BYTES, FILL_PACKET_BYTES, GCR_INVALIDATE,
    GCR_PACKET_BYTES, GCR_WRITEBACK, GFX1250_GCR_PACKET_BYTES, GpuCopyOptions,
    GpuCopySignalOperation, SDMA_CONST_FILL_DWORD, SDMA_COPY_LINEAR, SDMA_COPY_SWAP,
    SDMA_SIGNAL_ADD64, SDMA_SIGNAL_SUB64, SDMA_SIGNAL_WRITE64, SDMA_TIMESTAMP_GET_GLOBAL_GFX9,
    SDMA_TIMESTAMP_GET_GLOBAL_SYS, SDMA_USER_GCR, SWAP_PACKET_BYTES, SdmaProtocol,
    TIMESTAMP_PACKET_BYTES,
};

fn encode_cache_envelope(command: &mut [u32], trailing_gcr: usize) {
    command.fill(0);
    command[0] = SDMA_USER_GCR;
    command[2] = GCR_WRITEBACK | GCR_INVALIDATE;
    command[trailing_gcr] = SDMA_USER_GCR;
    command[trailing_gcr + 2] = GCR_WRITEBACK;
}

fn encode_copy(command: &mut [u32], destination: u64, source: u64, size: u32) {
    // The command allocation is reused only after native retirement. Every
    // word is written, including reserved fields, before the release fence.
    encode_cache_envelope(command, 12);
    encode_copy_packet(command, 5, destination, source, size);
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "packet addresses are deliberately split into their low and high dwords"
)]
fn encode_copy_packet(
    command: &mut [u32],
    offset: usize,
    destination: u64,
    source: u64,
    size: u32,
) {
    command[offset] = SDMA_COPY_LINEAR;
    command[offset + 1] = size - 1;
    command[offset + 3] = source as u32;
    command[offset + 4] = (source >> 32) as u32;
    command[offset + 5] = destination as u32;
    command[offset + 6] = (destination >> 32) as u32;
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the packet address is deliberately split into low and high dwords"
)]
fn encode_fill(command: &mut [u32], destination: u64, value: u32, count: u32) {
    encode_cache_envelope(command, 10);
    command[5] = SDMA_CONST_FILL_DWORD;
    command[6] = destination as u32;
    command[7] = (destination >> 32) as u32;
    command[8] = value;
    command[9] = (count - 1) * 4;
}

fn encode_gfx1250_cache_envelope(command: &mut [u32], trailing_gcr: usize) {
    command.fill(0);
    command[0] = SDMA_USER_GCR;
    command[3] = (1 << 4) | (1 << 14);
    command[trailing_gcr] = SDMA_USER_GCR;
    command[trailing_gcr + 3] = (1 << 4) | (1 << 15);
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "packet addresses are deliberately split into low and high dwords"
)]
fn encode_gfx1250_copy_packet(
    command: &mut [u32],
    offset: usize,
    destination: u64,
    source: u64,
    size: u32,
) {
    command[offset] = SDMA_COPY_LINEAR | (1 << 28);
    command[offset + 1] = size - 1;
    command[offset + 2] = (3 << 18) | (3 << 26);
    command[offset + 3] = source as u32;
    command[offset + 4] = (source >> 32) as u32;
    command[offset + 5] = destination as u32;
    command[offset + 6] = (destination >> 32) as u32;
}

fn encode_gfx1250_copy(
    command: &mut [u32],
    destination: u64,
    source: u64,
    size: u32,
    timing: Option<TimingAddresses>,
) -> u64 {
    let copy_offset = 6 + usize::from(timing.is_some_and(|timing| timing.start.is_some())) * 3;
    let end_offset = copy_offset + COPY_PACKET_BYTES / 4;
    let trailing_gcr = end_offset + usize::from(timing.is_some()) * 3;
    encode_gfx1250_cache_envelope(command, trailing_gcr);
    if let Some(start) = timing.and_then(|timing| timing.start) {
        encode_timestamp(command, 6, start);
    }
    encode_gfx1250_copy_packet(command, copy_offset, destination, source, size);
    if let Some(timing) = timing {
        encode_timestamp(command, end_offset, timing.end);
    }
    ((trailing_gcr + GFX1250_GCR_PACKET_BYTES / 4) * 4) as u64
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "packet address is deliberately split into low and high dwords"
)]
fn encode_gfx1250_fill(command: &mut [u32], destination: u64, value: u32, count: u32) -> u64 {
    let trailing_gcr = 6 + FILL_PACKET_BYTES / 4;
    encode_gfx1250_cache_envelope(command, trailing_gcr);
    command[6] = SDMA_CONST_FILL_DWORD | (3 << 24) | (1 << 29);
    command[7] = destination as u32;
    command[8] = (destination >> 32) as u32;
    command[9] = value;
    command[10] = (count - 1) * 4;
    ((trailing_gcr + GFX1250_GCR_PACKET_BYTES / 4) * 4) as u64
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "packet addresses are deliberately split into low and high dwords"
)]
fn encode_gfx1250_swap(
    command: &mut [u32],
    address_a: u64,
    address_b: u64,
    size: u32,
    timing: Option<TimingAddresses>,
) -> u64 {
    let swap_offset = 6 + usize::from(timing.is_some_and(|timing| timing.start.is_some())) * 3;
    let end_offset = swap_offset + SWAP_PACKET_BYTES / 4;
    let trailing_gcr = end_offset + usize::from(timing.is_some()) * 3;
    encode_gfx1250_cache_envelope(command, trailing_gcr);
    if let Some(start) = timing.and_then(|timing| timing.start) {
        encode_timestamp(command, 6, start);
    }
    command[swap_offset] = SDMA_COPY_SWAP;
    command[swap_offset + 1] = size - 1;
    command[swap_offset + 2] = (3 << 18) | (3 << 26);
    command[swap_offset + 3] = address_a as u32;
    command[swap_offset + 4] = (address_a >> 32) as u32;
    command[swap_offset + 5] = address_b as u32;
    command[swap_offset + 6] = (address_b >> 32) as u32;
    if let Some(timing) = timing {
        encode_timestamp(command, end_offset, timing.end);
    }
    ((trailing_gcr + GFX1250_GCR_PACKET_BYTES / 4) * 4) as u64
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "packet addresses and operands are deliberately split into dwords"
)]
fn encode_gfx1250_controlled(
    command: &mut [u32],
    destination: u64,
    source: u64,
    size: u32,
    swap: bool,
    options: GpuCopyOptions,
    timing: Option<TimingAddresses>,
) -> u64 {
    let packet_offset = 6 + usize::from(timing.is_some_and(|timing| timing.start.is_some())) * 3;
    let copy_offset = packet_offset + 1 + usize::from(options.wait.is_some()) * 7;
    let signal_offset = copy_offset + 6;
    let end_offset = signal_offset + usize::from(options.signal.is_some()) * 5;
    let trailing_gcr = end_offset + usize::from(timing.is_some()) * 3;
    encode_gfx1250_cache_envelope(command, trailing_gcr);
    if let Some(start) = timing.and_then(|timing| timing.start) {
        encode_timestamp(command, 6, start);
    }

    command[packet_offset] = (if swap {
        SDMA_COPY_SWAP
    } else {
        SDMA_COPY_LINEAR
    }) | u32::from(options.indirect_source) << 20
        | u32::from(options.indirect_destination) << 21
        | u32::from(options.wait.is_some()) << 30
        | u32::from(options.signal.is_some()) << 31;
    if let Some(wait) = options.wait {
        let offset = packet_offset + 1;
        command[offset] = u32::from(wait.function) | (3 << 18);
        command[offset + 1] = wait.address as u32;
        command[offset + 2] = (wait.address >> 32) as u32;
        command[offset + 3] = wait.reference as u32;
        command[offset + 4] = (wait.reference >> 32) as u32;
        command[offset + 5] = wait.mask as u32;
        command[offset + 6] = (wait.mask >> 32) as u32;
    }
    command[copy_offset] = size - 1;
    command[copy_offset + 1] = if options.indirect_source || options.indirect_destination {
        (3 << 10) | (3 << 18) | (3 << 26)
    } else {
        (3 << 18) | (3 << 26)
    };
    command[copy_offset + 2] = source as u32;
    command[copy_offset + 3] = (source >> 32) as u32;
    command[copy_offset + 4] = destination as u32;
    command[copy_offset + 5] = (destination >> 32) as u32;
    if let Some(signal) = options.signal {
        command[signal_offset] = (match signal.operation {
            GpuCopySignalOperation::Write => SDMA_SIGNAL_WRITE64,
            GpuCopySignalOperation::Add => SDMA_SIGNAL_ADD64,
            GpuCopySignalOperation::Subtract => SDMA_SIGNAL_SUB64,
        }) | (3 << 18);
        command[signal_offset + 1] = signal.address as u32;
        command[signal_offset + 2] = (signal.address >> 32) as u32;
        command[signal_offset + 3] = signal.data as u32;
        command[signal_offset + 4] = (signal.data >> 32) as u32;
    }
    if let Some(timing) = timing {
        encode_timestamp(command, end_offset, timing.end);
    }
    ((trailing_gcr + GFX1250_GCR_PACKET_BYTES / 4) * 4) as u64
}

/// Optional GPU timestamp destinations for one encoded command.
#[derive(Clone, Copy)]
pub(super) struct TimingAddresses {
    /// First-packet timestamp address, when start timing is requested.
    pub(super) start: Option<u64>,
    /// Address written after the last packet.
    pub(super) end: u64,
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "packet addresses are deliberately split into their low and high dwords"
)]
fn encode_timestamp(command: &mut [u32], offset: usize, address: u64) {
    command[offset] = SDMA_TIMESTAMP_GET_GLOBAL_SYS;
    command[offset + 1] = address as u32;
    command[offset + 2] = (address >> 32) as u32;
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "the timestamp address is deliberately split into low and high dwords"
)]
pub(super) fn encode_gfx9_timestamp(command: &mut [u32], offset: usize, address: u64) {
    command[offset] = SDMA_TIMESTAMP_GET_GLOBAL_GFX9;
    command[offset + 1] = address as u32;
    command[offset + 2] = (address >> 32) as u32;
}

fn encode_timed_copy(
    command: &mut [u32],
    destination: u64,
    source: u64,
    size: u32,
    timing: TimingAddresses,
) -> u64 {
    let copy_offset = 5 + usize::from(timing.start.is_some()) * TIMESTAMP_PACKET_BYTES / 4;
    let end_offset = copy_offset + COPY_PACKET_BYTES / 4;
    let trailing_gcr = end_offset + TIMESTAMP_PACKET_BYTES / 4;
    encode_cache_envelope(command, trailing_gcr);
    if let Some(start) = timing.start {
        encode_timestamp(command, 5, start);
    }
    encode_copy_packet(command, copy_offset, destination, source, size);
    encode_timestamp(command, end_offset, timing.end);
    ((trailing_gcr + GCR_PACKET_BYTES / 4) * 4) as u64
}

/// Native SDMA operation encoded into the shared indirect buffer.
#[derive(Clone, Copy)]
pub(super) enum SdmaCommand {
    /// Copies bytes from source to destination.
    Copy {
        destination: u64,
        source: u64,
        size: u32,
        options: GpuCopyOptions,
    },
    /// Writes a dword value to a consecutive destination range.
    Fill {
        destination: u64,
        value: u32,
        count: u32,
    },
    /// Exchanges equal-sized source and destination ranges.
    Swap {
        address_a: u64,
        address_b: u64,
        size: u32,
        options: GpuCopyOptions,
    },
}

impl SdmaCommand {
    #[allow(
        clippy::cast_possible_truncation,
        reason = "SDMA packet addresses are split deliberately into low and high dwords"
    )]
    pub(super) fn encode(
        &self,
        words: &mut [u32],
        timing: Option<TimingAddresses>,
        protocol: SdmaProtocol,
    ) -> u64 {
        match *self {
            Self::Copy {
                destination,
                source,
                size,
                options,
            } => match protocol {
                SdmaProtocol::Gfx9 => {
                    debug_assert!(options.is_plain() && timing.is_none());
                    words.fill(0);
                    encode_copy_packet(words, 0, destination, source, size);
                    COPY_PACKET_BYTES as u64
                }
                SdmaProtocol::Gfx12_0 => {
                    debug_assert!(options.is_plain());
                    if let Some(timing) = timing {
                        encode_timed_copy(words, destination, source, size, timing)
                    } else {
                        encode_copy(words, destination, source, size);
                        COMMAND_BYTES as u64
                    }
                }
                SdmaProtocol::Gfx12_5 => {
                    if options.is_plain() {
                        encode_gfx1250_copy(words, destination, source, size, timing)
                    } else {
                        encode_gfx1250_controlled(
                            words,
                            destination,
                            source,
                            size,
                            false,
                            options,
                            timing,
                        )
                    }
                }
            },
            Self::Fill {
                destination,
                value,
                count,
            } => {
                debug_assert!(timing.is_none());
                match protocol {
                    SdmaProtocol::Gfx9 => {
                        words.fill(0);
                        words[0] = SDMA_CONST_FILL_DWORD;
                        words[1] = destination as u32;
                        words[2] = (destination >> 32) as u32;
                        words[3] = value;
                        words[4] = (count - 1) * 4;
                        FILL_PACKET_BYTES as u64
                    }
                    SdmaProtocol::Gfx12_0 => {
                        encode_fill(words, destination, value, count);
                        FILL_COMMAND_BYTES as u64
                    }
                    SdmaProtocol::Gfx12_5 => encode_gfx1250_fill(words, destination, value, count),
                }
            }
            Self::Swap {
                address_a,
                address_b,
                size,
                options,
            } => {
                debug_assert!(!matches!(protocol, SdmaProtocol::Gfx9));
                if options.is_plain() {
                    encode_gfx1250_swap(words, address_a, address_b, size, timing)
                } else {
                    encode_gfx1250_controlled(
                        words, address_b, address_a, size, true, options, timing,
                    )
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::TIMED_COMMAND_BYTES;
    use super::*;

    #[test]
    fn gfx1201_linear_copy_encodes_system_cache_controls_and_full_addresses() {
        let mut command = [u32::MAX; COMMAND_BYTES / 4];
        encode_copy(
            &mut command,
            0x1234_5678_9abc_def0,
            0xfeed_face_cafe_babe,
            4096,
        );
        assert_eq!(command[0], SDMA_USER_GCR);
        assert_eq!(command[2], GCR_WRITEBACK | GCR_INVALIDATE);
        assert_eq!(
            command[5..12],
            [
                1,
                4095,
                0,
                0xcafe_babe,
                0xfeed_face,
                0x9abc_def0,
                0x1234_5678
            ]
        );
        assert_eq!(command[12], SDMA_USER_GCR);
        assert_eq!(command[14], GCR_WRITEBACK);
        assert_eq!(command[16], 0);
    }

    #[test]
    fn gfx1201_timed_copy_brackets_packets_with_global_gpu_timestamps() {
        let copy = SdmaCommand::Copy {
            destination: 0x1234_5678_9abc_def0,
            source: 0xfeed_face_cafe_babe,
            size: 64,
            options: GpuCopyOptions::default(),
        };
        let mut command = [u32::MAX; TIMED_COMMAND_BYTES / 4];
        let byte_length = copy.encode(
            &mut command,
            Some(TimingAddresses {
                start: Some(0xaabb_ccdd_1122_3344),
                end: 0x5566_7788_99aa_bbcc,
            }),
            SdmaProtocol::Gfx12_0,
        );
        assert_eq!(byte_length, TIMED_COMMAND_BYTES as u64);
        assert_eq!(command[0], SDMA_USER_GCR);
        assert_eq!(command[2], GCR_WRITEBACK | GCR_INVALIDATE);
        assert_eq!(
            &command[5..18],
            &[
                SDMA_TIMESTAMP_GET_GLOBAL_SYS,
                0x1122_3344,
                0xaabb_ccdd,
                SDMA_COPY_LINEAR,
                63,
                0,
                0xcafe_babe,
                0xfeed_face,
                0x9abc_def0,
                0x1234_5678,
                SDMA_TIMESTAMP_GET_GLOBAL_SYS,
                0x99aa_bbcc,
                0x5566_7788,
            ]
        );
        assert_eq!(command[18], SDMA_USER_GCR);
        assert_eq!(command[20], GCR_WRITEBACK);
        assert_eq!(&command[21..], &[0, 0]);

        let byte_length = copy.encode(
            &mut command,
            Some(TimingAddresses {
                start: None,
                end: 0x5566_7788_99aa_bbcc,
            }),
            SdmaProtocol::Gfx12_0,
        );
        assert_eq!(
            byte_length,
            (TIMED_COMMAND_BYTES - TIMESTAMP_PACKET_BYTES) as u64
        );
        assert_eq!(command[5], SDMA_COPY_LINEAR);
        assert_eq!(command[12], SDMA_TIMESTAMP_GET_GLOBAL_SYS);
        assert_eq!(command[15], SDMA_USER_GCR);
        assert_eq!(&command[20..], &[0, 0, 0]);
    }

    #[test]
    fn gfx1201_fill_encodes_dword_count_and_cache_controls() {
        let mut command = [u32::MAX; COMMAND_BYTES / 4];
        encode_fill(&mut command, 0x1234_5678_9abc_def0, 0xaabb_ccdd, 1024);
        assert_eq!(command[0], SDMA_USER_GCR);
        assert_eq!(command[2], GCR_WRITEBACK | GCR_INVALIDATE);
        assert_eq!(command[5], SDMA_CONST_FILL_DWORD);
        assert_eq!(
            command[6..10],
            [0x9abc_def0, 0x1234_5678, 0xaabb_ccdd, 4092]
        );
        assert_eq!(command[10], SDMA_USER_GCR);
        assert_eq!(command[12], GCR_WRITEBACK);
        assert_eq!(command[14], 0);
    }
}
