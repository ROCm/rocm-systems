// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Qualified copy recipes composed with the shared native ABCE encoders.

use super::{GpuCopyOptions, GpuCopySignalOperation, SdmaProtocol};
use crate::device::gpu::abce::{
    self,
    abi::{abce_builder_t, abce_isa_version_t, abce_wait_signal_params_t},
};

/// Optional GPU timestamp destinations for one encoded command.
#[derive(Clone, Copy)]
pub(super) struct TimingAddresses {
    pub(super) start: Option<u64>,
    pub(super) end: u64,
}

fn builder(protocol: SdmaProtocol) -> abce_builder_t {
    let isa = match protocol {
        SdmaProtocol::Gfx9 => abce_isa_version_t {
            major: 9,
            minor: 0,
            stepping: 0,
        },
        SdmaProtocol::Gfx12_0 => abce_isa_version_t {
            major: 12,
            minor: 0,
            stepping: 0,
        },
        SdmaProtocol::Gfx12_5 => abce_isa_version_t {
            major: 12,
            minor: 5,
            stepping: 0,
        },
    };
    #[allow(unsafe_code)]
    // SAFETY: Defaults are valid for every qualified dialect and initialize
    // every field before the value is read.
    unsafe {
        let mut builder = std::mem::MaybeUninit::uninit();
        abce::abce_builder_initialize(isa, std::ptr::null(), builder.as_mut_ptr());
        builder.assume_init()
    }
}

fn controls(options: GpuCopyOptions) -> abce_wait_signal_params_t {
    #[allow(unsafe_code)]
    // SAFETY: The initializer has no pointer inputs.
    let mut params = unsafe { abce::abce_wait_signal_none() };
    if let Some(wait) = options.wait {
        params.wait_addr = wait.address as *mut _;
        params.wait_function = u32::from(wait.function);
        params.wait_reference = wait.reference;
        params.wait_mask = wait.mask;
    }
    if let Some(signal) = options.signal {
        params.signal_addr = signal.address as *mut _;
        params.signal_operation = match signal.operation {
            GpuCopySignalOperation::Write => 0x67,
            GpuCopySignalOperation::Add => 0x6f,
            GpuCopySignalOperation::Subtract => 0x70,
        };
        params.signal_value = signal.data;
    }
    params
}

#[allow(unsafe_code)]
pub(super) fn encode_gfx9_timestamp(words: &mut [u32], offset: usize, address: u64) {
    let output = &mut words[offset..offset + 3];
    output.fill(0);
    // SAFETY: A timestamp occupies exactly the three checked dwords. The GPU
    // address is encoded, not dereferenced by the host.
    unsafe {
        abce::abce_builder_build_get_global_timestamp(
            &builder(SdmaProtocol::Gfx9),
            output.as_mut_ptr().cast(),
            address as *mut _,
        );
    }
}

/// Native SDMA operation encoded into the shared indirect buffer.
#[derive(Clone, Copy)]
pub(super) enum SdmaCommand {
    Copy {
        destination: u64,
        source: u64,
        size: u32,
        options: GpuCopyOptions,
    },
    Fill {
        destination: u64,
        value: u32,
        count: u32,
    },
    Swap {
        address_a: u64,
        address_b: u64,
        size: u32,
        options: GpuCopyOptions,
    },
}

impl SdmaCommand {
    #[allow(
        unsafe_code,
        clippy::too_many_lines,
        reason = "the complete bounded packet recipe is kept together"
    )]
    pub(super) fn encode(
        &self,
        words: &mut [u32],
        timing: Option<TimingAddresses>,
        protocol: SdmaProtocol,
    ) -> u64 {
        let builder = builder(protocol);
        // These safe entry points must bound the raw emitter even if a future
        // caller forgets CopyResources' chunking contract.
        match *self {
            Self::Copy { size, .. } => assert!(size > 0 && size as usize <= builder.max_copy_size),
            Self::Fill { count, .. } => assert!(count > 0 && count <= builder.max_fill_dwords),
            Self::Swap { size, .. } => assert!(size > 0 && size <= 0x3f_ffff),
        }
        let gcr = match protocol {
            SdmaProtocol::Gfx9 => 0,
            SdmaProtocol::Gfx12_0 => 5,
            SdmaProtocol::Gfx12_5 => 6,
        };
        let options = match self {
            Self::Copy { options, .. } | Self::Swap { options, .. } => *options,
            Self::Fill { .. } => GpuCopyOptions::default(),
        };
        let body = if matches!(self, Self::Fill { .. }) {
            5
        } else {
            7 + usize::from(options.wait.is_some()) * 7 + usize::from(options.signal.is_some()) * 5
        };
        let start_words = usize::from(timing.is_some_and(|t| t.start.is_some())) * 3;
        let end_words = usize::from(timing.is_some()) * 3;
        let length = gcr * 2 + start_words + body + end_words;
        words.fill(0);
        let output = &mut words[..length];
        let mut timestamp_builder = builder;
        // Preserve the qualified GFX12 timestamp recipe's system scope, even
        // on the OSS5 copy dialect whose COPY packet has no scope fields.
        timestamp_builder.scope_fields = !matches!(protocol, SdmaProtocol::Gfx9);
        let body_offset = gcr + start_words;
        let end_offset = body_offset + body;
        // SAFETY: Each checked slice has the exact packet size. Copy/fill
        // chunks are bounded by CopyResources before reaching this encoder;
        // foreign GPU addresses are only encoded into words here.
        unsafe {
            if gcr != 0 {
                abce::abce_builder_build_gcr(&raw const builder, output.as_mut_ptr().cast(), true);
                abce::abce_builder_build_gcr(
                    &raw const builder,
                    output.as_mut_ptr().add(end_offset + end_words).cast(),
                    false,
                );
            }
            if let Some(start) = timing.and_then(|t| t.start) {
                abce::abce_builder_build_get_global_timestamp(
                    &raw const timestamp_builder,
                    output.as_mut_ptr().add(gcr).cast(),
                    start as *mut _,
                );
            }
            let target = output.as_mut_ptr().add(body_offset).cast();
            match *self {
                Self::Copy {
                    destination,
                    source,
                    size,
                    options,
                } => {
                    if options.indirect_source || options.indirect_destination {
                        // The public ABCE indirect helper fixes EQ/SUB(1).
                        // Native options support arbitrary comparisons and
                        // atomic operands. Reuse the configurable copy block
                        // and add the indirect flags/scope to this one packet.
                        abce::abce_builder_build_wait_signal_copy(
                            &raw const builder,
                            target,
                            destination as *mut _,
                            source as *const _,
                            size as usize,
                            &controls(options),
                        );
                        output[body_offset] |= u32::from(options.indirect_source) << 20
                            | u32::from(options.indirect_destination) << 21;
                        output[body_offset + 2 + usize::from(options.wait.is_some()) * 7] |=
                            3 << 10;
                    } else if !options.is_plain() {
                        abce::abce_builder_build_wait_signal_copy(
                            &raw const builder,
                            target,
                            destination as *mut _,
                            source as *const _,
                            size as usize,
                            &controls(options),
                        );
                    } else {
                        abce::abce_builder_build_copy(
                            &raw const builder,
                            target,
                            destination as *mut _,
                            source as *const _,
                            size as usize,
                        );
                    }
                }
                Self::Fill {
                    destination,
                    value,
                    count,
                } => {
                    abce::abce_builder_build_fill(
                        &raw const builder,
                        target,
                        destination as *mut _,
                        value,
                        count as usize,
                    );
                    if matches!(protocol, SdmaProtocol::Gfx12_5) {
                        // Keep this service's existing single-command NPD
                        // recipe. ABCE batch frames use their own ordered fill.
                        output[body_offset] |= 1 << 29;
                    }
                }
                Self::Swap {
                    address_a,
                    address_b,
                    size,
                    options,
                } => {
                    if options.is_plain() {
                        abce::abce_builder_build_swap_copy(
                            &raw const builder,
                            target,
                            address_a as *mut _,
                            address_b as *mut _,
                            size as usize,
                        );
                    } else {
                        abce::abce_builder_build_wait_signal_swap(
                            &raw const builder,
                            target,
                            address_a as *mut _,
                            address_b as *mut _,
                            size as usize,
                            size as usize,
                            &controls(options),
                        );
                    }
                }
            }
            if let Some(timing) = timing {
                abce::abce_builder_build_get_global_timestamp(
                    &raw const timestamp_builder,
                    output.as_mut_ptr().add(end_offset).cast(),
                    timing.end as *mut _,
                );
            }
        }
        (length * 4) as u64
    }
}

#[cfg(test)]
fn encode_copy(words: &mut [u32], destination: u64, source: u64, size: u32) {
    SdmaCommand::Copy {
        destination,
        source,
        size,
        options: GpuCopyOptions::default(),
    }
    .encode(words, None, SdmaProtocol::Gfx12_0);
}

#[cfg(test)]
fn encode_fill(words: &mut [u32], destination: u64, value: u32, count: u32) {
    SdmaCommand::Fill {
        destination,
        value,
        count,
    }
    .encode(words, None, SdmaProtocol::Gfx12_0);
}

#[cfg(test)]
mod tests {
    use super::super::{
        COMMAND_BYTES, COPY_PACKET_BYTES, GCR_INVALIDATE, GCR_WRITEBACK, SDMA_CONST_FILL_DWORD,
        SDMA_COPY_LINEAR, SDMA_TIMESTAMP_GET_GLOBAL_SYS, SDMA_USER_GCR, TIMED_COMMAND_BYTES,
        TIMESTAMP_PACKET_BYTES,
    };
    use super::*;

    #[test]
    fn indirect_copy_preserves_native_comparison_and_atomic_operands() {
        use super::super::{GpuCopySignal, GpuCopyWait};
        for (operation, opcode) in [
            (GpuCopySignalOperation::Write, 0x67),
            (GpuCopySignalOperation::Add, 0x6f),
            (GpuCopySignalOperation::Subtract, 0x70),
        ] {
            let copy = SdmaCommand::Copy {
                destination: 0x2000,
                source: 0x1000,
                size: 64,
                options: GpuCopyOptions {
                    indirect_source: true,
                    wait: Some(GpuCopyWait {
                        address: 0x3000,
                        function: 6,
                        reference: 7,
                        mask: u64::MAX,
                    }),
                    signal: Some(GpuCopySignal {
                        address: 0x4000,
                        operation,
                        data: 0x1_0000_002a,
                    }),
                    ..GpuCopyOptions::default()
                },
            };
            let mut words = [0; 64];
            assert_eq!(copy.encode(&mut words, None, SdmaProtocol::Gfx12_5), 124);
            assert_eq!(words[6], 1 | (1 << 20) | (1 << 30) | (1 << 31));
            assert_eq!(words[7], 6 | (3 << 18));
            assert_eq!(words[15], (3 << 10) | (3 << 18) | (3 << 26));
            assert_eq!(&words[20..25], &[opcode | (3 << 18), 0x4000, 0, 42, 1]);
        }
    }

    #[test]
    fn gfx9_indirect_timing_retains_both_timestamp_destinations() {
        let copy = SdmaCommand::Copy {
            destination: 0x1000,
            source: 0x2000,
            size: 64,
            options: GpuCopyOptions::default(),
        };
        let mut words = [0; 64];
        let length = copy.encode(
            &mut words,
            Some(TimingAddresses {
                start: Some(0x1_0000_3000),
                end: 0x2_0000_4000,
            }),
            SdmaProtocol::Gfx9,
        );
        assert_eq!(
            length,
            (COPY_PACKET_BYTES + 2 * TIMESTAMP_PACKET_BYTES) as u64
        );
        assert_eq!(&words[..3], &[0x20d, 0x3000, 1]);
        assert_eq!(&words[10..13], &[0x20d, 0x4000, 2]);
        let length = copy.encode(
            &mut words,
            Some(TimingAddresses {
                start: None,
                end: 0x2_0000_4000,
            }),
            SdmaProtocol::Gfx9,
        );
        assert_eq!(length, (COPY_PACKET_BYTES + TIMESTAMP_PACKET_BYTES) as u64);
        assert_eq!(&words[7..10], &[0x20d, 0x4000, 2]);
    }

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
