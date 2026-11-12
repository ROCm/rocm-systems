// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! One KFD PC sampling registration and its retryable native lifetime.

use super::{errno, error, memory, native_error, uapi};
use crate::device::gpu::profiling::{PcSamplingIntervalUnit, PcSamplingMethod, PcSamplingRequest};
use crate::driver::PcSamplingResource;
use crate::memory::storage::{Owned, Shared};
use crate::{Error, ErrorKind};
use std::num::NonZeroU32;

/// Last known collection state after a KFD start or stop request.
#[derive(Clone, Copy, Eq, PartialEq)]
enum SamplingState {
    /// Collection has not started or was stopped successfully.
    Stopped,
    /// Collection started successfully.
    Running,
    /// KFD did not establish whether a state change completed.
    Unknown,
}

/// Retains the KFD context that owns a native trace ID until its destruction
/// has been proved. An ambiguous destroy leaks that context for process exit.
pub(crate) struct KfdPcSampling {
    /// Keeps the KFD context alive until trace destruction is conclusive.
    vm: Shared<memory::DeviceVm>,
    /// Native GPU owning the trace registration.
    gpu_id: u32,
    /// Registration ID, cleared only after a successful destroy.
    trace_id: Option<u32>,
    /// Last known collection state.
    state: SamplingState,
    /// Prevents replay of a possibly completed native destroy.
    destroy_uncertain: bool,
    /// Whether this trace ID was supplied by an external owner.
    adopted: bool,
}

impl KfdPcSampling {
    /// Takes ownership of a stopped registration in the primary KFD process
    /// context, which can be created through another descriptor by a tool.
    /// The caller retains ownership if this function returns an error.
    pub(super) fn adopt(
        vm: Shared<memory::DeviceVm>,
        gpu_id: u32,
        trace_id: NonZeroU32,
    ) -> Result<Owned<Self>, Error> {
        vm.check()?;
        if !vm
            .kfd()
            .uses_primary_context()
            .map_err(|source| native_error("KFD PC sampling context check", source))?
        {
            return Err(error(
                ErrorKind::Unsupported,
                "external PC sampling trace IDs require the primary KFD context",
            ));
        }
        Ok(Owned::try_new_uninit()?.write(Self {
            vm,
            gpu_id,
            trace_id: Some(trace_id.get()),
            state: SamplingState::Stopped,
            destroy_uncertain: false,
            adopted: true,
        }))
    }

    pub(super) fn create(
        vm: Shared<memory::DeviceVm>,
        gpu_id: u32,
        request: PcSamplingRequest,
    ) -> Result<Owned<Self>, Error> {
        if request.interval == 0 {
            return Err(error(
                ErrorKind::InvalidArgument,
                "PC sampling interval must be nonzero",
            ));
        }
        vm.check()?;
        let storage = Owned::try_new_uninit()?;
        let mut info = [uapi::PcSampleInfo {
            interval: request.interval,
            method: match request.method {
                PcSamplingMethod::HostTrap => uapi::PC_SAMPLE_METHOD_HOST_TRAP,
                PcSamplingMethod::Stochastic => uapi::PC_SAMPLE_METHOD_STOCHASTIC,
            },
            unit: match request.unit {
                PcSamplingIntervalUnit::Microseconds => uapi::PC_SAMPLE_UNIT_MICROSECONDS,
                PcSamplingIntervalUnit::ClockCycles => uapi::PC_SAMPLE_UNIT_CYCLES,
                PcSamplingIntervalUnit::Instructions => uapi::PC_SAMPLE_UNIT_INSTRUCTIONS,
            },
            ..uapi::PcSampleInfo::default()
        }];
        let mut args = uapi::PcSampleArgs {
            operation: uapi::PC_SAMPLE_CREATE,
            gpu_id,
            ..uapi::PcSampleArgs::default()
        };
        if let Err(source) = vm.kfd().pc_sample(&mut args, &mut info) {
            if args.trace_id != 0 {
                let mut cleanup = uapi::PcSampleArgs {
                    operation: uapi::PC_SAMPLE_DESTROY,
                    gpu_id,
                    trace_id: args.trace_id,
                    ..uapi::PcSampleArgs::default()
                };
                if vm.kfd().pc_sample(&mut cleanup, &mut []).is_err() {
                    std::mem::forget(vm);
                }
            } else if matches!(source.raw_os_error(), Some(errno::EINTR | errno::EFAULT)) {
                // The kernel may have registered a trace without returning
                // its ID. Retain the owning connection until process exit.
                std::mem::forget(vm);
            }
            return Err(if source.raw_os_error() == Some(errno::EEXIST) {
                Error::NativeOperation {
                    kind: ErrorKind::Busy,
                    operation: "KFD PC sampling creation",
                    source,
                }
            } else {
                native_error("KFD PC sampling creation", source)
            });
        }
        if args.trace_id == 0 {
            std::mem::forget(vm);
            return Err(error(
                ErrorKind::DriverContract,
                "KFD created a PC sampler without a trace ID",
            ));
        }
        Ok(storage.write(Self {
            vm,
            gpu_id,
            trace_id: Some(args.trace_id),
            state: SamplingState::Stopped,
            destroy_uncertain: false,
            adopted: false,
        }))
    }

    fn control(&self, operation: u32, trace_id: u32) -> Result<(), Error> {
        let mut args = uapi::PcSampleArgs {
            operation,
            gpu_id: self.gpu_id,
            trace_id,
            ..uapi::PcSampleArgs::default()
        };
        self.vm
            .kfd()
            .pc_sample(&mut args, &mut [])
            .map_err(|source| native_error("KFD PC sampling control", source))
    }
}

impl PcSamplingResource for KfdPcSampling {
    fn relinquish_adopted(&mut self) -> Result<(), Error> {
        if !self.adopted
            || self.trace_id.is_none()
            || self.state != SamplingState::Stopped
            || self.destroy_uncertain
        {
            return Err(error(
                ErrorKind::InvalidArgument,
                "only an unstarted adopted PC sampler can be relinquished",
            ));
        }
        self.trace_id = None;
        self.adopted = false;
        Ok(())
    }

    fn start(&mut self) -> Result<(), Error> {
        let Some(trace_id) = self.trace_id else {
            return Err(error(ErrorKind::InvalidArgument, "PC sampler is destroyed"));
        };
        if self.destroy_uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD PC sampler destruction outcome is uncertain",
            ));
        }
        if self.state == SamplingState::Running {
            return Ok(());
        }
        if self.state == SamplingState::Unknown {
            return Err(error(
                ErrorKind::Busy,
                "PC sampler start outcome must be resolved by stopping it",
            ));
        }
        self.vm.check()?;
        match self.control(uapi::PC_SAMPLE_START, trace_id) {
            Ok(()) => {
                self.state = SamplingState::Running;
                Ok(())
            }
            Err(source) => {
                self.state = SamplingState::Unknown;
                Err(source)
            }
        }
    }

    fn stop(&mut self) -> Result<(), Error> {
        let Some(trace_id) = self.trace_id else {
            return Ok(());
        };
        if self.state == SamplingState::Stopped {
            return Ok(());
        }
        match self.control(uapi::PC_SAMPLE_STOP, trace_id) {
            Ok(()) => {
                self.state = SamplingState::Stopped;
                Ok(())
            }
            Err(source) if source.native_error_code() == Some(errno::EALREADY) => {
                // KFD found this trace but reports that collection is already
                // disabled, which is sufficient to permit its destruction.
                self.state = SamplingState::Stopped;
                Ok(())
            }
            Err(source) => {
                self.state = SamplingState::Unknown;
                Err(source)
            }
        }
    }

    fn destroy(&mut self) -> Result<(), Error> {
        let Some(trace_id) = self.trace_id else {
            return Ok(());
        };
        if self.destroy_uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "KFD PC sampler destruction outcome is uncertain",
            ));
        }
        self.stop()?;
        let mut args = uapi::PcSampleArgs {
            operation: uapi::PC_SAMPLE_DESTROY,
            gpu_id: self.gpu_id,
            trace_id,
            ..uapi::PcSampleArgs::default()
        };
        match self.vm.kfd().pc_sample(&mut args, &mut []) {
            Ok(()) => {
                self.trace_id = None;
                Ok(())
            }
            Err(source) => {
                self.destroy_uncertain =
                    matches!(source.raw_os_error(), Some(errno::EINTR | errno::EFAULT));
                Err(native_error("KFD PC sampling destruction", source))
            }
        }
    }
}

impl Drop for KfdPcSampling {
    fn drop(&mut self) {
        if self.destroy().is_err() && self.trace_id.is_some() {
            std::mem::forget(self.vm.clone());
        }
    }
}
