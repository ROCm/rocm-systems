//! Owned KFD PC-sampling registrations and capability translation.
//!
//! Capability records are copied out of the synchronous KFD query and
//! translated into implementation-neutral units. A live registration retains
//! its KFD endpoint and trace identifier. Failed stop or destroy operations keep
//! the registration owned for retry; ambiguous results are never replayed as if
//! no native side effect occurred.

use super::memory::{error, native_error};
use super::{sys, uapi};
use crate::host_storage::{Allocator, Buffer, Owned, Shared};
use crate::profiling::{
    PC_SAMPLING_INTERVAL_POWER_OF_TWO, PcSamplingConfiguration, PcSamplingMethod, PcSamplingUnits,
};
use crate::{Error, ErrorKind};
use std::io;

/// One KFD PC-sampling trace registration and its lifecycle state.
pub(crate) struct KfdPcSampling {
    kfd: Shared<sys::Kfd>,
    gpu_id: u32,
    trace_id: Option<u32>,
    active: bool,
    uncertain: bool,
}

impl KfdPcSampling {
    pub(super) fn configurations(
        kfd: &Shared<sys::Kfd>,
        gpu_id: u32,
        allocator: Allocator,
    ) -> Result<Buffer<PcSamplingConfiguration>, Error> {
        let native = kfd
            .pc_sampling_capabilities(gpu_id)
            .map_err(|source| pc_sampling_error("KFD PC sampling capability query", source))?;
        let mut configurations = Buffer::try_with_capacity(native.len(), allocator)?;
        for configuration in &native {
            let method = match configuration.method {
                uapi::PC_SAMPLE_METHOD_HOSTTRAP => PcSamplingMethod::HostTrapV1,
                uapi::PC_SAMPLE_METHOD_STOCHASTIC => PcSamplingMethod::StochasticV1,
                _ => {
                    return Err(error(
                        ErrorKind::DriverContract,
                        "KFD returned an unknown PC sampling method",
                    ));
                }
            };
            let units = match configuration.sample_type {
                uapi::PC_SAMPLE_TYPE_TIME_US => PcSamplingUnits::Microseconds,
                uapi::PC_SAMPLE_TYPE_CLOCK_CYCLES => PcSamplingUnits::ClockCycles,
                uapi::PC_SAMPLE_TYPE_INSTRUCTIONS => PcSamplingUnits::Instructions,
                _ => {
                    return Err(error(
                        ErrorKind::DriverContract,
                        "KFD returned unknown PC sampling interval units",
                    ));
                }
            };
            if configuration.interval_min == 0
                || configuration.interval_min > configuration.interval_max
            {
                return Err(error(
                    ErrorKind::DriverContract,
                    "KFD returned an invalid PC sampling interval range",
                ));
            }
            configurations.try_push(PcSamplingConfiguration {
                method,
                units,
                minimum_interval: configuration.interval_min,
                maximum_interval: configuration.interval_max,
                flags: configuration.flags,
            })?;
        }
        Ok(configurations)
    }

    pub(super) fn create(
        kfd: Shared<sys::Kfd>,
        gpu_id: u32,
        configuration: PcSamplingConfiguration,
        interval: u64,
        allocator: Allocator,
    ) -> Result<Owned<Self>, Error> {
        validate_configuration(configuration, interval)?;
        let storage = Owned::try_new_uninit(allocator)?;
        let mut native = to_native(configuration, interval);
        let mut trace_id = 0;
        if let Err(source) = kfd.pc_sampling_create(gpu_id, &mut native, &mut trace_id) {
            if trace_id != 0 && kfd.pc_sampling_destroy(gpu_id, trace_id).is_err() {
                // A failing create that nevertheless published an ID may have
                // left kernel state reachable through this endpoint.
                std::mem::forget(kfd);
            }
            return Err(pc_sampling_error("KFD PC sampling create", source));
        }
        Ok(storage.write(Self {
            kfd,
            gpu_id,
            trace_id: Some(trace_id),
            active: false,
            uncertain: false,
        }))
    }

    pub(super) fn adopt(
        kfd: Shared<sys::Kfd>,
        gpu_id: u32,
        trace_id: u32,
        allocator: Allocator,
    ) -> Result<Owned<Self>, Error> {
        if gpu_id == 0 || trace_id == 0 {
            return Err(error(
                ErrorKind::InvalidArgument,
                "invalid KFD PC sampling identity",
            ));
        }
        Ok(Owned::new(
            Self {
                kfd,
                gpu_id,
                trace_id: Some(trace_id),
                active: false,
                uncertain: false,
            },
            allocator,
        )?)
    }

    pub(crate) fn trace_id(&self) -> Result<u32, Error> {
        self.trace_id.ok_or_else(|| {
            error(
                ErrorKind::InvalidArgument,
                "PC sampling session was destroyed",
            )
        })
    }

    pub(crate) fn start(&mut self) -> Result<(), Error> {
        if self.active {
            return Ok(());
        }
        if self.uncertain {
            return Err(error(
                ErrorKind::DriverContract,
                "PC sampling state is uncertain",
            ));
        }
        let trace_id = self.trace_id()?;
        let result = self.kfd.pc_sampling_start(self.gpu_id, trace_id);
        match result {
            Ok(()) => {
                self.active = true;
                Ok(())
            }
            Err(source) => {
                self.uncertain = source.raw_os_error() == Some(4);
                Err(pc_sampling_error("KFD PC sampling start", source))
            }
        }
    }

    pub(crate) fn stop(&mut self) -> Result<(), Error> {
        if !self.active && !self.uncertain {
            return Ok(());
        }
        let trace_id = self.trace_id()?;
        let result = self.kfd.pc_sampling_stop(self.gpu_id, trace_id);
        match result {
            Ok(()) => {
                self.active = false;
                self.uncertain = false;
                Ok(())
            }
            Err(source) => {
                self.uncertain = source.raw_os_error() == Some(4);
                Err(pc_sampling_error("KFD PC sampling stop", source))
            }
        }
    }

    pub(crate) fn destroy(&mut self) -> Result<(), Error> {
        let Some(trace_id) = self.trace_id else {
            return Ok(());
        };
        if self.active || self.uncertain {
            self.stop()?;
        }
        self.kfd
            .pc_sampling_destroy(self.gpu_id, trace_id)
            .map_err(|source| pc_sampling_error("KFD PC sampling destroy", source))?;
        self.trace_id = None;
        Ok(())
    }
}

impl Drop for KfdPcSampling {
    fn drop(&mut self) {
        if self.destroy().is_err() && self.trace_id.is_some() {
            // Keep the exact KFD endpoint alive when native sampling state may
            // still refer to this process. Process teardown is the final owner.
            std::mem::forget(self.kfd.clone());
        }
    }
}

fn validate_configuration(
    configuration: PcSamplingConfiguration,
    interval: u64,
) -> Result<(), Error> {
    if configuration.minimum_interval == 0
        || configuration.minimum_interval > configuration.maximum_interval
        || interval < configuration.minimum_interval
        || interval > configuration.maximum_interval
        || (configuration.flags & PC_SAMPLING_INTERVAL_POWER_OF_TWO != 0
            && !interval.is_power_of_two())
    {
        return Err(error(
            ErrorKind::InvalidArgument,
            "invalid PC sampling interval",
        ));
    }
    Ok(())
}

fn to_native(configuration: PcSamplingConfiguration, interval: u64) -> uapi::PcSampleInfo {
    uapi::PcSampleInfo {
        interval,
        method: match configuration.method {
            PcSamplingMethod::HostTrapV1 => uapi::PC_SAMPLE_METHOD_HOSTTRAP,
            PcSamplingMethod::StochasticV1 => uapi::PC_SAMPLE_METHOD_STOCHASTIC,
        },
        sample_type: match configuration.units {
            PcSamplingUnits::Microseconds => uapi::PC_SAMPLE_TYPE_TIME_US,
            PcSamplingUnits::ClockCycles => uapi::PC_SAMPLE_TYPE_CLOCK_CYCLES,
            PcSamplingUnits::Instructions => uapi::PC_SAMPLE_TYPE_INSTRUCTIONS,
        },
        ..uapi::PcSampleInfo::default()
    }
}

fn pc_sampling_error(operation: &'static str, source: io::Error) -> Error {
    match source.raw_os_error() {
        Some(22) => Error::NativeOperation {
            kind: ErrorKind::InvalidArgument,
            operation,
            source,
        },
        Some(11 | 16) => Error::NativeOperation {
            kind: ErrorKind::Busy,
            operation,
            source,
        },
        Some(12 | 28) => Error::NativeOperation {
            kind: ErrorKind::ResourceExhausted,
            operation,
            source,
        },
        _ => native_error(operation, source),
    }
}

#[cfg(test)]
#[allow(clippy::panic, clippy::unwrap_used)]
mod tests {
    use super::*;
    use crate::driver::builtin::linux_kfd::sys::{Call, IoctlHook, Kfd};
    use std::fs::File;
    use std::sync::{Arc, Mutex};

    fn endpoint(hook: IoctlHook) -> Shared<Kfd> {
        Shared::new(
            Kfd::with_hook(File::open("/dev/null").unwrap(), hook),
            Allocator::default(),
        )
        .unwrap()
    }

    fn configuration() -> PcSamplingConfiguration {
        PcSamplingConfiguration {
            method: PcSamplingMethod::HostTrapV1,
            units: PcSamplingUnits::Microseconds,
            minimum_interval: 512,
            maximum_interval: u64::MAX,
            flags: 0,
        }
    }

    #[test]
    fn interval_validation_enforces_the_advertised_range_and_flags() {
        let configuration = PcSamplingConfiguration {
            maximum_interval: 4096,
            flags: 1,
            ..configuration()
        };
        for interval in [512, 1024, 4096] {
            assert!(validate_configuration(configuration, interval).is_ok());
        }
        for interval in [0, 511, 513, 8192] {
            assert_eq!(
                validate_configuration(configuration, interval)
                    .unwrap_err()
                    .kind(),
                ErrorKind::InvalidArgument
            );
        }
    }

    #[test]
    fn native_session_lifecycle_is_idempotent_and_ordered() {
        let operations = Arc::new(Mutex::new(Vec::new()));
        let observed = operations.clone();
        let kfd = endpoint(Arc::new(move |call| {
            let Call::PcSampling(args, configurations) = call else {
                panic!("unexpected ioctl")
            };
            observed.lock().unwrap().push(args.operation);
            if args.operation == uapi::PC_SAMPLE_OP_CREATE {
                assert_eq!(configurations.len(), 1);
                assert_eq!(configurations[0].interval, 512);
                args.trace_id = 17;
            } else {
                assert!(configurations.is_empty());
                assert_eq!(args.trace_id, 17);
            }
            Ok(())
        }));
        let mut sampling =
            KfdPcSampling::create(kfd, 42, configuration(), 512, Allocator::default()).unwrap();
        assert_eq!(sampling.trace_id().unwrap(), 17);
        sampling.start().unwrap();
        sampling.start().unwrap();
        sampling.stop().unwrap();
        sampling.stop().unwrap();
        sampling.destroy().unwrap();
        sampling.destroy().unwrap();
        drop(sampling);
        assert_eq!(
            *operations.lock().unwrap(),
            [
                uapi::PC_SAMPLE_OP_CREATE,
                uapi::PC_SAMPLE_OP_START,
                uapi::PC_SAMPLE_OP_STOP,
                uapi::PC_SAMPLE_OP_DESTROY
            ]
        );
    }

    #[test]
    fn active_session_drop_stops_before_destroying() {
        let operations = Arc::new(Mutex::new(Vec::new()));
        let observed = operations.clone();
        let kfd = endpoint(Arc::new(move |call| {
            let Call::PcSampling(args, _) = call else {
                panic!("unexpected ioctl")
            };
            observed.lock().unwrap().push(args.operation);
            if args.operation == uapi::PC_SAMPLE_OP_CREATE {
                args.trace_id = 17;
            }
            Ok(())
        }));
        let mut sampling =
            KfdPcSampling::create(kfd, 42, configuration(), 512, Allocator::default()).unwrap();
        sampling.start().unwrap();
        drop(sampling);
        assert_eq!(
            *operations.lock().unwrap(),
            [
                uapi::PC_SAMPLE_OP_CREATE,
                uapi::PC_SAMPLE_OP_START,
                uapi::PC_SAMPLE_OP_STOP,
                uapi::PC_SAMPLE_OP_DESTROY
            ]
        );
    }

    #[test]
    fn failed_create_releases_a_published_trace_id() {
        let operations = Arc::new(Mutex::new(Vec::new()));
        let observed = operations.clone();
        let kfd = endpoint(Arc::new(move |call| {
            let Call::PcSampling(args, _) = call else {
                panic!("unexpected ioctl")
            };
            observed.lock().unwrap().push(args.operation);
            if args.operation == uapi::PC_SAMPLE_OP_CREATE {
                args.trace_id = 17;
                Err(io::Error::from_raw_os_error(4))
            } else {
                assert_eq!(args.operation, uapi::PC_SAMPLE_OP_DESTROY);
                assert_eq!(args.trace_id, 17);
                Ok(())
            }
        }));
        let error = KfdPcSampling::create(kfd, 42, configuration(), 512, Allocator::default())
            .err()
            .unwrap();
        assert_eq!(error.native_error_code(), Some(4));
        assert_eq!(
            *operations.lock().unwrap(),
            [uapi::PC_SAMPLE_OP_CREATE, uapi::PC_SAMPLE_OP_DESTROY]
        );
    }
}
