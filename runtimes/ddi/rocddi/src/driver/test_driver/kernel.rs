// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Injectable native acceptance/retirement, without packet execution.

use super::*;
use crate::device::gpu::kernel_queue::{
    KernelCommand, KernelQueueFormat, KernelQueueStatus, KernelQueueWait,
};
use crate::driver::{GpuKernelQueueOperations, KernelQueueResource};

pub(crate) struct KernelControls {
    pub(crate) status: Mutex<KernelQueueStatus>,
    pub(crate) reject: AtomicBool,
    pub(crate) submitted: AtomicU64,
    pub(crate) allocations: Mutex<Vec<(u64, u64, DeviceAccess)>>,
}

impl Default for KernelControls {
    fn default() -> Self {
        Self {
            status: Mutex::new(KernelQueueStatus {
                retired_submission: 0,
                terminal: None,
            }),
            reject: AtomicBool::new(false),
            submitted: AtomicU64::new(0),
            allocations: Mutex::new(Vec::new()),
        }
    }
}

pub(crate) struct TestKernelQueue {
    controls: Arc<KernelControls>,
}

fn error(kind: ErrorKind) -> Error {
    Error::Operation {
        kind,
        detail: "injected kernel queue outcome",
    }
}

impl KernelQueueResource for TestKernelQueue {
    #[allow(unsafe_code)]
    unsafe fn submit(&self, command: KernelCommand) -> Result<u64, Error> {
        if self.controls.reject.load(Ordering::Acquire) {
            return Err(error(ErrorKind::Driver));
        }
        let allocations = self
            .controls
            .allocations
            .lock()
            .map_err(|_| error(ErrorKind::Internal))?;
        if !allocations.iter().any(|&(base, size, access)| {
            command.device_address >= base
                && command
                    .device_address
                    .checked_add(command.byte_length)
                    .is_some_and(|end| end <= base + size)
                && access.contains(DeviceAccess::EXECUTE)
        }) {
            return Err(error(ErrorKind::InvalidArgument));
        }
        Ok(self.controls.submitted.fetch_add(1, Ordering::AcqRel) + 1)
    }

    fn status(&self) -> KernelQueueStatus {
        *self
            .controls
            .status
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn refresh_status(&self) -> Result<KernelQueueStatus, Error> {
        Ok(self.status())
    }

    fn wait(&self, submission: u64, _: u64, _: u64) -> Result<KernelQueueWait, Error> {
        Ok(if self.status().retired_submission >= submission {
            KernelQueueWait::Retired
        } else {
            KernelQueueWait::TimedOut
        })
    }

    fn destroy(&mut self) -> Result<(), Error> {
        if self.status().retired_submission < self.controls.submitted.load(Ordering::Acquire) {
            return Err(error(ErrorKind::Busy));
        }
        Ok(())
    }
}

impl GpuKernelQueueOperations for TestDriver {
    type KernelQueue = TestKernelQueue;

    fn available_sdma_engines(&self, _: &TestDevice) -> Result<u32, Error> {
        self.kernel_controls
            .as_ref()
            .map(|_| 1)
            .ok_or_else(|| error(ErrorKind::Unsupported))
    }

    fn create_kernel_queue(
        &self,
        _: &TestDevice,
        format: KernelQueueFormat,
    ) -> Result<Owned<TestKernelQueue>, Error> {
        if format != KernelQueueFormat::SdmaOnEngine(0) {
            return Err(error(ErrorKind::Unsupported));
        }
        let controls = self
            .kernel_controls
            .clone()
            .ok_or_else(|| error(ErrorKind::Unsupported))?;
        Ok(Owned::new(TestKernelQueue { controls })?)
    }
}
