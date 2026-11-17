// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Process-context second-level traps matching debugger ABI version 11.

use std::sync::atomic::AtomicBool;

use rocddi::device::Device;
#[cfg(target_os = "linux")]
use rocddi::device::gpu::profiling::linux::ProcessTrap;
use rocddi::session::{DebuggerRegistrationMode, Session};
use rocddi::topology::GpuInfo;

use crate::ffi::{ERROR, OUT_OF_RESOURCES, Status};
use crate::runtime::map_error;

/// Exact trap build target and target/revision fixup rules from `ROCr`.
pub(crate) fn trap_bytes(info: &GpuInfo) -> Option<&'static [u8]> {
    Some(match (info.gfx_major, info.gfx_minor, info.gfx_stepping) {
        (9, 4, _) => include_bytes!("traps/trap_942.bin"),
        (9, 5, _) => include_bytes!("traps/trap_950.bin"),
        (9, 0, 8 | 10) => include_bytes!("traps/trap_9.bin"),
        (10, 1, _) => include_bytes!("traps/trap_1010.bin"),
        (10, 3, _) => include_bytes!("traps/trap_10.bin"),
        (11, _, _) => include_bytes!("traps/trap_11.bin"),
        (12, 0, _) => include_bytes!("traps/trap_12.bin"),
        (12, 5, 0) if info.asic_revision == 0 => include_bytes!("traps/trap_1250_rev0.bin"),
        (12, 5, _) => include_bytes!("traps/trap_1250.bin"),
        _ => return None,
    })
}

/// A session handle to normal trap backing owned by the shared native context.
#[cfg(target_os = "linux")]
struct GpuDebugger {
    /// Session device used to locate the corresponding sampling restore token.
    device: Device,
    /// Native backing retained independently of this HSA generation.
    trap: ProcessTrap,
}

/// Keeps the exact native normal trap available while sampling replaces it.
#[derive(Clone)]
pub(crate) struct TrapRestore {
    /// Saved normal code and exact VM, with no session driver reference.
    #[cfg(target_os = "linux")]
    trap: ProcessTrap,
}

impl TrapRestore {
    /// Resolves the saved trap in the exact GPU VM whose binding will be restored.
    pub(crate) fn address(&self, device: &Device) -> Result<u64, Status> {
        #[cfg(target_os = "linux")]
        return self
            .trap
            .address(&device.gpu().map_err(map_error)?)
            .map_err(map_error);
        #[cfg(not(target_os = "linux"))]
        {
            let _ = device;
            Err(ERROR)
        }
    }
}

/// Debugger owner retained through partial initialization and active generations.
pub(crate) struct PendingDebugger {
    /// Native context ownership precedes publication, including on failure.
    #[cfg(target_os = "linux")]
    gpus: Vec<GpuDebugger>,
}

impl PendingDebugger {
    /// Retains the existing trap binding before a sampling owner replaces it.
    pub(crate) fn restore_token(&self, device: &Device) -> Option<TrapRestore> {
        #[cfg(target_os = "linux")]
        {
            let owner = self
                .gpus
                .iter()
                .find(|owner| owner.device.endpoint().id == device.endpoint().id)?;
            Some(TrapRestore {
                trap: owner.trap.clone(),
            })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = device;
            None
        }
    }

    /// Stages the stable rendezvous and TTMP setup before native activation.
    pub(crate) fn new(session: &mut Session) -> Result<Self, Status> {
        // SAFETY: The static record and breakpoint live until process exit. The
        // loader retains each linked object until its delete notification ends.
        #[cfg(target_os = "linux")]
        unsafe {
            session.request_debugger_registration(
                super::record_address(),
                true,
                DebuggerRegistrationMode::Prefer,
            )
        }
        .map_err(map_error)?;
        #[cfg(not(target_os = "linux"))]
        let _ = session;
        Ok(Self {
            #[cfg(target_os = "linux")]
            gpus: Vec::new(),
        })
    }

    /// Retains a GPU's trap owner before starting fallible native publication.
    pub(crate) fn install(&mut self, device: &Device, cancel: &AtomicBool) -> Result<(), Status> {
        #[cfg(target_os = "linux")]
        {
            let gpu = device.gpu().map_err(map_error)?;
            // Preserve the registration/binding of a client that activated first.
            if !gpu
                .debugger_registration_matches(super::record_address(), true)
                .map_err(map_error)?
            {
                return Ok(());
            }
            self.gpus.try_reserve(1).map_err(|_| OUT_OF_RESOURCES)?;
            let code = trap_bytes(gpu.info()).ok_or(ERROR)?;
            // SAFETY: Target-selected ROCr normal trap code uses debugger ABI 11
            // with no TMA. HSA serializes its normal/sampling binding changes.
            let trap = unsafe { gpu.install_process_trap(super::record_address(), code, cancel) }
                .map_err(map_error)?;
            self.gpus.push(GpuDebugger {
                device: device.clone(),
                trap,
            });
            Ok(())
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = (device, cancel);
            Err(ERROR)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trap_fixups_follow_asic_revision_without_affecting_later_steppings() {
        let mut info = GpuInfo {
            gfx_major: 12,
            gfx_minor: 5,
            ..GpuInfo::default()
        };
        let rev0 = trap_bytes(&info);
        info.asic_revision = 1;
        let later = trap_bytes(&info);
        assert_ne!(rev0, later);
        info.gfx_stepping = 1;
        info.asic_revision = 0;
        assert_eq!(trap_bytes(&info), later);
        info.gfx_major = 13;
        assert!(trap_bytes(&info).is_none());
    }
}
