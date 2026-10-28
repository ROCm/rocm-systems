// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! ROCr-compatible IPC signal representation and retained native backing.

use super::*;
use crate::platform::memory::{self as linux_interop, KfdIpcMemoryHandle};

/// IPC signal allocation size in bytes.
pub(super) const IPC_SIGNAL_ALLOCATION_BYTES: u64 = 4096;
/// Magic identifier used to validate an IPC signal block.
const IPC_SIGNAL_ID: u64 = 0x71fc_ca6a_3d5d_5276;

/// The mapped record exchanged with another HSA process.
///
/// `ROCr` peers read the signal prefix, `core_signal`, and `id` at fixed offsets.
/// The other bytes are reserved wire storage, not per-process Rust state.
#[repr(C, align(64))]
pub(super) struct IpcSignalBlock {
    /// Public signal record stored in shared memory.
    signal: AmdSignal,
    /// Reserved storage retained for the native layout.
    reserved0: u64,
    /// Native signal address exported through the IPC handle.
    core_signal: u64,
    /// Stable identifier used to validate an attached signal.
    id: u64,
    /// Reserved storage retained for the native layout.
    reserved: [u8; 40],
}

impl IpcSignalBlock {
    /// Initializes the public signal and IPC identity in shared storage.
    pub(super) fn new(value: SignalValue) -> Self {
        Self {
            signal: AmdSignal::user(value),
            reserved0: 0,
            core_signal: 0,
            id: IPC_SIGNAL_ID,
            reserved: [0; 40],
        }
    }

    /// Checks only the immutable IPC signature in an imported mapping.
    ///
    /// # Safety
    /// `host` must address at least `size` readable bytes retained by an IPC
    /// allocation throughout this call.
    pub(super) unsafe fn validate_mapped(host: usize, size: u64) -> bool {
        if host % align_of::<Self>() != 0 || size < size_of::<Self>() as u64 {
            return false;
        }
        let block = host as *const Self;
        // SAFETY: The caller retains the readable, aligned IPC mapping.
        let core_signal = unsafe { std::ptr::addr_of!((*block).core_signal).read() };
        // SAFETY: The same mapping contains the immutable signature.
        let id = unsafe { std::ptr::addr_of!((*block).id).read() };
        core_signal == 0 && id == IPC_SIGNAL_ID
    }
}

const _: () = {
    assert!(size_of::<IpcSignalBlock>() == 128);
    assert!(align_of::<IpcSignalBlock>() == 64);
    assert!(std::mem::offset_of!(IpcSignalBlock, signal) == 0);
    assert!(std::mem::offset_of!(IpcSignalBlock, core_signal) == 72);
    assert!(std::mem::offset_of!(IpcSignalBlock, id) == 80);
};

/// Exporter-owned allocation retained for the public signal lifetime.
pub(super) struct OwnedIpcSignal {
    /// Native shared allocation retained while this signal is live.
    pub(super) allocation: Allocation,
}

/// Imported mapping with the number of local public references.
pub(super) struct ImportedIpcSignal {
    /// Native shared allocation retained while this signal is live.
    pub(super) allocation: Allocation,
    /// Exported handle used to attach the shared allocation.
    pub(super) ipc_handle: [u32; 8],
    /// Number of local HSA handles sharing this imported signal.
    pub(super) references: u32,
}

impl SignalState {
    /// Creates IPC signal and records its ownership.
    pub(crate) fn create_ipc_signal(
        &mut self,
        gpus: &[crate::runtime::Gpu],
        host_page_size: usize,
        value: SignalValue,
    ) -> Result<HsaSignal, Status> {
        reserve_wait_signal_slots(1)?;
        let allocation = {
            let (gpu, peers) = gpus.split_first().ok_or(OUT_OF_RESOURCES)?;
            let peers = peers.iter().map(|peer| &peer.device).collect::<Vec<_>>();
            gpu.device
                .allocate_with_peers(
                    &peers,
                    MemoryKind::System,
                    IPC_SIGNAL_ALLOCATION_BYTES.max(host_page_size as u64),
                    host_page_size as u64,
                    DeviceAccess::READ | DeviceAccess::WRITE,
                )
                .map_err(map_error)?
        };
        let info = allocation.info();
        let host = info.host_address.ok_or(OUT_OF_RESOURCES)?;
        if info.device_address != host as u64 {
            return Err(OUT_OF_RESOURCES);
        }
        self.owned_ipc_signals
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        // SAFETY: The dedicated allocation is writable, page-sized, and retained
        // by owned_ipc_signals for the complete public signal lifetime.
        unsafe {
            std::ptr::write_bytes(host as *mut u8, 0, IPC_SIGNAL_ALLOCATION_BYTES as usize);
            (host as *mut IpcSignalBlock).write(IpcSignalBlock::new(value));
        }
        match self.owned_ipc_signals.entry(host) {
            Entry::Vacant(entry) => {
                entry.insert(OwnedIpcSignal { allocation });
                publish_wait_signal(host);
            }
            Entry::Occupied(_) => return Err(OUT_OF_RESOURCES),
        }
        Ok(HsaSignal {
            handle: host as u64,
        })
    }

    /// Attaches an exported IPC signal to native shared storage.
    pub(crate) unsafe fn attach_ipc_signal(
        &mut self,
        session: &rocddi::session::Session,
        gpus: &[crate::runtime::Gpu],
        words: [u32; 8],
    ) -> Result<HsaSignal, Status> {
        reserve_wait_signal_slots(1)?;
        if let Some((address, imported)) = self
            .imported_ipc_signals
            .iter_mut()
            .find(|(_, signal)| signal.ipc_handle == words)
        {
            imported.references = imported.references.checked_add(1).ok_or(OUT_OF_RESOURCES)?;
            let address = *address;
            if let Some(record) = self.async_signal_refs.get_mut(&address) {
                if record.references == 0 && record.retired.load(Ordering::Acquire) {
                    self.async_signal_refs.remove(&address);
                } else {
                    record.revive();
                }
            }
            publish_wait_signal(address);
            return Ok(HsaSignal {
                handle: address as u64,
            });
        }

        let allocation = {
            let devices = gpus.iter().map(|gpu| &gpu.device).collect::<Vec<_>>();
            linux_interop::import_kfd_ipc_memory(
                session,
                &devices,
                &devices,
                KfdIpcMemoryHandle { words },
                IPC_SIGNAL_ALLOCATION_BYTES,
            )
            .map_err(|error| match error.kind() {
                rocddi::ErrorKind::InvalidArgument | rocddi::ErrorKind::Unsupported => {
                    INVALID_ARGUMENT
                }
                rocddi::ErrorKind::ResourceExhausted => OUT_OF_RESOURCES,
                _ => ERROR,
            })?
        };
        let info = allocation.info();
        let host = info.host_address.ok_or(INVALID_ARGUMENT)?;
        if info.device_address != host as u64 {
            return Err(INVALID_ARGUMENT);
        }
        // SAFETY: The imported allocation retains the readable mapped range.
        if !unsafe { IpcSignalBlock::validate_mapped(host, info.size) } {
            return Err(INVALID_ARGUMENT);
        }
        self.imported_ipc_signals
            .try_reserve(1)
            .map_err(|_| OUT_OF_RESOURCES)?;
        match self.imported_ipc_signals.entry(host) {
            Entry::Vacant(entry) => {
                entry.insert(ImportedIpcSignal {
                    allocation,
                    ipc_handle: words,
                    references: 1,
                });
                publish_wait_signal(host);
            }
            Entry::Occupied(_) => return Err(OUT_OF_RESOURCES),
        }
        Ok(HsaSignal {
            handle: host as u64,
        })
    }
    /// Encodes an owned IPC signal as an attachable native handle.
    pub(crate) fn export_ipc_signal(&self, signal: HsaSignal) -> Result<[u32; 8], Status> {
        let address = signal.handle as usize;
        if !self.owns_signal(signal) {
            return Err(INVALID_ARGUMENT);
        }
        let allocation = self
            .owned_ipc_signals
            .get(&address)
            .map(|signal| &signal.allocation)
            .or_else(|| {
                self.imported_ipc_signals
                    .get(&address)
                    .map(|signal| &signal.allocation)
            })
            .ok_or(INVALID_ARGUMENT)?;
        linux_interop::export_kfd_ipc_memory(allocation)
            .map(|handle| handle.words)
            .map_err(map_error)
    }
}

#[cfg(test)]
/// IPC signal layout and identity checks.
mod tests {
    use super::*;

    #[test]
    /// Checks the shared signal layout and validates a mapped IPC block.
    fn ipc_signal_storage_matches_the_rocr_shared_layout() {
        assert_eq!(size_of::<HsaAmdIpcSignal>(), 32);
        assert_eq!(align_of::<HsaAmdIpcSignal>(), 4);
        assert_eq!(size_of::<IpcSignalBlock>(), 128);
        assert_eq!(align_of::<IpcSignalBlock>(), 64);
        assert_eq!(std::mem::offset_of!(IpcSignalBlock, signal), 0);
        assert_eq!(std::mem::offset_of!(IpcSignalBlock, core_signal), 72);
        assert_eq!(std::mem::offset_of!(IpcSignalBlock, id), 80);

        let signal = IpcSignalBlock::new(17);
        // SAFETY: The local block is aligned and readable for its full size.
        assert!(unsafe {
            IpcSignalBlock::validate_mapped(
                std::ptr::from_ref(&signal) as usize,
                size_of::<IpcSignalBlock>() as u64,
            )
        });
        assert_eq!(
            signal.signal.kind.load(Ordering::Relaxed),
            AMD_SIGNAL_KIND_USER
        );
        assert_eq!(signal.signal.value.load(Ordering::Relaxed), 17);
    }
}
