// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Stable native `r_debug`/`link_map` records with paired loader notifications.

use std::ffi::CString;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI32, AtomicUsize, Ordering, fence};

use crate::ffi::{ERROR, OUT_OF_RESOURCES, Status};

/// ABI version shared by the trap blobs and the current AMD V2 queue layout.
const DEBUG_ABI_VERSION: i32 = 11;
/// Native SVR4 debugger states, also consumed by `rocdbgapi`.
const CONSISTENT: i32 = 0;
/// Native list mutation is adding one or more code objects.
const ADDING: i32 = 1;
/// Native list mutation is removing one or more code objects.
const DELETING: i32 = 2;

/// Native Linux/AMD rendezvous layout. Atomics preserve C field sizes/offsets.
#[repr(C)]
struct RuntimeDebugRecord {
    /// Trap and queue interpretation protocol consumed by `rocdbgapi`.
    version: i32,
    /// First currently published code object.
    head: AtomicUsize,
    /// Stable function address where a debugger places a host breakpoint.
    breakpoint: extern "C" fn(),
    /// Whether the linked list is consistent or being changed.
    state: AtomicI32,
    /// Reserved native linker base, unused by AMD's loader protocol.
    linker_base: usize,
}

impl RuntimeDebugRecord {
    /// Constructs a complete native record without runtime allocation.
    const fn new() -> Self {
        Self {
            version: DEBUG_ABI_VERSION,
            head: AtomicUsize::new(0),
            breakpoint: loader_debug_state,
            state: AtomicI32::new(CONSISTENT),
            linker_base: 0,
        }
    }

    /// Begins a mutation before changing any debugger-visible pointer.
    fn begin(&self, state: i32) {
        self.state.store(state, Ordering::Relaxed);
        fence(Ordering::SeqCst);
        (self.breakpoint)();
        fence(Ordering::SeqCst);
    }

    /// Publishes the completed list before the second breakpoint notification.
    fn finish(&self) {
        self.state.store(CONSISTENT, Ordering::Release);
        (self.breakpoint)();
    }
}

/// Side effect keeps the debugger breakpoint call and its unique address live.
static BREAKPOINT_SIDE_EFFECT: AtomicUsize = AtomicUsize::new(0);

/// Native loader rendezvous; the atomic side effect forbids eliding the call.
#[inline(never)]
extern "C" fn loader_debug_state() {
    BREAKPOINT_SIDE_EFFECT.fetch_xor(1, Ordering::Relaxed);
}

/// Process-lifetime native record registered through KFD's runtime-enable ioctl.
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
static _amdgpu_r_debug: RuntimeDebugRecord = RuntimeDebugRecord::new();

/// Returns the stable record for pre-activation native debugger registration.
pub(crate) fn record_address() -> usize {
    std::ptr::addr_of!(_amdgpu_r_debug) as usize
}

/// Native `link_map` prefix consumed by the debugger's host-memory reads.
#[repr(C)]
struct LinkMap {
    /// Wrapping difference between load address and ELF virtual base.
    load_bias: u64,
    /// NUL-terminated URI whose owned bytes survive this map's publication.
    name: usize,
    /// Dynamic-section pointer is unused by the AMD code-object protocol.
    dynamic: usize,
    /// Neighbor pointers remain stable through consistent-state publication.
    next: AtomicUsize,
    /// Previous published object, or zero for the head.
    previous: AtomicUsize,
}

/// Stable map allocation and the backing for its borrowed native URI address.
struct Entry {
    /// Debugger-visible record remains fixed while its owning Box moves.
    map: LinkMap,
    /// Retains the native URI independently of the loader's query metadata.
    _uri: CString,
}

impl Entry {
    /// Allocates URI storage before creating its native pointer.
    fn new(load_bias: u64, uri: &[u8]) -> Result<Box<Self>, Status> {
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(uri.len().checked_add(1).ok_or(OUT_OF_RESOURCES)?)
            .map_err(|_| OUT_OF_RESOURCES)?;
        bytes.extend_from_slice(uri);
        let uri = CString::new(bytes).map_err(|_| ERROR)?;
        Ok(Box::new(Self {
            map: LinkMap {
                load_bias,
                name: uri.as_ptr() as usize,
                dynamic: 0,
                next: AtomicUsize::new(0),
                previous: AtomicUsize::new(0),
            },
            _uri: uri,
        }))
    }

    /// Native address within this stable allocation.
    fn address(&self) -> usize {
        std::ptr::addr_of!(self.map) as usize
    }
}

/// Serializes native linked-list mutations, including destruction during unwind.
struct Registry {
    /// Last map owned by a still-live executable.
    tail: usize,
}

/// Shared across all HSA executable owners in this process.
static REGISTRY: Mutex<Registry> = Mutex::new(Registry { tail: 0 });

impl Registry {
    /// Links a fully prepared executable without further allocation.
    fn add(&mut self, record: &RuntimeDebugRecord, entries: &[Box<Entry>]) {
        record.begin(ADDING);
        for entry in entries {
            let address = entry.address();
            entry.map.previous.store(self.tail, Ordering::Relaxed);
            entry.map.next.store(0, Ordering::Relaxed);
            if self.tail == 0 {
                record.head.store(address, Ordering::Relaxed);
            } else {
                // SAFETY: Every linked map belongs to a live executable. The
                // registry lock excludes unlink/drop while this pointer is used.
                unsafe { &*(self.tail as *const LinkMap) }
                    .next
                    .store(address, Ordering::Release);
            }
            self.tail = address;
        }
        record.finish();
    }

    /// Unlinks before the executable can release names or image source storage.
    fn remove(&mut self, record: &RuntimeDebugRecord, entries: &[Box<Entry>]) {
        record.begin(DELETING);
        for entry in entries {
            let previous = entry.map.previous.load(Ordering::Relaxed);
            let next = entry.map.next.load(Ordering::Relaxed);
            if previous == 0 {
                record.head.store(next, Ordering::Relaxed);
            } else {
                // SAFETY: The previous map is still linked and retained under
                // the registry lock; removal completes before any owner drops.
                unsafe { &*(previous as *const LinkMap) }
                    .next
                    .store(next, Ordering::Release);
            }
            if next == 0 {
                self.tail = previous;
            } else {
                // SAFETY: The following map is still linked and retained.
                unsafe { &*(next as *const LinkMap) }
                    .previous
                    .store(previous, Ordering::Release);
            }
            entry.map.previous.store(0, Ordering::Relaxed);
            entry.map.next.store(0, Ordering::Relaxed);
        }
        record.finish();
    }
}

/// Native code-object records owned by one HSA executable.
#[derive(Default)]
pub(crate) struct ExecutableDebugInfo {
    /// Boxes keep every native map address stable as the vector grows/moves.
    #[allow(
        clippy::vec_box,
        reason = "native debugger links require stable element addresses"
    )]
    entries: Vec<Box<Entry>>,
    /// Only frozen executable records enter the native linked list.
    published: bool,
}

impl ExecutableDebugInfo {
    /// Completes all fallible preparation before native executable freeze begins.
    pub(crate) fn prepare(
        &mut self,
        executable: &rocloader::Executable<crate::loader::LoadedMetadata>,
    ) -> Result<(), Status> {
        if self.published {
            return Err(ERROR);
        }
        let mut entries = Vec::new();
        entries
            .try_reserve_exact(executable.loaded().len())
            .map_err(|_| OUT_OF_RESOURCES)?;
        for object in executable.loaded() {
            entries.push(Entry::new(
                object
                    .image
                    .device_base()
                    .wrapping_sub(object.image.virtual_base()),
                &object.metadata.uri,
            )?);
        }
        self.entries = entries;
        Ok(())
    }

    /// Publishes only after image copies and code-cache invalidations complete.
    pub(crate) fn publish(&mut self) {
        if self.published || self.entries.is_empty() {
            return;
        }
        REGISTRY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .add(&_amdgpu_r_debug, &self.entries);
        self.published = true;
    }

    /// Removes debugger reachability before code-object owners are destroyed.
    pub(crate) fn unpublish(&mut self) {
        if !self.published {
            return;
        }
        REGISTRY
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&_amdgpu_r_debug, &self.entries);
        self.published = false;
    }
}

impl Drop for ExecutableDebugInfo {
    fn drop(&mut self) {
        self.unpublish();
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;
    use std::mem::{offset_of, size_of};

    #[test]
    fn debugger_records_match_native_svr4_and_amd_layouts() {
        assert_eq!(size_of::<RuntimeDebugRecord>(), 40);
        assert_eq!(offset_of!(RuntimeDebugRecord, head), 8);
        assert_eq!(offset_of!(RuntimeDebugRecord, breakpoint), 16);
        assert_eq!(offset_of!(RuntimeDebugRecord, state), 24);
        assert_eq!(offset_of!(RuntimeDebugRecord, linker_base), 32);
        assert_eq!(size_of::<LinkMap>(), 40);
        assert_eq!(offset_of!(LinkMap, name), 8);
        assert_eq!(offset_of!(LinkMap, next), 24);
        assert_eq!(offset_of!(LinkMap, previous), 32);
        assert_eq!(RuntimeDebugRecord::new().version, 11);
    }

    #[test]
    fn removing_one_executable_preserves_other_stable_links_and_names() {
        let record = RuntimeDebugRecord::new();
        let mut registry = Registry { tail: 0 };
        let first = [Entry::new(0_u64.wrapping_sub(0x100), b"memory://1#offset=0&size=8").unwrap()];
        let second = [Entry::new(0x2000, b"file:///code.hsaco").unwrap()];
        registry.add(&record, &first);
        registry.add(&record, &second);
        assert_eq!(record.head.load(Ordering::Acquire), first[0].address());
        assert_eq!(
            first[0].map.next.load(Ordering::Acquire),
            second[0].address()
        );
        assert_eq!(
            second[0].map.previous.load(Ordering::Acquire),
            first[0].address()
        );
        assert_eq!(first[0].map.load_bias, u64::MAX - 255);
        registry.remove(&record, &first);
        drop(first);
        assert_eq!(record.head.load(Ordering::Acquire), second[0].address());
        assert_eq!(second[0].map.previous.load(Ordering::Acquire), 0);
        // SAFETY: The still-published entry retains its NUL-terminated name.
        let name = unsafe { std::ffi::CStr::from_ptr(second[0].map.name as *const _) };
        assert_eq!(name.to_bytes(), b"file:///code.hsaco");
        registry.remove(&record, &second);
        assert_eq!(record.head.load(Ordering::Acquire), 0);
        assert_eq!(registry.tail, 0);
        assert_eq!(record.state.load(Ordering::Acquire), CONSISTENT);
    }
}
