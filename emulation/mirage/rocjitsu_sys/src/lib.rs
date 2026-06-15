//! `rocjitsu_sys` — runtime FFI bindings to the rocjitsu VM C API.
//!
//! mirage drives the rocjitsu functional emulator directly through its
//! public C API (`rj_vm_*`, declared in `rocjitsu/vm/rj_vm.h`) instead
//! of shelling out to the `rocjitsu` command-line tool. This crate is
//! the thin, unsafe binding layer between Rust and that C API.
//!
//! # Why runtime loading?
//!
//! The rocjitsu shared library is *discovered at runtime* — it ships in
//! a ROCm wheel / system install and is never present when mirage is
//! built, so we cannot link it at build time. Instead we `dlopen` it
//! (via [`libloading`]) and resolve the handful of `rj_vm_*` symbols we
//! need. The single self-contained `librocjitsu_kmd.so` exports the full
//! VM API in addition to the LD_PRELOAD interposer, so loading that one
//! library is enough to both interpose a workload *and* host the daemon.
//!
//! # Safety
//!
//! Every function here is `unsafe`: callers must uphold the C API's
//! contract (valid pointers, correct lifetimes, single-threaded VM
//! creation, etc.). Higher layers (`mirage_rocjitsu`) wrap these in
//! safe, RAII-managed abstractions.

use std::ffi::{CStr, OsStr};
use std::os::raw::{c_char, c_int, c_void};

/// Status codes returned by the rocjitsu C API (`rj_status_t`).
pub type RjStatus = c_int;

/// Operation completed successfully.
pub const ROCJITSU_STATUS_SUCCESS: RjStatus = 0;
/// Unspecified error.
pub const ROCJITSU_STATUS_ERROR: RjStatus = 1;
/// One or more arguments are invalid.
pub const ROCJITSU_STATUS_INVALID_ARGUMENT: RjStatus = 2;
/// Insufficient resources to complete the operation.
pub const ROCJITSU_STATUS_OUT_OF_RESOURCES: RjStatus = 3;
/// The supplied code object is malformed or unsupported.
pub const ROCJITSU_STATUS_INVALID_CODE_OBJECT: RjStatus = 4;
/// A required file could not be opened or read.
pub const ROCJITSU_STATUS_INVALID_FILE: RjStatus = 5;

/// Platform-specific handle type (`rj_handle_t`); an fd on Linux.
pub type RjHandle = c_int;

/// Opaque VM handle (`rj_vm_t`). Only ever held behind a pointer.
#[repr(C)]
pub struct RjVm {
    _private: [u8; 0],
}

/// VM creation mode (`rj_vm_mode_t`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RjVmMode {
    /// Standalone simulation driven by the caller.
    Default = 0,
    /// Single-process serving for an LD_PRELOAD interposer (in-process).
    Local = 1,
    /// Multi-process serving (daemon): GPU allocations are memfd-backed
    /// for cross-process sharing via `SCM_RIGHTS`.
    Daemon = 2,
}

/// Device command descriptor (`rj_vm_cmd_t`).
#[repr(C)]
#[derive(Debug)]
pub struct RjVmCmd {
    /// Platform-specific command number (an `AMDKFD_IOC_*` ioctl).
    pub cmd: u32,
    /// Command arguments buffer (with inlined arrays).
    pub buf: *mut c_void,
    /// Total size of the arguments buffer.
    pub buf_size: usize,
    /// `[out]` Return code (0 on success, negative errno on failure).
    pub result: i32,
    /// `[out]` Backing handle for shareable allocations, or -1.
    pub shared_handle: RjHandle,
}

/// Device memory mapping descriptor (`rj_vm_map_t`).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct RjVmMap {
    /// Requested mapping address.
    pub addr: u64,
    /// Length in bytes to map.
    pub length: u64,
    /// Platform-specific offset encoding.
    pub offset: i64,
    /// Memory protection flags.
    pub prot: u32,
    /// Mapping flags.
    pub flags: u32,
    /// `[out]` Address the mapping was placed at.
    pub mapped_addr: u64,
}

/// Device memory unmapping descriptor (`rj_vm_unmap_t`).
#[repr(C)]
#[derive(Debug, Default, Clone, Copy)]
pub struct RjVmUnmap {
    /// Address of the mapping to unmap.
    pub addr: u64,
    /// Length in bytes to unmap.
    pub length: u64,
}

// Raw C function-pointer signatures for the symbols we resolve.
type FnVmCreate = unsafe extern "C" fn(*const c_char, RjVmMode, *mut *mut RjVm) -> RjStatus;
type FnVmCreateFromString =
    unsafe extern "C" fn(*const c_char, RjVmMode, *mut *mut RjVm) -> RjStatus;
type FnVmRun = unsafe extern "C" fn(*mut RjVm, *mut u64) -> RjStatus;
type FnVmRequestExit = unsafe extern "C" fn(*mut RjVm, *const c_char);
type FnVmDestroy = unsafe extern "C" fn(*mut RjVm);
type FnVmDeviceOpen = unsafe extern "C" fn(*mut RjVm, *mut u32) -> RjStatus;
type FnVmDeviceClose = unsafe extern "C" fn(*mut RjVm, u32) -> RjStatus;
type FnVmExecuteAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmCmd) -> RjStatus;
type FnVmDeviceMapAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmMap) -> RjStatus;
type FnVmDeviceUnmapAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmUnmap) -> RjStatus;
type FnVmGpuId = unsafe extern "C" fn(*mut RjVm, *mut u32) -> RjStatus;
type FnVmTopologyPath = unsafe extern "C" fn(*mut RjVm, *mut *const c_char) -> RjStatus;
type FnVmDrmPath = unsafe extern "C" fn(*mut RjVm, *mut *const c_char) -> RjStatus;
type FnVmGetSharedMemAs = unsafe extern "C" fn(*mut RjVm, u32, i64, *mut RjHandle) -> RjStatus;

/// A loaded rocjitsu shared library with its `rj_vm_*` entry points
/// resolved.
///
/// The [`libloading::Library`] is kept alive for the lifetime of this
/// struct so the resolved function pointers remain valid. All methods
/// are `unsafe`: they call directly into C and require the caller to
/// uphold the rocjitsu API contract.
pub struct Lib {
    // Resolved function pointers. The owning library must outlive them,
    // so it is kept in `_lib` and dropped last.
    vm_create: FnVmCreate,
    vm_create_from_string: FnVmCreateFromString,
    vm_run: FnVmRun,
    vm_request_exit: FnVmRequestExit,
    vm_destroy: FnVmDestroy,
    vm_device_open: FnVmDeviceOpen,
    vm_device_close: FnVmDeviceClose,
    vm_execute_as: FnVmExecuteAs,
    vm_device_map_as: FnVmDeviceMapAs,
    vm_device_unmap_as: FnVmDeviceUnmapAs,
    vm_gpu_id: FnVmGpuId,
    vm_topology_path: FnVmTopologyPath,
    vm_drm_path: FnVmDrmPath,
    vm_get_shared_mem_as: FnVmGetSharedMemAs,
    _lib: libloading::Library,
}

// The resolved entry points are plain C function pointers and the VM
// they operate on is internally synchronised by rocjitsu (the daemon
// shares one VM across an engine thread and many client threads via the
// `*_as` API), so the handle is safe to move and share across threads.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

impl Lib {
    /// Load the rocjitsu shared library at `path` and resolve the
    /// `rj_vm_*` entry points.
    ///
    /// # Safety
    /// Loading an arbitrary shared library runs its initialisers; the
    /// caller must ensure `path` is a trusted rocjitsu library.
    pub unsafe fn open(path: impl AsRef<OsStr>) -> Result<Self, libloading::Error> {
        unsafe {
            let lib = libloading::Library::new(path.as_ref())?;
            // `*symbol` copies out the raw fn pointer; the symbol's
            // borrow of `lib` ends here but `lib` is moved into the
            // returned struct, keeping the code mapped.
            let vm_create = *lib.get::<FnVmCreate>(b"rj_vm_create\0")?;
            let vm_create_from_string =
                *lib.get::<FnVmCreateFromString>(b"rj_vm_create_from_string\0")?;
            let vm_run = *lib.get::<FnVmRun>(b"rj_vm_run\0")?;
            let vm_request_exit = *lib.get::<FnVmRequestExit>(b"rj_vm_request_exit\0")?;
            let vm_destroy = *lib.get::<FnVmDestroy>(b"rj_vm_destroy\0")?;
            let vm_device_open = *lib.get::<FnVmDeviceOpen>(b"rj_vm_device_open\0")?;
            let vm_device_close = *lib.get::<FnVmDeviceClose>(b"rj_vm_device_close\0")?;
            let vm_execute_as = *lib.get::<FnVmExecuteAs>(b"rj_vm_execute_as\0")?;
            let vm_device_map_as = *lib.get::<FnVmDeviceMapAs>(b"rj_vm_device_map_as\0")?;
            let vm_device_unmap_as = *lib.get::<FnVmDeviceUnmapAs>(b"rj_vm_device_unmap_as\0")?;
            let vm_gpu_id = *lib.get::<FnVmGpuId>(b"rj_vm_gpu_id\0")?;
            let vm_topology_path = *lib.get::<FnVmTopologyPath>(b"rj_vm_topology_path\0")?;
            let vm_drm_path = *lib.get::<FnVmDrmPath>(b"rj_vm_drm_path\0")?;
            let vm_get_shared_mem_as =
                *lib.get::<FnVmGetSharedMemAs>(b"rj_vm_get_shared_mem_as\0")?;
            Ok(Self {
                vm_create,
                vm_create_from_string,
                vm_run,
                vm_request_exit,
                vm_destroy,
                vm_device_open,
                vm_device_close,
                vm_execute_as,
                vm_device_map_as,
                vm_device_unmap_as,
                vm_gpu_id,
                vm_topology_path,
                vm_drm_path,
                vm_get_shared_mem_as,
                _lib: lib,
            })
        }
    }

    /// Create a VM from a JSON config file. Returns the status and the
    /// (possibly null) VM handle.
    ///
    /// # Safety
    /// `json_path` must be a valid C string path; the returned VM must
    /// eventually be released with [`Lib::vm_destroy`].
    pub unsafe fn vm_create(&self, json_path: &CStr, mode: RjVmMode) -> (RjStatus, *mut RjVm) {
        let mut vm: *mut RjVm = std::ptr::null_mut();
        let status = unsafe { (self.vm_create)(json_path.as_ptr(), mode, &mut vm) };
        (status, vm)
    }

    /// Create a VM from a JSON config string.
    ///
    /// # Safety
    /// See [`Lib::vm_create`].
    pub unsafe fn vm_create_from_string(
        &self,
        json: &CStr,
        mode: RjVmMode,
    ) -> (RjStatus, *mut RjVm) {
        let mut vm: *mut RjVm = std::ptr::null_mut();
        let status = unsafe { (self.vm_create_from_string)(json.as_ptr(), mode, &mut vm) };
        (status, vm)
    }

    /// Run the simulation engine until [`Lib::vm_request_exit`] is
    /// called (or the configured tick limit is reached). Blocks.
    ///
    /// # Safety
    /// `vm` must be a live handle from [`Lib::vm_create`].
    pub unsafe fn vm_run(&self, vm: *mut RjVm) -> RjStatus {
        unsafe { (self.vm_run)(vm, std::ptr::null_mut()) }
    }

    /// Ask the engine to stop at the next opportunity. Thread-safe.
    ///
    /// # Safety
    /// `vm` must be a live handle; `reason` (if any) a valid C string.
    pub unsafe fn vm_request_exit(&self, vm: *mut RjVm, reason: &CStr) {
        unsafe { (self.vm_request_exit)(vm, reason.as_ptr()) }
    }

    /// Destroy a VM handle.
    ///
    /// # Safety
    /// `vm` must not be used after this call.
    pub unsafe fn vm_destroy(&self, vm: *mut RjVm) {
        unsafe { (self.vm_destroy)(vm) }
    }

    /// Open the simulated device, creating a new KFD process. Returns
    /// the status and the new process id.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_device_open(&self, vm: *mut RjVm) -> (RjStatus, u32) {
        let mut pid: u32 = 0;
        let status = unsafe { (self.vm_device_open)(vm, &mut pid) };
        (status, pid)
    }

    /// Close a KFD process by id.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_device_close(&self, vm: *mut RjVm, process_id: u32) -> RjStatus {
        unsafe { (self.vm_device_close)(vm, process_id) }
    }

    /// Execute a device command on behalf of `process_id` (daemon mode).
    ///
    /// # Safety
    /// `vm` must be live and `cmd` a valid, writable descriptor whose
    /// `buf`/`buf_size` describe an accessible buffer.
    pub unsafe fn vm_execute_as(
        &self,
        vm: *mut RjVm,
        process_id: u32,
        cmd: *mut RjVmCmd,
    ) -> RjStatus {
        unsafe { (self.vm_execute_as)(vm, process_id, cmd) }
    }

    /// Map device memory on behalf of `process_id` (daemon mode).
    ///
    /// # Safety
    /// `vm` must be live and `map` a valid, writable descriptor.
    pub unsafe fn vm_device_map_as(
        &self,
        vm: *mut RjVm,
        process_id: u32,
        map: *mut RjVmMap,
    ) -> RjStatus {
        unsafe { (self.vm_device_map_as)(vm, process_id, map) }
    }

    /// Unmap device memory on behalf of `process_id` (daemon mode).
    ///
    /// # Safety
    /// `vm` must be live and `unmap` a valid descriptor.
    pub unsafe fn vm_device_unmap_as(
        &self,
        vm: *mut RjVm,
        process_id: u32,
        unmap: *mut RjVmUnmap,
    ) -> RjStatus {
        unsafe { (self.vm_device_unmap_as)(vm, process_id, unmap) }
    }

    /// Get the KFD `gpu_id` for the simulated device.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_gpu_id(&self, vm: *mut RjVm) -> (RjStatus, u32) {
        let mut gpu_id: u32 = 0;
        let status = unsafe { (self.vm_gpu_id)(vm, &mut gpu_id) };
        (status, gpu_id)
    }

    /// Get the sysfs topology directory path (owned by the VM).
    ///
    /// # Safety
    /// `vm` must be a live handle. The returned string borrows VM-owned
    /// memory valid until the VM is destroyed.
    pub unsafe fn vm_topology_path(&self, vm: *mut RjVm) -> Option<&CStr> {
        let mut ptr: *const c_char = std::ptr::null();
        let status = unsafe { (self.vm_topology_path)(vm, &mut ptr) };
        if status != ROCJITSU_STATUS_SUCCESS || ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) })
    }

    /// Get the DRM sysfs directory path (owned by the VM).
    ///
    /// # Safety
    /// See [`Lib::vm_topology_path`].
    pub unsafe fn vm_drm_path(&self, vm: *mut RjVm) -> Option<&CStr> {
        let mut ptr: *const c_char = std::ptr::null();
        let status = unsafe { (self.vm_drm_path)(vm, &mut ptr) };
        if status != ROCJITSU_STATUS_SUCCESS || ptr.is_null() {
            return None;
        }
        Some(unsafe { CStr::from_ptr(ptr) })
    }

    /// Get the backing memory handle (memfd) for `process_id` at the
    /// given KFD mmap `offset`, or `None` when there is no backing fd.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_get_shared_mem_as(
        &self,
        vm: *mut RjVm,
        process_id: u32,
        offset: i64,
    ) -> Option<RjHandle> {
        let mut handle: RjHandle = -1;
        let status = unsafe { (self.vm_get_shared_mem_as)(vm, process_id, offset, &mut handle) };
        if status != ROCJITSU_STATUS_SUCCESS || handle < 0 {
            return None;
        }
        Some(handle)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C `rj_vm_*` structs must match the rocjitsu headers exactly,
    /// or every FFI call corrupts memory. Pin the sizes the daemon RPC
    /// code relies on.
    #[test]
    fn struct_sizes_match_c_abi() {
        assert_eq!(std::mem::size_of::<RjVmMap>(), 40);
        assert_eq!(std::mem::size_of::<RjVmUnmap>(), 16);
        // rj_vm_cmd_t: u32 + (pad) + ptr + usize + i32 + i32 on 64-bit.
        assert_eq!(std::mem::size_of::<RjVmCmd>(), 32);
        assert_eq!(RjVmMode::Daemon as i32, 2);
    }
}
