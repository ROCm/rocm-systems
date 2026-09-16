//! `rocjitsu_sys` — runtime FFI bindings to the rocjitsu C API.
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
//! need. The single self-contained `librocjitsu.so` exports the full
//! VM API in addition to the LD_PRELOAD interposer, so loading that one
//! library is enough to both interpose a workload *and* host the daemon.
//!
//! # Safety
//!
//! The raw [`Lib`] bindings are `unsafe`: callers must uphold the C API's
//! contract (valid pointers, correct lifetimes, single-threaded VM
//! creation, and so on).
//!
//! Safe, RAII-managed wrappers live here too, in [`daemon`], rather than
//! in the crates that use them. This crate is the one place in the
//! workspace permitted to write `unsafe`; every other crate is
//! `forbid(unsafe_code)`. Keeping the wrapper here means the invariants
//! and the `unsafe` that relies on them sit in the same file and are
//! reviewed together, instead of the invariants living in a doc comment
//! that a caller in another crate has to remember to honour.

use std::ffi::{CStr, OsStr};
use std::fmt;
use std::mem::offset_of;
use std::os::raw::{c_char, c_int, c_void};

type FnGetVersionString = unsafe extern "C" fn() -> *const c_char;

/// Load and copy the formatted build identity exported by `librocjitsu.so`.
///
/// # Errors
///
/// Returns a diagnostic when the library cannot be loaded, predates the version
/// API, or violates the API's non-null string contract.
pub fn version_string(path: impl AsRef<OsStr>) -> Result<String, String> {
    // SAFETY: Mirage's RocJITsu discovery selects the library. The resolved
    // symbol has the C signature declared in rj_version.h, and its static string
    // is copied while the library remains loaded.
    unsafe {
        let library = libloading::Library::new(path.as_ref()).map_err(|error| error.to_string())?;
        let get_version = *library
            .get::<FnGetVersionString>(b"rj_get_version_string\0")
            .map_err(|error| error.to_string())?;
        let version = get_version();
        if version.is_null() {
            return Err("rj_get_version_string returned null".to_string());
        }
        Ok(CStr::from_ptr(version).to_string_lossy().into_owned())
    }
}

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
/// The requested operation is not supported by this configuration.
pub const ROCJITSU_STATUS_UNSUPPORTED: RjStatus = 6;

/// Platform-specific handle type (`rj_handle_t`); an fd on Linux.
pub type RjHandle = c_int;

/// Opaque VM handle (`rj_vm_t`). Only ever held behind a pointer.
///
/// `Debug` prints the type name and nothing else — there is nothing to
/// print. It exists so a caller can derive `Debug` on a struct holding
/// `*mut RjVm` without having to hand-write an impl.
#[repr(C)]
#[derive(Debug)]
pub struct RjVm {
    _private: [u8; 0],
}

/// Opaque daemon handle (`rj_daemon_t`). Only ever held behind a pointer.
///
/// `Debug` is present for the same reason as on [`RjVm`].
#[repr(C)]
#[derive(Debug)]
pub struct RjDaemon {
    _private: [u8; 0],
}

/// Observable daemon lifecycle state (`rj_daemon_status_t`).
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RjDaemonStatus {
    Stopped = 0,
    Starting = 1,
    Running = 2,
    Stopping = 3,
    Error = 4,
}

impl TryFrom<c_int> for RjDaemonStatus {
    type Error = c_int;

    fn try_from(value: c_int) -> Result<Self, c_int> {
        match value {
            0 => Ok(Self::Stopped),
            1 => Ok(Self::Starting),
            2 => Ok(Self::Running),
            3 => Ok(Self::Stopping),
            4 => Ok(Self::Error),
            value => Err(value),
        }
    }
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
    /// `[in/out]` Client-provided fd (e.g. a debugger notifier), or -1.
    /// In daemon mode the VM substitutes it into DBG_TRAP ENABLE and, on
    /// adoption, clears it to -1 so the caller does not close it.
    pub in_handle: RjHandle,
    /// `[in/out]` Debugger-authorized target `/proc/pid/mem` fd, or -1.
    pub in_mem_handle: RjHandle,
    /// `[in]` Pinned target `/proc/pid` directory fd, or -1.
    pub in_proc_handle: RjHandle,
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
    /// `[out]` errno captured at the failing mmap (0 on success).
    pub map_errno: i32,
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

/// Simulated GPU metadata (`rj_vm_gpu_info_t`, 312 bytes).
///
/// Sent verbatim inside the daemon handshake response so a workload's
/// interposer can emulate libdrm/DRM device queries client-side. The
/// layout must match `rocjitsu/vm/rj_vm.h` byte-for-byte.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RjVmGpuInfo {
    /// Non-zero when this payload was populated by the VM.
    pub present: u32,
    pub gpu_id: u32,
    pub gfx_target_version: u32,
    pub vendor_id: u32,
    pub device_id: u32,
    pub family_id: u32,
    pub unique_id: u64,
    pub location_id: u32,
    pub domain: u32,
    pub hive_id: u64,
    pub drm_render_minor: u32,
    pub revision_id: u32,
    pub pci_revision_id: u32,
    pub simd_count: u32,
    pub max_waves_per_simd: u32,
    pub num_shader_engines: u32,
    pub num_shader_arrays_per_engine: u32,
    pub num_cu_per_sh: u32,
    pub simd_per_cu: u32,
    pub wave_front_size: u32,
    pub num_xcc: u32,
    pub max_slots_scratch_cu: u32,
    pub local_mem_size: u64,
    pub vram_type: u32,
    pub lds_size_kb: u32,
    pub mem_width: u32,
    pub mem_clk_max: u32,
    pub l1_size_kb: u32,
    pub l1_line_size: u32,
    pub l1_assoc: u32,
    pub l2_size_kb: u32,
    pub l2_line_size: u32,
    pub l2_assoc: u32,
    pub num_sdma_engines: u32,
    pub num_sdma_xgmi_engines: u32,
    pub num_cp_queues: u32,
    pub max_engine_clk_fcompute: u32,
    pub capability: u32,
    pub capability2: u32,
    pub debug_prop: u64,
    pub fw_version: u32,
    pub sdma_fw_version: u32,
    pub marketing_name: [c_char; 128],
}

impl Default for RjVmGpuInfo {
    fn default() -> Self {
        // All-zero is a valid "absent" payload (`present == 0`).
        unsafe { std::mem::zeroed() }
    }
}

impl RjVmGpuInfo {
    /// The raw little-/native-endian bytes of this struct, for wire
    /// serialisation. Sound because the struct is `#[repr(C)]` POD that
    /// matches `rj_vm_gpu_info_t` exactly.
    pub fn as_bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, size_of::<Self>()) }
    }
}

// ---------------------------------------------------------------------
// C ABI layout assertions
// ---------------------------------------------------------------------
//
// Every `#[repr(C)]` type above is written into by C. A field that is
// missing, reordered, or the wrong width does not fail to compile and
// does not trip a lint: `#[repr(C)]` describes the layout Rust *will*
// produce, not the layout C expects, so the two silently disagree and the
// C side reads or writes at offsets the Rust allocation may not even
// cover. `rj_vm_map_t` was exactly that — the Rust struct lacked the
// trailing `map_errno`, so every `rj_vm_device_map*` call wrote four
// bytes past the end of the caller's `RjVmMap`.
//
// The numbers below were read off
// `rocjitsu/lib/rocjitsu/include/rocjitsu/vm/rj_vm.h` (`rj_vm_cmd_t` at
// line 56, `rj_vm_map_t` at 68, `rj_vm_unmap_t` at 79, `rj_vm_gpu_info_t`
// at 85) and `rocjitsu/daemon/rj_daemon.h` (`rj_daemon_status_t` at 26),
// as a 64-bit LP64 target sees them, and confirmed against a `sizeof` /
// `offsetof` probe compiled from those headers.
//
// Be clear about the guarantee: these are a snapshot. They catch a
// careless edit *on the Rust side* — a dropped field, a `u32` widened to
// a `u64`, a field moved across a padding boundary — at compile time, on
// every build, with no rocjitsu installed. They cannot notice the *C*
// side changing, because nothing here reads the header. Neither can they
// catch a swap of two same-width neighbours, which does not move
// anything. For both of those, `tests/abi_layout.rs` compiles a probe
// against the real header when one can be found and compares field by
// name; it skips when the header or a C compiler is absent, which on a
// mirage checkout is the common case.
//
// So: if a number here changes, the header is the authority. Re-read it,
// and fix the struct rather than the assertion.

const _: () = {
    // rj_vm_cmd_t — 48 bytes; `cmd` is followed by 4 bytes of padding so
    // the `buf` pointer lands on an 8-byte boundary, and the three
    // trailing handles are padded out to a multiple of 8.
    assert!(size_of::<RjVmCmd>() == 48);
    assert!(offset_of!(RjVmCmd, cmd) == 0);
    assert!(offset_of!(RjVmCmd, buf) == 8);
    assert!(offset_of!(RjVmCmd, buf_size) == 16);
    assert!(offset_of!(RjVmCmd, result) == 24);
    assert!(offset_of!(RjVmCmd, shared_handle) == 28);
    assert!(offset_of!(RjVmCmd, in_handle) == 32);
    assert!(offset_of!(RjVmCmd, in_mem_handle) == 36);
    assert!(offset_of!(RjVmCmd, in_proc_handle) == 40);

    // rj_vm_map_t — 48 bytes. `prot`/`flags` pair up into one 8-byte
    // slot, and the trailing `map_errno` is padded out to 48.
    assert!(size_of::<RjVmMap>() == 48);
    assert!(offset_of!(RjVmMap, addr) == 0);
    assert!(offset_of!(RjVmMap, length) == 8);
    assert!(offset_of!(RjVmMap, offset) == 16);
    assert!(offset_of!(RjVmMap, prot) == 24);
    assert!(offset_of!(RjVmMap, flags) == 28);
    assert!(offset_of!(RjVmMap, mapped_addr) == 32);
    assert!(offset_of!(RjVmMap, map_errno) == 40);

    // rj_vm_unmap_t — two u64s, no padding.
    assert!(size_of::<RjVmUnmap>() == 16);
    assert!(offset_of!(RjVmUnmap, addr) == 0);
    assert!(offset_of!(RjVmUnmap, length) == 8);

    // rj_vm_gpu_info_t — 312 bytes, which is also the size of the
    // `RpcGpuInfo` the daemon handshake embeds verbatim (there is a
    // matching static_assert in the C++ `rpc.h`), so this number is load
    // bearing for the wire protocol as well as for the FFI call.
    //
    // The u32 runs in this struct are interrupted by four u64s; those are
    // the only places padding can appear, so they are the offsets worth
    // pinning. Each one below sits at a position that is already 8-byte
    // aligned, i.e. the C struct happens to have no padding holes at all
    // — if an edit inserts or removes a u32 anywhere before one of them,
    // the anchor moves and this block fails.
    assert!(size_of::<RjVmGpuInfo>() == 312);
    assert!(align_of::<RjVmGpuInfo>() == 8);
    assert!(offset_of!(RjVmGpuInfo, present) == 0);
    assert!(offset_of!(RjVmGpuInfo, unique_id) == 24);
    assert!(offset_of!(RjVmGpuInfo, hive_id) == 40);
    assert!(offset_of!(RjVmGpuInfo, local_mem_size) == 96);
    assert!(offset_of!(RjVmGpuInfo, debug_prop) == 168);
    assert!(offset_of!(RjVmGpuInfo, marketing_name) == 184);

    // Scalar widths the signatures below depend on: rj_status_t and
    // rj_vm_mode_t are C enums (4 bytes on every target rocjitsu builds
    // for), rj_handle_t is an `int`, and rj_daemon_status_t is an
    // explicit `int32_t`. rj_client_pid_t is `int32_t` too, but it has
    // no named binding here — it appears as a bare `i32` argument in
    // `FnVmDeviceOpen`.
    assert!(size_of::<RjStatus>() == 4);
    assert!(size_of::<RjHandle>() == 4);
    assert!(size_of::<RjVmMode>() == 4);
    assert!(size_of::<RjDaemonStatus>() == 4);

    // Discriminants are part of the ABI just as much as the layouts are.
    assert!(RjVmMode::Default as i32 == 0);
    assert!(RjVmMode::Local as i32 == 1);
    assert!(RjVmMode::Daemon as i32 == 2);
    assert!(RjDaemonStatus::Stopped as i32 == 0);
    assert!(RjDaemonStatus::Starting as i32 == 1);
    assert!(RjDaemonStatus::Running as i32 == 2);
    assert!(RjDaemonStatus::Stopping as i32 == 3);
    assert!(RjDaemonStatus::Error as i32 == 4);
};

// Raw C function-pointer signatures for the symbols we resolve.
type FnVmCreate = unsafe extern "C" fn(*const c_char, RjVmMode, *mut *mut RjVm) -> RjStatus;
type FnVmCreateFromString =
    unsafe extern "C" fn(*const c_char, RjVmMode, *mut *mut RjVm) -> RjStatus;
type FnVmLoadPlugins = unsafe extern "C" fn(*mut RjVm, *const c_char, *const c_char) -> RjStatus;
type FnVmRun = unsafe extern "C" fn(*mut RjVm, *mut u64) -> RjStatus;
type FnVmRequestExit = unsafe extern "C" fn(*mut RjVm, *const c_char);
type FnVmDestroy = unsafe extern "C" fn(*mut RjVm);
type FnVmDeviceOpen = unsafe extern "C" fn(*mut RjVm, i32, *mut u32) -> RjStatus;
type FnVmDeviceClose = unsafe extern "C" fn(*mut RjVm, u32) -> RjStatus;
type FnVmExecuteAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmCmd) -> RjStatus;
type FnVmDeviceMapAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmMap) -> RjStatus;
type FnVmDeviceUnmapAs = unsafe extern "C" fn(*mut RjVm, u32, *mut RjVmUnmap) -> RjStatus;
type FnVmGpuId = unsafe extern "C" fn(*mut RjVm, *mut u32) -> RjStatus;
type FnVmGpuInfo = unsafe extern "C" fn(*mut RjVm, *mut RjVmGpuInfo) -> RjStatus;
type FnVmTopologyPath = unsafe extern "C" fn(*mut RjVm, *mut *const c_char) -> RjStatus;
type FnVmDrmPath = unsafe extern "C" fn(*mut RjVm, *mut *const c_char) -> RjStatus;
type FnVmGetSharedMemAs = unsafe extern "C" fn(*mut RjVm, u32, i64, *mut RjHandle) -> RjStatus;
type FnDaemonStart =
    unsafe extern "C" fn(*const c_char, *const c_char, *mut *mut RjDaemon) -> RjStatus;
type FnDaemonStop = unsafe extern "C" fn(*mut RjDaemon) -> RjStatus;
type FnDaemonStatus = unsafe extern "C" fn(*const RjDaemon) -> c_int;

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
    // Optional: only present in rocjitsu libraries that ship the runtime
    // plugin loader. When absent, plugin selection in the config is a no-op
    // for a C-API host (the daemon), matching an older library.
    vm_load_plugins: Option<FnVmLoadPlugins>,
    vm_run: FnVmRun,
    vm_request_exit: FnVmRequestExit,
    vm_destroy: FnVmDestroy,
    vm_device_open: FnVmDeviceOpen,
    vm_device_close: FnVmDeviceClose,
    vm_execute_as: FnVmExecuteAs,
    vm_device_map_as: FnVmDeviceMapAs,
    vm_device_unmap_as: FnVmDeviceUnmapAs,
    vm_gpu_id: FnVmGpuId,
    // Optional: only present in protocol-v3+ rocjitsu libraries. When
    // absent, daemon clients fall back to a zeroed (absent) gpu_info.
    vm_gpu_info: Option<FnVmGpuInfo>,
    vm_topology_path: FnVmTopologyPath,
    vm_drm_path: FnVmDrmPath,
    vm_get_shared_mem_as: FnVmGetSharedMemAs,
    daemon_start: FnDaemonStart,
    daemon_stop: FnDaemonStop,
    daemon_status: FnDaemonStatus,
    _lib: libloading::Library,
}

// The resolved entry points are plain C function pointers and the VM
// they operate on is internally synchronised by rocjitsu (the daemon
// shares one VM across an engine thread and many client threads via the
// `*_as` API), so the handle is safe to move and share across threads.
unsafe impl Send for Lib {}
unsafe impl Sync for Lib {}

// Printing forty function-pointer addresses would be noise. What a reader
// of a log actually wants to know about a loaded library is which of the
// optional entry points it turned out to have, because that is what
// decides whether plugin selection works and whether a daemon client gets
// real `gpu_info` — and it is the first thing to check when an older
// library behaves unexpectedly.
impl fmt::Debug for Lib {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lib")
            .field("vm_load_plugins", &self.vm_load_plugins.is_some())
            .field("vm_gpu_info", &self.vm_gpu_info.is_some())
            .finish_non_exhaustive()
    }
}

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
            // Optional symbol: tolerate older libraries that predate the loader.
            let vm_load_plugins = lib
                .get::<FnVmLoadPlugins>(b"rj_vm_load_plugins\0")
                .map(|s| *s)
                .ok();
            let vm_run = *lib.get::<FnVmRun>(b"rj_vm_run\0")?;
            let vm_request_exit = *lib.get::<FnVmRequestExit>(b"rj_vm_request_exit\0")?;
            let vm_destroy = *lib.get::<FnVmDestroy>(b"rj_vm_destroy\0")?;
            let vm_device_open = *lib.get::<FnVmDeviceOpen>(b"rj_vm_device_open\0")?;
            let vm_device_close = *lib.get::<FnVmDeviceClose>(b"rj_vm_device_close\0")?;
            let vm_execute_as = *lib.get::<FnVmExecuteAs>(b"rj_vm_execute_as\0")?;
            let vm_device_map_as = *lib.get::<FnVmDeviceMapAs>(b"rj_vm_device_map_as\0")?;
            let vm_device_unmap_as = *lib.get::<FnVmDeviceUnmapAs>(b"rj_vm_device_unmap_as\0")?;
            let vm_gpu_id = *lib.get::<FnVmGpuId>(b"rj_vm_gpu_id\0")?;
            // Optional symbol: tolerate older libraries that predate it.
            let vm_gpu_info = lib.get::<FnVmGpuInfo>(b"rj_vm_gpu_info\0").map(|s| *s).ok();
            let vm_topology_path = *lib.get::<FnVmTopologyPath>(b"rj_vm_topology_path\0")?;
            let vm_drm_path = *lib.get::<FnVmDrmPath>(b"rj_vm_drm_path\0")?;
            let vm_get_shared_mem_as =
                *lib.get::<FnVmGetSharedMemAs>(b"rj_vm_get_shared_mem_as\0")?;
            let daemon_start = *lib.get::<FnDaemonStart>(b"rj_daemon_start\0")?;
            let daemon_stop = *lib.get::<FnDaemonStop>(b"rj_daemon_stop\0")?;
            let daemon_status = *lib.get::<FnDaemonStatus>(b"rj_daemon_status\0")?;
            Ok(Self {
                vm_create,
                vm_create_from_string,
                vm_load_plugins,
                vm_run,
                vm_request_exit,
                vm_destroy,
                vm_device_open,
                vm_device_close,
                vm_execute_as,
                vm_device_map_as,
                vm_device_unmap_as,
                vm_gpu_id,
                vm_gpu_info,
                vm_topology_path,
                vm_drm_path,
                vm_get_shared_mem_as,
                daemon_start,
                daemon_stop,
                daemon_status,
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

    /// Load and attach the execution plugins declared in `config_json`
    /// (its `plugins` / `sinks` sections) to `vm`.
    ///
    /// `plugin_dir`, when non-empty, is a trusted directory the plugin
    /// shared objects are loaded from by explicit path — required in
    /// daemon mode, where the process is not re-`exec`'d and so cannot
    /// rely on a launcher-populated `LD_LIBRARY_PATH`.
    ///
    /// Returns `None` when the loaded library predates the
    /// `rj_vm_load_plugins` symbol (an older rocjitsu without the runtime
    /// plugin loader); otherwise the C API status.
    ///
    /// # Safety
    /// `vm` must be a live handle from [`Lib::vm_create`].
    pub unsafe fn vm_load_plugins(
        &self,
        vm: *mut RjVm,
        config_json: &CStr,
        plugin_dir: &CStr,
    ) -> Option<RjStatus> {
        let load = self.vm_load_plugins?;
        Some(unsafe { load(vm, config_json.as_ptr(), plugin_dir.as_ptr()) })
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
    ///
    /// `client_pid` is the connecting client's OS PID (enables daemon-mode
    /// cross-process memory access and process reuse); pass 0 in local mode.
    pub unsafe fn vm_device_open(&self, vm: *mut RjVm, client_pid: i32) -> (RjStatus, u32) {
        let mut pid: u32 = 0;
        let status = unsafe { (self.vm_device_open)(vm, client_pid, &mut pid) };
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

    /// Fetch the simulated device metadata (`rj_vm_gpu_info`). Returns
    /// `None` when the loaded library predates the symbol or the call
    /// fails; callers should then send a zeroed (absent) payload.
    ///
    /// # Safety
    /// `vm` must be a live handle.
    pub unsafe fn vm_gpu_info(&self, vm: *mut RjVm) -> Option<RjVmGpuInfo> {
        let f = self.vm_gpu_info?;
        let mut info = RjVmGpuInfo::default();
        let status = unsafe { f(vm, &mut info) };
        if status != ROCJITSU_STATUS_SUCCESS {
            return None;
        }
        Some(info)
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

    /// Start a daemon and return its opaque handle.
    ///
    /// # Safety
    /// `json` and `socket_path` must be valid C strings. A non-null returned
    /// handle must be released exactly once with [`Lib::daemon_stop`].
    pub unsafe fn daemon_start(
        &self,
        json: &CStr,
        socket_path: &CStr,
    ) -> (RjStatus, *mut RjDaemon) {
        let mut daemon = std::ptr::null_mut();
        let status =
            unsafe { (self.daemon_start)(json.as_ptr(), socket_path.as_ptr(), &mut daemon) };
        (status, daemon)
    }

    /// Stop a daemon and release its handle.
    ///
    /// # Safety
    /// `daemon` must be null or a live handle from [`Lib::daemon_start`] and
    /// must not be used after this call.
    pub unsafe fn daemon_stop(&self, daemon: *mut RjDaemon) -> RjStatus {
        unsafe { (self.daemon_stop)(daemon) }
    }

    /// Return the current lifecycle state of a daemon.
    ///
    /// Returns the unrecognized raw value when an ABI-incompatible library
    /// produces something other than a valid [`RjDaemonStatus`] discriminant.
    ///
    /// # Safety
    /// `daemon` must be null or remain live for the duration of this call.
    pub unsafe fn daemon_status(&self, daemon: *const RjDaemon) -> Result<RjDaemonStatus, c_int> {
        RjDaemonStatus::try_from(unsafe { (self.daemon_status)(daemon) })
    }
}

pub mod daemon;

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    // The sizes, field offsets and discriminants that used to be checked
    // here at run time are now checked at *compile* time, in the "C ABI
    // layout assertions" block above. A `#[test]` for them was strictly
    // weaker: it only ran when someone ran the suite, and it could not
    // express a field offset without duplicating the struct.

    #[test]
    fn status_codes_match_c_api() {
        const C_STATUS_HEADER: &str = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../rocjitsu/lib/rocjitsu/include/rocjitsu/base/rj_status.h"
        ));
        let c_statuses: Vec<String> = C_STATUS_HEADER
            .lines()
            .map(str::trim)
            .filter(|line| line.starts_with("ROCJITSU_STATUS_"))
            .map(|line| line.trim_end_matches(',').to_owned())
            .collect();
        let rust_statuses = vec![
            format!("ROCJITSU_STATUS_SUCCESS = {ROCJITSU_STATUS_SUCCESS}"),
            format!("ROCJITSU_STATUS_ERROR = {ROCJITSU_STATUS_ERROR}"),
            format!("ROCJITSU_STATUS_INVALID_ARGUMENT = {ROCJITSU_STATUS_INVALID_ARGUMENT}"),
            format!("ROCJITSU_STATUS_OUT_OF_RESOURCES = {ROCJITSU_STATUS_OUT_OF_RESOURCES}"),
            format!("ROCJITSU_STATUS_INVALID_CODE_OBJECT = {ROCJITSU_STATUS_INVALID_CODE_OBJECT}"),
            format!("ROCJITSU_STATUS_INVALID_FILE = {ROCJITSU_STATUS_INVALID_FILE}"),
            format!("ROCJITSU_STATUS_UNSUPPORTED = {ROCJITSU_STATUS_UNSUPPORTED}"),
        ];

        assert_eq!(c_statuses, rust_statuses);
    }

    #[test]
    fn daemon_status_rejects_unknown_discriminants() {
        assert_eq!(RjDaemonStatus::try_from(0), Ok(RjDaemonStatus::Stopped));
        assert_eq!(RjDaemonStatus::try_from(1), Ok(RjDaemonStatus::Starting));
        assert_eq!(RjDaemonStatus::try_from(2), Ok(RjDaemonStatus::Running));
        assert_eq!(RjDaemonStatus::try_from(3), Ok(RjDaemonStatus::Stopping));
        assert_eq!(RjDaemonStatus::try_from(4), Ok(RjDaemonStatus::Error));
        assert_eq!(RjDaemonStatus::try_from(-1), Err(-1));
        assert_eq!(RjDaemonStatus::try_from(5), Err(5));
    }
}
