// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! `ROCr` tool ABI tables and dispatch ownership.
//!
//! Each public HSA symbol jumps through an atomic table-owner slot and then
//! reads the current function pointer. The table given to a tool contains the
//! preceding implementation pointer, so a replacement can call through to the
//! runtime. Initial edits are published after `OnLoad` succeeds; later
//! attachment edits remain visible through the same stable table storage.

use std::collections::HashSet;
use std::ffi::{CStr, CString, c_char, c_void};
use std::mem::size_of;
use std::sync::Mutex;
use std::sync::atomic::{AtomicPtr, Ordering};

mod generated {
    include!(concat!(env!("OUT_DIR"), "/api_trace_generated.rs"));
}

/// `ROCr` tool ABI version prefix for one function table.
#[repr(C)]
#[derive(Clone, Copy)]
struct ApiTableVersion {
    /// Major ABI version for this table.
    major_id: u32,
    /// Byte size of the table, as required by the tool ABI.
    minor_id: u32,
    /// Revision within the major table layout.
    step_id: u32,
    /// Reserved native ABI word.
    reserved: u32,
}

impl ApiTableVersion {
    fn new(major_id: u32, minor_id: usize, step_id: u32) -> Self {
        Self {
            major_id,
            minor_id: minor_id as u32,
            step_id,
            reserved: 0,
        }
    }
}

/// One stable array of interceptable HSA function slots.
#[repr(C)]
struct ApiTable<const N: usize> {
    /// ABI version and complete byte size of this table.
    version: ApiTableVersion,
    /// Mutable function pointers visible to attached tools.
    functions: [*mut c_void; N],
}

impl<const N: usize> ApiTable<N> {
    fn new(major: u32, step: u32, functions: [*mut c_void; N]) -> Self {
        Self {
            version: ApiTableVersion::new(major, size_of::<Self>(), step),
            functions,
        }
    }
}

/// Root table of pointers passed to a `ROCr` tool during attachment.
#[repr(C)]
#[derive(Clone, Copy)]
struct RootTable {
    /// Root ABI version and byte size.
    version: ApiTableVersion,
    /// Core HSA entry points.
    core: *mut ApiTable<{ generated::CORE_COUNT }>,
    /// AMD extension entry points.
    amd_ext: *mut ApiTable<{ generated::AMD_EXT_COUNT }>,
    /// Finalizer extension entry points.
    finalizer_ext: *mut ApiTable<{ generated::FINALIZER_EXT_COUNT }>,
    /// Image extension entry points.
    image_ext: *mut ApiTable<{ generated::IMAGE_EXT_COUNT }>,
    /// Tool support entry points.
    tools: *mut ApiTable<{ generated::TOOLS_COUNT }>,
    /// PC sampling extension entry points.
    pc_sampling_ext: *mut ApiTable<{ generated::PC_SAMPLING_EXT_COUNT }>,
}

/// Pinned storage for the root and every concrete API function table.
struct ApiTables {
    /// Points into the table fields of this same pinned allocation.
    root: RootTable,
    core: ApiTable<{ generated::CORE_COUNT }>,
    amd_ext: ApiTable<{ generated::AMD_EXT_COUNT }>,
    finalizer_ext: ApiTable<{ generated::FINALIZER_EXT_COUNT }>,
    image_ext: ApiTable<{ generated::IMAGE_EXT_COUNT }>,
    tools: ApiTable<{ generated::TOOLS_COUNT }>,
    pc_sampling_ext: ApiTable<{ generated::PC_SAMPLING_EXT_COUNT }>,
}

// SAFETY: Root pointers refer only to fields within the pinned Box<ApiTables>.
// The process tool mutex owns runtime-side mutation. Exported trampolines and
// tool callbacks read aligned function slots in the retained table; attached
// tools own synchronization of their subsequent slot edits.
unsafe impl Send for ApiTables {}

impl ApiTables {
    fn new() -> Box<Self> {
        let original = generated::initial_functions();
        let mut tables = Box::new(Self {
            root: RootTable {
                version: ApiTableVersion::new(3, size_of::<RootTable>(), 1),
                core: std::ptr::null_mut(),
                amd_ext: std::ptr::null_mut(),
                finalizer_ext: std::ptr::null_mut(),
                image_ext: std::ptr::null_mut(),
                tools: std::ptr::null_mut(),
                pc_sampling_ext: std::ptr::null_mut(),
            },
            core: ApiTable::new(2, 0, original.core),
            amd_ext: ApiTable::new(2, generated::AMD_EXT_STEP, original.amd_ext),
            finalizer_ext: ApiTable::new(2, 0, original.finalizer_ext),
            image_ext: ApiTable::new(2, 1, original.image_ext),
            tools: ApiTable::new(1, 0, original.tools),
            pc_sampling_ext: ApiTable::new(1, 0, original.pc_sampling_ext),
        });
        tables.root.core = &raw mut tables.core;
        tables.root.amd_ext = &raw mut tables.amd_ext;
        tables.root.finalizer_ext = &raw mut tables.finalizer_ext;
        tables.root.image_ext = &raw mut tables.image_ext;
        tables.root.tools = &raw mut tables.tools;
        tables.root.pc_sampling_ext = &raw mut tables.pc_sampling_ext;
        tables
    }

    fn functions(&self) -> generated::InitialFunctions {
        generated::InitialFunctions {
            core: self.core.functions,
            amd_ext: self.amd_ext.functions,
            finalizer_ext: self.finalizer_ext.functions,
            image_ext: self.image_ext.functions,
            tools: self.tools.functions,
            pc_sampling_ext: self.pc_sampling_ext.functions,
        }
    }

    fn restore(&mut self, saved: &generated::InitialFunctions) {
        self.core.functions = saved.core;
        self.amd_ext.functions = saved.amd_ext;
        self.finalizer_ext.functions = saved.finalizer_ext;
        self.image_ext.functions = saved.image_ext;
        self.tools.functions = saved.tools;
        self.pc_sampling_ext.functions = saved.pc_sampling_ext;
    }
}

/// Owned `dlopen` reference whose code stays mapped through process exit.
struct ToolLibrary(usize);

impl ToolLibrary {
    fn open(name: &CStr) -> Option<Self> {
        // RTLD_NODELETE retains code until process exit: a shutdown wrapper
        // may still be returning through this library after OnUnload.
        // SAFETY: name is NUL terminated and dlopen owns its returned handle.
        let handle = unsafe {
            libc::dlopen(
                name.as_ptr(),
                libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NODELETE,
            )
        };
        (!handle.is_null()).then_some(Self(handle as usize))
    }

    fn open_loaded(name: &CStr) -> Option<Self> {
        // SAFETY: RTLD_NOLOAD resolves only an existing process mapping and
        // takes one reference when it succeeds.
        let handle = unsafe {
            libc::dlopen(
                name.as_ptr(),
                libc::RTLD_NOLOAD | libc::RTLD_NOW | libc::RTLD_LOCAL | libc::RTLD_NODELETE,
            )
        };
        (!handle.is_null()).then_some(Self(handle as usize))
    }

    fn symbol(&self, name: &CStr) -> *mut c_void {
        // SAFETY: The handle remains owned by this ToolLibrary.
        unsafe { libc::dlsym(self.0 as *mut c_void, name.as_ptr()) }
    }
}

impl Drop for ToolLibrary {
    fn drop(&mut self) {
        // SAFETY: This closes one owned dlopen reference. RTLD_NODELETE keeps
        // code mapped while a tool wrapper can still be on the call stack.
        unsafe { libc::dlclose(self.0 as *mut c_void) };
    }
}

/// Process-owned API tables and attached tool libraries.
struct ToolState {
    /// Pinned function tables published to tools and HSA trampolines.
    tables: Box<ApiTables>,
    /// Libraries whose callbacks may still be in flight.
    libraries: Vec<ToolLibrary>,
    /// Optional library that supplied the tool registration entry point.
    registration: Option<ToolLibrary>,
    /// Whether this tool set has completed initialization.
    active: bool,
}

/// Candidate tool and its requested attachment order.
struct ToolCandidate {
    /// Opened tool library.
    library: ToolLibrary,
    /// Name supplied to its load callback.
    name: CString,
    /// Ordering key used before calling tool entry points.
    order: u64,
}

static STATE: Mutex<Option<ToolState>> = Mutex::new(None);
static SCRATCH_TABLE: AtomicPtr<c_void> = AtomicPtr::new(std::ptr::null_mut());

pub(crate) fn scratch_callback(index: usize) -> Option<unsafe extern "C" fn(*const c_void) -> u32> {
    if index >= generated::TOOLS_COUNT {
        return None;
    }
    let table = SCRATCH_TABLE.load(Ordering::Acquire);
    if table.is_null() {
        return None;
    }
    // SAFETY: The pinned API table has TOOLS_COUNT pointer slots. Tools may
    // replace them after registration; read the current aligned slot so late
    // attachment is visible without taking the tool-loader mutex.
    let pointer = unsafe { table.cast::<*mut c_void>().add(index).read_volatile() };
    if pointer.is_null() {
        return None;
    }
    // SAFETY: The tool table field has the hsa_amd_tool_event signature.
    // RTLD_NODELETE retains registered tool code through callback delivery.
    Some(unsafe {
        std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_void) -> u32>(pointer)
    })
}

fn publish(tables: &ApiTables) {
    generated::publish(tables);
    SCRATCH_TABLE.store(
        (&raw const tables.tools.functions[0]) as *mut c_void,
        Ordering::Release,
    );
}

type OnLoad = unsafe extern "C" fn(*mut RootTable, u64, u64, *const *const c_char) -> bool;
type OnUnload = unsafe extern "C" fn();
type RegisterImport = extern "C" fn() -> u32;
type Register = unsafe extern "C" fn(
    *const c_char,
    RegisterImport,
    u32,
    *mut *mut c_void,
    u64,
    *mut u64,
) -> u32;

const HSA_REGISTER_VERSION: u32 = 12_100;

#[unsafe(no_mangle)]
pub extern "C" fn rocprofiler_register_import_hsa() -> u32 {
    HSA_REGISTER_VERSION
}

fn secure_execution() -> bool {
    // SAFETY: These libc queries have no pointer arguments or retained state.
    unsafe {
        libc::geteuid() != libc::getuid()
            || libc::getegid() != libc::getgid()
            || libc::getauxval(libc::AT_SECURE) != 0
    }
}

fn parse_tool_names(names: &str) -> Vec<String> {
    let mut parsed = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut escaped = false;
    for ch in names.chars() {
        if escaped {
            current.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            quoted = !quoted;
        } else if ch.is_ascii_whitespace() && !quoted {
            if !current.is_empty() {
                parsed.push(std::mem::take(&mut current));
            }
        } else {
            current.push(ch);
        }
    }
    if escaped {
        current.push('\\');
    }
    if !current.is_empty() {
        parsed.push(current);
    }
    parsed
}

fn register_profiler(state: &mut ToolState) -> bool {
    let Some(library) = ToolLibrary::open(c"librocprofiler-register.so.0") else {
        return false;
    };
    let symbol = library.symbol(c"rocprofiler_register_library_api_table");
    if symbol.is_null() {
        return false;
    }
    // SAFETY: The symbol name and signature are the public
    // rocprofiler-register C ABI. The table Box stays pinned for its lifetime.
    let register: Register = unsafe { std::mem::transmute(symbol) };
    let saved = state.tables.functions();
    let saved_root = state.tables.root;
    let mut root = (&raw mut state.tables.root).cast::<c_void>();
    let mut identifier = 0u64;
    // SAFETY: The registration call borrows one writable root table pointer.
    let status = unsafe {
        register(
            c"hsa".as_ptr(),
            rocprofiler_register_import_hsa,
            HSA_REGISTER_VERSION,
            &raw mut root,
            1,
            &raw mut identifier,
        )
    };
    state.tables.root = saved_root;
    // NO_TOOLS (1) can still install the attachment queue shim. Retain its
    // table edits and load legacy tools afterwards, matching ROCr's protocol.
    if status > 1 || !generated::valid(&state.tables) {
        state.tables.restore(&saved);
        return false;
    }
    publish(&state.tables);
    state.registration = Some(library);
    status == 0
}

unsafe extern "C" fn collect_loaded_library(
    info: *mut libc::dl_phdr_info,
    _size: usize,
    data: *mut c_void,
) -> libc::c_int {
    if info.is_null() || data.is_null() {
        return 0;
    }
    // SAFETY: dl_iterate_phdr passes a live info record and our Vec pointer
    // synchronously. Copy each name before the callback returns.
    let name = unsafe { (*info).dlpi_name };
    if !name.is_null() {
        let name = unsafe { CStr::from_ptr(name) };
        if !name.is_empty() {
            let names = unsafe { &mut *data.cast::<Vec<CString>>() };
            names.push(name.to_owned());
        }
    }
    0
}

fn loaded_tool_candidates(env_count: usize) -> Vec<ToolCandidate> {
    let mut loaded = Vec::<CString>::new();
    // SAFETY: The callback borrows loaded only during this synchronous call.
    unsafe {
        libc::dl_iterate_phdr(Some(collect_loaded_library), (&raw mut loaded).cast());
    }
    let mut candidates = Vec::new();
    for name in loaded {
        let Some(library) = ToolLibrary::open_loaded(&name) else {
            continue;
        };
        let priority = library.symbol(c"HSA_AMD_TOOL_PRIORITY");
        if priority.is_null() {
            continue;
        }
        // SAFETY: This exported symbol is a uint32_t data object in the
        // loaded tool, and its module remains mapped through ToolLibrary.
        let priority = unsafe { priority.cast::<u32>().read_unaligned() };
        candidates.push(ToolCandidate {
            library,
            name,
            order: u64::from(priority) + env_count as u64,
        });
    }
    candidates
}

fn load_legacy_tools(state: &mut ToolState) {
    let names = if secure_execution() {
        Vec::new()
    } else {
        std::env::var_os("HSA_TOOLS_LIB")
            .map(|names| parse_tool_names(&names.to_string_lossy()))
            .unwrap_or_default()
    };
    let mut candidates = loaded_tool_candidates(names.len());
    let mut failed = Vec::<CString>::new();
    for (order, name) in names.into_iter().enumerate() {
        let Ok(name) = CString::new(name) else {
            continue;
        };
        let Some(library) = ToolLibrary::open(&name) else {
            failed.push(name);
            continue;
        };
        candidates.push(ToolCandidate {
            library,
            name,
            order: order as u64,
        });
    }
    candidates.sort_by_key(|candidate| candidate.order);
    let mut seen = HashSet::new();
    for candidate in candidates {
        let ToolCandidate { library, name, .. } = candidate;
        if !seen.insert(library.0) {
            continue;
        }
        let symbol = library.symbol(c"OnLoad");
        if symbol.is_null() || state.libraries.try_reserve(1).is_err() {
            failed.push(name);
            continue;
        }
        // SAFETY: OnLoad has the published HSA tool C ABI and runs while the
        // table owner is held exclusively by the process tool mutex.
        let on_load: OnLoad = unsafe { std::mem::transmute(symbol) };
        let saved = state.tables.functions();
        let saved_root = state.tables.root;
        let failed_names = failed.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();
        let failed_ptr = if failed_names.is_empty() {
            std::ptr::null()
        } else {
            failed_names.as_ptr()
        };
        let success = unsafe {
            on_load(
                &raw mut state.tables.root,
                u64::from(state.tables.root.version.major_id),
                failed_names.len() as u64,
                failed_ptr,
            )
        };
        // The root table points to the pinned tables owned by this runtime.
        // Tools may replace individual function pointers, not these owners.
        state.tables.root = saved_root;
        if success && generated::valid(&state.tables) {
            publish(&state.tables);
            state.libraries.push(library);
        } else {
            state.tables.restore(&saved);
            failed.push(name);
        }
    }
}

pub(crate) fn load() {
    let mut guard = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let state = guard.get_or_insert_with(|| ToolState {
        tables: ApiTables::new(),
        libraries: Vec::new(),
        registration: None,
        active: false,
    });
    if state.active {
        return;
    }
    let registered = register_profiler(state);
    let v1_requested = std::env::var("HSA_TOOLS_ROCPROFILER_V1_TOOLS")
        .is_ok_and(|value| !matches!(value.as_str(), "0" | "off" | "false" | "no"));
    if !registered || v1_requested {
        load_legacy_tools(state);
    }
    state.active = true;
}

pub(crate) fn unload() {
    let mut guard = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = guard.as_mut().filter(|state| state.active) else {
        return;
    };
    for library in state.libraries.iter().rev() {
        let symbol = library.symbol(c"OnUnload");
        if !symbol.is_null() {
            // SAFETY: OnUnload has the published HSA tool C ABI; the library
            // remains loaded through this callback.
            let on_unload: OnUnload = unsafe { std::mem::transmute(symbol) };
            unsafe { on_unload() };
        }
    }
    generated::reset();
    SCRATCH_TABLE.store(std::ptr::null_mut(), Ordering::Release);
    state.tables.restore(&generated::initial_functions());
    state.libraries.clear();
    state.registration = None;
    // rocprofiler-register can retain the root pointer for later attachment,
    // so its storage remains stable across HSA init/shutdown generations.
    state.active = false;
}

/// The v1 image extension has a different field order from the tool trace
/// table. Return public trampolines in the extension's own ABI order.
pub(crate) fn image_extension_table() -> [usize; 14] {
    [
        "hsa_ext_image_get_capability",
        "hsa_ext_image_data_get_info",
        "hsa_ext_image_create",
        "hsa_ext_image_destroy",
        "hsa_ext_image_copy",
        "hsa_ext_image_import",
        "hsa_ext_image_export",
        "hsa_ext_image_clear",
        "hsa_ext_sampler_create",
        "hsa_ext_sampler_destroy",
        "hsa_ext_image_get_capability_with_layout",
        "hsa_ext_image_data_get_info_with_layout",
        "hsa_ext_image_create_with_layout",
        "hsa_ext_sampler_create_v2",
    ]
    .map(|name| generated::public_address(name).unwrap_or(0))
}

pub(crate) fn loader_extension_table() -> [usize; 7] {
    [
        "hsa_ven_amd_loader_query_host_address",
        "hsa_ven_amd_loader_query_segment_descriptors",
        "hsa_ven_amd_loader_query_executable",
        "hsa_ven_amd_loader_executable_iterate_loaded_code_objects",
        "hsa_ven_amd_loader_loaded_code_object_get_info",
        "hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size",
        "hsa_ven_amd_loader_iterate_executables",
    ]
    .map(|name| generated::public_address(name).unwrap_or(0))
}

pub(crate) fn pc_sampling_extension_table() -> [usize; 7] {
    [
        "hsa_ven_amd_pcs_iterate_configuration",
        "hsa_ven_amd_pcs_create",
        "hsa_ven_amd_pcs_create_from_id",
        "hsa_ven_amd_pcs_destroy",
        "hsa_ven_amd_pcs_start",
        "hsa_ven_amd_pcs_stop",
        "hsa_ven_amd_pcs_flush",
    ]
    .map(|name| generated::public_address(name).unwrap_or(0))
}
