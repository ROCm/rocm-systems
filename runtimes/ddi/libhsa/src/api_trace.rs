// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! `ROCr` tool ABI tables and dispatch ownership.
//!
//! Each public HSA symbol jumps through an atomic function slot. The
//! registrar uses generation-leased update transactions and immutable snapshots.
//! Legacy `OnLoad` edits are snapshotted after its callback returns; staging
//! storage and published snapshots remain pinned for process lifetime.

use crate::platform::tooling;
use rocddi::host::module::ResidentModule;
use std::collections::HashSet;
use std::ffi::{CStr, CString, c_char, c_void};
use std::mem::size_of;
use std::ptr::NonNull;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

/// Generated support for this frontend.
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
    /// Builds a table version with its exact ABI byte size.
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
    /// Builds a stable function table with version metadata.
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
    /// Pinned table of core HSA function pointers.
    core: ApiTable<{ generated::CORE_COUNT }>,
    /// Pinned table of AMD extension function pointers.
    amd_ext: ApiTable<{ generated::AMD_EXT_COUNT }>,
    /// Pinned table of finalizer extension pointers.
    finalizer_ext: ApiTable<{ generated::FINALIZER_EXT_COUNT }>,
    /// Pinned table of image extension pointers.
    image_ext: ApiTable<{ generated::IMAGE_EXT_COUNT }>,
    /// Pinned table of tool support function pointers.
    tools: ApiTable<{ generated::TOOLS_COUNT }>,
    /// Pinned table of PC sampling extension pointers.
    pc_sampling_ext: ApiTable<{ generated::PC_SAMPLING_EXT_COUNT }>,
}

// SAFETY: Root pointers refer only to fields within the pinned Box<ApiTables>.
// The process tool mutex owns Rust-side mutation of private and legacy
// tables. Legacy tools must finish their edits before OnLoad returns;
// afterward the staging table is retained for tool-owned pointers.
unsafe impl Send for ApiTables {}

impl ApiTables {
    /// Allocates the root table and all child tables at stable addresses.
    fn new() -> Box<Self> {
        Self::from_functions(&generated::initial_functions())
    }

    /// Creates a private staging table from the last published functions.
    fn from_functions(functions: &generated::InitialFunctions) -> Box<Self> {
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
            core: ApiTable::new(2, 0, functions.core),
            amd_ext: ApiTable::new(2, generated::AMD_EXT_STEP, functions.amd_ext),
            finalizer_ext: ApiTable::new(2, 0, functions.finalizer_ext),
            image_ext: ApiTable::new(2, generated::IMAGE_EXT_STEP, functions.image_ext),
            tools: ApiTable::new(1, 0, functions.tools),
            pc_sampling_ext: ApiTable::new(1, 0, functions.pc_sampling_ext),
        });
        tables.root.core = &raw mut tables.core;
        tables.root.amd_ext = &raw mut tables.amd_ext;
        tables.root.finalizer_ext = &raw mut tables.finalizer_ext;
        tables.root.image_ext = &raw mut tables.image_ext;
        tables.root.tools = &raw mut tables.tools;
        tables.root.pc_sampling_ext = &raw mut tables.pc_sampling_ext;
        tables
    }

    /// Snapshots the functions in every API table.
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

    /// Replaces the private function slots from a validated snapshot.
    fn restore(&mut self, saved: &generated::InitialFunctions) {
        self.core.functions = saved.core;
        self.amd_ext.functions = saved.amd_ext;
        self.finalizer_ext.functions = saved.finalizer_ext;
        self.image_ext.functions = saved.image_ext;
        self.tools.functions = saved.tools;
        self.pc_sampling_ext.functions = saved.pc_sampling_ext;
    }
}

/// Tool module whose exported code remains resident through process exit.
struct ToolLibrary(ResidentModule);

impl ToolLibrary {
    /// Acquires a named tool using the qualified frontend naming convention.
    fn open(name: &CStr) -> Option<Self> {
        // SAFETY: HSA tool configuration opts into the named foreign tool ABI.
        unsafe { tooling::acquire(name, false) }.map(Self)
    }

    /// Retains a candidate from an already-loaded image snapshot.
    fn open_loaded(name: &CStr) -> Option<Self> {
        // SAFETY: The existing HSA tool ABI permits already-loaded candidates.
        unsafe { tooling::acquire(name, true) }.map(Self)
    }

    /// Looks up an untyped tool symbol; each caller checks its exact C ABI.
    fn symbol(&self, name: &CStr) -> *mut c_void {
        // SAFETY: Tool acquisition accepts foreign resolver execution. The
        // native owner pins code before any table can retain the address.
        unsafe { self.0.symbol(name) }
            .ok()
            .flatten()
            .map_or(std::ptr::null_mut(), NonNull::as_ptr)
    }
}

/// Process-owned API tables and attached tool libraries.
struct ToolState {
    /// Private source for the last published function snapshot.
    tables: Box<ApiTables>,
    /// One process registration reused across all HSA generations.
    profiler_id: Option<u64>,
    /// A tool can retain a table pointer after `OnUnload`; keep every staging
    /// table alive until process exit, including failed attachment attempts.
    #[allow(clippy::vec_box, reason = "each C table needs a stable address")]
    legacy_tables: Vec<Box<ApiTables>>,
    /// Libraries whose callbacks may still be in flight.
    libraries: Vec<ToolLibrary>,
    /// Keeps the registrar mapped while it retains any HSA table pointer.
    registration: Option<ToolLibrary>,
    /// Attachment state of the current HSA generation.
    phase: ToolPhase,
}

/// Serialized tool attachment phase for one HSA generation.
#[derive(Clone, Copy, Eq, PartialEq)]
enum ToolPhase {
    /// No tool callbacks are active for an HSA generation.
    Inactive,
    /// Legacy tool callbacks are running.
    Loading,
    /// Tool callbacks have completed and wrappers are published.
    Active,
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

/// Loaded tools and stable API tables for the current process.
static STATE: Mutex<Option<ToolState>> = Mutex::new(None);
/// Runtime-owned callback slots read by queue operations.
static SCRATCH_SLOTS: [AtomicPtr<c_void>; generated::TOOLS_COUNT] =
    [const { AtomicPtr::new(std::ptr::null_mut()) }; generated::TOOLS_COUNT];

/// Native no-op target for a scratch bridge without a registered callback.
extern "C" fn no_scratch_callback(_: *const c_void) -> u32 {
    0
}

/// Returns a registered scratch callback at the given table index.
pub(crate) fn scratch_callback(index: usize) -> Option<unsafe extern "C" fn(*const c_void) -> u32> {
    let pointer = SCRATCH_SLOTS.get(index)?.load(Ordering::Acquire);
    if pointer.is_null() {
        return None;
    }
    // SAFETY: The tool table field has the hsa_amd_tool_event signature.
    // RTLD_NODELETE retains registered tool code through callback delivery.
    Some(unsafe {
        std::mem::transmute::<*mut c_void, unsafe extern "C" fn(*const c_void) -> u32>(pointer)
    })
}

/// Atomically publishes the current API tables to export trampolines.
fn publish(tables: &ApiTables) {
    generated::publish(tables);
    for (slot, callback) in SCRATCH_SLOTS.iter().zip(tables.tools.functions) {
        slot.store(callback, Ordering::Release);
    }
}

/// Tool entry point called with mutable HSA API tables on attach.
type OnLoad = unsafe extern "C" fn(*mut RootTable, u64, u64, *const *const c_char) -> bool;
/// Tool entry point called when HSA unloads the tool.
type OnUnload = unsafe extern "C" fn();
/// rocprofiler import-registration entry point.
type RegisterImport = extern "C" fn() -> u32;
/// Synchronous mutator supplied by the versioned registrar. It may retain
/// staging addresses for reads, but all writes finish before it returns.
type ProfilerUpdater = unsafe extern "C" fn(*mut c_void, *mut *mut c_void, u64, u64) -> i32;
/// Runtime-owned transaction entry point used by registrar ABI version 2.
type ProfilerTransaction = unsafe extern "C" fn(*mut c_void, ProfilerUpdater, *mut c_void) -> u32;
/// Executes a worker's bounded synchronous native work under its generation.
type ProfilerCleanup = unsafe extern "C" fn(
    *mut c_void,
    u64,
    unsafe extern "C" fn(*mut c_void) -> i32,
    *mut c_void,
) -> u32;

/// ABI shared with rocprofiler-register's v2 provider header.
#[repr(C)]
struct ProfilerProvider {
    /// Extensible structure byte size.
    size: usize,
    /// Publication protocol version, independently of the HSA table version.
    version: u64,
    /// Process-lived provider context; this runtime uses static state.
    data: *mut c_void,
    /// Runs an updater inside admission and publishes its complete result.
    update: ProfilerTransaction,
    /// Admits bounded worker operations while active or preparing retirement.
    run_cleanup: ProfilerCleanup,
}

/// Registers the process-owned version 2 provider.
type Register = unsafe extern "C" fn(
    *const c_char,
    RegisterImport,
    u32,
    *const ProfilerProvider,
    *mut u64,
) -> u32;
/// Refreshes or quiesces one process registration identifier.
type Refresh = unsafe extern "C" fn(u64) -> u32;
/// Prepares SDK worker retirement while native callback services remain usable.
type Prepare = unsafe extern "C" fn(u64, u64) -> u32;

/// Last profiler layer, protected independently of legacy attachment. This
/// mutex is never acquired while the HSA lifecycle mutex is held.
static PROFILER: Mutex<Option<Box<ApiTables>>> = Mutex::new(None);
/// Tool cleanup may use a private native-call lease after public admission
/// closes. That lease must never authorize another table update.
static PROFILER_QUIESCING: AtomicBool = AtomicBool::new(true);

/// Runs bounded native work without permitting a worker to carry its lease
/// into another callback or runtime generation.
unsafe extern "C" fn profiler_cleanup(
    _: *mut c_void,
    generation: u64,
    work: unsafe extern "C" fn(*mut c_void) -> i32,
    data: *mut c_void,
) -> u32 {
    let Ok(token) = crate::lifecycle::enter_delegated_tool_cleanup(generation) else {
        return 9; // ROCP_REG_ATTACHMENT_NOT_AVAILABLE
    };
    // SAFETY: The v2 callback is synchronous and must not unwind or retain the
    // supplied native-call lease beyond this call.
    crate::admission::ApiLeaseScope::run(token, || if unsafe { work(data) } == 0 { 0 } else { 6 })
}

/// Executes a versioned foreign update without exposing its mutable table to
/// dispatch readers. Closing process admission fences this transaction against
/// final shutdown, including native calls made synchronously by the updater.
unsafe extern "C" fn profiler_update(
    _: *mut c_void,
    updater: ProfilerUpdater,
    data: *mut c_void,
) -> u32 {
    const INVALID_ARGUMENT: u32 = 8;
    const INACTIVE: u32 = 9;
    if PROFILER_QUIESCING.load(Ordering::Acquire) {
        return INACTIVE;
    }
    let Some(token) = crate::admission::process_admission().enter() else {
        return INACTIVE;
    };
    let generation = token.generation();
    crate::admission::ApiLeaseScope::run(token, || {
        // Registrar serializes its calls and rejects recursive updates. The
        // independent mutex also protects the retained seed across generations.
        let Ok(mut profiler) = PROFILER.lock() else {
            return INVALID_ARGUMENT;
        };
        let staging = profiler.as_ref().map_or_else(ApiTables::new, |tables| {
            ApiTables::from_functions(&tables.functions())
        });
        // Pin before invoking foreign code: even a failed updater may retain the
        // root or children. No subsequent runtime mutation reaches this table.
        let staging = Box::leak(staging);
        let mut root = (&raw mut staging.root).cast::<c_void>();
        // SAFETY: The registrar's versioned contract permits synchronous writes
        // to this private table and requires all such writes to finish on return.
        if unsafe { updater(data, &raw mut root, 1, generation) } != 0 || !generated::valid(staging)
        {
            return 6; // ROCP_REG_ROCPROFILER_ERROR
        }
        let snapshot = ApiTables::from_functions(&staging.functions());
        generated::publish_profiler(&snapshot);
        *profiler = Some(snapshot);
        0
    })
}

/// ABI version value for HSA register.
const HSA_REGISTER_VERSION: u32 = 12_100;

#[unsafe(no_mangle)]
/// Reports the HSA import registration version to rocprofiler.
pub extern "C" fn rocprofiler_register_import_hsa() -> u32 {
    HSA_REGISTER_VERSION
}

/// Splits and validates the configured list of tool library names.
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

/// Registers one process provider, or refreshes its existing generation.
/// The old retained-mutable-table ABI is deliberately not called: it has no
/// writer fence. Synchronous legacy `OnLoad` tools remain available below.
fn register_profiler(state: &mut ToolState) -> bool {
    if state.registration.is_none() {
        state.registration = ToolLibrary::open(tooling::REGISTRAR);
    }
    let Some(library) = state.registration.as_ref() else {
        return false;
    };
    let register = library.symbol(c"rocprofiler_register_library_api_table_v2");
    let refresh = library.symbol(c"rocprofiler_register_refresh_library_api_table_v2");
    let quiesce = library.symbol(c"rocprofiler_register_quiesce_library_api_table_v2");
    let prepare = library.symbol(c"rocprofiler_register_prepare_library_api_table_v2");
    if register.is_null() || refresh.is_null() || quiesce.is_null() || prepare.is_null() {
        return false;
    }
    // SAFETY: Symbol names select the exact v2 C signatures defined above.
    let register: Register = unsafe { std::mem::transmute(register) };
    let refresh: Refresh = unsafe { std::mem::transmute(refresh) };
    let status = if let Some(identifier) = state.profiler_id {
        unsafe { refresh(identifier) }
    } else {
        let provider = ProfilerProvider {
            size: size_of::<ProfilerProvider>(),
            version: 2,
            data: std::ptr::null_mut(),
            update: profiler_update,
            run_cleanup: profiler_cleanup,
        };
        let mut identifier = u64::MAX;
        // SAFETY: Registrar copies the descriptor synchronously. Its callback
        // and context remain valid through process exit, including failure.
        let status = unsafe {
            register(
                c"hsa".as_ptr(),
                rocprofiler_register_import_hsa,
                HSA_REGISTER_VERSION,
                &raw const provider,
                &raw mut identifier,
            )
        };
        if identifier != u64::MAX {
            state.profiler_id = Some(identifier);
        }
        status
    };
    if status > 1 {
        return false;
    }
    state.tables.restore(&generated::bridge_functions());
    publish(&state.tables);
    status == 0
}

/// Combines environment and already-loaded libraries into tool candidates.
fn loaded_tool_candidates(env_count: usize) -> Vec<ToolCandidate> {
    let loaded = tooling::loaded_names();
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

/// Attaches configured legacy tools and lets each modify the API tables.
fn load_legacy_tools(state: &mut ToolState) {
    let names = if tooling::secure_execution() {
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
        if !seen.insert(library.0.id()) {
            continue;
        }
        let symbol = library.symbol(c"OnLoad");
        if symbol.is_null() {
            failed.push(name);
            continue;
        }
        if state.libraries.try_reserve(1).is_err() || state.legacy_tables.try_reserve(1).is_err() {
            failed.push(name);
            continue;
        }
        // SAFETY: OnLoad has the published HSA tool C ABI. The table is pinned
        // and its edits must finish before OnLoad returns; the tool-loader
        // mutex keeps runtime-side table publication serialized.
        let on_load: OnLoad = unsafe { std::mem::transmute(symbol) };
        let mut staging = ApiTables::from_functions(&state.tables.functions());
        let failed_names = failed.iter().map(|name| name.as_ptr()).collect::<Vec<_>>();
        let failed_ptr = if failed_names.is_empty() {
            std::ptr::null()
        } else {
            failed_names.as_ptr()
        };
        let success = unsafe {
            on_load(
                &raw mut staging.root,
                u64::from(staging.root.version.major_id),
                failed_names.len() as u64,
                failed_ptr,
            )
        };
        // A tool can retain the root and child tables after OnUnload. Keep the
        // storage pinned even when OnLoad fails or supplies an invalid table.
        if success {
            if generated::valid(&staging) {
                state.tables.restore(&staging.functions());
                publish(&state.tables);
            } else {
                // OnLoad succeeded, so the tool may still retain its table or
                // need OnUnload. Keep both alive even though its edits cannot
                // be published to HSA callers.
                failed.push(name);
            }
            state.libraries.push(library);
        } else {
            failed.push(name);
        }
        state.legacy_tables.push(staging);
    }
}

/// Loads and attaches legacy HSA tools in configured order.
pub(crate) fn load() {
    {
        let mut guard = STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let state = guard.get_or_insert_with(|| ToolState {
            tables: ApiTables::new(),
            profiler_id: None,
            legacy_tables: Vec::new(),
            libraries: Vec::new(),
            registration: None,
            phase: ToolPhase::Inactive,
        });
        if state.phase != ToolPhase::Inactive {
            return;
        }
        state.phase = ToolPhase::Loading;
        PROFILER_QUIESCING.store(false, Ordering::Release);
    }
    let mut guard = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = guard.as_mut() else {
        return;
    };
    // A generation is built from native functions. The registrar reinstalls
    // the attachment layer before the SDK; feeding the previous SDK wrappers
    // back into its initializer could make their downstream calls recursive.
    *PROFILER
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner) = None;
    let registered = register_profiler(state);
    let v1_requested = std::env::var("HSA_TOOLS_ROCPROFILER_V1_TOOLS")
        .is_ok_and(|value| !matches!(value.as_str(), "0" | "off" | "false" | "no"));
    if !registered || v1_requested {
        load_legacy_tools(state);
    }
    state.phase = ToolPhase::Active;
}

/// Stops existing SDK work while the runtime's explicit cleanup window permits
/// its workers to finish native operations. Public table updates stay closed.
pub(crate) fn prepare_shutdown(generation: u64) -> crate::ffi::Status {
    PROFILER_QUIESCING.store(true, Ordering::Release);
    let guard = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = guard
        .as_ref()
        .filter(|state| state.phase == ToolPhase::Active)
    else {
        return crate::ffi::SUCCESS;
    };
    if let (Some(library), Some(identifier)) = (&state.registration, state.profiler_id) {
        let symbol = library.symbol(c"rocprofiler_register_prepare_library_api_table_v2");
        if symbol.is_null() {
            return crate::ffi::ERROR;
        }
        // SAFETY: The named versioned symbol has the two-identifier C signature.
        let prepare: Prepare = unsafe { std::mem::transmute(symbol) };
        if unsafe { prepare(identifier, generation) } > 1 {
            return crate::ffi::ERROR;
        }
    }
    crate::ffi::SUCCESS
}

/// Retires the SDK after native queues and callback producers have stopped.
pub(crate) fn retire_profiler() -> crate::ffi::Status {
    PROFILER_QUIESCING.store(true, Ordering::Release);
    let mut guard = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = guard
        .as_mut()
        .filter(|state| state.phase == ToolPhase::Active)
    else {
        return crate::ffi::SUCCESS;
    };
    // The lifecycle owner supplies a synchronous native cleanup lease through
    // retirement and legacy unload, after prepare and native stop succeed.
    if let (Some(library), Some(identifier)) = (&state.registration, state.profiler_id) {
        let symbol = library.symbol(c"rocprofiler_register_quiesce_library_api_table_v2");
        if symbol.is_null() {
            return crate::ffi::ERROR;
        }
        // SAFETY: The v2 symbol uses the same one-identifier C signature.
        let quiesce: Refresh = unsafe { std::mem::transmute(symbol) };
        if unsafe { quiesce(identifier) } > 1 {
            return crate::ffi::ERROR;
        }
    }
    crate::ffi::SUCCESS
}

/// Detaches loaded tools and restores original API function pointers.
pub(crate) fn unload() -> crate::ffi::Status {
    PROFILER_QUIESCING.store(true, Ordering::Release);
    let mut guard = STATE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let Some(state) = guard
        .as_mut()
        .filter(|state| state.phase == ToolPhase::Active)
    else {
        return crate::ffi::SUCCESS;
    };
    for library in state.libraries.iter().rev() {
        let symbol = library.symbol(c"OnUnload");
        if !symbol.is_null() {
            // SAFETY: OnUnload has the published HSA tool C ABI; the
            // library remains loaded through this callback.
            let on_unload: OnUnload = unsafe { std::mem::transmute(symbol) };
            unsafe { on_unload() };
        }
    }
    generated::reset();
    generated::reset_profiler();
    for slot in &SCRATCH_SLOTS {
        slot.store(std::ptr::null_mut(), Ordering::Release);
    }
    state.tables.restore(&generated::initial_functions());
    state.libraries.clear();
    state.phase = ToolPhase::Inactive;
    crate::ffi::SUCCESS
}

/// The v1 image extension has a different field order from the tool trace
/// table. Return public trampolines in the extension's own ABI order.
pub(crate) fn image_extension_table() -> [usize; 15] {
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
        "hsa_ext_sampler_create_v3",
    ]
    .map(|name| generated::public_address(name).unwrap_or(0))
}

/// Returns the AMD loader extension function table.
pub(crate) fn loader_extension_table() -> [usize; 9] {
    [
        "hsa_ven_amd_loader_query_host_address",
        "hsa_ven_amd_loader_query_segment_descriptors",
        "hsa_ven_amd_loader_query_executable",
        "hsa_ven_amd_loader_executable_iterate_loaded_code_objects",
        "hsa_ven_amd_loader_loaded_code_object_get_info",
        "hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size",
        "hsa_ven_amd_loader_iterate_executables",
        "hsa_ven_amd_loader_code_object_reader_get_load_size",
        "hsa_ven_amd_loader_executable_load_agent_code_object_at_address",
    ]
    .map(|name| generated::public_address(name).unwrap_or(0))
}

/// Returns the PC sampling extension function table.
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
