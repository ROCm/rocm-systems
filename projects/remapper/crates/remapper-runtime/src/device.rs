//! Device memory and code-object loading through the process's HSA runtime.
//!
//! Every transfer is a host access or an `hsa_memory_copy`: the remapper runs inside kernel replay callbacks and
//! never submits work to the application's queues.

use std::ffi::{c_void, CStr, CString};

use anyhow::{bail, Context, Result};
use hsa_sys::*;

/// The rewritten kernel as the HSA loader placed it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadedKernel {
    pub kernel_object: u64,
    pub kernarg_segment_size: u32,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
}

/// What the pass schedule needs from a GPU. Addresses are device virtual addresses.
pub trait Device {
    /// Coarse-grained device memory, zero-filled.
    fn alloc(&mut self, bytes: u64) -> Result<u64>;
    fn write(&mut self, va: u64, data: &[u8]) -> Result<()>;
    fn read(&mut self, va: u64, len: u64) -> Result<Vec<u8>>;
    /// Load a code object into a new executable on this agent and return `symbol`'s kernel.
    fn load_kernel(&mut self, code: &[u8], symbol: &str) -> Result<LoadedKernel>;
}

fn check(status: hsa_status_t, what: &str) -> Result<()> {
    if status == HSA_STATUS_SUCCESS {
        return Ok(());
    }
    let mut text: *const std::ffi::c_char = std::ptr::null();
    // SAFETY: hsa_status_string writes a pointer to a static string or leaves it null.
    let known =
        unsafe { hsa_status_string(status, &mut text) } == HSA_STATUS_SUCCESS && !text.is_null();
    let reason = if known {
        // SAFETY: non-null and NUL-terminated per the HSA API.
        unsafe { CStr::from_ptr(text) }
            .to_string_lossy()
            .into_owned()
    } else {
        String::new()
    };
    bail!("{what} failed: {status:#x} {reason}")
}

fn agent_info<T: Default>(agent: hsa_agent_t, attribute: hsa_agent_info_t) -> Result<T> {
    let mut v = T::default();
    // SAFETY: every attribute queried here is a plain integer of type T.
    check(
        unsafe { hsa_agent_get_info(agent, attribute, (&mut v as *mut T).cast()) },
        "hsa_agent_get_info",
    )?;
    Ok(v)
}

/// The GPU agent whose KFD node id is `node_id`.
pub fn gpu_agent_by_node(node_id: u32) -> Result<hsa_agent_t> {
    unsafe extern "C" fn each(agent: hsa_agent_t, data: *mut c_void) -> hsa_status_t {
        // SAFETY: `data` is the `Vec` passed below and outlives the iteration.
        let out = unsafe { &mut *data.cast::<Vec<hsa_agent_t>>() };
        out.push(agent);
        HSA_STATUS_SUCCESS
    }
    let mut agents: Vec<hsa_agent_t> = Vec::new();
    // SAFETY: the callback only appends to `agents`.
    check(
        unsafe { hsa_iterate_agents(each, (&mut agents as *mut Vec<hsa_agent_t>).cast()) },
        "hsa_iterate_agents",
    )?;
    for agent in agents {
        let device: hsa_device_type_t = agent_info(agent, HSA_AGENT_INFO_DEVICE)?;
        if device != HSA_DEVICE_TYPE_GPU {
            continue;
        }
        let node: u32 = agent_info(agent, HSA_AMD_AGENT_INFO_DRIVER_NODE_ID)?;
        if node == node_id {
            return Ok(agent);
        }
    }
    bail!("no GPU agent with KFD node id {node_id}")
}

pub fn shader_engines(agent: hsa_agent_t) -> Result<u32> {
    agent_info(agent, HSA_AMD_AGENT_INFO_NUM_SHADER_ENGINES)
}

fn coarse_pool(agent: hsa_agent_t) -> Result<hsa_amd_memory_pool_t> {
    unsafe extern "C" fn each(pool: hsa_amd_memory_pool_t, data: *mut c_void) -> hsa_status_t {
        let mut segment: hsa_amd_segment_t = u32::MAX;
        let mut flags = 0u32;
        let mut alloc = false;
        // SAFETY: the queried attributes are a u32 enum, a u32 bit field and a bool.
        unsafe {
            if hsa_amd_memory_pool_get_info(
                pool,
                HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                (&mut segment as *mut u32).cast(),
            ) != HSA_STATUS_SUCCESS
                || hsa_amd_memory_pool_get_info(
                    pool,
                    HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                    (&mut flags as *mut u32).cast(),
                ) != HSA_STATUS_SUCCESS
                || hsa_amd_memory_pool_get_info(
                    pool,
                    HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                    (&mut alloc as *mut bool).cast(),
                ) != HSA_STATUS_SUCCESS
            {
                return HSA_STATUS_SUCCESS;
            }
        }
        if segment == HSA_AMD_SEGMENT_GLOBAL
            && flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED != 0
            && alloc
        {
            // SAFETY: `data` is the `Option` passed below.
            unsafe { *data.cast::<Option<hsa_amd_memory_pool_t>>() = Some(pool) };
            return HSA_STATUS_INFO_BREAK;
        }
        HSA_STATUS_SUCCESS
    }
    let mut found: Option<hsa_amd_memory_pool_t> = None;
    // SAFETY: the callback only writes `found`.
    let status = unsafe {
        hsa_amd_agent_iterate_memory_pools(
            agent,
            each,
            (&mut found as *mut Option<hsa_amd_memory_pool_t>).cast(),
        )
    };
    if status != HSA_STATUS_INFO_BREAK {
        check(status, "hsa_amd_agent_iterate_memory_pools")?;
    }
    found.context("GPU agent has no coarse-grained global memory pool")
}

/// One GPU agent's coarse-grained pool, and the executables and buffers the remapper created on it. Dropping it
/// frees them.
pub struct HsaDevice {
    agent: hsa_agent_t,
    pool: hsa_amd_memory_pool_t,
    buffers: Vec<*mut c_void>,
    executables: Vec<(hsa_executable_t, hsa_code_object_reader_t)>,
    /// A code object reader reads from its bytes until it is destroyed.
    code_objects: Vec<Box<[u8]>>,
}

// SAFETY: the raw pointers are device allocations owned by this struct; HSA calls on them are thread-safe.
unsafe impl Send for HsaDevice {}

impl HsaDevice {
    pub fn new(agent: hsa_agent_t) -> Result<Self> {
        Ok(Self {
            agent,
            pool: coarse_pool(agent)?,
            buffers: Vec::new(),
            executables: Vec::new(),
            code_objects: Vec::new(),
        })
    }

    pub fn agent(&self) -> hsa_agent_t {
        self.agent
    }

    fn symbol_info<T: Default>(
        symbol: hsa_executable_symbol_t,
        attribute: hsa_executable_symbol_info_t,
    ) -> Result<T> {
        let mut v = T::default();
        // SAFETY: every attribute queried here is a plain integer of type T.
        check(
            unsafe { hsa_executable_symbol_get_info(symbol, attribute, (&mut v as *mut T).cast()) },
            "hsa_executable_symbol_get_info",
        )?;
        Ok(v)
    }
}

impl Device for HsaDevice {
    fn alloc(&mut self, bytes: u64) -> Result<u64> {
        let bytes = usize::try_from(bytes.max(8))?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `pool` was discovered on `agent` and allows runtime allocation.
        check(
            unsafe { hsa_amd_memory_pool_allocate(self.pool, bytes, 0, &mut ptr) },
            "hsa_amd_memory_pool_allocate",
        )?;
        self.buffers.push(ptr);
        // SAFETY: `ptr` is a fresh allocation of `bytes` bytes.
        check(
            unsafe { hsa_amd_agents_allow_access(1, &self.agent, std::ptr::null(), ptr) },
            "hsa_amd_agents_allow_access",
        )?;
        let va = ptr as u64;
        self.write(va, &vec![0u8; bytes])?;
        Ok(va)
    }

    fn write(&mut self, va: u64, data: &[u8]) -> Result<()> {
        if data.is_empty() {
            return Ok(());
        }
        // SAFETY: `va` addresses a buffer from `alloc` (or another HSA allocation) of at least `data.len()` bytes.
        check(
            unsafe { hsa_memory_copy(va as *mut c_void, data.as_ptr().cast(), data.len()) },
            "hsa_memory_copy (to device)",
        )
    }

    fn read(&mut self, va: u64, len: u64) -> Result<Vec<u8>> {
        let mut out = vec![0u8; usize::try_from(len)?];
        if !out.is_empty() {
            // SAFETY: `va` addresses at least `len` readable bytes of HSA-visible memory.
            check(
                unsafe { hsa_memory_copy(out.as_mut_ptr().cast(), va as *const c_void, out.len()) },
                "hsa_memory_copy (from device)",
            )?;
        }
        Ok(out)
    }

    fn load_kernel(&mut self, code: &[u8], symbol: &str) -> Result<LoadedKernel> {
        let profile: hsa_profile_t = agent_info(self.agent, HSA_AGENT_INFO_PROFILE)?;
        let rounding: hsa_default_float_rounding_mode_t =
            agent_info(self.agent, HSA_AGENT_INFO_DEFAULT_FLOAT_ROUNDING_MODE)?;
        self.code_objects.push(code.into());
        let code = self.code_objects.last().context("code object")?;
        let mut reader = hsa_code_object_reader_t::default();
        // SAFETY: `code` is kept in `code_objects` until the reader is destroyed in `drop`.
        check(
            unsafe {
                hsa_code_object_reader_create_from_memory(
                    code.as_ptr().cast(),
                    code.len(),
                    &mut reader,
                )
            },
            "hsa_code_object_reader_create_from_memory",
        )?;
        let mut exe = hsa_executable_t::default();
        // SAFETY: plain out-parameter.
        if let Err(e) = check(
            unsafe { hsa_executable_create_alt(profile, rounding, std::ptr::null(), &mut exe) },
            "hsa_executable_create_alt",
        ) {
            // SAFETY: `reader` was created above.
            unsafe { hsa_code_object_reader_destroy(reader) };
            return Err(e);
        }
        self.executables.push((exe, reader));
        // SAFETY: `exe` is unfrozen and `reader` holds a code object for this agent's ISA.
        check(
            unsafe {
                hsa_executable_load_agent_code_object(
                    exe,
                    self.agent,
                    reader,
                    std::ptr::null(),
                    std::ptr::null_mut(),
                )
            },
            "hsa_executable_load_agent_code_object",
        )?;
        // SAFETY: `exe` was created above.
        check(
            unsafe { hsa_executable_freeze(exe, std::ptr::null()) },
            "hsa_executable_freeze",
        )?;
        let name = CString::new(format!(
            "{}.kd",
            symbol.strip_suffix(".kd").unwrap_or(symbol)
        ))?;
        let mut sym = hsa_executable_symbol_t::default();
        // SAFETY: `name` is NUL-terminated and `exe` is frozen.
        check(
            unsafe { hsa_executable_get_symbol_by_name(exe, name.as_ptr(), &self.agent, &mut sym) },
            "hsa_executable_get_symbol_by_name",
        )
        .with_context(|| {
            format!(
                "looking up {} in the rewritten code object",
                name.to_string_lossy()
            )
        })?;
        Ok(LoadedKernel {
            kernel_object: Self::symbol_info(sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT)?,
            kernarg_segment_size: Self::symbol_info(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
            )?,
            group_segment_size: Self::symbol_info(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
            )?,
            private_segment_size: Self::symbol_info(
                sym,
                HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
            )?,
        })
    }
}

impl Drop for HsaDevice {
    fn drop(&mut self) {
        // SAFETY: each handle and pointer was created by this device and is released once.
        unsafe {
            for (exe, reader) in self.executables.drain(..) {
                hsa_executable_destroy(exe);
                hsa_code_object_reader_destroy(reader);
            }
            for ptr in self.buffers.drain(..) {
                hsa_amd_memory_pool_free(ptr);
            }
        }
    }
}
