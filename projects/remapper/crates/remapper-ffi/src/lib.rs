//! `libremapper.so`: the C ABI in `include/remapper.h`.
#![allow(non_camel_case_types)]

use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::{Mutex, Once};

use anyhow::{bail, Context, Result};
use remapper_runtime::{
    device, Config, Dispatch, HsaDevice, KernelInfo, PassPlan, Remapper, Session,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct remapper_dim3_t {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl From<remapper_dim3_t> for [u32; 3] {
    fn from(d: remapper_dim3_t) -> Self {
        [d.x, d.y, d.z]
    }
}

impl From<[u32; 3]> for remapper_dim3_t {
    fn from(d: [u32; 3]) -> Self {
        Self {
            x: d[0],
            y: d[1],
            z: d[2],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct remapper_dispatch_t {
    pub kernel_object: u64,
    pub kernarg_address: u64,
    pub workgroup_size: remapper_dim3_t,
    pub grid_size: remapper_dim3_t,
    pub cluster_size: remapper_dim3_t,
    pub group_segment_size: u32,
    pub private_segment_size: u32,
}

impl From<remapper_dispatch_t> for Dispatch {
    fn from(d: remapper_dispatch_t) -> Self {
        Dispatch {
            kernel_object: d.kernel_object,
            kernarg_address: d.kernarg_address,
            workgroup_size: d.workgroup_size.into(),
            grid_size: d.grid_size.into(),
            cluster_size: d.cluster_size.into(),
            group_segment_size: d.group_segment_size,
            private_segment_size: d.private_segment_size,
        }
    }
}

impl From<Dispatch> for remapper_dispatch_t {
    fn from(d: Dispatch) -> Self {
        Self {
            kernel_object: d.kernel_object,
            kernarg_address: d.kernarg_address,
            workgroup_size: d.workgroup_size.into(),
            grid_size: d.grid_size.into(),
            cluster_size: d.cluster_size.into(),
            group_segment_size: d.group_segment_size,
            private_segment_size: d.private_segment_size,
        }
    }
}

#[repr(C)]
pub struct remapper_kernel_t {
    pub size: u64,
    pub dispatch_id: u64,
    pub kernel_name: *const c_char,
    pub code_object: *const c_void,
    pub code_object_size: u64,
    pub agent_node_id: u32,
    pub reserved: u32,
    pub dispatch: remapper_dispatch_t,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct remapper_pass_plan_t {
    pub size: u64,
    pub replace_dispatch: i32,
    pub trace: i32,
    pub dispatch: remapper_dispatch_t,
    pub target_cu: u32,
    pub simd: u32,
    pub shader_engine_mask: u64,
}

impl remapper_pass_plan_t {
    fn set(&mut self, plan: &PassPlan) {
        let size = self.size;
        *self = Self {
            size,
            ..Self::default()
        };
        if let Some(d) = plan.dispatch {
            self.replace_dispatch = 1;
            self.dispatch = d.into();
        }
        if let Some(t) = plan.trace {
            self.trace = 1;
            self.target_cu = t.target_cu;
            self.simd = t.simd;
            self.shader_engine_mask = t.shader_engine_mask;
        }
    }
}

pub struct remapper_t(Mutex<Remapper>);

pub struct remapper_session_t {
    session: Session<HsaDevice>,
    dir: CString,
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_error(message: String) {
    let message = CString::new(message.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = message);
}

/// Run `f`, turning an error or a panic into `fallback` and the thread's last error.
fn guard<T>(fallback: T, f: impl FnOnce() -> Result<T>) -> T {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_error(format!("{e:#}"));
            fallback
        }
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".into());
            set_error(format!("remapper panicked: {what}"));
            fallback
        }
    }
}

struct StderrLogger;

impl log::Log for StderrLogger {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::max_level()
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            eprintln!("[remapper] {}: {}", record.level(), record.args());
        }
    }
    fn flush(&self) {}
}

fn init_logging() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let level = match std::env::var("REMAPPER_LOG").as_deref() {
            Ok("debug") => log::LevelFilter::Debug,
            Ok("info") => log::LevelFilter::Info,
            Ok("error") => log::LevelFilter::Error,
            Ok("off") => log::LevelFilter::Off,
            _ => log::LevelFilter::Warn,
        };
        if log::set_logger(&StderrLogger).is_ok() {
            log::set_max_level(level);
        }
    });
}

/// # Safety
/// `p` is NULL or a NUL-terminated string.
unsafe fn opt_str<'a>(p: *const c_char) -> Result<Option<&'a str>> {
    if p.is_null() {
        return Ok(None);
    }
    // SAFETY: per the contract above.
    Ok(Some(
        unsafe { CStr::from_ptr(p) }
            .to_str()
            .context("string is not UTF-8")?,
    ))
}

/// # Safety
/// `output_dir` is a NUL-terminated path; `options` is NULL or a NUL-terminated string.
#[no_mangle]
pub unsafe extern "C" fn remapper_create(
    output_dir: *const c_char,
    options: *const c_char,
) -> *mut remapper_t {
    init_logging();
    guard(std::ptr::null_mut(), || {
        // SAFETY: per the function contract.
        let dir = unsafe { opt_str(output_dir) }?.context("output_dir is NULL")?;
        let mut config = Config {
            output_dir: PathBuf::from(dir),
            ..Config::default()
        };
        // SAFETY: per the function contract.
        if let Some(options) = unsafe { opt_str(options) }? {
            config.apply_options(options)?;
        }
        Ok(Box::into_raw(Box::new(remapper_t(Mutex::new(
            Remapper::new(config)?,
        )))))
    })
}

/// # Safety
/// `remapper` is NULL or came from `remapper_create` and is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn remapper_destroy(remapper: *mut remapper_t) {
    if !remapper.is_null() {
        // SAFETY: per the function contract.
        drop(unsafe { Box::from_raw(remapper) });
    }
}

/// # Safety
/// `remapper` came from `remapper_create`.
#[no_mangle]
pub unsafe extern "C" fn remapper_wants_dispatch(remapper: *mut remapper_t) -> c_int {
    guard(0, || {
        // SAFETY: per the function contract.
        let r = unsafe { remapper.as_ref() }.context("remapper is NULL")?;
        Ok(r.0
            .lock()
            .map_err(|_| anyhow::anyhow!("remapper lock poisoned"))?
            .wants_more() as c_int)
    })
}

/// # Safety
/// `remapper` came from `remapper_create`; `kernel` points to a filled `remapper_kernel_t` whose pointers stay valid
/// for the call.
#[no_mangle]
pub unsafe extern "C" fn remapper_begin_dispatch(
    remapper: *mut remapper_t,
    kernel: *const remapper_kernel_t,
) -> *mut remapper_session_t {
    guard(std::ptr::null_mut(), || {
        // SAFETY: per the function contract.
        let r = unsafe { remapper.as_ref() }.context("remapper is NULL")?;
        // SAFETY: per the function contract.
        let k = unsafe { kernel.as_ref() }.context("kernel is NULL")?;
        if (k.size as usize) < std::mem::size_of::<remapper_kernel_t>() {
            bail!(
                "remapper_kernel_t size {} is smaller than {}",
                k.size,
                std::mem::size_of::<remapper_kernel_t>()
            );
        }
        // SAFETY: per the function contract.
        let name = unsafe { opt_str(k.kernel_name) }?.context("kernel_name is NULL")?;
        if k.code_object.is_null() || k.code_object_size == 0 {
            bail!("no code object bytes for {name}");
        }
        // SAFETY: the caller guarantees `code_object_size` readable bytes.
        let code = unsafe {
            std::slice::from_raw_parts(
                k.code_object.cast::<u8>(),
                usize::try_from(k.code_object_size)?,
            )
        };
        let agent = device::gpu_agent_by_node(k.agent_node_id)?;
        let info = KernelInfo {
            dispatch_id: k.dispatch_id,
            symbol: name,
            code_object: code,
            app: k.dispatch.into(),
            shader_engines: device::shader_engines(agent)?,
        };
        let mut r =
            r.0.lock()
                .map_err(|_| anyhow::anyhow!("remapper lock poisoned"))?;
        let session = r.begin_dispatch(&info, HsaDevice::new(agent)?)?;
        let dir = CString::new(session.dir().to_string_lossy().into_owned())?;
        Ok(Box::into_raw(Box::new(remapper_session_t { session, dir })))
    })
}

/// # Safety
/// `session` came from `remapper_begin_dispatch`; `plan` points to a `remapper_pass_plan_t` with `size` set.
#[no_mangle]
pub unsafe extern "C" fn remapper_pass_enter(
    session: *mut remapper_session_t,
    pass: u64,
    plan: *mut remapper_pass_plan_t,
) -> c_int {
    guard(-1, || {
        // SAFETY: per the function contract.
        let plan = unsafe { plan.as_mut() }.context("plan is NULL")?;
        if (plan.size as usize) < std::mem::size_of::<remapper_pass_plan_t>() {
            bail!(
                "remapper_pass_plan_t size {} is smaller than {}",
                plan.size,
                std::mem::size_of::<remapper_pass_plan_t>()
            );
        }
        plan.set(&PassPlan {
            dispatch: None,
            trace: None,
        });
        // SAFETY: per the function contract.
        let s = unsafe { session.as_mut() }.context("session is NULL")?;
        plan.set(&s.session.pass_enter(pass)?);
        Ok(0)
    })
}

/// # Safety
/// `session` came from `remapper_begin_dispatch`.
#[no_mangle]
pub unsafe extern "C" fn remapper_pass_exit(session: *mut remapper_session_t, pass: u64) -> c_int {
    guard(-1, || {
        // SAFETY: per the function contract.
        let s = unsafe { session.as_mut() }.context("session is NULL")?;
        Ok(s.session.pass_exit(pass)? as c_int)
    })
}

/// # Safety
/// `session` came from `remapper_begin_dispatch`.
#[no_mangle]
pub unsafe extern "C" fn remapper_session_dir(session: *mut remapper_session_t) -> *const c_char {
    // SAFETY: per the function contract.
    match unsafe { session.as_ref() } {
        Some(s) => s.dir.as_ptr(),
        None => c"".as_ptr(),
    }
}

/// # Safety
/// `session` is NULL or came from `remapper_begin_dispatch` and is not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn remapper_end_dispatch(session: *mut remapper_session_t) -> c_int {
    if session.is_null() {
        return -1;
    }
    // SAFETY: per the function contract.
    let s = unsafe { Box::from_raw(session) };
    guard(-1, move || {
        Ok(match (s.session.done(), s.session.failed()) {
            (false, _) => -1,
            (true, true) => 1,
            (true, false) => 0,
        })
    })
}

/// `REMAPPER_ABI_VERSION` in `remapper.h`.
pub const ABI_VERSION: u32 = 1;

#[no_mangle]
pub extern "C" fn remapper_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub extern "C" fn remapper_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;

    #[test]
    fn abi_version_matches_remapper_h() {
        let header = include_str!("../../../include/remapper.h");
        assert!(header.contains(&format!("#define REMAPPER_ABI_VERSION {ABI_VERSION}\n")));
        assert_eq!(remapper_abi_version(), ABI_VERSION);
    }

    #[test]
    fn layouts_match_remapper_h() {
        assert_eq!(size_of::<remapper_dispatch_t>(), 60 + 4);
        assert_eq!(offset_of!(remapper_dispatch_t, workgroup_size), 16);
        assert_eq!(offset_of!(remapper_dispatch_t, cluster_size), 40);
        assert_eq!(offset_of!(remapper_dispatch_t, private_segment_size), 56);
        assert_eq!(offset_of!(remapper_kernel_t, agent_node_id), 40);
        assert_eq!(offset_of!(remapper_kernel_t, dispatch), 48);
        assert_eq!(size_of::<remapper_kernel_t>(), 112);
        assert_eq!(offset_of!(remapper_pass_plan_t, dispatch), 16);
        assert_eq!(offset_of!(remapper_pass_plan_t, target_cu), 80);
        assert_eq!(offset_of!(remapper_pass_plan_t, shader_engine_mask), 88);
        assert_eq!(size_of::<remapper_pass_plan_t>(), 96);
    }

    #[test]
    fn errors_reach_the_caller_and_null_is_rejected() {
        // SAFETY: NULL and literal strings only.
        unsafe {
            assert!(remapper_create(std::ptr::null(), std::ptr::null()).is_null());
            let msg = CStr::from_ptr(remapper_last_error())
                .to_string_lossy()
                .into_owned();
            assert!(msg.contains("output_dir is NULL"), "{msg}");
            let dir = tempfile_dir();
            let path = CString::new(dir.to_string_lossy().into_owned()).unwrap();
            assert!(remapper_create(path.as_ptr(), c"records=0".as_ptr()).is_null());
            assert!(remapper_create(path.as_ptr(), c"nonsense=1".as_ptr()).is_null());
            let r = remapper_create(path.as_ptr(), c"records=2,max-dispatches=1".as_ptr());
            assert!(!r.is_null());
            assert_eq!(remapper_wants_dispatch(r), 1);
            assert!(remapper_begin_dispatch(r, std::ptr::null()).is_null());
            let mut plan = remapper_pass_plan_t {
                size: 8,
                ..Default::default()
            };
            assert_eq!(remapper_pass_enter(std::ptr::null_mut(), 0, &mut plan), -1);
            plan.size = size_of::<remapper_pass_plan_t>() as u64;
            assert_eq!(remapper_pass_enter(std::ptr::null_mut(), 0, &mut plan), -1);
            assert_eq!((plan.replace_dispatch, plan.trace), (0, 0));
            assert_eq!(remapper_pass_exit(std::ptr::null_mut(), 0), -1);
            assert_eq!(remapper_end_dispatch(std::ptr::null_mut()), -1);
            remapper_destroy(r);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }

    fn tempfile_dir() -> PathBuf {
        let dir = std::env::temp_dir().join(format!("remapper-ffi-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }
}
