#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::ffi::{CStr, CString, NulError};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

pub mod ffi {
    #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
    #![allow(clippy::all)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub mod fbs_generated {
    #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
    #![allow(dead_code, unsafe_op_in_unsafe_fn, unused_imports)]
    #![allow(mismatched_lifetime_syntaxes)]
    #![allow(clippy::all)]

    include!(concat!(env!("OUT_DIR"), "/fbs/mod.rs"));
}

pub mod fb {
    pub use crate::fbs_generated::rocjitsu::fb::*;
}

pub const SCHEMA_DIR: &str = env!("ROCJITSU_SYS_SCHEMA_DIR");

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    Status(ffi::rj_status_t),
    NullHandle(&'static str),
    Nul(NulError),
}

impl Error {
    pub fn status(&self) -> Option<ffi::rj_status_t> {
        match self {
            Self::Status(status) => Some(*status),
            Self::NullHandle(_) | Self::Nul(_) => None,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Status(status) => write!(formatter, "rocjitsu returned {}", status_name(*status)),
            Self::NullHandle(name) => write!(formatter, "rocjitsu returned a null {name} handle"),
            Self::Nul(err) => err.fmt(formatter),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Nul(err) => Some(err),
            Self::Status(_) | Self::NullHandle(_) => None,
        }
    }
}

impl From<NulError> for Error {
    fn from(err: NulError) -> Self {
        Self::Nul(err)
    }
}

pub struct Vm {
    ptr: NonNull<ffi::rj_vm_t>,
}

impl Vm {
    pub fn create(json_path: impl AsRef<Path>, schema_path: impl AsRef<Path>) -> Result<Self> {
        let json_path = path_to_cstring(json_path.as_ref())?;
        let schema_path = path_to_cstring(schema_path.as_ref())?;
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { ffi::rj_vm_create(json_path.as_ptr(), schema_path.as_ptr(), &mut handle) };
        status_to_result(status)?;
        Self::from_raw(handle, "VM")
    }

    pub fn create_with_default_schema(json_path: impl AsRef<Path>) -> Result<Self> {
        Self::create(json_path, default_simulation_schema())
    }

    pub fn create_from_string(json: &str, schema_path: impl AsRef<Path>) -> Result<Self> {
        let json = CString::new(json)?;
        let schema_path = path_to_cstring(schema_path.as_ref())?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            ffi::rj_vm_create_from_string(json.as_ptr(), schema_path.as_ptr(), &mut handle)
        };
        status_to_result(status)?;
        Self::from_raw(handle, "VM")
    }

    pub fn create_from_string_with_default_schema(json: &str) -> Result<Self> {
        Self::create_from_string(json, default_simulation_schema())
    }

    pub fn restore_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_to_cstring(path.as_ref())?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { ffi::rj_vm_restore_checkpoint(path.as_ptr(), &mut handle) };
        status_to_result(status)?;
        Self::from_raw(handle, "VM")
    }

    pub fn step(&mut self) -> Result<bool> {
        let mut active = 0;
        let status = unsafe { ffi::rj_vm_step(self.ptr.as_ptr(), &mut active) };
        status_to_result(status).map(|()| active != 0)
    }

    pub fn run(&mut self) -> Result<u64> {
        let mut ticks_executed = 0;
        let status = unsafe { ffi::rj_vm_run(self.ptr.as_ptr(), &mut ticks_executed) };
        status_to_result(status).map(|()| ticks_executed)
    }

    pub fn save_checkpoint(&self, path: impl AsRef<Path>, tick: u64) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        let status = unsafe { ffi::rj_vm_save_checkpoint(self.ptr.as_ptr(), path.as_ptr(), tick) };
        status_to_result(status)
    }

    pub fn as_ptr(&self) -> *mut ffi::rj_vm_t {
        self.ptr.as_ptr()
    }

    fn from_raw(handle: *mut ffi::rj_vm_t, name: &'static str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(handle).ok_or(Error::NullHandle(name))?,
        })
    }
}

impl Drop for Vm {
    fn drop(&mut self) {
        unsafe { ffi::rj_vm_destroy(self.ptr.as_ptr()) }
    }
}

pub struct Decoder {
    ptr: NonNull<ffi::rj_code_decoder_t>,
}

impl Decoder {
    pub fn create(arch: ffi::rj_code_arch_t) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { ffi::rj_code_decoder_create(arch, &mut handle) };
        status_to_result(status)?;
        Self::from_raw(handle, "decoder")
    }

    pub fn decode<'decoder>(
        &'decoder mut self,
        binary_inst: &ffi::rj_code_binary_inst_t,
    ) -> Result<Instruction<'decoder>> {
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { ffi::rj_code_decoder_decode(self.ptr.as_ptr(), binary_inst, &mut handle) };
        status_to_result(status)?;
        Instruction::from_raw(handle, "instruction")
    }

    pub fn as_ptr(&self) -> *mut ffi::rj_code_decoder_t {
        self.ptr.as_ptr()
    }

    fn from_raw(handle: *mut ffi::rj_code_decoder_t, name: &'static str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(handle).ok_or(Error::NullHandle(name))?,
        })
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        unsafe { ffi::rj_code_decoder_destroy(self.ptr.as_ptr()) }
    }
}

pub struct Executable {
    ptr: NonNull<ffi::rj_code_executable_t>,
}

impl Executable {
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_to_cstring(path.as_ref())?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { ffi::rj_code_executable_create(path.as_ptr(), &mut handle) };
        status_to_result(status)?;
        Self::from_raw(handle, "executable")
    }

    pub fn num_code_objects(&self, target: ffi::rj_code_target_id_t) -> u32 {
        unsafe { ffi::rj_code_executable_num_code_objects(self.ptr.as_ptr(), target) }
    }

    pub fn code_object(
        &self,
        target: ffi::rj_code_target_id_t,
        index: u32,
    ) -> Result<CodeObject<'_>> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe {
            ffi::rj_code_executable_get_code_object(self.ptr.as_ptr(), target, index, &mut handle)
        };
        status_to_result(status)?;
        CodeObject::from_raw(handle, "code object")
    }

    pub fn as_ptr(&self) -> *mut ffi::rj_code_executable_t {
        self.ptr.as_ptr()
    }

    fn from_raw(handle: *mut ffi::rj_code_executable_t, name: &'static str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(handle).ok_or(Error::NullHandle(name))?,
        })
    }
}

impl Drop for Executable {
    fn drop(&mut self) {
        unsafe { ffi::rj_code_executable_destroy(self.ptr.as_ptr()) }
    }
}

pub struct CodeObject<'exec> {
    ptr: NonNull<ffi::rj_code_object_t>,
    _exec: PhantomData<&'exec Executable>,
}

impl<'exec> CodeObject<'exec> {
    pub fn instruction_list(
        &mut self,
        target: ffi::rj_code_target_id_t,
    ) -> Result<InstructionList> {
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { ffi::rj_code_inst_list_create(self.ptr.as_ptr(), target, &mut handle) };
        status_to_result(status)?;
        InstructionList::from_raw(handle, "instruction list")
    }

    pub fn basic_blocks(&mut self, target: ffi::rj_code_target_id_t) -> Result<BasicBlockList> {
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { ffi::rj_code_basic_block_list_create(self.ptr.as_ptr(), target, &mut handle) };
        status_to_result(status)?;
        BasicBlockList::from_raw(handle, "basic block list")
    }

    pub fn as_ptr(&self) -> *mut ffi::rj_code_object_t {
        self.ptr.as_ptr()
    }

    fn from_raw(handle: *mut ffi::rj_code_object_t, name: &'static str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(handle).ok_or(Error::NullHandle(name))?,
            _exec: PhantomData,
        })
    }
}

impl Drop for CodeObject<'_> {
    fn drop(&mut self) {
        unsafe {
            ffi::rj_code_object_destroy(self.ptr.as_ptr());
            ffi::rj_code_object_release(self.ptr.as_ptr());
        }
    }
}

pub struct InstructionList {
    ptr: NonNull<ffi::rj_code_inst_list_t>,
}

impl InstructionList {
    pub fn as_ptr(&self) -> *mut ffi::rj_code_inst_list_t {
        self.ptr.as_ptr()
    }

    fn from_raw(handle: *mut ffi::rj_code_inst_list_t, name: &'static str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(handle).ok_or(Error::NullHandle(name))?,
        })
    }
}

impl Drop for InstructionList {
    fn drop(&mut self) {
        unsafe { ffi::rj_code_inst_list_destroy(self.ptr.as_ptr()) }
    }
}

pub struct BasicBlockList {
    ptr: NonNull<ffi::rj_code_basic_block_list_t>,
}

impl BasicBlockList {
    pub fn len(&self) -> u32 {
        unsafe { ffi::rj_code_basic_block_list_size(self.ptr.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get(&self, index: u32) -> Result<BasicBlock<'_>> {
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { ffi::rj_code_basic_block_list_get(self.ptr.as_ptr(), index, &mut handle) };
        status_to_result(status)?;
        BasicBlock::from_raw(handle, "basic block")
    }

    pub fn as_ptr(&self) -> *mut ffi::rj_code_basic_block_list_t {
        self.ptr.as_ptr()
    }

    fn from_raw(handle: *mut ffi::rj_code_basic_block_list_t, name: &'static str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(handle).ok_or(Error::NullHandle(name))?,
        })
    }
}

impl Drop for BasicBlockList {
    fn drop(&mut self) {
        unsafe { ffi::rj_code_basic_block_list_destroy(self.ptr.as_ptr()) }
    }
}

pub struct BasicBlock<'list> {
    ptr: NonNull<ffi::rj_code_basic_block_t>,
    _list: PhantomData<&'list BasicBlockList>,
}

impl<'list> BasicBlock<'list> {
    pub fn start_offset(&self) -> u64 {
        unsafe { ffi::rj_code_basic_block_start_offset(self.ptr.as_ptr()) }
    }

    pub fn size(&self) -> u32 {
        unsafe { ffi::rj_code_basic_block_size(self.ptr.as_ptr()) }
    }

    pub fn num_instructions(&self) -> u32 {
        unsafe { ffi::rj_code_basic_block_num_instructions(self.ptr.as_ptr()) }
    }

    pub fn first_instruction(&self) -> Option<Instruction<'_>> {
        let handle = unsafe { ffi::rj_code_basic_block_first_inst(self.ptr.as_ptr()) };
        NonNull::new(handle.cast_mut()).map(|ptr| Instruction {
            ptr,
            _owner: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *mut ffi::rj_code_basic_block_t {
        self.ptr.as_ptr()
    }

    fn from_raw(handle: *mut ffi::rj_code_basic_block_t, name: &'static str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(handle).ok_or(Error::NullHandle(name))?,
            _list: PhantomData,
        })
    }
}

impl Drop for BasicBlock<'_> {
    fn drop(&mut self) {
        unsafe {
            ffi::rj_code_basic_block_destroy(self.ptr.as_ptr());
            ffi::rj_code_basic_block_release(self.ptr.as_ptr());
        }
    }
}

pub struct Instruction<'owner> {
    ptr: NonNull<ffi::rj_code_inst_t>,
    _owner: PhantomData<&'owner ()>,
}

impl<'owner> Instruction<'owner> {
    pub fn mnemonic(&self) -> Option<&str> {
        let mnemonic = unsafe { ffi::rj_code_inst_mnemonic(self.ptr.as_ptr()) };
        if mnemonic.is_null() {
            return None;
        }

        unsafe { CStr::from_ptr(mnemonic) }.to_str().ok()
    }

    pub fn size(&self) -> u32 {
        unsafe { ffi::rj_code_inst_size(self.ptr.as_ptr()) }
    }

    pub fn flags(&self) -> u32 {
        unsafe { ffi::rj_code_inst_flags(self.ptr.as_ptr()) }
    }

    pub fn disassemble(&self) -> Result<String> {
        let mut buffer = vec![0_i8; 256];
        let status = unsafe {
            ffi::rj_code_inst_disassemble(
                self.ptr.as_ptr(),
                buffer.as_mut_ptr(),
                buffer.len() as u32,
            )
        };
        status_to_result(status)?;
        Ok(unsafe { CStr::from_ptr(buffer.as_ptr()) }
            .to_string_lossy()
            .into_owned())
    }

    pub fn next(&self) -> Option<Instruction<'owner>> {
        let handle = unsafe { ffi::rj_code_inst_next(self.ptr.as_ptr()) };
        NonNull::new(handle.cast_mut()).map(|ptr| Instruction {
            ptr,
            _owner: PhantomData,
        })
    }

    pub fn as_ptr(&self) -> *mut ffi::rj_code_inst_t {
        self.ptr.as_ptr()
    }

    fn from_raw(handle: *mut ffi::rj_code_inst_t, name: &'static str) -> Result<Self> {
        Ok(Self {
            ptr: NonNull::new(handle).ok_or(Error::NullHandle(name))?,
            _owner: PhantomData,
        })
    }
}

fn status_to_result(status: ffi::rj_status_t) -> Result<()> {
    if status == ffi::rj_status_e::ROCJITSU_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::Status(status))
    }
}

fn status_name(status: ffi::rj_status_t) -> &'static str {
    match status {
        ffi::rj_status_e::ROCJITSU_STATUS_SUCCESS => "ROCJITSU_STATUS_SUCCESS",
        ffi::rj_status_e::ROCJITSU_STATUS_ERROR => "ROCJITSU_STATUS_ERROR",
        ffi::rj_status_e::ROCJITSU_STATUS_INVALID_ARGUMENT => "ROCJITSU_STATUS_INVALID_ARGUMENT",
        ffi::rj_status_e::ROCJITSU_STATUS_OUT_OF_RESOURCES => "ROCJITSU_STATUS_OUT_OF_RESOURCES",
        ffi::rj_status_e::ROCJITSU_STATUS_INVALID_CODE_OBJECT => {
            "ROCJITSU_STATUS_INVALID_CODE_OBJECT"
        }
        ffi::rj_status_e::ROCJITSU_STATUS_INVALID_FILE => "ROCJITSU_STATUS_INVALID_FILE",
    }
}

pub fn default_simulation_schema() -> PathBuf {
    Path::new(SCHEMA_DIR).join("simulation_config.fbs")
}

fn path_to_cstring(path: &Path) -> Result<CString> {
    #[cfg(unix)]
    {
        use std::os::unix::ffi::OsStrExt;
        Ok(CString::new(path.as_os_str().as_bytes())?)
    }

    #[cfg(not(unix))]
    {
        Ok(CString::new(path.as_os_str().to_string_lossy().as_bytes())?)
    }
}
