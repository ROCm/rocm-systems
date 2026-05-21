//! Low-level Rust bindings and light RAII wrappers for rocjitsu.
//!
//! This crate builds bindings from rocjitsu's public C API and generates Rust
//! types for rocjitsu's FlatBuffers schemas at Cargo build time. Raw bindgen
//! output is available under [`ffi`], generated schema types are re-exported
//! under [`fb`], and the top-level wrapper types provide ownership-aware access
//! to common rocjitsu handles.
//!
//! The build script discovers rocjitsu from the repository layout by default.
//! Override discovery with `ROCJITSU_ROOT`, `ROCJITSU_INCLUDE_DIR`,
//! `ROCJITSU_SCHEMA_DIR`, `ROCJITSU_LIB_DIR`, or `FLATC` when building against
//! an installed or out-of-tree rocjitsu.
//!
//! ```no_run
//! use rocjitsu_sys::{ffi, Decoder};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut decoder = Decoder::create(ffi::rj_code_arch_e::ROCJITSU_CODE_ARCH_CDNA4)?;
//! let s_nop = 0xBF80_0000;
//! let inst = decoder.decode(&s_nop)?;
//! assert_eq!(inst.mnemonic(), Some("s_nop"));
//! # Ok(())
//! # }
//! ```
//!
#![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]

use std::ffi::{CStr, CString, NulError};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

/// Raw bindgen output for rocjitsu's public C API.
///
/// Prefer the safe wrapper types in this crate for normal Rust code. Use this
/// module when a rocjitsu API surface does not yet have a wrapper.
pub mod ffi {
    #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
    #![allow(rustdoc::bare_urls, rustdoc::broken_intra_doc_links)]
    #![allow(clippy::all)]

    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

/// Generated Rust modules for rocjitsu's FlatBuffers schemas.
///
/// The module layout mirrors `flatc --rust --rust-module-root-file` output.
pub mod fbs_generated {
    #![allow(non_camel_case_types, non_snake_case, non_upper_case_globals)]
    #![allow(dead_code, unsafe_op_in_unsafe_fn, unused_imports)]
    #![allow(mismatched_lifetime_syntaxes)]
    #![allow(rustdoc::bare_urls, rustdoc::invalid_html_tags)]
    #![allow(clippy::all)]

    include!(concat!(env!("OUT_DIR"), "/fbs/mod.rs"));
}

/// Re-exports of generated FlatBuffers table types and helper functions.
pub mod fb {
    pub use crate::fbs_generated::rocjitsu::fb::*;
}

/// Schema directory embedded by the build script.
pub const SCHEMA_DIR: &str = env!("ROCJITSU_SYS_SCHEMA_DIR");

/// Result type returned by the safe rocjitsu wrappers.
pub type Result<T> = std::result::Result<T, Error>;

/// Error returned by the safe rocjitsu wrappers.
#[derive(Debug)]
pub enum Error {
    /// rocjitsu returned a non-success status code.
    Status(ffi::rj_status_t),
    /// rocjitsu returned success but did not fill an expected output handle.
    NullHandle(&'static str),
    /// A Rust string or path contained an interior NUL byte before crossing FFI.
    Nul(NulError),
}

impl Error {
    /// Returns the rocjitsu status code when this error came from the C API.
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

/// Owned rocjitsu virtual-machine handle.
///
/// The handle is destroyed when dropped. VM construction uses rocjitsu's JSON
/// configuration loader and therefore needs the matching FlatBuffers schema.
pub struct Vm {
    ptr: NonNull<ffi::rj_vm_t>,
}

impl Vm {
    /// Creates a VM from a JSON configuration file and schema file.
    pub fn create(json_path: impl AsRef<Path>, schema_path: impl AsRef<Path>) -> Result<Self> {
        let json_path = path_to_cstring(json_path.as_ref())?;
        let schema_path = path_to_cstring(schema_path.as_ref())?;
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { ffi::rj_vm_create(json_path.as_ptr(), schema_path.as_ptr(), &mut handle) };
        status_to_result(status)?;
        Self::from_raw(handle, "VM")
    }

    /// Creates a VM from a JSON configuration file using the generated default schema path.
    pub fn create_with_default_schema(json_path: impl AsRef<Path>) -> Result<Self> {
        Self::create(json_path, default_simulation_schema())
    }

    /// Creates a VM from a JSON configuration string and schema file.
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

    /// Creates a VM from a JSON configuration string using the generated default schema path.
    pub fn create_from_string_with_default_schema(json: &str) -> Result<Self> {
        Self::create_from_string(json, default_simulation_schema())
    }

    /// Restores a VM from a rocjitsu checkpoint file.
    pub fn restore_checkpoint(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_to_cstring(path.as_ref())?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { ffi::rj_vm_restore_checkpoint(path.as_ptr(), &mut handle) };
        status_to_result(status)?;
        Self::from_raw(handle, "VM")
    }

    /// Advances the VM by one simulation tick.
    ///
    /// Returns `true` when rocjitsu reports that wavefronts are still active.
    pub fn step(&mut self) -> Result<bool> {
        let mut active = 0;
        let status = unsafe { ffi::rj_vm_step(self.ptr.as_ptr(), &mut active) };
        status_to_result(status).map(|()| active != 0)
    }

    /// Runs the VM until rocjitsu reaches completion or its configured tick limit.
    ///
    /// Returns the number of ticks executed.
    pub fn run(&mut self) -> Result<u64> {
        let mut ticks_executed = 0;
        let status = unsafe { ffi::rj_vm_run(self.ptr.as_ptr(), &mut ticks_executed) };
        status_to_result(status).map(|()| ticks_executed)
    }

    /// Saves a VM checkpoint to `path` with the provided checkpoint tick.
    pub fn save_checkpoint(&self, path: impl AsRef<Path>, tick: u64) -> Result<()> {
        let path = path_to_cstring(path.as_ref())?;
        let status = unsafe { ffi::rj_vm_save_checkpoint(self.ptr.as_ptr(), path.as_ptr(), tick) };
        status_to_result(status)
    }

    /// Returns the raw rocjitsu VM pointer.
    ///
    /// The pointer remains owned by this wrapper and must not be destroyed by the caller.
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

/// Owned rocjitsu instruction decoder.
pub struct Decoder {
    ptr: NonNull<ffi::rj_code_decoder_t>,
}

impl Decoder {
    /// Creates a decoder for a rocjitsu ISA architecture.
    pub fn create(arch: ffi::rj_code_arch_t) -> Result<Self> {
        let mut handle = std::ptr::null_mut();
        let status = unsafe { ffi::rj_code_decoder_create(arch, &mut handle) };
        status_to_result(status)?;
        Self::from_raw(handle, "decoder")
    }

    /// Decodes one instruction from raw instruction bits.
    ///
    /// The returned instruction borrows the decoder because rocjitsu owns the
    /// backing decoder-specific instruction storage.
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

    /// Returns the raw rocjitsu decoder pointer.
    ///
    /// The pointer remains owned by this wrapper and must not be destroyed by the caller.
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

/// Owned rocjitsu executable handle.
pub struct Executable {
    ptr: NonNull<ffi::rj_code_executable_t>,
}

impl Executable {
    /// Loads an executable from disk.
    pub fn create(path: impl AsRef<Path>) -> Result<Self> {
        let path = path_to_cstring(path.as_ref())?;
        let mut handle = std::ptr::null_mut();
        let status = unsafe { ffi::rj_code_executable_create(path.as_ptr(), &mut handle) };
        status_to_result(status)?;
        Self::from_raw(handle, "executable")
    }

    /// Returns the number of code objects in this executable for `target`.
    pub fn num_code_objects(&self, target: ffi::rj_code_target_id_t) -> u32 {
        unsafe { ffi::rj_code_executable_num_code_objects(self.ptr.as_ptr(), target) }
    }

    /// Gets a code object from this executable.
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

    /// Returns the raw rocjitsu executable pointer.
    ///
    /// The pointer remains owned by this wrapper and must not be destroyed by the caller.
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

/// Code object borrowed from an [`Executable`].
pub struct CodeObject<'exec> {
    ptr: NonNull<ffi::rj_code_object_t>,
    _exec: PhantomData<&'exec Executable>,
}

impl<'exec> CodeObject<'exec> {
    /// Decodes this code object into a rocjitsu instruction list.
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

    /// Builds the basic-block list for this code object.
    pub fn basic_blocks(&mut self, target: ffi::rj_code_target_id_t) -> Result<BasicBlockList> {
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { ffi::rj_code_basic_block_list_create(self.ptr.as_ptr(), target, &mut handle) };
        status_to_result(status)?;
        BasicBlockList::from_raw(handle, "basic block list")
    }

    /// Returns the raw rocjitsu code-object pointer.
    ///
    /// The pointer remains owned by this wrapper and must not be destroyed by the caller.
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

/// Owned rocjitsu decoded instruction-list handle.
pub struct InstructionList {
    ptr: NonNull<ffi::rj_code_inst_list_t>,
}

impl InstructionList {
    /// Returns the raw rocjitsu instruction-list pointer.
    ///
    /// The pointer remains owned by this wrapper and must not be destroyed by the caller.
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

/// Owned rocjitsu basic-block-list handle.
pub struct BasicBlockList {
    ptr: NonNull<ffi::rj_code_basic_block_list_t>,
}

impl BasicBlockList {
    /// Returns the number of basic blocks in the list.
    pub fn len(&self) -> u32 {
        unsafe { ffi::rj_code_basic_block_list_size(self.ptr.as_ptr()) }
    }

    /// Returns whether the list contains no basic blocks.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Gets a basic block by index.
    pub fn get(&self, index: u32) -> Result<BasicBlock<'_>> {
        let mut handle = std::ptr::null_mut();
        let status =
            unsafe { ffi::rj_code_basic_block_list_get(self.ptr.as_ptr(), index, &mut handle) };
        status_to_result(status)?;
        BasicBlock::from_raw(handle, "basic block")
    }

    /// Returns the raw rocjitsu basic-block-list pointer.
    ///
    /// The pointer remains owned by this wrapper and must not be destroyed by the caller.
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

/// Basic block borrowed from a [`BasicBlockList`].
pub struct BasicBlock<'list> {
    ptr: NonNull<ffi::rj_code_basic_block_t>,
    _list: PhantomData<&'list BasicBlockList>,
}

impl<'list> BasicBlock<'list> {
    /// Returns the byte offset of the first instruction in the block.
    pub fn start_offset(&self) -> u64 {
        unsafe { ffi::rj_code_basic_block_start_offset(self.ptr.as_ptr()) }
    }

    /// Returns the block size in bytes.
    pub fn size(&self) -> u32 {
        unsafe { ffi::rj_code_basic_block_size(self.ptr.as_ptr()) }
    }

    /// Returns the number of instructions in the block.
    pub fn num_instructions(&self) -> u32 {
        unsafe { ffi::rj_code_basic_block_num_instructions(self.ptr.as_ptr()) }
    }

    /// Returns the first instruction in the block, if present.
    pub fn first_instruction(&self) -> Option<Instruction<'_>> {
        let handle = unsafe { ffi::rj_code_basic_block_first_inst(self.ptr.as_ptr()) };
        NonNull::new(handle.cast_mut()).map(|ptr| Instruction {
            ptr,
            _owner: PhantomData,
        })
    }

    /// Returns the raw rocjitsu basic-block pointer.
    ///
    /// The pointer remains owned by this wrapper and must not be destroyed by the caller.
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

/// Instruction borrowed from a decoder or basic block.
pub struct Instruction<'owner> {
    ptr: NonNull<ffi::rj_code_inst_t>,
    _owner: PhantomData<&'owner ()>,
}

impl<'owner> Instruction<'owner> {
    /// Returns the instruction mnemonic when rocjitsu provides valid UTF-8 text.
    pub fn mnemonic(&self) -> Option<&str> {
        let mnemonic = unsafe { ffi::rj_code_inst_mnemonic(self.ptr.as_ptr()) };
        if mnemonic.is_null() {
            return None;
        }

        unsafe { CStr::from_ptr(mnemonic) }.to_str().ok()
    }

    /// Returns the instruction size in bytes.
    pub fn size(&self) -> u32 {
        unsafe { ffi::rj_code_inst_size(self.ptr.as_ptr()) }
    }

    /// Returns rocjitsu's instruction flag bitmask.
    pub fn flags(&self) -> u32 {
        unsafe { ffi::rj_code_inst_flags(self.ptr.as_ptr()) }
    }

    /// Returns rocjitsu's disassembly text for this instruction.
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

    /// Returns the next instruction in the same basic block, if present.
    pub fn next(&self) -> Option<Instruction<'owner>> {
        let handle = unsafe { ffi::rj_code_inst_next(self.ptr.as_ptr()) };
        NonNull::new(handle.cast_mut()).map(|ptr| Instruction {
            ptr,
            _owner: PhantomData,
        })
    }

    /// Returns the raw rocjitsu instruction pointer.
    ///
    /// The pointer is borrowed and must not be destroyed by the caller.
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

/// Returns the default `simulation_config.fbs` path discovered by the build script.
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
