//! Legacy HSAIL finalizer entry points from the public HSA extension ABI.
//!
//! The optional finalizer is not implemented or advertised by this frontend.
//! `ROCr` still exports these six entry points even when its finalizer plugin is
//! absent, so callers must be able to resolve them and receive the same
//! `NOT_INITIALIZED` result as `ROCr`'s missing-plugin implementation.

use crate::ffi::{
    HsaCodeObject, HsaExtControlDirectives, HsaExtProgram, HsaIsa, NOT_INITIALIZED, Status,
};
use std::ffi::{c_char, c_void};

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_program_create(
    _machine_model: u32,
    _profile: u32,
    _default_float_rounding_mode: u32,
    _options: *const c_char,
    _program: *mut HsaExtProgram,
) -> Status {
    NOT_INITIALIZED
}

#[unsafe(no_mangle)]
pub extern "C" fn hsa_ext_program_destroy(_program: HsaExtProgram) -> Status {
    NOT_INITIALIZED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_program_add_module(
    _program: HsaExtProgram,
    _module: *mut c_void,
) -> Status {
    NOT_INITIALIZED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_program_iterate_modules(
    _program: HsaExtProgram,
    _callback: Option<unsafe extern "C" fn(HsaExtProgram, *mut c_void, *mut c_void) -> Status>,
    _data: *mut c_void,
) -> Status {
    NOT_INITIALIZED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_program_get_info(
    _program: HsaExtProgram,
    _attribute: u32,
    _value: *mut c_void,
) -> Status {
    NOT_INITIALIZED
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn hsa_ext_program_finalize(
    _program: HsaExtProgram,
    _isa: HsaIsa,
    _call_convention: i32,
    _control_directives: HsaExtControlDirectives,
    _options: *const c_char,
    _code_object_type: u32,
    _code_object: *mut HsaCodeObject,
) -> Status {
    NOT_INITIALIZED
}
