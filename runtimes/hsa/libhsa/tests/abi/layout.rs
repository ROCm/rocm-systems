// SPDX-License-Identifier: MIT

//! Rust half of the public-header ABI comparison.

#[allow(dead_code)]
#[path = "../../src/ffi.rs"]
mod ffi;

use ffi::*;
use std::mem::{align_of, size_of, MaybeUninit};

fn field<T>(name: &str, base: *const u8, pointer: *const T) {
    println!(
        "field {name} {} {}",
        (pointer as usize) - (base as usize),
        size_of::<T>()
    );
}

macro_rules! record {
    ($name:ident) => {
        println!(
            "type {} {} {}",
            stringify!($name),
            size_of::<$name>(),
            align_of::<$name>()
        );
    };
}

macro_rules! member {
    ($name:ident, $field:ident) => {{
        let record = MaybeUninit::<$name>::uninit();
        let base = record.as_ptr();
        // SAFETY: addr_of only projects an address within aligned storage.
        let pointer = unsafe { std::ptr::addr_of!((*base).$field) };
        field(
            concat!(stringify!($name), ".", stringify!($field)),
            base.cast(),
            pointer,
        );
    }};
}

fn main() {
    record!(HsaSignal);
    member!(HsaSignal, handle);
    record!(HsaDim3);
    member!(HsaDim3, x);
    member!(HsaDim3, y);
    member!(HsaDim3, z);
    record!(HsaQueue);
    member!(HsaQueue, queue_type);
    member!(HsaQueue, features);
    member!(HsaQueue, base_address);
    member!(HsaQueue, doorbell_signal);
    member!(HsaQueue, size);
    member!(HsaQueue, reserved);
    member!(HsaQueue, id);
    record!(HsaExtControlDirectives);
    member!(HsaExtControlDirectives, control_directives_mask);
    member!(HsaExtControlDirectives, break_exceptions_mask);
    member!(HsaExtControlDirectives, detect_exceptions_mask);
    member!(HsaExtControlDirectives, max_dynamic_group_size);
    member!(HsaExtControlDirectives, max_flat_grid_size);
    member!(HsaExtControlDirectives, max_flat_workgroup_size);
    member!(HsaExtControlDirectives, reserved1);
    member!(HsaExtControlDirectives, required_grid_size);
    member!(HsaExtControlDirectives, required_workgroup_size);
    member!(HsaExtControlDirectives, required_dim);
    member!(HsaExtControlDirectives, reserved2);
}
