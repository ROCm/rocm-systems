// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! HSA runtime ABI frontend backed directly by `rocddi`.
//!
//! This frontend is early-access runtime software, not a drop-in replacement
//! for the production `ROCr` HSA runtime. Its packaging, symbol-versioning,
//! coexistence, platform coverage, and hardware qualification remain subject
//! to change. Callers must select it explicitly.
//!
//! The crate owns HSA-visible process state and translates each C entry point
//! into the private rocddi mechanisms. Public handles are integer or pointer
//! encodings into runtime-owned objects; the individual modules document the
//! lifetime and synchronization rules for those objects.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_ptr_alignment,
    clippy::cast_sign_loss,
    clippy::manual_let_else,
    clippy::match_same_arms,
    clippy::must_use_candidate,
    clippy::semicolon_if_nothing_returned,
    clippy::too_many_lines,
    clippy::wildcard_imports
)]

// Unit-test executables call the Rust entry points directly. They do not
// export the shared library's versioned ABI symbols.
#[cfg(all(target_os = "linux", not(test)))]
core::arch::global_asm!(include_str!("exports_linux.S"));
#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
core::arch::global_asm!(include_str!(concat!(
    env!("OUT_DIR"),
    "/api_trace_dispatch_linux.S"
)));

#[cfg(all(target_os = "linux", target_arch = "x86_64", not(test)))]
mod api_trace;
mod callback_arg;
mod ffi;
mod finalizer;
mod image_abi;
mod image_gfx12;
mod isa;
mod loader;
mod memory;
mod pc_sampling;
mod platform;
mod queue;
mod queue_intercept;
mod runtime;
mod signal;
mod tool_events;

mod api;
pub(crate) use api::agent::hsa_agent_get_info;

#[cfg(test)]
pub(crate) use api::lifecycle::{hsa_init, hsa_shut_down};
#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests;
