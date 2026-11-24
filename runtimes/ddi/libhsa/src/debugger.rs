// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Native debugger rendezvous, code-object publication, and retained trap code.

mod loader;
mod trap;

pub(crate) use loader::{ExecutableDebugInfo, diagnostic_record, record_address};
pub(crate) use trap::{PendingDebugger, TrapRestore, trap_bytes};
