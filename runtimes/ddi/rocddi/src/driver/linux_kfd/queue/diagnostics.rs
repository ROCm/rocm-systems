// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Failure-only queue backing ranges from retained ownership metadata.

use super::{KfdQueue, QueueContext};
use std::io::{self, Write};

impl KfdQueue {
    #[cold]
    pub(crate) fn diagnose_incomplete_cache_wait(&self) {
        if let Some(queue_id) = self.id {
            self.vm.diagnose_incomplete_cache_wait(queue_id);
        }
    }

    #[cold]
    pub(super) fn diagnose_backing_ranges(&self, queue_id: u32) {
        if self.vm.kfd().check_process().is_err()
            || std::env::var_os("ROCDDI_CODE_CACHE_DIAGNOSTICS").is_none_or(|value| value != "1")
        {
            return;
        }
        let _ = self.write_backing_range_diagnostic(&mut io::stderr().lock(), queue_id);
    }

    /// Uses cached owner metadata; no allocation health check or header read.
    pub(super) fn write_backing_range_diagnostic(
        &self,
        output: &mut impl Write,
        queue_id: u32,
    ) -> io::Result<()> {
        if let Some(eop) = &self.backing[2] {
            eop.write_queue_backing_diagnostic(output, queue_id, "eop")?;
        } else {
            writeln!(
                output,
                "rocddi: queue_backing; queue_id={queue_id}; role=eop; backing=absent"
            )?;
        }
        let context_size = match &self.context {
            Some(QueueContext::Native(storage)) => {
                storage.write_queue_backing_diagnostic(output, queue_id, "context")?;
                Some(storage.cached_info().native_size)
            }
            Some(QueueContext::Svm(storage)) => {
                writeln!(
                    output,
                    "rocddi: queue_backing; queue_id={queue_id}; role=context; kind=svm; \
                     cpu={:#x}; gpu={:#x}; native_size_bytes={}; cpu_writable={}; \
                     svm_registration=completed_at_creation; gpu_permissions=unavailable(not_retained)",
                    storage.address(),
                    storage.address(),
                    storage.usable_size(),
                    storage.has_writable_mapping(),
                )?;
                Some(storage.usable_size() as u64)
            }
            None => {
                writeln!(
                    output,
                    "rocddi: queue_backing; queue_id={queue_id}; role=context; backing=absent"
                )?;
                None
            }
        };
        write!(
            output,
            "rocddi: queue_backing; queue_id={queue_id}; context_layout="
        )?;
        match (self.aql, context_size) {
            (Some(aql), Some(total_size))
                if aql.properties.context_size != 0 && aql.properties.control_stack_size != 0 =>
            {
                let context_span =
                    u64::from(aql.properties.context_size) * u64::from(aql.properties.xcc_count);
                write!(
                    output,
                    "retained_aql_properties; xcc_count={}; context_stride_bytes={}; \
                     context_span_size_bytes={context_span}; control_stack_prefix_size_bytes_per_xcc={}; \
                     debug_offset_bytes={context_span}; debug_size_bytes=unavailable(not_retained); \
                     post_context_allocation_tail_size_bytes=",
                    aql.properties.xcc_count,
                    aql.properties.context_size,
                    aql.properties.control_stack_size,
                )?;
                match total_size.checked_sub(context_span) {
                    Some(size) => write!(output, "{size}")?,
                    None => write!(output, "unavailable(inconsistent_extent)")?,
                }
            }
            _ => write!(output, "unavailable(not_retained)")?,
        }
        writeln!(output, "; vm_trap_owner=unavailable(no_lifecycle_lock)")
    }
}
