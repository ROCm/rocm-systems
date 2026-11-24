// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

//! Failure-only queue backing ranges from retained ownership metadata.

use super::super::{
    drm,
    memory::diagnostics::{FaultCacheUnavailable, write_fault_cache},
};
use super::{DeviceVm, KfdQueue, QueueContext, uapi};
use std::io::{self, Write};
use std::sync::atomic::{AtomicU64, Ordering};

/// Sequence only of enabled private cache captures, not native queue IDs.
static NEXT_CACHE_CAPTURE: AtomicU64 = AtomicU64::new(1);

/// Bounded activation evidence retained until a publication failure.
pub(super) struct CacheQueueActivation {
    /// Distinguishes captures even when native IDs and allocation VAs are reused.
    ordinal: u64,
    /// Exact CREATE input before KFD overlays its output fields.
    request: uapi::CreateQueue,
    /// Independent retained-render sample immediately before native CREATE.
    before_create: Result<drm::GpuVmFault, FaultCacheUnavailable>,
    /// Independent sample after successful CREATE, before any producer store.
    pub(super) after_create: Option<Result<drm::GpuVmFault, FaultCacheUnavailable>>,
}

impl CacheQueueActivation {
    #[cold]
    pub(super) fn capture(vm: &DeviceVm, request: uapi::CreateQueue) -> Option<Self> {
        if !vm.cache_diagnostics_enabled() {
            return None;
        }
        let ordinal = NEXT_CACHE_CAPTURE.fetch_add(1, Ordering::Relaxed);
        Some(Self {
            ordinal,
            request,
            before_create: vm.passive_fault_cache(),
            after_create: None,
        })
    }
}

impl KfdQueue {
    #[cold]
    pub(crate) fn diagnose_incomplete_cache_wait(&self) {
        if let Some(queue_id) = self.id {
            self.diagnose_cache_activation("code-cache-wait-incomplete-before-UPDATE_QUEUE");
            self.vm.diagnose_incomplete_cache_wait(queue_id);
        }
    }

    #[cold]
    pub(super) fn diagnose_cache_activation(&self, boundary: &str) {
        if self.cache_activation.is_none() || !self.vm.cache_diagnostics_enabled() {
            return;
        }
        let _ = self.write_cache_activation(&mut io::stderr().lock(), boundary);
    }

    /// All samples are already captured; formatting performs no native calls.
    pub(super) fn write_cache_activation(
        &self,
        output: &mut impl Write,
        boundary: &str,
    ) -> io::Result<()> {
        let Some(activation) = &self.cache_activation else {
            return Ok(());
        };
        writeln!(
            output,
            "rocddi: cache_queue_activation; report_boundary={boundary}; vm_owner={:p}; \
             cache_queue_capture_ordinal={}; queue_id={:?}; expected_gpu_id={}; create_request={:?}",
            &*self.vm, activation.ordinal, self.id, self.vm.gpu_id, activation.request,
        )?;
        write!(
            output,
            "rocddi: cache_queue_capture_ordinal={}; ",
            activation.ordinal
        )?;
        self.vm.write_fresh_render_fault(output)?;
        for (boundary, fault) in [
            (
                "code-cache-before-CREATE_QUEUE",
                Some(&activation.before_create),
            ),
            (
                "code-cache-after-CREATE_QUEUE-before-producer-store",
                activation.after_create.as_ref(),
            ),
        ] {
            write!(
                output,
                "rocddi: cache_queue_capture_ordinal={}; boundary={boundary}; drm_gpuvm_fault_cache=",
                activation.ordinal,
            )?;
            match fault {
                Some(fault) => write_fault_cache(output, fault)?,
                None => write!(output, "unavailable(not_captured)")?,
            }
            writeln!(
                output,
                "; semantics=untimestamped_may_be_stale_or_incomplete"
            )?;
        }
        Ok(())
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
