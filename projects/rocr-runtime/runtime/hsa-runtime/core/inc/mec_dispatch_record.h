////////////////////////////////////////////////////////////////////////////////
//
// MEC firmware dispatch record layout (userspace mirror for sizing).
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_MEC_DISPATCH_RECORD_H_
#define HSA_RUNTIME_CORE_INC_MEC_DISPATCH_RECORD_H_

#include <cstdint>

namespace rocr {
namespace AMD {

/// 16-byte record written by MEC firmware per dispatch event.
///
/// The firmware writes two records per dispatch (one at dispatch-start
/// and one at EOP / dispatch-end), tagged with FW-defined `record_type`
/// values. HSA does NOT assign stable start-vs-end semantics to those
/// tag values: the start/end mapping is FW-version-specific (e.g. on
/// gfx950 the two values are reversed relative to the original 2026-03-30
/// contract; see the kernel_dispatch_record tracepoint comment in
/// lttng/rocm_hsa_tp.h). Host-side tooling correlates records by
/// (queue_id, dispatch_idx) and either orders by gpu_ts or applies its
/// own FW-version-specific record_type interpretation. Kernel identity
/// is resolved host-side via the ROCr code-object symbol table, not
/// embedded in the record.
///
/// Design rationale (2026-03-30):
///   - Minimises firmware complexity and risk of GPU hangs.
///   - 16 bytes = 4 DWords = single TC write transaction.
///   - Kernel name / object VA not needed in the record because the host
///     already knows which kernel was submitted to each queue slot.
///   - Richer per-dispatch metadata (doorbell, ring_index, etc.) can be
///     added later by extending this struct and the firmware write path,
///     but is not required for the primary use-case of late-attach
///     timestamp profiling.
///
/// Field-name history note: the trailing slot is named `reserved` for
/// historical reasons (the original 2026-03-30 contract reserved it as
/// padding "must be 0"). Subsequent firmware revisions repurposed this
/// slot to carry the per-queue monotonic dispatch index that the FW
/// writes alongside every record. Host-side consumers therefore treat
/// this slot as the FW-written dispatch_idx; see the
/// `mec_dispatch_record_16` mirror in core/inc/dispatch_log.h, which
/// names the slot `dispatch_idx` to reflect its current FW semantics,
/// and the rocm_hsa:kernel_dispatch_record tracepoint contract in
/// lttng/rocm_hsa_tp.h. The `reserved` name is retained on this struct
/// for source-compatibility with existing consumers; it is NOT zero in
/// FW-written records.
struct mec_dispatch_record {
  uint32_t ts_lo;        // GPU clock counter [31:0]
  uint32_t ts_hi;        // GPU clock counter [63:32]
  uint32_t record_type;  // FW-defined event tag; HSA does not assign
                         // stable start/end semantics. The start-vs-end
                         // mapping is FW-version-specific (see struct
                         // doc above and lttng/rocm_hsa_tp.h). Host
                         // consumers join on (queue_id, dispatch_idx)
                         // and order by gpu_ts.
  uint32_t reserved;     // Historical name; FW writes per-queue monotonic
                         // dispatch_idx here (see struct doc above).
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_MEC_DISPATCH_RECORD_H_
