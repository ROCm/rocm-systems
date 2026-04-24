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
/// The firmware writes two records per dispatch: one at dispatch-start
/// (record_type=1) and one at EOP / dispatch-end (record_type=2).
/// Host-side tooling correlates records by dispatch order and pairs
/// start/end timestamps.  Kernel identity is resolved host-side via
/// the ROCr code-object symbol table, not embedded in the record.
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
struct mec_dispatch_record {
  uint32_t ts_lo;        // GPU clock counter [31:0]
  uint32_t ts_hi;        // GPU clock counter [63:32]
  uint32_t record_type;  // 1 = dispatch-start, 2 = EOP (dispatch-end)
  uint32_t reserved;     // Must be 0
};

}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_MEC_DISPATCH_RECORD_H_
