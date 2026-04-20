#ifndef HOTSWAP_TRANSPILER_USER_SGPR_LAYOUT_HPP
#define HOTSWAP_TRANSPILER_USER_SGPR_LAYOUT_HPP

#include "code_object_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace transpiler {

// UserSgprLayout — what each SGPR contains at function entry on the source
// ISA, derived from the kernel descriptor (KernelMeta::kernelCodeProperties,
// kernargPreload, computePgmRsrc2).
//
// This is the single source of truth for the source-ISA SGPR ABI inside the
// raiser. The previous implementation hardcoded "kernarg_segment_ptr at
// s[0:1] / workgroup_id_x at s[2] / workgroup_id_y at s[3]" in raiser.cpp
// Phase 4 and "isKernarg = (baseIdx == 0)" in handle_smem.cpp; both are
// wrong as soon as kernarg-preload or a non-default kernel_code_properties
// value is in play (gfx1250 Triton kernels routinely have both).
//
// The factory `fromKernelMeta` walks the bits of `kernel_code_properties`
// in their canonical order (see LLVM's KERNEL_CODE_PROPERTY_* enum), then
// appends `kernarg_preload_length` PreloadedKernarg entries (each one
// dword), then appends the system-SGPR workgroup ids enabled by
// `compute_pgm_rsrc2`. The resulting `entries` vector is indexed by SGPR
// index, so `entries[i]` describes the contents of SGPR i at kernel entry.
//
// The convenience integer fields (`kernargSegmentPtrSgpr` etc.) are the
// SGPR indices of the *first* dword of the corresponding source. They are
// -1 when the source is disabled. Handlers that want to ask "is this SGPR
// the kernarg pointer?" should compare against these fields rather than
// re-walking `entries`.
//
// We never silently fall back to a hardcoded layout: if
// `KernelMeta::hasKernelDescriptor` is false the factory aborts via
// llvm::report_fatal_error. The caller is expected to refuse the lift
// before reaching this point.
struct UserSgprLayout {
  enum class Source : uint8_t {
    Unset,
    PrivateSegmentBuffer, // 4 dwords (s[i:i+3]) — Shader Resource Descriptor
    DispatchPtr,          // 2 dwords — pointer to the AQL dispatch packet
    QueuePtr,             // 2 dwords — pointer to the HSA queue object
    KernargSegmentPtr,    // 2 dwords — pointer to the kernarg segment
    DispatchId,           // 2 dwords — dispatch identifier
    FlatScratchInit,      // 2 dwords — flat scratch base/size init
    PrivateSegmentSize,   // 1 dword  — size of private segment per work-item
    PreloadedKernarg,     // 1 dword  — preloaded kernarg dword (gfx1250 ABI)
    WorkgroupIdX,         // 1 dword  — system SGPR (compute_pgm_rsrc2 bit 7)
    WorkgroupIdY,         // 1 dword  — system SGPR (compute_pgm_rsrc2 bit 8)
    WorkgroupIdZ,         // 1 dword  — system SGPR (compute_pgm_rsrc2 bit 9)
    WorkgroupInfo,        // 1 dword  — system SGPR (compute_pgm_rsrc2 bit 10)
  };

  struct Entry {
    Source source = Source::Unset;
    // For multi-dword sources (DispatchPtr, KernargSegmentPtr, ...) this is
    // the dword index within the source: 0 = lo, 1 = hi for 2-dword
    // sources, 0..3 for the 4-dword PrivateSegmentBuffer SRD.
    uint8_t subDword = 0;
    // For PreloadedKernarg only: the byte offset within the kernarg segment
    // that this dword originated from. Computed as
    // `(kernarg_preload_offset + i) * 4` per the gfx1250 ABI. Used by the
    // raiser to look up the matching kernarg parameter and extract the
    // appropriate dword from it.
    uint16_t kernargByteOffset = 0;
  };

  std::vector<Entry> entries;
  uint8_t userSgprCount = 0; // == entries.size() at end of user-SGPR region

  // Convenience: SGPR index that holds the *low* dword of the corresponding
  // source. -1 when the source is not enabled in `kernel_code_properties`
  // / `compute_pgm_rsrc2`. Handlers that need to identify the kernarg
  // pointer SGPR should compare against `kernargSegmentPtrSgpr` rather than
  // assuming `0`.
  int kernargSegmentPtrSgpr = -1;
  int dispatchPtrSgpr = -1;
  int queuePtrSgpr = -1;
  int dispatchIdSgpr = -1;
  int flatScratchInitSgpr = -1;
  int privateSegmentBufferSgpr = -1;
  int privateSegmentSizeSgpr = -1;
  int workgroupIdXSgpr = -1;
  int workgroupIdYSgpr = -1;
  int workgroupIdZSgpr = -1;
  int workgroupInfoSgpr = -1;

  // SGPR index of the *first* preloaded kernarg dword, or -1 when
  // kernargPreload.length == 0. Useful for debugging / diagnostics.
  int firstPreloadedKernargSgpr = -1;
  uint8_t preloadedKernargLength = 0;
  uint16_t preloadedKernargByteOffset = 0;

  // Build the layout from a parsed kernel descriptor. Aborts via
  // report_fatal_error if `meta.hasKernelDescriptor` is false; the caller
  // is responsible for refusing the lift earlier in the pipeline.
  static UserSgprLayout fromKernelMeta(const KernelMeta &meta);

  // Render a one-line debug summary: useful for HSA_HOTSWAP_DEBUG output
  // and for failure diagnostics. Format:
  //   "user_sgpr=N: s[0:1]=KernargSegmentPtr s[2]=PreloadedKernarg(off=0) ..."
  std::string toString() const;
};

} // namespace transpiler

#endif
