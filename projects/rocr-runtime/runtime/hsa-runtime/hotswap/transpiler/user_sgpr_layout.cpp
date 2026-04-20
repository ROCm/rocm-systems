#include "user_sgpr_layout.hpp"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/AMDHSAKernelDescriptor.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <sstream>

namespace transpiler {

namespace {

// Append `count` Entry rows to `entries` describing a multi-dword source
// that occupies `count` consecutive SGPRs. Returns the SGPR index of the
// first (low) dword, which the factory stores into the convenience
// `*Sgpr` field for the corresponding source.
int appendSource(std::vector<UserSgprLayout::Entry> &entries,
                 UserSgprLayout::Source src, unsigned count) {
  int firstIdx = static_cast<int>(entries.size());
  for (unsigned i = 0; i < count; ++i) {
    UserSgprLayout::Entry e;
    e.source = src;
    e.subDword = static_cast<uint8_t>(i);
    entries.push_back(e);
  }
  return firstIdx;
}

const char *sourceName(UserSgprLayout::Source s) {
  switch (s) {
  case UserSgprLayout::Source::Unset:                return "Unset";
  case UserSgprLayout::Source::PrivateSegmentBuffer: return "PrivateSegmentBuffer";
  case UserSgprLayout::Source::DispatchPtr:          return "DispatchPtr";
  case UserSgprLayout::Source::QueuePtr:             return "QueuePtr";
  case UserSgprLayout::Source::KernargSegmentPtr:    return "KernargSegmentPtr";
  case UserSgprLayout::Source::DispatchId:           return "DispatchId";
  case UserSgprLayout::Source::FlatScratchInit:      return "FlatScratchInit";
  case UserSgprLayout::Source::PrivateSegmentSize:   return "PrivateSegmentSize";
  case UserSgprLayout::Source::PreloadedKernarg:     return "PreloadedKernarg";
  case UserSgprLayout::Source::WorkgroupIdX:         return "WorkgroupIdX";
  case UserSgprLayout::Source::WorkgroupIdY:         return "WorkgroupIdY";
  case UserSgprLayout::Source::WorkgroupIdZ:         return "WorkgroupIdZ";
  case UserSgprLayout::Source::WorkgroupInfo:        return "WorkgroupInfo";
  }
  return "<invalid>";
}

} // namespace

UserSgprLayout UserSgprLayout::fromKernelMeta(const KernelMeta &meta) {
  if (!meta.hasKernelDescriptor) {
    // No fallback: a missing KD means we cannot derive the SGPR ABI, and
    // every downstream decision (SGPR seeding in Phase 4, kernarg
    // detection in handle_smem) would silently produce wrong results.
    // Per the user-rules, we surface this loudly rather than guessing.
    llvm::report_fatal_error(
        llvm::Twine("transpiler: UserSgprLayout::fromKernelMeta: kernel '") +
        meta.name +
        "' has no parsed kernel descriptor. Cannot derive user-SGPR ABI; "
        "refuse the lift instead of guessing a hardcoded layout.");
  }

  using namespace llvm::amdhsa;

  UserSgprLayout layout;

  const uint16_t kcp = meta.kernelCodeProperties;

  // Walk the canonical KERNEL_CODE_PROPERTY bit order.
  // Source: LLVM AMDHSAKernelDescriptor.h (bits 0..6 in ascending order).
  if (kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER)
    layout.privateSegmentBufferSgpr =
        appendSource(layout.entries, Source::PrivateSegmentBuffer, 4);
  if (kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR)
    layout.dispatchPtrSgpr =
        appendSource(layout.entries, Source::DispatchPtr, 2);
  if (kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR)
    layout.queuePtrSgpr = appendSource(layout.entries, Source::QueuePtr, 2);
  if (kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR)
    layout.kernargSegmentPtrSgpr =
        appendSource(layout.entries, Source::KernargSegmentPtr, 2);
  if (kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID)
    layout.dispatchIdSgpr = appendSource(layout.entries, Source::DispatchId, 2);
  if (kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT)
    layout.flatScratchInitSgpr =
        appendSource(layout.entries, Source::FlatScratchInit, 2);
  if (kcp & KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE)
    layout.privateSegmentSizeSgpr =
        appendSource(layout.entries, Source::PrivateSegmentSize, 1);

  // Kernarg preload (gfx1250+): N dwords from kernarg memory at byte
  // offset (preloadOffset * 4) get loaded into the user SGPRs immediately
  // above the enable_sgpr_*-selected ones, before kernel entry. The
  // sequence is little-endian dword-aligned: dword i goes into
  // s[user_sgpr_count_so_far + i] and corresponds to kernarg bytes
  // [(offset+i)*4 .. (offset+i+1)*4 - 1].
  const uint8_t preloadLen = static_cast<uint8_t>(
      (meta.kernargPreload >> KERNARG_PRELOAD_SPEC_LENGTH_SHIFT) &
      ((1 << KERNARG_PRELOAD_SPEC_LENGTH_WIDTH) - 1));
  const uint16_t preloadOffsetDwords = static_cast<uint16_t>(
      (meta.kernargPreload >> KERNARG_PRELOAD_SPEC_OFFSET_SHIFT) &
      ((1 << KERNARG_PRELOAD_SPEC_OFFSET_WIDTH) - 1));
  layout.preloadedKernargLength = preloadLen;
  layout.preloadedKernargByteOffset =
      static_cast<uint16_t>(preloadOffsetDwords * 4);
  if (preloadLen > 0) {
    layout.firstPreloadedKernargSgpr = static_cast<int>(layout.entries.size());
    for (unsigned i = 0; i < preloadLen; ++i) {
      Entry e;
      e.source = Source::PreloadedKernarg;
      // Each preloaded dword is its own independent SGPR (no multi-dword
      // bundling from the KD's perspective — the byte offset alone identifies
      // which kernarg slice it carries). subDword stays 0 so Phase 4's
      // "act on subDword==0 only" loop visits every preload entry.
      e.subDword = 0;
      e.kernargByteOffset =
          static_cast<uint16_t>((preloadOffsetDwords + i) * 4);
      layout.entries.push_back(e);
    }
  }

  layout.userSgprCount = static_cast<uint8_t>(layout.entries.size());

  // Sanity-check against compute_pgm_rsrc2.USER_SGPR_COUNT. We deliberately
  // read 5 bits (the GFX6-GFX120 width) rather than 6 (gfx125): for the
  // current corpus all kernels fit in 31 user SGPRs and the GFX125 wider
  // field is not exercised. If a future kernel sets bit 6 we'll trip the
  // mismatch assertion below and revisit.
  const uint8_t pgmRsrc2UserSgprCount = static_cast<uint8_t>(
      (meta.computePgmRsrc2 >> COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_SHIFT) &
      ((1 << COMPUTE_PGM_RSRC2_GFX6_GFX120_USER_SGPR_COUNT_WIDTH) - 1));
  if (pgmRsrc2UserSgprCount != layout.userSgprCount) {
    llvm::report_fatal_error(
        llvm::Twine("transpiler: UserSgprLayout::fromKernelMeta: kernel '") +
        meta.name + "' has compute_pgm_rsrc2.USER_SGPR_COUNT=" +
        llvm::Twine(pgmRsrc2UserSgprCount) +
        " but kernel_code_properties + kernarg_preload imply " +
        llvm::Twine(layout.userSgprCount) +
        ". KD is inconsistent — refusing to guess the layout.");
  }

  // Workgroup ID SGPRs sit immediately above the user-SGPR region.
  if (meta.computePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X)
    layout.workgroupIdXSgpr =
        appendSource(layout.entries, Source::WorkgroupIdX, 1);
  if (meta.computePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y)
    layout.workgroupIdYSgpr =
        appendSource(layout.entries, Source::WorkgroupIdY, 1);
  if (meta.computePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z)
    layout.workgroupIdZSgpr =
        appendSource(layout.entries, Source::WorkgroupIdZ, 1);
  if (meta.computePgmRsrc2 & COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO)
    layout.workgroupInfoSgpr =
        appendSource(layout.entries, Source::WorkgroupInfo, 1);

  return layout;
}

std::string UserSgprLayout::toString() const {
  std::ostringstream os;
  os << "user_sgpr_count=" << static_cast<int>(userSgprCount);
  for (size_t i = 0; i < entries.size(); ++i) {
    const auto &e = entries[i];
    os << " s[" << i << "]=" << sourceName(e.source);
    if (e.source == Source::PreloadedKernarg)
      os << "(off=" << e.kernargByteOffset << ")";
    else if (e.subDword > 0)
      os << "[" << static_cast<int>(e.subDword) << "]";
  }
  return os.str();
}

} // namespace transpiler
