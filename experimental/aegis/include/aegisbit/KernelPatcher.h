//===-- aegisbit/KernelPatcher.h - Kernel Patching Orchestrator --*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Kernel patching orchestrator that coordinates the instrumentation pipeline.
/// Caches patched kernels for reuse across multiple dispatches.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_KERNEL_PATCHER_H
#define AEGISBIT_KERNEL_PATCHER_H

#include "aegisbit/CoalescingAnalyzer.h"
#include "aegisbit/DispatchInterceptor.h"
#include "aegisbit/InstrumentationPlan.h"
#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aegisbit {

class Disassembler;
class KernelLauncher;

/// Information about a patched kernel
struct PatchedKernel {
  /// Patched ELF bytes (instrumented code object)
  std::vector<uint8_t> PatchedELF;

  /// HSA kernel object handle for dispatching
  uint64_t KernelObject = 0;

  /// Original kernel argument size
  uint32_t OriginalKernargSize = 0;

  /// Original kernel name
  std::string KernelName;

  /// GPU architecture this kernel was compiled for
  std::string GPUArch;

  /// Number of additional VGPRs reserved for instrumentation
  uint32_t AdditionalVGPRs = 0;

  /// Number of additional SGPRs reserved for instrumentation
  uint32_t AdditionalSGPRs = 0;

  /// Instrumentation mode used
  InstrumentationMode Mode = InstrumentationMode::MEMORY_ONLY;

  /// Number of basic blocks in the kernel
  uint32_t NumBasicBlocks = 0;

  /// Number of instructions in the kernel
  uint32_t NumInstructions = 0;

  /// Trace buffer configuration (set when using buildInstrumented).
  /// Buffer addresses are baked into the trampoline as immediates.
  std::optional<TraceConfig> Trace;

  /// Typed summary of the chosen instrumentation configuration.
  std::optional<InstrumentationPlan> Plan;

  /// Number of instrumented memory sites
  uint32_t NumMemorySites = 0;

  /// Additional scratch bytes required by instrumentation (for VGPR spill to scratch)
  uint32_t AdditionalScratchBytes = 0;

  /// Per-site instruction metadata for coalescing reports
  std::vector<SiteInfo> SiteMap;
};

/// Cache key for patched kernels.
///
/// `VariantID` selects among the K complementary patched ELFs built by
/// `KernelPatcher::getOrPatchVariants` when instrumentation replay is
/// enabled.  Defaulted to 0 so existing brace-initialization call sites
/// continue to compile and mean "single variant" (unchanged behavior).
struct PatchCacheKey {
  uint64_t CodeObjectId;    ///< Code object identifier
  uint64_t KernelId;        ///< Kernel identifier within code object
  InstrumentationMode Mode; ///< Instrumentation mode
  uint32_t VariantID = 0;   ///< Replay variant index (0 = single variant)

  bool operator==(const PatchCacheKey& Other) const {
    return CodeObjectId == Other.CodeObjectId &&
           KernelId == Other.KernelId &&
           Mode == Other.Mode &&
           VariantID == Other.VariantID;
  }
};

} // namespace aegisbit

// Hash function for cache key
namespace std {
template<>
struct hash<aegisbit::PatchCacheKey> {
  size_t operator()(const aegisbit::PatchCacheKey& K) const {
    size_t H = hash<uint64_t>{}(K.CodeObjectId);
    H ^= hash<uint64_t>{}(K.KernelId) << 1;
    H ^= hash<int>{}(static_cast<int>(K.Mode)) << 2;
    H ^= hash<uint32_t>{}(K.VariantID) << 3;
    return H;
  }
};
} // namespace std

namespace aegisbit {

/// Kernel patching orchestrator.
///
/// Coordinates the full instrumentation pipeline:
/// 1. Parse code object (CodeObjectHandler)
/// 2. Build CFG (CFGBuilder)
/// 3. Identify memory instruction sites and allocate above-the-count registers
/// 4. Generate trampolines with profiling payloads (TrampolineBridge)
/// 5. Update descriptors and build output (CodeObjectHandler)
/// 6. Load patched kernel (KernelLauncher)
///
/// Patched kernels are cached for reuse.
///
/// Usage:
/// \code
///   auto Patcher = KernelPatcher::create("gfx942");
///   auto* Patched = Patcher->getOrPatch(CodeObj, Symbol, Mode);
///   // Use Patched->KernelObject for dispatch
/// \endcode
class KernelPatcher {
public:
  /// Create a kernel patcher for the specified GPU architecture.
  /// \param GPUArch GPU architecture string (e.g., "gfx942")
  /// \return KernelPatcher instance or error
  static llvm::Expected<std::unique_ptr<KernelPatcher>>
  create(llvm::StringRef GPUArch);

  /// Destructor
  ~KernelPatcher();

  /// Get or create a patched version of a kernel.
  ///
  /// If the kernel has already been patched with the same mode, returns
  /// the cached version. Otherwise, runs the full patching pipeline.
  ///
  /// \param CodeObj Captured code object containing the kernel
  /// \param Symbol Captured kernel symbol information
  /// \param Mode Instrumentation mode
  /// \param Trace If non-null, use buildInstrumented with per-lane address
  ///              capture. Buffer addresses are baked into the trampoline.
  /// \return Pointer to patched kernel info (owned by patcher) or error
  llvm::Expected<const PatchedKernel*>
  getOrPatch(const CapturedCodeObject& CodeObj,
             const CapturedKernelSymbol& Symbol,
             InstrumentationMode Mode,
             const TraceConfig* Trace = nullptr);

  /// Callback supplied by the caller of `getOrPatchVariants`: returns a
  /// pointer to the `TraceConfig` to use for variant `V`, or `nullptr` to
  /// tell the patcher to stop building variants (e.g. buffer OOM).
  ///
  /// The patcher invokes this serially, in ascending `V` order, *only* for
  /// variants it actually intends to build — this is the mechanism by
  /// which lazy allocation works, so the engine can avoid reserving a
  /// 64 MB persistent trace buffer per variant up-front.  The returned
  /// pointer must remain valid until `getOrPatchVariants` returns.
  using TraceConfigProvider =
      llvm::unique_function<const TraceConfig *(uint32_t VariantID)>;

  /// Get or build a bounded set of **complementary** patched kernels.
  ///
  /// For kernels whose site count exceeds what one island layout can
  /// trampoline within ±128 KB (e.g. FLASH_ATTN_EXT: 447/824), this builds
  /// up to `MaxVariants` patched ELFs.  Variant 0 instruments whichever
  /// sites it can reach; variant N+1 gets an `ExcludedPCs` set covering
  /// the union of prior variants and lays out its island range for the
  /// remaining sites.  The engine later rotates variants per dispatch;
  /// the sink merges their samples back by original PC.
  ///
  /// Ordering invariant (caller's responsibility): each `TraceConfig*`
  /// returned by `ProvideTrace` must describe a **distinct** persistent
  /// trace buffer — the trampoline bakes `BufferAddr`/`CounterAddr` as
  /// 64-bit immediates, so sharing a buffer across variants would collide
  /// writes.
  ///
  /// \param MaxVariants Upper bound on the number of variants to build.
  ///        Plateau-termination (see `DebugFlags::ReplayAuto`) may stop
  ///        the loop sooner; the provider returning `nullptr` also stops
  ///        it.  Must be >= 1.
  /// \param ProvideTrace Callback invoked per variant to obtain its
  ///        `TraceConfig*`.  Returning `nullptr` ends the loop (the
  ///        patcher returns whatever variants were already built, minus
  ///        any error from variant 0).
  /// \return Pointers to cached `PatchedKernel`s in variant order.  The
  ///         first variant is always present on success; later variants
  ///         may be absent if plateau triggered, the provider returned
  ///         `nullptr`, or a downstream patching error occurred.  Pointers
  ///         are owned by this `KernelPatcher` and stable until
  ///         `clearCache()`.
  llvm::Expected<llvm::SmallVector<const PatchedKernel*, 4>>
  getOrPatchVariants(const CapturedCodeObject& CodeObj,
                     const CapturedKernelSymbol& Symbol,
                     InstrumentationMode Mode,
                     uint32_t MaxVariants,
                     TraceConfigProvider ProvideTrace);

  /// Clear the cache of patched kernels.
  void clearCache();

  /// Get cache statistics.
  struct CacheStats {
    size_t CacheHits = 0;
    size_t CacheMisses = 0;
    size_t TotalPatched = 0;
    size_t TotalPatchErrors = 0;
  };
  CacheStats getCacheStats() const;

  /// Get the GPU architecture.
  llvm::StringRef getGPUArch() const { return GPUArch; }

private:
  KernelPatcher() = default;

  /// Run the patching pipeline for a kernel.
  ///
  /// \param ExcludedPCs If non-null, any instrumentation site whose
  ///        original PC is in this set is dropped at the `findSites` stage
  ///        before trampoline layout.  Used by `getOrPatchVariants` to
  ///        build complementary patched ELFs — variant N excludes all PCs
  ///        already covered by variants 0..N-1 so the island allocator
  ///        spends its ±128 KB range on fresh sites.  `nullptr` == no
  ///        exclusion (legacy single-variant behavior).
  llvm::Expected<PatchedKernel>
  patchKernel(const CapturedCodeObject& CodeObj,
              const CapturedKernelSymbol& Symbol,
              InstrumentationMode Mode,
              const TraceConfig* Trace = nullptr,
              const std::unordered_set<uint64_t>* ExcludedPCs = nullptr);

  std::string GPUArch;
  std::unique_ptr<Disassembler> Disasm;

  // Cache for patched kernels.
  //
  // Value is a `vector<unique_ptr<PatchedKernel>>` so we can hand out
  // stable `const PatchedKernel*` pointers even as later variants are
  // appended (a `vector<PatchedKernel>` would invalidate references on
  // growth).  For the common single-variant case the vector holds exactly
  // one entry and behavior is identical to the legacy map.
  std::unordered_map<PatchCacheKey, std::vector<std::unique_ptr<PatchedKernel>>>
      Cache;
  mutable std::mutex CacheMutex;

  // Statistics
  mutable CacheStats Stats;

  // Non-copyable
  KernelPatcher(const KernelPatcher&) = delete;
  KernelPatcher& operator=(const KernelPatcher&) = delete;
};

} // namespace aegisbit

#endif // AEGISBIT_KERNEL_PATCHER_H
