//===-- aegisbit/TracingEngine.h - Tracing Engine ---------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Main tracing engine that coordinates dispatch interception, kernel
/// patching, and trace collection.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TRACING_ENGINE_H
#define AEGISBIT_TRACING_ENGINE_H

#include "aegisbit/ActiveDispatch.h"
#include "aegisbit/CoalescingAnalyzer.h"
#include "aegisbit/DispatchRegistry.h"
#include "aegisbit/HSAPoolManager.h"
#include "aegisbit/KernargPool.h"
#include "aegisbit/KernelLauncher.h"
#include "aegisbit/KernelPatcher.h"
#include "aegisbit/LoadedKernelCache.h"
#include "aegisbit/PersistentBufferCache.h"
#include "aegisbit/ProfilingResultsSink.h"
#include "aegisbit/SignalMonitor.h"
#include "aegisbit/TraceBufferAllocator.h"
#include "aegisbit/RuntimeConfig.h"
#include "aegisbit/Types.h"
#include "aegisbit/VariantSelector.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

// Forward declarations
struct hsa_queue_s;
typedef struct hsa_queue_s hsa_queue_t;
struct hsa_kernel_dispatch_packet_s;
typedef struct hsa_kernel_dispatch_packet_s hsa_kernel_dispatch_packet_t;

namespace aegisbit {

/// Main tracing engine for AegisBit.
///
/// Coordinates the end-to-end profiling workflow:
/// 1. Receives dispatch notifications from HSAInterceptor
/// 2. Decides whether to trace based on RuntimeConfig
/// 3. Gets or creates patched kernel via KernelPatcher
/// 4. Modifies dispatch packet to use patched kernel
/// 5. Analyzes coalescing data on completion via CoalescingAnalyzer
///
/// The engine is a singleton that is initialized from AegisBit's shared-library
/// lifecycle hooks and finalized during HSA/library shutdown.
///
/// Usage:
/// \code
///   TracingEngine::getInstance().initialize();
///   // Set up HSAInterceptor callback to call onDispatch
///   // ... application runs ...
///   TracingEngine::getInstance().finalize();
/// \endcode
class TracingEngine {
public:
  /// Destructor - ensures proper cleanup of threads
  ~TracingEngine();

  /// Get the singleton instance.
  static TracingEngine& getInstance();

  /// Initialize the tracing engine.
  /// Sets up patchers, trace writers, and output directory.
  /// \return Error on failure
  llvm::Error initialize();

  /// Finalize the tracing engine.
  /// Waits for pending dispatches, writes final traces, and cleans up.
  void finalize();

  /// Check if engine is initialized.
  bool isInitialized() const;

  /// Called by HSAInterceptor for each kernel dispatch.
  ///
  /// This method:
  /// 1. Checks if kernel should be traced
  /// 2. Gets patched kernel from cache or patches it
  /// 3. Allocates trace buffer
  /// 4. Creates extended kernarg with TraceArgs
  /// 5. Modifies packet to use patched kernel
  ///
  /// \param Queue HSA queue (may be null)
  /// \param Packet Dispatch packet (modified in place)
  /// \param OriginalKernelObject Original kernel object handle
  /// \param OriginalKernarg Original kernel arguments
  /// \param OriginalKernargSize Size of original kernel arguments
  /// \return true to proceed with dispatch, false to skip
  bool onDispatch(hsa_queue_t* Queue,
                  hsa_kernel_dispatch_packet_t* Packet,
                  uint64_t OriginalKernelObject,
                  void* OriginalKernarg,
                  uint32_t OriginalKernargSize);

  /// Called when a dispatch completes.
  /// Collects trace data from buffer and writes to file.
  /// \param DispatchID Dispatch identifier
  void onDispatchComplete(uint32_t DispatchID);

  /// Get statistics about tracing.
  struct Stats {
    uint64_t TotalDispatches = 0;      ///< Total dispatches seen
    uint64_t TracedDispatches = 0;     ///< Dispatches that were traced
    uint64_t SkippedDispatches = 0;    ///< Dispatches skipped (not matching filter)
    uint64_t ErrorDispatches = 0;      ///< Dispatches with errors
    uint64_t TimeoutDispatches = 0;    ///< Dispatches that timed out
    uint64_t OverflowDispatches = 0;   ///< Dispatches where trace buffer overflowed
    uint64_t TracesWritten = 0;        ///< Trace files written
    uint64_t TotalTraceBytes = 0;      ///< Total bytes written to traces
  };
  Stats getStats() const;

private:
  TracingEngine() = default;

  /// Look up kernel symbol by kernel object handle
  const CapturedKernelSymbol* lookupKernelSymbol(uint64_t KernelObject);

  /// Load patched kernel or return cached version
  llvm::Expected<const LoadedKernel*>
  getOrLoadKernel(const PatchedKernel& Patched, const PatchCacheKey& Key);

  /// Lazily create the `KernelPatcher` on first dispatch, detecting the GPU
  /// arch from the code object when possible. Returns true if the patcher is
  /// ready on exit; false if patcher creation failed (caller should pass the
  /// dispatch through unmodified and record an error stat).
  bool ensurePatcher(const CapturedCodeObject& CodeObj);

  /// MEMORY_ONLY profiler fast path: load the patched kernel, create our
  /// completion signal, swap the kernel object on `Packet`, and register an
  /// `ActiveDispatch` with the signal monitor. Always returns true — the
  /// caller must proceed with dispatch (either patched or, on error /
  /// `AEGISBIT_SKIP_SIGNAL`, with the original signal forwarded).
  ///
  /// `VariantID` is stashed in `ActiveDispatch` so `onDispatchComplete` can
  /// route the variant-specific buffer + SiteMap to the sink.
  bool launchProfilerDispatch(hsa_kernel_dispatch_packet_t* Packet,
                              uint64_t OriginalKernelObject,
                              const CapturedKernelSymbol& Symbol,
                              const PatchedKernel& Patched,
                              const LoadedKernel& Loaded,
                              uint32_t VariantID,
                              uint32_t DispatchID);

  /// Instrumentation Replay (Phase 4b): a single kernel can have multiple
  /// patched-ELF variants, each with its own persistent trace buffer and
  /// its own loaded HSA kernel.  `VariantBundle` bundles those three
  /// handles for a single variant; `ensureVariants` produces one per
  /// active variant for a given kernel.
  struct VariantBundle {
    const PatchedKernel *Patched = nullptr;
    PersistentTraceBuffer *Buffer = nullptr;
    const LoadedKernel *Loaded = nullptr;
  };

  /// Produce (and cache) the variant bundles for `Symbol`.  On success the
  /// returned `ArrayRef` has at least one entry.  Buffer-before-patch
  /// ordering is enforced inside: every variant's persistent trace buffer
  /// is allocated *before* its ELF is patched, since the trampoline bakes
  /// the buffer/counter addresses as 64-bit immediates.
  ///
  /// With `AEGISBIT_REPLAY` unset (`MaxVariants=1`) this returns a
  /// 1-element view and the flow is byte-for-byte equivalent to the
  /// pre-replay path.
  llvm::Expected<llvm::ArrayRef<VariantBundle>>
  ensureVariants(const CapturedCodeObject &CodeObj,
                 const CapturedKernelSymbol &Symbol);

  /// Cleanup dispatch resources (GPU kernarg, signals).
  /// When DuringFinalize is true, skip all HSA API calls — the runtime may be
  /// partially torn down and calling into it corrupts glibc heap metadata.
  void cleanupDispatch(ActiveDispatch& Dispatch, bool DuringFinalize = false);

  bool Initialized = false;
  std::string GPUArch;
  std::unique_ptr<KernelPatcher> Patcher;

  // Kernel launcher for loading patched kernels
  std::unique_ptr<KernelLauncher> Launcher;

  // Cache for loaded kernels (patched ELF -> loaded executable).
  // Owns its own mutex; safe to `clear()` during shutdown because
  // `LoadedKernel` is a handle-only value with no HSA-releasing destructor.
  LoadedKernelCache KernelCache;

  // In-flight dispatches + monotonic ID counter (thread-safe).
  DispatchRegistry Dispatches;

  // HSA GPU agent + kernarg/fine-grained memory pool discovery.
  // Owns its own once_flag; safe to outlive HSA runtime.
  // MUST be declared before any member that holds a reference to it, so that
  // the implicit ctor-init order (top-to-bottom) initializes it first.
  HSAPoolManager HSAPools;

  // Allocates GPU-visible trace buffers out of the discovered pools.
  TraceBufferAllocator TraceBuffers{HSAPools};

  // Pre-allocated GPU kernarg pool used to avoid HSA allocs in the dispatch
  // callback. Runs a background thread to pre-populate.
  KernargPool Kernargs{HSAPools};

  // Default trace buffer size (256KB)
  static constexpr size_t kDefaultTraceBufferSize = 256 * 1024;

  // Signal monitoring timeout per dispatch (seconds)
  static constexpr int kSignalTimeoutSeconds = 30;

  // Per-kernel error tracking: skip re-patching kernels that already failed
  std::unordered_set<std::string> FailedKernels;
  mutable std::mutex FailedKernelsMutex;

  /// Per-kernel persistent trace buffer cache.
  /// Buffer addresses are baked into the trampoline, so we reuse the same
  /// buffer across dispatches of the same kernel. Counter is reset per
  /// dispatch. Finalize leaks rather than frees (HSA teardown unsafe).
  PersistentBufferCache PersistBuffers;

  /// Allocate or retrieve a persistent trace buffer for `(KernelName,
  /// VariantID)`.  Thin wrapper around `PersistBuffers.getOrAlloc` that
  /// supplies the HSA-backed allocator closure; each variant receives its
  /// own buffer so the trampoline's baked addresses never alias.
  PersistentTraceBuffer& getOrAllocPersistentBuffer(
      const std::string& KernelName, uint32_t VariantID = 0);

  /// Variant bundles per kernel name (Phase 4b).  Populated lazily on the
  /// first dispatch of each kernel and reused thereafter.
  std::unordered_map<std::string,
                     llvm::SmallVector<VariantBundle, 4>> Bundles;
  mutable std::mutex BundlesMutex;

  /// Round-robin variant picker used by `onDispatch` to decide which
  /// bundle to swap into the next dispatch packet.  Stateless per-kernel
  /// atomic counter; no HSA state.
  VariantSelector Selector;

  /// Completion signal monitor. Constructed lazily in `initialize()` once
  /// this object's lifetime is fully established, so the callbacks can
  /// safely capture `this`. Destroyed in `finalize()`.
  std::unique_ptr<SignalMonitor> SignalMon;

  // Mutex to protect initialization (make thread-safe)
  mutable std::mutex InitMutex;

  // Accumulated profiling results for JSON output (thread-safe sink).
  ProfilingResultsSink Results;

  // Statistics
  mutable Stats Statistics;
  mutable std::mutex StatsMutex;

  // Non-copyable
  TracingEngine(const TracingEngine&) = delete;
  TracingEngine& operator=(const TracingEngine&) = delete;
};

} // namespace aegisbit

#endif // AEGISBIT_TRACING_ENGINE_H
