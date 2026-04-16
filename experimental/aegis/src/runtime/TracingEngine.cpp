//===-- TracingEngine.cpp - Tracing Engine ----------------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the main tracing engine.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/TracingEngine.h"
#include "aegisbit/CoalescingAnalyzer.h"
#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/DispatchInterceptor.h"
#include "aegisbit/KernelLauncher.h"
#include "aegisbit/KernelPatcher.h"
#include "aegisbit/RuntimeConfig.h"
#include "aegisbit/Types.h"
#include "llvm/Support/Error.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime.h>
#endif

#include <cstring>
#include <filesystem>
#include <sstream>
#include <thread>

using namespace llvm;

namespace aegisbit {

TracingEngine& TracingEngine::getInstance() {
  static TracingEngine Instance;
  return Instance;
}

TracingEngine::~TracingEngine() {
  // Ensure threads are properly joined on destruction.
  // This handles the case where finalize() wasn't called before process exit.
  // During static destruction (__cxa_finalize), HSA/ROCm may already be torn
  // down, so thread operations can throw. Catch everything to avoid
  // "terminate called without an active exception" during process exit.
  try {
    if (SignalMon)
      SignalMon->stop();
    Kernargs.joinInit();
  } catch (...) {
    // Swallow exceptions during static destruction — process is exiting anyway
  }
}

Error TracingEngine::initialize() {
  // Use mutex to make initialization thread-safe
  std::lock_guard<std::mutex> Lock(InitMutex);

  if (Initialized) {
    return Error::success();
  }

  RuntimeConfig& Cfg = RuntimeConfig::getInstance();

  if (!Cfg.Enabled) {
    Cfg.log("Tracing disabled by configuration");
    return Error::success();
  }

  // Note: GPU architecture detection and patcher creation are deferred
  // to the first dispatch. Calling HIP functions during rocprofiler's
  // toolInit callback can cause deadlocks due to recursive initialization.
  // The GPUArch will be detected from the code object's ELF metadata when
  // we first need to patch a kernel.
  GPUArch = "";  // Will be populated on first code object load
  Cfg.log("Tracing engine: patcher creation deferred to first dispatch");

  // Spin up the completion-signal monitor. Construct lazily so `this` is
  // guaranteed fully alive when the callbacks capture it.
  if (!SignalMon) {
    SignalMon = std::make_unique<SignalMonitor>(
        Dispatches,
        [this](uint32_t ID) { onDispatchComplete(ID); },
        [this](uint32_t /*ID*/, ActiveDispatch Dispatch) {
          cleanupDispatch(Dispatch);
          std::lock_guard<std::mutex> Lock(StatsMutex);
          Statistics.TimeoutDispatches++;
        },
        kSignalTimeoutSeconds);
  }
  SignalMon->start();

  Initialized = true;
  Cfg.log("Tracing engine initialized");

  return Error::success();
}

// HSA pool discovery has moved to aegisbit/HSAPoolManager.{h,cpp}.
// Trace buffer allocation/free have moved to TraceBufferAllocator.{h,cpp}.
// Kernarg pool management has moved to KernargPool.{h,cpp}.
// Completion signal monitoring has moved to SignalMonitor.{h,cpp}.

void TracingEngine::cleanupDispatch(ActiveDispatch& Dispatch,
                                    bool DuringFinalize) {
  RuntimeConfig& Cfg = RuntimeConfig::getInstance();

#ifdef AEGISBIT_HAS_GPU
  if (DuringFinalize) {
    // During finalize (triggered by rocprofiler-sdk atexit → toolFini), the HSA
    // runtime may be partially torn down: SharedSignalPool.clear() runs during
    // Runtime::Unload() while IS_OPEN() still passes.  Calling
    // hsa_signal_destroy at this point runs `delete Signal` which returns slots
    // to a freed pool, corrupting glibc heap metadata.  Leak everything — the
    // OS reclaims all memory at process exit.
    Dispatch.GpuKernargPtr = nullptr;
    Dispatch.CompletionSignalHandle = 0;
    Dispatch.OriginalSignalHandle = 0;
    return;
  }

  if (Dispatch.GpuKernargPtr) {
    Kernargs.release(Dispatch.GpuKernargPtr);
    Dispatch.GpuKernargPtr = nullptr;
  }

  // Destroy our completion signal
  if (Dispatch.CompletionSignalHandle != 0) {
    hsa_signal_t Signal = {Dispatch.CompletionSignalHandle};
    hsa_signal_destroy(Signal);
    Dispatch.CompletionSignalHandle = 0;
  }

  // Forward completion to app's original signal.
  // Must use screlease to ensure the kernel's memory writes are globally visible
  // before the app observes this signal. relaxed would allow the app to proceed
  // before GPU writes propagate, causing data races between sequential kernels.
  if (Dispatch.OriginalSignalHandle != 0) {
    hsa_signal_t OrigSignal = {Dispatch.OriginalSignalHandle};
    hsa_signal_store_relaxed(OrigSignal, 0);
    Cfg.log("Forwarded completion to app signal for dispatch " +
            std::to_string(Dispatch.DispatchID));
  }
#endif
}

void TracingEngine::finalize() {
  if (!Initialized) {
    return;
  }

  RuntimeConfig& Cfg = RuntimeConfig::getInstance();

  // Join background pool initialization thread FIRST (finalize ordering:
  // pool-init must complete before we stop the signal monitor so that no
  // HSA interactions happen after teardown begins).
  Kernargs.joinInit();

  // Stop signal monitor thread. `stop()` is idempotent and joins internally.
  if (SignalMon) {
    SignalMon->stop();
    Cfg.log("Signal monitor thread stopped");
  }

  // Clean up any pending dispatches.  Pass DuringFinalize=true so that
  // cleanupDispatch skips all HSA API calls — the HSA runtime may be
  // partially torn down at this point (see test/zerosGPR_crash_repro.md).
  for (auto &[ID, Dispatch] : Dispatches.drainAll()) {
    (void)ID;
    cleanupDispatch(Dispatch, /*DuringFinalize=*/true);
  }

  // Log statistics
  {
    std::lock_guard<std::mutex> Lock(StatsMutex);
    std::ostringstream SS;
    SS << "Tracing statistics:\n"
       << "  Total dispatches: " << Statistics.TotalDispatches << "\n"
       << "  Traced: " << Statistics.TracedDispatches << "\n"
       << "  Skipped: " << Statistics.SkippedDispatches << "\n"
       << "  Errors: " << Statistics.ErrorDispatches << "\n"
       << "  Timeouts: " << Statistics.TimeoutDispatches << "\n"
       << "  Overflows: " << Statistics.OverflowDispatches << "\n"
       << "  Traces written: " << Statistics.TracesWritten << "\n"
       << "  Total trace bytes: " << Statistics.TotalTraceBytes;
    Cfg.log(SS.str());
  }

  KernelCache.clear();

  // During __cxa_finalize (triggered by rocprofiler-sdk teardown), HSA
  // internals may be partially destroyed.  Calling hsa_amd_memory_pool_free()
  // at this point corrupts the heap.  Leak GPU allocations instead — the OS
  // reclaims device memory at process exit.
#ifdef AEGISBIT_HAS_GPU
  // Intentional leak: HSA memory pool free during __cxa_finalize corrupts
  // glibc heap. OS reclaims GPU memory at process exit.
  PersistBuffers.clear();
  Kernargs.clear();
  Cfg.log("Skipped GPU memory free (unsafe during __cxa_finalize)");
#endif

  // Write accumulated results as JSON if output path is configured.
  // Sink is responsible for its own locking and clears itself on flush.
  if (!Cfg.JSONOutputPath.empty()) {
    Results.flush(Cfg.JSONOutputPath);
    Cfg.log("JSON output written to " + Cfg.JSONOutputPath);
  }

  // Intentionally leak Patcher (LLVM MC objects) and Launcher (HSA
  // executables).  During __cxa_finalize (triggered by rocprofiler-sdk
  // teardown), both LLVM statics and HSA internals may already be in a
  // partially-destroyed state.  Destroying them triggers SIGSEGV or heap
  // corruption.  At process exit the OS reclaims all memory anyway.
  (void)Launcher.release();
  (void)Patcher.release();
  Initialized = false;

  Cfg.log("Tracing engine finalized");
}

bool TracingEngine::isInitialized() const {
  return Initialized;
}

PersistentTraceBuffer&
TracingEngine::getOrAllocPersistentBuffer(const std::string& KernelName,
                                          uint32_t VariantID) {
  return PersistBuffers.getOrAlloc(
      KernelName, VariantID,
      [this](const std::string &Name) -> PersistentTraceBuffer {
        PersistentTraceBuffer PB;

        const PayloadStrategy ChosenStrategy =
            (RuntimeConfig::getInstance().Transform.StrategyOverride ==
             "full_capture")
                ? PayloadStrategy::FullCapture
                : PayloadStrategy::OnGpuReduce;

        PB.Config.Strategy = ChosenStrategy;

        size_t AllocSize;
        if (ChosenStrategy == PayloadStrategy::FullCapture) {
          constexpr size_t MaxRecords = 4'000'000;
          AllocSize = MaxRecords * TraceConfig::RecordSize;
        } else {
          // OnGpuReduce: per-site {total_cache_lines u32, total_samples u32}
          constexpr size_t MaxSites = 4096;
          AllocSize = MaxSites * TraceConfig::CounterSize;
        }
        PB.BufferSize = AllocSize;

        auto [BufPtr, CtrPtr, ActualSize, FineGrained] =
            TraceBuffers.allocate(PB.BufferSize + 64);
        if (BufPtr && CtrPtr) {
          PB.BufferPtr = BufPtr;
          PB.CounterPtr = CtrPtr;
          PB.BufferSize = ActualSize > 64 ? ActualSize - 64 : ActualSize;
          std::memset(PB.BufferPtr, 0, PB.BufferSize);
          std::memset(PB.CounterPtr, 0, 8);

          PB.Config.BufferAddr = reinterpret_cast<uint64_t>(PB.BufferPtr);
          PB.Config.CounterAddr = reinterpret_cast<uint64_t>(PB.CounterPtr);
          PB.Config.BufferSize = PB.BufferSize;
          PB.Config.SupportsGPUAtomics = FineGrained;

          RuntimeConfig::getInstance().log(
              "Allocated persistent trace buffer for " + Name +
              ": buf=" + std::to_string(PB.Config.BufferAddr) +
              " ctr=" + std::to_string(PB.Config.CounterAddr) +
              " size=" + std::to_string(PB.BufferSize) +
              (FineGrained ? " (atomics OK)" : ""));
        } else {
          RuntimeConfig::getInstance().log(
              "WARNING: Failed to allocate persistent buffer for " + Name);
        }

        return PB;
      });
}

bool TracingEngine::onDispatch(hsa_queue_t* /*Queue*/,
                                hsa_kernel_dispatch_packet_t* Packet,
                                uint64_t OriginalKernelObject,
                                void* OriginalKernarg,
                                uint32_t OriginalKernargSize) {
  // Re-entrancy guard: prevent infinite loop when our own patched kernel launches
  // This is the classic DBI bootstrap problem - we intercept dispatches, but we
  // also LAUNCH dispatches (the patched kernels). Without this guard, we'd
  // intercept our own patched kernel, patch it again, launch again, etc.
  static thread_local bool InsideAegisBit = false;
  if (InsideAegisBit) {
    // This is our own dispatch - let it through unmodified
    return true;
  }

  if (!Initialized) {
    return true;  // Pass through
  }

  // Set guard for the duration of this function
  InsideAegisBit = true;

  // Ensure guard is cleared on all exit paths (normal return, exceptions, etc.)
  struct GuardClearer {
    ~GuardClearer() { InsideAegisBit = false; }
  } guard_clearer;

  RuntimeConfig& Cfg = RuntimeConfig::getInstance();

  {
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.TotalDispatches++;
  }

  // Look up kernel symbol
  const CapturedKernelSymbol* Symbol = lookupKernelSymbol(OriginalKernelObject);
  if (!Symbol) {
    Cfg.log("Unknown kernel object, passing through");
    return true;
  }

  // Check if kernel should be traced
  if (!Cfg.shouldTraceKernel(Symbol->KernelName)) {
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.SkippedDispatches++;
    return true;
  }

  // Skip internal HIP/ROCclr kernels - these are used for memory operations
  // and tracing them causes re-entrancy issues (we'd be allocating trace
  // buffers while inside a hipMemcpy which uses these kernels)
  if (Symbol->KernelName.find("__amd_rocclr_") != std::string::npos) {
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.SkippedDispatches++;
    return true;
  }


  // Skip kernels that have previously failed patching (avoid log spam)
  {
    std::lock_guard<std::mutex> Lock(FailedKernelsMutex);
    if (FailedKernels.count(Symbol->KernelName)) {
      std::lock_guard<std::mutex> SLock(StatsMutex);
      Statistics.SkippedDispatches++;
      return true;  // Proceed with original
    }
  }

  Cfg.log("Tracing kernel: " + Symbol->KernelName);

  // Start background HSA pool discovery and kernarg pool initialization.
  // HSA APIs are safe to call from any context (unlike HIP).
  Kernargs.startInitAsync();

  // Discover HSA pools synchronously if not done yet (needed for trace buffer
  // allocation). Safe because HSA APIs don't have the HIP locking issues.
  HSAPools.ensureDiscovered();

  // Get code object
  const CapturedCodeObject* CodeObj =
      DispatchInterceptor::getCodeObject(Symbol->CodeObjectId);
  if (!CodeObj) {
    Cfg.log("Code object not found for kernel: " + Symbol->KernelName);
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.ErrorDispatches++;
    return true;  // Proceed with original
  }

  if (!ensurePatcher(*CodeObj))
    return true; // Patcher creation failed; pass dispatch through unmodified.

  // Replay-aware path: ensureVariants lazily builds up to N patched ELFs
  // (each with its own persistent buffer + loaded HSA kernel) on the first
  // dispatch of this kernel.  With AEGISBIT_REPLAY unset this is exactly
  // one variant and is behaviorally identical to the pre-replay path.
  auto BundlesOrErr = ensureVariants(*CodeObj, *Symbol);
  if (!BundlesOrErr) {
    Cfg.log("Failed to build variants for kernel '" + Symbol->KernelName +
            "': " + toString(BundlesOrErr.takeError()));
    {
      std::lock_guard<std::mutex> Lock(FailedKernelsMutex);
      FailedKernels.insert(Symbol->KernelName);
    }
    Cfg.log("Kernel '" + Symbol->KernelName +
            "' added to failed list (will skip future dispatches)");
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.ErrorDispatches++;
    return true;
  }
  llvm::ArrayRef<VariantBundle> Variants = *BundlesOrErr;
  if (Variants.empty()) {
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.ErrorDispatches++;
    return true;
  }

  // Pick which variant to dispatch this time.  Round-robin per kernel name;
  // collapses to 0 when only one variant is bundled.
  uint32_t Idx =
      Selector.pick(Symbol->KernelName, static_cast<uint32_t>(Variants.size()));
  const VariantBundle &Chosen = Variants[Idx];

  // Reset the chosen variant's counter for this dispatch (per-dispatch zero).
  // The buffer itself is cumulative within each variant across dispatches and
  // the sink aggregates it canonically by PC at completion.
  if (Chosen.Buffer && Chosen.Buffer->CounterPtr)
    std::memset(Chosen.Buffer->CounterPtr, 0, 8);

  uint32_t DispatchID = Dispatches.nextId();

  if (Chosen.Patched && Chosen.Patched->Trace && Chosen.Loaded) {
    return launchProfilerDispatch(Packet, OriginalKernelObject, *Symbol,
                                  *Chosen.Patched, *Chosen.Loaded, Idx,
                                  DispatchID);
  }

  Cfg.log("WARNING: Non-profiler dispatch path reached — mode must be MEMORY_ONLY");
  std::lock_guard<std::mutex> Lock(StatsMutex);
  Statistics.ErrorDispatches++;
  return true;
}

Expected<llvm::ArrayRef<TracingEngine::VariantBundle>>
TracingEngine::ensureVariants(const CapturedCodeObject &CodeObj,
                               const CapturedKernelSymbol &Symbol) {
  RuntimeConfig &Cfg = RuntimeConfig::getInstance();

  // Serialize per-kernel build on BundlesMutex.  Builds are once-per-kernel
  // (cached thereafter), so the coarse lock is acceptable; downstream
  // patcher / loader / buffer caches have their own finer-grained locks.
  std::lock_guard<std::mutex> Lock(BundlesMutex);
  auto &Slot = Bundles[Symbol.KernelName];
  if (!Slot.empty())
    return llvm::ArrayRef<VariantBundle>(Slot);

  // Decide the target variant count.  Off by default (1 variant, legacy
  // behavior).  Non-MEMORY_ONLY dispatches don't use instrumented trampolines,
  // so replay is meaningless there.
  //
  // For auto mode we lift the historical hard cap (was 4) to
  // `Cfg.Debug.ReplayMax` (default 32).  Plateau detection inside
  // `KernelPatcher::getOrPatchVariants` is the real terminator — dense
  // kernels still finish in 2-3 variants — so the cap only matters for
  // pathologically sparse kernels like `flash_attn_tile<64,64,64,1>` which
  // need ~17 variants for full coverage.  Because the buffer allocation
  // below is *lazy* (provider callback), raising the cap no longer wastes
  // GPU memory on kernels that converge quickly.
  uint32_t MaxVariants = 1;
  if (Cfg.Mode == InstrumentationMode::MEMORY_ONLY) {
    if (Cfg.Debug.ReplayVariants > 0)
      MaxVariants = Cfg.Debug.ReplayVariants;
    else if (Cfg.Debug.ReplayAuto)
      MaxVariants = Cfg.Debug.ReplayMax;
  }

  // Buffer-before-patch invariant still holds *per variant* — the trampoline
  // bakes Buffer/Counter addresses as 64-bit immediates, so each variant
  // must own a distinct buffer.  What changed is that we no longer eagerly
  // reserve all N buffers up-front; instead we hand the patcher a provider
  // callback that allocates the next variant's buffer only when the patcher
  // is actually about to build it.  When plateau fires (or MaxVariants is
  // reached) the remaining buffers are never allocated.
  llvm::SmallVector<PersistentTraceBuffer *, 4> PerVariantBuffers;
  PerVariantBuffers.reserve(MaxVariants);
  auto ProvideTrace =
      [this, &Symbol, &Cfg,
       &PerVariantBuffers](uint32_t V) -> const TraceConfig * {
    PersistentTraceBuffer &PB =
        getOrAllocPersistentBuffer(Symbol.KernelName, V);
    if (!PB.BufferPtr) {
      Cfg.log("Variant " + std::to_string(V) +
              ": buffer alloc failed; stopping variant loop");
      return nullptr;
    }
    PerVariantBuffers.push_back(&PB);
    return &PB.Config;
  };

  auto PatchedOrErr = Patcher->getOrPatchVariants(
      CodeObj, Symbol, Cfg.Mode, MaxVariants,
      KernelPatcher::TraceConfigProvider(std::move(ProvideTrace)));
  if (!PatchedOrErr)
    return PatchedOrErr.takeError();

  llvm::SmallVector<VariantBundle, 4> Result;
  Result.reserve(PatchedOrErr->size());
  for (size_t V = 0; V < PatchedOrErr->size(); ++V) {
    const PatchedKernel *Patched = (*PatchedOrErr)[V];
    // Per-variant PatchCacheKey keys the LoadedKernel cache so each variant
    // gets its own HSA executable (different ELF bytes, different kernel
    // object handles).
    PatchCacheKey Key{CodeObj.CodeObjectId, Symbol.KernelId, Cfg.Mode,
                      static_cast<uint32_t>(V)};
    auto LoadedOrErr = getOrLoadKernel(*Patched, Key);
    if (!LoadedOrErr) {
      if (V == 0)
        return LoadedOrErr.takeError();
      Cfg.log("Variant " + std::to_string(V) +
              ": HSA load failed; stopping variant loop (" +
              toString(LoadedOrErr.takeError()) + ")");
      break;
    }
    VariantBundle Bundle;
    Bundle.Patched = Patched;
    Bundle.Buffer = PerVariantBuffers[V];
    Bundle.Loaded = *LoadedOrErr;
    Result.push_back(Bundle);
  }

  Slot = std::move(Result);
  Cfg.log("Kernel '" + Symbol.KernelName + "': bundled " +
          std::to_string(Slot.size()) + " variant(s)");
  return llvm::ArrayRef<VariantBundle>(Slot);
}

bool TracingEngine::ensurePatcher(const CapturedCodeObject &CodeObj) {
  if (Patcher)
    return true;

  RuntimeConfig &Cfg = RuntimeConfig::getInstance();

  // We defer patcher creation until first dispatch to avoid calling HIP
  // functions during rocprofiler toolInit.
  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(CodeObj.Bytes);
  if (HandlerOrErr) {
    std::string DetectedArch = HandlerOrErr->getGPUArch();
    if (!DetectedArch.empty()) {
      GPUArch = DetectedArch;
      Cfg.log("Detected GPU from code object: " + GPUArch);
    }
  } else {
    consumeError(HandlerOrErr.takeError());
  }

  if (GPUArch.empty()) {
    GPUArch = "gfx942";
    Cfg.log("Using default GPU arch: " + GPUArch);
  }

  auto PatcherOrErr = KernelPatcher::create(GPUArch);
  if (!PatcherOrErr) {
    Cfg.log("Failed to create patcher: " + toString(PatcherOrErr.takeError()));
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.ErrorDispatches++;
    return false;
  }
  Patcher = std::move(*PatcherOrErr);
  Cfg.log("Kernel patcher initialized for " + GPUArch);
  return true;
}

bool TracingEngine::launchProfilerDispatch(
    hsa_kernel_dispatch_packet_t *Packet, uint64_t OriginalKernelObject,
    const CapturedKernelSymbol &Symbol, const PatchedKernel &Patched,
    const LoadedKernel &Loaded, uint32_t VariantID, uint32_t DispatchID) {
#ifdef AEGISBIT_HAS_GPU
  RuntimeConfig &Cfg = RuntimeConfig::getInstance();

  hsa_signal_t OurSignal;
  hsa_status_t HsaStatus = hsa_signal_create(1, 0, nullptr, &OurSignal);
  if (HsaStatus != HSA_STATUS_SUCCESS) {
    Cfg.log("Failed to create completion signal");
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.ErrorDispatches++;
    return true;
  }
  uint64_t OriginalSignal = Packet->completion_signal.handle;

  // AEGISBIT_SKIP_SIGNAL diagnostic: use the patched kernel but keep the
  // original signal. This isolates "patched code" from the signal-forwarding
  // mechanism when narrowing down a bug.
  const bool SkipSignal = Cfg.Debug.SkipSignal;

  Packet->kernel_object = Loaded.KernelSymbol;
  if (Patched.AdditionalScratchBytes > 0)
    Packet->private_segment_size += Patched.AdditionalScratchBytes;

  if (SkipSignal) {
    hsa_signal_destroy(OurSignal);
    Cfg.log("SKIP_SIGNAL: using patched kernel but keeping original signal");
    std::lock_guard<std::mutex> Lock(StatsMutex);
    Statistics.TracedDispatches++;
    return true;
  }

  Packet->completion_signal = OurSignal;

  Cfg.log("Profiler dispatch " + std::to_string(DispatchID) + ": " +
          Symbol.KernelName + " [variant " + std::to_string(VariantID) +
          "] → patched kernel (object: " +
          std::to_string(Loaded.KernelSymbol) + ")");

  ActiveDispatch Dispatch;
  Dispatch.DispatchID = DispatchID;
  Dispatch.KernelName = Symbol.KernelName;
  Dispatch.Params.WorkgroupSizeX = Packet->workgroup_size_x;
  Dispatch.Params.WorkgroupSizeY = Packet->workgroup_size_y;
  Dispatch.Params.WorkgroupSizeZ = Packet->workgroup_size_z;
  Dispatch.Params.GridSizeX = Packet->grid_size_x;
  Dispatch.Params.GridSizeY = Packet->grid_size_y;
  Dispatch.Params.GridSizeZ = Packet->grid_size_z;
  Dispatch.StartTime = std::chrono::steady_clock::now();
  Dispatch.OriginalKernelObject = OriginalKernelObject;
  Dispatch.PatchedKernelObject = Loaded.KernelSymbol;
  Dispatch.CompletionSignalHandle = OurSignal.handle;
  Dispatch.OriginalSignalHandle = OriginalSignal;
  Dispatch.SiteMap = Patched.SiteMap;
  Dispatch.VariantID = VariantID;
  Dispatches.insert(DispatchID, std::move(Dispatch));

  std::lock_guard<std::mutex> Lock(StatsMutex);
  Statistics.TracedDispatches++;
  return true;
#else
  (void)Packet;
  (void)OriginalKernelObject;
  (void)Symbol;
  (void)Patched;
  (void)Loaded;
  (void)VariantID;
  (void)DispatchID;
  return true;
#endif
}

void TracingEngine::onDispatchComplete(uint32_t DispatchID) {
  auto Taken = Dispatches.take(DispatchID);
  if (!Taken)
    return; // Unknown ID — already cleaned up by timeout path or finalize.
  ActiveDispatch Dispatch = std::move(*Taken);

  if (!Dispatch.KernelName.empty()) {
    // Persistent buffer pointers are stable for the kernel's lifetime, so it's
    // safe to read through them outside the cache's lock.  Variant-aware
    // lookup routes each dispatch's counters to its own per-variant buffer;
    // the sink re-joins them canonically by original PC (Phase 1b).
    if (const PersistentTraceBuffer *PB =
            PersistBuffers.find(Dispatch.KernelName, Dispatch.VariantID)) {
      Results.ingest(*PB, Dispatch.SiteMap, Dispatch.KernelName,
                     Dispatch.VariantID);
    }
  }

  cleanupDispatch(Dispatch);
}

TracingEngine::Stats TracingEngine::getStats() const {
  std::lock_guard<std::mutex> Lock(StatsMutex);
  return Statistics;
}

const CapturedKernelSymbol*
TracingEngine::lookupKernelSymbol(uint64_t KernelObject) {
  // Search through all captured kernel symbols
  auto AllSymbols = DispatchInterceptor::getAllKernelSymbols();
  for (const auto* Symbol : AllSymbols) {
    if (Symbol->KernelObject == KernelObject) {
      return Symbol;
    }
  }
  return nullptr;
}

Expected<const LoadedKernel*>
TracingEngine::getOrLoadKernel(const PatchedKernel& Patched,
                                const PatchCacheKey& Key) {
  RuntimeConfig& Cfg = RuntimeConfig::getInstance();

  if (const LoadedKernel *Hit = KernelCache.find(Key))
    return Hit;

  // Lazy initialization of KernelLauncher
  if (!Launcher) {
    uint64_t AgentHandle = getDefaultGPUAgent();
    if (AgentHandle == 0) {
      return createStringError(inconvertibleErrorCode(),
                               "No GPU agent available for loading kernels");
    }

    auto LauncherOrErr = KernelLauncher::create(AgentHandle);
    if (!LauncherOrErr) {
      return LauncherOrErr.takeError();
    }
    Launcher = std::move(*LauncherOrErr);
    Cfg.log("KernelLauncher created for GPU: " + Launcher->getGPUName());
  }

  // Strip .kd suffix if present (rocprofiler-sdk appends it)
  std::string CleanName = Patched.KernelName;
  if (CleanName.size() > 3 &&
      CleanName.substr(CleanName.size() - 3) == ".kd") {
    CleanName = CleanName.substr(0, CleanName.size() - 3);
  }

  // Load the patched kernel
  auto LoadedOrErr = Launcher->loadKernel(
      Patched.PatchedELF,
      CleanName,
      Patched.OriginalKernargSize);
  if (!LoadedOrErr) {
    return LoadedOrErr.takeError();
  }

  const LoadedKernel *Cached =
      KernelCache.insert(Key, std::move(*LoadedOrErr));
  Cfg.log("Loaded patched kernel into HSA: " + CleanName +
          " (kernel object: " + std::to_string(Cached->KernelSymbol) + ")");
  return Cached;
}

} // namespace aegisbit
