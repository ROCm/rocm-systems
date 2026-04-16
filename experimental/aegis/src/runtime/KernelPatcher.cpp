//===-- KernelPatcher.cpp - Kernel Patching Orchestrator --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of the kernel patching orchestrator.
///
/// Pipeline: CFG -> findMemorySites -> above-the-count registers ->
///           TrampolineBridge -> in-place ELF patching.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/KernelPatcher.h"
#include "aegisbit/CFGBuilder.h"
#include "aegisbit/CoalescingAnalyzer.h"
#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/DescriptorUpdater.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/InstrumentationPlan.h"
#include "aegisbit/JumpHeuristics.h"
#include "aegisbit/RegisterHelper.h"
#include "aegisbit/RuntimeConfig.h"
#include "aegisbit/ScratchRegisters.h"
#include "aegisbit/SourceMapper.h"
#include "aegisbit/TrampolineBridge.h"
#include "aegisbit/Types.h"
#include "aegisbit/VariantCoverageAccumulator.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <vector>

using namespace llvm;

namespace aegisbit {

//===----------------------------------------------------------------------===//
// PatchContext and pipeline stage helpers
//
// `patchKernel` is orchestration only. Each stage below has a single
// responsibility and communicates with the next via PatchContext.
//===----------------------------------------------------------------------===//

namespace {

/// State carried between patching stages. Initialised incrementally by
/// the stage functions below and finally consumed by assemblePatchedText /
/// applyDescriptorAndRebuildELF / buildSiteMap.
struct PatchContext {
  // Inputs
  const CapturedCodeObject *CodeObj = nullptr;
  const CapturedKernelSymbol *Symbol = nullptr;
  InstrumentationMode Mode = InstrumentationMode::MEMORY_ONLY;
  const TraceConfig *Trace = nullptr;
  /// Optional set of original PCs that should be skipped in `findSites`.
  /// Non-null only when `getOrPatchVariants` is building a complementary
  /// variant and passes the union of prior variants' covered PCs here.
  const std::unordered_set<uint64_t> *ExcludedPCs = nullptr;

  // Loaded code object
  std::optional<CodeObjectHandler> Handler;
  const KernelInfo *KInfo = nullptr;
  ArrayRef<uint8_t> KCode;
  uint64_t KStart = 0;
  uint64_t KEnd = 0;

  // CFG
  std::unique_ptr<ControlFlowGraph> CFG;

  // Plan + scratch
  InstrumentationPlan Plan;
  ScratchRegisters Scratch;

  // Sites
  std::vector<InstrumentationSite> Sites;

  // Layout adjustments applied during assembly
  uint64_t PadSize = 0;
  uint64_t AlignedPrologue = 0;
};

/// Step 0 — optional input-ELF dump for offline reproduction.
void dumpInputCodeObjectIfRequested(const RuntimeConfig &Cfg,
                                    const CapturedCodeObject &CodeObj,
                                    const CapturedKernelSymbol &Symbol) {
  if (Cfg.Debug.DumpInputELFPrefix.empty())
    return;

  const std::string &Prefix = Cfg.Debug.DumpInputELFPrefix;
  std::string Path = Prefix + "_" + Symbol.KernelName + ".bin";
  std::error_code DumpEC;
  llvm::raw_fd_ostream DumpOS(Path, DumpEC);
  if (!DumpEC)
    DumpOS.write(reinterpret_cast<const char *>(CodeObj.Bytes.data()),
                 CodeObj.Bytes.size());

  std::string MetaPath = Prefix + "_" + Symbol.KernelName + ".meta";
  std::error_code MetaEC;
  llvm::raw_fd_ostream MetaOS(MetaPath, MetaEC);
  if (!MetaEC)
    MetaOS << "KernelName=" << Symbol.KernelName << "\n"
           << "KernelId=" << Symbol.KernelId << "\n"
           << "CodeObjectId=" << Symbol.CodeObjectId << "\n"
           << "KernargSegmentSize=" << Symbol.KernargSegmentSize << "\n"
           << "GroupSegmentSize=" << Symbol.GroupSegmentSize << "\n"
           << "PrivateSegmentSize=" << Symbol.PrivateSegmentSize << "\n"
           << "SGPRCount=" << Symbol.SGPRCount << "\n"
           << "VGPRCount=" << Symbol.VGPRCount << "\n";
}

/// Step 1 — parse ELF, extract target kernel, slice its bytes.
Error loadCodeObjectAndExtractKernel(PatchContext &PC) {
  if (PC.CodeObj->Bytes.empty())
    return createStringError(inconvertibleErrorCode(),
                             "Code object has no bytes");

  auto HandlerOrErr = CodeObjectHandler::loadFromBytes(PC.CodeObj->Bytes);
  if (!HandlerOrErr)
    return HandlerOrErr.takeError();
  PC.Handler.emplace(std::move(*HandlerOrErr));

  PC.KInfo = PC.Handler->getKernel(PC.Symbol->KernelName);
  if (!PC.KInfo)
    return createStringError(inconvertibleErrorCode(),
                             "Kernel not found: " + PC.Symbol->KernelName);

  ArrayRef<uint8_t> TextSection = PC.Handler->getTextSection();
  if (TextSection.empty())
    return createStringError(inconvertibleErrorCode(), "Empty .text section");

  PC.KStart = PC.KInfo->CodeOffset;
  PC.KEnd = std::min(PC.KStart + PC.KInfo->CodeSize,
                     static_cast<uint64_t>(TextSection.size()));
  PC.KCode = TextSection.slice(PC.KStart, PC.KEnd - PC.KStart);
  if (PC.KCode.empty())
    return createStringError(inconvertibleErrorCode(), "Empty kernel code");

  return Error::success();
}

/// Step 2 — disassemble kernel into a CFG.
Error buildCFG(Disassembler &Disasm, PatchContext &PC) {
  CFGBuilder CfgB(Disasm);
  auto CFGOrErr = CfgB.build(PC.KCode, PC.KStart);
  if (!CFGOrErr)
    return CFGOrErr.takeError();
  PC.CFG = std::make_unique<ControlFlowGraph>(std::move(*CFGOrErr));
  return Error::success();
}

/// Step 3a — derive the instrumentation plan.
Error planInstrumentation(PatchContext &PC) {
  InstrumentationPlanningFacts PlanningFacts;
  PlanningFacts.Instrumented = (PC.Trace != nullptr);
  if (PC.Trace) {
    auto InstrumentedScratch =
        ScratchRegisters::fromDescriptorInstrumented(PC.KInfo->Descriptor);
    uint32_t NewSGPRCount =
        PC.KInfo->Descriptor.SGPRCount + InstrumentedScratch.ExtraSGPRs;
    uint32_t Rounded = DescriptorUpdater::roundUpSGPR(NewSGPRCount);
    PlanningFacts.SGPROverflow = (Rounded > DescriptorUpdater::MAX_SGPRS);
    PlanningFacts.SupportsGPUAtomics = PC.Trace->SupportsGPUAtomics;
    PlanningFacts.RequestedPayload = PC.Trace->Strategy;
  }

  auto PlanOrErr = buildInstrumentationPlan(PlanningFacts);
  if (!PlanOrErr)
    return PlanOrErr.takeError();
  PC.Plan = *PlanOrErr;
  return Error::success();
}

/// Step 3b — choose scratch-register allocation for this plan.
/// Encapsulates the three fallback paths (uninstrumented / zero-SGPR /
/// instrumented) plus the AccVGPR refinement / spill fallback.
void allocateScratch(const RuntimeConfig &Cfg, Disassembler &Disasm,
                     PatchContext &PC) {
  if (!PC.Plan.Instrumented) {
    PC.Scratch = ScratchRegisters::fromDescriptor(PC.KInfo->Descriptor);
    return;
  }

  if (PC.Plan.Register == RegisterMode::ZeroSGPR) {
    auto InstrumentedScratch =
        ScratchRegisters::fromDescriptorInstrumented(PC.KInfo->Descriptor);
    uint32_t NewSGPRCount =
        PC.KInfo->Descriptor.SGPRCount + InstrumentedScratch.ExtraSGPRs;
    uint32_t Rounded = DescriptorUpdater::roundUpSGPR(NewSGPRCount);
    Cfg.log("SGPR overflow (" + std::to_string(Rounded) + " > " +
            std::to_string(DescriptorUpdater::MAX_SGPRS) +
            "); falling back to zero-SGPR (VCC-temp) trampoline");
    PC.Scratch =
        ScratchRegisters::fromDescriptorZeroSGPR(PC.KInfo->Descriptor);
  } else {
    PC.Scratch =
        ScratchRegisters::fromDescriptorInstrumented(PC.KInfo->Descriptor);
  }

  // AccVGPR path: try to reclaim unused regular VGPRs; if none, spill
  // victim VGPRs to scratch memory (default) or AccVGPR slots (legacy).
  if (PC.Scratch.HasAccumVGPRs && !PC.Scratch.isValid()) {
    uint32_t AO = PC.KInfo->Descriptor.AccumOffset;
    uint32_t CurrentScratch = PC.KInfo->Descriptor.PrivateSegmentFixedSize;
    if (PC.Scratch.refineScratchVGPRs(*PC.CFG, Disasm, AO)) {
      Cfg.log("AccVGPR scratch: found unused v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.ScratchVGPR)) +
              ", v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.LaneVGPR)) +
              ", v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.TempVGPR)) +
              " (scanned " + std::to_string(AO) + " regular VGPRs)");
      return;
    }

    const bool UseAccVGPRSpill = Cfg.Transform.AccVGPRSpill;
    uint32_t VC = PC.KInfo->Descriptor.VGPRCount;
    if (UseAccVGPRSpill) {
      PC.Scratch.setupAccVGPRSpill(AO, VC);
      Cfg.log("AccVGPR spill (legacy): all " + std::to_string(AO) +
              " regular VGPRs in use; spilling v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.ScratchVGPR)) +
              "/v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.LaneVGPR)) +
              "/v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.TempVGPR)) +
              " to a" +
              std::to_string(
                  RegisterHelper::getAGPRIndex(PC.Scratch.SpillAGPR0)) +
              "/a" +
              std::to_string(
                  RegisterHelper::getAGPRIndex(PC.Scratch.SpillAGPR1)) +
              "/a" +
              std::to_string(
                  RegisterHelper::getAGPRIndex(PC.Scratch.SpillAGPR2)));
    } else {
      PC.Scratch.setupScratchSpill(AO, CurrentScratch);
      Cfg.log("Scratch spill: all " + std::to_string(AO) +
              " regular VGPRs in use; spilling v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.ScratchVGPR)) +
              "/v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.LaneVGPR)) +
              "/v" +
              std::to_string(
                  RegisterHelper::getVGPRIndex(PC.Scratch.TempVGPR)) +
              " to scratch offset " +
              std::to_string(PC.Scratch.ScratchSpillOffset) + " (+" +
              std::to_string(PC.Scratch.ExtraScratchBytes) + " bytes)");
    }
  }
}

/// Step 4 — find memory sites, sort them, apply debug caps.
Error findAndSortSites(const RuntimeConfig &Cfg, Disassembler &Disasm,
                       PatchContext &PC) {
  bool AtomicsOK = PC.Trace && PC.Trace->SupportsGPUAtomics;
  PC.Sites = TrampolineBridge::findMemorySites(*PC.CFG, PC.KStart, Disasm,
                                               PC.Scratch, AtomicsOK);

  // Instrumentation-replay exclusion: drop any site whose original PC is in
  // `ExcludedPCs` before downstream layout (sort, SwapPC upgrade, bridge).
  // This is how complementary variants avoid reinstrumenting PCs already
  // covered by an earlier variant and buy back island-range budget.
  if (PC.ExcludedPCs && !PC.ExcludedPCs->empty()) {
    size_t Before = PC.Sites.size();
    PC.Sites.erase(
        std::remove_if(
            PC.Sites.begin(), PC.Sites.end(),
            [&](const InstrumentationSite &S) {
              return PC.ExcludedPCs->count(S.Address) > 0;
            }),
        PC.Sites.end());
    size_t Dropped = Before - PC.Sites.size();
    if (Dropped > 0) {
      Cfg.log("ExcludedPCs: dropped " + std::to_string(Dropped) +
              " sites before bridge (variant building)");
    }
  }

  if (PC.Scratch.NeedsAccVGPRSpill)
    TrampolineBridge::computePreSpillDrainValues(*PC.CFG, PC.Sites,
                                                  PC.Scratch, Disasm);

  std::sort(PC.Sites.begin(), PC.Sites.end(),
            [](const InstrumentationSite &A, const InstrumentationSite &B) {
              return A.Offset < B.Offset;
            });

  uint32_t VMEMCount = 0, LDSCount = 0;
  for (const auto &S : PC.Sites) {
    if (S.IsGlobal) VMEMCount++; else LDSCount++;
  }
  Cfg.log("CFG: " + std::to_string(PC.CFG->BasicBlocks.size()) + " BBs, " +
          std::to_string(PC.KCode.size()) + " bytes, " +
          std::to_string(VMEMCount) + " VMEM + " +
          std::to_string(LDSCount) + " LDS memory ops");

  if (PC.Sites.empty())
    return createStringError(inconvertibleErrorCode(),
                             "No memory ops to instrument in " +
                                 PC.Symbol->KernelName);

  if (Cfg.Debug.MaxSites > 0 && PC.Sites.size() > Cfg.Debug.MaxSites)
    PC.Sites.resize(Cfg.Debug.MaxSites);

  return Error::success();
}

/// Step 5 — promote to SwapPC shared-body mode for large kernels.
void maybeUpgradeToSwapPC(const RuntimeConfig &Cfg, PatchContext &PC) {
  if (!PC.Trace)
    return;

  const uint64_t TextSize = PC.Handler->getTextSection().size();
  if (!shouldUseSwapPCSharedBody(PC.Plan, TextSize, PC.KStart,
                                 PC.Sites.size()))
    return;

  auto SwapScratch =
      ScratchRegisters::fromDescriptorSwapPC(PC.KInfo->Descriptor);
  uint32_t NewSGPRCount =
      PC.KInfo->Descriptor.SGPRCount + SwapScratch.ExtraSGPRs;
  uint32_t Rounded = DescriptorUpdater::roundUpSGPR(NewSGPRCount);
  if (Rounded > DescriptorUpdater::MAX_SGPRS) {
    Cfg.log("SwapPC shared-body requested by range heuristic but SGPR "
            "allocation would overflow; keeping standard shared-body");
    return;
  }

  Cfg.log("Kernel too large for s_call_b64 shared-body range; "
          "upgrading to SwapPC shared-body mode");
  SwapScratch.ScratchVGPR = PC.Scratch.ScratchVGPR;
  SwapScratch.LaneVGPR = PC.Scratch.LaneVGPR;
  SwapScratch.TempVGPR = PC.Scratch.TempVGPR;
  SwapScratch.ExtraVGPRs = PC.Scratch.ExtraVGPRs;
  SwapScratch.HasAccumVGPRs = PC.Scratch.HasAccumVGPRs;
  SwapScratch.NeedsScratchSpill = PC.Scratch.NeedsScratchSpill;
  SwapScratch.NeedsAccVGPRSpill = PC.Scratch.NeedsAccVGPRSpill;
  SwapScratch.ScratchSpillOffset = PC.Scratch.ScratchSpillOffset;
  SwapScratch.ExtraScratchBytes = PC.Scratch.ExtraScratchBytes;
  SwapScratch.FirstFreeVGPRIdx = PC.Scratch.FirstFreeVGPRIdx;
  PC.Plan.Jump = JumpStrategy::SwapPCSharedBody;
  PC.Scratch = SwapScratch;
}

/// Step 6 — shift the kernel forward in .text so relay stubs have
/// ±128 KB space behind large zero-SGPR kernels placed near offset 0.
void applyPreKernelPadding(const RuntimeConfig &Cfg, PatchContext &PC,
                            uint64_t TextSize) {
  uint64_t KernelSize = PC.KEnd - PC.KStart;
  PC.PadSize = computePreKernelPadSize(PC.Plan, PC.KStart, KernelSize, TextSize);
  if (PC.PadSize == 0)
    return;

  Cfg.log("Pre-kernel padding: adding " + std::to_string(PC.PadSize) +
          " bytes (kernel " + std::to_string(KernelSize) +
          " bytes, original KStart=" + std::to_string(PC.KStart) + ")");
  PC.KStart += PC.PadSize;
  PC.KEnd += PC.PadSize;
  for (auto &S : PC.Sites)
    S.Address += PC.PadSize;
}

/// Step 8 — assemble patched .text: zero-pad prefix, copy original bytes,
/// then overlay prologue / patch slots / islands at absolute offsets.
std::vector<uint8_t> assemblePatchedText(const CodeObjectHandler &Handler,
                                          const BridgeResult &BR,
                                          uint64_t KStart, uint64_t PadSize,
                                          uint64_t AlignedPrologue,
                                          uint64_t PrologueSize) {
  ArrayRef<uint8_t> TextSection = Handler.getTextSection();
  uint64_t RequiredSize = PadSize + TextSection.size();
  for (const auto &Isl : BR.Islands)
    RequiredSize = std::max(
        RequiredSize, Isl.Offset + AlignedPrologue + Isl.Bytes.size());

  std::vector<uint8_t> NewText(RequiredSize, 0x00);
  std::memcpy(NewText.data() + PadSize, TextSection.data(), TextSection.size());

  if (PrologueSize > 0) {
    uint64_t PrologueOffset = KStart - PrologueSize;
    std::memcpy(NewText.data() + PrologueOffset, BR.PrologueBytes.data(),
                BR.PrologueBytes.size());
  }

  for (const auto &Slot : BR.Slots) {
    uint64_t AbsOffset = Slot.OriginalPC + AlignedPrologue;
    std::memcpy(NewText.data() + AbsOffset, Slot.PatchBytes.data(),
                Slot.PatchBytes.size());
  }

  for (const auto &Isl : BR.Islands)
    std::memcpy(NewText.data() + Isl.Offset + AlignedPrologue,
                Isl.Bytes.data(), Isl.Bytes.size());

  return NewText;
}

/// Step 10 — build source-mapped SiteMap entries for the patched sites.
void buildSiteMap(const PatchContext &PC, Disassembler &Disasm,
                  const SourceMapper *SrcMapper, uint32_t InstrumentedCount,
                  PatchedKernel &Result) {
  // `Sites[i].Address` was shifted by the *initial* pre-kernel pad (before
  // the prologue shift); the later prologue bump adds `AlignedPrologue` to
  // `PC.PadSize` but does not re-shift the site addresses.  Undo that
  // initial pad to recover the pre-padding PC — stable across variants.
  const uint64_t InitialPad =
      PC.PadSize >= PC.AlignedPrologue ? PC.PadSize - PC.AlignedPrologue : 0;
  for (uint32_t i = 0; i < InstrumentedCount; ++i) {
    SiteInfo SI;
    SI.SiteID = i;
    SI.PC = PC.Sites[i].Address;
    SI.OriginalPC = PC.Sites[i].Address - InitialPad;
    std::string OpcodeName = Disasm.getInstructionName(PC.Sites[i].OrigInst);
    SI.InstrName = Disasm.getAsmMnemonic(PC.Sites[i].OrigInst);
    SI.IsLoad = PC.Sites[i].IsLoad;
    SI.ElemSize = CoalescingAnalyzer::inferElemSize(OpcodeName);
    SI.IsLDS = !PC.Sites[i].IsGlobal;
    SI.DSOffset0 = PC.Sites[i].DSOffset0;
    SI.DSOffset1 = PC.Sites[i].DSOffset1;
    SI.IsDualDS = PC.Sites[i].IsDualDS;

    if (SrcMapper) {
      auto Loc = SrcMapper->lookup(PC.Sites[i].Address);
      if (Loc.isValid()) {
        SI.SourceFile = Loc.shortFile();
        SI.SourceFileFull = Loc.File;
        SI.SourceLine = Loc.Line;
        SI.SourceColumn = Loc.Column;
      }
    }

    Result.SiteMap.push_back(std::move(SI));
  }
}

} // namespace

Expected<std::unique_ptr<KernelPatcher>>
KernelPatcher::create(StringRef GPUArch) {
  auto Patcher = std::unique_ptr<KernelPatcher>(new KernelPatcher());
  Patcher->GPUArch = GPUArch.str();

  std::string CPU = GPUArch.str();
  auto DisasmOrErr = Disassembler::create("amdgcn-amd-amdhsa", CPU, "+wavefrontsize64");
  if (!DisasmOrErr) {
    return DisasmOrErr.takeError();
  }
  Patcher->Disasm = std::move(*DisasmOrErr);

  return Patcher;
}

KernelPatcher::~KernelPatcher() = default;

Expected<const PatchedKernel*>
KernelPatcher::getOrPatch(const CapturedCodeObject& CodeObj,
                           const CapturedKernelSymbol& Symbol,
                           InstrumentationMode Mode,
                           const TraceConfig* Trace) {
  // Thin wrapper over getOrPatchVariants with a single trace config.
  // Preserves every existing caller's expected "single patched ELF"
  // semantics — the returned pointer is the first (and only) variant.
  TraceConfig Single{};
  if (Trace)
    Single = *Trace;
  auto Provider = [&Single, Trace](uint32_t V) -> const TraceConfig * {
    if (V != 0)
      return nullptr;
    return Trace ? &Single : nullptr;
  };
  auto VariantsOrErr =
      getOrPatchVariants(CodeObj, Symbol, Mode, /*MaxVariants=*/1u,
                         TraceConfigProvider(std::move(Provider)));
  if (!VariantsOrErr)
    return VariantsOrErr.takeError();
  if (VariantsOrErr->empty())
    return createStringError(inconvertibleErrorCode(),
                             "getOrPatchVariants returned zero variants");
  return VariantsOrErr->front();
}

Expected<llvm::SmallVector<const PatchedKernel*, 4>>
KernelPatcher::getOrPatchVariants(const CapturedCodeObject &CodeObj,
                                   const CapturedKernelSymbol &Symbol,
                                   InstrumentationMode Mode,
                                   uint32_t MaxVariants,
                                   TraceConfigProvider ProvideTrace) {
  RuntimeConfig &Cfg = RuntimeConfig::getInstance();

  if (MaxVariants == 0)
    MaxVariants = 1;

  // Parent key: VariantID=0 carries all variants for this (CodeObj, Kernel,
  // Mode).  We hand out per-variant pointers out of the vector stored at
  // this key.  Per-variant `PatchCacheKey` values (with VariantID>0) are
  // used downstream for `LoadedKernelCache` keying.
  PatchCacheKey ParentKey{CodeObj.CodeObjectId, Symbol.KernelId, Mode,
                          /*VariantID=*/0};

  {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    auto It = Cache.find(ParentKey);
    if (It != Cache.end() && !It->second.empty()) {
      Stats.CacheHits++;
      llvm::SmallVector<const PatchedKernel *, 4> Out;
      Out.reserve(It->second.size());
      for (auto &UP : It->second)
        Out.push_back(UP.get());
      return Out;
    }
  }

  Cfg.log("Patching kernel: " + Symbol.KernelName +
          (MaxVariants > 1
               ? " (requesting up to " + std::to_string(MaxVariants) +
                     " variants)"
               : ""));

  VariantCoverageAccumulator Cov;
  std::vector<std::unique_ptr<PatchedKernel>> Built;
  Built.reserve(MaxVariants);

  for (uint32_t V = 0; V < MaxVariants; ++V) {
    // Lazy-allocate the next variant's trace buffer.  A null return tells
    // us the caller can't / won't supply another variant (e.g. buffer
    // OOM) — bail with whatever we have.  V=0 returning null means "no
    // trace config at all" which we still honor (patchKernel tolerates a
    // null Trace).
    const TraceConfig *TracePtr = ProvideTrace ? ProvideTrace(V) : nullptr;
    if (V > 0 && TracePtr == nullptr) {
      Cfg.log("Variant " + std::to_string(V) +
              ": provider returned null; stopping variant loop");
      break;
    }

    const std::unordered_set<uint64_t> &Excluded = Cov.covered();

    auto PatchedOrErr = patchKernel(
        CodeObj, Symbol, Mode, TracePtr,
        Excluded.empty() ? nullptr : &Excluded);
    if (!PatchedOrErr) {
      {
        std::lock_guard<std::mutex> Lock(CacheMutex);
        Stats.TotalPatchErrors++;
      }
      if (V == 0) {
        // Couldn't build even the first variant — propagate the error
        // so the caller falls back to passing the dispatch through.
        return PatchedOrErr.takeError();
      }
      // Later variant failed: log and stop iterating with whatever we
      // have.  This matches the plateau path (caller still gets the
      // variants already built).
      Cfg.log("Variant " + std::to_string(V) + ": patching failed (" +
              toString(PatchedOrErr.takeError()) +
              "); stopping variant loop");
      break;
    }

    // Collect the *original* PCs actually instrumented in this variant.
    // `SiteInfo::OriginalPC` is the stable, pre-padding address — the same
    // domain that `findAndSortSites` uses when applying `ExcludedPCs`, so
    // downstream variants can reliably skip sites already covered.
    std::vector<uint64_t> PatchedPCs;
    PatchedPCs.reserve(PatchedOrErr->SiteMap.size());
    for (const auto &SI : PatchedOrErr->SiteMap)
      PatchedPCs.push_back(SI.OriginalPC);
    size_t NewCount = Cov.markCovered(PatchedPCs);

    Cfg.log(llvm::formatv("Variant {0}: {1} sites patched (+{2} new), "
                          "union {3} sites so far",
                          V, PatchedPCs.size(), NewCount, Cov.coveredCount())
                .str());

    Built.push_back(std::make_unique<PatchedKernel>(std::move(*PatchedOrErr)));

    if (Cfg.Debug.ReplayAuto && NewCount == 0 && V > 0) {
      Cfg.log("Variant " + std::to_string(V) +
              " added nothing new; stopping (plateau)");
      break;
    }
  }

  Cfg.log("Replay: " + std::to_string(Built.size()) + "/" +
          std::to_string(MaxVariants) + " variants built, union " +
          std::to_string(Cov.coveredCount()) + " sites");

  llvm::SmallVector<const PatchedKernel *, 4> Out;
  Out.reserve(Built.size());

  std::lock_guard<std::mutex> Lock(CacheMutex);
  Stats.CacheMisses++;
  Stats.TotalPatched += Built.size();
  Cache[ParentKey] = std::move(Built);
  for (auto &UP : Cache[ParentKey])
    Out.push_back(UP.get());
  return Out;
}

void KernelPatcher::clearCache() {
  std::lock_guard<std::mutex> Lock(CacheMutex);
  Cache.clear();
}

KernelPatcher::CacheStats KernelPatcher::getCacheStats() const {
  std::lock_guard<std::mutex> Lock(CacheMutex);
  return Stats;
}

//===----------------------------------------------------------------------===//
// Patching pipeline (above-the-count registers)
//
// 1. Parse code object, extract kernel
// 2. Build CFG, find global memory sites
// 3. Pick scratch registers above the kernel's declared count
// 4. Build trampoline (empty: displaced instruction + jump back)
// 5. Patch .text in-place, bump descriptor, rebuild ELF
//===----------------------------------------------------------------------===//

Expected<PatchedKernel>
KernelPatcher::patchKernel(const CapturedCodeObject& CodeObj,
                            const CapturedKernelSymbol& Symbol,
                            InstrumentationMode Mode,
                            const TraceConfig* Trace,
                            const std::unordered_set<uint64_t>* ExcludedPCs) {
  RuntimeConfig &Cfg = RuntimeConfig::getInstance();
  Cfg.log("Patching " + Symbol.KernelName +
          (Trace ? " [instrumented]" : " [empty]"));

  PatchContext PC;
  PC.CodeObj = &CodeObj;
  PC.Symbol = &Symbol;
  PC.Mode = Mode;
  PC.Trace = Trace;
  PC.ExcludedPCs = ExcludedPCs;

  dumpInputCodeObjectIfRequested(Cfg, CodeObj, Symbol);

  if (auto E = loadCodeObjectAndExtractKernel(PC))
    return std::move(E);

  if (auto E = buildCFG(*Disasm, PC))
    return std::move(E);

  if (auto E = planInstrumentation(PC))
    return std::move(E);

  allocateScratch(Cfg, *Disasm, PC);

  if (PC.Plan.Instrumented) {
    Cfg.log("Instrumentation plan: register=" +
            std::string(toString(PC.Plan.Register)) +
            ", payload=" +
            (PC.Plan.Payload == PayloadStrategy::OnGpuReduce
                 ? "on-gpu-reduce"
                 : "full-capture") +
            ", jump=" + std::string(toString(PC.Plan.Jump)) +
            (PC.Plan.SupportsGPUAtomics ? ", gpu-atomics=yes"
                                        : ", gpu-atomics=no"));
  } else {
    Cfg.log("Instrumentation plan: empty trampoline");
  }

  if (auto E = findAndSortSites(Cfg, *Disasm, PC))
    return std::move(E);

  maybeUpgradeToSwapPC(Cfg, PC);

  Cfg.log("Scratch: " + std::to_string(PC.Scratch.ExtraSGPRs) + " SGPRs + " +
          std::to_string(PC.Scratch.ExtraVGPRs) + " VGPRs above-the-count" +
          " (kernel uses " +
          std::to_string(PC.KInfo->Descriptor.VGPRCount) + " VGPRs, " +
          std::to_string(PC.KInfo->Descriptor.SGPRCount) + " SGPRs" +
          (PC.KInfo->Descriptor.AccumOffset > 0
               ? ", accum_offset=" +
                     std::to_string(PC.KInfo->Descriptor.AccumOffset)
               : "") +
          ")");

  const uint64_t TextSize = PC.Handler->getTextSection().size();
  applyPreKernelPadding(Cfg, PC, TextSize);

  // ---- Build trampoline bridge ----
  auto BridgeOrErr = TrampolineBridge::create(GPUArch, *Disasm);
  if (!BridgeOrErr)
    return BridgeOrErr.takeError();

  // .text may contain many other kernels/device functions. Register them
  // as occupied so the island allocator skips over them.
  TrampolineBridge::OccupiedRanges Occupied;
  for (const auto &R : PC.Handler->getTextFunctionRanges())
    Occupied.emplace_back(R.first + PC.PadSize, R.second + PC.PadSize);

  uint64_t EffectiveTextEnd = PC.KEnd;
  Expected<BridgeResult> BROrErr =
      Trace ? (*BridgeOrErr)
                  ->buildInstrumented(PC.KCode, PC.KStart, EffectiveTextEnd,
                                       PC.Sites, PC.Plan, PC.Scratch, *Trace,
                                       PC.KStart, Occupied)
            : (*BridgeOrErr)
                  ->buildEmpty(PC.KCode, PC.KStart, EffectiveTextEnd,
                               PC.Sites, PC.Scratch, PC.KStart, Occupied);
  if (!BROrErr)
    return BROrErr.takeError();
  BridgeResult &BR = *BROrErr;

  {
    uint64_t TotalIslandBytes = 0;
    for (const auto &Isl : BR.Islands)
      TotalIslandBytes += Isl.Bytes.size();
    Cfg.log("Trampoline: " + std::to_string(BR.PatchedCount) + " sites, " +
            std::to_string(BR.Islands.size()) + " island(s), " +
            std::to_string(TotalIslandBytes) + " total island bytes");
  }

  // ---- Optional prologue bytes (SwapPC path) shift everything further ----
  uint64_t PrologueSize = BR.PrologueBytes.size();
  PC.AlignedPrologue = 0;
  if (PrologueSize > 0) {
    PC.AlignedPrologue = (PrologueSize + 3) & ~3ULL;
    PC.PadSize += PC.AlignedPrologue;
    PC.KStart += PC.AlignedPrologue;
    PC.KEnd += PC.AlignedPrologue;
    Cfg.log("Prepended trampoline prologue: " + std::to_string(PrologueSize) +
            " bytes");
  }

  // ---- Assemble new .text and apply to the handler ----
  std::vector<uint8_t> NewText = assemblePatchedText(
      *PC.Handler, BR, PC.KStart, PC.PadSize, PC.AlignedPrologue,
      PrologueSize);
  PC.Handler->setTextSection(NewText);

  // ---- Bump descriptor + rebuild ELF ----
  uint32_t AdditionalKernargSize = Trace ? 0 : sizeof(TraceArgs);
  uint32_t AdditionalScratchSize =
      PC.Scratch.NeedsScratchSpill ? PC.Scratch.ExtraScratchBytes : 0;
  int64_t EntryAdjust = static_cast<int64_t>(PC.PadSize) -
                        static_cast<int64_t>(PrologueSize);

  if (auto ApplyErr = PC.Handler->applyPatch(
          NewText, Symbol.KernelName, PC.Scratch.ExtraVGPRs,
          PC.Scratch.ExtraSGPRs, AdditionalKernargSize, AdditionalScratchSize,
          EntryAdjust))
    return std::move(ApplyErr);

  auto OutputOrErr = PC.Handler->build();
  if (!OutputOrErr)
    return OutputOrErr.takeError();

  if (!Cfg.Debug.DumpELFPath.empty()) {
    std::error_code EC;
    llvm::raw_fd_ostream OS(Cfg.Debug.DumpELFPath, EC);
    if (!EC)
      OS.write(reinterpret_cast<const char *>(OutputOrErr->data()),
               OutputOrErr->size());
  }

  // ---- Populate result ----
  PatchedKernel Result;
  Result.PatchedELF = std::move(*OutputOrErr);
  Result.KernelObject = 0;
  Result.OriginalKernargSize = Symbol.KernargSegmentSize;
  Result.KernelName = Symbol.KernelName;
  Result.GPUArch = GPUArch;
  Result.AdditionalVGPRs = PC.Scratch.ExtraVGPRs;
  Result.AdditionalSGPRs = PC.Scratch.ExtraSGPRs;
  Result.Mode = Mode;
  Result.Plan = PC.Plan;
  Result.NumBasicBlocks = static_cast<uint32_t>(PC.CFG->BasicBlocks.size());
  Result.NumInstructions = BR.PatchedCount;
  Result.NumMemorySites = BR.PatchedCount;
  Result.AdditionalScratchBytes =
      PC.Scratch.NeedsScratchSpill ? PC.Scratch.ExtraScratchBytes : 0;
  if (Trace)
    Result.Trace = *Trace;

  // ---- Source mapping ----
  auto SrcMapper = SourceMapper::create(CodeObj.Bytes);
  if (SrcMapper)
    Cfg.log("DWARF source mapper: " +
            std::to_string(SrcMapper->entryCount()) + " line entries");

  uint32_t InstrumentedCount = BR.PatchedCount;
  if (InstrumentedCount < PC.Sites.size()) {
    Cfg.log("Partial instrumentation: " + std::to_string(InstrumentedCount) +
            "/" + std::to_string(PC.Sites.size()) +
            " sites (s_branch range limit)");
  }
  buildSiteMap(PC, *Disasm, SrcMapper.get(), InstrumentedCount, Result);

  if (Cfg.LogEnabled) {
    std::string ElfPath = "/tmp/aegisbit_" + Symbol.KernelName + ".elf";
    std::ofstream ElfOut(ElfPath, std::ios::binary);
    if (ElfOut) {
      ElfOut.write(reinterpret_cast<const char *>(Result.PatchedELF.data()),
                   Result.PatchedELF.size());
      Cfg.log("Wrote patched ELF to " + ElfPath);
    }
  }

  Cfg.log("Done: " + Symbol.KernelName + " — " +
          std::to_string(BR.PatchedCount) + " trampolines, " +
          std::to_string(Result.PatchedELF.size()) + " byte ELF");

  return Result;
}

} // namespace aegisbit
