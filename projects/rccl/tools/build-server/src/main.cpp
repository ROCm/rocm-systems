#include "clang/Basic/DiagnosticOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "lld/Common/Driver.h"

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCTargetOptions.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

LLD_HAS_DRIVER(elf)

// ---- Trace (Chrome Trace Format) ----

using Clock = std::chrono::high_resolution_clock;
using TimePoint = Clock::time_point;

struct TraceEvent {
  std::string Name;
  std::string Cat;
  int Tid;
  double StartUs;
  double DurUs;
};

class Tracer {
  std::mutex Mu;
  std::vector<TraceEvent> Events;
  TimePoint Epoch;

public:
  Tracer() : Epoch(Clock::now()) {}
  void setEpoch(TimePoint T) { Epoch = T; }

  void addEvent(const std::string &Name, const std::string &Cat,
                int Tid, TimePoint Start, TimePoint End) {
    double S = std::chrono::duration<double, std::micro>(Start - Epoch).count();
    double D = std::chrono::duration<double, std::micro>(End - Start).count();
    std::lock_guard<std::mutex> Lock(Mu);
    Events.push_back({Name, Cat, Tid, S, D});
  }

  bool writeJSON(const std::string &Path) const {
    std::error_code EC;
    llvm::raw_fd_ostream OS(Path, EC);
    if (EC) return false;
    OS << "[\n";
    bool First = true;
    for (const auto &E : Events) {
      if (!First) OS << ",\n";
      First = false;
      OS << "  {\"name\":\"" << E.Name
         << "\",\"cat\":\"" << E.Cat
         << "\",\"ph\":\"X\""
         << ",\"ts\":" << llvm::format("%.1f", E.StartUs)
         << ",\"dur\":" << llvm::format("%.1f", E.DurUs)
         << ",\"pid\":1,\"tid\":" << E.Tid << "}";
    }
    OS << "\n]\n";
    return true;
  }
};

static Tracer GTrace;

// ---- Data Structures ----

struct BuildConfig {
  std::string GPUArch = "gfx942";
  std::string RcclBuild = "/work/lmeadows/rocm-systems/projects/rccl/build";
  std::string ClangPath = "/work/lmeadows/rocm/aomp_23.0-1/bin/amdclang++";
  std::string ResourceDir =
      "/work/lmeadows/rocm/aomp_23.0-1/lib/llvm/lib/clang/23";
  std::string RocmPath = "/opt/rocm";
  std::string TraceFile = "build-trace.json";
  unsigned NumThreads = 0;
};

struct DeviceMetadata {
  int MaxVGPR = 0, MaxAGPR = 0, MaxSGPR = 0, MaxNamedBarrier = 0;
};

struct CompileResult {
  std::string Name;
  std::string AsmText;
  llvm::SmallVector<char, 0> ObjBuffer;
  DeviceMetadata Meta;
  bool IsKernel = false;
  bool Success = false;
};

struct SourceInfo {
  std::string Path;
  std::string Name;
  bool IsKernel;
};

// ---- LLVM Init ----

static void initLLVMTargets() {
  LLVMInitializeAMDGPUTargetInfo();
  LLVMInitializeAMDGPUTarget();
  LLVMInitializeAMDGPUTargetMC();
  LLVMInitializeAMDGPUAsmPrinter();
  LLVMInitializeAMDGPUAsmParser();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmPrinter();
  LLVMInitializeX86AsmParser();
}

static std::unique_ptr<llvm::TargetMachine>
createAMDGPUTargetMachine(llvm::StringRef GPUArch) {
  llvm::Triple Triple("amdgcn-amd-amdhsa");
  std::string Error;
  const llvm::Target *T = llvm::TargetRegistry::lookupTarget(Triple, Error);
  if (!T) return nullptr;
  llvm::TargetOptions Opts;
  return std::unique_ptr<llvm::TargetMachine>(T->createTargetMachine(
      Triple, GPUArch, "", Opts, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive));
}

// ---- CC1 arg capture ----
// Call createInvocation() ONCE to translate driver args -> cc1 args.
// Then reuse the cc1 args for every TU, only swapping the source file.

static std::vector<std::string>
captureCC1Args(const BuildConfig &Cfg, const std::string &DummySrc,
               bool IsFuncOnly) {
  std::string H = Cfg.RcclBuild + "/hipify";
  std::vector<std::string> DriverArgs = {
      Cfg.ClangPath, "-x", "hip", "-std=c++17",
      "-fgpu-rdc", "--offload-device-only",
      "--offload-arch=" + Cfg.GPUArch,
      "-emit-llvm", "-c", "-O3",
      "-resource-dir", Cfg.ResourceDir,
      "--rocm-path=" + Cfg.RocmPath, "-nogpulib",
      "-I" + Cfg.RcclBuild + "/include",
      "-I" + H + "/src", "-I" + H + "/src/device",
      "-I" + H + "/src/device/network/unpack",
      "-I" + H + "/src/include", "-I" + H + "/src/include/mlx5",
      "-I" + H + "/src/include/nccl_device",
      "-I" + H + "/src/include/ionic",
      "-I" + H + "/src/include/plugin",
      "-I" + H + "/gensrc", "-I/opt/rocm/include",
      "-DENABLE_COLLTRACE", "-DNDEBUG",
      "-DUSE_ROCM_SMI64CONFIG", "-DUSE_ROCM_SMI_THREAD_ONLY_MUTEX",
      "-DROCTX_ENABLE", "-DHIP_CONTIGUOUS_MEMORY",
      "-DHIP_UNCACHED_MEMORY", "-DHIP_HOST_UNCACHED_MEMORY",
      "-DUSE_INDIRECT_FUNCTION_CALL", "-DENABLE_LL128",
      "-DENABLE_FAULT_INJECTION", "-DRCCL_ROCPROFILER_REGISTER=1",
      "-Werror=uninitialized", "-Werror=sometimes-uninitialized", "-Wall",
      "-Werror=deprecated-copy-with-user-provided-copy",
      "-Wno-format-nonliteral", "-Wno-unused-function",
      "-gline-tables-only",
      "-mllvm", "--amdgpu-kernarg-preload-count=16",
      "-fvisibility=hidden",
  };
  if (IsFuncOnly) DriverArgs.push_back("-DNCCL_FUNC_ONLY");
  DriverArgs.push_back("-o");
  DriverArgs.push_back("/dev/null");
  DriverArgs.push_back(DummySrc);

  std::vector<const char *> Ptrs;
  for (const auto &A : DriverArgs) Ptrs.push_back(A.c_str());

  clang::CreateInvocationOptions Opts;
  Opts.RecoverOnError = false;
  std::vector<std::string> CC1Args;
  Opts.CC1Args = &CC1Args;

  auto CI = clang::createInvocation(Ptrs, Opts);
  if (!CI) {
    llvm::errs() << "captureCC1Args: createInvocation failed\n";
    return {};
  }

  llvm::errs() << "Captured " << CC1Args.size() << " cc1 args\n";
  return CC1Args;
}

// Replace the source file in a cc1 arg list. The source file is the
// last positional arg (not starting with '-').
static std::vector<std::string>
replaceSourceFile(const std::vector<std::string> &CC1Args,
                  const std::string &NewSrc) {
  std::vector<std::string> Result = CC1Args;
  // Find and replace the source file. In cc1 args it's typically the
  // last argument, or the argument after -main-file-name might need
  // updating too. The source file appears as a bare positional arg.
  for (int i = Result.size() - 1; i >= 0; --i) {
    if (!Result[i].empty() && Result[i][0] != '-') {
      Result[i] = NewSrc;
      break;
    }
  }
  // Also update -main-file-name if present
  for (unsigned i = 0; i + 1 < Result.size(); ++i) {
    if (Result[i] == "-main-file-name") {
      Result[i + 1] = llvm::sys::path::filename(NewSrc).str();
      break;
    }
  }
  return Result;
}

// ---- Clang Frontend (cc1 args, no Driver) ----

static std::unique_ptr<llvm::Module>
compileCC1(llvm::LLVMContext &Ctx,
           const std::vector<std::string> &CC1Args) {
  std::vector<const char *> Ptrs;
  for (const auto &A : CC1Args) Ptrs.push_back(A.c_str());

  clang::DiagnosticOptions DiagOpts;
  auto *Printer = new clang::TextDiagnosticPrinter(llvm::errs(), DiagOpts);
  llvm::IntrusiveRefCntPtr<clang::DiagnosticIDs> DiagID(
      new clang::DiagnosticIDs());
  clang::DiagnosticsEngine Diags(DiagID, DiagOpts, Printer, true);

  auto CI = std::make_shared<clang::CompilerInvocation>();
  if (!clang::CompilerInvocation::CreateFromArgs(*CI, Ptrs, Diags))
    return nullptr;

  CI->getFrontendOpts().OutputFile = "/dev/null";

  clang::CompilerInstance Compiler;
  Compiler.getInvocation() = std::move(*CI);
  Compiler.createDiagnostics(Printer, false);
  if (!Compiler.hasDiagnostics())
    return nullptr;

  clang::EmitLLVMOnlyAction Action(&Ctx);
  if (!Compiler.ExecuteAction(Action))
    return nullptr;

  return Action.takeModule();
}

// ---- AMDGPU Backend ----

static bool
emitAssembly(llvm::Module &M, llvm::TargetMachine &TM,
             std::string &AsmText) {
  M.setDataLayout(TM.createDataLayout());
  M.setTargetTriple(TM.getTargetTriple());
  llvm::SmallVector<char, 0> Buf;
  llvm::raw_svector_ostream OS(Buf);
  llvm::legacy::PassManager PM;
  if (TM.addPassesToEmitFile(PM, OS, nullptr,
                             llvm::CodeGenFileType::AssemblyFile))
    return false;
  PM.run(M);
  AsmText.assign(Buf.data(), Buf.size());
  return true;
}

// ---- In-memory assembler ----

static bool
assembleToObject(llvm::StringRef AsmText, llvm::StringRef GPUArch,
                 llvm::SmallVectorImpl<char> &ObjBuffer) {
  llvm::Triple Triple("amdgcn-amd-amdhsa");
  std::string Error;
  const llvm::Target *T = llvm::TargetRegistry::lookupTarget(Triple, Error);
  if (!T) return false;

  std::unique_ptr<llvm::MCRegisterInfo> MRI(T->createMCRegInfo(Triple));
  if (!MRI) return false;
  llvm::MCTargetOptions MCOpts;
  std::unique_ptr<llvm::MCAsmInfo> MAI(
      T->createMCAsmInfo(*MRI, Triple, MCOpts));
  if (!MAI) return false;
  std::unique_ptr<llvm::MCSubtargetInfo> STI(
      T->createMCSubtargetInfo(Triple, GPUArch, ""));
  if (!STI) return false;
  std::unique_ptr<llvm::MCInstrInfo> MII(T->createMCInstrInfo());

  llvm::SourceMgr SrcMgr;
  SrcMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(AsmText, "<asm>"), llvm::SMLoc());

  llvm::MCContext MCCtx(Triple, MAI.get(), MRI.get(), STI.get(), &SrcMgr);
  std::unique_ptr<llvm::MCObjectFileInfo> MOFI(
      T->createMCObjectFileInfo(MCCtx, true));
  MCCtx.setObjectFileInfo(MOFI.get());

  llvm::raw_svector_ostream ObjOS(ObjBuffer);
  auto CE = std::unique_ptr<llvm::MCCodeEmitter>(
      T->createMCCodeEmitter(*MII, MCCtx));
  auto MAB = std::unique_ptr<llvm::MCAsmBackend>(
      T->createMCAsmBackend(*STI, *MRI, MCOpts));
  auto OW = std::unique_ptr<llvm::MCObjectWriter>(
      MAB->createObjectWriter(ObjOS));
  auto Str = std::unique_ptr<llvm::MCStreamer>(
      T->createMCObjectStreamer(Triple, MCCtx, std::move(MAB),
                                std::move(OW), std::move(CE), *STI));

  auto Parser = std::unique_ptr<llvm::MCAsmParser>(
      llvm::createMCAsmParser(SrcMgr, MCCtx, *Str, *MAI));
  auto TAP = std::unique_ptr<llvm::MCTargetAsmParser>(
      T->createMCAsmParser(*STI, *Parser, *MII, MCOpts));
  if (!TAP) return false;
  Parser->setTargetParser(*TAP);
  return !Parser->Run(false);
}

// ---- Metadata ----

static int parseIntAfter(llvm::StringRef Text, llvm::StringRef Pat) {
  auto Pos = Text.find(Pat);
  if (Pos == llvm::StringRef::npos) return 0;
  Pos += Pat.size();
  while (Pos < Text.size() && (Text[Pos] == ' ' || Text[Pos] == ',')) Pos++;
  int V = 0;
  while (Pos < Text.size() && Text[Pos] >= '0' && Text[Pos] <= '9')
    V = V * 10 + (Text[Pos++] - '0');
  return V;
}

static DeviceMetadata extractMetadata(llvm::StringRef Asm) {
  return {
    parseIntAfter(Asm, ".set amdgpu.max_num_vgpr,"),
    parseIntAfter(Asm, ".set amdgpu.max_num_agpr,"),
    parseIntAfter(Asm, ".set amdgpu.max_num_sgpr,"),
    parseIntAfter(Asm, ".set amdgpu.max_num_named_barrier,"),
  };
}

// ---- Source discovery ----

static std::vector<SourceInfo>
discoverSources(const BuildConfig &Cfg) {
  std::vector<SourceInfo> S;
  std::string Dir = Cfg.RcclBuild + "/hipify/gensrc";
  std::error_code EC;
  for (llvm::sys::fs::directory_iterator DI(Dir, EC), DE;
       !EC && DI != DE; DI.increment(EC)) {
    llvm::StringRef P = DI->path();
    if (!P.ends_with(".cpp")) continue;
    S.push_back({P.str(), llvm::sys::path::stem(P).str(), false});
  }
  std::sort(S.begin(), S.end(),
            [](const SourceInfo &A, const SourceInfo &B) {
              return A.Name < B.Name;
            });
  S.push_back(
      {Cfg.RcclBuild + "/hipify/src/device/common.cu.cpp", "common", true});
  return S;
}

static std::mutex PrintMutex;

// ---- Main ----

int main(int argc, char **argv) {
  auto T0 = Clock::now();
  GTrace.setEpoch(T0);
  initLLVMTargets();

  BuildConfig Cfg;
  if (argc > 1) Cfg.GPUArch = argv[1];
  if (argc > 2) Cfg.RcclBuild = argv[2];
  if (argc > 3) Cfg.NumThreads = std::atoi(argv[3]);
  if (argc > 4) Cfg.TraceFile = argv[4];
  if (Cfg.NumThreads == 0)
    Cfg.NumThreads = std::min(128u, std::thread::hardware_concurrency());

  llvm::errs() << "GPU=" << Cfg.GPUArch
               << " threads=" << Cfg.NumThreads
               << " trace=" << Cfg.TraceFile << "\n";

  auto Sources = discoverSources(Cfg);
  llvm::errs() << Sources.size() << " device TUs\n";

  // ---- Capture cc1 args once (callee and kernel variants) ----
  auto T_cc1 = Clock::now();

  // Use first callee source as template for cc1 capture
  std::string FirstCallee;
  std::string KernelSrc;
  for (const auto &S : Sources) {
    if (S.IsKernel) KernelSrc = S.Path;
    else if (FirstCallee.empty()) FirstCallee = S.Path;
  }

  auto CalleeCC1 = captureCC1Args(Cfg, FirstCallee, /*IsFuncOnly=*/true);
  auto KernelCC1 = captureCC1Args(Cfg, KernelSrc, /*IsFuncOnly=*/false);

  if (CalleeCC1.empty() || KernelCC1.empty()) {
    llvm::errs() << "Failed to capture cc1 args\n";
    return 1;
  }

  auto T_cc1_end = Clock::now();
  double CC1CaptureMs =
      std::chrono::duration<double, std::milli>(T_cc1_end - T_cc1).count();
  llvm::errs() << "CC1 capture: " << llvm::format("%.0f", CC1CaptureMs)
               << " ms (callee: " << CalleeCC1.size()
               << " args, kernel: " << KernelCC1.size() << " args)\n";

  std::vector<CompileResult> Results(Sources.size());

  // ===========================================================
  // Phase 1: Parallel frontend + backend using cc1 args directly.
  //          No Driver, no global locks.
  // ===========================================================
  auto TP1 = Clock::now();

  std::atomic<unsigned> NextIdx{0};
  std::atomic<unsigned> DoneCount{0};

  auto Worker = [&]() {
    while (true) {
      unsigned Idx = NextIdx.fetch_add(1);
      if (Idx >= Sources.size()) break;

      auto &Src = Sources[Idx];
      auto &R = Results[Idx];
      R.Name = Src.Name;
      R.IsKernel = Src.IsKernel;

      // Build per-TU cc1 args by swapping the source file
      auto CC1 = replaceSourceFile(
          Src.IsKernel ? KernelCC1 : CalleeCC1, Src.Path);

      // -- Frontend (no Driver involved) --
      auto T_fe0 = Clock::now();
      llvm::LLVMContext Ctx;
      auto M = compileCC1(Ctx, CC1);
      auto T_fe1 = Clock::now();
      GTrace.addEvent(Src.Name, "frontend", Idx, T_fe0, T_fe1);

      if (!M) continue;

      // -- Backend (asm) --
      auto TM = createAMDGPUTargetMachine(Cfg.GPUArch);
      if (!TM) continue;

      auto T_be0 = Clock::now();
      if (!emitAssembly(*M, *TM, R.AsmText)) continue;
      auto T_be1 = Clock::now();
      GTrace.addEvent(Src.Name, "backend", Idx, T_be0, T_be1);

      if (!Src.IsKernel)
        R.Meta = extractMetadata(R.AsmText);

      R.Success = true;

      unsigned Done = DoneCount.fetch_add(1) + 1;
      if (Done % 10 == 0 || Done == Sources.size()) {
        std::lock_guard<std::mutex> Lock(PrintMutex);
        llvm::errs() << "\r  Phase 1: " << Done << "/" << Sources.size()
                      << " compiled     ";
      }
    }
  };

  {
    std::vector<std::thread> Threads;
    for (unsigned i = 0; i < Cfg.NumThreads; ++i)
      Threads.emplace_back(Worker);
    for (auto &T : Threads)
      T.join();
  }
  llvm::errs() << "\n";

  auto TP2 = Clock::now();
  double Phase1Ms =
      std::chrono::duration<double, std::milli>(TP2 - TP1).count();

  DeviceMetadata GlobalMeta;
  unsigned P1Ok = 0, P1Fail = 0;
  for (unsigned i = 0; i < Sources.size(); ++i) {
    if (!Results[i].Success) { P1Fail++; continue; }
    P1Ok++;
    if (!Sources[i].IsKernel) {
      GlobalMeta.MaxVGPR = std::max(GlobalMeta.MaxVGPR, Results[i].Meta.MaxVGPR);
      GlobalMeta.MaxAGPR = std::max(GlobalMeta.MaxAGPR, Results[i].Meta.MaxAGPR);
      GlobalMeta.MaxSGPR = std::max(GlobalMeta.MaxSGPR, Results[i].Meta.MaxSGPR);
      GlobalMeta.MaxNamedBarrier =
          std::max(GlobalMeta.MaxNamedBarrier, Results[i].Meta.MaxNamedBarrier);
    }
  }

  llvm::errs() << "\n=== Phase 1 ===\n"
               << "OK=" << P1Ok << " FAIL=" << P1Fail
               << " Wall=" << llvm::format("%.1f", Phase1Ms) << "ms\n"
               << "Meta: V=" << GlobalMeta.MaxVGPR
               << " A=" << GlobalMeta.MaxAGPR
               << " S=" << GlobalMeta.MaxSGPR << "\n";

  if (P1Fail > 0) {
    for (unsigned i = 0; i < Sources.size(); ++i)
      if (!Results[i].Success)
        llvm::errs() << "  FAIL: " << Sources[i].Name << "\n";
    return 1;
  }

  // ===========================================================
  // Phase 2: Assemble asm -> obj (parallel)
  // ===========================================================
  auto TP3 = Clock::now();
  NextIdx.store(0);
  DoneCount.store(0);

  auto AsmWorker = [&]() {
    while (true) {
      unsigned Idx = NextIdx.fetch_add(1);
      if (Idx >= Sources.size()) break;
      if (!Results[Idx].Success) continue;
      auto T0a = Clock::now();
      bool OK = assembleToObject(Results[Idx].AsmText, Cfg.GPUArch,
                                 Results[Idx].ObjBuffer);
      auto T1a = Clock::now();
      GTrace.addEvent(Results[Idx].Name, "assemble", Idx, T0a, T1a);
      if (!OK) Results[Idx].Success = false;
      Results[Idx].AsmText.clear();
      Results[Idx].AsmText.shrink_to_fit();
      DoneCount.fetch_add(1);
    }
  };

  {
    std::vector<std::thread> Threads;
    for (unsigned i = 0; i < Cfg.NumThreads; ++i)
      Threads.emplace_back(AsmWorker);
    for (auto &T : Threads) T.join();
  }

  auto TP4 = Clock::now();
  double Phase2Ms =
      std::chrono::duration<double, std::milli>(TP4 - TP3).count();
  size_t TotalObj = 0;
  for (auto &R : Results) TotalObj += R.ObjBuffer.size();
  llvm::errs() << "\n=== Phase 2 (asm->obj) ===\n"
               << "Wall=" << llvm::format("%.1f", Phase2Ms) << "ms"
               << " ObjTotal=" << TotalObj << "\n";

  // ===========================================================
  // Phase 3: lld -r
  // ===========================================================
  auto TP5 = Clock::now();
  std::string TmpDir = "/dev/shm/rccl-build-" + std::to_string(getpid());
  llvm::sys::fs::create_directories(TmpDir);

  std::vector<std::string> ObjPaths;
  for (unsigned i = 0; i < Results.size(); ++i) {
    if (Results[i].ObjBuffer.empty()) continue;
    std::string P = TmpDir + "/" + Results[i].Name + ".o";
    std::error_code EC;
    llvm::raw_fd_ostream F(P, EC);
    if (EC) continue;
    F.write(Results[i].ObjBuffer.data(), Results[i].ObjBuffer.size());
    ObjPaths.push_back(P);
  }

  std::string Combined = TmpDir + "/combined.o";
  std::vector<const char *> LLDArgs = {"ld.lld", "-r", "-o", Combined.c_str()};
  for (const auto &P : ObjPaths) LLDArgs.push_back(P.c_str());

  auto T_l0 = Clock::now();
  llvm::raw_null_ostream NullOS;
  lld::Result LR = lld::lldMain(
      LLDArgs, NullOS, llvm::errs(), {{lld::Gnu, &lld::elf::link}});
  auto T_l1 = Clock::now();
  GTrace.addEvent("lld -r", "link", Sources.size(), T_l0, T_l1);

  uint64_t CombSize = 0;
  if (LR.retCode == 0) llvm::sys::fs::file_size(Combined, CombSize);

  auto TP6 = Clock::now();
  llvm::errs() << "\n=== Phase 3 (lld -r) ===\n"
               << "RC=" << LR.retCode
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(TP6 - TP5).count())
               << "ms Size=" << CombSize << "\n";

  for (const auto &P : ObjPaths) llvm::sys::fs::remove(P);
  llvm::sys::fs::remove(Combined);
  llvm::sys::fs::remove(TmpDir);

  double TotalMs =
      std::chrono::duration<double, std::milli>(Clock::now() - T0).count();
  llvm::errs() << "\n=== TOTAL: " << llvm::format("%.1f", TotalMs)
               << "ms (" << llvm::format("%.1f", TotalMs / 1000) << "s) ===\n";

  if (GTrace.writeJSON(Cfg.TraceFile))
    llvm::errs() << "Trace: " << Cfg.TraceFile << "\n";

  return LR.retCode;
}
