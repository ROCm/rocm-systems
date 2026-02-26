#include "clang/Basic/DiagnosticOptions.h"
#include "clang/CodeGen/CodeGenAction.h"
#include "clang/Driver/CreateInvocationFromArgs.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"

#include "lld/Common/Driver.h"

#include "llvm/ADT/Hashing.h"
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
#include "llvm/Support/CommandLine.h"
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
#include <dirent.h>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <set>
#include <signal.h>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <sys/mman.h>
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

struct ScopedTrace {
  Tracer &Tr;
  std::string Name, Cat;
  int Tid;
  TimePoint Start;
  ScopedTrace(Tracer &Tr, llvm::StringRef N, llvm::StringRef C, int T)
    : Tr(Tr), Name(N.str()), Cat(C.str()), Tid(T), Start(Clock::now()) {}
  ~ScopedTrace() { Tr.addEvent(Name, Cat, Tid, Start, Clock::now()); }
  ScopedTrace(const ScopedTrace &) = delete;
  ScopedTrace &operator=(const ScopedTrace &) = delete;
};

// ---- Debug Probe ----
//
// Dumps intermediate pipeline artifacts for comparison with the CMake build.
// Controlled entirely by environment variables — no code changes needed to use.
//
//   RCCL_BUILD_PROBE=ir,asm,obj   Stages to dump (comma-separated).
//   RCCL_BUILD_PROBE_DIR=/path    Output directory (default: <build>/probe).
//   RCCL_BUILD_PROBE_TU=name,...  Specific TU name(s) (default: all).
//
// Files are written as:
//   <dir>/ir/<name>.ll
//   <dir>/asm/<name>.s
//   <dir>/obj/<name>.o

struct ProbeConfig {
  bool DumpIR  = false;
  bool DumpAsm = false;
  bool DumpObj = false;
  std::string Dir;
  std::set<std::string> TUs;

  bool active() const { return DumpIR || DumpAsm || DumpObj; }

  bool shouldProbe(const std::string &Name) const {
    return TUs.empty() || TUs.count(Name);
  }

  void writeIR(const std::string &Name, const llvm::Module &M) const {
    if (!DumpIR || !shouldProbe(Name)) return;
    std::string P = Dir + "/ir/" + Name + ".ll";
    std::error_code EC;
    llvm::raw_fd_ostream OS(P, EC);
    if (!EC) M.print(OS, nullptr);
  }

  void writeAsm(const std::string &Name, llvm::StringRef Asm) const {
    if (!DumpAsm || !shouldProbe(Name)) return;
    std::string P = Dir + "/asm/" + Name + ".s";
    std::error_code EC;
    llvm::raw_fd_ostream OS(P, EC);
    if (!EC) OS << Asm;
  }

  void writeObj(const std::string &Name, int Fd) const {
    if (!DumpObj || !shouldProbe(Name) || Fd < 0) return;
    std::string FdPath = "/proc/self/fd/" + std::to_string(Fd);
    auto BufOrErr = llvm::MemoryBuffer::getFile(FdPath);
    if (!BufOrErr) return;
    std::string P = Dir + "/obj/" + Name + ".o";
    std::error_code EC;
    llvm::raw_fd_ostream OS(P, EC);
    if (!EC) OS.write((*BufOrErr)->getBufferStart(),
                      (*BufOrErr)->getBufferSize());
  }
};

static ProbeConfig parseProbeEnv(const std::string &DefaultDir) {
  ProbeConfig P;
  P.Dir = DefaultDir;

  if (const char *V = ::getenv("RCCL_BUILD_PROBE")) {
    llvm::StringRef Stages(V);
    llvm::SmallVector<llvm::StringRef, 4> Parts;
    Stages.split(Parts, ',');
    for (auto S : Parts) {
      S = S.trim();
      if (S == "ir")  P.DumpIR  = true;
      else if (S == "asm") P.DumpAsm = true;
      else if (S == "obj") P.DumpObj = true;
    }
  }

  if (const char *V = ::getenv("RCCL_BUILD_PROBE_DIR"))
    P.Dir = V;

  if (const char *V = ::getenv("RCCL_BUILD_PROBE_TU")) {
    llvm::StringRef TUs(V);
    llvm::SmallVector<llvm::StringRef, 8> Parts;
    TUs.split(Parts, ',');
    for (auto S : Parts)
      P.TUs.insert(S.trim().str());
  }

  if (P.active()) {
    auto MkDir = [](const std::string &D) {
      if (auto EC = llvm::sys::fs::create_directories(D))
        llvm::errs() << "Warning: cannot create " << D << ": "
                     << EC.message() << "\n";
    };
    if (P.DumpIR)  MkDir(P.Dir + "/ir");
    if (P.DumpAsm) MkDir(P.Dir + "/asm");
    if (P.DumpObj) MkDir(P.Dir + "/obj");
    llvm::errs() << "Probe: "
                 << (P.DumpIR  ? "ir "  : "")
                 << (P.DumpAsm ? "asm " : "")
                 << (P.DumpObj ? "obj " : "")
                 << "-> " << P.Dir;
    if (!P.TUs.empty()) {
      llvm::errs() << " (";
      bool First = true;
      for (const auto &N : P.TUs) {
        if (!First) llvm::errs() << ",";
        First = false;
        llvm::errs() << N;
      }
      llvm::errs() << ")";
    }
    llvm::errs() << "\n";
  }

  return P;
}

// ---- Data Structures ----

struct BuildConfig {
  std::string GPUArch;
  std::string RcclBuild;
  std::string ClangPath;
  std::string ResourceDir;
  std::string TraceFile = "build-trace.json";
  unsigned NumThreads = 0;

  std::vector<std::string> CalleeFlags;
  std::vector<std::string> KernelFlags;
  std::vector<std::string> BackendFlags;
  std::vector<std::string> CalleeSources;
  std::vector<std::string> KernelSources;

  // Host compilation
  std::vector<std::string> HostFlags;
  std::vector<std::string> HostSources;
  std::string OnerankSource;
  std::vector<std::string> OnerankFlags;

  // SPLIT post-device commands
  std::vector<std::string> SplitCobjArgs;
  std::vector<std::string> SplitHipfbArgs;
  std::vector<std::string> SplitHostArgs;

  // Final link
  std::vector<std::string> LinkFlags;
  std::vector<std::string> LinkLibraries;
  std::vector<std::string> LinkPath;
  std::vector<std::string> LinkObjects;
  std::string LinkSoname;
};

// ---- Config file parser ----

static bool loadFlagConfig(BuildConfig &Cfg, const std::string &Path) {
  std::ifstream In(Path);
  if (!In.is_open()) {
    llvm::errs() << "Cannot open flag config: " << Path << "\n"
                 << "Run: python3 extract-ninja-flags.py " << Cfg.RcclBuild
                 << "\n";
    return false;
  }

  enum Section {
    None, Meta, CalleeFlags, KernelFlags, BackendFlags,
    CalleeSources, KernelSources,
    HostFlags, HostSources, OnerankSource, OnerankFlags,
    SplitCobj, SplitHipfb, SplitHost,
    LinkFlags, LinkLibraries, LinkPath, LinkObjects, LinkSoname
  };
  Section Sec = None;
  std::string Line;
  while (std::getline(In, Line)) {
    if (Line.empty() || Line[0] == '#') continue;

    if (Line[0] == '[') {
      if (Line == "[meta]")            Sec = Meta;
      else if (Line == "[callee_flags]")    Sec = CalleeFlags;
      else if (Line == "[kernel_flags]")    Sec = KernelFlags;
      else if (Line == "[backend_flags]")   Sec = BackendFlags;
      else if (Line == "[callee_sources]")  Sec = CalleeSources;
      else if (Line == "[kernel_sources]")  Sec = KernelSources;
      else if (Line == "[host_flags]")      Sec = HostFlags;
      else if (Line == "[host_sources]")    Sec = HostSources;
      else if (Line == "[onerank_source]")  Sec = OnerankSource;
      else if (Line == "[onerank_flags]")   Sec = OnerankFlags;
      else if (Line == "[split_cobj]")      Sec = SplitCobj;
      else if (Line == "[split_hipfb]")     Sec = SplitHipfb;
      else if (Line == "[split_host]")      Sec = SplitHost;
      else if (Line == "[link_flags]")      Sec = LinkFlags;
      else if (Line == "[link_libraries]")  Sec = LinkLibraries;
      else if (Line == "[link_path]")       Sec = LinkPath;
      else if (Line == "[link_objects]")    Sec = LinkObjects;
      else if (Line == "[link_soname]")     Sec = LinkSoname;
      else Sec = None;
      continue;
    }

    switch (Sec) {
    case Meta: {
      auto Eq = Line.find('=');
      if (Eq == std::string::npos) continue;
      std::string Key = Line.substr(0, Eq);
      std::string Val = Line.substr(Eq + 1);
      if (Key == "compiler")     Cfg.ClangPath = Val;
      if (Key == "gpu_arch")     Cfg.GPUArch = Val;
      if (Key == "resource_dir") Cfg.ResourceDir = Val;
      break;
    }
    case CalleeFlags:   Cfg.CalleeFlags.push_back(Line); break;
    case KernelFlags:   Cfg.KernelFlags.push_back(Line); break;
    case BackendFlags:  Cfg.BackendFlags.push_back(Line); break;
    case CalleeSources: Cfg.CalleeSources.push_back(Line); break;
    case KernelSources: Cfg.KernelSources.push_back(Line); break;
    case HostFlags:     Cfg.HostFlags.push_back(Line); break;
    case HostSources:   Cfg.HostSources.push_back(Line); break;
    case OnerankSource: Cfg.OnerankSource = Line; break;
    case OnerankFlags:  Cfg.OnerankFlags.push_back(Line); break;
    case SplitCobj:     Cfg.SplitCobjArgs.push_back(Line); break;
    case SplitHipfb:    Cfg.SplitHipfbArgs.push_back(Line); break;
    case SplitHost:     Cfg.SplitHostArgs.push_back(Line); break;
    case LinkFlags:     Cfg.LinkFlags.push_back(Line); break;
    case LinkLibraries: Cfg.LinkLibraries.push_back(Line); break;
    case LinkPath:      Cfg.LinkPath.push_back(Line); break;
    case LinkObjects:   Cfg.LinkObjects.push_back(Line); break;
    case LinkSoname:    Cfg.LinkSoname = Line; break;
    default: break;
    }
  }

  if (Cfg.ClangPath.empty() || Cfg.GPUArch.empty() ||
      Cfg.CalleeFlags.empty() || Cfg.KernelFlags.empty()) {
    llvm::errs() << "Incomplete flag config: " << Path << "\n";
    return false;
  }

  return true;
}

struct DeviceMetadata {
  int MaxVGPR = 0, MaxAGPR = 0, MaxSGPR = 0, MaxNamedBarrier = 0;
};

struct CompileResult {
  std::string Name;
  std::string AsmText;
  std::string ErrorMsg;
  int ObjFd = -1;
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

// Apply LLVM backend options extracted from the SPLIT[asm] commands.
// These are the -mllvm flags from the CMake build (e.g.
// -amdgpu-allow-lds-in-non-entry-functions).  Must be called once
// before any backend pass execution.
static void applyBackendFlags(const std::vector<std::string> &Flags) {
  if (Flags.empty()) return;

  std::vector<const char *> Argv;
  Argv.push_back("rccl-build-server");
  for (const auto &F : Flags)
    Argv.push_back(F.c_str());

  llvm::cl::ParseCommandLineOptions(Argv.size(), Argv.data(),
                                    "RCCL build server backend options\n");

  llvm::errs() << "Backend LLVM options:";
  for (const auto &F : Flags) llvm::errs() << " " << F;
  llvm::errs() << "\n";
}

// Create a TargetMachine for the AMDGPU backend.  When a Module is
// provided, extract target-cpu and target-features from its functions
// so the backend uses the same sub-target configuration as the frontend.
static std::unique_ptr<llvm::TargetMachine>
createAMDGPUTargetMachine(llvm::StringRef GPUArch,
                          const llvm::Module *M = nullptr) {
  llvm::Triple Triple("amdgcn-amd-amdhsa");
  std::string Error;
  const llvm::Target *T = llvm::TargetRegistry::lookupTarget(Triple, Error);
  if (!T) return nullptr;

  std::string CPU = GPUArch.str();
  std::string Features;

  if (M) {
    for (const auto &F : *M) {
      if (F.isDeclaration()) continue;
      auto A = F.getFnAttribute("target-cpu");
      if (A.isStringAttribute() && !A.getValueAsString().empty())
        CPU = A.getValueAsString().str();
      auto B = F.getFnAttribute("target-features");
      if (B.isStringAttribute() && !B.getValueAsString().empty())
        Features = B.getValueAsString().str();
      break;
    }
  }

  llvm::TargetOptions Opts;
  Opts.MCOptions.AsmVerbose = true;
  return std::unique_ptr<llvm::TargetMachine>(T->createTargetMachine(
      Triple, CPU, Features, Opts, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive));
}

// ---- CC1 arg capture ----

static std::vector<std::string>
captureCC1Args(const BuildConfig &Cfg, const std::string &DummySrc,
               const std::vector<std::string> &Flags) {
  std::vector<std::string> DriverArgs;
  DriverArgs.push_back(Cfg.ClangPath);
  DriverArgs.insert(DriverArgs.end(), Flags.begin(), Flags.end());
  if (!Cfg.ResourceDir.empty()) {
    DriverArgs.push_back("-resource-dir");
    DriverArgs.push_back(Cfg.ResourceDir);
  }
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

// Replace the source file in a cc1 arg list and update associated fields.
static std::vector<std::string>
replaceSourceFile(const std::vector<std::string> &CC1Args,
                  const std::string &NewSrc) {
  std::vector<std::string> Result = CC1Args;
  int LastBare = -1;
  for (unsigned i = 0; i < Result.size(); ++i) {
    if (Result[i] == "-main-file-name" && i + 1 < Result.size()) {
      Result[++i] = llvm::sys::path::filename(NewSrc).str();
    } else if (llvm::StringRef(Result[i]).starts_with("-cuid=")) {
      uint64_t H = llvm::hash_value(llvm::StringRef(NewSrc));
      llvm::SmallString<32> Buf;
      llvm::raw_svector_ostream OS(Buf);
      OS << llvm::format_hex_no_prefix(H, 16);
      Result[i] = ("-cuid=" + Buf).str();
    } else if (!Result[i].empty() && Result[i][0] != '-') {
      LastBare = i;
    }
  }
  if (LastBare >= 0) Result[LastBare] = NewSrc;
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
  llvm::StringRef Rest = Text.substr(Pos + Pat.size()).ltrim(" ,");
  unsigned V = 0;
  Rest.consumeInteger(10, V);
  return static_cast<int>(V);
}

static DeviceMetadata extractMetadata(llvm::StringRef Asm) {
  return {
    parseIntAfter(Asm, ".set amdgpu.max_num_vgpr,"),
    parseIntAfter(Asm, ".set amdgpu.max_num_agpr,"),
    parseIntAfter(Asm, ".set amdgpu.max_num_sgpr,"),
    parseIntAfter(Asm, ".set amdgpu.max_num_named_barrier,"),
  };
}

// ---- Kernel asm patching ----
//
// Mirrors cmake/scripts/patch_kernel_metadata.cmake.  Patches the kernel
// TU's assembly so that:
//   1. Module-wide .set amdgpu.max_num_{vgpr,agpr,sgpr} reflect the callee
//      maximums (needed for correct kernel descriptors).
//   2. Forward/undefined references to those symbols inside max() expressions
//      are resolved to literal values (some assembler versions choke otherwise).
//   3. YAML .note metadata for IFC kernels (.uses_dynamic_stack: true) gets
//      register counts bumped to max(kernel_own, callee_max).

static std::string patchKernelAsm(llvm::StringRef AsmIn,
                                  const DeviceMetadata &CM) {
  std::string Asm = AsmIn.str();

  // Step 1: Replace .set definitions with callee maximums.
  auto replaceSet = [&](const char *Field, int Val) {
    std::string Pat = std::string(".set\tamdgpu.max_num_") + Field + ",";
    size_t Pos = Asm.find(Pat);
    if (Pos == std::string::npos) {
      Pat = std::string(".set amdgpu.max_num_") + Field + ",";
      Pos = Asm.find(Pat);
    }
    if (Pos == std::string::npos) return;
    size_t VS = Pos + Pat.size();
    while (VS < Asm.size() && (Asm[VS] == ' ' || Asm[VS] == '\t')) ++VS;
    size_t VE = VS;
    while (VE < Asm.size() && isdigit(static_cast<unsigned char>(Asm[VE]))) ++VE;
    Asm.replace(VS, VE - VS, std::to_string(Val));
  };
  replaceSet("vgpr", CM.MaxVGPR);
  replaceSet("agpr", CM.MaxAGPR);
  replaceSet("sgpr", CM.MaxSGPR);

  // Step 2: Resolve forward references inside max() call-sites.
  auto resolveRef = [&](const char *Field, int Val) {
    std::string Old = std::string("amdgpu.max_num_") + Field + ")";
    std::string New = std::to_string(Val) + ")";
    size_t Pos = 0;
    while ((Pos = Asm.find(Old, Pos)) != std::string::npos) {
      Asm.replace(Pos, Old.size(), New);
      Pos += New.size();
    }
  };
  resolveRef("vgpr", CM.MaxVGPR);
  resolveRef("agpr", CM.MaxAGPR);
  resolveRef("sgpr", CM.MaxSGPR);
  resolveRef("named_barrier", CM.MaxNamedBarrier);

  // Step 3: Patch YAML .note metadata for IFC kernels.
  const std::string UDS = ".uses_dynamic_stack: true";
  size_t SP = 0;
  unsigned Patched = 0;
  while ((SP = Asm.find(UDS, SP)) != std::string::npos) {
    size_t ES = Asm.rfind("  - .agpr_count:", SP);
    if (ES == std::string::npos) { SP += UDS.size(); continue; }

    size_t After = SP + UDS.size();
    size_t NE = Asm.find("  - .agpr_count:", After);
    size_t EM = Asm.find(".end_amdgpu_metadata", After);
    size_t EE = (NE != std::string::npos) ? NE
              : (EM != std::string::npos) ? EM
              : Asm.size();

    auto patchField = [&](const char *Field, int CalleeMax) {
      std::string Pat = std::string(".") + Field + ":";
      size_t FP = Asm.find(Pat, ES);
      if (FP == std::string::npos || FP >= EE) return;
      size_t VS = FP + Pat.size();
      while (VS < EE && (Asm[VS] == ' ' || Asm[VS] == '\t')) ++VS;
      size_t VE = VS;
      while (VE < EE && isdigit(static_cast<unsigned char>(Asm[VE]))) ++VE;
      if (VE == VS) return;
      int Cur = std::stoi(Asm.substr(VS, VE - VS));
      int NV = std::max(Cur, CalleeMax);
      std::string NS = std::to_string(NV);
      int Delta = static_cast<int>(NS.size()) - static_cast<int>(VE - VS);
      Asm.replace(VS, VE - VS, NS);
      EE += Delta;
    };

    patchField("agpr_count", CM.MaxAGPR);
    patchField("sgpr_count", CM.MaxSGPR);
    patchField("vgpr_count", CM.MaxVGPR);
    ++Patched;
    SP = EE;
  }
  llvm::errs() << "Patched kernel asm: V=" << CM.MaxVGPR
               << " A=" << CM.MaxAGPR << " S=" << CM.MaxSGPR
               << " B=" << CM.MaxNamedBarrier
               << " (" << Patched << " YAML entries)\n";
  return Asm;
}

// ---- Source discovery ----

static std::vector<SourceInfo>
discoverSources(const BuildConfig &Cfg) {
  std::vector<SourceInfo> S;
  for (const auto &P : Cfg.CalleeSources) {
    std::string Name = llvm::sys::path::stem(P).str();
    S.push_back({P, Name, false});
  }
  std::sort(S.begin(), S.end(),
            [](const SourceInfo &A, const SourceInfo &B) {
              return A.Name < B.Name;
            });
  for (const auto &P : Cfg.KernelSources) {
    std::string Name = llvm::sys::path::stem(
        llvm::sys::path::stem(P)).str();
    S.push_back({P, Name, true});
  }
  return S;
}

struct BuildContext {
  BuildConfig Cfg;
  ProbeConfig Probe;
  Tracer Trace;
  std::mutex PrintMu;
};

// ---- Parallel dispatch ----

static void parallelFor(unsigned N, unsigned NumThreads,
                        const std::function<void(unsigned)> &Body) {
  std::atomic<unsigned> Next{0};
  std::vector<std::thread> Threads;
  for (unsigned i = 0; i < NumThreads; ++i)
    Threads.emplace_back([&] {
      while (true) {
        unsigned Idx = Next.fetch_add(1);
        if (Idx >= N) break;
        Body(Idx);
      }
    });
  for (auto &T : Threads) T.join();
}

// ---- /dev/shm cleanup ----

static bool processAlive(pid_t Pid) {
  return kill(Pid, 0) == 0 || errno != ESRCH;
}

static void cleanStaleTmpDirs() {
  const char *Base = "/dev/shm";
  DIR *D = opendir(Base);
  if (!D) return;

  unsigned Removed = 0;
  while (struct dirent *E = readdir(D)) {
    llvm::StringRef Name(E->d_name);
    if (!Name.starts_with("rccl-build-")) continue;

    llvm::StringRef PidStr = Name.drop_front(strlen("rccl-build-"));
    unsigned long Pid = 0;
    if (PidStr.getAsInteger(10, Pid)) continue;
    if (processAlive(static_cast<pid_t>(Pid))) continue;

    std::string Dir = std::string(Base) + "/" + Name.str();
    std::error_code EC;
    for (llvm::sys::fs::directory_iterator I(Dir, EC), End; I != End && !EC;
         I.increment(EC))
      if (auto RE = llvm::sys::fs::remove(I->path()))
        llvm::errs() << "Warning: remove " << I->path() << ": " << RE.message() << "\n";
    if (auto RE = llvm::sys::fs::remove(Dir))
      llvm::errs() << "Warning: rmdir " << Dir << ": " << RE.message() << "\n";
    ++Removed;
  }
  closedir(D);

  if (Removed)
    llvm::errs() << "Cleaned " << Removed << " stale /dev/shm/rccl-build-* dirs\n";
}


// ---- Phase functions ----

// Phase 1: Parallel frontend (source→IR) + backend (IR→asm) for a range.
// Compiles Sources[Begin..End) into Results[Begin..End).
// Returns {ok_count, fail_count}.
static std::pair<unsigned, unsigned>
runPhase1(BuildContext &BC,
          const std::vector<SourceInfo> &Sources,
          std::vector<CompileResult> &Results,
          unsigned Begin, unsigned End,
          const std::vector<std::string> &CalleeCC1,
          const std::vector<std::string> &KernelCC1) {
  unsigned N = End - Begin;
  std::atomic<unsigned> DoneCount{0};

  parallelFor(N, BC.Cfg.NumThreads, [&](unsigned Rel) {
    unsigned Idx = Begin + Rel;
    auto &Src = Sources[Idx];
    auto &R = Results[Idx];
    R.Name = Src.Name;
    R.IsKernel = Src.IsKernel;

    auto CC1 = replaceSourceFile(
        Src.IsKernel ? KernelCC1 : CalleeCC1, Src.Path);

    llvm::LLVMContext Ctx;
    std::unique_ptr<llvm::Module> M;
    {
      ScopedTrace ST(BC.Trace, Src.Name, "frontend", Idx);
      M = compileCC1(Ctx, CC1);
    }

    if (!M) { R.ErrorMsg = "frontend failed"; return; }

    BC.Probe.writeIR(Src.Name, *M);

    auto TM = createAMDGPUTargetMachine(BC.Cfg.GPUArch, M.get());
    if (!TM) { R.ErrorMsg = "TargetMachine creation failed"; return; }

    {
      ScopedTrace ST(BC.Trace, Src.Name, "backend", Idx);
      if (!emitAssembly(*M, *TM, R.AsmText)) { R.ErrorMsg = "backend failed"; return; }
    }

    BC.Probe.writeAsm(Src.Name, R.AsmText);

    if (!Src.IsKernel)
      R.Meta = extractMetadata(R.AsmText);

    R.Success = true;

    unsigned Done = DoneCount.fetch_add(1) + 1;
    if (Done % 10 == 0 || Done == N) {
      std::lock_guard<std::mutex> Lock(BC.PrintMu);
      llvm::errs() << "\r  Phase 1: " << Done << "/" << N
                    << " compiled     ";
    }
  });
  llvm::errs() << "\n";

  unsigned Ok = 0, Fail = 0;
  for (unsigned i = Begin; i < End; ++i) {
    if (Results[i].Success) ++Ok; else ++Fail;
  }
  return {Ok, Fail};
}

// Phase 2: Parallel assembly (asm→obj) into memfds.
static void
runPhase2(BuildContext &BC, std::vector<CompileResult> &Results) {
  parallelFor(Results.size(), BC.Cfg.NumThreads, [&](unsigned Idx) {
    if (!Results[Idx].Success) return;

    llvm::SmallVector<char, 0> ObjBuf;
    bool OK;
    {
      ScopedTrace ST(BC.Trace, Results[Idx].Name, "assemble", Idx);
      OK = assembleToObject(Results[Idx].AsmText, BC.Cfg.GPUArch, ObjBuf);
    }
    if (!OK) {
      Results[Idx].Success = false;
      Results[Idx].ErrorMsg = "assembly failed";
      return;
    }

    int Fd = memfd_create(Results[Idx].Name.c_str(), MFD_CLOEXEC);
    if (Fd < 0) {
      Results[Idx].Success = false;
      Results[Idx].ErrorMsg = "memfd_create failed";
      return;
    }
    ssize_t W = ::write(Fd, ObjBuf.data(), ObjBuf.size());
    if (W != static_cast<ssize_t>(ObjBuf.size())) {
      ::close(Fd);
      Results[Idx].Success = false;
      Results[Idx].ErrorMsg = "memfd write failed";
      return;
    }
    Results[Idx].ObjFd = Fd;

    BC.Probe.writeObj(Results[Idx].Name, Fd);

    Results[Idx].AsmText.clear();
    Results[Idx].AsmText.shrink_to_fit();
  });
}

// Phase 3: lld -r (relocatable link) using memfds via /proc/self/fd/N.
// Writes the combined object to OutputPath.
// Returns lld exit code and combined object size via CombSizeOut.
static int
runPhase3(BuildContext &BC, std::vector<CompileResult> &Results,
          const std::string &OutputPath, uint64_t &CombSizeOut) {
  std::vector<std::string> ObjPaths;
  for (auto &R : Results) {
    if (R.ObjFd < 0) continue;
    ObjPaths.push_back("/proc/self/fd/" + std::to_string(R.ObjFd));
  }

  std::vector<const char *> LLDArgs = {"ld.lld", "-r", "-o",
                                        OutputPath.c_str()};
  for (const auto &P : ObjPaths) LLDArgs.push_back(P.c_str());

  llvm::raw_null_ostream NullOS;
  lld::Result LR;
  {
    ScopedTrace ST(BC.Trace, "lld -r", "link", Results.size());
    LR = lld::lldMain(
        LLDArgs, NullOS, llvm::errs(), {{lld::Gnu, &lld::elf::link}});
  }

  for (auto &R : Results) {
    if (R.ObjFd >= 0) { ::close(R.ObjFd); R.ObjFd = -1; }
  }

  CombSizeOut = 0;
  if (LR.retCode == 0)
    if (auto EC = llvm::sys::fs::file_size(OutputPath, CombSizeOut))
      llvm::errs() << "Warning: file_size " << OutputPath << ": "
                   << EC.message() << "\n";

  return LR.retCode;
}

// ---- Memfd helpers ----

static int writeToMemfd(const char *Name, const void *Data, size_t Size) {
  int Fd = memfd_create(Name, MFD_CLOEXEC);
  if (Fd < 0) return -1;
  ssize_t W = ::write(Fd, Data, Size);
  if (W != static_cast<ssize_t>(Size)) { ::close(Fd); return -1; }
  return Fd;
}

static int readFileToMemfd(const char *Name, const std::string &Path) {
  auto BufOrErr = llvm::MemoryBuffer::getFile(Path);
  if (!BufOrErr) return -1;
  auto &Buf = *BufOrErr;
  return writeToMemfd(Name, Buf->getBufferStart(), Buf->getBufferSize());
}

static std::string fdPath(int Fd) {
  return "/proc/self/fd/" + std::to_string(Fd);
}

// ---- x86 target machine helper ----

static std::unique_ptr<llvm::TargetMachine>
createX86TargetMachine(const llvm::Module *M = nullptr) {
  llvm::Triple Triple("x86_64-unknown-linux-gnu");
  std::string Error;
  const llvm::Target *T = llvm::TargetRegistry::lookupTarget(Triple, Error);
  if (!T) return nullptr;
  std::string CPU = "x86-64";
  std::string Features;
  if (M) {
    for (const auto &F : *M) {
      if (F.isDeclaration()) continue;
      auto A = F.getFnAttribute("target-cpu");
      if (A.isStringAttribute() && !A.getValueAsString().empty())
        CPU = A.getValueAsString().str();
      auto B = F.getFnAttribute("target-features");
      if (B.isStringAttribute() && !B.getValueAsString().empty())
        Features = B.getValueAsString().str();
      break;
    }
  }
  llvm::TargetOptions Opts;
  return std::unique_ptr<llvm::TargetMachine>(T->createTargetMachine(
      Triple, CPU, Features, Opts, llvm::Reloc::PIC_,
      std::nullopt, llvm::CodeGenOptLevel::Aggressive));
}

// ---- Phase 4: SPLIT[cobj] — lld -shared ----
// Input: combined.<arch>.o on disk (from Phase 3)
// Output: combined.<arch>.so on disk (consumed by Phase 5)

static bool
runPhase4_SplitCobj(BuildContext &BC, const std::string &InputPath,
                    const std::string &OutputPath) {
  if (BC.Cfg.SplitCobjArgs.size() < 2) {
    llvm::errs() << "Phase 4: no split_cobj config\n";
    return false;
  }

  std::vector<const char *> Args = {"ld.lld", "-shared", "-o",
                                     OutputPath.c_str(), InputPath.c_str()};
  llvm::raw_null_ostream NullOS;
  lld::Result LR;
  {
    ScopedTrace ST(BC.Trace, "lld -shared", "split_cobj", 0);
    LR = lld::lldMain(Args, NullOS, llvm::errs(),
                       {{lld::Gnu, &lld::elf::link}});
  }
  return LR.retCode == 0;
}

// ---- Phase 5: SPLIT[hipfb] — clang-offload-bundler subprocess ----
// Input: combined.<arch>.so on disk (from Phase 4)
// Output: combined.hipfb on disk (consumed by Phase 6)

static bool
runPhase5_SplitHipfb(BuildContext &BC, const std::string &SoPath,
                     const std::string &HipfbPath) {
  if (BC.Cfg.SplitHipfbArgs.size() < 2) {
    llvm::errs() << "Phase 5: no split_hipfb config\n";
    return false;
  }

  std::vector<std::string> Args;
  for (const auto &A : BC.Cfg.SplitHipfbArgs) {
    if (llvm::StringRef(A).starts_with("--input=") &&
        !llvm::StringRef(A).starts_with("--input=/dev/null"))
      Args.push_back("--input=" + SoPath);
    else if (llvm::StringRef(A).starts_with("--output="))
      Args.push_back("--output=" + HipfbPath);
    else
      Args.push_back(A);
  }

  std::vector<const char *> Ptrs;
  for (const auto &A : Args) Ptrs.push_back(A.c_str());
  Ptrs.push_back(nullptr);

  int RC;
  {
    ScopedTrace ST(BC.Trace, "offload-bundler", "split_hipfb", 0);
    pid_t Pid = fork();
    if (Pid == 0) {
      execv(Ptrs[0], const_cast<char **>(Ptrs.data()));
      _exit(127);
    }
    int Status = 0;
    waitpid(Pid, &Status, 0);
    RC = WIFEXITED(Status) ? WEXITSTATUS(Status) : 1;
  }

  if (RC != 0) {
    llvm::errs() << "Phase 5: clang-offload-bundler failed (rc=" << RC << ")\n";
    return false;
  }
  return true;
}

// ---- Phase 6: SPLIT[host] — compile host stub with embedded hipfb ----
// Input: combined.hipfb on disk (from Phase 5)
// Output: common.host.o as memfd (passed to Phase 7 final link)

static int
runPhase6_SplitHost(BuildContext &BC, const std::string &HipfbPath) {
  if (BC.Cfg.SplitHostArgs.size() < 2) {
    llvm::errs() << "Phase 6: no split_host config\n";
    return -1;
  }

  // Build driver args, substituting hipfb path and output.
  // The hipfb path appears as: -Xclang -fcuda-include-gpubinary -Xclang <path>
  // We find and replace <path> (the token containing ".hipfb").
  std::vector<std::string> DriverArgs;
  bool NextIsOutput = false;
  for (const auto &A : BC.Cfg.SplitHostArgs) {
    if (NextIsOutput) {
      DriverArgs.push_back("/dev/null");
      NextIsOutput = false;
      continue;
    }
    if (A == "-o") {
      DriverArgs.push_back(A);
      NextIsOutput = true;
      continue;
    }
    if (llvm::StringRef(A).ends_with(".hipfb") ||
        llvm::StringRef(A).contains("/combined.hipfb")) {
      DriverArgs.push_back(HipfbPath);
      continue;
    }
    DriverArgs.push_back(A);
  }

  if (!BC.Cfg.ResourceDir.empty()) {
    DriverArgs.push_back("-resource-dir");
    DriverArgs.push_back(BC.Cfg.ResourceDir);
  }

  std::vector<const char *> Ptrs;
  for (const auto &A : DriverArgs) Ptrs.push_back(A.c_str());

  clang::CreateInvocationOptions InvOpts;
  InvOpts.RecoverOnError = false;
  std::vector<std::string> CC1Args;
  InvOpts.CC1Args = &CC1Args;

  auto CI = clang::createInvocation(Ptrs, InvOpts);
  if (!CI || CC1Args.empty()) {
    llvm::errs() << "Phase 6: failed to capture cc1 args\n";
    return -1;
  }

  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> M;
  {
    ScopedTrace ST(BC.Trace, "host-stub-fe", "split_host", 0);
    M = compileCC1(Ctx, CC1Args);
  }

  if (!M) {
    llvm::errs() << "Phase 6: frontend failed\n";
    return -1;
  }

  auto TM = createX86TargetMachine(M.get());
  if (!TM) {
    llvm::errs() << "Phase 6: failed to create x86 TargetMachine\n";
    return -1;
  }

  M->setDataLayout(TM->createDataLayout());
  M->setTargetTriple(TM->getTargetTriple());

  llvm::SmallVector<char, 0> ObjBuf;
  {
    ScopedTrace ST(BC.Trace, "host-stub-be", "split_host", 0);
    llvm::raw_svector_ostream OS(ObjBuf);
    llvm::legacy::PassManager PM;
    if (TM->addPassesToEmitFile(PM, OS, nullptr,
                                llvm::CodeGenFileType::ObjectFile)) {
      llvm::errs() << "Phase 6: x86 backend setup failed\n";
      return -1;
    }
    PM.run(*M);
  }

  int ObjFd = writeToMemfd("common.host.o", ObjBuf.data(), ObjBuf.size());
  if (ObjFd < 0) {
    llvm::errs() << "Phase 6: memfd write failed\n";
    return -1;
  }
  return ObjFd;
}

// ---- Phase H: Parallel host compilation ----

struct HostCompileResult {
  std::string Name;
  int ObjFd = -1;
  bool Success = false;
  std::string ErrorMsg;
};

// Compile a single host .cc file using in-process cc1 + x86 backend.
static HostCompileResult
compileHostTU(BuildContext &BC, const std::string &SrcPath,
              const std::vector<std::string> &HostCC1, int Tid) {
  HostCompileResult R;
  R.Name = llvm::sys::path::stem(SrcPath).str();

  auto CC1 = replaceSourceFile(HostCC1, SrcPath);

  llvm::LLVMContext Ctx;
  std::unique_ptr<llvm::Module> M;
  {
    ScopedTrace ST(BC.Trace, R.Name, "host_frontend", Tid);
    M = compileCC1(Ctx, CC1);
  }
  if (!M) { R.ErrorMsg = "host frontend failed"; return R; }

  auto TM = createX86TargetMachine(M.get());
  if (!TM) { R.ErrorMsg = "x86 TM creation failed"; return R; }

  M->setDataLayout(TM->createDataLayout());
  M->setTargetTriple(TM->getTargetTriple());

  llvm::SmallVector<char, 0> ObjBuf;
  {
    ScopedTrace ST(BC.Trace, R.Name, "host_backend", Tid);
    llvm::raw_svector_ostream OS(ObjBuf);
    llvm::legacy::PassManager PM;
    if (TM->addPassesToEmitFile(PM, OS, nullptr,
                                llvm::CodeGenFileType::ObjectFile)) {
      R.ErrorMsg = "x86 backend failed";
      return R;
    }
    PM.run(*M);
  }

  R.ObjFd = writeToMemfd(R.Name.c_str(), ObjBuf.data(), ObjBuf.size());
  if (R.ObjFd < 0) { R.ErrorMsg = "memfd write failed"; return R; }
  R.Success = true;
  return R;
}

// Compile onerank.cu.cpp as a subprocess (needs full HIP pipeline).
static HostCompileResult
compileOnerankSubprocess(BuildContext &BC) {
  HostCompileResult R;
  R.Name = "onerank.cu.cpp";

  if (BC.Cfg.OnerankSource.empty()) {
    R.ErrorMsg = "no onerank source";
    return R;
  }

  // Build the full command: compiler <flags> -o <tmpfile> -c <source>
  std::vector<std::string> CmdArgs;
  CmdArgs.push_back(BC.Cfg.ClangPath);
  for (const auto &F : BC.Cfg.OnerankFlags)
    CmdArgs.push_back(F);
  if (!BC.Cfg.ResourceDir.empty()) {
    CmdArgs.push_back("-resource-dir");
    CmdArgs.push_back(BC.Cfg.ResourceDir);
  }

  // Use a tmpfile for output since subprocess can't write to our memfd
  std::string TmpObj = BC.Cfg.RcclBuild + "/split_device/host_obj/onerank.cu.cpp.o";
  CmdArgs.push_back("-o");
  CmdArgs.push_back(TmpObj);
  CmdArgs.push_back("-c");
  CmdArgs.push_back(BC.Cfg.OnerankSource);

  std::vector<const char *> Ptrs;
  for (const auto &A : CmdArgs) Ptrs.push_back(A.c_str());
  Ptrs.push_back(nullptr);

  int RC;
  {
    ScopedTrace ST(BC.Trace, "onerank.cu.cpp", "host_subprocess", 0);
    pid_t Pid = fork();
    if (Pid == 0) {
      execv(Ptrs[0], const_cast<char **>(Ptrs.data()));
      _exit(127);
    }
    int Status = 0;
    waitpid(Pid, &Status, 0);
    RC = WIFEXITED(Status) ? WEXITSTATUS(Status) : 1;
  }

  if (RC != 0) {
    R.ErrorMsg = "subprocess failed (rc=" + std::to_string(RC) + ")";
    return R;
  }

  R.ObjFd = readFileToMemfd("onerank.cu.cpp.o", TmpObj);
  if (R.ObjFd < 0) {
    R.ErrorMsg = "failed to read onerank obj into memfd";
    return R;
  }
  R.Success = true;
  return R;
}

// Run all host compilations: parallel cc1 for ~80 .cc files + onerank subprocess.
// Returns vector of memfd file descriptors for all host objects.
static std::vector<HostCompileResult>
runHostCompilation(BuildContext &BC,
                   const std::vector<std::string> &HostCC1,
                   unsigned NumThreads) {
  unsigned N = BC.Cfg.HostSources.size();
  std::vector<HostCompileResult> Results(N);
  std::atomic<unsigned> DoneCount{0};

  // onerank runs as async subprocess in parallel with the cc1 compilations
  auto OnerankFut = std::async(std::launch::async, [&]() {
    return compileOnerankSubprocess(BC);
  });

  parallelFor(N, NumThreads, [&](unsigned Idx) {
    Results[Idx] = compileHostTU(BC, BC.Cfg.HostSources[Idx], HostCC1,
                                  1000 + Idx);
    unsigned Done = DoneCount.fetch_add(1) + 1;
    if (Done % 10 == 0 || Done == N) {
      std::lock_guard<std::mutex> Lock(BC.PrintMu);
      llvm::errs() << "\r  Phase H: " << Done << "/" << N
                    << " host TUs compiled     ";
    }
  });
  llvm::errs() << "\n";

  // Collect onerank result
  auto OnerankResult = OnerankFut.get();
  Results.push_back(std::move(OnerankResult));

  return Results;
}

// ---- Phase 7: Final link — compiler driver subprocess → librccl.so ----
//
// The CMake link flags (LINK_FLAGS, LINK_LIBRARIES, etc.) are designed for the
// Clang driver, not direct LLD invocation.  We run the link as a subprocess
// through amdclang++ to get correct flag translation.
//
// Host object memfds are passed via /proc/self/fd/N (visible to the child
// after fork, since we strip MFD_CLOEXEC before exec).

static int
runPhase7_FinalLink(BuildContext &BC,
                    const std::vector<HostCompileResult> &HostResults,
                    int FatObjFd) {
  std::string OutputPath = BC.Cfg.RcclBuild + "/librccl.so.1.0";
  std::string Soname = BC.Cfg.LinkSoname.empty() ? "librccl.so.1"
                                                   : BC.Cfg.LinkSoname;

  std::vector<std::string> ArgStrs;
  ArgStrs.push_back(BC.Cfg.ClangPath);
  ArgStrs.push_back("-fPIC");
  ArgStrs.push_back("-shared");
  ArgStrs.push_back("-Wl,-soname," + Soname);
  ArgStrs.push_back("-o");
  ArgStrs.push_back(OutputPath);

  // Link path
  for (const auto &P : BC.Cfg.LinkPath)
    ArgStrs.push_back(P);

  // Link flags (skip the fat obj path — we pass it separately)
  std::string FatObjDiskPath =
      BC.Cfg.RcclBuild + "/split_device/host_obj/common.host.o";
  for (const auto &F : BC.Cfg.LinkFlags) {
    if (F == FatObjDiskPath) continue;
    ArgStrs.push_back(F);
  }

  // Host object memfds via /proc/self/fd/N
  // Collect fds that need CLOEXEC cleared for the child
  std::vector<int> ChildFds;
  for (const auto &R : HostResults) {
    if (R.ObjFd >= 0) {
      ArgStrs.push_back(fdPath(R.ObjFd));
      ChildFds.push_back(R.ObjFd);
    }
  }

  // Fat object (common.host.o from Phase 6)
  if (FatObjFd >= 0) {
    ArgStrs.push_back(fdPath(FatObjFd));
    ChildFds.push_back(FatObjFd);
  }

  // Link libraries
  for (const auto &L : BC.Cfg.LinkLibraries)
    ArgStrs.push_back(L);

  std::vector<const char *> Ptrs;
  for (const auto &A : ArgStrs)
    Ptrs.push_back(A.c_str());
  Ptrs.push_back(nullptr);

  int RC;
  {
    ScopedTrace ST(BC.Trace, "link librccl.so", "final_link", 0);
    pid_t Pid = fork();
    if (Pid == 0) {
      // Clear CLOEXEC on memfds so the child (linker) can read them
      for (int Fd : ChildFds) {
        int Flags = fcntl(Fd, F_GETFD);
        if (Flags >= 0)
          fcntl(Fd, F_SETFD, Flags & ~FD_CLOEXEC);
      }
      execv(Ptrs[0], const_cast<char **>(Ptrs.data()));
      _exit(127);
    }
    int Status = 0;
    waitpid(Pid, &Status, 0);
    RC = WIFEXITED(Status) ? WEXITSTATUS(Status) : 1;
  }

  // Close all host memfds
  for (const auto &R : HostResults) {
    if (R.ObjFd >= 0) ::close(R.ObjFd);
  }
  if (FatObjFd >= 0) ::close(FatObjFd);

  if (RC == 0) {
    std::string SoDir = BC.Cfg.RcclBuild;
    std::string SymSo1 = SoDir + "/librccl.so.1";
    std::string SymSo = SoDir + "/librccl.so";
    ::unlink(SymSo1.c_str());
    ::unlink(SymSo.c_str());
    ::symlink("librccl.so.1.0", SymSo1.c_str());
    ::symlink("librccl.so.1.0", SymSo.c_str());
  }

  return RC;
}

// ---- Main ----

int main(int argc, char **argv) {
  auto T0 = Clock::now();
  initLLVMTargets();
  cleanStaleTmpDirs();

  BuildContext BC;
  BC.Trace.setEpoch(T0);

  if (argc < 2) {
    llvm::errs() << "Usage: rccl-build-server <build_dir> [num_threads] "
                    "[trace_file]\n";
    return 1;
  }
  BC.Cfg.RcclBuild = argv[1];
  if (argc > 2) BC.Cfg.NumThreads = std::atoi(argv[2]);
  if (argc > 3) BC.Cfg.TraceFile = argv[3];

  std::string ConfigPath = BC.Cfg.RcclBuild + "/build_server_flags.conf";
  if (!loadFlagConfig(BC.Cfg, ConfigPath))
    return 1;

  applyBackendFlags(BC.Cfg.BackendFlags);

  if (BC.Cfg.NumThreads == 0)
    BC.Cfg.NumThreads = std::min(128u, std::thread::hardware_concurrency());

  BC.Probe = parseProbeEnv(BC.Cfg.RcclBuild + "/probe");

  bool FullPipeline = !BC.Cfg.HostSources.empty() && !BC.Cfg.LinkFlags.empty();

  llvm::errs() << "GPU=" << BC.Cfg.GPUArch
               << " threads=" << BC.Cfg.NumThreads
               << " trace=" << BC.Cfg.TraceFile
               << " mode=" << (FullPipeline ? "full" : "device-only") << "\n";

  auto Sources = discoverSources(BC.Cfg);
  unsigned NumCallees = BC.Cfg.CalleeSources.size();
  unsigned NumKernels = BC.Cfg.KernelSources.size();
  llvm::errs() << Sources.size() << " device TUs ("
               << NumCallees << " callee + " << NumKernels << " kernel)";
  if (FullPipeline)
    llvm::errs() << " + " << BC.Cfg.HostSources.size() << " host TUs";
  llvm::errs() << "\n";

  // Capture cc1 args once (callee and kernel variants)
  auto T_cc1 = Clock::now();

  std::string FirstCallee;
  std::string KernelSrc;
  for (const auto &S : Sources) {
    if (S.IsKernel) KernelSrc = S.Path;
    else if (FirstCallee.empty()) FirstCallee = S.Path;
  }

  auto CalleeCC1 = captureCC1Args(BC.Cfg, FirstCallee, BC.Cfg.CalleeFlags);
  auto KernelCC1 = captureCC1Args(BC.Cfg, KernelSrc, BC.Cfg.KernelFlags);

  if (CalleeCC1.empty() || KernelCC1.empty()) {
    llvm::errs() << "Failed to capture cc1 args\n";
    return 1;
  }

  llvm::errs() << "CC1 capture: "
               << llvm::format("%.0f", std::chrono::duration<double, std::milli>(
                                            Clock::now() - T_cc1).count())
               << " ms (callee: " << CalleeCC1.size()
               << " args, kernel: " << KernelCC1.size() << " args)\n";

  std::vector<CompileResult> Results(Sources.size());

  // Output paths
  std::string DevObjDir = BC.Cfg.RcclBuild + "/split_device/dev_obj";
  std::string HostObjDir = BC.Cfg.RcclBuild + "/split_device/host_obj";
  std::string FatObjDir = BC.Cfg.RcclBuild + "/split_device/fat_obj";
  for (const auto &Dir : {DevObjDir, HostObjDir, FatObjDir})
    if (auto EC = llvm::sys::fs::create_directories(Dir))
      llvm::errs() << "Warning: cannot create " << Dir << ": "
                   << EC.message() << "\n";
  std::string CombinedPath = DevObjDir + "/combined." + BC.Cfg.GPUArch + ".o";

  // --- Launch host compilation in background (Phase H) ---
  // Allocate threads: host gets 25%, device gets the rest.
  unsigned HostThreads = 0;
  unsigned DeviceThreads = BC.Cfg.NumThreads;
  std::future<std::vector<HostCompileResult>> HostFuture;
  std::vector<std::string> HostCC1;

  if (FullPipeline) {
    HostThreads = std::max(1u, BC.Cfg.NumThreads / 4);
    DeviceThreads = BC.Cfg.NumThreads - HostThreads;

    // Capture host cc1 args (with --offload-host-only)
    std::vector<std::string> HostDriverFlags = BC.Cfg.HostFlags;
    HostDriverFlags.push_back("--offload-host-only");
    std::string FirstHost = BC.Cfg.HostSources.empty()
                                ? "" : BC.Cfg.HostSources[0];
    HostCC1 = captureCC1Args(BC.Cfg, FirstHost, HostDriverFlags);

    if (HostCC1.empty()) {
      llvm::errs() << "Failed to capture host cc1 args\n";
      return 1;
    }
    llvm::errs() << "Host cc1: " << HostCC1.size() << " args, "
                 << HostThreads << " threads\n";

    HostFuture = std::async(std::launch::async, [&]() {
      return runHostCompilation(BC, HostCC1, HostThreads);
    });
  }

  // --- Phase 1a: compile callee TUs (frontend + backend) ---
  auto Saved = BC.Cfg.NumThreads;
  BC.Cfg.NumThreads = DeviceThreads;

  auto TP1a = Clock::now();
  llvm::errs() << "\n--- Phase 1a: " << NumCallees << " callees ---\n";
  auto [P1aOk, P1aFail] = runPhase1(BC, Sources, Results,
                                      0, NumCallees,
                                      CalleeCC1, KernelCC1);
  auto TP1a_end = Clock::now();

  if (P1aFail > 0) {
    for (unsigned i = 0; i < NumCallees; ++i)
      if (!Results[i].Success)
        llvm::errs() << "  FAIL: " << Sources[i].Name
                     << " (" << Results[i].ErrorMsg << ")\n";
    return 1;
  }

  // Compute global metadata from callee TUs
  DeviceMetadata GlobalMeta;
  for (unsigned i = 0; i < NumCallees; ++i) {
    if (!Results[i].Success) continue;
    GlobalMeta.MaxVGPR = std::max(GlobalMeta.MaxVGPR, Results[i].Meta.MaxVGPR);
    GlobalMeta.MaxAGPR = std::max(GlobalMeta.MaxAGPR, Results[i].Meta.MaxAGPR);
    GlobalMeta.MaxSGPR = std::max(GlobalMeta.MaxSGPR, Results[i].Meta.MaxSGPR);
    GlobalMeta.MaxNamedBarrier =
        std::max(GlobalMeta.MaxNamedBarrier, Results[i].Meta.MaxNamedBarrier);
  }

  llvm::errs() << "OK=" << P1aOk
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TP1a_end - TP1a).count())
               << "ms\nGlobalMeta: V=" << GlobalMeta.MaxVGPR
               << " A=" << GlobalMeta.MaxAGPR
               << " S=" << GlobalMeta.MaxSGPR
               << " B=" << GlobalMeta.MaxNamedBarrier << "\n";

  // --- Phase 1b: compile kernel TUs + patch asm with callee metadata ---
  auto TP1b = Clock::now();
  llvm::errs() << "\n--- Phase 1b: " << NumKernels << " kernel(s) ---\n";
  auto [P1bOk, P1bFail] = runPhase1(BC, Sources, Results,
                                      NumCallees, Sources.size(),
                                      CalleeCC1, KernelCC1);

  if (P1bFail > 0) {
    for (unsigned i = NumCallees; i < Sources.size(); ++i)
      if (!Results[i].Success)
        llvm::errs() << "  FAIL: " << Sources[i].Name
                     << " (" << Results[i].ErrorMsg << ")\n";
    return 1;
  }

  for (unsigned i = NumCallees; i < Sources.size(); ++i) {
    if (!Results[i].Success) continue;
    Results[i].AsmText = patchKernelAsm(Results[i].AsmText, GlobalMeta);
    BC.Probe.writeAsm(Results[i].Name + ".patched", Results[i].AsmText);
  }
  auto TP1b_end = Clock::now();

  llvm::errs() << "OK=" << P1bOk
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TP1b_end - TP1b).count()) << "ms\n";

  // --- Phase 2: assemble all TUs into memfds ---
  auto TP2 = Clock::now();
  runPhase2(BC, Results);
  auto TP2_end = Clock::now();

  unsigned P2Ok = 0, P2Fail = 0;
  unsigned ActiveFds = 0;
  for (auto &R : Results) {
    if (R.ObjFd >= 0) { ++P2Ok; ++ActiveFds; }
    else if (!R.Success) ++P2Fail;
  }
  llvm::errs() << "\n=== Phase 2 (asm->memfd) ===\n"
               << "OK=" << P2Ok << " FAIL=" << P2Fail
               << " memfds=" << ActiveFds
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TP2_end - TP2).count()) << "ms\n";

  if (P2Fail > 0) {
    for (auto &R : Results)
      if (!R.Success && !R.ErrorMsg.empty())
        llvm::errs() << "  FAIL: " << R.Name
                     << " (" << R.ErrorMsg << ")\n";
  }

  // --- Phase 3: lld -r via /proc/self/fd/N ---
  auto TP3 = Clock::now();
  uint64_t CombSize = 0;
  int LldRC = runPhase3(BC, Results, CombinedPath, CombSize);
  auto TP3_end = Clock::now();

  llvm::errs() << "\n=== Phase 3 (lld -r) ===\n"
               << "RC=" << LldRC
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TP3_end - TP3).count())
               << "ms Size=" << CombSize
               << "\nOutput: " << CombinedPath << "\n";

  BC.Cfg.NumThreads = Saved;

  if (LldRC != 0) {
    llvm::errs() << "Phase 3 failed, aborting.\n";
    return LldRC;
  }

  if (!FullPipeline) {
    double TotalMs =
        std::chrono::duration<double, std::milli>(Clock::now() - T0).count();
    llvm::errs() << "\n=== TOTAL: " << llvm::format("%.1f", TotalMs)
                 << "ms (" << llvm::format("%.1f", TotalMs / 1000)
                 << "s) [device-only] ===\n";

    if (BC.Trace.writeJSON(BC.Cfg.TraceFile))
      llvm::errs() << "Trace: " << BC.Cfg.TraceFile << "\n";
    return 0;
  }

  // Disk paths for SPLIT intermediates
  std::string CobjPath = DevObjDir + "/combined." + BC.Cfg.GPUArch + ".so";
  std::string HipfbPath = FatObjDir + "/combined.hipfb";

  // --- Phase 4: SPLIT[cobj] — lld -shared → combined.so ---
  auto TP4 = Clock::now();
  bool P4OK = runPhase4_SplitCobj(BC, CombinedPath, CobjPath);
  auto TP4_end = Clock::now();
  llvm::errs() << "\n=== Phase 4 (SPLIT[cobj]) ===\n"
               << (P4OK ? "OK" : "FAIL")
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TP4_end - TP4).count()) << "ms\n";
  if (!P4OK) return 1;

  // --- Phase 5: SPLIT[hipfb] — offload-bundler → combined.hipfb ---
  auto TP5 = Clock::now();
  bool P5OK = runPhase5_SplitHipfb(BC, CobjPath, HipfbPath);
  auto TP5_end = Clock::now();
  llvm::errs() << "\n=== Phase 5 (SPLIT[hipfb]) ===\n"
               << (P5OK ? "OK" : "FAIL")
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TP5_end - TP5).count()) << "ms\n";
  if (!P5OK) return 1;

  // --- Phase 6: SPLIT[host] — host stub with embedded hipfb → common.host.o memfd ---
  auto TP6 = Clock::now();
  int FatObjFd = runPhase6_SplitHost(BC, HipfbPath);
  auto TP6_end = Clock::now();
  llvm::errs() << "\n=== Phase 6 (SPLIT[host]) ===\n"
               << (FatObjFd >= 0 ? "OK" : "FAIL")
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TP6_end - TP6).count()) << "ms\n";
  if (FatObjFd < 0) return 1;

  // --- Wait for host compilation (Phase H) ---
  auto TPH = Clock::now();
  auto HostResults = HostFuture.get();
  auto TPH_end = Clock::now();

  unsigned HostOk = 0, HostFail = 0;
  for (const auto &R : HostResults) {
    if (R.Success) ++HostOk;
    else {
      ++HostFail;
      if (!R.ErrorMsg.empty())
        llvm::errs() << "  FAIL: " << R.Name << " (" << R.ErrorMsg << ")\n";
    }
  }
  llvm::errs() << "\n=== Phase H (host compile) ===\n"
               << "OK=" << HostOk << " FAIL=" << HostFail
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TPH_end - TPH).count())
               << "ms (wait, actual work ran in parallel)\n";
  if (HostFail > 0) return 1;

  // --- Phase 7: final link → librccl.so ---
  auto TP7 = Clock::now();
  int LinkRC = runPhase7_FinalLink(BC, HostResults, FatObjFd);
  auto TP7_end = Clock::now();

  std::string LibPath = BC.Cfg.RcclBuild + "/librccl.so.1.0";
  uint64_t LibSize = 0;
  llvm::sys::fs::file_size(LibPath, LibSize);

  llvm::errs() << "\n=== Phase 7 (final link) ===\n"
               << "RC=" << LinkRC
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(
                        TP7_end - TP7).count())
               << "ms Size=" << LibSize
               << "\nOutput: " << LibPath << "\n";

  double TotalMs =
      std::chrono::duration<double, std::milli>(Clock::now() - T0).count();
  llvm::errs() << "\n=== TOTAL: " << llvm::format("%.1f", TotalMs)
               << "ms (" << llvm::format("%.1f", TotalMs / 1000)
               << "s) [full pipeline] ===\n";

  if (BC.Trace.writeJSON(BC.Cfg.TraceFile))
    llvm::errs() << "Trace: " << BC.Cfg.TraceFile << "\n";

  return LinkRC;
}
