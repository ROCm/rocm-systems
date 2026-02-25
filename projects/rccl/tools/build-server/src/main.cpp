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
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <set>
#include <signal.h>
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

  void writeObj(const std::string &Name,
                const llvm::SmallVectorImpl<char> &Buf) const {
    if (!DumpObj || !shouldProbe(Name)) return;
    std::string P = Dir + "/obj/" + Name + ".o";
    std::error_code EC;
    llvm::raw_fd_ostream OS(P, EC);
    if (!EC) OS.write(Buf.data(), Buf.size());
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
    CalleeSources, KernelSources
  };
  Section Sec = None;
  std::string Line;
  while (std::getline(In, Line)) {
    if (Line.empty() || Line[0] == '#') continue;

    if (Line == "[meta]")            { Sec = Meta; continue; }
    if (Line == "[callee_flags]")    { Sec = CalleeFlags; continue; }
    if (Line == "[kernel_flags]")    { Sec = KernelFlags; continue; }
    if (Line == "[backend_flags]")   { Sec = BackendFlags; continue; }
    if (Line == "[callee_sources]")  { Sec = CalleeSources; continue; }
    if (Line == "[kernel_sources]")  { Sec = KernelSources; continue; }

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

struct TempDir {
  std::string Path;
  TempDir(llvm::StringRef Prefix)
    : Path(Prefix.str() + "-" + std::to_string(getpid())) {
    if (auto EC = llvm::sys::fs::create_directories(Path))
      llvm::errs() << "Warning: cannot create " << Path << ": "
                   << EC.message() << "\n";
  }
  ~TempDir() {
    std::error_code EC;
    for (llvm::sys::fs::directory_iterator I(Path, EC), End;
         I != End && !EC; I.increment(EC))
      if (auto RE = llvm::sys::fs::remove(I->path()))
        llvm::errs() << "Warning: remove " << I->path() << ": "
                     << RE.message() << "\n";
    if (auto RE = llvm::sys::fs::remove(Path))
      llvm::errs() << "Warning: rmdir " << Path << ": "
                   << RE.message() << "\n";
  }
  TempDir(const TempDir &) = delete;
  TempDir &operator=(const TempDir &) = delete;
};

// ---- Phase functions ----

// Phase 1: Parallel frontend (source→IR) + backend (IR→asm).
// Returns {ok_count, fail_count}.
static std::pair<unsigned, unsigned>
runPhase1(BuildContext &BC,
          const std::vector<SourceInfo> &Sources,
          std::vector<CompileResult> &Results,
          const std::vector<std::string> &CalleeCC1,
          const std::vector<std::string> &KernelCC1) {
  std::atomic<unsigned> DoneCount{0};

  parallelFor(Sources.size(), BC.Cfg.NumThreads, [&](unsigned Idx) {
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
    if (Done % 10 == 0 || Done == Sources.size()) {
      std::lock_guard<std::mutex> Lock(BC.PrintMu);
      llvm::errs() << "\r  Phase 1: " << Done << "/" << Sources.size()
                    << " compiled     ";
    }
  });
  llvm::errs() << "\n";

  unsigned Ok = 0, Fail = 0;
  for (const auto &R : Results) {
    if (R.Success) ++Ok; else ++Fail;
  }
  return {Ok, Fail};
}

// Phase 2: Parallel assembly (asm→obj).
static void
runPhase2(BuildContext &BC, std::vector<CompileResult> &Results) {
  parallelFor(Results.size(), BC.Cfg.NumThreads, [&](unsigned Idx) {
    if (!Results[Idx].Success) return;
    bool OK;
    {
      ScopedTrace ST(BC.Trace, Results[Idx].Name, "assemble", Idx);
      OK = assembleToObject(Results[Idx].AsmText, BC.Cfg.GPUArch,
                            Results[Idx].ObjBuffer);
    }
    if (!OK) {
      Results[Idx].Success = false;
      Results[Idx].ErrorMsg = "assembly failed";
    }

    BC.Probe.writeObj(Results[Idx].Name, Results[Idx].ObjBuffer);

    Results[Idx].AsmText.clear();
    Results[Idx].AsmText.shrink_to_fit();
  });
}

// Phase 3: lld -r (relocatable link).
// Returns lld exit code and combined object size via CombSizeOut.
static int
runPhase3(BuildContext &BC, const std::vector<CompileResult> &Results,
          unsigned NumTUs, uint64_t &CombSizeOut) {
  TempDir Tmp("/dev/shm/rccl-build");

  std::vector<std::string> ObjPaths;
  for (unsigned i = 0; i < Results.size(); ++i) {
    if (Results[i].ObjBuffer.empty()) continue;
    std::string P = Tmp.Path + "/" + Results[i].Name + ".o";
    std::error_code EC;
    llvm::raw_fd_ostream F(P, EC);
    if (EC) continue;
    F.write(Results[i].ObjBuffer.data(), Results[i].ObjBuffer.size());
    ObjPaths.push_back(P);
  }

  std::string Combined = Tmp.Path + "/combined.o";
  std::vector<const char *> LLDArgs = {"ld.lld", "-r", "-o", Combined.c_str()};
  for (const auto &P : ObjPaths) LLDArgs.push_back(P.c_str());

  llvm::raw_null_ostream NullOS;
  lld::Result LR;
  {
    ScopedTrace ST(BC.Trace, "lld -r", "link", NumTUs);
    LR = lld::lldMain(
        LLDArgs, NullOS, llvm::errs(), {{lld::Gnu, &lld::elf::link}});
  }

  CombSizeOut = 0;
  if (LR.retCode == 0)
    if (auto EC = llvm::sys::fs::file_size(Combined, CombSizeOut))
      llvm::errs() << "Warning: file_size " << Combined << ": "
                   << EC.message() << "\n";

  return LR.retCode;
  // ~TempDir cleans up automatically
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

  llvm::errs() << "GPU=" << BC.Cfg.GPUArch
               << " threads=" << BC.Cfg.NumThreads
               << " trace=" << BC.Cfg.TraceFile << "\n";

  auto Sources = discoverSources(BC.Cfg);
  llvm::errs() << Sources.size() << " device TUs ("
               << BC.Cfg.CalleeSources.size() << " callee + "
               << BC.Cfg.KernelSources.size() << " kernel)\n";

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

  // Phase 1: frontend + backend
  auto TP1 = Clock::now();
  auto [P1Ok, P1Fail] = runPhase1(BC, Sources, Results,
                                   CalleeCC1, KernelCC1);
  auto TP2 = Clock::now();

  DeviceMetadata GlobalMeta;
  for (unsigned i = 0; i < Sources.size(); ++i) {
    if (!Results[i].Success || Sources[i].IsKernel) continue;
    GlobalMeta.MaxVGPR = std::max(GlobalMeta.MaxVGPR, Results[i].Meta.MaxVGPR);
    GlobalMeta.MaxAGPR = std::max(GlobalMeta.MaxAGPR, Results[i].Meta.MaxAGPR);
    GlobalMeta.MaxSGPR = std::max(GlobalMeta.MaxSGPR, Results[i].Meta.MaxSGPR);
    GlobalMeta.MaxNamedBarrier =
        std::max(GlobalMeta.MaxNamedBarrier, Results[i].Meta.MaxNamedBarrier);
  }

  llvm::errs() << "\n=== Phase 1 ===\n"
               << "OK=" << P1Ok << " FAIL=" << P1Fail
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(TP2 - TP1).count())
               << "ms\nMeta: V=" << GlobalMeta.MaxVGPR
               << " A=" << GlobalMeta.MaxAGPR
               << " S=" << GlobalMeta.MaxSGPR << "\n";

  if (P1Fail > 0) {
    for (unsigned i = 0; i < Sources.size(); ++i)
      if (!Results[i].Success)
        llvm::errs() << "  FAIL: " << Sources[i].Name
                     << " (" << Results[i].ErrorMsg << ")\n";
    return 1;
  }

  // Phase 2: asm→obj
  auto TP3 = Clock::now();
  runPhase2(BC, Results);
  auto TP4 = Clock::now();

  size_t TotalObj = 0;
  for (auto &R : Results) TotalObj += R.ObjBuffer.size();
  llvm::errs() << "\n=== Phase 2 (asm->obj) ===\n"
               << "Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(TP4 - TP3).count())
               << "ms ObjTotal=" << TotalObj << "\n";

  // Phase 3: lld -r
  auto TP5 = Clock::now();
  uint64_t CombSize = 0;
  int LldRC = runPhase3(BC, Results, Sources.size(), CombSize);
  auto TP6 = Clock::now();

  llvm::errs() << "\n=== Phase 3 (lld -r) ===\n"
               << "RC=" << LldRC
               << " Wall=" << llvm::format("%.1f",
                    std::chrono::duration<double, std::milli>(TP6 - TP5).count())
               << "ms Size=" << CombSize << "\n";

  double TotalMs =
      std::chrono::duration<double, std::milli>(Clock::now() - T0).count();
  llvm::errs() << "\n=== TOTAL: " << llvm::format("%.1f", TotalMs)
               << "ms (" << llvm::format("%.1f", TotalMs / 1000) << "s) ===\n";

  if (BC.Trace.writeJSON(BC.Cfg.TraceFile))
    llvm::errs() << "Trace: " << BC.Cfg.TraceFile << "\n";

  return LldRC;
}
