// Per-file raiser CLI — two modes.
//
// Usage:
//   raise_cli <code-object.co|.hsaco> [--isa=<arch>] [--target-isa=<arch>]
//   raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] [--isa=<arch>]
//                                     [--target-isa=<arch>]
//
// Default (kerneldex-coverage) mode. For each kernel in the code object,
// forks a child that runs raiseToIR so that a fatal error
// (report_fatal_error / asan trap / ...) in one kernel doesn't poison
// the whole file.  Emits one line per kernel on stdout:
//
//   OK   <kernel-name> (<lifted>/<total>)
//   FAIL <kernel-name> -> <mnemonic> [<format>]
//
// Kernels that crashed in the child (signal, non-zero exit, incomplete
// shm) are reported as a FAIL with mnemonic ``__crash__`` and format
// ``signal_<N>`` so they still land in the kerneldex worklist instead of
// being silently dropped.
//
// Exits 0 iff every kernel succeeded; otherwise 1.  ISA is auto-detected
// from the filename (look for ``gfx<digits>[a-z]?``) when ``--isa=`` is
// not passed.
//
// --emit-ir mode. Designed for lit tests. Runs raiseToIR in-process (no
// fork), dumps the raised LLVM IR for a single kernel on stdout, and
// leaves stderr alone so FileCheck can match warnings / abort-gate
// diagnostics. Selects the only kernel when the code object has one, or
// requires the ``=<kernel>`` form when there are multiple. Exits 0 iff
// the kernel raised successfully; non-zero otherwise.
//
// --target-isa=<arch>. Optional. Controls the target ISA the raiser
// lowers for; defaults to the source ISA (same-wave translation). Use
// to exercise cross-wave paths from a single CO (e.g. a gfx1250 CO
// compiled for a wave64 target).
//
// --enable-writelane-rewrite. Optional; default off. Turns on the
// post-raise rewrite of cross-widen-divergent `v_writelane_b32` /
// `v_readlane_b32` sites into per-source-wave `select` / `ds_bpermute`
// primitives — see `rewrite_cross_lane_divergent.{hpp,cpp}` and
// hotswap/docs/wave-size-translation.md §5.6.3. Provided as an
// explicit flag during the graduation rollout: lit fixtures that
// exercise the rewrite opt in per RUN line, while the
// `c1_wave_id_lift_scalarized` refusal fixture keeps the honest
// silent-miscompile gate with the flag off.
//
// --enable-wave-native. Optional; default off. Selects
// `WaveNativeProjection` instead of `ModuloReplicationProjection`
// for wave32 source → wave64 target cross-widening. Under wave-
// native the kernel entry emits `@llvm.amdgcn.init_whole_wave` so
// hardware EXEC = -1 for the body, which makes the WMMA → MFMA
// pipeline in `wmma_lowering.cpp` correct on the upper half of the
// Wave64 target. See `wave_projection.{hpp,cpp}` for the projection
// class and hotswap/docs/wave-size-translation.md §2.2 for the
// projection ladder. Lit fixtures that rely on the wave-native IR
// shape (`v_cmpx_ballot`, the four `wmma_*`) opt in per RUN line;
// kernels outside the matmul / WMMA subset stay on the modulo-
// replication default until the corpus sweep validates the
// broader flip.

#include "code_object_utils.hpp"
#include "pipeline.hpp"
#include "raiser.hpp"

// raiser.hpp forward-declares llvm::LLVMContext and llvm::Module but
// RaiseResult holds them by unique_ptr, so the destructor synthesized in
// main() needs the complete types.
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// Shared-memory block handed from each per-kernel child back to the parent.
// Using a fixed-size POD struct keeps the IPC trivially safe across fork().
struct KernelRaiseStats {
  bool done;
  bool success;
  int lifted;
  int total;
  char failMnemonic[128];
  char failFormat[64];
};

std::string autoDetectIsa(const std::string &path) {
  // Look for ``gfx<digits>[a-z]?`` anywhere in the filename.
  for (size_t i = 0; i + 3 < path.size(); ++i) {
    if (path[i] == 'g' && path[i + 1] == 'f' && path[i + 2] == 'x') {
      size_t j = i + 3;
      while (j < path.size() &&
             std::isdigit(static_cast<unsigned char>(path[j])))
        ++j;
      if (j > i + 3) {
        if (j < path.size() && path[j] >= 'a' && path[j] <= 'z')
          ++j;
        return path.substr(i, j - i);
      }
    }
  }
  return {};
}

int usage() {
  std::fprintf(
      stderr,
      "usage:\n"
      "  raise_cli <code-object.co|.hsaco> [--isa=<arch>] "
      "[--target-isa=<arch>] [--enable-writelane-rewrite] "
      "[--enable-wave-native]\n"
      "  raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] "
      "[--isa=<arch>] [--target-isa=<arch>] "
      "[--enable-writelane-rewrite] [--enable-wave-native]\n"
      "  raise_cli <code-object.co|.hsaco> --write-hsaco=<path> "
      "[--kernel=<name>] [--isa=<arch>] [--target-isa=<arch>] "
      "[--enable-writelane-rewrite] [--enable-wave-native]\n"
      "\n"
      "Default mode: emits per-kernel OK/FAIL lines on stdout in the format\n"
      "  kerneldex coverage expects. Exits 0 iff every kernel raises.\n"
      "--emit-ir mode: dumps raised LLVM IR for a single kernel on stdout.\n"
      "  No fork; stderr left alone for FileCheck.\n"
      "--write-hsaco mode: runs the full pipeline (raise + llc + lld)\n"
      "  for a single kernel and writes the produced HSACO to <path>.\n"
      "  Intended for post-rewrite disassembly triage (see\n"
      "  hotswap/docs/wave-size-translation.md \u00a75.6.3).\n"
      "--target-isa: overrides the target ISA (default: same as --isa).\n"
      "--enable-writelane-rewrite: turn on the cross-widen-divergent\n"
      "  writelane/readlane rewrite (default off; see wave-size-\n"
      "  translation.md \u00a75.6.3).\n"
      "--enable-wave-native: select WaveNativeProjection for wave32\n"
      "  source \u2192 wave64 target cross-widening (default off; see\n"
      "  wave-size-translation.md \u00a72.2).\n"
      "ISA is inferred from the filename when --isa is not given.\n");
  return 2;
}

} // namespace

int main(int argc, char **argv) {
  std::string coPath;
  std::string isa;
  std::string targetIsa;
  bool emitIr = false;
  bool enableWritelaneRewrite = false;
  bool enableWaveNative = false;
  std::string emitIrKernel;
  std::string writeHsacoPath;
  std::string writeHsacoKernel;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--isa=", 0) == 0) {
      isa = a.substr(6);
    } else if (a == "--isa") {
      if (i + 1 >= argc)
        return usage();
      isa = argv[++i];
    } else if (a.rfind("--target-isa=", 0) == 0) {
      targetIsa = a.substr(13);
    } else if (a == "--target-isa") {
      if (i + 1 >= argc)
        return usage();
      targetIsa = argv[++i];
    } else if (a == "--emit-ir") {
      emitIr = true;
    } else if (a.rfind("--emit-ir=", 0) == 0) {
      emitIr = true;
      emitIrKernel = a.substr(10);
    } else if (a.rfind("--write-hsaco=", 0) == 0) {
      writeHsacoPath = a.substr(14);
    } else if (a.rfind("--kernel=", 0) == 0) {
      writeHsacoKernel = a.substr(9);
    } else if (a == "--enable-writelane-rewrite") {
      enableWritelaneRewrite = true;
    } else if (a == "--enable-wave-native") {
      enableWaveNative = true;
    } else if (!a.empty() && a[0] == '-') {
      std::fprintf(stderr, "raise_cli: unknown flag: %s\n", a.c_str());
      return usage();
    } else if (coPath.empty()) {
      coPath = a;
    } else {
      std::fprintf(stderr, "raise_cli: unexpected positional arg: %s\n",
                   a.c_str());
      return usage();
    }
  }
  if (coPath.empty())
    return usage();

  // Read the file up-front so we can fall back to the ELF e_flags
  // ISA when the filename heuristic fails (kerneldex corpora often
  // store kernels under hashed names with no `gfx*` substring; the
  // ELF MACH field is the only deterministic source).
  auto coData = transpiler::readFile(coPath);
  if (coData.empty()) {
    std::fprintf(stderr, "raise_cli: cannot read %s\n", coPath.c_str());
    return 2;
  }

  if (isa.empty()) {
    isa = autoDetectIsa(coPath);
    if (isa.empty())
      isa = transpiler::detectIsaFromElf(coData);
    if (isa.empty()) {
      std::fprintf(stderr,
                   "raise_cli: could not infer ISA from %s; pass --isa=<arch>\n",
                   coPath.c_str());
      return 2;
    }
  }

  auto kernelNames = transpiler::listKernelNames(coData);
  if (kernelNames.empty()) {
    std::fprintf(stderr, "raise_cli: no kernels in %s\n", coPath.c_str());
    return 2;
  }

  auto text = transpiler::extractTextSection(coData);
  if (!text.valid) {
    std::fprintf(stderr, "raise_cli: could not extract .text from %s\n",
                 coPath.c_str());
    return 2;
  }

  // --emit-ir path — no fork, no stderr redirect. Used by lit tests that
  // FileCheck the raised IR on stdout and the raiser diagnostics on
  // stderr. One kernel per invocation.
  if (emitIr) {
    std::string target;
    if (emitIrKernel.empty()) {
      if (kernelNames.size() != 1) {
        std::fprintf(stderr,
                     "raise_cli: --emit-ir requires =<kernel> when the "
                     "code object has %zu kernels\n",
                     kernelNames.size());
        return 2;
      }
      target = kernelNames.front();
    } else {
      bool found = false;
      for (const auto &kn : kernelNames)
        if (kn == emitIrKernel) {
          target = kn;
          found = true;
          break;
        }
      if (!found) {
        std::fprintf(stderr,
                     "raise_cli: kernel '%s' not found in %s\n",
                     emitIrKernel.c_str(), coPath.c_str());
        return 2;
      }
    }
    auto meta = transpiler::extractKernelMeta(coData, target);
    uint64_t kernelOffset =
        transpiler::findKernelSymbolOffset(coData, target);
    auto raised = transpiler::raiseToIR(text.bytes, isa, target, meta,
                                        kernelOffset, targetIsa,
                                        enableWritelaneRewrite,
                                        enableWaveNative);
    if (!raised.success) {
      // Contract: raiseToIR only populates RaiseResult::irText on the
      // success path (the last write before setting `success = true`),
      // so we cannot dump partial IR here. Callers that need stderr
      // diagnostics (abort-gate lit tests, etc.) FileCheck the raiser's
      // stderr — we leave that untouched.
      std::fprintf(stderr,
                   "raise_cli: kernel '%s' failed to raise: %s [%s]"
                   " @offset=0x%llx%s%s\n",
                   target.c_str(),
                   raised.failure.mnemonic.empty()
                       ? "unknown"
                       : raised.failure.mnemonic.c_str(),
                   raised.failure.format.empty()
                       ? "unknown"
                       : raised.failure.format.c_str(),
                   static_cast<unsigned long long>(raised.failure.offset),
                   raised.failure.detail.empty() ? "" : " :: ",
                   raised.failure.detail.empty()
                       ? ""
                       : raised.failure.detail.c_str());
      return 1;
    }
    std::fwrite(raised.irText.data(), 1, raised.irText.size(), stdout);
    return 0;
  }

  // --write-hsaco path — runs the full pipeline (raise + llc + lld)
  // for a single kernel and writes the resulting HSACO to disk.
  // Triage-mode only: lets downstream tools (llvm-objdump) inspect the
  // exact bytes the gtest harness would launch, so we can walk the
  // Phase 6.5 rewrite end-to-end through the final ISA.
  if (!writeHsacoPath.empty()) {
    std::string target;
    if (writeHsacoKernel.empty()) {
      if (kernelNames.size() != 1) {
        std::fprintf(stderr,
                     "raise_cli: --write-hsaco requires --kernel=<name> when "
                     "the code object has %zu kernels\n",
                     kernelNames.size());
        return 2;
      }
      target = kernelNames.front();
    } else {
      bool found = false;
      for (const auto &kn : kernelNames)
        if (kn == writeHsacoKernel) {
          target = kn;
          found = true;
          break;
        }
      if (!found) {
        std::fprintf(stderr,
                     "raise_cli: kernel '%s' not found in %s\n",
                     writeHsacoKernel.c_str(), coPath.c_str());
        return 2;
      }
    }
    std::string effectiveTargetIsa = targetIsa.empty() ? isa : targetIsa;
    auto pipe = transpiler::runPipeline(coData, isa, effectiveTargetIsa,
                                        target, enableWritelaneRewrite,
                                        enableWaveNative);
    if (!pipe.success) {
      std::fprintf(stderr,
                   "raise_cli: pipeline failed for kernel '%s' (lifted=%d/%d, "
                   "failMnemonic='%s')\n",
                   target.c_str(), pipe.liftedCount, pipe.totalCount,
                   pipe.failMnemonic.c_str());
      return 1;
    }
    FILE *fp = std::fopen(writeHsacoPath.c_str(), "wb");
    if (!fp) {
      std::fprintf(stderr, "raise_cli: cannot open %s for writing\n",
                   writeHsacoPath.c_str());
      return 2;
    }
    size_t wrote =
        std::fwrite(pipe.hsaco.data(), 1, pipe.hsaco.size(), fp);
    std::fclose(fp);
    if (wrote != pipe.hsaco.size()) {
      std::fprintf(stderr,
                   "raise_cli: short write to %s (%zu of %zu bytes)\n",
                   writeHsacoPath.c_str(), wrote, pipe.hsaco.size());
      return 2;
    }
    std::fprintf(stderr,
                 "raise_cli: wrote %zu byte HSACO for kernel '%s' to %s "
                 "(lifted %d/%d)\n",
                 pipe.hsaco.size(), target.c_str(), writeHsacoPath.c_str(),
                 pipe.liftedCount, pipe.totalCount);
    return 0;
  }

  int totalKernels = 0, okKernels = 0, failKernels = 0, crashKernels = 0;

  for (auto &kName : kernelNames) {
    ++totalKernels;
    auto *shm = static_cast<KernelRaiseStats *>(
        mmap(nullptr, sizeof(KernelRaiseStats), PROT_READ | PROT_WRITE,
             MAP_SHARED | MAP_ANONYMOUS, -1, 0));
    if (shm == MAP_FAILED) {
      std::fprintf(stderr, "raise_cli: mmap failed\n");
      return 3;
    }
    std::memset(shm, 0, sizeof(KernelRaiseStats));

    pid_t pid = fork();
    if (pid == 0) {
      // Silence the child's stderr: LLVM chatters a lot, and kerneldex
      // only cares about OK/FAIL on stdout plus the last stderr line
      // when the process as a whole crashes.
      int devnull = open("/dev/null", O_WRONLY);
      if (devnull >= 0) {
        dup2(devnull, STDERR_FILENO);
        close(devnull);
      }
      auto meta = transpiler::extractKernelMeta(coData, kName);
      auto raised = transpiler::raiseToIR(text.bytes, isa, kName, meta,
                                          /*kernelOffset=*/0, targetIsa,
                                          enableWritelaneRewrite,
                                          enableWaveNative);
      shm->done = true;
      shm->success = raised.success;
      shm->lifted = raised.liftedCount;
      shm->total = raised.totalCount;
      if (!raised.success) {
        const char *mn = raised.failure.mnemonic.empty()
                             ? "unknown"
                             : raised.failure.mnemonic.c_str();
        const char *fmt = raised.failure.format.empty()
                              ? "unknown"
                              : raised.failure.format.c_str();
        std::strncpy(shm->failMnemonic, mn, sizeof(shm->failMnemonic) - 1);
        std::strncpy(shm->failFormat, fmt, sizeof(shm->failFormat) - 1);
      }
      _exit(0);
    }

    int st = 0;
    waitpid(pid, &st, 0);

    if (!shm->done || WIFSIGNALED(st) || (WIFEXITED(st) && WEXITSTATUS(st) != 0)) {
      // Child never wrote the shm marker, or died by signal, or exited
      // with a nonzero status: surface this as a FAIL row with a
      // synthetic mnemonic so kerneldex still counts the kernel.
      ++crashKernels;
      int sig = WIFSIGNALED(st) ? WTERMSIG(st) : 0;
      std::printf("FAIL %s -> __crash__ [signal_%d]\n", kName.c_str(), sig);
    } else if (shm->success) {
      ++okKernels;
      std::printf("OK %s (%d/%d)\n", kName.c_str(), shm->lifted, shm->total);
    } else {
      ++failKernels;
      std::printf("FAIL %s -> %s [%s]\n", kName.c_str(), shm->failMnemonic,
                  shm->failFormat);
    }

    munmap(shm, sizeof(KernelRaiseStats));
  }

  std::fprintf(stderr,
               "raise_cli: %d kernels, %d ok, %d fail, %d crash (%s)\n",
               totalKernels, okKernels, failKernels, crashKernels,
               coPath.c_str());

  return (failKernels + crashKernels) == 0 ? 0 : 1;
}
