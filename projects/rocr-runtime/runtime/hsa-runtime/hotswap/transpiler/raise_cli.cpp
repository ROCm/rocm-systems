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
// --enable-writelane-rewrite / --disable-writelane-rewrite. Default
// **on** (post-Triton-corpus graduation; see raiser.hpp for the full
// rationale).  Controls the post-raise rewrite of cross-widen-divergent
// `v_writelane_b32` / `v_readlane_b32` sites into per-source-wave
// `select` / `ds_bpermute` primitives — see
// `rewrite_cross_lane_divergent.{hpp,cpp}` and
// hotswap/docs/wave-size-translation.md §5.6.3.
//
// `--enable-writelane-rewrite` is accepted for backward compatibility
// (the canonical flag name used by existing lit fixtures) and is a
// no-op since the default is already on; `--disable-writelane-rewrite`
// forces the pre-rewrite path and is used by the `REFUSE` / `UNCHANGED`
// sibling RUN lines in the writelane/readlane regression fixtures to
// pin the pre-rewrite contract.  Later-wins between the two flags is
// by command-line order (last occurrence decides).
//
// --enable-wave-native / --disable-wave-native. Default **on** as
// of the WaveNative graduation. Selects `WaveNativeProjection`
// instead of `ModuloReplicationProjection` for wave32 source →
// wave64 target cross-widening. Under wave-native the kernel entry
// emits `@llvm.amdgcn.init_whole_wave` so hardware EXEC = -1 for
// the body, which:
//   * makes the WMMA → MFMA pipeline in `wmma_lowering.cpp`
//     correct on the upper half of the Wave64 target (the original
//     design motivation — see wave-size-translation.md §5.6.1);
//   * projects kernels with `num_warps > 1` correctly by giving
//     each target lane its own modeled-EXEC bit (fixes the
//     `swiglu_fp32` / `corpus_layernorm_fp32` class documented in
//     hotswap/docs/modrep-predicate-chain.md §4.3 sub-case 1);
//   * renders the C5 classifier's MODREP-specific refusal
//     rationale inapplicable — target lanes have their own
//     modeled-EXEC bits rather than sharing source wave 0's. The
//     classifier's `waveNative` gate suppresses refusal on this
//     path. For `canary_bpermute_scan_fp32`, the underlying
//     miscompile that would otherwise surface is closed by the
//     VOPD-cndmask SGPR-condition fix
//     (modrep-predicate-chain.md §6.4) rather than by the
//     projection choice itself.
//
// `--disable-wave-native` opts back into `ModuloReplicationProjection`
// for the narrow class of pointwise / independent-half kernels where
// MODREP's "replicas of source wave 0" model is correct AND where
// the C5 refusal under MODREP is the desired loud-fail signal.
// No env-var override exists; `HSA_SALMON_WAVE_NATIVE` was a
// transient test hook during the graduation sweep and has been
// removed so the opt-out path isn't silently bypassed.

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
      "[--target-isa=<arch>] [--disable-writelane-rewrite] "
      "[--disable-wave-native]\n"
      "  raise_cli <code-object.co|.hsaco> --emit-ir[=<kernel>] "
      "[--isa=<arch>] [--target-isa=<arch>] "
      "[--disable-writelane-rewrite] [--disable-wave-native]\n"
      "  raise_cli <code-object.co|.hsaco> --write-hsaco=<path> "
      "[--kernel=<name>] [--isa=<arch>] [--target-isa=<arch>] "
      "[--disable-writelane-rewrite] [--disable-wave-native]\n"
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
      "--enable-writelane-rewrite / --disable-writelane-rewrite: controls\n"
      "  the cross-widen-divergent writelane/readlane rewrite (default on;\n"
      "  see wave-size-translation.md \u00a75.6.3). The `--enable-` form is\n"
      "  kept for backward compatibility (existing REWRITE lit RUN lines);\n"
      "  `--disable-` pins the pre-rewrite REFUSE / UNCHANGED path for the\n"
      "  sibling RUN lines. Later-wins on the command line.\n"
      "--enable-wave-native / --disable-wave-native: select between\n"
      "  WaveNativeProjection (post-graduation default) and\n"
      "  ModuloReplicationProjection for wave32 source \u2192 wave64\n"
      "  target cross-widening. The `--enable-` form is kept for\n"
      "  backward compatibility; `--disable-` pins the MODREP path\n"
      "  for lit fixtures and for kernels outside WaveNative's\n"
      "  class coverage (see wave-size-translation.md \u00a7\u00a72.2 / 5.6.1\n"
      "  and modrep-predicate-chain.md \u00a76 for the graduation\n"
      "  rationale). Later-wins on the command line.\n"
      "ISA is inferred from the filename when --isa is not given.\n");
  return 2;
}

} // namespace

int main(int argc, char **argv) {
  std::string coPath;
  std::string isa;
  std::string targetIsa;
  bool emitIr = false;
  // Default on as of the Triton-corpus graduation (see this file's
  // top-of-file comment and raiser.hpp for the rationale).  The
  // `--disable-writelane-rewrite` flag (parsed below) forces the
  // pre-rewrite path for the lit fixtures that pin the
  // REFUSE / UNCHANGED sibling contracts.
  bool enableWritelaneRewrite = true;
  bool enableWaveNative = true;
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
    } else if (a == "--disable-writelane-rewrite") {
      // Later-wins on the command line: the last occurrence of an
      // --enable- / --disable- pair decides the effective value.  This
      // matches the behaviour every lit fixture's REFUSE / REWRITE RUN
      // lines implicitly rely on (one flag per RUN line).
      enableWritelaneRewrite = false;
    } else if (a == "--enable-wave-native") {
      enableWaveNative = true;
    } else if (a == "--disable-wave-native") {
      // Later-wins on the command line, symmetric with
      // --enable-/--disable-writelane-rewrite. Post-graduation the
      // default is on; --disable-wave-native is the opt-out path for
      // lit fixtures that pin MODREP-specific IR shapes (the
      // `cross_wave_warn` warn-only contract, the narrow-O1 C5
      // refusal siblings) and for producer flows that want MODREP's
      // "independent halves" throughput on pointwise kernels. See
      // this file's top-of-file comment.
      enableWaveNative = false;
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
