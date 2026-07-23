////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include <libelf.h>
#include <limits.h>
#if defined(__linux__)
#include <link.h>
#include <unistd.h>
#else
#include <cstdint>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <atomic>
#include <fstream>
#include <mutex>
#include "inc/amd_hsa_elf.h"
#include "inc/amd_hsa_kernel_code.h"
#include "core/inc/agent.h"
#include "core/inc/amd_hsa_code.hpp"
#include "core/inc/amd_loader_context.hpp"
#include "core/inc/hotswap_env.hpp"
#include "amd_hsa_code_util.hpp"
#include "amd_options.hpp"
#include "core/util/utils.h"

#include "executable.hpp"

#include "AMDHSAKernelDescriptor.h"

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
#include "amd_comgr.h"
#include "hotswap_kernel_registry.h"
#endif

using namespace rocr::amd::hsa;
using namespace rocr::amd::hsa::common;

// r_version history:
// 1: Initial debug protocol
// 2: New trap handler ABI. The reason for halting a wave is recorded in ttmp11[8:7].
// 3: New trap handler ABI. A wave halted at S_ENDPGM rewinds its PC by 8 bytes, and sets ttmp11[9]=1.
// 4: New trap handler ABI. Save the trap id in ttmp11[16:9]
// 5: New trap handler ABI. Save the PC in ttmp11[22:7] ttmp6[31:0], and park the wave if stopped
// 6: New trap handler ABI. ttmp6[25:0] contains dispatch index modulo queue size
// 7: New trap handler ABI. Send interrupts as a bitmask, coalescing concurrent exceptions.
// 8: New trap handler ABI. for gfx942: Initialize ttmp[4:5] if ttmp11[31] == 0.
// 9: New trap handler ABI. For gfx11: Save PC in ttmp11[22:7] ttmp6[31:0], and park the wave if stopped.
// 10: New trap handler ABI. Set status.skip_export when halting the wave.
//                           For gfx942, set ttmp6[31] = 0 if ttmp11[31] == 0.

HSA_API r_debug _amdgpu_r_debug;
static __forceinline link_map*& r_debug_tail() {
  static link_map* r_debug_tail_ = nullptr;
  return r_debug_tail_;
}

namespace rocr {
  // Having a side effect prevents call site optimization that allows removal of a noinline function call
  // with no side effect.
#if defined(__linux__)
  __attribute__((noinline))
#else
  __declspec(noinline)
#endif
void _loader_debug_state() {
  static volatile int function_needs_a_side_effect = 0;
  function_needs_a_side_effect ^= 1;
}

namespace amd {
namespace hsa {
namespace loader {

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
static std::string JsonEscape(const std::string& s);
static void AppendHotSwapProofJson(const std::string& fields);
static void RegisterHotSwapKernelObject(uint64_t address, const std::string& name,
                                        HotSwapKernelKind kind);

// A source-ISA code object whose translation is deferred until first dispatch.
// It retains everything COMGR needs to translate an individual kernel later:
// the source ELF, the source/target ISA names, and the target agent/profile.
struct HotSwapLazyCodeObject {
  std::vector<uint8_t> SourceElf;
  std::string SourceIsa;
  std::string TargetIsa;
  std::string SourceGfx;
  std::string TargetGfx;
  uint8_t SourceMach = 0;
  hsa_agent_t Agent = {};
  hsa_profile_t Profile = HSA_PROFILE_FULL;
};

namespace {

HotSwapKernelRegistry& HotSwapKernelRegistryInstance() {
  static auto* Registry = new HotSwapKernelRegistry();
  return *Registry;
}

std::vector<std::shared_ptr<ExecutableImpl>>&
HotSwapRuntimeLifetimeTranslatedExecutables() {
  // Translated one-kernel code objects back AQL packet kernel descriptors.
  // Queue packets and runtime teardown can outlive the source loaded-code-object
  // destruction point, so these private target code objects are owned for the
  // runtime lifetime instead of by source-symbol registry entries.
  static auto* Executables = new std::vector<std::shared_ptr<ExecutableImpl>>();
  return *Executables;
}

std::mutex& HotSwapRuntimeLifetimeTranslatedExecutablesMutex() {
  static auto* Mutex = new std::mutex();
  return *Mutex;
}

// Lazy target executables are owned by HotSwap's runtime-lifetime list, not by
// AmdHsaCodeLoader::executables, so give them an ID outside the loader index
// namespace.
constexpr size_t HotSwapRuntimeTranslatedExecutableId = static_cast<size_t>(-1);

thread_local HotSwapKernelKind CurrentHotSwapKernelKind =
    HotSwapKernelKind::Untranslated;
thread_local std::shared_ptr<HotSwapLazyCodeObject> CurrentHotSwapLazyCodeObject;
thread_local bool CurrentHotSwapLoadingTranslatedTarget = false;

void RegisterHotSwapLoadedKernelObject(uint64_t address,
                                       const std::string& name) {
  if (!ShouldCheckHotSwapDispatchKernelObjects()) return;
  RegisterHotSwapKernelObject(address, name, CurrentHotSwapKernelKind);
}

}  // namespace

static void RegisterHotSwapKernelObject(uint64_t address, const std::string& name,
                                        HotSwapKernelKind kind) {
  HotSwapKernelRegistryInstance().RegisterKernelObject(
      address, name, kind, CurrentHotSwapLazyCodeObject);
}

// Registers ROCr's own blit shaders, which are already built for the execution
// ISA, as target kernels so the dispatch intercept passes them through unchanged
// instead of trying to translate them.
void RegisterHotSwapRocrBlitTargetKernelObject(uint64_t address,
                                               size_t size,
                                               const char* target_gfx) {
  if (!ShouldCheckHotSwapDispatchKernelObjects()) return;
  std::string TargetGfx = target_gfx ? std::string(target_gfx) : std::string();
  const std::string Name = "rocr_blit";
  const std::string SourceKind = "rocr_blit";
  const std::string Reason = "embedded_rocr_target_blit_shader";
  const std::string InvalidReason =
      "rocr_blit target ISA or allocation could not be identified";
  std::shared_ptr<HotSwapKernelRecord> Record =
      HotSwapKernelRegistryInstance().RegisterRocrBlitTargetKernelObject(
          address, size, TargetGfx, Reason, InvalidReason);
  const HotSwapKernelKind Kind =
      Record->Kind.load(std::memory_order_acquire);
  const bool HasTargetIsa = Kind == HotSwapKernelKind::RuntimeTargetInternal;

  std::ostringstream proof;
  proof << "\"event\":\"hotswap_runtime_target_internal_registered\""
        << ",\"kernel_name\":\"" << JsonEscape(Name) << "\""
        << ",\"kernel_object\":\"0x" << std::hex << address << "\""
        << std::dec
        << ",\"object_size\":" << size
        << ",\"kind\":\"" << HotSwapKernelKindName(
                                 HasTargetIsa ? HotSwapKernelKind::RuntimeTargetInternal
                                              : HotSwapKernelKind::Untranslated)
        << "\""
        << ",\"source_kind\":\"" << JsonEscape(SourceKind) << "\"";
  if (!TargetGfx.empty())
    proof << ",\"target_gfx\":\"" << JsonEscape(TargetGfx) << "\"";
  proof << ",\"reason\":\""
        << JsonEscape(HasTargetIsa ? Reason
                                   : InvalidReason)
        << "\"";
  AppendHotSwapProofJson(proof.str());
}

void UnregisterHotSwapRocrBlitTargetKernelObject(uint64_t address) {
  if (!ShouldCheckHotSwapDispatchKernelObjects()) return;
  HotSwapKernelRegistryInstance().UnregisterRocrBlitTargetKernelObject(address);
}

static void UnregisterHotSwapKernelObject(uint64_t address) {
  HotSwapKernelRegistryInstance().UnregisterLoadedKernelObject(address);
}

static std::shared_ptr<HotSwapKernelRecord> LookupHotSwapKernelRecord(uint64_t address) {
  return HotSwapKernelRegistryInstance().Lookup(address);
}

static void UpdateHotSwapKernelObjectLaunchMetadata(uint64_t address,
                                                    uint32_t groupSegmentSize,
                                                    uint32_t privateSegmentSize) {
  HotSwapKernelRegistryInstance().UpdateLaunchMetadata(
      address, groupSegmentSize, privateSegmentSize);
}

// Master gate for the dispatch-time intercept: true only in presentation mode
// (HSA_HOTSWAP_PRESENT_ISA set and HotSwap not disabled). When false every
// HotSwap check below short-circuits and the runtime behaves as it does today.
bool ShouldCheckHotSwapDispatchKernelObjects() {
  return rocr::hotswap::IsPresentationModeEnabled();
}

#endif

class LoaderOptions {
public:
  explicit LoaderOptions(std::ostream &error = std::cerr);

  const amd::options::NoArgOption* Help() const { return &help; }
  const amd::options::NoArgOption* DumpCode() const { return &dump_code; }
  const amd::options::NoArgOption* DumpIsa() const { return &dump_isa; }
  const amd::options::NoArgOption* DumpExec() const { return &dump_exec; }
  const amd::options::NoArgOption* DumpAll() const { return &dump_all; }
  const amd::options::ValueOption<std::string>* DumpDir() const { return &dump_dir; }
  const amd::options::PrefixOption* Substitute() const { return &substitute; }

  bool TrampolineEnabled() const { return trampoline_enabled_; }
  bool TrampolineNoWaEnabled() const { return trampoline_no_wa_enabled_; }

  bool ParseOptions(const std::string& options);
  void Reset();
  void PrintHelp(std::ostream& out) const;

private:
  /// @brief Copy constructor - not available.
  LoaderOptions(const LoaderOptions&);

  /// @brief Assignment operator - not available.
  LoaderOptions& operator=(const LoaderOptions&);

  amd::options::NoArgOption help;
  amd::options::NoArgOption dump_code;
  amd::options::NoArgOption dump_isa;
  amd::options::NoArgOption dump_exec;
  amd::options::NoArgOption dump_all;
  amd::options::ValueOption<std::string> dump_dir;
  amd::options::PrefixOption substitute;
  amd::options::OptionParser option_parser;
  bool trampoline_enabled_ = false;
  bool trampoline_no_wa_enabled_ = false;
};

LoaderOptions::LoaderOptions(std::ostream& error) :
  help("help", "print help"),
  dump_code("dump-code", "Dump finalizer output code object"),
  dump_isa("dump-isa", "Dump finalizer output to ISA text file"),
  dump_exec("dump-exec", "Dump executable to text file"),
  dump_all("dump-all", "Dump all finalizer input and output (as above)"),
  dump_dir("dump-dir", "Dump directory"),
  substitute("substitute", "Substitute code object with given index or index range on loading from file"),
  option_parser(false, error)
{
  option_parser.AddOption(&help);
  option_parser.AddOption(&dump_code);
  option_parser.AddOption(&dump_isa);
  option_parser.AddOption(&dump_exec);
  option_parser.AddOption(&dump_all);
  option_parser.AddOption(&dump_dir);
  option_parser.AddOption(&substitute);

  // LOADER_ENABLE_TRAMPOLINE=1: enable gfx125x kernel-entry trampolines.
  // LOADER_ENABLE_TRAMPOLINE_NO_WA=1: enable trampolines without the global_wb
  // cache-writeback workaround (s_mov + s_set_pc only). Trampolines are disabled
  // by default; these env vars are for testing only.
  const char* enable_trampoline = getenv("LOADER_ENABLE_TRAMPOLINE");
  if (enable_trampoline && std::strcmp(enable_trampoline, "1") == 0) {
    trampoline_enabled_ = true;
  }
  const char* enable_trampoline_no_wa = getenv("LOADER_ENABLE_TRAMPOLINE_NO_WA");
  if (enable_trampoline_no_wa && std::strcmp(enable_trampoline_no_wa, "1") == 0) {
    trampoline_no_wa_enabled_ = true;
  }
}

bool LoaderOptions::ParseOptions(const std::string& options)
{
  return option_parser.ParseOptions(options.c_str());
}

void LoaderOptions::Reset()
{
  option_parser.Reset();
}

void LoaderOptions::PrintHelp(std::ostream& out) const
{
  option_parser.PrintHelp(out);
}

static const char *LOADER_DUMP_PREFIX = "amdcode";

// Kernel-entry trampoline (gfx125x / RDNA4).
//
// We cannot reserve space immediately in front of each kernel entry: that would
// require a non-uniform relayout of the loaded code segment, which breaks every
// intra-segment PC-relative reference the compiler baked in. Instead we allocate
// a separate *executable* region (AMDGPU_HSA_SEGMENT_CODE_AGENT, which carries
// AllocateExecutable in the loader context) and, for each kernel, emit a stub
// that jumps to the real entry; the kernel descriptor's entry offset is then
// rewritten so dispatch lands in the stub first.
//
// The jump is absolute (the pool is not within S_BRANCH range of the code), so
// the stub does a global cache writeback (SCOPE_CU) and a v_nop, then loads the
// 64-bit entry address into a scratch SGPR pair and sets PC.
// s[100:101] is a safe fixed scratch: RDNA gives every wave 128 physical SGPRs and
// these indices are well above the preloaded user+system SGPRs (<= ~20), so they
// are never a live kernel input -- the kernel writes them before it reads them.
//
// gfx1250 encodings verified with: llvm-mc --arch=amdgcn --mcpu=gfx1250 --show-encoding
//   global_wb   <scope:SCOPE_CU>       ->   0xEE0B007C, 0x00000000, 0x00000000
//   v_nop        (padding)              ->  0x7E000000
//   s_mov_b32    s100, <lit> + literal  ->  0xBEE400FF
//   s_mov_b32    s101, <lit> + literal  ->  0xBEE500FF
//   s_set_pc_i64 s[100:101]             ->  0xBE804864
//   s_code_end   (padding)              ->  0xBF9F0000
static constexpr size_t kTrampolineStubStride =
    AMD_ISA_ALIGN_BYTES;  // 256: one stub, entry-aligned

// The CP (CPC) instruction-prefetches forward from a kernel's entry PC when it
// dispatches. Because dispatch now lands on a stub inside our pool, that prefetch
// reads ahead from the stub and would run off the end of the pool into the next,
// unmapped page -- a CPC read page/permission fault (observed on gfx1250). The
// prefetch length is per-kernel: COMPUTE_PGM_RSRC3.INST_PREF_SIZE (6 bits, GFX11+)
// counts 128-byte instruction-cache lines to prefetch ahead of the entry. We size
// a trailing guard from the largest INST_PREF_SIZE in the pool so the prefetch from
// any stub always lands in mapped, readable memory inside this same allocation. The
// guard is never executed (the stub sets PC away first); it only needs to be present
// and readable, which the allocation's zero-fill already guarantees.
static constexpr size_t kInstPrefUnitBytes = 128;  // GFX11+ CP I$ prefetch line size

// The GFX1250 unclaused-VMEM workaround prologue (llvm PR #208467). The compiler
// emits these 4 dwords -- global_wb <scope:SCOPE_CU> followed by v_nop -- at every
// hardware kernel entry so the first VMEM instruction is unclaused. It is exactly
// the sequence the entry trampoline itself prepends, so when a kernel's entry
// already begins with it the trampoline would only duplicate the workaround.
static constexpr uint32_t kGfx1250UnclausedVmemPrologue[4] = {
    0xEE0B007C,  // global_wb <scope:SCOPE_CU>
    0x00000000,  // :
    0x00000000,  // :
    0x7E000000,  // v_nop
};

static void BuildTrampolineGfx1250(uint8_t* buf, uint64_t target) {
  auto* w = reinterpret_cast<uint32_t*>(buf);

  w[0] = kGfx1250UnclausedVmemPrologue[0];  // global_wb <scope:SCOPE_CU>
  w[1] = kGfx1250UnclausedVmemPrologue[1];  // :
  w[2] = kGfx1250UnclausedVmemPrologue[2];  // :
  w[3] = kGfx1250UnclausedVmemPrologue[3];  // v_nop (padding)
  w[4] = 0xBEE400FF;  // s_mov_b32 s100, target_lo
  w[5] = static_cast<uint32_t>(target);
  w[6] = 0xBEE500FF;  // s_mov_b32 s101, target_hi
  w[7] = static_cast<uint32_t>(target >> 32);
  w[8] = 0xBE804864;  // s_set_pc_i64 s[100:101]
  for (size_t i = 9; i < kTrampolineStubStride / sizeof(uint32_t); ++i)
    w[i] = 0xBF9F0000;  // s_code_end (prefetch-safe padding)
}

// Minimal stub: load the 64-bit entry address into s[100:101] and set PC, with no
// global_wb / v_nop cache-writeback workaround.
static void BuildTrampolineGfx1250NoWa(uint8_t* buf, uint64_t target) {
  auto* w = reinterpret_cast<uint32_t*>(buf);

  w[0] = 0xBEE400FF;  // s_mov_b32 s100, target_lo
  w[1] = static_cast<uint32_t>(target);
  w[2] = 0xBEE500FF;  // s_mov_b32 s101, target_hi
  w[3] = static_cast<uint32_t>(target >> 32);
  w[4] = 0xBE804864;  // s_set_pc_i64 s[100:101]
  for (size_t i = 5; i < kTrampolineStubStride / sizeof(uint32_t); ++i)
    w[i] = 0xBF9F0000;  // s_code_end (prefetch-safe padding)
}

// gfx12.5 family: CO v3+ reports either a generic mach name (gfx12-5-generic) or
// discrete targets (gfx1250, gfx1251, …) in the amdgcn-amd-amdhsa--<target> ISA string.
static bool CodeObjectIsaIsGfx125Family(const std::string& codeIsa) {
  if (codeIsa.find("gfx12-5-generic") != std::string::npos) return true;
  return codeIsa.find("gfx125") != std::string::npos;
}

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
static std::string JsonEscape(const std::string& s) {
  std::ostringstream os;
  for (char c : s) {
    switch (c) {
      case '\\':
        os << "\\\\";
        break;
      case '"':
        os << "\\\"";
        break;
      case '\n':
        os << "\\n";
        break;
      case '\r':
        os << "\\r";
        break;
      case '\t':
        os << "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          os << "\\u" << std::hex << std::setw(4) << std::setfill('0')
             << static_cast<unsigned>(static_cast<unsigned char>(c)) << std::dec
             << std::setfill(' ');
        } else {
          os << c;
        }
    }
  }
  return os.str();
}

// Proof logging is a process-run decision: cache the selected path so all proof
// emission uses the same destination even if the environment changes later.
static const std::string& HotSwapProofLogPath() {
  static const std::string Path = [] {
    constexpr char EnvName[] = "HSA_HOTSWAP_PROOF_LOG";
    if (!rocr::os::IsEnvVarSet(EnvName)) return std::string();
    return rocr::os::GetEnvVar(EnvName);
  }();
  return Path;
}

// Keep call sites cheap and explicit without repeating the path lookup rule.
static bool HotSwapProofLogEnabled() {
  return !HotSwapProofLogPath().empty();
}

// Appends one JSON object (the given comma-separated fields) as a line to the
// file named by HSA_HOTSWAP_PROOF_LOG. Best-effort and inert when unset; used to
// prove which kernels were translated/dispatched during a run.
static void AppendHotSwapProofJson(const std::string& jsonFields) {
  const std::string& Path = HotSwapProofLogPath();
  if (Path.empty()) return;
  static std::mutex ProofLogMutex;
  std::lock_guard<std::mutex> Guard(ProofLogMutex);
  static std::ofstream Out;
  if (!Out.is_open()) {
    Out.clear();
    Out.open(Path, std::ios::app);
  }
  if (!Out) {
    std::cerr << "hotswap: failed to open proof log '" << Path
              << "' for append\n";
    return;
  }
  Out << "{" << jsonFields << "}\n";
  Out.flush();
  if (!Out) {
    std::cerr << "hotswap: failed to write proof log '" << Path << "'\n";
    Out.close();
    Out.clear();
  }
}

// Returns the trailing gfx processor token (e.g. "gfx1250") from an ISA name, or
// "" if none is present.
static std::string ExtractGfxName(const std::string& isa) {
  size_t pos = isa.rfind("gfx");
  if (pos == std::string::npos) return "";
  size_t end = pos + 3;
  while (end < isa.size() &&
         ((isa[end] >= '0' && isa[end] <= '9') || (isa[end] >= 'a' && isa[end] <= 'z'))) {
    ++end;
  }
  return isa.substr(pos, end - pos);
}

// Returns the gfx name of the agent's physical execution ISA (what the hardware
// actually runs), which may differ from the ISA presented to the application.
static std::string ExecutionGfxName(hsa_agent_t agent) {
  if (agent.handle == 0) return "";
  const rocr::core::Agent* agent_object = rocr::core::Agent::Convert(agent);
  if (!agent_object || agent_object->execution_isas().empty()) return "";
  return agent_object->execution_isas()[0]->GetProcessorName();
}

// Reports the target agent's per-workgroup LDS (group segment) capacity so a
// translated kernel's group-segment usage can be validated before dispatch.
// Returns false with a reason in `failure` if it cannot be determined.
static bool GetAgentGroupSegmentLimit(hsa_agent_t agent, uint32_t& limit,
                                      std::string& failure) {
  limit = 0;
  if (agent.handle == 0) {
    failure = "translated HotSwap kernel has no target agent";
    return false;
  }
  const rocr::core::Agent* agent_object = rocr::core::Agent::Convert(agent);
  if (!agent_object) {
    failure = "translated HotSwap target agent is invalid";
    return false;
  }
  // HSA exposes the per-workgroup LDS capacity as the agent's group segment.
  for (const auto& region : agent_object->regions()) {
    if (!region) continue;
    hsa_region_segment_t segment = HSA_REGION_SEGMENT_READONLY;
    if (region->GetInfo(HSA_REGION_INFO_SEGMENT, &segment) != HSA_STATUS_SUCCESS ||
        segment != HSA_REGION_SEGMENT_GROUP)
      continue;
    size_t size = 0;
    if (region->GetInfo(HSA_REGION_INFO_SIZE, &size) == HSA_STATUS_SUCCESS) {
      limit = size > UINT32_MAX ? UINT32_MAX : static_cast<uint32_t>(size);
      return true;
    }
  }
  failure = "translated HotSwap target agent has no LDS region";
  return false;
}

static std::string EnvGfxName(const char* name) {
  const char* env = std::getenv(name);
  if (!env || !env[0] || std::strcmp(env, "0") == 0) return "";
  if (std::strcmp(env, "1") == 0) return "1";
  std::string gfx = ExtractGfxName(env);
  if (gfx.empty() && std::strncmp(env, "gfx", 3) == 0) gfx = env;
  return gfx;
}

// Debug aid: writes each incoming (pre-translation) code object to
// HSA_HOTSWAP_INPUT_DUMP_DIR. Inert unless that variable is set.
static void DumpHotSwapInputCodeObject(const void* elfData, size_t elfSize, uint32_t codeNum,
                                       const std::string& codeIsa) {
  const char* DumpDir = std::getenv("HSA_HOTSWAP_INPUT_DUMP_DIR");
  if (!DumpDir || !DumpDir[0] || !elfData || elfSize == 0) return;

  std::string Gfx = ExtractGfxName(codeIsa);
  if (Gfx.empty()) Gfx = "unknown";
  std::ostringstream Path;
  Path << DumpDir << "/hotswap_input_" << std::setfill('0') << std::setw(5) << codeNum << "_" << Gfx
       << ".co";
  std::ofstream Out(Path.str(), std::ios::binary | std::ios::trunc);
  if (!Out) return;
  Out.write(reinterpret_cast<const char*>(elfData), elfSize);
}

static const char* LookupStatusString(amd_comgr_hotswap_cache_lookup_status_t status) {
  switch (status) {
    case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED:
      return "disabled";
    case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_BYPASSED:
      return "bypassed";
    case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_MISS:
      return "miss";
    case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_HIT:
      return "hit";
    case AMD_COMGR_HOTSWAP_CACHE_LOOKUP_INVALID:
      return "invalid";
  }
  return "invalid";
}

static const char* WriteStatusString(amd_comgr_hotswap_cache_write_status_t status) {
  switch (status) {
    case AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED:
      return "not_attempted";
    case AMD_COMGR_HOTSWAP_CACHE_WRITE_SUCCESS:
      return "success";
    case AMD_COMGR_HOTSWAP_CACHE_WRITE_FAILED:
      return "failed";
  }
  return "failed";
}

static bool GetComgrResultString(amd_comgr_hotswap_transpile_result_t result,
                                 amd_comgr_hotswap_transpile_result_string_t field,
                                 std::string& out) {
  size_t size = 0;
  amd_comgr_status_t status =
      amd_comgr_hotswap_transpile_result_get_string(result, field, &size, nullptr);
  if (status != AMD_COMGR_STATUS_SUCCESS) return false;
  out.assign(size, '\0');
  status = amd_comgr_hotswap_transpile_result_get_string(result, field, &size, out.data());
  if (status != AMD_COMGR_STATUS_SUCCESS) return false;
  if (!out.empty() && out.back() == '\0') out.pop_back();
  return true;
}

template <typename T>
static bool GetComgrResultInfo(amd_comgr_hotswap_transpile_result_t result,
                               amd_comgr_hotswap_transpile_result_info_t info, T& value) {
  return amd_comgr_hotswap_transpile_result_get_info(result, info, &value) ==
      AMD_COMGR_STATUS_SUCCESS;
}

static void AppendHotSwapComgrProof(const char* event, amd_comgr_hotswap_transpile_result_t result,
                                    size_t elfSize = 0, const char* overrideFailReason = nullptr,
                                    const char* overrideFailDetail = nullptr) {
  if (!HotSwapProofLogEnabled()) return;

  bool success = false;
  bool cacheHit = false;
  int64_t lifted = 0;
  int64_t total = 0;
  amd_comgr_hotswap_cache_lookup_status_t lookupStatus = AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  amd_comgr_hotswap_cache_write_status_t writeStatus = AMD_COMGR_HOTSWAP_CACHE_WRITE_NOT_ATTEMPTED;
  std::string backend, sourceGfx, targetGfx, cacheKey, cacheDetail, kernelName;
  std::string cacheMetadata, cacheObject, failReason, failDetail;

  if (!GetComgrResultInfo(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SUCCESS, success) ||
      !GetComgrResultInfo(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_HIT, cacheHit) ||
      !GetComgrResultInfo(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_LOOKUP, lookupStatus) ||
      !GetComgrResultInfo(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_WRITE, writeStatus) ||
      !GetComgrResultInfo(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_LIFTED_COUNT, lifted) ||
      !GetComgrResultInfo(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TOTAL_COUNT, total) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_BACKEND, backend) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SOURCE_GFX, sourceGfx) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TARGET_GFX, targetGfx) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_KEY, cacheKey) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_DETAIL, cacheDetail) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_METADATA_PATH,
                            cacheMetadata) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_OBJECT_PATH,
                            cacheObject) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_REASON, failReason) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_FAIL_DETAIL, failDetail) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_KERNEL_NAME,
                            kernelName)) {
    std::cerr << "hotswap: failed to query COMGR result metadata\n";
    return;
  }

  if (overrideFailReason) {
    success = false;
    failReason = overrideFailReason;
    failDetail = overrideFailDetail ? overrideFailDetail : "";
  }

  std::ostringstream proof;
  proof << "\"event\":\"" << event << "\""
        << ",\"backend\":\"" << JsonEscape(backend) << "\""
        << ",\"source_gfx\":\"" << JsonEscape(sourceGfx) << "\""
        << ",\"target_gfx\":\"" << JsonEscape(targetGfx) << "\""
        << ",\"success\":" << (success ? "true" : "false")
        << ",\"cache_hit\":" << (cacheHit ? "true" : "false") << ",\"cache_status\":\""
        << LookupStatusString(lookupStatus) << "\""
        << ",\"cache_write_status\":\"" << WriteStatusString(writeStatus) << "\""
        << ",\"lifted_count\":" << lifted << ",\"total_count\":" << total;
  if (elfSize > 0) proof << ",\"elf_size\":" << elfSize;
  if (!cacheKey.empty()) proof << ",\"cache_key\":\"" << JsonEscape(cacheKey) << "\"";
  if (!cacheMetadata.empty())
    proof << ",\"cache_metadata\":\"" << JsonEscape(cacheMetadata) << "\"";
  if (!cacheObject.empty()) proof << ",\"cache_object\":\"" << JsonEscape(cacheObject) << "\"";
  if (!cacheDetail.empty()) proof << ",\"cache_detail\":\"" << JsonEscape(cacheDetail) << "\"";
  if (!kernelName.empty()) proof << ",\"kernel_name\":\"" << JsonEscape(kernelName) << "\"";
  if (!failReason.empty()) proof << ",\"fail_reason\":\"" << JsonEscape(failReason) << "\"";
  if (!failDetail.empty()) proof << ",\"fail_detail\":\"" << JsonEscape(failDetail) << "\"";
  AppendHotSwapProofJson(proof.str());
}

static void LogComgrCacheDebug(amd_comgr_hotswap_transpile_result_t result) {
  const char* cacheDebug = std::getenv("HSA_HOTSWAP_CACHE_DEBUG");
  if (!cacheDebug || !cacheDebug[0]) return;
  amd_comgr_hotswap_cache_lookup_status_t lookupStatus = AMD_COMGR_HOTSWAP_CACHE_LOOKUP_DISABLED;
  std::string sourceGfx, targetGfx;
  if (!GetComgrResultInfo(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_CACHE_LOOKUP, lookupStatus) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SOURCE_GFX, sourceGfx) ||
      !GetComgrResultString(result, AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_TARGET_GFX, targetGfx))
    return;
  std::cerr << "hotswap_cache: " << LookupStatusString(lookupStatus) << " (" << sourceGfx << " -> "
            << targetGfx << ")\n";
}

static bool LoadLazyTranslatedKernel(std::shared_ptr<HotSwapKernelRecord> record,
                                     std::string& failure) {
  std::shared_ptr<HotSwapLazyCodeObject> lazy = record->LazyCodeObject;
  if (!lazy) {
    failure = "lazy source record has no source code object";
    return false;
  }

  amd_comgr_data_t comgrInput = {0};
  amd_comgr_status_t comgrStatus =
      amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &comgrInput);
  if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
    failure = "COMGR failed to create HotSwap input data";
    return false;
  }

  comgrStatus = amd_comgr_set_data(
      comgrInput, lazy->SourceElf.size(),
      reinterpret_cast<const char*>(lazy->SourceElf.data()));
  if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
    amd_comgr_release_data(comgrInput);
    failure = "COMGR failed to set HotSwap input data";
    return false;
  }

  amd_comgr_data_t comgrOutput = {0};
  amd_comgr_hotswap_transpile_result_t comgrResult = {0};
  amd_comgr_hotswap_transpile_options_v2_t comgrOptions = {};
  const std::string lazyCacheDir = ResolveLazyHotSwapCacheDir(
      std::getenv("HSA_HOTSWAP_LAZY_CACHE_DIR"),
      std::getenv("HSA_HOTSWAP_CACHE_DIR"));
  comgrOptions.version = AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_VERSION_2;
  comgrOptions.cache_directory = lazyCacheDir.empty() ? nullptr : lazyCacheDir.c_str();
  comgrOptions.cache_skip_kernels = std::getenv("HSA_HOTSWAP_CACHE_SKIP_KERNELS");
  comgrOptions.hotswap_rules_path = std::getenv("HSA_HOTSWAP_RULES");
  comgrOptions.kernel_name = record->Name.c_str();
  comgrOptions.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_USE_KERNEL_NAME;
  if (rocr::hotswap::IsEnvFlagEnabled("HSA_HOTSWAP_CACHE_DISABLE"))
    comgrOptions.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_CACHE_DISABLE;
  if (rocr::hotswap::IsEnvFlagEnabled("HSA_HOTSWAP_CACHE_READONLY"))
    comgrOptions.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_CACHE_READONLY;
  if (rocr::hotswap::IsEnvFlagEnabled("HSA_HOTSWAP_STRICT"))
    comgrOptions.flags |= AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_STRICT;
  if (rocr::hotswap::IsEnvFlagEnabled("HSA_HOTSWAP_ASSUME_GLOBAL_OFFSET_ZERO") ||
      rocr::hotswap::IsEnvFlagEnabled("HSA_HOTSWAP_ASSUME_HIP_GLOBAL_OFFSET_ZERO"))
    comgrOptions.flags |=
        AMD_COMGR_HOTSWAP_TRANSPILE_OPTIONS_V2_ASSUME_HIP_GLOBAL_OFFSET_ZERO;
  comgrStatus = amd_comgr_hotswap_transpile_with_options_v2(
      comgrInput, lazy->SourceIsa.c_str(), lazy->TargetIsa.c_str(), &comgrOptions,
      &comgrOutput, &comgrResult);
  amd_comgr_release_data(comgrInput);

  if (comgrResult.handle) {
    AppendHotSwapComgrProof("hotswap_cache", comgrResult);
    LogComgrCacheDebug(comgrResult);
  }

  if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
    failure = "COMGR HotSwap per-kernel transpilation failed for " + record->Name;
    if (comgrResult.handle) {
      AppendHotSwapComgrProof("hotswap_result", comgrResult);
      amd_comgr_destroy_hotswap_transpile_result(comgrResult);
    }
    if (comgrOutput.handle) amd_comgr_release_data(comgrOutput);
    return false;
  }

  size_t comgrOutputSize = 0;
  comgrStatus = amd_comgr_get_data(comgrOutput, &comgrOutputSize, nullptr);
  if (comgrStatus != AMD_COMGR_STATUS_SUCCESS || comgrOutputSize == 0) {
    amd_comgr_release_data(comgrOutput);
    if (comgrResult.handle) {
      AppendHotSwapComgrProof("hotswap_result", comgrResult, 0,
                              "comgr_empty_output",
                              "COMGR produced no translated HSACO bytes");
      amd_comgr_destroy_hotswap_transpile_result(comgrResult);
    }
    failure = "COMGR produced no translated HSACO bytes for " + record->Name;
    return false;
  }

  std::vector<uint8_t> targetBytes(comgrOutputSize, 0);
  comgrStatus = amd_comgr_get_data(
      comgrOutput, &comgrOutputSize, reinterpret_cast<char*>(targetBytes.data()));
  amd_comgr_release_data(comgrOutput);
  if (comgrStatus != AMD_COMGR_STATUS_SUCCESS) {
    if (comgrResult.handle) {
      AppendHotSwapComgrProof("hotswap_result", comgrResult, 0,
                              "comgr_read_output",
                              "COMGR translated HSACO read failed");
      amd_comgr_destroy_hotswap_transpile_result(comgrResult);
    }
    failure = "failed to read COMGR translated HSACO for " + record->Name;
    return false;
  }

  auto targetExec = std::make_shared<ExecutableImpl>(
      lazy->Profile, std::make_unique<rocr::amd::LoaderContext>(),
      HotSwapRuntimeTranslatedExecutableId,
      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT);
  hsa_code_object_t targetCodeObject = {
      reinterpret_cast<uint64_t>(targetBytes.data())};
  const bool previousTargetLoad = CurrentHotSwapLoadingTranslatedTarget;
  CurrentHotSwapLoadingTranslatedTarget = true;
  hsa_status_t status = targetExec->LoadCodeObject(
      lazy->Agent, targetCodeObject, nullptr, "hotswap-lazy-kernel", nullptr);
  CurrentHotSwapLoadingTranslatedTarget = previousTargetLoad;
  if (status != HSA_STATUS_SUCCESS) {
    failure = "ROCR failed to load translated HotSwap kernel " + record->Name;
    if (comgrResult.handle) {
      AppendHotSwapComgrProof("hotswap_result", comgrResult, targetBytes.size(),
                              "target_load_failed", failure.c_str());
      amd_comgr_destroy_hotswap_transpile_result(comgrResult);
    }
    return false;
  }
  status = targetExec->Freeze(nullptr);
  if (status != HSA_STATUS_SUCCESS) {
    failure = "ROCR failed to freeze translated HotSwap kernel " + record->Name;
    if (comgrResult.handle) {
      AppendHotSwapComgrProof("hotswap_result", comgrResult, targetBytes.size(),
                              "target_freeze_failed", failure.c_str());
      amd_comgr_destroy_hotswap_transpile_result(comgrResult);
    }
    return false;
  }

  std::string targetSymbolName = record->Name + ".kd";
  Symbol* symbol = targetExec->GetSymbol(targetSymbolName.c_str(), &lazy->Agent);
  if (!symbol) {
    failure = "translated HotSwap kernel symbol not found: " + targetSymbolName;
    if (comgrResult.handle) {
      AppendHotSwapComgrProof("hotswap_result", comgrResult, targetBytes.size(),
                              "target_symbol_missing", failure.c_str());
      amd_comgr_destroy_hotswap_transpile_result(comgrResult);
    }
    return false;
  }

  uint64_t targetKernelObject = 0;
  uint32_t targetPrivateSegmentSize = 0;
  uint32_t targetGroupSegmentSize = 0;
  if (!symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                       &targetKernelObject) ||
      !symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                       &targetPrivateSegmentSize) ||
      !symbol->GetInfo(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                       &targetGroupSegmentSize) ||
      targetKernelObject == 0) {
    failure = "translated HotSwap kernel metadata missing: " + record->Name;
    if (comgrResult.handle) {
      AppendHotSwapComgrProof("hotswap_result", comgrResult, targetBytes.size(),
                              "target_metadata_missing", failure.c_str());
      amd_comgr_destroy_hotswap_transpile_result(comgrResult);
    }
    return false;
  }

  uint32_t targetGroupSegmentLimit = 0;
  if (!GetAgentGroupSegmentLimit(lazy->Agent, targetGroupSegmentLimit, failure)) {
    if (comgrResult.handle) {
      AppendHotSwapComgrProof("hotswap_result", comgrResult, targetBytes.size(),
                              "target_lds_limit_missing", failure.c_str());
      amd_comgr_destroy_hotswap_transpile_result(comgrResult);
    }
    return false;
  }

  record->TargetKernelObject = targetKernelObject;
  record->TargetPrivateSegmentSize = targetPrivateSegmentSize;
  record->TargetGroupSegmentSize = targetGroupSegmentSize;
  record->TargetGroupSegmentLimit = targetGroupSegmentLimit;
  int64_t scaledDispatchFactor = 1;
  GetComgrResultInfo(comgrResult,
                     AMD_COMGR_HOTSWAP_TRANSPILE_RESULT_SCALED_DISPATCH_FACTOR,
                     scaledDispatchFactor);
  record->ScaledDispatchFactor = static_cast<uint32_t>(scaledDispatchFactor);
  record->TranslationSucceeded = true;
  {
    std::lock_guard<std::mutex> Guard(
        HotSwapRuntimeLifetimeTranslatedExecutablesMutex());
    HotSwapRuntimeLifetimeTranslatedExecutables().push_back(std::move(targetExec));
  }
  record->Kind.store(HotSwapKernelKind::Translated, std::memory_order_release);
  if (comgrResult.handle) {
    AppendHotSwapComgrProof("hotswap_result", comgrResult, targetBytes.size());
    amd_comgr_destroy_hotswap_transpile_result(comgrResult);
  }
  return true;
}

[[noreturn]] static void FatalHotSwapDispatch(uint64_t address,
                                              const std::string& name,
                                              HotSwapKernelKind kind,
                                              const std::string& reason,
                                              const std::string& sourceKind = "",
                                              const std::string& targetGfx = "") {
  std::ostringstream proof;
  proof << "\"event\":\"hotswap_lazy_dispatch_reject\""
        << ",\"kernel_name\":\"" << JsonEscape(name.empty() ? "<unknown>" : name)
        << "\""
        << ",\"kernel_object\":\"0x" << std::hex << address << "\""
        << std::dec
        << ",\"kind\":\"" << HotSwapKernelKindName(kind) << "\"";
  if (!sourceKind.empty())
    proof << ",\"source_kind\":\"" << JsonEscape(sourceKind) << "\"";
  if (!targetGfx.empty())
    proof << ",\"target_gfx\":\"" << JsonEscape(targetGfx) << "\"";
  proof << ",\"reason\":\"" << JsonEscape(reason) << "\"";
  AppendHotSwapProofJson(proof.str());
  std::cerr << "hotswap: refusing kernel dispatch";
  if (!name.empty()) std::cerr << " for " << name;
  std::cerr << " (kind=" << HotSwapKernelKindName(kind)
            << (sourceKind.empty() ? "" : ", source_kind=") << sourceKind
            << (targetGfx.empty() ? "" : ", target_gfx=") << targetGfx
            << "): " << reason << "\n";
  std::abort();
}

// Applies the translated-target rewrite at dispatch time and preserves the
// existing fail-closed diagnostics and proof logging on any patch failure.
static void PatchHotSwapDispatchToTargetOrAbort(
    const std::shared_ptr<HotSwapKernelRecord>& record, uint64_t* address,
    uint32_t* private_segment_size, uint32_t* group_segment_size,
    uint32_t* scaled_dispatch_factor, HotSwapKernelKind kind) {
  const uint64_t sourceKernelObject = *address;
  std::string segmentFailure;
  if (!PatchHotSwapTranslatedDispatch(
          *record, *address, *private_segment_size, *group_segment_size,
          segmentFailure))
    FatalHotSwapDispatch(sourceKernelObject, record->Name, kind, segmentFailure);

  *scaled_dispatch_factor = record->ScaledDispatchFactor;

  std::ostringstream proof;
  proof << "\"event\":\"hotswap_lazy_dispatch_patch\""
        << ",\"kernel_name\":\"" << JsonEscape(record->Name) << "\""
        << ",\"source_kernel_object\":\"0x" << std::hex << sourceKernelObject
        << "\""
        << ",\"target_kernel_object\":\"0x" << record->TargetKernelObject
        << "\"" << std::dec
        << ",\"private_segment_size\":" << *private_segment_size
        << ",\"group_segment_size\":" << *group_segment_size;
  if (record->ScaledDispatchFactor > 1)
    proof << ",\"scaled_dispatch_factor\":" << record->ScaledDispatchFactor;
  AppendHotSwapProofJson(proof.str());
}

// Dispatch-time hook invoked by the AQL doorbell intercept for each dispatched
// kernel. Given the packet's kernel_object it looks up the HotSwap record and, on
// success, rewrites *address (and the segment sizes) to the translated target
// kernel: already-translated kernels pass through, lazy-source kernels are
// translated via COMGR on first use and cached, and anything unsupported or
// unregistered aborts the dispatch. Returns true once the packet is safe to run.
bool PrepareHotSwapDispatchKernelObject(uint64_t* address,
                                        uint32_t* private_segment_size,
                                        uint32_t* group_segment_size,
                                        uint32_t* scaled_dispatch_factor) {
  if (!address || !private_segment_size || !group_segment_size ||
      !scaled_dispatch_factor)
    FatalHotSwapDispatch(0, "", HotSwapKernelKind::Untranslated,
                         "invalid dispatch packet pointer");
  // A kernel needs x-extent scaling only if it was raised under the comgr
  // ScaledModuloReplicationProjection; PatchHotSwapDispatchToTargetOrAbort
  // reports the factor once the target record is known.
  *scaled_dispatch_factor = 1;

  std::shared_ptr<HotSwapKernelRecord> record =
      LookupHotSwapKernelRecord(*address);
  if (!record)
    FatalHotSwapDispatch(*address, "", HotSwapKernelKind::Untranslated,
                         "unregistered kernel object");

  HotSwapKernelKind kind = record->Kind.load(std::memory_order_acquire);
  if (kind == HotSwapKernelKind::Translated) {
    if (record->TargetKernelObject != 0)
      PatchHotSwapDispatchToTargetOrAbort(record, address, private_segment_size,
                                          group_segment_size,
                                          scaled_dispatch_factor, kind);
    return true;
  }

  if (kind == HotSwapKernelKind::RuntimeTargetInternal) {
    if (!record->IsRegisteredRocrBlit(*address))
      FatalHotSwapDispatch(*address, record->Name, kind,
                           "runtime target-internal kernel is not a registered ROCR blit shader",
                           record->SourceKind, record->TargetGfx);

    std::ostringstream proof;
    proof << "\"event\":\"hotswap_runtime_target_internal_dispatch\""
          << ",\"kernel_name\":\"" << JsonEscape(record->Name) << "\""
          << ",\"kernel_object\":\"0x" << std::hex << *address << "\""
          << std::dec
          << ",\"kind\":\"runtime_target_internal\"";
    if (!record->SourceKind.empty())
      proof << ",\"source_kind\":\"" << JsonEscape(record->SourceKind) << "\"";
    if (!record->TargetGfx.empty())
      proof << ",\"target_gfx\":\"" << JsonEscape(record->TargetGfx) << "\"";
    if (!record->Failure.empty())
      proof << ",\"reason\":\"" << JsonEscape(record->Failure) << "\"";
    AppendHotSwapProofJson(proof.str());
    return true;
  }

  if (kind != HotSwapKernelKind::LazySource)
    FatalHotSwapDispatch(*address, record->Name, kind,
                         record->Failure.empty()
                             ? "kernel object is not translated"
                             : record->Failure,
                         record->SourceKind, record->TargetGfx);

  std::lock_guard<std::mutex> Guard(record->Mutex);
  if (!record->TranslationAttempted) {
    record->TranslationAttempted = true;
    std::string failure;
    // Submitting the source kernel object would execute foreign-ISA code.
    // Treat unsupported lazy translation as a fail-closed process abort rather
    // than letting a bad dispatch escape to hardware.
    if (!LoadLazyTranslatedKernel(record, failure)) {
      record->Failure = failure;
      FatalHotSwapDispatch(*address, record->Name, kind, failure);
    }
  }
  if (!record->TranslationSucceeded || record->TargetKernelObject == 0)
    FatalHotSwapDispatch(*address, record->Name, kind,
                         record->Failure.empty()
                             ? "lazy translation did not produce a target kernel"
                             : record->Failure);

  kind = record->Kind.load(std::memory_order_acquire);
  PatchHotSwapDispatchToTargetOrAbort(record, address, private_segment_size,
                                      group_segment_size,
                                      scaled_dispatch_factor, kind);

  return true;
}

struct HotSwapGfxMach {
  const char* name;
  uint32_t mach;
};

// AMDGPU e_flags processor values are the only target identity available before
// the ROCR code object parser accepts a presented-source object. Keep this table
// limited to the gfx names supported by the HotSwap presentation mode.
static constexpr HotSwapGfxMach kHotSwapGfxMachMap[] = {
    {"gfx900", 0x02c},  {"gfx902", 0x02d},  {"gfx904", 0x02e},  {"gfx906", 0x02f},
    {"gfx908", 0x030},  {"gfx909", 0x031},  {"gfx90a", 0x03f},  {"gfx90c", 0x032},
    {"gfx942", 0x04c},  {"gfx950", 0x04f},  {"gfx1010", 0x033}, {"gfx1011", 0x034},
    {"gfx1012", 0x035}, {"gfx1030", 0x036}, {"gfx1031", 0x037}, {"gfx1032", 0x038},
    {"gfx1033", 0x039}, {"gfx1034", 0x03e}, {"gfx1035", 0x03d}, {"gfx1100", 0x041},
    {"gfx1101", 0x046}, {"gfx1102", 0x047}, {"gfx1103", 0x044}, {"gfx1150", 0x043},
    {"gfx1151", 0x04a}, {"gfx1200", 0x048}, {"gfx1201", 0x04e}, {"gfx1250", 0x049},
    {"gfx1251", 0x05a}, {nullptr, 0},
};

static bool HotSwapMachFromGfxName(const std::string& gfx, uint32_t& mach) {
  for (const HotSwapGfxMach* entry = kHotSwapGfxMachMap; entry->name; ++entry) {
    if (gfx == entry->name) {
      mach = entry->mach;
      return true;
    }
  }
  return false;
}

#endif

Loader* Loader::Create(Context* context)
{
  return new AmdHsaCodeLoader(context);
}

void Loader::Destroy(Loader *loader)
{
  // Loader resets the link_map, but the executables and loaded code objects are not deleted.
  _amdgpu_r_debug.r_map = nullptr;
  _amdgpu_r_debug.r_state = r_debug::RT_CONSISTENT;
  r_debug_tail() = nullptr;
}

Executable* AmdHsaCodeLoader::CreateExecutable(
  hsa_profile_t profile, const char *options, hsa_default_float_rounding_mode_t default_float_rounding_mode)
{
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);

  executables.push_back(std::make_shared<ExecutableImpl>(profile, context, executables.size(), default_float_rounding_mode));
  return executables.back().get();
}

Executable* AmdHsaCodeLoader::CreateExecutable(
      std::unique_ptr<Context> isolated_context,
      hsa_profile_t profile,
      const char *options,
      hsa_default_float_rounding_mode_t default_float_rounding_mode)
{
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);

  executables.push_back(std::make_shared<ExecutableImpl>(profile, std::move(isolated_context), executables.size(), default_float_rounding_mode));
  return executables.back().get();
}

static void AddCodeObjectInfoIntoDebugMap(link_map* map) {
  if (r_debug_tail()) {
      r_debug_tail()->l_next = map;
      map->l_prev = r_debug_tail();
      map->l_next = nullptr;
  } else {
      _amdgpu_r_debug.r_map = map;
      map->l_prev = nullptr;
      map->l_next = nullptr;
  }
  r_debug_tail() = map;
}

static void RemoveCodeObjectInfoFromDebugMap(link_map* map) {
  if (r_debug_tail() == map) {
      r_debug_tail() = map->l_prev;
  }
  if (_amdgpu_r_debug.r_map == map) {
      _amdgpu_r_debug.r_map = map->l_next;
  }

  if (map->l_prev) {
      map->l_prev->l_next = map->l_next;
  }
  if (map->l_next) {
      map->l_next->l_prev = map->l_prev;
  }

  free(const_cast<char*>(map->l_name));
  memset(map, 0, sizeof(link_map));
}

hsa_status_t AmdHsaCodeLoader::FreezeExecutable(Executable *executable, const char *options) {
  hsa_status_t  status = executable->Freeze(options);
  if (status != HSA_STATUS_SUCCESS) {
    return status;
  }

  // Assuming runtime atomic implements C++ std::memory_order
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);
  atomic::Store(&_amdgpu_r_debug.r_state, r_debug::RT_ADD, std::memory_order_relaxed);
  atomic::Fence(std::memory_order_acq_rel);
  _loader_debug_state();
  atomic::Fence(std::memory_order_acq_rel);
  for (const auto &lco : reinterpret_cast<ExecutableImpl*>(executable)->loaded_code_objects) {
    AddCodeObjectInfoIntoDebugMap(&(lco->r_debug_info));
  }
  atomic::Store(&_amdgpu_r_debug.r_state, r_debug::RT_CONSISTENT, std::memory_order_release);
  _loader_debug_state();

  return HSA_STATUS_SUCCESS;
}

void AmdHsaCodeLoader::DestroyExecutable(Executable *executable) {
  // Assuming runtime atomic implements C++ std::memory_order
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);
  atomic::Store(&_amdgpu_r_debug.r_state, r_debug::RT_DELETE, std::memory_order_relaxed);
  atomic::Fence(std::memory_order_acq_rel);
  _loader_debug_state();
  atomic::Fence(std::memory_order_acq_rel);
  for (const auto &lco : reinterpret_cast<ExecutableImpl*>(executable)->loaded_code_objects) {
    RemoveCodeObjectInfoFromDebugMap(&(lco->r_debug_info));
  }
  atomic::Store(&_amdgpu_r_debug.r_state, r_debug::RT_CONSISTENT, std::memory_order_release);
  _loader_debug_state();

  executables[static_cast<ExecutableImpl*>(executable)->id()].reset();
}

hsa_status_t AmdHsaCodeLoader::IterateExecutables(
  hsa_status_t (*callback)(
    hsa_executable_t executable,
    void *data),
  void *data)
{
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);
  assert(callback);

  for (const auto &exec : executables) {
    if(exec != nullptr){
      hsa_status_t status = callback(Executable::Handle(exec.get()), data);
      if (status != HSA_STATUS_SUCCESS) {
        return status;
      }
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t AmdHsaCodeLoader::QuerySegmentDescriptors(
  hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
  size_t *num_segment_descriptors)
{
  if (!num_segment_descriptors) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (*num_segment_descriptors == 0 && segment_descriptors) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }
  if (*num_segment_descriptors != 0 && !segment_descriptors) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  this->EnableReadOnlyMode();

  size_t actual_num_segment_descriptors = 0;
  for (const auto &executable : executables) {
    if (executable) {
      actual_num_segment_descriptors += executable->GetNumSegmentDescriptors();
    }
  }

  if (*num_segment_descriptors == 0) {
    *num_segment_descriptors = actual_num_segment_descriptors;
    this->DisableReadOnlyMode();
    return HSA_STATUS_SUCCESS;
  }
  if (*num_segment_descriptors != actual_num_segment_descriptors) {
    this->DisableReadOnlyMode();
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  }

  size_t i = 0;
  for (const auto &executable : executables) {
    if (executable) {
      i += executable->QuerySegmentDescriptors(segment_descriptors, actual_num_segment_descriptors, i);
    }
  }

  this->DisableReadOnlyMode();
  return HSA_STATUS_SUCCESS;
}

uint64_t AmdHsaCodeLoader::FindHostAddress(uint64_t device_address)
{
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  if (device_address == 0) {
    return 0;
  }

  for (const auto &exec : executables) {
    if (exec != nullptr) {
      uint64_t host_address = exec->FindHostAddress(device_address);
      if (host_address != 0) {
        return host_address;
      }
    }
  }
  return 0;
}

void AmdHsaCodeLoader::PrintHelp(std::ostream& out)
{
  LoaderOptions().PrintHelp(out);
}

void AmdHsaCodeLoader::EnableReadOnlyMode()
{
  rw_lock_.ReaderLock();
  for (const auto &executable : executables) {
    if (executable) {
      ((ExecutableImpl*)executable.get())->EnableReadOnlyMode();
    }
  }
}

void AmdHsaCodeLoader::DisableReadOnlyMode()
{
  rw_lock_.ReaderUnlock();
  for (const auto &executable : executables) {
    if (executable) {
      ((ExecutableImpl*)executable.get())->DisableReadOnlyMode();
    }
  }
}

//===----------------------------------------------------------------------===//
// SymbolImpl.                                                                    //
//===----------------------------------------------------------------------===//

bool SymbolImpl::GetInfo(hsa_symbol_info32_t symbol_info, void *value) {
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_TYPE) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_TYPE)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_TYPE) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_TYPE)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_NAME_LENGTH) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_NAME) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_NAME)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME_LENGTH)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_MODULE_NAME) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_MODULE_NAME)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_LINKAGE) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_LINKAGE)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_IS_DEFINITION) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_IS_DEFINITION)),
    "attributes are not compatible"
  );

  assert(value);

  switch (symbol_info) {
    case HSA_CODE_SYMBOL_INFO_TYPE: {
      *((hsa_symbol_kind_t*)value) = kind;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_NAME_LENGTH: {
      *((uint32_t*)value) = symbol_name.size();
      break;
    }
    case HSA_CODE_SYMBOL_INFO_NAME: {
      memset(value, 0x0, symbol_name.size());
      memcpy(value, symbol_name.c_str(), symbol_name.size());
      break;
    }
    case HSA_CODE_SYMBOL_INFO_MODULE_NAME_LENGTH: {
      *((uint32_t*)value) = module_name.size();
      break;
    }
    case HSA_CODE_SYMBOL_INFO_MODULE_NAME: {
      memset(value, 0x0, module_name.size());
      memcpy(value, module_name.c_str(), module_name.size());
      break;
    }
    case HSA_CODE_SYMBOL_INFO_LINKAGE: {
      *((hsa_symbol_linkage_t*)value) = linkage;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_IS_DEFINITION: {
      *((bool*)value) = is_definition;
      break;
    }
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_CALL_CONVENTION: {
      *((uint32_t*)value) = 0;
      break;
    }
    case HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT:
    case HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS: {
      if (!is_loaded) {
        return false;
      }
      *((uint64_t*)value) = address;
      break;
    }
    case HSA_EXECUTABLE_SYMBOL_INFO_AGENT: {
      if (!is_loaded) {
        return false;
      }
      *((hsa_agent_t*)value) = agent;
      break;
    }
    default: {
      return false;
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// KernelSymbol.                                                              //
//===----------------------------------------------------------------------===//

bool KernelSymbol::GetInfo(hsa_symbol_info32_t symbol_info, void *value) {
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK)),
    "attributes are not compatible"
  );

  assert(value);

  switch (symbol_info) {
    case HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE: {
      *((uint32_t*)value) = kernarg_segment_size;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT: {
      *((uint32_t*)value) = kernarg_segment_alignment;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE: {
      *((uint32_t*)value) = group_segment_size;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE: {
      *((uint32_t*)value) = private_segment_size;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK: {
      *((bool*)value) = is_dynamic_callstack;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_KERNEL_WAVEFRONT_SIZE: {
      *((uint32_t*)value) = wavefront_size;
      break;
    }
    case HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_SIZE: {
      *((uint32_t*)value) = size;
      break;
    }
    case HSA_EXT_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT_ALIGN: {
      *((uint32_t*)value) = alignment;
      break;
    }
    default: {
      return SymbolImpl::GetInfo(symbol_info, value);
    }
  }

  return true;
}

//===----------------------------------------------------------------------===//
// VariableSymbol.                                                            //
//===----------------------------------------------------------------------===//

bool VariableSymbol::GetInfo(hsa_symbol_info32_t symbol_info, void *value) {
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALLOCATION)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SEGMENT)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ALIGNMENT)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE)),
    "attributes are not compatible"
  );
  static_assert(
    (symbol_attribute32_t(HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST) ==
     symbol_attribute32_t(HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_IS_CONST)),
    "attributes are not compatible"
  );

  switch (symbol_info) {
    case HSA_CODE_SYMBOL_INFO_VARIABLE_ALLOCATION: {
      *((hsa_variable_allocation_t*)value) = allocation;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_VARIABLE_SEGMENT: {
      *((hsa_variable_segment_t*)value) = segment;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_VARIABLE_ALIGNMENT: {
      *((uint32_t*)value) = alignment;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_VARIABLE_SIZE: {
      *((uint32_t*)value) = size;
      break;
    }
    case HSA_CODE_SYMBOL_INFO_VARIABLE_IS_CONST: {
      *((bool*)value) = is_constant;
      break;
    }
    default: {
      return SymbolImpl::GetInfo(symbol_info, value);
    }
  }

  return true;
}

bool LoadedCodeObjectImpl::GetInfo(amd_loaded_code_object_info_t attribute, void *value)
{
  assert(value);

  switch (attribute) {
    case AMD_LOADED_CODE_OBJECT_INFO_ELF_IMAGE:
      ((hsa_code_object_t*)value)->handle = reinterpret_cast<uint64_t>(elf_data);
      break;
    case AMD_LOADED_CODE_OBJECT_INFO_ELF_IMAGE_SIZE:
      *((size_t*)value) = elf_size;
      break;
    default: {
      return false;
    }
  }

  return true;
}

hsa_status_t LoadedCodeObjectImpl::IterateLoadedSegments(
  hsa_status_t (*callback)(
    amd_loaded_segment_t loaded_segment,
    void *data),
  void *data)
{
  assert(callback);

  for (auto &loaded_segment : loaded_segments) {
    hsa_status_t status = callback(LoadedSegment::Handle(loaded_segment), data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  return HSA_STATUS_SUCCESS;
}

void LoadedCodeObjectImpl::Print(std::ostream& out)
{
  out << "Code Object" << std::endl;
}

bool Segment::GetInfo(amd_loaded_segment_info_t attribute, void *value)
{
  assert(value);

  switch (attribute) {
    case AMD_LOADED_SEGMENT_INFO_TYPE: {
      *((amdgpu_hsa_elf_segment_t*)value) = segment;
      break;
    }
    case AMD_LOADED_SEGMENT_INFO_ELF_BASE_ADDRESS: {
      *((uint64_t*)value) = vaddr;
      break;
    }
    case AMD_LOADED_SEGMENT_INFO_LOAD_BASE_ADDRESS: {
      *((uint64_t*)value) = reinterpret_cast<uint64_t>(this->Address(this->VAddr()));
      break;
    }
    case AMD_LOADED_SEGMENT_INFO_SIZE: {
      *((size_t*)value) = size;
      break;
    }
    default: {
      return false;
    }
  }

  return true;
}

uint64_t Segment::Offset(uint64_t addr)
{
  assert(IsAddressInSegment(addr));
  return addr - vaddr;
}

void* Segment::Address(uint64_t addr)
{
  return owner->context()->SegmentAddress(segment, agent, ptr, Offset(addr));
}

bool Segment::Freeze()
{
  return !frozen ? (frozen = owner->context()->SegmentFreeze(segment, agent, ptr, size)) : true;
}

bool Segment::IsAddressInSegment(uint64_t addr)
{
  return vaddr <= addr && addr < vaddr + size;
}

bool Segment::IsAddressInSegment(uint64_t addr, size_t copy_size)
{
  if (addr < vaddr) { return false; }
  const uint64_t offset = addr - vaddr;
  // Mirror the 1-arg bound (offset < size, i.e. addr < vaddr + size). Express
  // the exclusive end offset + copy_size <= size without overflow via
  // copy_size <= size - offset (equivalent to addr + copy_size <= vaddr + size).
  return offset < size && copy_size <= size - offset;
}

bool Segment::Copy(uint64_t addr, const void* src, size_t size)
{
  // loader must do copies before freezing.
  assert(!frozen);

  if (size > 0) {
    // addr/size can be derived from attacker-controlled code-object fields
    // (e.g. a relocation's section sh_addr + r_offset). Offset() only asserts
    // the start address via IsAddressInSegment() and that assert compiles out
    // under NDEBUG, so validate the entire destination range here to prevent a
    // heap out-of-bounds write when loading a crafted code object.
    if (!IsAddressInSegment(addr, size)) {
      return false;
    }
    owner->context()->SegmentCopy(segment, agent, ptr, Offset(addr), src, size);
  }
  return true;
}

void Segment::Print(std::ostream& out)
{
  out << "Segment" << std::endl
    << "    Type: " << AmdHsaElfSegmentToString(segment)
    << "    Size: " << size
    << "    VAddr: " << vaddr << std::endl
    << "    Ptr: " << std::hex << ptr << std::dec
    << std::endl;
}

void Segment::Destroy()
{
  owner->context()->SegmentFree(segment, agent, ptr, size);
}

void LoadedCodeObjectImpl::Destroy() {
#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
  for (uint64_t address : hotswap_kernel_objects_) {
    UnregisterHotSwapKernelObject(address);
  }
  hotswap_kernel_objects_.clear();
#endif
}

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
void LoadedCodeObjectImpl::RecordHotSwapKernelObject(uint64_t address) {
  hotswap_kernel_objects_.push_back(address);
}
#endif

//===----------------------------------------------------------------------===//
// ExecutableImpl.                                                                //
//===----------------------------------------------------------------------===//

ExecutableImpl::ExecutableImpl(
    const hsa_profile_t &_profile,
    Context *context,
    size_t id,
    hsa_default_float_rounding_mode_t default_float_rounding_mode)
  : Executable()
  , profile_(_profile)
  , context_(context)
  , id_(id)
  , default_float_rounding_mode_(default_float_rounding_mode)
  , state_(HSA_EXECUTABLE_STATE_UNFROZEN)
  , program_allocation_segment(nullptr)
{
}

ExecutableImpl::ExecutableImpl(
    const hsa_profile_t &_profile,
    std::unique_ptr<Context> unique_context,
    size_t id,
    hsa_default_float_rounding_mode_t default_float_rounding_mode)
  : Executable()
  , profile_(_profile)
  , unique_context_(std::move(unique_context))
  , id_(id)
  , default_float_rounding_mode_(default_float_rounding_mode)
  , state_(HSA_EXECUTABLE_STATE_UNFROZEN)
  , program_allocation_segment(nullptr)
{
  context_ = unique_context_.get();
}

ExecutableImpl::~ExecutableImpl() {
  for (const auto& o : objects) {
    o->Destroy();
  }
  objects.clear();
}

hsa_status_t ExecutableImpl::DefineProgramExternalVariable(
  const char *name, void *address)
{
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);
  assert(name);

  if (HSA_EXECUTABLE_STATE_FROZEN == state_) {
    return HSA_STATUS_ERROR_FROZEN_EXECUTABLE;
  }

  auto symbol_entry = program_symbols_.find(std::string(name));
  if (symbol_entry != program_symbols_.end()) {
    return HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED;
  }

  program_symbols_.insert(
    std::make_pair(std::string(name),
                   std::make_shared<VariableSymbol>(true,
                                      "", // Only program linkage symbols can be
                                          // defined.
                                      std::string(name),
                                      HSA_SYMBOL_LINKAGE_PROGRAM,
                                      true,
                                      HSA_VARIABLE_ALLOCATION_PROGRAM,
                                      HSA_VARIABLE_SEGMENT_GLOBAL,
                                      0,     // TODO: size.
                                      0,     // TODO: align.
                                      false, // TODO: const.
                                      true,
                                      reinterpret_cast<uint64_t>(address))));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::DefineAgentExternalVariable(
  const char *name,
  hsa_agent_t agent,
  hsa_variable_segment_t segment,
  void *address)
{
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);
  assert(name);

  if (HSA_EXECUTABLE_STATE_FROZEN == state_) {
    return HSA_STATUS_ERROR_FROZEN_EXECUTABLE;
  }

  auto symbol_entry = agent_symbols_.find(std::make_pair(std::string(name), agent));
  if (symbol_entry != agent_symbols_.end()) {
    return HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED;
  }

  auto insert_status = agent_symbols_.insert(
    std::make_pair(std::make_pair(std::string(name), agent),
                   std::make_shared<VariableSymbol>(true,
                                      "", // Only program linkage symbols can be
                                          // defined.
                                      std::string(name),
                                      HSA_SYMBOL_LINKAGE_PROGRAM,
                                      true,
                                      HSA_VARIABLE_ALLOCATION_AGENT,
                                      segment,
                                      0,     // TODO: size.
                                      0,     // TODO: align.
                                      false, // TODO: const.
                                      true,
                                      reinterpret_cast<uint64_t>(address))));
  assert(insert_status.second);
  insert_status.first->second->agent = agent;

  return HSA_STATUS_SUCCESS;
}

bool ExecutableImpl::IsProgramSymbol(const char *symbol_name) {
  assert(symbol_name);

  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  return program_symbols_.find(std::string(symbol_name)) != program_symbols_.end();
}

Symbol* ExecutableImpl::GetSymbol(
  const char *symbol_name,
  const hsa_agent_t *agent)
{
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  return this->GetSymbolInternal(symbol_name, agent);
}

Symbol* ExecutableImpl::GetSymbolInternal(
  const char *symbol_name,
  const hsa_agent_t *agent)
{
  assert(symbol_name);

  std::string mangled_name = std::string(symbol_name);
  if (mangled_name.empty()) {
    return nullptr;
  }

  if (!agent) {
    auto program_symbol = program_symbols_.find(mangled_name);
    if (program_symbol != program_symbols_.end()) {
      return program_symbol->second.get();
    }
    return nullptr;
  }

  auto agent_symbol = agent_symbols_.find(std::make_pair(mangled_name, *agent));
  if (agent_symbol != agent_symbols_.end()) {
    return agent_symbol->second.get();
  }
  return nullptr;
}

hsa_status_t ExecutableImpl::IterateSymbols(
  iterate_symbols_f callback, void *data)
{
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  assert(callback);

  for (auto &symbol_entry : program_symbols_) {
    hsa_status_t hsc =
      callback(Executable::Handle(this), Symbol::Handle(symbol_entry.second.get()), data);
    if (HSA_STATUS_SUCCESS != hsc) {
      return hsc;
    }
  }
  for (auto &symbol_entry : agent_symbols_) {
    hsa_status_t hsc =
      callback(Executable::Handle(this), Symbol::Handle(symbol_entry.second.get()), data);
    if (HSA_STATUS_SUCCESS != hsc) {
      return hsc;
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::IterateAgentSymbols(
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t exec,
                             hsa_agent_t agent,
                             hsa_executable_symbol_t symbol,
                             void *data),
    void *data) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  assert(callback);

  for (auto &symbol_entry : agent_symbols_) {
    if (symbol_entry.second->GetAgent().handle != agent.handle) {
      continue;
    }

    hsa_status_t status = callback(
        Executable::Handle(this), agent, Symbol::Handle(symbol_entry.second.get()),
        data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::IterateProgramSymbols(
    hsa_status_t (*callback)(hsa_executable_t exec,
                             hsa_executable_symbol_t symbol,
                             void *data),
    void *data) {
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  assert(callback);

  for (auto &symbol_entry : program_symbols_) {
    hsa_status_t status = callback(
        Executable::Handle(this), Symbol::Handle(symbol_entry.second.get()), data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::IterateLoadedCodeObjects(
  hsa_status_t (*callback)(
    hsa_executable_t executable,
    hsa_loaded_code_object_t loaded_code_object,
    void *data),
  void *data)
{
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  assert(callback);

  for (const auto& loaded_code_object : loaded_code_objects) {
    hsa_status_t status = callback(
        Executable::Handle(this),
        LoadedCodeObject::Handle(loaded_code_object.get()),
        data);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  return HSA_STATUS_SUCCESS;
}

size_t ExecutableImpl::GetNumSegmentDescriptors()
{
  // assuming we are in readonly mode.
  size_t actual_num_segment_descriptors = 0;
  for (const auto &obj : loaded_code_objects) {
    actual_num_segment_descriptors += obj->LoadedSegments().size();
  }
  return actual_num_segment_descriptors;
}

size_t ExecutableImpl::QuerySegmentDescriptors(
  hsa_ven_amd_loader_segment_descriptor_t *segment_descriptors,
  size_t total_num_segment_descriptors,
  size_t first_empty_segment_descriptor)
{
  // assuming we are in readonly mode.
  assert(segment_descriptors);
  assert(first_empty_segment_descriptor < total_num_segment_descriptors);

  size_t i = first_empty_segment_descriptor;
  for (const auto &obj : loaded_code_objects) {
    assert(i < total_num_segment_descriptors);
    for (auto &seg : obj->LoadedSegments()) {
      segment_descriptors[i].agent = seg->Agent();
      segment_descriptors[i].executable = Executable::Handle(seg->Owner());
      segment_descriptors[i].code_object_storage_type = HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
      segment_descriptors[i].code_object_storage_base = obj->ElfData();
      segment_descriptors[i].code_object_storage_size = obj->ElfSize();
      segment_descriptors[i].code_object_storage_offset = seg->StorageOffset();
      segment_descriptors[i].segment_base = seg->Address(seg->VAddr());
      segment_descriptors[i].segment_size = seg->Size();
      ++i;
    }
  }

  return i - first_empty_segment_descriptor;
}

hsa_agent_t LoadedCodeObjectImpl::getAgent() const {
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return loaded_segments.front()->Agent();
}
hsa_executable_t LoadedCodeObjectImpl::getExecutable() const {
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return Executable::Handle(loaded_segments.front()->Owner());
}
uint64_t LoadedCodeObjectImpl::getElfData() const {
  return reinterpret_cast<uint64_t>(elf_data);
}
uint64_t LoadedCodeObjectImpl::getElfSize() const {
  return (uint64_t)elf_size;
}
uint64_t LoadedCodeObjectImpl::getStorageOffset() const {
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return (uint64_t)loaded_segments.front()->StorageOffset();
}
uint64_t LoadedCodeObjectImpl::getLoadBase() const {
  // TODO Add support for code objects with 0 segments.
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return reinterpret_cast<uint64_t>(loaded_segments.front()->Address(0));
}
uint64_t LoadedCodeObjectImpl::getLoadSize() const {
  // TODO Add support for code objects with 0 or >1 segments.
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return (uint64_t)loaded_segments.front()->Size();
}
int64_t LoadedCodeObjectImpl::getDelta() const {
  // TODO Add support for code objects with 0 segments.
  assert(loaded_segments.size() == 1 && "Only supports code objects v2+");
  return getLoadBase() - loaded_segments.front()->VAddr();
}

std::string LoadedCodeObjectImpl::getUri() const {
  return std::string(r_debug_info.l_name);
}

hsa_executable_t AmdHsaCodeLoader::FindExecutable(uint64_t device_address)
{
  hsa_executable_t execHandle = {0};
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  if (device_address == 0) {
    return execHandle;
  }

  for (const auto &exec : executables) {
    if (exec != nullptr) {
      uint64_t host_address = exec->FindHostAddress(device_address);
      if (host_address != 0) {
        return Executable::Handle(exec.get());
      }
    }
  }
  return execHandle;
}

uint64_t ExecutableImpl::FindHostAddress(uint64_t device_address)
{
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);
  for (const auto &obj : loaded_code_objects) {
    assert(obj);
    for (auto &seg : obj->LoadedSegments()) {
      assert(seg);
      uint64_t paddr = (uint64_t)(uintptr_t)seg->Address(seg->VAddr());
      if (paddr <= device_address && device_address < paddr + seg->Size()) {
        void *haddr = context_->SegmentHostAddress(
          seg->ElfSegment(), seg->Agent(), seg->Ptr(), device_address - paddr);
        return nullptr == haddr ? 0 : (uint64_t)(uintptr_t)haddr;
      }
    }
  }
  return 0;
}

void ExecutableImpl::EnableReadOnlyMode()
{
  rw_lock_.ReaderLock();
}

void ExecutableImpl::DisableReadOnlyMode()
{
  rw_lock_.ReaderUnlock();
}

#define HSAERRCHECK(hsc)                                                       \
  if (hsc != HSA_STATUS_SUCCESS) {                                             \
    assert(false);                                                             \
    return hsc;                                                                \
  }                                                                            \


hsa_status_t ExecutableImpl::GetInfo(
    hsa_executable_info_t executable_info, void *value)
{
  ReaderLockGuard<ReaderWriterLock> reader_lock(rw_lock_);

  assert(value);

  switch (executable_info) {
    case HSA_EXECUTABLE_INFO_PROFILE: {
      *((hsa_profile_t*)value) = profile_;;
      break;
    }
    case HSA_EXECUTABLE_INFO_STATE: {
      *((hsa_executable_state_t*)value) = state_;
      break;
    }
    case HSA_EXECUTABLE_INFO_DEFAULT_FLOAT_ROUNDING_MODE: {
      *((hsa_default_float_rounding_mode_t*)value) =
          default_float_rounding_mode_;
      break;
    }
    default: {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }
  }

  return HSA_STATUS_SUCCESS;
}

static uint32_t NextCodeObjectNum()
{
  static std::atomic_uint_fast32_t dumpN(1);
  return dumpN++;
}

hsa_status_t ExecutableImpl::LoadCodeObject(
  hsa_agent_t agent,
  hsa_code_object_t code_object,
  const char *options,
  const std::string &uri,
  hsa_loaded_code_object_t *loaded_code_object)
{
  return LoadCodeObject(agent, code_object, 0, options, uri, loaded_code_object);
}

hsa_status_t ExecutableImpl::LoadCodeObject(
  hsa_agent_t agent,
  hsa_code_object_t code_object,
  size_t code_object_size,
  const char *options,
  const std::string &uri,
  hsa_loaded_code_object_t *loaded_code_object)
{
  WriterLockGuard<ReaderWriterLock> writer_lock(rw_lock_);
  if (HSA_EXECUTABLE_STATE_FROZEN == state_) {
    logger_ << "LoaderError: executable is already frozen\n";
    return HSA_STATUS_ERROR_FROZEN_EXECUTABLE;
  }

  if (code_object_size == 0) {
    const void* elf_data = reinterpret_cast<const void*>(code_object.handle);
    if (!elf_data) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    code_object_size = amd::elf::ElfSize(elf_data, 0);
    if (code_object_size == 0) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
  }

  LoaderOptions loaderOptions;
  if (options && !loaderOptions.ParseOptions(options)) {
    return HSA_STATUS_ERROR;
  }

  const char *options_append = getenv("LOADER_OPTIONS_APPEND");
  if (options_append && !loaderOptions.ParseOptions(options_append)) {
    return HSA_STATUS_ERROR;
  }

  typedef std::tuple<uint32_t, uint32_t, std::string> Substitute;
  std::vector<Substitute> substitutes;

  for (const std::string& s : loaderOptions.Substitute()->values()) {
    std::string::size_type vi = s.find('=');
    if (vi == std::string::npos) { return HSA_STATUS_ERROR; }
    std::string value = s.substr(vi + 1);
    std::string range = s.substr(0, vi);
    std::string::size_type mi = range.find('-');
    uint32_t n1 = UINT32_MAX, n2 = UINT32_MAX;
    if (mi != std::string::npos) {
      std::string s1, s2;
      s1 = range.substr(0, mi - 1);
      s2 = range.substr(mi + 1);
      std::istringstream is1(s1); is1 >> n1;
      std::istringstream is2(s2); is2 >> n2;
    } else {
      std::istringstream is(range); is >> n1;
      n2 = n1;
    }
    substitutes.push_back(std::make_tuple(n1, n2, value));
  }

  uint32_t codeNum = NextCodeObjectNum();

  code = std::make_unique<code::AmdHsaCode>();

  std::string substituteFileName;
  for (const Substitute& ss : substitutes) {
    if (codeNum >= std::get<0>(ss) && codeNum <= std::get<1>(ss)) {
      substituteFileName = std::get<2>(ss);
      break;
    }
  }
  std::vector<char> buffer;
  if (substituteFileName.empty()) {
    if (!code->InitAsBuffer(reinterpret_cast<const void*>(code_object.handle),
                            code_object_size)) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
  } else {
    if (!ReadFileIntoBuffer(substituteFileName, buffer)) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    if (!code->InitAsBuffer(&buffer[0], buffer.size())) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
  }

  if (loaderOptions.DumpAll()->is_set() || loaderOptions.DumpCode()->is_set()) {
    if (!code->SaveToFile(amd::hsa::DumpFileName(loaderOptions.DumpDir()->value(), LOADER_DUMP_PREFIX, "hsaco", codeNum))) {
      // Ignore error.
    }
  }
  if (loaderOptions.DumpAll()->is_set() || loaderOptions.DumpIsa()->is_set()) {
    if (!code->PrintToFile(amd::hsa::DumpFileName(loaderOptions.DumpDir()->value(), LOADER_DUMP_PREFIX, "isa", codeNum))) {
      // Ignore error.
    }
  }

  std::string codeIsa;
  unsigned genericVersion;
  if (!code->GetIsa(codeIsa, &genericVersion)) {
    logger_ << "LoaderError: failed to determine code object's ISA\n";
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
  DumpHotSwapInputCodeObject(code->ElfData(), code->ElfSize(), codeNum, codeIsa);
#endif

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
  if (HotSwapProofLogEnabled()) {
    std::ostringstream proof;
    proof << "\"event\":\"hsa_load_code_object\""
          << ",\"code_isa\":\"" << JsonEscape(codeIsa) << "\""
          << ",\"source_gfx\":\"" << JsonEscape(ExtractGfxName(codeIsa)) << "\""
          << ",\"elf_size\":" << code->ElfSize();
    AppendHotSwapProofJson(proof.str());
  }
#endif

  // Kernel-entry trampolines (gfx125x). Disabled by default for gfx125x.
  // Set LOADER_ENABLE_TRAMPOLINE=1 or LOADER_ENABLE_TRAMPOLINE_NO_WA=1 to enable
  // (for testing only). NO_WA selects the minimal stub without global_wb.
  trampoline_no_wa_gfx125x_ = loaderOptions.TrampolineNoWaEnabled();
  trampoline_enabled_gfx125x_ =
      (loaderOptions.TrampolineEnabled() || trampoline_no_wa_gfx125x_) &&
      CodeObjectIsaIsGfx125Family(codeIsa);
  kd_fixups_.clear();

  uint32_t majorVersion, minorVersion;
  if (!code->GetCodeObjectVersion(&majorVersion, &minorVersion)) {
    logger_ << "LoaderError: failed to determine code object's version\n";
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  if (majorVersion < 1 || majorVersion > 6) {
    logger_ << "LoaderError: unsupported code object version: " << majorVersion << "\n";
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  if (agent.handle == 0 && majorVersion == 1) {
    logger_ << "LoaderError: code object v1 requires non-null agent\n";
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  uint32_t codeHsailMajor;
  uint32_t codeHsailMinor;
  hsa_profile_t codeProfile;
  hsa_machine_model_t codeMachineModel;
  hsa_default_float_rounding_mode_t codeRoundingMode;
  if (!code->GetNoteHsail(&codeHsailMajor, &codeHsailMinor, &codeProfile, &codeMachineModel, &codeRoundingMode)) {
    codeProfile = profile_;
  }
  if (profile_ != codeProfile) {
    logger_ << "LoaderError: mismatched profiles\n";
    return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  }

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
  std::vector<uint8_t> ownedCodeObject;
  bool ownsCodeObject = false;
  bool translatedInThisCall = CurrentHotSwapLoadingTranslatedTarget;
  bool lazySourceInThisCall = false;
  std::shared_ptr<HotSwapLazyCodeObject> lazyCodeObjectForThisCall;
  const std::string presentedGfx = EnvGfxName("HSA_HOTSWAP_PRESENT_ISA");
  const bool presentationMode =
      !rocr::hotswap::IsEnvFlagEnabled("HSA_HOTSWAP_DISABLE") && !presentedGfx.empty();
  std::string hotswapTargetGfx;
  std::string hotswapTargetFailure;
  if (!ResolveHotSwapPresentationTarget(
          presentationMode, EnvGfxName("HSA_HOTSWAP_TARGET"),
          EnvGfxName("HSA_HOTSWAP_ISA_OVERRIDE"), ExecutionGfxName(agent),
          hotswapTargetGfx, hotswapTargetFailure)) {
    logger_ << "LoaderError: " << hotswapTargetFailure << "\n";
    return HSA_STATUS_ERROR_INVALID_ISA_NAME;
  }

  if (CurrentHotSwapLoadingTranslatedTarget) {
    ownedCodeObject.assign(reinterpret_cast<const uint8_t*>(code->ElfData()),
                           reinterpret_cast<const uint8_t*>(code->ElfData()) +
                               code->ElfSize());
    code = std::make_unique<code::AmdHsaCode>();
    if (!code->InitAsBuffer(ownedCodeObject.data(), ownedCodeObject.size())) {
      logger_ << "LoaderError: lazy translated HotSwap HSACO failed to initialize\n";
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    }
    ownsCodeObject = true;
  }

  if (presentationMode) {
    std::string targetGfx = hotswapTargetGfx;

    if (!targetGfx.empty()) {
      std::string sourceGfx;
      uint8_t sourceMach = 0;
      const std::string codeGfx = ExtractGfxName(codeIsa);
      bool presentationInterceptForHotSwap = false;

      if (!codeGfx.empty() && codeGfx == presentedGfx && codeGfx != targetGfx) {
        uint32_t autoSourceMach = 0;
        if (HotSwapMachFromGfxName(codeGfx, autoSourceMach)) {
          sourceGfx = codeGfx;
          sourceMach = static_cast<uint8_t>(autoSourceMach & 0xffu);
          presentationInterceptForHotSwap = true;
        }
      } else if (!codeGfx.empty() && codeGfx != targetGfx) {
        std::ostringstream proof;
        proof << "\"event\":\"hotswap_skip\""
              << ",\"backend\":\"comgr\""
              << ",\"reason\":\"not_presented_source\""
              << ",\"source_gfx\":\"" << JsonEscape(codeGfx) << "\""
              << ",\"target_gfx\":\"" << JsonEscape(targetGfx) << "\""
              << ",\"elf_size\":" << code->ElfSize();
        AppendHotSwapProofJson(proof.str());
      }
      if (presentationInterceptForHotSwap && sourceGfx != targetGfx) {
        const std::string sourceIsa = std::string("amdgcn-amd-amdhsa--") + sourceGfx;
        const std::string targetIsa = std::string("amdgcn-amd-amdhsa--") + targetGfx;

        {
          std::ostringstream proof;
          proof << "\"event\":\"transpile_decision\""
                << ",\"source\":\"presented_source\""
                << ",\"source_gfx\":\"" << JsonEscape(sourceGfx) << "\""
                << ",\"target_gfx\":\"" << JsonEscape(targetGfx) << "\""
                << ",\"orig_mach\":\"0x" << std::hex << static_cast<unsigned>(sourceMach)
                << std::dec << "\""
                << ",\"code_isa\":\"" << JsonEscape(codeIsa) << "\"";
          AppendHotSwapProofJson(proof.str());
        }

        std::vector<uint8_t> comgrSource(
            reinterpret_cast<const uint8_t*>(code->ElfData()),
            reinterpret_cast<const uint8_t*>(code->ElfData()) + code->ElfSize());

        ownedCodeObject = std::move(comgrSource);
        code = std::make_unique<code::AmdHsaCode>();
        if (!code->InitAsBuffer(ownedCodeObject.data(), ownedCodeObject.size())) {
          logger_ << "LoaderError: HotSwap source HSACO failed to initialize\n";
          return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
        }
        ownsCodeObject = true;
        lazySourceInThisCall = true;
        lazyCodeObjectForThisCall = std::make_shared<HotSwapLazyCodeObject>();
        lazyCodeObjectForThisCall->SourceElf = ownedCodeObject;
        lazyCodeObjectForThisCall->SourceIsa = sourceIsa;
        lazyCodeObjectForThisCall->TargetIsa = targetIsa;
        lazyCodeObjectForThisCall->SourceGfx = sourceGfx;
        lazyCodeObjectForThisCall->TargetGfx = targetGfx;
        lazyCodeObjectForThisCall->SourceMach = sourceMach;
        lazyCodeObjectForThisCall->Agent = agent;
        lazyCodeObjectForThisCall->Profile = profile();
        if (!code->GetIsa(codeIsa, &genericVersion)) {
          logger_ << "LoaderError: failed to determine HotSwap source code object's ISA\n";
          return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
        }
        if (!code->GetCodeObjectVersion(&majorVersion, &minorVersion)) {
          logger_ << "LoaderError: failed to determine HotSwap source code object's version\n";
          return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
        }
        std::ostringstream proof;
        proof << "\"event\":\"hotswap_lazy_source_registered\""
              << ",\"backend\":\"comgr\""
              << ",\"source_gfx\":\"" << JsonEscape(sourceGfx) << "\""
              << ",\"target_gfx\":\"" << JsonEscape(targetGfx) << "\""
              << ",\"elf_size\":" << ownedCodeObject.size();
        AppendHotSwapProofJson(proof.str());
      }
    }
  }
#endif

  hsa_isa_t objectsIsa = context_->IsaFromName(codeIsa.c_str());
  if (!objectsIsa.handle) {
    logger_ << "LoaderError: code object's ISA (" << codeIsa.c_str() << ") is invalid\n";
    return HSA_STATUS_ERROR_INVALID_ISA_NAME;
  }

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
  if (presentationMode && agent.handle != 0) {
    const std::string codeGfx = ExtractGfxName(codeIsa);
    const bool targetObject = !hotswapTargetGfx.empty() && codeGfx == hotswapTargetGfx;
    if (targetObject && !translatedInThisCall) {
      logger_ << "LoaderError: target-ISA code object (" << codeIsa.c_str()
              << ") was not produced by HotSwap in this load call\n";
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    }
  }
#endif

  if (agent.handle != 0 && !context_->IsaSupportedByAgent(agent, objectsIsa, genericVersion)) {
#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
    if (lazySourceInThisCall) {
      std::ostringstream proof;
      proof << "\"event\":\"hotswap_lazy_source_isa_bypass\""
            << ",\"backend\":\"comgr\""
            << ",\"code_isa\":\"" << JsonEscape(codeIsa) << "\""
            << ",\"source_gfx\":\"" << JsonEscape(ExtractGfxName(codeIsa)) << "\""
            << ",\"target_gfx\":\"" << JsonEscape(hotswapTargetGfx) << "\""
            << ",\"reason\":\"lazy_source_requires_dispatch_translation\"";
      AppendHotSwapProofJson(proof.str());
    } else
#endif
    {
      logger_ << "LoaderError: code object's ISA (" << codeIsa.c_str()
              << ") is not supported by the agent\n";
      return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
    }
  }

  hsa_status_t status;

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
  if (ownsCodeObject) {
    objects.push_back(
        std::make_shared<LoadedCodeObjectImpl>(this, agent, std::move(ownedCodeObject)));
  } else
#endif
  {
    objects.push_back(
        std::make_shared<LoadedCodeObjectImpl>(this, agent, code->ElfData(), code->ElfSize()));
  }
  loaded_code_objects.push_back(std::static_pointer_cast<LoadedCodeObjectImpl>(objects.back()));

  status = LoadSegments(agent, code.get(), majorVersion);
  if (status != HSA_STATUS_SUCCESS) return status;

#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
  HotSwapKernelKind PreviousHotSwapKernelKind = CurrentHotSwapKernelKind;
  std::shared_ptr<HotSwapLazyCodeObject> PreviousHotSwapLazyCodeObject =
      CurrentHotSwapLazyCodeObject;
  CurrentHotSwapKernelKind = translatedInThisCall
      ? HotSwapKernelKind::Translated
      : (lazySourceInThisCall ? HotSwapKernelKind::LazySource
                              : HotSwapKernelKind::Untranslated);
  CurrentHotSwapLazyCodeObject =
      lazySourceInThisCall ? lazyCodeObjectForThisCall : nullptr;
#endif
  for (size_t i = 0; i < code->SymbolCount(); ++i) {
    if (majorVersion >= 2 &&
        code->GetSymbol(i)->elfSym()->type() != STT_AMDGPU_HSA_KERNEL &&
        code->GetSymbol(i)->elfSym()->binding() == STB_LOCAL)
      continue;

    status = LoadSymbol(agent, code->GetSymbol(i), majorVersion);
    if (status != HSA_STATUS_SUCCESS) {
#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
      CurrentHotSwapKernelKind = PreviousHotSwapKernelKind;
      CurrentHotSwapLazyCodeObject = PreviousHotSwapLazyCodeObject;
#endif
      return status;
    }
  }
#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
  CurrentHotSwapKernelKind = PreviousHotSwapKernelKind;
  CurrentHotSwapLazyCodeObject = PreviousHotSwapLazyCodeObject;
#endif

  status = ApplyRelocations(agent, code.get());
  if (status != HSA_STATUS_SUCCESS) { return status; }

  // Emit kernel-entry trampolines into the host shadow now that the image is
  // final (post-relocation) and still unfrozen. The single Freeze DMA carries
  // them to device along with the rewritten descriptors.
  if (trampoline_enabled_gfx125x_ && !kd_fixups_.empty()) {
    status = InstallTrampolinesGfx125x(agent);
    if (status != HSA_STATUS_SUCCESS) {
      return status;
    }
  }

  code.reset();

  if (loaderOptions.DumpAll()->is_set() || loaderOptions.DumpExec()->is_set()) {
    if (!PrintToFile(amd::hsa::DumpFileName(loaderOptions.DumpDir()->value(), LOADER_DUMP_PREFIX, "exec", codeNum))) {
      // Ignore error.
    }
  }

  loaded_code_objects.back()->r_debug_info.l_addr = (decltype(link_map::l_addr))loaded_code_objects.back()->getDelta();
  loaded_code_objects.back()->r_debug_info.l_name = strdup(uri.c_str());
  loaded_code_objects.back()->r_debug_info.l_prev = nullptr;
  loaded_code_objects.back()->r_debug_info.l_next = nullptr;

  if (nullptr != loaded_code_object) { *loaded_code_object = LoadedCodeObject::Handle(loaded_code_objects.back().get()); }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::LoadSegments(hsa_agent_t agent,
                                          const code::AmdHsaCode *c,
                                          uint32_t majorVersion) {
  if (majorVersion < 2)
    return LoadSegmentsV1(agent, c);
  else
    return LoadSegmentsV2(agent, c);
}

hsa_status_t ExecutableImpl::LoadSegmentsV1(hsa_agent_t agent,
                                            const code::AmdHsaCode *c) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  for (size_t i = 0; i < c->DataSegmentCount(); ++i) {
    status = LoadSegmentV1(agent, c->DataSegment(i));
    if (status != HSA_STATUS_SUCCESS) return status;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::LoadSegmentsV2(hsa_agent_t agent,
                                            const code::AmdHsaCode *c) {
  assert(c->Machine() == ELF::EM_AMDGPU && "Program code objects are not supported");

  if (!c->DataSegmentCount()) return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;

  uint64_t vaddr = c->DataSegment(0)->vaddr();
  uint64_t size = c->DataSegment(c->DataSegmentCount() - 1)->vaddr() +
                  c->DataSegment(c->DataSegmentCount() - 1)->memSize();

  void *ptr = context_->SegmentAlloc(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, size,
      AMD_ISA_ALIGN_BYTES, true);
  if (!ptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  std::shared_ptr<Segment> load_segment = std::make_shared<Segment>(this, agent, AMDGPU_HSA_SEGMENT_CODE_AGENT,
      ptr, size, vaddr, c->DataSegment(0)->offset());
  if (!load_segment) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  hsa_status_t status = HSA_STATUS_SUCCESS;
  for (size_t i = 0; i < c->DataSegmentCount(); ++i) {
    status = LoadSegmentV2(c->DataSegment(i), load_segment.get());
    if (status != HSA_STATUS_SUCCESS) return status;
  }

  objects.push_back(load_segment);
  loaded_code_objects.back()->LoadedSegments().push_back(load_segment.get());

  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::LoadSegmentV1(hsa_agent_t agent,
                                           const code::Segment *s) {
  assert(s->type() < PT_LOOS + AMDGPU_HSA_SEGMENT_LAST);
  if (s->memSize() == 0)
    return HSA_STATUS_SUCCESS;
  amdgpu_hsa_elf_segment_t segment = (amdgpu_hsa_elf_segment_t)(s->type() - PT_LOOS);
  std::shared_ptr<Segment> new_seg;
  bool need_alloc = true;
  if (segment == AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM && nullptr != program_allocation_segment) {
    new_seg = program_allocation_segment;
    need_alloc = false;
  }
  if (need_alloc) {
    if (s->imageSize() > s->memSize()) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    // Reject a segment whose file contents cannot be sourced from within the
    // backing image (crafted p_offset). See ROCM-26177 finding #1.
    const char* segment_data = s->data();
    if (s->imageSize() > 0 && segment_data == nullptr) {
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
    }
    void* ptr = context_->SegmentAlloc(segment, agent, s->memSize(), s->align(), true);
    if (!ptr) { return HSA_STATUS_ERROR_OUT_OF_RESOURCES; }
    new_seg = std::make_shared<Segment>(this, agent, segment, ptr, s->memSize(), s->vaddr(), s->offset());
    // Copy() return unchecked: imageSize <= memSize was validated above and the
    // destination is this segment's own [vaddr, vaddr + imageSize) range.
    new_seg->Copy(s->vaddr(), segment_data, s->imageSize());
    objects.push_back(new_seg);

    if (segment == AMDGPU_HSA_SEGMENT_GLOBAL_PROGRAM) {
      program_allocation_segment = new_seg;
    }
  }
  assert(new_seg);
  loaded_code_objects.back()->LoadedSegments().push_back(new_seg.get());
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::LoadSegmentV2(const code::Segment *data_segment,
                                           loader::Segment *load_segment) {
  assert(data_segment && load_segment);
  if (data_segment->imageSize() > data_segment->memSize()) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  // The combined code segment allocation is sized from the last data segment
  // only (see LoadSegmentsV2). A crafted code object with non-monotonic,
  // overlapping, or out-of-range p_vaddr values could otherwise drive the copy
  // below past the end of that allocation (heap OOB write). Bound the copy
  // destination [offset, offset + imageSize) against the allocation explicitly,
  // since Segment::Offset() only asserts the start address and compiles out
  // under NDEBUG. See ROCM-26177 finding #1.
  const uint64_t seg_vaddr = data_segment->vaddr();
  const uint64_t base_vaddr = load_segment->VAddr();
  const size_t alloc_size = load_segment->Size();
  if (seg_vaddr < base_vaddr) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  const uint64_t offset = seg_vaddr - base_vaddr;
  if (offset > alloc_size ||
      data_segment->imageSize() > alloc_size - offset) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  // Reject a segment whose file contents cannot be sourced from within the
  // backing image (crafted p_offset). See ROCM-26177 finding #1.
  const char* segment_data = data_segment->data();
  if (data_segment->imageSize() > 0 && segment_data == nullptr) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  load_segment->Copy(data_segment->vaddr(), segment_data,
                     data_segment->imageSize());

  return HSA_STATUS_SUCCESS;
}

// Returns true if the kernel entry at entry_vaddr already begins with the GFX1250
// unclaused-VMEM workaround prologue (see kGfx1250UnclausedVmemPrologue). The check
// reads the post-relocation host shadow of the code segment (valid pre-Freeze).
static bool KernelEntryHasUnclausedVmemPrologue(Context* context, Segment* code_seg,
                                                uint64_t entry_vaddr) {
  static constexpr size_t kPrologueBytes = sizeof(kGfx1250UnclausedVmemPrologue);
  if (!code_seg->IsAddressInSegment(entry_vaddr) ||
      !code_seg->IsAddressInSegment(entry_vaddr + kPrologueBytes - 1)) {
    return false;
  }
  void* host = context->SegmentHostAddress(code_seg->ElfSegment(), code_seg->Agent(),
                                           code_seg->Ptr(), code_seg->Offset(entry_vaddr));
  if (host == nullptr) return false;
  const uint32_t* w = reinterpret_cast<const uint32_t*>(host);
  return w[0] == kGfx1250UnclausedVmemPrologue[0] &&
         w[1] == kGfx1250UnclausedVmemPrologue[1] &&
         w[2] == kGfx1250UnclausedVmemPrologue[2] &&
         w[3] == kGfx1250UnclausedVmemPrologue[3];
}

hsa_status_t ExecutableImpl::InstallTrampolinesGfx125x(hsa_agent_t agent) {
  // Skip kernels whose entry already carries the compiler-inserted unclaused-VMEM
  // workaround prologue (llvm PR #208467): a trampoline would only duplicate the
  // global_wb/v_nop that is already there, so dispatch can go straight to the real
  // entry. Kernels still needing the workaround keep their trampoline.
  std::vector<KdFixup> fixups;
  fixups.reserve(kd_fixups_.size());
  for (const auto& f : kd_fixups_) {
    if (KernelEntryHasUnclausedVmemPrologue(context_, f.code_seg, f.kd_vaddr + f.entry_off)) {
      continue;
    }
    fixups.push_back(f);
  }
  if (fixups.empty()) return HSA_STATUS_SUCCESS;

  const size_t n = fixups.size();

  // Size the trailing prefetch guard from the largest CP instruction-prefetch
  // window among this pool's kernels (INST_PREF_SIZE lines * 128 B). The forward
  // prefetch from the last stub reaches its_entry + INST_PREF_SIZE*128; since that
  // stub's own slot (one stub stride) already lies inside the pool, only the
  // remainder, (INST_PREF_SIZE*128 - stub_size), can spill past the pool and needs
  // a guard. (Clamp to 0 when the window fits within a stub slot.)
  uint32_t max_pref_lines = 0;
  for (const auto& f : fixups) max_pref_lines = std::max(max_pref_lines, f.inst_pref);
  const size_t pref_bytes = static_cast<size_t>(max_pref_lines) * kInstPrefUnitBytes;
  const size_t guard = pref_bytes > kTrampolineStubStride ? pref_bytes - kTrampolineStubStride : 0;
  const size_t pool = n * kTrampolineStubStride + guard;

  // AMDGPU_HSA_SEGMENT_CODE_AGENT yields *executable* device memory: the loader
  // context backs it with RegionMemory(..., is_code=true), which sets
  // core::MemoryRegion::AllocateExecutable (see amd_loader_context.cpp).
  void* ptr = context_->SegmentAlloc(AMDGPU_HSA_SEGMENT_CODE_AGENT, agent, pool,
                                     AMD_ISA_ALIGN_BYTES, /*zero=*/true);
  if (!ptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;

  // vaddr == 0: Address()/Copy() index by raw byte offset into the pool.
  auto tramp = std::make_shared<Segment>(this, agent, AMDGPU_HSA_SEGMENT_CODE_AGENT, ptr, pool,
                                         /*vaddr=*/0, /*storage_offset=*/0);
  objects.push_back(tramp);               // freed via Destroy() in ~ExecutableImpl
  trampoline_segments_.push_back(tramp);  // frozen in ExecutableImpl::Freeze

  for (size_t i = 0; i < n; ++i) {
    const KdFixup& f = fixups[i];
    const uint64_t stub_off = i * kTrampolineStubStride;
    // Device addresses are valid pre-Freeze (RegionMemory::ptr_ is set at alloc).
    const uint64_t kd_dev = reinterpret_cast<uint64_t>(f.code_seg->Address(f.kd_vaddr));
    const uint64_t entry_dev =
        reinterpret_cast<uint64_t>(f.code_seg->Address(f.kd_vaddr + f.entry_off));
    const uint64_t stub_dev = reinterpret_cast<uint64_t>(tramp->Address(stub_off));

    uint8_t blob[kTrampolineStubStride];
    if (trampoline_no_wa_gfx125x_) {
      BuildTrampolineGfx1250NoWa(blob, entry_dev);
    } else {
      BuildTrampolineGfx1250(blob, entry_dev);
    }
    tramp->Copy(stub_off, blob, sizeof(blob));  // -> trampoline host shadow

    // Gated by LOADER_ENABLE_LOGGING=1 (see Logger).
    logger_ << "Loader: injecting gfx125x entry trampoline for kernel " << f.name << "\n";

    // Redirect dispatch onto the stub: kernel_object(kd_dev) + new_off == stub.
    int64_t new_off = static_cast<int64_t>(stub_dev) - static_cast<int64_t>(kd_dev);
    f.code_seg->Copy(f.kd_vaddr + llvm::amdhsa::KERNEL_CODE_ENTRY_BYTE_OFFSET_OFFSET, &new_off,
                     sizeof(new_off));  // -> code host shadow
  }

  // The prefetch guard is left as the allocation's zero-fill (zero=true): it is
  // committed and readable -- all the CP prefetch needs -- and is never executed.
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::LoadSymbol(hsa_agent_t agent,
                                        code::Symbol* sym,
                                        uint32_t majorVersion)
{
  if (sym->IsDeclaration()) {
    return LoadDeclarationSymbol(agent, sym, majorVersion);
  } else {
    return LoadDefinitionSymbol(agent, sym, majorVersion);
  }
}

namespace {

bool string_ends_with(const std::string &str, const std::string &suf) {
  return str.size() >= suf.size() ? str.compare(str.size() - suf.size(), suf.size(), suf) == 0 : false;
}

}

hsa_status_t ExecutableImpl::LoadDefinitionSymbol(hsa_agent_t agent,
                                                  code::Symbol* sym,
                                                  uint32_t majorVersion)
{
  bool isAgent = sym->IsAgent();
  if (majorVersion >= 2) {
    isAgent = agent.handle != 0;
  }
  if (isAgent) {
    auto agent_symbol = agent_symbols_.find(std::make_pair(sym->Name(), agent));
    if (agent_symbol != agent_symbols_.end()) {
      // TODO(spec): this is not spec compliant.
      return HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED;
    }
  } else {
    auto program_symbol = program_symbols_.find(sym->Name());
    if (program_symbol != program_symbols_.end()) {
      // TODO(spec): this is not spec compliant.
      return HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED;
    }
  }

  uint64_t address = SymbolAddress(agent, sym);
  std::shared_ptr<SymbolImpl> symbol;
  if (string_ends_with(sym->GetSymbolName(), ".kd")) {
    // V3.
    llvm::amdhsa::kernel_descriptor_t kd;
    sym->GetSection()->getData(sym->SectionOffset(), &kd, sizeof(kd));

    if (trampoline_enabled_gfx125x_) {
      // Record this descriptor; the trampoline is installed after relocations.
      // sym->VAddr() is the descriptor's ELF vaddr (matches SymbolAddress below).
      // INST_PREF_SIZE = number of 128B I$ lines the CP prefetches ahead of the
      // entry; captured here to size the trampoline's prefetch guard.
      uint32_t inst_pref = AMDHSA_BITS_GET(
          kd.compute_pgm_rsrc3, rocr::llvm::amdhsa::COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE);
      kd_fixups_.push_back({SymbolSegment(agent, sym), sym->VAddr(),
                            kd.kernel_code_entry_byte_offset, inst_pref, sym->GetSymbolName()});
    }

    uint32_t kernarg_segment_size = kd.kernarg_size; // FIXME: If 0 then the compiler is not specifying the size.
    uint32_t kernarg_segment_alignment = 16;         // FIXME: Use the minumum HSA required alignment.
    uint32_t group_segment_size = kd.group_segment_fixed_size;
    uint32_t private_segment_size = kd.private_segment_fixed_size;
    bool is_dynamic_callstack = AMDHSA_BITS_GET(kd.kernel_code_properties, rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK);
    bool uses_wave32 = AMDHSA_BITS_GET( kd.kernel_code_properties, rocr::llvm::amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32);

    uint64_t size = sym->Size();

    std::shared_ptr<KernelSymbol> kernel_symbol = std::make_shared<KernelSymbol>(true,
                                    sym->GetModuleName(),
                                    sym->GetSymbolName(),
                                    sym->Linkage(),
                                    true, // sym->IsDefinition()
                                    kernarg_segment_size,
                                    kernarg_segment_alignment,
                                    group_segment_size,
                                    private_segment_size,
                                    is_dynamic_callstack,
                                    size,
                                    64,
                                    uses_wave32 ? 32 : 64,
                                    address);
#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
    if (ShouldCheckHotSwapDispatchKernelObjects()) {
      std::string KernelName;
      std::string KernelNameFailure;
      if (!DeriveHotSwapKernelRecordName(kernel_symbol->symbol_name, true,
                                         KernelName, KernelNameFailure)) {
        logger_ << "LoaderError: " << KernelNameFailure << ": "
                << kernel_symbol->symbol_name << "\n";
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      RegisterHotSwapLoadedKernelObject(address, KernelName);
      UpdateHotSwapKernelObjectLaunchMetadata(address, group_segment_size,
                                              private_segment_size);
      loaded_code_objects.back()->RecordHotSwapKernelObject(address);
    }
#endif
    symbol = kernel_symbol;
  } else if (sym->IsVariableSymbol()) {
    symbol = std::make_shared<VariableSymbol>(true,
                       sym->GetModuleName(),
                       sym->GetSymbolName(),
                       sym->Linkage(),
                       true, // sym->IsDefinition()
                       sym->Allocation(),
                       sym->Segment(),
                       sym->Size(),
                       sym->Alignment(),
                       sym->IsConst(),
                       false,
                       address);
  } else if (sym->IsKernelSymbol()) {
      amd_kernel_code_t akc;
      sym->GetSection()->getData(sym->SectionOffset(), &akc, sizeof(akc));

      uint32_t kernarg_segment_size =
        uint32_t(akc.kernarg_segment_byte_size);
      uint32_t kernarg_segment_alignment =
        uint32_t(1 << akc.kernarg_segment_alignment);
      uint32_t group_segment_size =
        uint32_t(akc.workgroup_group_segment_byte_size);
      uint32_t private_segment_size =
        uint32_t(akc.workitem_private_segment_byte_size);
      bool is_dynamic_callstack =
        AMD_HSA_BITS_GET(akc.kernel_code_properties, AMD_KERNEL_CODE_PROPERTIES_IS_DYNAMIC_CALLSTACK) ? true : false;
      bool uses_wave32 = akc.wavefront_size == AMD_POWERTWO_32;

      uint64_t size = sym->Size();

      if (!size && sym->SectionOffset() < sym->GetSection()->size()) {
        // ORCA Runtime relies on symbol size equal to size of kernel ISA. If symbol size is 0 in ELF,
        // calculate end of segment - symbol value.
        size = sym->GetSection()->size() - sym->SectionOffset();
      }
      std::shared_ptr<KernelSymbol> kernel_symbol = std::make_shared<KernelSymbol>(true,
                                      sym->GetModuleName(),
                                      sym->GetSymbolName(),
                                      sym->Linkage(),
                                      true, // sym->IsDefinition()
                                      kernarg_segment_size,
                                      kernarg_segment_alignment,
                                      group_segment_size,
                                      private_segment_size,
                                      is_dynamic_callstack,
                                      size,
                                      256,
                                      uses_wave32 ? 32 : 64,
                                      address);
      kernel_symbol->debug_info.elf_raw = code->ElfData();
      kernel_symbol->debug_info.elf_size = code->ElfSize();
      kernel_symbol->debug_info.kernel_name = kernel_symbol->full_name.c_str();
      kernel_symbol->debug_info.owning_segment = (void*)SymbolSegment(agent, sym)->Address(sym->GetSection()->addr());
#ifdef ROCR_HOTSWAP_COMGR_ADAPTER
    if (ShouldCheckHotSwapDispatchKernelObjects()) {
      std::string KernelName;
      std::string KernelNameFailure;
      if (!DeriveHotSwapKernelRecordName(kernel_symbol->full_name, false,
                                         KernelName, KernelNameFailure)) {
        logger_ << "LoaderError: " << KernelNameFailure << ": "
                << kernel_symbol->full_name << "\n";
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      RegisterHotSwapLoadedKernelObject(address, KernelName);
      UpdateHotSwapKernelObjectLaunchMetadata(address, group_segment_size,
                                              private_segment_size);
      loaded_code_objects.back()->RecordHotSwapKernelObject(address);
    }
#endif
      symbol = kernel_symbol;

      // \todo kzhuravl 10/15/15 This is a debugger backdoor: needs to be
      // removed.
      uint64_t target_address = sym->GetSection()->addr() + sym->SectionOffset() + ((size_t)(&((amd_kernel_code_t*)0)->runtime_loader_kernel_symbol));
      uint64_t source_value = (uint64_t) (uintptr_t) &kernel_symbol->debug_info;
      // Copy() return unchecked: debugger backdoor for compiler-generated kernel
      // symbols; target is a fixed offsetof within the symbol's loaded section.
      SymbolSegment(agent, sym)->Copy(target_address, &source_value, sizeof(source_value));
  } else {
    assert(!"Unexpected symbol type in LoadDefinitionSymbol");
    return HSA_STATUS_ERROR;
  }

  assert(symbol);
  if (isAgent) {
    symbol->agent = agent;
    agent_symbols_.insert(std::make_pair(std::make_pair(sym->Name(), agent), symbol));
  } else {
    program_symbols_.insert(std::make_pair(sym->Name(), symbol));
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::LoadDeclarationSymbol(hsa_agent_t agent,
                                                   code::Symbol* sym,
                                                   uint32_t majorVersion)
{
  auto program_symbol = program_symbols_.find(sym->Name());
  if (program_symbol == program_symbols_.end()) {
    auto agent_symbol = agent_symbols_.find(std::make_pair(sym->Name(), agent));
    if (agent_symbol == agent_symbols_.end()) {
      logger_ << "LoaderError: symbol \"" << sym->Name() << "\" is undefined\n";

      // TODO(spec): this is not spec compliant.
      return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
    }
  }
  return HSA_STATUS_SUCCESS;
}

Segment* ExecutableImpl::VirtualAddressSegment(uint64_t vaddr)
{
  for (auto &seg : loaded_code_objects.back()->LoadedSegments()) {
    if (seg->IsAddressInSegment(vaddr)) {
      return seg;
    }
  }
  return 0;
}

uint64_t ExecutableImpl::SymbolAddress(hsa_agent_t agent, code::Symbol* sym)
{
  code::Section* sec = sym->GetSection();
  Segment* seg = SectionSegment(agent, sec);
  return nullptr == seg ? 0 : (uint64_t) (uintptr_t) seg->Address(sym->VAddr());
}

uint64_t ExecutableImpl::SymbolAddress(hsa_agent_t agent, elf::Symbol* sym)
{
  elf::Section* sec = sym->section();
  if(!sec) { return NULL; }

  Segment* seg = SectionSegment(agent, sec);
  uint64_t vaddr = sec->addr() + sym->value();
  return nullptr == seg ? 0 : (uint64_t) (uintptr_t) seg->Address(vaddr);
}

Segment* ExecutableImpl::SymbolSegment(hsa_agent_t agent, code::Symbol* sym)
{
  return SectionSegment(agent, sym->GetSection());
}

Segment* ExecutableImpl::SectionSegment(hsa_agent_t agent, code::Section* sec)
{
  for (Segment* seg : loaded_code_objects.back()->LoadedSegments()) {
    if (seg->IsAddressInSegment(sec->addr())) {
      return seg;
    }
  }
  return 0;
}

hsa_status_t ExecutableImpl::ApplyRelocations(hsa_agent_t agent, amd::hsa::code::AmdHsaCode *c)
{
  hsa_status_t status = HSA_STATUS_SUCCESS;

  uint32_t majorVersion, minorVersion;
  if (!c->GetCodeObjectVersion(&majorVersion, &minorVersion)) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }

  for (size_t i = 0; i < c->RelocationSectionCount(); ++i) {
    if (c->GetRelocationSection(i)->targetSection()) {
      // Static relocations may be present if --emit-relocs
      // option was passed to lld, but they cannot be applied
      // again, so skip it for code object v2 and up.
      if (majorVersion >= 2) {
        continue;
      }

      status = ApplyStaticRelocationSection(agent, c->GetRelocationSection(i));
    } else {
      // Dynamic relocations are supported starting code object v2.1.
      if (majorVersion < 2) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      if (majorVersion == 2 && minorVersion < 1) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      status = ApplyDynamicRelocationSection(agent, c->GetRelocationSection(i));
    }
    if (status != HSA_STATUS_SUCCESS) { return status; }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::ApplyStaticRelocationSection(hsa_agent_t agent, amd::hsa::code::RelocationSection* sec)
{
  // Skip link-time relocations (if any).
  if (!(sec->targetSection()->flags() & SHF_ALLOC)) { return HSA_STATUS_SUCCESS; }
  hsa_status_t status = HSA_STATUS_SUCCESS;
  for (size_t i = 0; i < sec->relocationCount(); ++i) {
    status = ApplyStaticRelocation(agent, sec->relocation(i));
    if (status != HSA_STATUS_SUCCESS) { return status; }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::ApplyStaticRelocation(hsa_agent_t agent, amd::hsa::code::Relocation *rel)
{
  hsa_status_t status = HSA_STATUS_SUCCESS;
  amd::elf::Symbol* sym = rel->symbol();
  code::RelocationSection* rsec = rel->section();
  code::Section* sec = rsec->targetSection();
  Segment* rseg = SectionSegment(agent, sec);
  // SectionSegment() returns nullptr when no loaded segment covers the target
  // section (crafted sh_info/sh_addr); reject rather than dereferencing it.
  if (!rseg) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
  // sec->addr() + rel->offset() can wrap on crafted input; Copy()'s range check
  // rejects any wrapped destination that falls outside the target segment.
  size_t reladdr = sec->addr() + rel->offset();
  switch (rel->type()) {
    case R_AMDGPU_V1_32_LOW:
    case R_AMDGPU_V1_32_HIGH:
    case R_AMDGPU_V1_64:
    {
      uint64_t addr;
      switch (sym->type()) {
        case STT_OBJECT:
        case STT_SECTION:
        case STT_AMDGPU_HSA_KERNEL:
        case STT_AMDGPU_HSA_INDIRECT_FUNCTION:
          addr = SymbolAddress(agent, sym);
          if (!addr) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
          break;
        case STT_COMMON: {
          hsa_agent_t *sagent = &agent;
          if (STA_AMDGPU_HSA_GLOBAL_PROGRAM == ELF64_ST_AMDGPU_ALLOCATION(sym->other())) {
            sagent = nullptr;
          }
          SymbolImpl* esym = (SymbolImpl*) GetSymbolInternal(sym->name().c_str(), sagent);
          if (!esym) {
            logger_ << "LoaderError: symbol \"" << sym->name() << "\" is undefined\n";
            return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
          }
          addr = esym->address;
          break;
        }
        default:
          return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      addr += rel->addend();

      uint32_t addr32 = 0;
      switch (rel->type()) {
        case R_AMDGPU_V1_32_HIGH:
          addr32 = uint32_t((addr >> 32) & 0xFFFFFFFF);
          if (!rseg->Copy(reladdr, &addr32, sizeof(addr32))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
          break;
        case R_AMDGPU_V1_32_LOW:
          addr32 = uint32_t(addr & 0xFFFFFFFF);
          if (!rseg->Copy(reladdr, &addr32, sizeof(addr32))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
          break;
        case R_AMDGPU_V1_64:
          if (!rseg->Copy(reladdr, &addr, sizeof(addr))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
          break;
        default:
          return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      break;
    }

    case R_AMDGPU_V1_INIT_SAMPLER:
    {
      if (STT_AMDGPU_HSA_METADATA != sym->type() ||
          SHT_PROGBITS != sym->section()->type() ||
          !(sym->section()->flags() & SHF_MERGE)) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      amdgpu_hsa_sampler_descriptor_t desc;
      if (!sym->section()->getData(sym->value(), &desc, sizeof(desc))) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      if (AMDGPU_HSA_METADATA_KIND_INIT_SAMP != desc.kind) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }

      hsa_ext_sampler_descriptor_t hsa_sampler_descriptor;
      hsa_sampler_descriptor.coordinate_mode =
        hsa_ext_sampler_coordinate_mode_t(desc.coord);
      hsa_sampler_descriptor.filter_mode =
        hsa_ext_sampler_filter_mode_t(desc.filter);
      hsa_sampler_descriptor.address_mode =
        hsa_ext_sampler_addressing_mode_t(desc.addressing);

      hsa_ext_sampler_t hsa_sampler = {0};
      status = context_->SamplerCreate(agent, &hsa_sampler_descriptor, &hsa_sampler);
      if (status != HSA_STATUS_SUCCESS) { return status; }
      assert(hsa_sampler.handle);
      if (!rseg->Copy(reladdr, &hsa_sampler, sizeof(hsa_sampler))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
      break;
    }

    case R_AMDGPU_V1_INIT_IMAGE:
    {
      if (STT_AMDGPU_HSA_METADATA != sym->type() ||
          SHT_PROGBITS != sym->section()->type() ||
          !(sym->section()->flags() & SHF_MERGE)) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }

      amdgpu_hsa_image_descriptor_t desc;
      if (!sym->section()->getData(sym->value(), &desc, sizeof(desc))) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }
      if (AMDGPU_HSA_METADATA_KIND_INIT_ROIMG != desc.kind &&
          AMDGPU_HSA_METADATA_KIND_INIT_WOIMG != desc.kind &&
          AMDGPU_HSA_METADATA_KIND_INIT_RWIMG != desc.kind) {
        return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
      }

      hsa_ext_image_format_t hsa_image_format;
      hsa_image_format.channel_order =
        hsa_ext_image_channel_order_t(desc.channel_order);
      hsa_image_format.channel_type =
        hsa_ext_image_channel_type_t(desc.channel_type);

      hsa_ext_image_descriptor_t hsa_image_descriptor;
      hsa_image_descriptor.geometry =
        hsa_ext_image_geometry_t(desc.geometry);
      hsa_image_descriptor.width = size_t(desc.width);
      hsa_image_descriptor.height = size_t(desc.height);
      hsa_image_descriptor.depth = size_t(desc.depth);
      hsa_image_descriptor.array_size = size_t(desc.array);
      hsa_image_descriptor.format = hsa_image_format;

      hsa_access_permission_t hsa_image_permission = HSA_ACCESS_PERMISSION_RO;
      switch (desc.kind) {
        case AMDGPU_HSA_METADATA_KIND_INIT_ROIMG: {
          hsa_image_permission = HSA_ACCESS_PERMISSION_RO;
          break;
        }
        case AMDGPU_HSA_METADATA_KIND_INIT_WOIMG: {
          hsa_image_permission = HSA_ACCESS_PERMISSION_WO;
          break;
        }
        case AMDGPU_HSA_METADATA_KIND_INIT_RWIMG: {
          hsa_image_permission = HSA_ACCESS_PERMISSION_RW;
          break;
        }
        default: {
          assert(false);
          return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
        }
      }

      hsa_ext_image_t hsa_image = {0};
      status = context_->ImageCreate(agent, hsa_image_permission,
                                  &hsa_image_descriptor,
                                  NULL, // TODO: image_data?
                                  &hsa_image);
      if (status != HSA_STATUS_SUCCESS) { return status; }
      if (!rseg->Copy(reladdr, &hsa_image, sizeof(hsa_image))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
      break;
    }

    default:
      // Ignore.
      break;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::ApplyDynamicRelocationSection(hsa_agent_t agent, amd::hsa::code::RelocationSection* sec)
{
  hsa_status_t status = HSA_STATUS_SUCCESS;
  for (size_t i = 0; i < sec->relocationCount(); ++i) {
    status = ApplyDynamicRelocation(agent, sec->relocation(i));
    if (status != HSA_STATUS_SUCCESS) { return status; }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::ApplyDynamicRelocation(hsa_agent_t agent, amd::hsa::code::Relocation *rel)
{
  Segment* relSeg = VirtualAddressSegment(rel->offset());
  // VirtualAddressSegment() returns nullptr when no loaded segment covers the
  // attacker-controlled r_offset; reject rather than dereferencing it.
  if (!relSeg) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
  uint64_t symAddr = 0;
  switch (rel->symbol()->type()) {
    case STT_OBJECT:
    case STT_AMDGPU_HSA_KERNEL:
    case STT_FUNC:
    {
      Segment* symSeg = VirtualAddressSegment(rel->symbol()->value());
      if (!symSeg) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
      symAddr = reinterpret_cast<uint64_t>(symSeg->Address(rel->symbol()->value()));
      break;
    }

    // External symbols, they must be defined prior loading.
    case STT_NOTYPE:
    {
      // TODO: Only agent allocation variables are supported in v2.1. How will
      // we distinguish between program allocation and agent allocation
      // variables?
      auto agent_symbol = agent_symbols_.find(std::make_pair(rel->symbol()->name(), agent));
      if (agent_symbol != agent_symbols_.end())
        symAddr = agent_symbol->second->address;
      break;
    }

    default:
      // Only objects and kernels are supported in v2.1.
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  symAddr += rel->addend();

  switch (rel->type()) {
    case ELF::R_AMDGPU_ABS32_HI:
    {
      if (!symAddr) {
        logger_ << "LoaderError: symbol \"" << rel->symbol()->name() << "\" is undefined\n";
        return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
      }

      uint32_t symAddr32 = uint32_t((symAddr >> 32) & 0xFFFFFFFF);
      if (!relSeg->Copy(rel->offset(), &symAddr32, sizeof(symAddr32))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
      break;
    }

    case ELF::R_AMDGPU_ABS32_LO:
    {
      if (!symAddr) {
        logger_ << "LoaderError: symbol \"" << rel->symbol()->name() << "\" is undefined\n";
        return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
      }

      uint32_t symAddr32 = uint32_t(symAddr & 0xFFFFFFFF);
      if (!relSeg->Copy(rel->offset(), &symAddr32, sizeof(symAddr32))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
      break;
    }

    case ELF::R_AMDGPU_ABS32:
    {
      if (!symAddr) {
        logger_ << "LoaderError: symbol \"" << rel->symbol()->name() << "\" is undefined\n";
        return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
      }

      uint32_t symAddr32 = uint32_t(symAddr);
      if (!relSeg->Copy(rel->offset(), &symAddr32, sizeof(symAddr32))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
      break;
    }

    case ELF::R_AMDGPU_ABS64:
    {
      if (!symAddr) {
        logger_ << "LoaderError: symbol \"" << rel->symbol()->name() << "\" is undefined\n";
        return HSA_STATUS_ERROR_VARIABLE_UNDEFINED;
      }

      if (!relSeg->Copy(rel->offset(), &symAddr, sizeof(symAddr))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
      break;
    }

    case ELF::R_AMDGPU_RELATIVE64:
    {
      int64_t baseDelta = reinterpret_cast<uint64_t>(relSeg->Address(0)) - relSeg->VAddr();
      uint64_t relocatedAddr = baseDelta + rel->addend();
      if (!relSeg->Copy(rel->offset(), &relocatedAddr, sizeof(relocatedAddr))) { return HSA_STATUS_ERROR_INVALID_CODE_OBJECT; }
      break;
    }

    default:
      return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t ExecutableImpl::Freeze(const char *options) {
  amd::hsa::common::WriterLockGuard<amd::hsa::common::ReaderWriterLock> writer_lock(rw_lock_);
  if (HSA_EXECUTABLE_STATE_FROZEN == state_) {
    return HSA_STATUS_ERROR_FROZEN_EXECUTABLE;
  }

  for (auto &lco : loaded_code_objects) {
    for (auto &ls : lco->LoadedSegments()) {
      ls->Freeze();
    }
  }

  // Trampoline pools are not part of any LoadedCodeObject's segment list
  // (that must stay size==1 for v2+); freeze them explicitly so their host->device
  // DMA and code-cache invalidation happen alongside the code segments.
  for (auto& ts : trampoline_segments_) {
    ts->Freeze();
  }

  state_ = HSA_EXECUTABLE_STATE_FROZEN;
  return HSA_STATUS_SUCCESS;
}

void ExecutableImpl::Print(std::ostream& out)
{
  out << "AMD Executable" << std::endl;
  out << "  Id: " << id()
      << "  Profile: " << HsaProfileToString(profile())
      << std::endl << std::endl;
  out << "Loaded Objects (total " << objects.size() << ")" << std::endl;
  size_t i = 0;
  for (const auto& o : objects) {
    out << "Loaded Object " << i++ << ": ";
    o->Print(out);
    out << std::endl;
  }
  out << "End AMD Executable" << std::endl;
}

bool ExecutableImpl::PrintToFile(const std::string& filename)
{
  std::ofstream out(filename);
  if (out.fail()) { return false; }
  Print(out);
  return out.fail();
}

} // namespace loader
} // namespace hsa
} // namespace amd
} // namespace rocr
