////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and HSA Software Development
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
//    notice, this list of conditions and the following disclaimers in the
//    documentation and/or other materials provided with the distribution.
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

#ifndef HSA_RUNTIME_CORE_INC_HOTSWAP_HPP_
#define HSA_RUNTIME_CORE_INC_HOTSWAP_HPP_

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "core/inc/amd_hsa_loader.hpp"
#include "inc/hsa.h"

namespace rocr {
namespace hotswap {

struct AgentGfxRevision;

using OwnedElfBuffer = std::unique_ptr<void, decltype(&std::free)>;

struct CodeObjectView {
  const void* data = nullptr;
  size_t size = 0;
  std::string uri;
};

// Entry-trampoline rewriting is opt-in while validation is ongoing.
inline constexpr bool kDefaultEntryTrampolinesEnabled = false;

struct RewriteOptions {
  bool entry_trampolines_enabled = kDefaultEntryTrampolinesEnabled;
  bool strict_mode_enabled = false;
};

struct RewriteDecision {
  std::string source_isa;
  std::string target_isa;
  bool request_entry_trampolines = false;
  bool request_strict_mode = false;
  bool rewrite_required = false;
};

enum class RetargetCodeObjectStatus {
  kSkipped,
  kRewritten,
  kRequiredRewriteFailed,
};

struct RetargetCodeObjectResult {
  RetargetCodeObjectStatus status = RetargetCodeObjectStatus::kSkipped;
  bool rewrite_required = false;
};

using LoadOriginalCodeObjectFn = hsa_status_t (*)(
    void* context, hsa_agent_t agent, hsa_code_object_t code_object,
    const char* options, const std::string& uri,
    hsa_loaded_code_object_t* loaded_code_object);

using LoadCodeObjectWithSizeFn = hsa_status_t (*)(
    void* context, hsa_agent_t agent, hsa_code_object_t code_object,
    size_t code_object_size, const char* options, const std::string& uri,
    hsa_loaded_code_object_t* loaded_code_object);

struct LoadAgentCodeObjectCallbacks {
  void* context = nullptr;
  LoadOriginalCodeObjectFn load_original_code_object = nullptr;
  LoadCodeObjectWithSizeFn load_rewritten_code_object = nullptr;
};

std::string GetCodeObjectIsaName(const void* elf_data, size_t elf_size);

bool RetargetCodeObject(const void* elf_data, size_t elf_size,
                        const char* source_isa, const char* target_isa,
                        OwnedElfBuffer* out_elf_buffer, size_t* out_elf_size,
                        bool request_entry_trampolines = false,
                        bool request_strict_mode = false);

RetargetCodeObjectResult TryRetargetCodeObject(
    const CodeObjectView& code_object, hsa_agent_t agent,
    OwnedElfBuffer* out_elf_buffer, size_t* out_elf_size);

RetargetCodeObjectResult TryRetargetCodeObject(
    amd::hsa::loader::CodeObjectReaderImpl* reader, hsa_agent_t agent,
    OwnedElfBuffer* out_elf_buffer, size_t* out_elf_size);

hsa_status_t LoadAgentCodeObjectWithHotswap(
    hsa_executable_t executable, hsa_agent_t agent,
    const CodeObjectView& code_object, const char* options,
    hsa_loaded_code_object_t* loaded_code_object,
    const LoadAgentCodeObjectCallbacks& callbacks);

void RetainRewrittenElfBuffer(hsa_executable_t executable,
                              OwnedElfBuffer elf_buffer);
void ReleaseRetainedRewrittenElfBuffers(hsa_executable_t executable);

// Lifecycle hooks for the background disk-cache writer thread. Called by the
// ROCr Runtime: HotswapCacheStartup() from Runtime::Load(), and
// HotswapCacheShutdown() from Runtime::Unload(). Both are idempotent and safe
// to call when the disk cache is unsupported/disabled (they become no-ops).
//
// Fork note: like ROCr generally, the writer is not fork-safe. A child forked
// while a write is in flight inherits a locked mutex and no writer thread; the
// child must not call these or trigger a retarget until it re-inits the runtime
// (Runtime::Unload/Load). HotswapCacheShutdown() in the child is unsafe if the
// mutex was held at fork; a fresh Runtime init in the child starts a new writer.
void HotswapCacheStartup();
void HotswapCacheShutdown();

#ifdef ROCR_HOTSWAP_TESTING
std::optional<RewriteDecision> DecideHotswapRewriteForTesting(
    const AgentGfxRevision& gfx, const std::string& source_isa,
    const std::string& target_isa, const RewriteOptions& options);
size_t RetainedRewrittenElfBufferCountForTesting(hsa_executable_t executable);
bool HotswapRewriteWithOptionsAvailableForTesting();
void ForceRetargetCodeObjectFailureForTesting(bool force);
size_t RetargetCacheSizeForTesting();
void ClearRetargetCacheForTesting();
// Bytes currently held by the in-memory retarget cache.
size_t RetargetCacheBytesForTesting();
// Sets the in-memory cache byte budget and re-evicts to honor it immediately.
void SetRetargetCacheByteBudgetForTesting(size_t budget);
// Inserts a synthetic success entry of `size` zero-filled bytes under `key`.
void PutSyntheticRetargetCacheEntryForTesting(uint64_t key, size_t size);
// Inserts a failure sentinel (no buffer, consumes no budget) under `key`.
void PutFailureRetargetCacheEntryForTesting(uint64_t key);
// Returns true if `key` is currently resident in the in-memory cache.
bool RetargetCacheContainsForTesting(uint64_t key);
// Performs a real cache Get (refreshes LRU recency) for `key`; returns true on
// a success hit and, if `out_bytes` is non-null, the shared buffer handle.
bool RetargetCacheGetForTesting(uint64_t key,
                                std::shared_ptr<std::vector<uint8_t>>* out_bytes);
// Synchronously writes a disk cache entry under `dir` (bypasses the async
// writer). Returns false if disk cache support is compiled out.
bool DiskCacheWriteForTesting(const std::string& dir, uint64_t key,
                              uint64_t salt,
                              const std::vector<uint8_t>& payload);
// Reads a disk cache entry; returns true and fills `out_payload` on a validated
// hit, false on miss/mismatch/unsupported.
bool DiskCacheReadForTesting(const std::string& dir, uint64_t key, uint64_t salt,
                             std::vector<uint8_t>* out_payload);
// Drives the async DiskWriter: start, enqueue `n` writes, Stop() (must drain
// all before joining), then count readable-back entries. Returns that count
// (== n if the drain-at-shutdown path is correct), or -1 if unsupported.
int DiskWriterDrainRoundTripForTesting(const std::string& dir, int n,
                                       const std::vector<uint8_t>& payload);
#endif

}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_HPP_
