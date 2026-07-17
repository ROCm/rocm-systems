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
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "core/inc/amd_hsa_loader.hpp"
#include "inc/hsa.h"

namespace rocr {
namespace hotswap {

struct AgentGfxRevision;

using OwnedElfBuffer = std::unique_ptr<void, decltype(&std::free)>;

class RetargetedElf final {
 public:
  RetargetedElf(OwnedElfBuffer bytes, size_t size) : bytes_(std::move(bytes)), size_(size) {}

  const void* data() const { return bytes_.get(); }
  size_t size() const { return size_; }

 private:
  OwnedElfBuffer bytes_;
  size_t size_ = 0;
};

using RetargetedElfRef = std::shared_ptr<const RetargetedElf>;

enum class RetargetError {
  kNone,
  kInvalidArgument,
  kComgrUnavailable,
  kComgrFailure,
  kOutOfResources,
};

enum class RetargetResultSource {
  kComputed,
  kReadyCache,
  kCoalesced,
};

struct RetargetOperationResult {
  RetargetedElfRef elf;
  RetargetError error = RetargetError::kNone;
  RetargetResultSource source = RetargetResultSource::kComputed;

  bool succeeded() const { return elf != nullptr; }
};

struct RetargetCacheKey {
  std::string source_isa;
  std::string target_isa;
  bool entry_trampolines = false;
  bool strict_mode = false;

  bool operator==(const RetargetCacheKey& other) const {
    return source_isa == other.source_isa && target_isa == other.target_isa &&
        entry_trampolines == other.entry_trampolines && strict_mode == other.strict_mode;
  }
};

class ReaderRetargetCache final {
 public:
  ReaderRetargetCache();
  ~ReaderRetargetCache();

  ReaderRetargetCache(const ReaderRetargetCache&) = delete;
  ReaderRetargetCache& operator=(const ReaderRetargetCache&) = delete;

  RetargetOperationResult GetOrCompute(const RetargetCacheKey& key,
                                       const std::function<RetargetOperationResult()>& producer);

#ifdef ROCR_HOTSWAP_TESTING
  size_t ReadyEntryCountForTesting() const;
  size_t WaiterCountForTesting(const RetargetCacheKey& key) const;
#endif

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

struct CodeObjectView {
  const void* data = nullptr;
  size_t size = 0;
  std::string uri;
  amd::hsa::loader::CodeObjectReaderImpl* reader = nullptr;
  std::shared_ptr<ReaderRetargetCache> retarget_cache;
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
  RetargetedElfRef elf;
  RetargetError error = RetargetError::kNone;
};

using LoadOriginalCodeObjectFn = hsa_status_t (*)(
    void* context, hsa_agent_t agent, hsa_code_object_t code_object,
    const char* options, const std::string& uri,
    hsa_loaded_code_object_t* loaded_code_object);

using LoadCodeObjectWithSizeFn = hsa_status_t (*)(
    void* context, hsa_agent_t agent, hsa_code_object_t code_object, size_t code_object_size,
    amd::hsa::loader::CodeObjectMemoryOwner code_object_owner, const char* options,
    const std::string& uri, hsa_loaded_code_object_t* loaded_code_object);

struct LoadAgentCodeObjectCallbacks {
  void* context = nullptr;
  LoadOriginalCodeObjectFn load_original_code_object = nullptr;
  LoadCodeObjectWithSizeFn load_rewritten_code_object = nullptr;
};

std::string GetCodeObjectIsaName(const void* elf_data, size_t elf_size);

RetargetOperationResult RetargetCodeObject(const void* elf_data, size_t elf_size,
                                           const char* source_isa, const char* target_isa,
                                           bool request_entry_trampolines = false,
                                           bool request_strict_mode = false);

RetargetCodeObjectResult TryRetargetCodeObject(const CodeObjectView& code_object,
                                               hsa_agent_t agent);

RetargetCodeObjectResult TryRetargetCodeObject(amd::hsa::loader::CodeObjectReaderImpl* reader,
                                               hsa_agent_t agent);

hsa_status_t LoadAgentCodeObjectWithHotswap(
    hsa_executable_t executable, hsa_agent_t agent,
    const CodeObjectView& code_object, const char* options,
    hsa_loaded_code_object_t* loaded_code_object,
    const LoadAgentCodeObjectCallbacks& callbacks);

#ifdef ROCR_HOTSWAP_TESTING
std::optional<RewriteDecision> DecideHotswapRewriteForTesting(const AgentGfxRevision& gfx,
                                                              const std::string& source_isa,
                                                              const std::string& target_isa,
                                                              const RewriteOptions& options);
bool HotswapRewriteWithOptionsAvailableForTesting();
void ForceRetargetCodeObjectFailureForTesting(bool force);
#endif

}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_HPP_
