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

// Entry-trampoline hotswap rewriting for gfx12.5 targets (the generic
// gfx12.5 rewrite path and the gfx1250 B0->A0 retarget) is opt-in: disabled
// by default and enabled only when the caller sets
// AMD_COMGR_HOTSWAP_ENTRY_TRAMPOLINES to a truthy value.
inline constexpr bool kDefaultEntryTrampolinesEnabled = false;

struct RewriteOptions {
  bool entry_trampolines_enabled = kDefaultEntryTrampolinesEnabled;
};

struct RewriteDecision {
  std::string source_isa;
  std::string target_isa;
  bool request_entry_trampolines = false;
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
                        bool request_entry_trampolines = false);

bool TryRetargetCodeObject(const CodeObjectView& code_object, hsa_agent_t agent,
                           OwnedElfBuffer* out_elf_buffer,
                           size_t* out_elf_size);

bool TryRetargetCodeObject(amd::hsa::loader::CodeObjectReaderImpl* reader,
                           hsa_agent_t agent, OwnedElfBuffer* out_elf_buffer,
                           size_t* out_elf_size);

hsa_status_t LoadAgentCodeObjectWithHotswap(
    hsa_executable_t executable, hsa_agent_t agent,
    const CodeObjectView& code_object, const char* options,
    hsa_loaded_code_object_t* loaded_code_object,
    const LoadAgentCodeObjectCallbacks& callbacks);

void RetainRewrittenElfBuffer(hsa_executable_t executable,
                              OwnedElfBuffer elf_buffer);
void ReleaseRetainedRewrittenElfBuffers(hsa_executable_t executable);

#ifdef ROCR_HOTSWAP_TESTING
std::optional<RewriteDecision> DecideHotswapRewriteForTesting(
    const AgentGfxRevision& gfx, const std::string& source_isa,
    const std::string& target_isa, const RewriteOptions& options);
size_t RetainedRewrittenElfBufferCountForTesting(hsa_executable_t executable);
bool EntryTrampolineRewriteAvailableForTesting();
#endif

}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_HPP_
