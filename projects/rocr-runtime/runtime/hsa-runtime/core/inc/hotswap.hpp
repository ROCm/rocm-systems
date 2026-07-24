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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/inc/amd_hsa_loader.hpp"
#include "inc/hsa.h"

namespace rocr {
namespace hotswap {

struct AgentGfxRevision;

using OwnedElfBuffer = std::unique_ptr<void, decltype(&std::free)>;

class RetargetCacheMemoryTracker;

class SourceSnapshot final {
 public:
  ~SourceSnapshot();

  const void* data() const { return bytes_.get(); }
  size_t size() const { return size_; }
  bool Equals(const void* data, size_t size) const;

 private:
  friend class ContentRetargetCache;
  friend class RetargetedElf;

  SourceSnapshot(std::unique_ptr<unsigned char[]> bytes, size_t size,
                 std::shared_ptr<RetargetCacheMemoryTracker> tracker);

  std::unique_ptr<unsigned char[]> bytes_;
  size_t size_ = 0;
  std::shared_ptr<RetargetCacheMemoryTracker> tracker_;
};

using SourceSnapshotRef = std::shared_ptr<const SourceSnapshot>;

class RetargetedElf final {
 public:
  RetargetedElf(OwnedElfBuffer bytes, size_t size, SourceSnapshotRef source = {});
  ~RetargetedElf();

  const void* data() const { return bytes_.get(); }
  size_t size() const { return size_; }
  const SourceSnapshotRef& source() const { return source_; }

 private:
  OwnedElfBuffer bytes_;
  size_t size_ = 0;
  SourceSnapshotRef source_;
  std::shared_ptr<RetargetCacheMemoryTracker> tracker_;
};

using RetargetedElfRef = std::shared_ptr<const RetargetedElf>;

enum class RetargetError {
  kNone,
  kInvalidArgument,
  kComgrUnavailable,
  kComgrFailure,
  kOutOfResources,
  kReentrantRequest,
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

struct RetargetCacheMetrics {
  uint64_t producer_calls = 0;
  uint64_t producer_failures = 0;
  uint64_t ready_hits = 0;
  uint64_t cross_reader_results = 0;
  uint64_t coalesced_results = 0;
  uint64_t reentrant_rejections = 0;
  uint64_t hash_bytes = 0;
  uint64_t hash_nanoseconds = 0;
  uint64_t exact_compare_bytes = 0;
  uint64_t exact_compare_nanoseconds = 0;
  uint64_t wait_nanoseconds = 0;
  uint64_t lock_hold_nanoseconds = 0;
  uint64_t source_snapshot_allocations = 0;
  uint64_t source_snapshot_bytes = 0;
  uint64_t live_source_snapshot_bytes = 0;
  uint64_t peak_live_source_snapshot_bytes = 0;
  uint64_t produced_output_bytes = 0;
  uint64_t live_output_bytes = 0;
  uint64_t peak_live_output_bytes = 0;
  size_t bucket_entries = 0;
  size_t content_bucket_entries = 0;
  size_t transform_bucket_entries = 0;
  size_t ready_entries = 0;
  size_t in_flight_entries = 0;
};

using DiskCacheDigest = std::array<uint8_t, 32>;

struct DiskCacheMetrics {
  uint64_t queued_tasks = 0;
  uint64_t queued_bytes = 0;
  uint64_t peak_queued_tasks = 0;
  uint64_t peak_queued_bytes = 0;
  uint64_t dropped_tasks = 0;
  uint64_t dropped_bytes = 0;
  uint64_t completed_tasks = 0;
  uint64_t completed_bytes = 0;
};

class ContentRetargetCache final {
 public:
  using Producer = std::function<RetargetOperationResult(const SourceSnapshotRef& source)>;

  ContentRetargetCache();
  ~ContentRetargetCache();

  ContentRetargetCache(const ContentRetargetCache&) = delete;
  ContentRetargetCache& operator=(const ContentRetargetCache&) = delete;

  RetargetOperationResult GetOrCompute(const void* source_data, size_t source_size,
                                       uint64_t reader_id, const RetargetCacheKey& key,
                                       const Producer& producer);

  RetargetCacheMetrics SnapshotMetrics() const;

#ifdef ROCR_HOTSWAP_TESTING
  using ContentHashFunction = std::function<uint64_t(const void*, size_t)>;

  explicit ContentRetargetCache(ContentHashFunction hash_function);

  size_t ReadyEntryCountForTesting() const;
  size_t WaiterCountForTesting(const void* source_data, size_t source_size,
                               const RetargetCacheKey& key) const;
  void ResetMetricsForTesting();
#endif

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

ContentRetargetCache& GetProcessRetargetCache();

struct CodeObjectView {
  const void* data = nullptr;
  size_t size = 0;
  std::string uri;
  uint64_t reader_id = 0;
  ContentRetargetCache* retarget_cache = nullptr;
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
                                           bool request_strict_mode = false,
                                           SourceSnapshotRef source_snapshot = {});

RetargetCodeObjectResult TryRetargetCodeObject(const CodeObjectView& code_object,
                                               hsa_agent_t agent);

RetargetCodeObjectResult TryRetargetCodeObject(amd::hsa::loader::CodeObjectReaderImpl* reader,
                                               hsa_agent_t agent);

hsa_status_t LoadAgentCodeObjectWithHotswap(
    hsa_executable_t executable, hsa_agent_t agent,
    const CodeObjectView& code_object, const char* options,
    hsa_loaded_code_object_t* loaded_code_object,
    const LoadAgentCodeObjectCallbacks& callbacks);

// Requests that the lazily-created disk writer stop and discards queued work.
// Runtime::Unload() calls this while serialized by the bootstrap lock.
void HotswapCacheShutdown();
// Joins at most the write already in progress. Runtime::Release() calls this
// after releasing the bootstrap lock.
void HotswapCacheWaitForShutdown();
DiskCacheMetrics GetDiskCacheMetrics();

#ifdef ROCR_HOTSWAP_TESTING
std::optional<RewriteDecision> DecideHotswapRewriteForTesting(const AgentGfxRevision& gfx,
                                                              const std::string& source_isa,
                                                              const std::string& target_isa,
                                                              const RewriteOptions& options);
bool HotswapRewriteWithOptionsAvailableForTesting();
bool ComgrCacheIdentifierAvailableForTesting();
void SetComgrCacheFingerprintForTesting(const DiskCacheDigest* fingerprint);
void ForceRetargetCodeObjectFailureForTesting(bool force);
DiskCacheDigest DigestBytesForTesting(const void* data, size_t size);
DiskCacheDigest ComputeRetargetCacheDigestForTesting(const void* elf_data, size_t elf_size,
                                                     const std::string& source_isa,
                                                     const std::string& target_isa,
                                                     bool entry_trampolines, bool strict_mode,
                                                     const DiskCacheDigest& comgr_fingerprint);
// Synchronously writes a disk cache entry under `dir` (bypasses the async
// writer). Returns false if disk cache support is compiled out.
bool DiskCacheWriteForTesting(const std::string& dir, const DiskCacheDigest& key,
                              const DiskCacheDigest& comgr_fingerprint,
                              const std::vector<uint8_t>& payload);
// Reads a disk cache entry; returns true and fills `out_payload` on a validated
// hit, false on miss/mismatch/unsupported.
bool DiskCacheReadForTesting(const std::string& dir, const DiskCacheDigest& key,
                             const DiskCacheDigest& comgr_fingerprint,
                             std::vector<uint8_t>* out_payload);
std::string DiskCachePathForTesting(const std::string& dir, const DiskCacheDigest& key,
                                    const DiskCacheDigest& comgr_fingerprint);
bool DiskWriterConstructedForTesting();
void ConfigureDiskWriterForTesting(size_t max_tasks, size_t max_bytes);
bool EnqueueDiskWriteForTesting(const std::string& dir, const DiskCacheDigest& key,
                                const DiskCacheDigest& comgr_fingerprint,
                                const std::vector<uint8_t>& payload);
void BlockDiskWritesForTesting(bool block);
bool WaitForDiskWriterIdleForTesting(uint64_t timeout_milliseconds);
bool DiskCacheEnabledForTesting();
void ResetDiskWriterForTesting();
#endif

}  // namespace hotswap
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_HOTSWAP_HPP_
