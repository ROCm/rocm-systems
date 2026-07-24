/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HSA_RUNTIME_LOADER_HOTSWAP_KERNEL_REGISTRY_H_
#define HSA_RUNTIME_LOADER_HOTSWAP_KERNEL_REGISTRY_H_

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Bookkeeping for the HotSwap dispatch intercept. The loader records every
// kernel object it loads here so the dispatch-time hook can, given a packet's
// kernel_object, decide whether to pass it through, translate it, or reject it,
// and can compute the correct segment sizes for the translated kernel. All state
// is header-only and inert unless presentation mode is active.
namespace rocr {
namespace amd {
namespace hsa {
namespace loader {

struct HotSwapLazyCodeObject;

// Registry invariants must never be violated at runtime; abort loudly if they are
// (rather than silently corrupting dispatch), so the failure is caught in testing.
[[noreturn]] inline void ReportHotSwapRegistryError(const char* reason) {
  std::fprintf(stderr, "hotswap: registry invariant violation: %s\n", reason);
  std::abort();
}

// How a registered kernel object should be treated at dispatch:
//   Untranslated          - not a HotSwap kernel; must not reach the hardware.
//   LazySource            - source-ISA kernel to translate via COMGR on first use.
//   Translated            - already translated to the execution ISA; pass
//                           through target records or rewrite promoted source
//                           records to their target kernel object.
//   RuntimeTargetInternal - runtime-owned kernel already built for the execution
//                           ISA (e.g. ROCr blit shaders); pass through.
enum class HotSwapKernelKind {
  Untranslated,
  LazySource,
  Translated,
  RuntimeTargetInternal,
};

inline const char* HotSwapKernelKindName(HotSwapKernelKind Kind) {
  switch (Kind) {
    case HotSwapKernelKind::Untranslated:
      return "untranslated";
    case HotSwapKernelKind::LazySource:
      return "lazy_source";
    case HotSwapKernelKind::Translated:
      return "translated";
    case HotSwapKernelKind::RuntimeTargetInternal:
      return "runtime_target_internal";
  }
  return "untranslated";
}

// Per-kernel-object state: identity and kind, the retained lazy source (until
// translated), the resulting target kernel object, and the source/target segment
// sizes used to patch each dispatch packet. Kind is atomic so a successful lazy
// translation can publish a terminal Translated state after target metadata is
// complete; the other mutable translation fields are guarded by Mutex.
struct HotSwapKernelRecord {
  std::string Name;
  std::atomic<HotSwapKernelKind> Kind{HotSwapKernelKind::Untranslated};
  std::string SourceKind;
  std::string TargetGfx;
  std::shared_ptr<HotSwapLazyCodeObject> LazyCodeObject;
  std::mutex Mutex;
  bool TranslationAttempted = false;
  bool TranslationSucceeded = false;
  std::string Failure;
  uint64_t SourceKernelObject = 0;
  uint64_t ObjectSize = 0;
  uint64_t TargetKernelObject = 0;
  uint32_t SourcePrivateSegmentSize = 0;
  uint32_t SourceGroupSegmentSize = 0;
  uint32_t TargetPrivateSegmentSize = 0;
  uint32_t TargetGroupSegmentSize = 0;
  uint32_t TargetGroupSegmentLimit = UINT32_MAX;

  bool IsRegisteredRocrBlit(uint64_t address) const {
    return Kind.load(std::memory_order_acquire) ==
               HotSwapKernelKind::RuntimeTargetInternal &&
           SourceKind == "rocr_blit" && address >= SourceKernelObject &&
           address - SourceKernelObject < ObjectSize;
  }
};

// Translates a dispatch packet's segment sizes from the source kernel to the
// translated target kernel: preserves any caller-supplied dynamic (runtime)
// portion while substituting the target's fixed sizes. Fails if the packet is
// smaller than the source fixed size, on overflow, or if the result exceeds the
// target's LDS limit.
inline bool ComputeHotSwapPatchedSegmentSizes(
    const HotSwapKernelRecord& record, uint32_t packet_private_segment_size,
    uint32_t packet_group_segment_size, uint32_t& patched_private_segment_size,
    uint32_t& patched_group_segment_size, std::string& failure) {
  if (packet_group_segment_size < record.SourceGroupSegmentSize) {
    failure = "dispatch group segment size is below source fixed size";
    return false;
  }
  if (packet_private_segment_size < record.SourcePrivateSegmentSize) {
    failure = "dispatch private segment size is below source fixed size";
    return false;
  }

  const uint32_t dynamic_group_segment_size =
      packet_group_segment_size - record.SourceGroupSegmentSize;
  const uint32_t dynamic_private_segment_size =
      packet_private_segment_size - record.SourcePrivateSegmentSize;
  if (dynamic_private_segment_size >
      UINT32_MAX - record.TargetPrivateSegmentSize) {
    failure = "patched private segment size overflows";
    return false;
  }
  if (dynamic_group_segment_size > UINT32_MAX - record.TargetGroupSegmentSize) {
    failure = "patched group segment size overflows";
    return false;
  }

  patched_private_segment_size =
      record.TargetPrivateSegmentSize + dynamic_private_segment_size;
  patched_group_segment_size =
      record.TargetGroupSegmentSize + dynamic_group_segment_size;
  if (patched_group_segment_size > record.TargetGroupSegmentLimit) {
    failure = "patched group segment size exceeds target LDS limit";
    return false;
  }
  return true;
}

// Rewrites a source-ISA dispatch packet to the translated target kernel while
// preserving any dynamic segment-size deltas carried by the packet.
inline bool PatchHotSwapTranslatedDispatch(
    const HotSwapKernelRecord& record, uint64_t& address,
    uint32_t& private_segment_size, uint32_t& group_segment_size,
    std::string& failure) {
  if (record.TargetKernelObject == 0) {
    failure = "translated kernel object is missing";
    return false;
  }

  uint32_t patched_private_segment_size = 0;
  uint32_t patched_group_segment_size = 0;
  if (!ComputeHotSwapPatchedSegmentSizes(record, private_segment_size,
                                         group_segment_size,
                                         patched_private_segment_size,
                                         patched_group_segment_size, failure))
    return false;

  address = record.TargetKernelObject;
  private_segment_size = patched_private_segment_size;
  group_segment_size = patched_group_segment_size;
  failure.clear();
  return true;
}

inline std::string ResolveLazyHotSwapCacheDir(const char* lazy_cache_dir,
                                              const char* shared_cache_dir) {
  if (lazy_cache_dir && lazy_cache_dir[0]) return lazy_cache_dir;
  if (!shared_cache_dir || !shared_cache_dir[0]) return "";

  std::string dir(shared_cache_dir);
  while (!dir.empty() && dir.back() == '/') dir.pop_back();
  if (dir.empty()) return "";
  return dir + "/lazy-v2";
}

// The dispatch intercept only supports MULTI queues; single-producer queues do
// not guarantee the doorbell-per-range ordering the intercept relies on.
inline bool IsHotSwapQueueTypeSupported(hsa_queue_type32_t queue_type) {
  return queue_type == HSA_QUEUE_TYPE_MULTI;
}

// Derives the canonical kernel name used as a registry/record key from a symbol
// name, stripping the ".kd" descriptor suffix so the descriptor and kernel
// symbols map to the same record.
inline bool DeriveHotSwapKernelRecordName(const std::string& symbol_name,
                                          bool descriptor_symbol,
                                          std::string& record_name,
                                          std::string& failure) {
  constexpr const char* DescriptorSuffix = ".kd";
  constexpr size_t DescriptorSuffixSize = 3;
  if (descriptor_symbol) {
    if (symbol_name.size() <= DescriptorSuffixSize ||
        symbol_name.compare(symbol_name.size() - DescriptorSuffixSize,
                            DescriptorSuffixSize, DescriptorSuffix) != 0) {
      failure = "kernel descriptor symbol does not end in .kd";
      record_name.clear();
      return false;
    }
    record_name = symbol_name.substr(0, symbol_name.size() - DescriptorSuffixSize);
  } else {
    record_name = symbol_name;
    if (record_name.size() > DescriptorSuffixSize &&
        record_name.compare(record_name.size() - DescriptorSuffixSize,
                            DescriptorSuffixSize, DescriptorSuffix) == 0)
      record_name.resize(record_name.size() - DescriptorSuffixSize);
  }
  if (record_name.empty()) {
    failure = "kernel symbol name is empty";
    return false;
  }
  failure.clear();
  return true;
}

// Thread-safe registry from exact kernel object address to its HotSwap record.
// ROCr blit shaders are also tracked in a small side list so range fallback only
// scans runtime-owned blit records, not every user kernel loaded in the process.
class HotSwapKernelRegistry {
 public:
  std::shared_ptr<HotSwapKernelRecord> RegisterKernelObject(
      uint64_t address, const std::string& name, HotSwapKernelKind kind,
      std::shared_ptr<HotSwapLazyCodeObject> lazy_code_object) {
    if (address == 0)
      ReportHotSwapRegistryError("kernel object address is zero");
    std::lock_guard<std::mutex> guard(mutex_);
    if (records_.find(address) != records_.end())
      ReportHotSwapRegistryError("duplicate kernel object address");
    auto record = std::make_shared<HotSwapKernelRecord>();
    record->Name = name;
    record->Kind.store(kind, std::memory_order_relaxed);
    record->SourceKernelObject = address;
    record->LazyCodeObject = std::move(lazy_code_object);
    records_[address] = record;
    return record;
  }

  std::shared_ptr<HotSwapKernelRecord> RegisterRocrBlitTargetKernelObject(
      uint64_t address, uint64_t size, const std::string& target_gfx,
      const std::string& reason, const std::string& invalid_reason) {
    const bool valid =
        address != 0 && size != 0 && !target_gfx.empty() &&
        target_gfx != "<unknown>";
    std::lock_guard<std::mutex> guard(mutex_);
    auto record = std::make_shared<HotSwapKernelRecord>();
    record->Name = "rocr_blit";
    record->Kind.store(valid ? HotSwapKernelKind::RuntimeTargetInternal
                             : HotSwapKernelKind::Untranslated,
                       std::memory_order_relaxed);
    record->SourceKind = "rocr_blit";
    record->TargetGfx = target_gfx;
    record->Failure = valid ? reason : invalid_reason;
    record->SourceKernelObject = address;
    record->ObjectSize = size;
    if (!valid) return record;
    if (records_.find(address) != records_.end())
      ReportHotSwapRegistryError("duplicate ROCR blit kernel object address");
    records_[address] = record;
    rocr_blit_records_.push_back(record);
    return record;
  }

  void UnregisterLoadedKernelObject(uint64_t address) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = records_.find(address);
    if (it == records_.end()) return;
    records_.erase(it);
  }

  bool UnregisterRocrBlitTargetKernelObject(uint64_t address) {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = records_.find(address);
    if (it == records_.end() || it->second->SourceKind != "rocr_blit")
      return false;
    auto blit_it =
        std::find(rocr_blit_records_.begin(), rocr_blit_records_.end(),
                  it->second);
    if (blit_it == rocr_blit_records_.end())
      ReportHotSwapRegistryError("missing ROCR blit range record");
    rocr_blit_records_.erase(blit_it);
    records_.erase(it);
    return true;
  }

  std::shared_ptr<HotSwapKernelRecord> Lookup(uint64_t address) const {
    std::lock_guard<std::mutex> guard(mutex_);
    auto it = records_.find(address);
    if (it != records_.end()) return it->second;
    for (const auto& record : rocr_blit_records_) {
      if (record->IsRegisteredRocrBlit(address)) return record;
    }
    return nullptr;
  }

  void UpdateLaunchMetadata(uint64_t address, uint32_t group_segment_size,
                            uint32_t private_segment_size) {
    std::shared_ptr<HotSwapKernelRecord> record;
    {
      std::lock_guard<std::mutex> guard(mutex_);
      auto it = records_.find(address);
      if (it == records_.end()) return;
      record = it->second;
    }
    std::lock_guard<std::mutex> record_guard(record->Mutex);
    record->SourceGroupSegmentSize = group_segment_size;
    record->SourcePrivateSegmentSize = private_segment_size;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<uint64_t, std::shared_ptr<HotSwapKernelRecord>> records_;
  std::vector<std::shared_ptr<HotSwapKernelRecord>> rocr_blit_records_;
};

}  // namespace loader
}  // namespace hsa
}  // namespace amd
}  // namespace rocr

#endif  // HSA_RUNTIME_LOADER_HOTSWAP_KERNEL_REGISTRY_H_
