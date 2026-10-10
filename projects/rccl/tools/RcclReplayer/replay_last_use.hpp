/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

#pragma once

#include <unordered_map>
#include <unordered_set>

// hipFree at the last use unless ncclMemFree owns that use: the address is still ncclMemAlloc'd, or unused since its ncclMemFree.
inline bool ReplayerFrees(bool ncclMemAllocated, int ncclMemFreeLine, int lastUseLine) {
  return !ncclMemAllocated && lastUseLine > ncclMemFreeLine;
}

// Last log line using each buffer and stream; a use inside ncclGroupStart/End counts at the outermost GroupEnd, where it launches.
class LastUseTracker {
 public:
  void UseBuffer(void* base, int line) { Use(base, line, &buffers_, &groupBuffers_); }
  void UseStream(void* stream, int line) { Use(stream, line, &streams_, &groupStreams_); }

  void GroupStart() { ++depth_; }

  void GroupEnd(int line) {
    if (depth_ == 0 || --depth_ > 0) {
      return;
    }
    for (void* base : groupBuffers_) {
      buffers_[base] = line;
    }
    for (void* stream : groupStreams_) {
      streams_[stream] = line;
    }
    groupBuffers_.clear();
    groupStreams_.clear();
  }

  // Nonzero at end of log means the last group was never closed, so its calls never launch.
  int depth() const { return depth_; }
  const std::unordered_map<void*, int>& buffers() const { return buffers_; }
  const std::unordered_map<void*, int>& streams() const { return streams_; }

 private:
  void Use(void* key, int line, std::unordered_map<void*, int>* lastUse, std::unordered_set<void*>* grouped) {
    (*lastUse)[key] = line;
    if (depth_ > 0) {
      grouped->insert(key);
    }
  }

  int depth_ = 0;
  std::unordered_map<void*, int> buffers_;
  std::unordered_map<void*, int> streams_;
  std::unordered_set<void*> groupBuffers_;
  std::unordered_set<void*> groupStreams_;
};
