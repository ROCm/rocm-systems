// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "smi_nic_log.h"

#include <atomic>

namespace amd::smi::nic::log {

namespace {
constexpr auto kUnmaskedTailChars = size_t{4};
constexpr auto kMaskPrefix = "****";

std::atomic<Sink_t> g_sink{nullptr};
}  // namespace

auto set_sink(Sink_t sink) -> void { g_sink.store(sink); }

auto is_enabled() -> bool { return (g_sink.load() != nullptr); }

auto debug(const char* func, const std::string& msg) -> void {
  const auto sink = g_sink.load();
  if (sink != nullptr) {
    sink(std::string(func) + " | " + msg);
  }
}

auto mask_tail(const std::string& value) -> std::string {
  if (value.empty()) {
    return value;
  }
  if (value.size() <= kUnmaskedTailChars) {
    return kMaskPrefix;
  }
  return kMaskPrefix + value.substr(value.size() - kUnmaskedTailChars);
}

}  // namespace amd::smi::nic::log
