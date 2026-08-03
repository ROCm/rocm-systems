// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_visibility.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstddef>
#include <format>

namespace rocjitsu::cli {
namespace {

std::string_view trim(std::string_view text) {
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())))
    text.remove_prefix(1);
  while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())))
    text.remove_suffix(1);
  return text;
}

std::string gpu_uuid(uint64_t unique_id) { return std::format("GPU-{:016X}", unique_id); }

constexpr std::string_view kGpuUuidPrefix = "GPU-";
constexpr std::string_view kNoUuidSentinel = "GPU-XX";
constexpr size_t kMinGpuUuidSelectorLength = kGpuUuidPrefix.size() + 1;
constexpr size_t kMaxGpuUuidSelectorLength = kGpuUuidPrefix.size() + 2 * sizeof(uint64_t);

/// @brief Whether @p token is spelled as a GPU UUID prefix rather than as a device ordinal.
/// @details The accepted window is ROCR's own: "GPU-" followed by at least one and at most 16 hex
/// digits of the 64-bit KFD unique ID, which is the "GPU-{:016X}" spelling gpu_uuid() produces and
/// the 5..20 token-length check in
/// projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_filter_device.cpp:119. "GPU-XX" is
/// the sentinel ROCR reports for agents without UUID support, so it never names a device. Both
/// selector paths share this predicate on purpose; only the matching below each call site differs.
bool is_gpu_uuid_selector(std::string_view token) {
  return token.size() >= kMinGpuUuidSelectorLength && token.size() <= kMaxGpuUuidSelectorLength &&
         token.starts_with(kGpuUuidPrefix) && token != kNoUuidSentinel;
}

std::string join_comma(const std::vector<std::string> &tokens) {
  std::string result;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (i != 0)
      result += ',';
    result += tokens[i];
  }
  return result;
}

std::optional<std::pair<std::string_view, std::string_view>>
client_selector(std::optional<std::string_view> hip_visible,
                std::optional<std::string_view> cuda_visible) {
  if (hip_visible && !hip_visible->empty())
    return std::pair<std::string_view, std::string_view>{"HIP_VISIBLE_DEVICES", *hip_visible};
  if (cuda_visible && !cuda_visible->empty())
    return std::pair<std::string_view, std::string_view>{"CUDA_VISIBLE_DEVICES", *cuda_visible};
  return std::nullopt;
}

} // namespace

std::vector<VisibleGpu> enumerate_kfd_gpus(const std::vector<VisibleGpu> &candidates) {
  std::vector<VisibleGpu> gpus;
  gpus.reserve(candidates.size());
  for (const VisibleGpu &candidate : candidates) {
    if (candidate.gpu_id == 0)
      continue;
    VisibleGpu gpu = candidate;
    gpu.ordinal = static_cast<uint32_t>(gpus.size());
    gpus.push_back(gpu);
  }
  return gpus;
}

std::vector<VisibleGpu> filter_rocr_visible_gpus(const std::vector<VisibleGpu> &gpus,
                                                 std::optional<std::string_view> selector) {
  if (!selector)
    return gpus;

  std::vector<VisibleGpu> filtered;
  std::string_view rest = *selector;
  while (!rest.empty() && filtered.size() < gpus.size()) {
    const size_t comma = rest.find(',');
    std::string token(trim(comma == std::string_view::npos ? rest : rest.substr(0, comma)));
    std::transform(token.begin(), token.end(), token.begin(),
                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

    std::optional<size_t> index;
    if (is_gpu_uuid_selector(token)) {
      for (size_t candidate = 0; candidate < gpus.size(); ++candidate) {
        if (gpus[candidate].unique_id == 0 ||
            !gpu_uuid(gpus[candidate].unique_id).starts_with(token))
          continue;
        if (index)
          return filtered;
        index = candidate;
      }
    } else {
      char *end = nullptr;
      const long parsed = std::strtol(token.c_str(), &end, 0);
      if (end != token.c_str() && *end == '\0' && parsed >= 0)
        index = static_cast<size_t>(parsed);
    }

    if (!index || *index >= gpus.size() ||
        std::any_of(filtered.begin(), filtered.end(),
                    [&](const VisibleGpu &gpu) { return gpu.ordinal == gpus[*index].ordinal; }))
      return filtered;
    filtered.push_back(gpus[*index]);

    if (comma == std::string_view::npos)
      break;
    rest.remove_prefix(comma + 1);
  }
  return filtered;
}

std::vector<VisibleGpu> filter_client_visible_gpus(const std::vector<VisibleGpu> &gpus,
                                                   std::optional<std::string_view> selector) {
  if (!selector)
    return gpus;

  std::vector<VisibleGpu> filtered;
  std::string_view rest = *selector;
  while (!rest.empty() && filtered.size() < gpus.size()) {
    const size_t comma = rest.find(',');
    std::string token(comma == std::string_view::npos ? rest : rest.substr(0, comma));

    // The spelling guard is the only thing this path borrows from filter_rocr_visible_gpus, and
    // its window makes rocjitsu STRICTER than the runtime this path models: CLR gates on a bare
    // find("GPU-") with no length or sentinel check
    // (projects/clr/rocclr/device/rocm/rocdevice.cpp:431-444), so CLR resolves the token "GPU-" to
    // agent 0. Diverging is safe only because CLR never sees a token rocjitsu rejected:
    // filter_client_visible_gpus runs only in DBT guest mode, where main.cpp unconditionally
    // rewrites the client selector to normalized_client_visible_devices()'s canonical numeric
    // list before execvp. Do not restore CLR parity here without also removing that rewrite.
    //
    // Everything below the guard stays CLR-shaped, preserving the deliberate ROCR-vs-CLR split
    // that DuplicateAndReorderedSelectorsMatchRuntimeBehavior documents: the break makes the
    // first match win rather than terminating on ambiguity, the compare is case-sensitive
    // because this path never uppercases, and a token that fails the guard or matches no agent
    // needs no early return -- it falls through to the numeric parse below, which rejects it and
    // terminates selection.
    if (is_gpu_uuid_selector(token)) {
      for (size_t candidate = 0; candidate < gpus.size(); ++candidate) {
        if (gpus[candidate].unique_id != 0 &&
            gpu_uuid(gpus[candidate].unique_id).starts_with(token)) {
          token = std::to_string(candidate);
          break;
        }
      }
    }

    uint32_t index = 0;
    const char *begin = token.data();
    const char *end = begin + token.size();
    auto [ptr, error] = std::from_chars(begin, end, index);
    if (error != std::errc{} || ptr != end || token != std::to_string(index) ||
        index >= gpus.size())
      return filtered;
    if (std::none_of(filtered.begin(), filtered.end(),
                     [&](const VisibleGpu &gpu) { return gpu.gpu_id == gpus[index].gpu_id; }))
      filtered.push_back(gpus[index]);

    if (comma == std::string_view::npos)
      break;
    rest.remove_prefix(comma + 1);
  }
  return filtered;
}

std::vector<VisibleGpu> effective_visible_gpus(const std::vector<VisibleGpu> &topology,
                                               std::optional<std::string_view> rocr_visible,
                                               std::optional<std::string_view> hip_visible,
                                               std::optional<std::string_view> cuda_visible) {
  std::vector<VisibleGpu> gpus = filter_rocr_visible_gpus(topology, rocr_visible);
  const auto client = client_selector(hip_visible, cuda_visible);
  return client ? filter_client_visible_gpus(gpus, client->second) : gpus;
}

std::optional<VisibilityOverride> normalized_client_visible_devices(
    const std::vector<VisibleGpu> &topology, std::optional<std::string_view> rocr_visible,
    std::optional<std::string_view> hip_visible, std::optional<std::string_view> cuda_visible,
    std::optional<uint32_t> first_gpu_id) {
  const auto client = client_selector(hip_visible, cuda_visible);
  if (!client && !first_gpu_id)
    return std::nullopt;

  const std::vector<VisibleGpu> rocr_gpus = filter_rocr_visible_gpus(topology, rocr_visible);
  std::vector<VisibleGpu> selected =
      client ? filter_client_visible_gpus(rocr_gpus, client->second) : rocr_gpus;
  if (first_gpu_id) {
    const auto first = std::find_if(selected.begin(), selected.end(), [&](const VisibleGpu &gpu) {
      return gpu.gpu_id == *first_gpu_id;
    });
    if (first != selected.end())
      std::rotate(selected.begin(), first, first + 1);
  }
  std::vector<std::string> ordinals;
  for (const VisibleGpu &gpu : selected) {
    auto match = std::find_if(rocr_gpus.begin(), rocr_gpus.end(), [&](const VisibleGpu &candidate) {
      return candidate.gpu_id == gpu.gpu_id;
    });
    if (match != rocr_gpus.end())
      ordinals.push_back(std::to_string(std::distance(rocr_gpus.begin(), match)));
  }
  const std::string name = client ? std::string(client->first) : "HIP_VISIBLE_DEVICES";
  return VisibilityOverride{name, join_comma(ordinals)};
}

std::optional<std::string>
expanded_rocr_visible_devices(const std::vector<VisibleGpu> &topology,
                              std::optional<std::string_view> rocr_visible) {
  if (!rocr_visible || topology.empty())
    return std::nullopt;

  const std::vector<VisibleGpu> selected = filter_rocr_visible_gpus(topology, rocr_visible);
  if (selected.empty())
    return std::nullopt;

  std::vector<std::string> ordinals;
  for (const VisibleGpu &gpu : selected)
    ordinals.push_back(std::to_string(gpu.ordinal));
  ordinals.push_back(std::to_string(topology.size()));

  std::string rewritten = join_comma(ordinals);
  return rewritten == *rocr_visible ? std::nullopt
                                    : std::optional<std::string>(std::move(rewritten));
}

HostSelection select_host_gpu(const std::vector<VisibleGpu> &visible_gpus,
                              uint32_t configured_gpu_id, uint32_t gfx_target_version) {
  if (configured_gpu_id != 0) {
    const auto match =
        std::find_if(visible_gpus.begin(), visible_gpus.end(),
                     [&](const VisibleGpu &gpu) { return gpu.gpu_id == configured_gpu_id; });
    if (match == visible_gpus.end())
      return {HostSelectionStatus::ExplicitGpuHidden, configured_gpu_id};
    return {match->gfx_target_version == gfx_target_version
                ? HostSelectionStatus::Selected
                : HostSelectionStatus::ExplicitGpuIsaMismatch,
            configured_gpu_id};
  }

  const auto match =
      std::find_if(visible_gpus.begin(), visible_gpus.end(), [&](const VisibleGpu &gpu) {
        return gpu.gfx_target_version == gfx_target_version;
      });
  return match == visible_gpus.end() ? HostSelection{HostSelectionStatus::NoIsaMatch, 0}
                                     : HostSelection{HostSelectionStatus::Selected, match->gpu_id};
}

} // namespace rocjitsu::cli
