// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocm_visibility.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>
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
    if (token.size() >= 5 && token.size() <= 20 && token.starts_with("GPU-") && token != "GPU-XX") {
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

    if (token.find("GPU-") != std::string::npos) {
      for (size_t candidate = 0; candidate < gpus.size(); ++candidate) {
        if (gpus[candidate].unique_id != 0 &&
            gpu_uuid(gpus[candidate].unique_id).find(token) != std::string::npos) {
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
  if (!client)
    return std::nullopt;

  const std::vector<VisibleGpu> rocr_gpus = filter_rocr_visible_gpus(topology, rocr_visible);
  std::vector<VisibleGpu> selected = filter_client_visible_gpus(rocr_gpus, client->second);
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
  return VisibilityOverride{std::string(client->first), join_comma(ordinals)};
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
