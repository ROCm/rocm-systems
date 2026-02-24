// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/amd_smi.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/gpu.hpp"
#include "timemory.hpp"

#include "logger/debug.hpp"

#include <cctype>
#include <string_view>

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
namespace rocprofsys
{
namespace amd_smi
{
namespace
{
std::string
get_setting_name(std::string_view input)
{
    constexpr auto prefix = std::string_view{ "rocprofsys_" };

    std::string result;
    result.reserve(input.size());
    for(auto c : input)
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if(result.compare(0, prefix.size(), prefix) == 0) return result.substr(prefix.size());

    return result;
}

#    define ROCPROFSYS_CONFIG_SETTING(TYPE, ENV_NAME, DESCRIPTION, INITIAL_VALUE, ...)   \
        [&]() {                                                                          \
            auto _ret = _config->insert<TYPE, TYPE>(                                     \
                ENV_NAME, get_setting_name(ENV_NAME), DESCRIPTION,                       \
                TYPE{ INITIAL_VALUE },                                                   \
                std::set<std::string>{ "custom", "rocprofsys", "librocprof-sys",         \
                                       __VA_ARGS__ });                                   \
            if(!_ret.second)                                                             \
            {                                                                            \
                LOG_WARNING("Duplicate setting: {} / {}", get_setting_name(ENV_NAME),    \
                            ENV_NAME);                                                   \
            }                                                                            \
            return _config->find(ENV_NAME)->second;                                      \
        }()
}  // namespace

void
config_settings(const std::shared_ptr<settings>& _config)
{
    if(!get_use_amd_smi() || !gpu::initialize_amdsmi()) return;

    std::string default_metrics = "busy, temp, power, mem_usage";
    // No distinction between busy and activity shown in description
    std::string jpeg_activity_support{};
    std::string vcn_activity_support{};
    std::string xgmi_support{};
    std::string pcie_support{};

    size_t device_count = gpu::get_processor_count();
    for(size_t i = 0; i < device_count; i++)
    {
        if(gpu::vcn_is_device_level_only(i) || gpu::is_vcn_busy_supported(i))
        {
            vcn_activity_support += ", vcn_activity";
            break;
        }
    }
    for(size_t i = 0; i < device_count; i++)
    {
        if(gpu::jpeg_is_device_level_only(i) || gpu::is_jpeg_busy_supported(i))
        {
            jpeg_activity_support += ", jpeg_activity";
            break;
        }
    }
    for(size_t i = 0; i < device_count; i++)
    {
        if(gpu::is_xgmi_supported(i))
        {
            xgmi_support += ", xgmi";
            break;
        }
    }
    for(size_t i = 0; i < device_count; i++)
    {
        if(gpu::is_pcie_supported(i))
        {
            pcie_support += ", pcie";
            break;
        }
    }

    ROCPROFSYS_CONFIG_SETTING(
        std::string, "ROCPROFSYS_AMD_SMI_METRICS",
        "amd-smi metrics to collect: " + default_metrics + jpeg_activity_support +
            vcn_activity_support + xgmi_support + pcie_support + ". " +
            "An empty value implies 'all' and 'none' suppresses all.",
        "busy, temp, power, mem_usage", "backend", "amd_smi", "rocm", "process_sampling");
}
}  // namespace amd_smi
}  // namespace rocprofsys

#else
namespace rocprofsys
{
namespace amd_smi
{
void
config_settings(const std::shared_ptr<settings>&)
{}
}  // namespace amd_smi
}  // namespace rocprofsys
#endif
