// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "core/amd_smi.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/gpu.hpp"
#include "timemory.hpp"

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
namespace rocprofsys
{
namespace amd_smi
{
namespace
{
std::string
get_setting_name(std::string _v)
{
    constexpr auto _prefix = tim::string_view_t{ "rocprofsys_" };
    for(auto& itr : _v)
        itr = tolower(itr);
    auto _pos = _v.find(_prefix);
    if(_pos == 0) return _v.substr(_prefix.length());
    return _v;
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
                ROCPROFSYS_PRINT("Warning! Duplicate setting: %s / %s\n",                \
                                 get_setting_name(ENV_NAME).c_str(), ENV_NAME);          \
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
    std::string jpeg_activity_support = "";
    std::string vcn_activity_support  = "";

    size_t device_count = gpu::get_processor_count();
    for(size_t i = 0; i < device_count; i++)
    {
        if(gpu::is_vcn_activity_supported(i) || gpu::is_vcn_busy_supported(i))
        {
            vcn_activity_support += ", vcn_activity";
            break;
        }
    }
    for(size_t i = 0; i < device_count; i++)
    {
        if(gpu::is_jpeg_activity_supported(i) || gpu::is_jpeg_busy_supported(i))
        {
            jpeg_activity_support += ", jpeg_activity";
            break;
        }
    }

    ROCPROFSYS_CONFIG_SETTING(
        std::string, "ROCPROFSYS_AMD_SMI_METRICS",
        "amd-smi metrics to collect: " + default_metrics + jpeg_activity_support +
            vcn_activity_support + ". " +
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
