// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/rocprofiler-sdk.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "timemory.hpp"
#include <regex>

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0

#    include <timemory/defines.h>
#    include <timemory/utility/demangle.hpp>

#    include <rocprofiler-sdk/agent.h>
#    include <rocprofiler-sdk/cxx/name_info.hpp>
#    include <rocprofiler-sdk/fwd.h>

#    include <algorithm>
#    include <cstdint>
#    include <set>
#    include <sstream>
#    include <string>
#    include <unordered_set>
#    include <vector>

#    define ROCPROFILER_CALL(result)                                                     \
        {                                                                                \
            rocprofiler_status_t CHECKSTATUS = (result);                                 \
            if(CHECKSTATUS != ROCPROFILER_STATUS_SUCCESS)                                \
            {                                                                            \
                auto        msg        = std::stringstream{};                            \
                std::string status_msg = rocprofiler_get_status_string(CHECKSTATUS);     \
                msg << "[" #result "][" << __FILE__ << ":" << __LINE__ << "] "           \
                    << "rocprofiler-sdk call [" << #result                               \
                    << "] failed with error code " << CHECKSTATUS                        \
                    << " :: " << status_msg;                                             \
                ROCPROFSYS_WARNING(0, "%s\n", msg.str().c_str());                        \
            }                                                                            \
        }

namespace rocprofsys
{
namespace rocprofiler_sdk
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

template <typename Tp>
std::string
to_lower(const Tp& _val)
{
    auto _v = std::string{ _val };
    for(auto& itr : _v)
        itr = ::tolower(itr);
    return _v;
}

struct operation_options
{
    std::string operations_include            = {};
    std::string operations_exclude            = {};
    std::string operations_annotate_backtrace = {};
};

auto callback_operation_option_names =
    std::unordered_map<rocprofiler_callback_tracing_kind_t, operation_options>{};
auto buffered_operation_option_names =
    std::unordered_map<rocprofiler_buffer_tracing_kind_t, operation_options>{};

std::unordered_set<int32_t>
get_operations_impl(rocprofiler_callback_tracing_kind_t kindv,
                    const std::string&                  optname = {})
{
    static const auto callback_tracing_info =
        rocprofiler::sdk::get_callback_tracing_names();

    if(optname.empty())
    {
        auto _ret = std::unordered_set<int32_t>{};
        for(auto iitr : callback_tracing_info[kindv].items())
        {
            if(iitr.second && *iitr.second != "none") _ret.emplace(iitr.first);
        }
        return _ret;
    }

    auto _val = get_setting_value<std::string>(optname);

    ROCPROFSYS_CONDITIONAL_ABORT_F(!_val, "no setting %s\n", optname.c_str());

    if(_val->empty()) return std::unordered_set<int32_t>{};

    auto _ret = std::unordered_set<int32_t>{};
    for(const auto& itr : tim::delimit(*_val, " ,;:\n\t"))
    {
        for(auto iitr : callback_tracing_info[kindv].items())
        {
            auto _re = std::regex{ itr, std::regex_constants::icase };
            if(iitr.second && std::regex_search(iitr.second->data(), _re))
            {
                ROCPROFSYS_PRINT_F("%s ('%s') matched: %s\n", optname.c_str(),
                                   itr.c_str(), iitr.second->data());
                _ret.emplace(iitr.first);
            }
        }
    }

    return _ret;
}

std::unordered_set<int32_t>
get_operations_impl(rocprofiler_buffer_tracing_kind_t kindv,
                    const std::string&                optname = {})
{
    static const auto buffered_tracing_info =
        rocprofiler::sdk::get_buffer_tracing_names();

    if(optname.empty())
    {
        auto _ret = std::unordered_set<int32_t>{};
        for(auto iitr : buffered_tracing_info[kindv].items())
        {
            if(iitr.second && *iitr.second != "none") _ret.emplace(iitr.first);
        }
        return _ret;
    }

    auto _val = get_setting_value<std::string>(optname);

    ROCPROFSYS_CONDITIONAL_ABORT_F(!_val, "no setting %s\n", optname.c_str());

    if(_val->empty()) return std::unordered_set<int32_t>{};

    auto _ret = std::unordered_set<int32_t>{};
    for(const auto& itr : tim::delimit(*_val, " ,;:\n\t"))
    {
        for(auto iitr : buffered_tracing_info[kindv].items())
        {
            auto _re = std::regex{ itr, std::regex_constants::icase };
            if(iitr.second && std::regex_search(iitr.second->data(), _re))
            {
                ROCPROFSYS_PRINT_F("%s ('%s') matched: %s\n", optname.c_str(),
                                   itr.c_str(), iitr.second->data());
                _ret.emplace(iitr.first);
            }
        }
    }
    return _ret;
}

std::vector<int32_t>
get_operations_impl(const std::unordered_set<int32_t>& _complete,
                    const std::unordered_set<int32_t>& _include,
                    const std::unordered_set<int32_t>& _exclude)
{
    auto _convert = [](const auto& _dset) {
        auto _dret = std::vector<int32_t>{};
        _dret.reserve(_dset.size());
        for(auto itr : _dset)
            _dret.emplace_back(itr);
        std::sort(_dret.begin(), _dret.end());
        return _dret;
    };

    if(_include.empty() && _exclude.empty()) return _convert(_complete);

    auto _ret = (_include.empty()) ? _complete : _include;
    for(auto itr : _exclude)
        _ret.erase(itr);

    return _convert(_ret);
}

}  // namespace

/// @brief Return the version of the rocprofiler-sdk
/// @return The version of the rocprofiler-sdk or 0 if not initialized
version_info&
get_version()
{
    static auto _version = version_info{ 0 };

    if(_version.formatted == 0)
    {
        uint32_t _major = 0;
        uint32_t _minor = 0;
        uint32_t _patch = 0;

        ROCPROFILER_CALL(rocprofiler_get_version(&_major, &_minor, &_patch));

        _version.major     = _major;
        _version.minor     = _minor;
        _version.patch     = _patch;
        _version.formatted = _major * 10000 + _minor * 100 + _patch;
    }

    return _version;
}

void
config_settings(const std::shared_ptr<settings>& _config)
{
    // const auto agents                = std::vector<rocprofiler_agent_t>{};
    const auto buffered_tracing_info = rocprofiler::sdk::get_buffer_tracing_names();
    const auto callback_tracing_info = rocprofiler::sdk::get_callback_tracing_names();

    auto _skip_domains =
        std::unordered_set<std::string_view>{ "none",
                                              "correlation_id_retirement",
                                              "marker_core_api",
                                              "marker_control_api",
                                              "marker_name_api",
                                              "code_object" };

    auto _domain_choices = std::vector<std::string>{};
    auto _add_domain     = [&_domain_choices, &_skip_domains](std::string_view _domain) {
        auto _v = to_lower(_domain);

        if(_skip_domains.count(_v) == 0)
        {
            auto itr = std::find(_domain_choices.begin(), _domain_choices.end(), _v);
            if(itr == _domain_choices.end()) _domain_choices.emplace_back(_v);
        }
    };

    static auto _option_names           = std::unordered_set<std::string>{};
    auto        _add_operation_settings = [&_config, &_skip_domains](
                                       std::string_view _domain_name, const auto& _domain,
                                       auto& _operation_option_names) {
        auto _v = to_lower(_domain_name);

        if(_skip_domains.count(_v) > 0) return;

        auto _op_option_name = JOIN('_', "ROCPROFSYS_ROCM", _domain_name, "OPERATIONS");
        auto _eop_option_name =
            JOIN('_', "ROCPROFSYS_ROCM", _domain_name, "OPERATIONS_EXCLUDE");
        auto _bt_option_name =
            JOIN('_', "ROCPROFSYS_ROCM", _domain_name, "OPERATIONS_ANNOTATE_BACKTRACE");

        auto _op_choices = std::vector<std::string>{};
        for(auto itr : _domain.operations)
            _op_choices.emplace_back(std::string{ itr });

        if(_op_choices.empty()) return;

        _operation_option_names.emplace(
            _domain.value,
            operation_options{ _op_option_name, _eop_option_name, _bt_option_name });

        if(_option_names.emplace(_op_option_name).second)
        {
            ROCPROFSYS_CONFIG_SETTING(
                std::string, _op_option_name.c_str(),
                "Inclusive filter for domain operations (for API domains, this selects "
                "the functions to trace) [regex supported]",
                std::string{}, "rocm", "rocprofiler-sdk", "advanced")
                ->set_choices(_op_choices);
        }

        if(_option_names.emplace(_eop_option_name).second)
        {
            ROCPROFSYS_CONFIG_SETTING(
                std::string, _eop_option_name.c_str(),
                "Exclusive filter for domain operations applied after the inclusive "
                "filter (for API domains, removes function from trace) [regex supported]",
                std::string{}, "rocm", "rocprofiler-sdk", "advanced")
                ->set_choices(_op_choices);
        }

        if(_option_names.emplace(_bt_option_name).second)
        {
            ROCPROFSYS_CONFIG_SETTING(
                std::string, _bt_option_name.c_str(),
                "Specification of domain operations which will record a backtrace (for "
                "API domains, this is a list of function names) [regex supported]",
                std::string{}, "rocm", "rocprofiler-sdk", "advanced")
                ->set_choices(_op_choices);
        }
    };

    _domain_choices.reserve(buffered_tracing_info.size());
    _domain_choices.reserve(callback_tracing_info.size());
    _add_domain("hip_api");
    _add_domain("hsa_api");
    _add_domain("marker_api");

    for(const auto& itr : buffered_tracing_info)
        _add_domain(itr.name);

    for(const auto& itr : callback_tracing_info)
        _add_domain(itr.name);

    std::sort(_domain_choices.begin(), _domain_choices.end());

    namespace join = ::timemory::join;
    auto _domain_description =
        JOIN("", "Specification of ROCm domains to trace/profile. Choices: ",
             join::join(join::array_config{ ", ", "", "" }, _domain_choices));
    auto _domain_defaults = std::string{ "hip_runtime_api,marker_api,kernel_dispatch,"
                                         "memory_copy,scratch_memory" };

#    if(ROCPROFILER_VERSION < 10000)
    _domain_defaults.append(",page_migration");
#    endif

    ROCPROFSYS_CONFIG_SETTING(std::string, "ROCPROFSYS_ROCM_DOMAINS", _domain_description,
                              _domain_defaults, "rocm", "rocprofiler-sdk")
        ->set_choices(_domain_choices);

    ROCPROFSYS_CONFIG_SETTING(
        std::string, "ROCPROFSYS_ROCM_EVENTS",
        "ROCm hardware counters. Use ':device=N' syntax to specify collection on device "
        "number N, e.g. ':device=0'. If no device specification is provided, the event "
        "is collected on every available device",
        "", "rocm", "hardware_counters");

    _skip_domains.emplace("kernel_dispatch");
    _skip_domains.emplace("page_migration");
    _skip_domains.emplace("scratch_memory");

    _add_operation_settings(
        "MARKER_API", callback_tracing_info[ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API],
        callback_operation_option_names);

    for(const auto& itr : callback_tracing_info)
        _add_operation_settings(itr.name, itr, callback_operation_option_names);

    for(const auto& itr : buffered_tracing_info)
        _add_operation_settings(itr.name, itr, buffered_operation_option_names);

    // Add the ROCPROFSYS_ROCM_GROUP_BY_QUEUE setting if the hip_stream domain is present
    // in supported ROCProfiler-SDK domains.
    auto _has_hip_stream = std::find(_domain_choices.begin(), _domain_choices.end(),
                                     "hip_stream") != _domain_choices.end();

    if(_has_hip_stream)
    {
        ROCPROFSYS_CONFIG_SETTING(
            bool, "ROCPROFSYS_ROCM_GROUP_BY_QUEUE",
            "By default, Perfetto trace will show the HIP streams to which kernel "
            "and memory copy operations submitted. With the "
            "`ROCPROFSYS_ROCM_GROUP_BY_QUEUE` option, the trace will display HSA queues "
            "to which these kernel and memory operations were submitted.",
            false, "rocm", "perfetto");
    }
}

std::unordered_set<rocprofiler_callback_tracing_kind_t>
get_callback_domains()
{
    const auto callback_tracing_info = rocprofiler::sdk::get_callback_tracing_names();
    auto       supported = std::unordered_set<rocprofiler_callback_tracing_kind_t>{
        ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
        ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
        ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
        ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API,
        ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
        ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API,
        ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API,
        ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
    };

    auto _version = get_version();
    ROCPROFSYS_WARNING_IF(_version.formatted == 0,
                          "Warning! rocprofiler-sdk version not initialized\n");

#    if(ROCPROFILER_VERSION >= 600)
    if(_version.formatted >= 600)
    {
        // Argument tracing is supported in rocprofiler-sdk 0.6.0 and later
        supported.emplace(ROCPROFILER_CALLBACK_TRACING_RCCL_API);
        supported.emplace(ROCPROFILER_CALLBACK_TRACING_ROCDECODE_API);
    }
#    endif
#    if(ROCPROFILER_VERSION >= 700)
    if(_version.formatted >= 700)
    {
        supported.emplace(ROCPROFILER_CALLBACK_TRACING_ROCJPEG_API);
    }
#    endif

    auto _data = std::unordered_set<rocprofiler_callback_tracing_kind_t>{};
    auto _domains =
        tim::delimit(config::get_setting_value<std::string>("ROCPROFSYS_ROCM_DOMAINS")
                         .value_or(std::string{}),
                     " ,;:\t\n");

    if(config::get_use_rcclp() && _version.formatted >= 600)
    {
        // Translate ROCPROFSYS_USE_RCCLP to entry in ROCPROFSYS_ROCM_DOMAINS
        _data.emplace(ROCPROFILER_CALLBACK_TRACING_RCCL_API);
    }

    const auto valid_choices =
        settings::instance()->at("ROCPROFSYS_ROCM_DOMAINS")->get_choices();

    auto invalid_domain = [&valid_choices](const auto& domainv) {
        return !std::any_of(valid_choices.begin(), valid_choices.end(),
                            [&domainv](const auto& aitr) { return (aitr == domainv); });
    };

    for(const auto& itr : _domains)
    {
        if(invalid_domain(itr))
        {
            ROCPROFSYS_THROW("unsupported ROCPROFSYS_ROCM_DOMAINS value: %s\n",
                             itr.c_str());
        }

        if(itr == "hsa_api")
        {
            for(auto eitr : { ROCPROFILER_CALLBACK_TRACING_HSA_CORE_API,
                              ROCPROFILER_CALLBACK_TRACING_HSA_AMD_EXT_API,
                              ROCPROFILER_CALLBACK_TRACING_HSA_IMAGE_EXT_API,
                              ROCPROFILER_CALLBACK_TRACING_HSA_FINALIZE_EXT_API })
                _data.emplace(eitr);
        }
        else if(itr == "hip_api")
        {
            for(auto eitr : { ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API,
                              ROCPROFILER_CALLBACK_TRACING_HIP_COMPILER_API })
                _data.emplace(eitr);
        }
        else if(itr == "marker_api" || itr == "roctx")
        {
            _data.emplace(ROCPROFILER_CALLBACK_TRACING_MARKER_CORE_API);
        }
        else
        {
            for(size_t idx = 0; idx < callback_tracing_info.size(); ++idx)
            {
                auto ditr = callback_tracing_info[idx];
                auto dval = static_cast<rocprofiler_callback_tracing_kind_t>(idx);
                if(itr == to_lower(ditr.name) && supported.count(dval) > 0)
                {
                    _data.emplace(dval);
                    break;
                }
            }
        }
    }

    return _data;
}

std::unordered_set<rocprofiler_buffer_tracing_kind_t>
get_buffered_domains()
{
    const auto buffer_tracing_info = rocprofiler::sdk::get_buffer_tracing_names();
    const auto supported = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{
        ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH,
        ROCPROFILER_BUFFER_TRACING_MEMORY_COPY,
#    if(ROCPROFILER_VERSION >= 600)
        ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION,
#    endif
#    if(ROCPROFILER_VERSION < 10000)
        ROCPROFILER_BUFFER_TRACING_PAGE_MIGRATION,
#    endif
        ROCPROFILER_BUFFER_TRACING_SCRATCH_MEMORY,
    };

    auto _data = std::unordered_set<rocprofiler_buffer_tracing_kind_t>{};
    auto _domains =
        tim::delimit(config::get_setting_value<std::string>("ROCPROFSYS_ROCM_DOMAINS")
                         .value_or(std::string{}),
                     " ,;:\t\n");
    const auto valid_choices =
        settings::instance()->at("ROCPROFSYS_ROCM_DOMAINS")->get_choices();

    auto invalid_domain = [&valid_choices](const auto& domainv) {
        return !std::any_of(valid_choices.begin(), valid_choices.end(),
                            [&domainv](const auto& aitr) { return (aitr == domainv); });
    };

    for(const auto& itr : _domains)
    {
        if(invalid_domain(itr))
        {
            ROCPROFSYS_THROW("unsupported ROCPROFSYS_ROCM_DOMAINS value: %s\n",
                             itr.c_str());
        }

        if(itr == "hsa_api")
        {
            for(auto eitr : { ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
                              ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API,
                              ROCPROFILER_BUFFER_TRACING_HSA_IMAGE_EXT_API,
                              ROCPROFILER_BUFFER_TRACING_HSA_FINALIZE_EXT_API })
                _data.emplace(eitr);
        }
        else if(itr == "hip_api")
        {
            for(auto eitr : { ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API,
                              ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API })
                _data.emplace(eitr);
        }
        else if(itr == "marker_api" || itr == "roctx")
        {
            _data.emplace(ROCPROFILER_BUFFER_TRACING_MARKER_CORE_API);
        }
#    if(ROCPROFILER_VERSION >= 600)
        else if(itr == "memory_allocation")
        {
            _data.emplace(ROCPROFILER_BUFFER_TRACING_MEMORY_ALLOCATION);
        }
#    endif
        else if(itr == "memory_copy")
        {
            _data.emplace(ROCPROFILER_BUFFER_TRACING_MEMORY_COPY);
        }
        else
        {
            for(size_t idx = 0; idx < buffer_tracing_info.size(); ++idx)
            {
                auto ditr = buffer_tracing_info[idx];
                auto dval = static_cast<rocprofiler_buffer_tracing_kind_t>(idx);
                if(itr == to_lower(ditr.name) && supported.count(dval) > 0)
                {
                    _data.emplace(dval);
                    break;
                }
            }
        }
    }

    return _data;
}

std::vector<std::string>
get_rocm_events()
{
    return tim::delimit(
        get_setting_value<std::string>("ROCPROFSYS_ROCM_EVENTS").value_or(std::string{}),
        " ,;\t\n");
}

bool
get_group_by_queue(void)
{
    std::optional<bool> _group_by_queue =
        config::get_setting_value<bool>("ROCPROFSYS_ROCM_GROUP_BY_QUEUE");
    bool _ret = _group_by_queue.value_or(true);

    return _ret;
}

std::vector<int32_t>
get_operations(rocprofiler_callback_tracing_kind_t kindv)
{
    ROCPROFSYS_CONDITIONAL_ABORT_F(
        callback_operation_option_names.count(kindv) == 0,
        "callback_operation_operation_names does not have value for %i\n", kindv);

    auto _complete = get_operations_impl(kindv);
    auto _include  = get_operations_impl(
        kindv, callback_operation_option_names.at(kindv).operations_include);
    auto _exclude = get_operations_impl(
        kindv, callback_operation_option_names.at(kindv).operations_exclude);

    return get_operations_impl(_complete, _include, _exclude);
}

std::vector<int32_t>
get_operations(rocprofiler_buffer_tracing_kind_t kindv)
{
    ROCPROFSYS_CONDITIONAL_ABORT_F(
        buffered_operation_option_names.count(kindv) == 0,
        "buffered_operation_option_names does not have value for %i\n", kindv);

    auto _complete = get_operations_impl(kindv);
    auto _include  = get_operations_impl(
        kindv, buffered_operation_option_names.at(kindv).operations_include);
    auto _exclude = get_operations_impl(
        kindv, buffered_operation_option_names.at(kindv).operations_exclude);

    return get_operations_impl(_complete, _include, _exclude);
}

std::unordered_set<int32_t>
get_backtrace_operations(rocprofiler_callback_tracing_kind_t kindv)
{
    ROCPROFSYS_CONDITIONAL_ABORT_F(
        callback_operation_option_names.count(kindv) == 0,
        "callback_operation_operation_names does not have value for %i\n", kindv);

    auto _data = get_operations_impl(
        kindv, callback_operation_option_names.at(kindv).operations_annotate_backtrace);
    auto _ret = std::unordered_set<int32_t>{};
    _ret.reserve(_data.size());
    for(auto itr : _data)
        _ret.emplace(itr);
    return _ret;
}

std::unordered_set<int32_t>
get_backtrace_operations(rocprofiler_buffer_tracing_kind_t kindv)
{
    ROCPROFSYS_CONDITIONAL_ABORT_F(
        buffered_operation_option_names.count(kindv) == 0,
        "buffered_operation_option_names does not have value for %i\n", kindv);

    auto _data = get_operations_impl(
        kindv, buffered_operation_option_names.at(kindv).operations_annotate_backtrace);
    auto _ret = std::unordered_set<int32_t>{};
    _ret.reserve(_data.size());
    for(auto itr : _data)
        _ret.emplace(itr);
    return _ret;
}
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

#else

namespace rocprofsys
{
namespace rocprofiler_sdk
{
void
config_settings(const std::shared_ptr<settings>&)
{}

version_info&
get_version()
{
    static auto _version = version_info{ 0 };
    return _version;
}
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

#endif
