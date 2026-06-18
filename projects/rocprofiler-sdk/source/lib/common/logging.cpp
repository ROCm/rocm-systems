// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/logging.hpp"
#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"

#include <absl/debugging/failure_signal_handler.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <fstream>
#include <mutex>
#include <string>
#include <unordered_map>

namespace rocprofiler
{
namespace common
{
namespace
{
namespace fs = ::rocprofiler::common::filesystem;

void
install_failure_signal_handler()
{
    static auto _once = std::once_flag{};
    std::call_once(
        _once, []() { absl::InstallFailureSignalHandler(absl::FailureSignalHandlerOptions{}); });
}

struct log_level_info
{
    int32_t severity_level = 0;
    int32_t verbose_level  = 0;
};

// Abseil log severity values: kInfo=0, kWarning=1, kError=2, kFatal=3
constexpr int32_t log_severity_info    = 0;
constexpr int32_t log_severity_warning = 1;
constexpr int32_t log_severity_error   = 2;
constexpr int32_t log_severity_fatal   = 3;

}  // namespace

void
init_logging(std::string_view env_prefix, logging_config cfg)
{
    static auto _once = std::once_flag{};
    std::call_once(_once, [env_prefix, &cfg]() {
        auto to_lower = [](std::string val) {
            for(auto& itr : val)
                itr = tolower(itr);
            return val;
        };

        const auto env_opts = std::unordered_map<std::string_view, log_level_info>{
            {"trace", {log_severity_info, ROCP_LOG_LEVEL_TRACE}},
            {"info", {log_severity_info, ROCP_LOG_LEVEL_INFO}},
            {"warning", {log_severity_warning, ROCP_LOG_LEVEL_WARNING}},
            {"error", {log_severity_error, ROCP_LOG_LEVEL_ERROR}},
            {"fatal", {log_severity_fatal, ROCP_LOG_LEVEL_NONE}}};

        auto supported = std::vector<std::string>{};
        supported.reserve(env_opts.size());
        for(auto itr : env_opts)
            supported.emplace_back(itr.first);

        if(cfg.name.empty()) cfg.name = to_lower(std::string{env_prefix});

        cfg.logdir       = get_env(fmt::format("{}_LOG_DIR", env_prefix), cfg.logdir);
        cfg.vlog_modules = get_env(fmt::format("{}_vmodule", env_prefix), cfg.vlog_modules);

        auto loglvl = to_lower(common::get_env(fmt::format("{}_LOG_LEVEL", env_prefix), ""));
        // default to warning
        auto& loglvl_v   = cfg.loglevel;
        auto& vlog_level = cfg.vlog_level;
        if(!loglvl.empty() && loglvl.find_first_not_of("-0123456789") == std::string::npos)
        {
            auto val = std::stol(loglvl);
            if(val < 0)
            {
                loglvl_v   = log_severity_fatal;
                vlog_level = val;
            }
            else
            {
                // default to trace in case val > ROCP_LOG_LEVEL_TRACE
                auto itr = env_opts.at("trace");
                for(auto oitr : env_opts)
                {
                    if(oitr.second.verbose_level == val)
                    {
                        itr = oitr.second;
                        break;
                    }
                }
                loglvl_v   = itr.severity_level;
                vlog_level = itr.verbose_level;
            }
        }
        else if(!loglvl.empty())
        {
            if(env_opts.find(loglvl) == env_opts.end())
            {
                // Write directly to stderr: absl logging is not yet initialized at this point.
                fmt::print(stderr,
                           "[rocprofiler-sdk][warning] invalid specifier for {}_LOG_LEVEL: "
                           "{}. Supported: {}. Falling back to default log level.\n",
                           env_prefix,
                           loglvl,
                           fmt::join(supported.begin(), supported.end(), ", "));
            }
            else
            {
                loglvl_v   = env_opts.at(loglvl).severity_level;
                vlog_level = env_opts.at(loglvl).verbose_level;
            }
        }

        update_logging(cfg);

        absl::InitializeLog();

        update_logging(cfg);

        ROCP_INFO << "logging initialized via " << fmt::format("{}_LOG_LEVEL", env_prefix)
                  << ". Log Level: " << loglvl << ". Verbose Log Level: " << vlog_level;
    });
}

void
update_logging(const logging_config& cfg)
{
    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{_mtx};

    absl::SetMinLogLevel(static_cast<absl::LogSeverityAtLeast>(cfg.loglevel));
    absl::SetStderrThreshold(static_cast<absl::LogSeverityAtLeast>(cfg.loglevel));
    absl::SetGlobalVLogLevel(cfg.vlog_level);

    if(cfg.install_failure_handler) install_failure_signal_handler();

    if(!cfg.logdir.empty() && !fs::exists(cfg.logdir))
    {
        fs::create_directories(cfg.logdir);
    }
}
}  // namespace common
}  // namespace rocprofiler
