// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc.
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

#include <fmt/format.h>
#include <glog/logging.h>
#include <glog/vlog_is_on.h>

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
void
install_failure_signal_handler()
{
    static auto _once = std::once_flag{};
    std::call_once(_once, []() { google::InstallFailureSignalHandler(); });
}
}  // namespace

void
init_logging(std::string_view env_var, logging_config cfg)
{
    static auto _once = std::once_flag{};
    std::call_once(_once, [env_var, &cfg]() {
        auto get_argv0 = []() {
            auto ifs  = std::ifstream{"/proc/self/cmdline"};
            auto sarg = std::string{};
            while(ifs && !ifs.eof())
            {
                ifs >> sarg;
                if(!sarg.empty()) break;
            }
            return sarg;
        };

        auto loglvl = common::get_env(env_var, "");
        for(auto& itr : loglvl)
            itr = tolower(itr);
        // default to warning
        auto& loglvl_v   = cfg.loglevel;
        auto& vlog_level = cfg.vlog_level;
        if(!loglvl.empty() && loglvl.find_first_not_of("0123456789") == std::string::npos)
        {
            loglvl_v   = std::stoul(loglvl);
            vlog_level = loglvl_v;
        }
        else if(!loglvl.empty())
        {
            const auto opts = std::unordered_map<std::string_view, std::pair<uint32_t, uint32_t>>{
                {"trace", {google::INFO, ROCP_LEVEL_TRACE}},
                {"info", {google::INFO, ROCP_LEVEL_INFO}},
                {"warning", {google::WARNING, ROCP_LEVEL_WARNING}},
                {"error", {google::ERROR, ROCP_NO_VLOG}},
                {"fatal", {google::ERROR, ROCP_NO_VLOG}}};
            if(opts.find(loglvl) == opts.end())
                throw std::runtime_error{fmt::format(
                    "invalid specifier for ROCPROFILER_LOG_LEVEL: {}. Supported: trace, info, "
                    "warning, error, fatal",
                    loglvl)};
            else
            {
                loglvl_v   = opts.at(loglvl).first;
                vlog_level = opts.at(loglvl).second;
            }
        }

        update_logging(cfg, true);

        if(!google::IsGoogleLoggingInitialized())
        {
            static auto argv0 = get_argv0();
            // Prevent glog from crashing if vmodule is empty
            if(FLAGS_vmodule.empty())
            {
                FLAGS_vmodule = " ";
            }

            google::InitGoogleLogging(argv0.c_str());
            ROCP_WARNING << "Log Level: " << loglvl << " VLOG Level: " << vlog_level;

            // Swap out memory to avoid leaking the string
            if(FLAGS_vmodule == " ")
            {
                std::string().swap(FLAGS_vmodule);
            }
        }

        update_logging(cfg);
        ROCP_INFO << "logging initialized via " << env_var;
    });
}

void
update_logging(const logging_config& cfg, bool setup_env, int env_override)
{
    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{_mtx};

    FLAGS_timestamp_in_logfile_name = false;
    FLAGS_minloglevel               = cfg.loglevel;
    FLAGS_stderrthreshold           = cfg.loglevel;
    FLAGS_logtostderr               = cfg.logtostderr;
    FLAGS_alsologtostderr           = cfg.alsologtostderr;
    FLAGS_v                         = cfg.vlog_level;

    if(cfg.install_failure_handler) install_failure_signal_handler();

    if(setup_env)
    {
        common::set_env("GLOG_v", cfg.vlog_level, env_override);
        common::set_env("GOOGLE_LOG_DIR", get_env("PWD", ""), env_override);
    }
}
}  // namespace common
}  // namespace rocprofiler
