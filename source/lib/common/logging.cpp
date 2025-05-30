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
namespace fs = ::rocprofiler::common::filesystem;

void
install_failure_signal_handler()
{
    static auto _once = std::once_flag{};
    std::call_once(_once, []() { google::InstallFailureSignalHandler(); });
}

struct log_level_info
{
    int32_t google_level  = 0;
    int32_t verbose_level = 0;
};

env_store
get_glog_env_config(const logging_config& cfg)
{
    auto as_env_config = [](std::string_view _var, auto _val) {
        return env_config{std::string{_var}, fmt::format("{}", _val), 1};
    };

    auto _data = std::vector<env_config>{
        as_env_config("GLOG_minloglevel", cfg.loglevel),
        as_env_config("GLOG_logtostderr", cfg.logtostderr ? 1 : 0),
        as_env_config("GLOG_alsologtostderr", cfg.alsologtostderr ? 1 : 0),
        as_env_config("GLOG_stderrthreshold", cfg.loglevel),
        as_env_config("GLOG_v", cfg.vlog_level),
    };

    if(!cfg.logdir.empty())
    {
        _data.emplace_back(as_env_config("GOOGLE_LOG_DIR", cfg.logdir));
        _data.emplace_back(as_env_config("GLOG_log_dir", cfg.logdir));
    }
    if(!cfg.vlog_modules.empty())
    {
        _data.emplace_back(as_env_config("GLOG_vmodule", cfg.vlog_modules));
    }

    return env_store{std::move(_data)};
}
}  // namespace

void
init_logging(std::string_view env_prefix, logging_config cfg)
{
    static auto _once = std::once_flag{};
    std::call_once(_once, [env_prefix, &cfg]() {
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

        auto to_lower = [](std::string val) {
            for(auto& itr : val)
                itr = tolower(itr);
            return val;
        };

        const auto env_opts = std::unordered_map<std::string_view, log_level_info>{
            {"trace", {google::INFO, ROCP_LOG_LEVEL_TRACE}},
            {"info", {google::INFO, ROCP_LOG_LEVEL_INFO}},
            {"warning", {google::WARNING, ROCP_LOG_LEVEL_WARNING}},
            {"error", {google::ERROR, ROCP_LOG_LEVEL_ERROR}},
            {"fatal", {google::FATAL, ROCP_LOG_LEVEL_NONE}}};

        auto supported = std::vector<std::string>{};
        supported.reserve(env_opts.size());
        for(auto itr : env_opts)
            supported.emplace_back(itr.first);

        if(cfg.name.empty()) cfg.name = to_lower(std::string{env_prefix});

        cfg.logdir       = get_env(fmt::format("{}_LOG_DIR", env_prefix), cfg.logdir);
        cfg.vlog_modules = get_env(fmt::format("{}_vmodule", env_prefix), cfg.vlog_modules);
        cfg.logtostderr  = cfg.logdir.empty();  // log to stderr if no log dir set
        // cfg.alsologtostderr = !cfg.logdir.empty();  // log to file if log dir set

        auto loglvl = to_lower(common::get_env(fmt::format("{}_LOG_LEVEL", env_prefix), ""));
        // default to warning
        auto& loglvl_v   = cfg.loglevel;
        auto& vlog_level = cfg.vlog_level;
        if(!loglvl.empty() && loglvl.find_first_not_of("-0123456789") == std::string::npos)
        {
            auto val = std::stol(loglvl);
            if(val < 0)
            {
                loglvl_v   = google::FATAL;
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
                loglvl_v   = itr.google_level;
                vlog_level = itr.verbose_level;
            }
        }
        else if(!loglvl.empty())
        {
            if(env_opts.find(loglvl) == env_opts.end())
                throw std::runtime_error{fmt::format(
                    "invalid specifier for {}_LOG_LEVEL: {}. Supported: {}",
                    env_prefix,
                    loglvl,
                    fmt::format("{}", fmt::join(supported.begin(), supported.end(), ", ")))};
            else
            {
                loglvl_v   = env_opts.at(loglvl).google_level;
                vlog_level = env_opts.at(loglvl).verbose_level;
            }
        }

        auto _env_store = get_glog_env_config(cfg);

        update_logging(cfg);
        _env_store.push();

        if(!google::IsGoogleLoggingInitialized())
        {
            static auto argv0 = get_argv0();
            // Prevent glog from crashing if vmodule is empty
            if(FLAGS_vmodule.empty()) FLAGS_vmodule = " ";

            google::InitGoogleLogging(argv0.c_str());

            // Swap out memory to avoid leaking the string
            if(!FLAGS_vmodule.empty()) std::string{}.swap(FLAGS_vmodule);
            if(!FLAGS_log_dir.empty()) std::string{}.swap(FLAGS_log_dir);
        }

        update_logging(cfg);

        ROCP_INFO << "logging initialized via " << fmt::format("{}_LOG_LEVEL", env_prefix)
                  << ". Log Level: " << loglvl << ". Verbose Log Level: " << vlog_level;

        _env_store.pop(false);
    });
}

void
update_logging(const logging_config& cfg)
{
    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{_mtx};

    FLAGS_timestamp_in_logfile_name = false;
    FLAGS_logtostderr               = cfg.logtostderr;
    FLAGS_minloglevel               = cfg.loglevel;
    FLAGS_stderrthreshold           = cfg.loglevel;
    FLAGS_alsologtostderr           = cfg.alsologtostderr;
    FLAGS_v                         = cfg.vlog_level;

    // if(!cfg.logdir.empty()) FLAGS_log_dir = cfg.logdir.c_str();

    if(cfg.install_failure_handler) install_failure_signal_handler();

    if(!cfg.logdir.empty() && !fs::exists(cfg.logdir))
    {
        fs::create_directories(cfg.logdir);

        if(cfg.logdir_gitignore)
        {
            auto ignore = fs::path{cfg.logdir} / ".gitignore";
            if(!fs::exists(ignore))
            {
                std::ofstream ofs{ignore.string()};
                ofs << "/**" << std::flush;
            }
        }
    }
}
}  // namespace common
}  // namespace rocprofiler
