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

#include "logging.hpp"
#include "environment.hpp"
#include "filesystem.hpp"

#if defined(_WIN32)
// WINDOWS-DIVERGENCE: <windows.h> drags in <wingdi.h>, which defines an
// ERROR macro that collides with google::ERROR (glog log severity). NOGDI
// suppresses GDI symbols (including the ERROR macro); WIN32_LEAN_AND_MEAN
// trims the rest of the unused Win32 surface. Both must be defined before
// any header that pulls in <windows.h>.
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOGDI
#        define NOGDI
#    endif
#    include <windows.h>
// Defensive guard: if any transitive include slipped in <wingdi.h> before
// our NOGDI took effect, drop the ERROR macro so glog severities resolve.
#    ifdef ERROR
#        undef ERROR
#    endif
#endif

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <glog/logging.h>

#include <fstream>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#if !defined(_WIN32)
#    include <sys/types.h>
#    include <unistd.h>
#endif

namespace rocprofiler_register
{
namespace logging
{
namespace
{
struct logging_config
{
    bool        logtostderr      = true;
    bool        alsologtostderr  = false;
    bool        logdir_gitignore = false;  // add .gitignore to logdir
    int32_t     loglevel         = google::GLOG_WARNING;
    std::string vlog_modules     = {};
    std::string name             = {};
    std::string logdir           = {};
};

struct log_level_info
{
    int32_t google_level  = 0;
    int32_t verbose_level = 0;
};

void
update_logging(const logging_config& cfg, bool setup_env = false, int env_override = 0)
{
    static auto _mtx = std::mutex{};
    auto        _lk  = std::unique_lock<std::mutex>{ _mtx };

    FLAGS_timestamp_in_logfile_name = false;
    FLAGS_logtostderr               = cfg.logtostderr;
    FLAGS_minloglevel               = cfg.loglevel;
    FLAGS_stderrthreshold           = cfg.loglevel;
    FLAGS_alsologtostderr           = cfg.alsologtostderr;

    // if(!cfg.logdir.empty()) FLAGS_log_dir = cfg.logdir.c_str();

    if(!cfg.logdir.empty() && !fs::exists(cfg.logdir))
    {
        fs::create_directories(cfg.logdir);

        if(cfg.logdir_gitignore)
        {
            auto ignore = fs::path{ cfg.logdir } / ".gitignore";
            if(!fs::exists(ignore))
            {
                std::ofstream ofs{ ignore.string() };
                ofs << "/**" << std::flush;
            }
        }
    }

    if(setup_env)
    {
        common::set_env("GLOG_minloglevel", cfg.loglevel, env_override);
        common::set_env("GLOG_logtostderr", cfg.logtostderr ? 1 : 0, env_override);
        common::set_env(
            "GLOG_alsologtostderr", cfg.alsologtostderr ? 1 : 0, env_override);
        common::set_env("GLOG_stderrthreshold", cfg.loglevel, env_override);
        if(!cfg.logdir.empty())
        {
            common::set_env("GOOGLE_LOG_DIR", cfg.logdir, env_override);
            common::set_env("GLOG_log_dir", cfg.logdir, env_override);
        }
        if(!cfg.vlog_modules.empty())
            common::set_env("GLOG_vmodule", cfg.vlog_modules, env_override);
    }
}

#define ROCP_REG_LOG_LEVEL_TRACE   4
#define ROCP_REG_LOG_LEVEL_INFO    3
#define ROCP_REG_LOG_LEVEL_WARNING 2
#define ROCP_REG_LOG_LEVEL_ERROR   1
#define ROCP_REG_LOG_LEVEL_NONE    0

void
init_logging(std::string_view env_prefix, logging_config cfg = logging_config{})
{
    static auto _once = std::once_flag{};
    std::call_once(_once, [env_prefix, &cfg]() {
        auto get_argv0 = []() {
#if defined(_WIN32)
            // WINDOWS-DIVERGENCE: no /proc/self/cmdline. The closest analogue
            // for argv[0] (executable path) is GetModuleFileNameW(NULL,...);
            // use the wide variant + UTF-8 conversion so non-ASCII install
            // paths survive (UTF-8 internal storage is the project rule).
            auto wbuf = std::wstring(MAX_PATH, L'\0');
            auto len  = ::GetModuleFileNameW(
                nullptr, wbuf.data(), static_cast<DWORD>(wbuf.size()));
            // Grow buffer if the path was truncated (long-path support).
            while(len == wbuf.size() &&
                  ::GetLastError() == ERROR_INSUFFICIENT_BUFFER)
            {
                wbuf.resize(wbuf.size() * 2);
                len = ::GetModuleFileNameW(
                    nullptr, wbuf.data(), static_cast<DWORD>(wbuf.size()));
            }
            if(len == 0) return std::string{};
            auto sz = ::WideCharToMultiByte(CP_UTF8,
                                            0,
                                            wbuf.data(),
                                            static_cast<int>(len),
                                            nullptr,
                                            0,
                                            nullptr,
                                            nullptr);
            if(sz <= 0) return std::string{};
            auto sarg = std::string(static_cast<std::size_t>(sz), '\0');
            ::WideCharToMultiByte(CP_UTF8,
                                  0,
                                  wbuf.data(),
                                  static_cast<int>(len),
                                  sarg.data(),
                                  sz,
                                  nullptr,
                                  nullptr);
            return sarg;
#else
            auto ifs  = std::ifstream{ "/proc/self/cmdline" };
            auto sarg = std::string{};
            while(ifs && !ifs.eof())
            {
                ifs >> sarg;
                if(!sarg.empty()) break;
            }
            return sarg;
#endif
        };

        auto to_lower = [](std::string val) {
            for(auto& itr : val)
                itr = tolower(itr);
            return val;
        };

        const auto env_opts = std::unordered_map<std::string_view, log_level_info>{
            { "trace", { google::GLOG_INFO, ROCP_REG_LOG_LEVEL_TRACE } },
            { "info", { google::GLOG_INFO, ROCP_REG_LOG_LEVEL_INFO } },
            { "warning", { google::GLOG_WARNING, ROCP_REG_LOG_LEVEL_WARNING } },
            { "error", { google::GLOG_ERROR, ROCP_REG_LOG_LEVEL_ERROR } },
            { "fatal", { google::GLOG_FATAL, ROCP_REG_LOG_LEVEL_NONE } }
        };

        auto supported = std::vector<std::string>{};
        supported.reserve(env_opts.size());
        for(auto itr : env_opts)
            supported.emplace_back(itr.first);

        if(cfg.name.empty()) cfg.name = to_lower(std::string{ env_prefix });

        cfg.logdir = common::get_env(fmt::format("{}_LOG_DIR", env_prefix), cfg.logdir);
        cfg.vlog_modules =
            common::get_env(fmt::format("{}_vmodule", env_prefix), cfg.vlog_modules);
        cfg.logtostderr = cfg.logdir.empty();  // log to stderr if no log dir set
        // cfg.alsologtostderr = !cfg.logdir.empty();  // log to file if log dir set

        auto loglvl =
            to_lower(common::get_env(fmt::format("{}_LOG_LEVEL", env_prefix), ""));
        // default to warning
        auto& loglvl_v = cfg.loglevel;
        if(!loglvl.empty() &&
           loglvl.find_first_not_of("-0123456789") == std::string::npos)
        {
            auto val = std::stol(loglvl);
            if(val < 0)
            {
                loglvl_v = google::GLOG_FATAL;
            }
            else
            {
                // default to trace in case val > ROCP_REG_LOG_LEVEL_TRACE
                auto itr = env_opts.at("trace");
                for(auto oitr : env_opts)
                {
                    if(oitr.second.verbose_level == val)
                    {
                        itr = oitr.second;
                        break;
                    }
                }
                loglvl_v = itr.google_level;
            }
        }
        else if(!loglvl.empty())
        {
            if(env_opts.find(loglvl) == env_opts.end())
                throw std::runtime_error{ fmt::format(
                    "invalid specifier for {}_LOG_LEVEL: {}. Supported: {}",
                    env_prefix,
                    loglvl,
                    fmt::format("{}",
                                fmt::join(supported.begin(), supported.end(), ", "))) };
            else
            {
                loglvl_v = env_opts.at(loglvl).google_level;
            }
        }

        update_logging(cfg, !google::IsGoogleLoggingInitialized());

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

        LOG(INFO) << "logging initialized via " << fmt::format("{}_LOG_LEVEL", env_prefix)
                  << ". Log Level: " << loglvl;
    });
}
}  // namespace

void
initialize()
{
    init_logging("ROCPROFILER_REGISTER");
}
}  // namespace logging
}  // namespace rocprofiler_register
