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

#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys
{

namespace
{

auto
to_lower(std::string_view s)
{
    std::string result;
    result.reserve(s.size());
    for(char c : s)
    {
        result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
};

}  // namespace

struct logger_settings_t
{
    logger_settings_t()
    : m_log_level(log_level_from_env(std::getenv("ROCPROFSYS_LOG_LEVEL")))
    , m_log_file(std::getenv("ROCPROFSYS_LOG_OUTPUT_FILENAME"))
    {
        const char* rocprofsys_monochrome_env = std::getenv("ROCPROFSYS_MONOCHROME");
        const char* monochrome_env            = std::getenv("MONOCHROME");
        if(rocprofsys_monochrome_env || monochrome_env)
        {
            const auto parse_monochrome_env = [](const char* env) {
                if(!env)
                {
                    return false;
                }
                const std::vector<std::string> true_values = { "1", "on", "true", "yes" };
                if(std::find(true_values.begin(), true_values.end(), std::string(env)) !=
                   true_values.end())
                {
                    return true;
                }
                return false;
            };

            m_monochrome = parse_monochrome_env(rocprofsys_monochrome_env) ||
                           parse_monochrome_env(monochrome_env);
        }
    }

    spdlog::level::level_enum log_level_from_env(const char* env)
    {
        if(!env)
        {
            return m_default_level;
        }

        return parse_level(env);
    }

    spdlog::level::level_enum parse_level(std::string_view level)
    {
        const auto lower = to_lower(level);

        if(lower == "trace") return spdlog::level::trace;
        if(lower == "debug") return spdlog::level::debug;
        if(lower == "info") return spdlog::level::info;
        if(lower == "warn" || lower == "warning") return spdlog::level::warn;
        if(lower == "error" || lower == "err") return spdlog::level::err;
        if(lower == "critical") return spdlog::level::critical;
        if(lower == "off") return spdlog::level::off;

        return m_default_level;
    }

    spdlog::level::level_enum get_log_level() const { return m_log_level; }

    std::string get_log_file() const
    {
        if(m_log_file == nullptr)
        {
            return {};
        }

        return { m_log_file };
    }

    const char* get_log_pattern() const
    {
        return m_monochrome ? m_log_pattern_monochrome : m_log_pattern;
    }

protected:
    const spdlog::level::level_enum m_default_level{ spdlog::level::info };
    const spdlog::level::level_enum m_log_level;
    const char*                     m_log_file;

    bool m_monochrome{ false };

    // Pattern:
    // [TIME][P:PID T:THREAD_ID][LOG_LEVEL][FILE:LINE FUNCTION] MESSAGE
    const char* m_log_pattern{ "%^[%H:%M:%S.%e][P:%P T:%t][%s:%# %!][%l]%$ %v" };
    const char* m_log_pattern_monochrome{ "[%H:%M:%S.%e][P:%P T:%t][%s:%# %!][%l] %v" };
};

class logger_t
{
public:
    static spdlog::logger& instance()
    {
        static auto _instance = [] {
            logger_settings_t logger_settings;

            std::vector<spdlog::sink_ptr> sinks;

            sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_mt>());

            auto log_file = logger_settings.get_log_file();
            if(!log_file.empty())
            {
                sinks.push_back(
                    std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true));
            }

            const auto* logger_name = "rocprofsys";
            auto        log =
                std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());

            log->set_pattern(logger_settings.get_log_pattern());
            log->set_level(logger_settings.get_log_level());

            spdlog::register_logger(log);
            return log;
        }();
        return *_instance;
    }

    logger_t() = delete;
};

}  // namespace rocprofsys
