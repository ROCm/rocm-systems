// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <new>
#include <pthread.h>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <sys/cdefs.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace rocprofsys
{

namespace
{

inline __attribute__((always_inline)) auto
to_lower(std::string_view s)
{
    std::string result;
    result.reserve(s.size());
    for(char c : s)
    {
        result += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return result;
}

std::string
include_process_id_in_filename(std::string_view filename)
{
    if(filename.empty())
    {
        return std::string{};
    }

    auto last_sep       = filename.find_last_of('/');
    auto filename_start = (last_sep == std::string_view::npos) ? 0 : last_sep + 1;
    auto dot_pos        = filename.find_last_of('.');

    bool has_extension =
        (dot_pos != std::string_view::npos) && (dot_pos > filename_start);

    std::string pid_suffix = "_" + std::to_string(getpid());

    if(!has_extension)
    {
        return std::string(filename) + pid_suffix;
    }

    return std::string(filename.substr(0, dot_pos)) + pid_suffix +
           std::string(filename.substr(dot_pos));
}

inline bool
parse_boolean_env(const char* env)
{
    if(!env)
    {
        return false;
    }
    constexpr std::array<const char*, 4> true_values = { "1", "on", "true", "yes" };

    auto lower = to_lower(env);
    return std::any_of(true_values.begin(), true_values.end(),
                       [&](const std::string& value) { return value == lower; });
}

}  // namespace

struct logger_settings_t
{
    logger_settings_t()
    : m_log_level(log_level_from_env(std::getenv("ROCPROFSYS_LOG_LEVEL")))
    , m_log_file(std::getenv("ROCPROFSYS_LOG_FILE"))
    {
        const char* rocprofsys_monochrome_env = std::getenv("ROCPROFSYS_MONOCHROME");
        const char* monochrome_env            = std::getenv("MONOCHROME");
        if(rocprofsys_monochrome_env || monochrome_env)
        {
            m_monochrome = parse_boolean_env(rocprofsys_monochrome_env) ||
                           parse_boolean_env(monochrome_env);
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
    const char* m_log_pattern{ "%^[%H:%M:%S.%e][P:%P T:%t][%s:%# %!][%l] %v %$" };
    const char* m_log_pattern_monochrome{ "[%H:%M:%S.%e][P:%P T:%t][%s:%# %!][%l] %v" };
};

class logger_t
{
public:
    static spdlog::logger& instance()
    {
        static std::shared_ptr<spdlog::logger> _instance;
        static std::atomic<bool>               _initialized{ false };
        static std::mutex                      _init_mutex;
        static std::atomic<bool>               _log_lock{ false };

        static std::once_flag _atfork_flag;
        std::call_once(_atfork_flag, [] {
            pthread_atfork(
                // prefork: lock _init_mutex.  Safe because it is only held
                // briefly during one-time logger creation, never during
                // normal log calls — so no lock-ordering inversion with
                // glibc-internal locks acquired by fork().
                [] { _init_mutex.lock(); },
                // parent postfork: unlock, resume normal operation.
                [] { _init_mutex.unlock(); },
                // child postfork: reset the atomic spinlock (well-defined
                // atomic store — no UB unlike placement-new on a mutex),
                // then safely delete the old logger (_st sinks have no
                // internal mutex).  No leak.
                [] {
                    _log_lock.store(false, std::memory_order_relaxed);
                    _instance.reset();
                    _initialized.store(false, std::memory_order_release);
                    _init_mutex.unlock();
                });
        });

        if(_initialized.load(std::memory_order_acquire))
        {
            return *_instance;
        }

        std::lock_guard<std::mutex> lock(_init_mutex);
        if(!_initialized.load(std::memory_order_relaxed))
        {
            _instance = create_logger(_log_lock);
            _initialized.store(true, std::memory_order_release);
        }

        return *_instance;
    }

    logger_t() = delete;

private:
    // Serializes _st sink access through an atomic spinlock.  Unlike a
    // mutex or shared_mutex the spinlock can be safely reset in a
    // post-fork child with a plain atomic store — no undefined behaviour,
    // no TSan complaints, and the old logger can be cleanly deleted.
    class fork_safe_logger : public spdlog::logger
    {
    public:
        template <typename It>
        fork_safe_logger(std::string name, It begin, It end, std::atomic<bool>& log_lock)
        : spdlog::logger(std::move(name), begin, end)
        , m_log_lock(log_lock)
        {}

    protected:
        void sink_it_(const spdlog::details::log_msg& msg) override
        {
            while(m_log_lock.exchange(true, std::memory_order_acquire))
                std::this_thread::yield();
            spdlog::logger::sink_it_(msg);
            m_log_lock.store(false, std::memory_order_release);
        }

        void flush_() override
        {
            while(m_log_lock.exchange(true, std::memory_order_acquire))
                std::this_thread::yield();
            spdlog::logger::flush_();
            m_log_lock.store(false, std::memory_order_release);
        }

    private:
        std::atomic<bool>& m_log_lock;
    };

    static std::shared_ptr<spdlog::logger> create_logger(std::atomic<bool>& log_lock)
    {
        logger_settings_t logger_settings;

        std::vector<spdlog::sink_ptr> sinks;

        // Use _st sinks — no internal mutex.  Thread safety is provided
        // by fork_safe_logger's atomic spinlock, which can be safely
        // reset after fork() with a plain atomic store (no UB).
        sinks.push_back(std::make_shared<spdlog::sinks::stdout_color_sink_st>());

        auto log_file = logger_settings.get_log_file();
        if(!log_file.empty())
        {
            log_file = include_process_id_in_filename(log_file);

            sinks.push_back(
                std::make_shared<spdlog::sinks::basic_file_sink_st>(log_file, true));
        }

        auto log = std::shared_ptr<spdlog::logger>(
            new fork_safe_logger(s_logger_name, sinks.begin(), sinks.end(), log_lock));

        log->set_pattern(logger_settings.get_log_pattern());
        log->set_level(logger_settings.get_log_level());

        return log;
    }

    static constexpr const char* s_logger_name = "rocprofiler-systems";
};

}  // namespace rocprofsys
