// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "debug.hpp"

// TODO: Whatever you do, do not use get_debug() directly. It will crash every bin due to some initlaization order nonsense.

// TODO: Fix this
#undef SPDLOG_ACTIVE_LEVEL
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE
// #define SPDLOG_ACRIVE_LEVEL SPDLOG_LEVEL_OFF

#include <spdlog/spdlog.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/rotating_file_sink.h>

// TODO: Determine what is NOT NEEDED
#include <timemory/api.hpp>
#include <timemory/backends/dmp.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/log/logger.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/signals/signal_handlers.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/locking.hpp>
#include <timemory/utility/utility.hpp>

#include "debug.hpp"

namespace rocprofsys
{
inline namespace config
{
// bool
// get_debug_tid() ROCPROFSYS_HOT;

// bool
// get_debug_pid() ROCPROFSYS_HOT;

// bool
// get_is_continuous_integration() ROCPROFSYS_HOT;

// bool
// get_debug_env() ROCPROFSYS_HOT;

// int
// get_verbose_env() ROCPROFSYS_HOT;

} // namespace config
namespace debug
{

// Standardized logger name
inline constexpr std::string_view std_logger_name = "rocprof-sys-logger";

// Global logger pointers
using logger_t = std::shared_ptr<spdlog::logger>;

// TODO: I shouldn't need to clean these up (VERIFY). i.e. run program without cleanup func
inline logger_t rocprofsys_npt_logger;
inline logger_t rocprofsys_nptf_logger;
inline logger_t rocprofsys_np_logger;
inline logger_t rocprofsys_npf_logger;

class standard_name_formatter : public spdlog::custom_flag_formatter {
public:
    void format(const spdlog::details::log_msg&, 
            const std::tm&, 
            spdlog::memory_buf_t& dest) override 
    {
        dest.append(std_logger_name.data(), 
                        std_logger_name.data() + std_logger_name.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<standard_name_formatter>();
    }
};

// Equivalent implementation of ROCPROFSYS_FUNCTION
// No compile time version to strip __FUNCTION__ of _hidden can be done
class function_name_formatter : public spdlog::custom_flag_formatter {
public:
    void format(const spdlog::details::log_msg& msg, 
            const std::tm&, 
            spdlog::memory_buf_t& dest) override {
        // Get function name from source location
        std::string_view func_name = msg.source.funcname;
        
        // Find and strip "_hidden" suffix
        auto hidden_pos = func_name.find("_hidden");
        if (hidden_pos != std::string_view::npos) {
            func_name = func_name.substr(0, hidden_pos);
        }
        
        dest.append(func_name.data(), func_name.data() + func_name.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<function_name_formatter>();
    }
};


// Equivalent implementation of ROCPROFSYS_DEBUG_PROCESS_IDENTIFIER
class process_identifier_formatter : public spdlog::custom_flag_formatter {
public:
    void format(const spdlog::details::log_msg&, 
               const std::tm&, 
               spdlog::memory_buf_t& dest) override {
        #if defined(ROCPROFSYS_USE_MPI)
        int process_id = static_cast<int>(::tim::dmp::rank());
        #elif defined(ROCPROFSYS_USE_MPI_HEADERS)
        int process_id = (::tim::dmp::is_initialized()) 
            ? static_cast<int>(::tim::dmp::rank())
            : static_cast<int>(spdlog::details::os::pid());
        #else
        int process_id = static_cast<int>(spdlog::details::os::pid()); // TODO: We might need to force use of timemory instead (do we use static_cast<int>(::tim::process::get_id()))?
        #endif
        auto id_str = std::to_string(process_id);
        dest.append(id_str.data(), id_str.data() + id_str.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<process_identifier_formatter>();
    }
};

// Equivalent implementation of ROCPROFSYS_DEBUG_THREAD_IDENTIFIER
class thread_identifier_formatter : public spdlog::custom_flag_formatter {
public:
    void format(const spdlog::details::log_msg&, 
               const std::tm&, 
               spdlog::memory_buf_t& dest) override {
        int thread_id = ::rocprofsys::debug::get_tid();
        auto id_str = std::to_string(thread_id);
        dest.append(id_str.data(), id_str.data() + id_str.size());
    }

    std::unique_ptr<custom_flag_formatter> clone() const override {
        return spdlog::details::make_unique<thread_identifier_formatter>();
    }
};

// Handles spdlog sink and logger initialization
void 
init_ddebug();

inline void
flush_spdlogimpl()
{
    // TODO: Better way to do this?
    // TODO: Is there a buffer length somewhere?
    if (rocprofsys_npt_logger) rocprofsys_npt_logger->flush();
    if (rocprofsys_nptf_logger) rocprofsys_nptf_logger->flush();
    if (rocprofsys_np_logger) rocprofsys_np_logger->flush();
    if (rocprofsys_npf_logger) rocprofsys_npf_logger->flush();

    // TODO: May need to flush streams?
}

// TODO: Make this actually good
// TODO: Also, the Backtrace [... name ...] is wrong
template <size_t Depth, int64_t Offset = 1>
inline void
log_demangled_backtrace(std::shared_ptr<spdlog::logger> logger,
                        const std::string& _info = "")
{
    auto bt = ::tim::get_demangled_unw_backtrace<Depth, Offset + 1>();
    
    // Build the backtrace string
    std::ostringstream oss;
    oss << "Backtrace";
    if(!_info.empty())
        oss << " " << _info;
    oss << ":\n";

    for(const auto& frame : bt)
    {
        if(frame.length() > 0)
            oss << "    " << frame << "\n";
    }
    SPDLOG_LOGGER_CRITICAL(logger, "{}", oss.str());
}

} // namespace debug
} // namespace rocprofsys

// General TODOS
// TODO: Do I need to flush before and/or after an spdlog call
// TODO: Revamp ROCPROFSYS_MONOCHROME as that is a flag

//--------------------------------------------------------------------------------------//
//
//  We keep old ROCPROFSYS_FUNCTION implementation for use in logging messages
//
//--------------------------------------------------------------------------------------//

#if defined(__clang__) || (__GNUC__ < 9)
#    define ROCPROFSYS_FUNCTION                                                          \
        std::string{ __FUNCTION__ }                                                      \
            .substr(0, std::string_view{ __FUNCTION__ }.find("_hidden"))                 \
            .c_str()
#    define ROCPROFSYS_PRETTY_FUNCTION                                                   \
        std::string{ __PRETTY_FUNCTION__ }                                               \
            .substr(0, std::string_view{ __PRETTY_FUNCTION__ }.find("_hidden"))          \
            .c_str()
#else
#    define ROCPROFSYS_FUNCTION                                                          \
        ::rocprofsys::debug::get_chars(                                                  \
            std::string_view{ __FUNCTION__ },                                            \
            std::make_index_sequence<std::min(                                           \
                std::string_view{ __FUNCTION__ }.find("_hidden"),                        \
                std::string_view{ __FUNCTION__ }.length())>{})                           \
            .data()
#    define ROCPROFSYS_PRETTY_FUNCTION                                                   \
        ::rocprofsys::debug::get_chars(                                                  \
            std::string_view{ __PRETTY_FUNCTION__ },                                     \
            std::make_index_sequence<std::min(                                           \
                std::string_view{ __PRETTY_FUNCTION__ }.find("_hidden"),                 \
                std::string_view{ __PRETTY_FUNCTION__ }.length())>{})                    \
            .data()
#endif



//--------------------------------------------------------------------------------------//
//
//  Log Type Handlers
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_HANDLE_TRACE_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)            \
    if(SHOW_TID && SHOW_FUNC)                                                               \
        SPDLOG_LOGGER_TRACE(::rocprofsys::debug::rocprofsys_nptf_logger, __VA_ARGS__);    \
    else if(SHOW_TID)                                                                        \
        SPDLOG_LOGGER_TRACE(::rocprofsys::debug::rocprofsys_npt_logger, __VA_ARGS__);     \
    else if(SHOW_FUNC)                                                                       \
        SPDLOG_LOGGER_TRACE(::rocprofsys::debug::rocprofsys_npf_logger, __VA_ARGS__);     \
    else                                                                                    \
        SPDLOG_LOGGER_TRACE(::rocprofsys::debug::rocprofsys_np_logger, __VA_ARGS__);      

#define ROCPROFSYS_HANDLE_DEBUG_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)            \
    if(SHOW_TID && SHOW_FUNC)                                                             \
    {                                                                                       \
        SPDLOG_LOGGER_DEBUG(::rocprofsys::debug::rocprofsys_nptf_logger, __VA_ARGS__);    \
    }                                                                                      \
    else if(SHOW_TID)                                                                        \
        SPDLOG_LOGGER_DEBUG(::rocprofsys::debug::rocprofsys_npt_logger, __VA_ARGS__);     \
    else if(SHOW_FUNC)                                                                       \
        SPDLOG_LOGGER_DEBUG(::rocprofsys::debug::rocprofsys_npf_logger, __VA_ARGS__);     \
    else                                                                                    \
        SPDLOG_LOGGER_DEBUG(::rocprofsys::debug::rocprofsys_np_logger, __VA_ARGS__);      

#define ROCPROFSYS_HANDLE_INFO_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)            \
    if(SHOW_TID && SHOW_FUNC)                                                               \
        SPDLOG_LOGGER_INFO(::rocprofsys::debug::rocprofsys_nptf_logger, __VA_ARGS__);    \
    else if(SHOW_TID)                                                                        \
        SPDLOG_LOGGER_INFO(::rocprofsys::debug::rocprofsys_npt_logger, __VA_ARGS__);     \
    else if(SHOW_FUNC)                                                                       \
        SPDLOG_LOGGER_INFO(::rocprofsys::debug::rocprofsys_npf_logger, __VA_ARGS__);     \
    else                                                                                    \
        SPDLOG_LOGGER_INFO(::rocprofsys::debug::rocprofsys_np_logger, __VA_ARGS__);      

#define ROCPROFSYS_HANDLE_WARN_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)            \
    if(SHOW_TID && SHOW_FUNC)                                                               \
        SPDLOG_LOGGER_WARN(::rocprofsys::debug::rocprofsys_nptf_logger, __VA_ARGS__);    \
    else if(SHOW_TID)                                                                        \
        SPDLOG_LOGGER_WARN(::rocprofsys::debug::rocprofsys_npt_logger, __VA_ARGS__);     \
    else if(SHOW_FUNC)                                                                       \
        SPDLOG_LOGGER_WARN(::rocprofsys::debug::rocprofsys_npf_logger, __VA_ARGS__);     \
    else                                                                                    \
        SPDLOG_LOGGER_WARN(::rocprofsys::debug::rocprofsys_np_logger, __VA_ARGS__);      

#define ROCPROFSYS_HANDLE_ERROR_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)            \
    if(SHOW_TID && SHOW_FUNC)                                                               \
        SPDLOG_LOGGER_ERROR(::rocprofsys::debug::rocprofsys_nptf_logger, __VA_ARGS__);    \
    else if(SHOW_TID)                                                                        \
        SPDLOG_LOGGER_ERROR(::rocprofsys::debug::rocprofsys_npt_logger, __VA_ARGS__);     \
    else if(SHOW_FUNC)                                                                       \
        SPDLOG_LOGGER_ERROR(::rocprofsys::debug::rocprofsys_npf_logger, __VA_ARGS__);     \
    else                                                                                    \
        SPDLOG_LOGGER_ERROR(::rocprofsys::debug::rocprofsys_np_logger, __VA_ARGS__);      

#define ROCPROFSYS_HANDLE_CRITICAL_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)            \
    if(SHOW_TID && SHOW_FUNC)                                                               \
        SPDLOG_LOGGER_CRITICAL(::rocprofsys::debug::rocprofsys_nptf_logger, __VA_ARGS__);    \
    else if(SHOW_TID)                                                                        \
        SPDLOG_LOGGER_CRITICAL(::rocprofsys::debug::rocprofsys_npt_logger, __VA_ARGS__);     \
    else if(SHOW_FUNC)                                                                       \
        SPDLOG_LOGGER_CRITICAL(::rocprofsys::debug::rocprofsys_npf_logger, __VA_ARGS__);     \
    else                                                                                    \
        SPDLOG_LOGGER_CRITICAL(::rocprofsys::debug::rocprofsys_np_logger, __VA_ARGS__);    



//--------------------------------------------------------------------------------------//
//
//  Conditional stuff (level above)
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_FAILURE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, METHOD, ...)   \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                         \
    {                                                                                       \
        ROCPROFSYS_HANDLE_CRITICAL_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, __VA_ARGS__)    \
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);                            \
        ::rocprofsys::debug::log_demangled_backtrace<64>(                                   \
            rocprofsys::debug::rocprofsys_npt_logger, TIMEMORY_FILE_LINE_FUNC_STRING);      \
        ::rocprofsys::debug::flush_spdlogimpl();                                            \
        METHOD;                                                                             \
    }

#define ROCPROFSYS_CONDITIONAL_PRINT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)             \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&               \
                           ::rocprofsys::config::get_debug_pid()))                          \
    {                                                                                       \
        ROCPROFSYS_HANDLE_DEBUG_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, __VA_ARGS__)       \
        ::rocprofsys::debug::flush_spdlogimpl();                                            \
    }

#define ROCPROFSYS_CONDITIONAL_WARN_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)              \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&               \
                           ::rocprofsys::config::get_debug_pid()))                          \
    {                                                                                       \
    ROCPROFSYS_HANDLE_WARN_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, __VA_ARGS__)            \
    ::rocprofsys::debug::flush_spdlogimpl();                                                \
    }                                                                                       \

    
// TODO: I actually have no idea how this works
#define ROCPROFSYS_CONDITIONAL_THROWER_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, TYPE, ...)       \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                          \
    {                                                                                        \
        bool _print_backtrace = ::rocprofsys::get_debug_env() ||                             \
                                ::rocprofsys::get_verbose() >= 2 ||                          \
                                ::rocprofsys::get_is_continuous_integration();               \
        auto _exception_msg = fmt::format(__VA_ARGS__);                                      \
        ROCPROFSYS_HANDLE_CRITICAL_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, "{}", _exception_msg); \
        if(_print_backtrace)                                                                 \
        {                                                                                    \
            if(SHOW_TID)                                                                     \
                ::rocprofsys::debug::log_demangled_backtrace<64>(                            \
                    ::rocprofsys::debug::rocprofsys_npt_logger,                              \
                    TIMEMORY_FILE_LINE_FUNC_STRING);                                         \
            else                                                                             \
                ::rocprofsys::debug::log_demangled_backtrace<64>(                            \
                    ::rocprofsys::debug::rocprofsys_np_logger,                               \
                    TIMEMORY_FILE_LINE_FUNC_STRING);                                         \
            throw ::rocprofsys::exception<TYPE>(_exception_msg, true);                       \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            throw ::rocprofsys::exception<TYPE>(_exception_msg, false);                      \
        }                                                                                    \
    }

//--------------------------------------------------------------------------------------//
//
//  CI Stuff
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CI_FAIL_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)               \
    ROCPROFSYS_CONDITIONAL_FAILURE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ::rocprofsys::get_is_continuous_integration() && (COND),                \
        ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)), __VA_ARGS__)

#define ROCPROFSYS_CI_ABORT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)                  \
    ROCPROFSYS_CONDITIONAL_FAILURE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ::rocprofsys::get_is_continuous_integration() && (COND),                \
        ROCPROFSYS_ESC(::std::abort()), __VA_ARGS__)

#define ROCPROFSYS_CI_THROW_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)                  \
    ROCPROFSYS_CONDITIONAL_THROWER_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ::rocprofsys::get_is_continuous_integration() && (COND),                \
        std::runtime_error, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Debug macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_DEBUG_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)                               \
    ROCPROFSYS_CONDITIONAL_PRINT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ::rocprofsys::get_debug_env(), __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Verbose macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_VERBOSE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, LEVEL, ...)                      \
    ROCPROFSYS_CONDITIONAL_PRINT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC,                            \
                                            ::rocprofsys::get_debug_env() ||                \
                                           (::rocprofsys::get_verbose_env() >= LEVEL),      \
                                            __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Warning macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_WARNING_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, LEVEL, ...)                         \
    ROCPROFSYS_CONDITIONAL_WARN_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ::rocprofsys::get_debug_env() || \
                                            (::rocprofsys::get_verbose_env() >= LEVEL),        \
                                            __VA_ARGS__)    

#define ROCPROFSYS_WARNING_IF_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)                        \
    ROCPROFSYS_CONDITIONAL_WARN_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Print macros. Unified.
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_PRINT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)                               \
    ROCPROFSYS_CONDITIONAL_PRINT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, true, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Throw macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_THROW_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)                               \
    ROCPROFSYS_CONDITIONAL_THROWER_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, true, std::runtime_error, __VA_ARGS__) \

    //TODO: Figure out a way to match this?
#define ROCPROFSYS_CONDITIONAL_THROW_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)                        \
    ROCPROFSYS_CONDITIONAL_THROWER_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, std::runtime_error, __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Fail macro
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_FAIL_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)                            \
    ROCPROFSYS_CONDITIONAL_FAILURE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, true,                \
        ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)), __VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Abort macro
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_ABORT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...)                           \
    ROCPROFSYS_CONDITIONAL_FAILURE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, true,                \
        ROCPROFSYS_ESC(::std::abort()), __VA_ARGS__)

// TODO: Combine this with one above
#define ROCPROFSYS_CONDITIONAL_ABORT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)         \
    ROCPROFSYS_CONDITIONAL_FAILURE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND,                \
        ROCPROFSYS_ESC(::std::abort()), __VA_ARGS__)



//--------------------------------------------------------------------------------------//
//
//  Assert Macro
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_REQUIRE_SPDLOGIMPL(COND, ...)     \
    if(!(COND))                           \
        ROCPROFSYS_CONDITIONAL_FAILURE_SPDLOGIMPL(true, true, true, \
            ROCPROFSYS_ESC(::std::exit(EXIT_FAILURE)), __VA_ARGS__)


// TODO: DELETE

/*
//--------------------------------------------------------------------------------------//
//
//  Conditional stuff (level above)
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CONDITIONAL_FAILURE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, METHOD, ...)   \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                         \
    {                                                                                       \
        ROCPROFSYS_HANDLE_CRITICAL_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, __VA_ARGS__)    \
        ::rocprofsys::set_state(::rocprofsys::State::Finalized);                            \
        ::rocprofsys::debug::log_demangled_backtrace<64>(                                   \
            rocprofsys::debug::rocprofsys_npt_logger, TIMEMORY_FILE_LINE_FUNC_STRING);      \
        ::rocprofsys::debug::flush_spdlogimpl();                                            \
        METHOD;                                                                             \
    }

#define ROCPROFSYS_CONDITIONAL_PRINT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)             \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&               \
                           ::rocprofsys::config::get_debug_pid()))                          \
    {                                                                                       \
        ROCPROFSYS_HANDLE_DEBUG_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, __VA_ARGS__)       \
        ::rocprofsys::debug::flush_spdlogimpl();                                            \
    }

#define ROCPROFSYS_CONDITIONAL_WARN_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...)              \
    if(ROCPROFSYS_UNLIKELY((COND) && ::rocprofsys::config::get_debug_tid() &&               \
                           ::rocprofsys::config::get_debug_pid()))                          \
    {                                                                                       \
    ROCPROFSYS_HANDLE_WARN_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, __VA_ARGS__)            \
    ::rocprofsys::debug::flush_spdlogimpl();                                                \
    }                                                                                       \

// TODO: I actually have no idea how this works
#define ROCPROFSYS_CONDITIONAL_THROWER_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, TYPE, ...)       \
    if(ROCPROFSYS_UNLIKELY((COND)))                                                          \
    {                                                                                        \
        bool _print_backtrace = ::rocprofsys::get_debug_env() ||                             \
                                ::rocprofsys::get_verbose() >= 2 ||                          \
                                ::rocprofsys::get_is_continuous_integration();               \
        auto _exception_msg = fmt::format(__VA_ARGS__);                                      \
        ROCPROFSYS_HANDLE_CRITICAL_LOG_TYPE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, "{}", _exception_msg); \
        if(_print_backtrace)                                                                 \
        {                                                                                    \
            if(SHOW_TID)                                                                     \
                ::rocprofsys::debug::log_demangled_backtrace<64>(                            \
                    ::rocprofsys::debug::rocprofsys_npt_logger,                              \
                    TIMEMORY_FILE_LINE_FUNC_STRING);                                         \
            else                                                                             \
                ::rocprofsys::debug::log_demangled_backtrace<64>(                            \
                    ::rocprofsys::debug::rocprofsys_np_logger,                               \
                    TIMEMORY_FILE_LINE_FUNC_STRING);                                         \
            throw ::rocprofsys::exception<TYPE>(_exception_msg, true);                       \
        }                                                                                    \
        else                                                                                 \
        {                                                                                    \
            throw ::rocprofsys::exception<TYPE>(_exception_msg, false);                      \
        }                                                                                    \
    }


//--------------------------------------------------------------------------------------//
//
//  CI Stuff
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_CI_FAIL_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_CI_FAIL_SPDLOGIMPL" << std::endl;

#define ROCPROFSYS_CI_ABORT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_CI_ABORT_SPDLOGIMPL" << std::endl;

#define ROCPROFSYS_CI_THROW_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_CI_THROW_SPDLOGIMPL" << std::endl;

//--------------------------------------------------------------------------------------//
//
//  Debug macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_DEBUG_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_DEBUG_SPDLOGIMPL" << std::endl;

//--------------------------------------------------------------------------------------//
//
//  Verbose macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_VERBOSE_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, LEVEL, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_VERBOSE_SPDLOGIMPL" << std::endl;

//--------------------------------------------------------------------------------------//
//
//  Warning macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_WARNING_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, LEVEL, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_WARNING_SPDLOGIMPL" << std::endl;

#define ROCPROFSYS_WARNING_IF_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_WARNING_IF_SPDLOGIMPL" << std::endl;

// TOOD: What in the world is this?
#define ROCPROFSYS_REQUIRE(...) TIMEMORY_REQUIRE(__VA_ARGS__)

//--------------------------------------------------------------------------------------//
//
//  Print macros. Unified.
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_PRINT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_PRINT_SPDLOGIMPL" << std::endl;

//--------------------------------------------------------------------------------------//
//
//  Throw macros
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_THROW_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_THROW_SPDLOGIMPL" << std::endl;

//TODO: Figure out a way to match this?
#define ROCPROFSYS_CONDITIONAL_THROW_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_CONDITIONAL_THROW_SPDLOGIMPL" << std::endl;

//--------------------------------------------------------------------------------------//
//
//  Fail macro
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_FAIL_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_FAIL_SPDLOGIMPL" << std::endl;

//--------------------------------------------------------------------------------------//
//
//  Abort macro
//
//--------------------------------------------------------------------------------------//

#define ROCPROFSYS_ABORT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_ABORT_SPDLOGIMPL" << std::endl;

// TODO: Combine this with one above
#define ROCPROFSYS_CONDITIONAL_ABORT_SPDLOGIMPL(SHOW_TID, SHOW_FUNC, COND, ...) \
    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] ROCPROFSYS_CONDITIONAL_ABPRT_SPDLOGIMPL" << std::endl;

*/

// TODO: Is below needed?
#include <string>

namespace std
{
inline std::string
to_string(bool _v)
{
    return (_v) ? "true" : "false";
}
}  // namespace std
