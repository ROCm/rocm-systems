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

#include "spdlogdebug.hpp"

namespace rocprofsys
{
namespace debug
{

void 
init_ddebug()
{
    // Function is called in library.cpp. Because some binaries won't enter some functions whereas others do, 
    //  init_ddebug() is put in multiple areas. To avoid multiple initializations, we must ensure this is
    //  called only once.

    // TODO: make this std::once
    static bool initialized = false;
    if (initialized) 
    {
        ROCPROFSYS_PRINT_SPDLOGIMPL(true, true, "Blocked reinitialization of spdlog");
    }
    initialized = true;

    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] Initializing spdlog..." << std::endl;

    // ----------------------------------------------------------------------------------
    spdlog::flush_on(spdlog::level::trace); // TODO: *not do this* Flush on every log (also terribly inneficient)
    spdlog::set_level(spdlog::level::trace);

    constexpr size_t megabyte = 1048576;


    constexpr const char* logger_color_info = "\033[1;34m";
    constexpr const char* logger_color_warn = "\033[01;33m";
    constexpr const char* logger_color_fatal = "\033[01;31m";
    constexpr const char* logger_color_source = "\033[01;32m";

    // ----------------------------------------------------------------------------------
    // May need to read some config thing to set verbosity level?
    // ----------------------------------------------------------------------------------

    // Sinks do not copy formatter, must create one for each sink (TODO: VERIFY)
    auto create_custom_formatter = [](const std::string& pattern) {
        auto base_formatter = std::make_unique<spdlog::pattern_formatter>();
        base_formatter->add_flag<standard_name_formatter>('N');
        base_formatter->add_flag<process_identifier_formatter>('Q');
        base_formatter->add_flag<thread_identifier_formatter>('U');
        base_formatter->add_flag<function_name_formatter>('J');
        base_formatter->set_pattern(pattern);
        return base_formatter;
    };

    auto set_console_colors = [](auto& sink) {
        sink->set_color(spdlog::level::trace,    logger_color_source); // Bold Green
        sink->set_color(spdlog::level::debug,    logger_color_source); // Bold Green
        sink->set_color(spdlog::level::info,     logger_color_info);   // Bold Blue
        sink->set_color(spdlog::level::warn,     logger_color_warn);   // Bold Yellow
        sink->set_color(spdlog::level::err,      logger_color_fatal);  // Bold Red
        sink->set_color(spdlog::level::critical, logger_color_fatal);  // Bold Red
    };

    auto assign_logger = [](const std::string& name, auto sink) {
        auto logger = std::make_shared<spdlog::logger>(name, sink);
        logger->set_level(spdlog::level::trace);
        return logger;
    };
    // ----------------------------------------------------------------------------------

    // Console sinks with different formatting patterns
    // N = standardized logger name, Q = process identifier, U = thread identifier, J = function identifier

    // In order, each sink will output:
    // [name][process][thread] message
    // [name][process][thread][function] message
    // [name][process] message
    // [name][process][function] message

    auto console_sink_npt = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto console_sink_nptf = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto console_sink_np = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    auto console_sink_npf = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

    // std::string placeholder_file = "logs/rocprof.log";

    // auto file_sink_npt = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
    //     placeholder_file, megabyte * 10, 5);
    // auto file_sink_nptf = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
    //     placeholder_file, megabyte * 10, 5);
    // auto file_sink_np = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
    //     placeholder_file, megabyte * 10, 5);
    // auto file_sink_npf = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
    //     placeholder_file, megabyte * 10, 5);


    console_sink_npt->set_formatter(create_custom_formatter("%^[%N][%Q][%U] %v%$"));
    console_sink_nptf->set_formatter(create_custom_formatter("%^[%N][%Q][%U][%J] %v%$"));
    console_sink_np->set_formatter(create_custom_formatter("%^[%N][%Q] %v%$"));
    console_sink_npf->set_formatter(create_custom_formatter("%^[%N][%Q][%J] %v%$"));

    // file_sink_npt->set_formatter(create_custom_formatter("[%n][%Q][%U] %v"));
    // file_sink_nptf->set_formatter(create_custom_formatter("[%n][%Q][%U][%J] %v"));
    // file_sink_np->set_formatter(create_custom_formatter("[%n][%Q] %v"));
    // file_sink_npf->set_formatter(create_custom_formatter("[%n][%Q][%J] %v"));

    set_console_colors(console_sink_npt);
    set_console_colors(console_sink_nptf);
    set_console_colors(console_sink_np);
    set_console_colors(console_sink_npf);

    // ----------------------------------------------------------------------------------
    // Assign loggers to pointers

    rocprofsys_npt_logger = assign_logger("rocprof-sys-npt", console_sink_npt);
    rocprofsys_nptf_logger = assign_logger("rocprof-sys-nptf", console_sink_nptf);
    rocprofsys_np_logger = assign_logger("rocprof-sys-np", console_sink_np);
    rocprofsys_npf_logger = assign_logger("rocprof-sys-npf", console_sink_npf);

    std::cerr << "[ROCPROFILER-SYSTEMS-CHECK] Initializing spdlog completed" << std::endl;

}

} // namespace debug
} // namespace rocprofsys
