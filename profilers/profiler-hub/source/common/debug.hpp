// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger.hpp"

#if PROFILER_HUB_ENABLE_LOGGING > 0

#    define LOG_CRITICAL(...)                                                            \
        do                                                                               \
        {                                                                                \
            profiler_hub::logger_t::instance().log(                                      \
                spdlog::source_loc{ __FILE__, __LINE__, __func__ },                      \
                spdlog::level::critical,                                                 \
                __VA_ARGS__);                                                            \
            profiler_hub::logger_t::instance().flush();                                  \
        } while(0)

#    define LOG_ERROR(...)                                                               \
        profiler_hub::logger_t::instance().log(                                          \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::err,                                                          \
            __VA_ARGS__)

#    define LOG_WARNING(...)                                                             \
        profiler_hub::logger_t::instance().log(                                          \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::warn,                                                         \
            __VA_ARGS__)

#    define LOG_INFO(...)                                                                \
        profiler_hub::logger_t::instance().log(                                          \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::info,                                                         \
            __VA_ARGS__)

#    define LOG_DEBUG(...)                                                               \
        profiler_hub::logger_t::instance().log(                                          \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::debug,                                                        \
            __VA_ARGS__)

#    define LOG_TRACE(...)                                                               \
        profiler_hub::logger_t::instance().log(                                          \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::trace,                                                        \
            __VA_ARGS__)

#else

#    define LOG_CRITICAL(...)
#    define LOG_ERROR(...)
#    define LOG_WARNING(...)
#    define LOG_INFO(...)
#    define LOG_DEBUG(...)
#    define LOG_TRACE(...)

#endif
