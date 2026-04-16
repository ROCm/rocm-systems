// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "logger.hpp"

#if ROCPDSNA_ENABLE_LOGGING > 0

#    define LOG_CRITICAL(...)                                                            \
        do                                                                               \
        {                                                                                \
            rocpdsna::logger_t::instance().log(                                          \
                spdlog::source_loc{ __FILE__, __LINE__, __func__ },                      \
                spdlog::level::critical,                                                 \
                __VA_ARGS__);                                                            \
            rocpdsna::logger_t::instance().flush();                                      \
        } while(0)

#    define LOG_ERROR(...)                                                               \
        rocpdsna::logger_t::instance().log(                                              \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::err,                                                          \
            __VA_ARGS__)

#    define LOG_WARNING(...)                                                             \
        rocpdsna::logger_t::instance().log(                                              \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::warn,                                                         \
            __VA_ARGS__)

#    define LOG_INFO(...)                                                                \
        rocpdsna::logger_t::instance().log(                                              \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::info,                                                         \
            __VA_ARGS__)

#    define LOG_DEBUG(...)                                                               \
        rocpdsna::logger_t::instance().log(                                              \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ },                          \
            spdlog::level::debug,                                                        \
            __VA_ARGS__)

#    define LOG_TRACE(...)                                                               \
        rocpdsna::logger_t::instance().log(                                              \
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
