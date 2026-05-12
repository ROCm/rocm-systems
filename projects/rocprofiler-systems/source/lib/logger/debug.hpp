// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger.hpp"

#define LOG_CRITICAL(...)                                                                \
    do                                                                                   \
    {                                                                                    \
        rocprofsys::logger_t::instance().log(                                            \
            spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::critical, \
            __VA_ARGS__);                                                                \
        rocprofsys::logger_t::instance().flush();                                        \
    } while(0)

#define LOG_ERROR(...)                                                                   \
    rocprofsys::logger_t::instance().log(                                                \
        spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::err,          \
        __VA_ARGS__)

#define LOG_WARNING(...)                                                                 \
    rocprofsys::logger_t::instance().log(                                                \
        spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::warn,         \
        __VA_ARGS__)

#define LOG_INFO(...)                                                                    \
    rocprofsys::logger_t::instance().log(                                                \
        spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::info,         \
        __VA_ARGS__)

#define LOG_DEBUG(...)                                                                   \
    rocprofsys::logger_t::instance().log(                                                \
        spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::debug,        \
        __VA_ARGS__)

#define LOG_TRACE(...)                                                                   \
    rocprofsys::logger_t::instance().log(                                                \
        spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::trace,        \
        __VA_ARGS__)

/// Log a std::exception (or any subclass) with its rendered stacktrace at the
/// given level. Uses `format_exception`, which honors an embedded trace on
/// `rocprofsys::exception` and falls back to a fresh capture otherwise.
///
/// The TU using this macro must `#include "common/diagnostic/format_exception.hpp"`
/// itself; we do not include it here so the logger header stays free of the
/// diagnostic library's transitive dependencies.
#define LOG_EXCEPTION(level, e)                                                          \
    LOG_##level("{}", ::rocprofsys::common::diagnostic::format_exception(e))
