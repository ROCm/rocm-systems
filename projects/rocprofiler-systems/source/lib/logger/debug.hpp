// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "logger.hpp"

#define LOG_CRITICAL(...)                                                                \
    do                                                                                   \
    {                                                                                    \
        if(!rocprofsys::logger_t::is_suppressed())                                       \
        {                                                                                \
            rocprofsys::logger_t::instance().log(                                        \
                spdlog::source_loc{ __FILE__, __LINE__, __func__ },                      \
                spdlog::level::critical, __VA_ARGS__);                                   \
            rocprofsys::logger_t::instance().flush();                                    \
        }                                                                                \
    } while(0)

#define LOG_ERROR(...)                                                                   \
    do                                                                                   \
    {                                                                                    \
        if(!rocprofsys::logger_t::is_suppressed())                                       \
            rocprofsys::logger_t::instance().log(                                        \
                spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::err,  \
                __VA_ARGS__);                                                            \
    } while(0)

#define LOG_WARNING(...)                                                                 \
    do                                                                                   \
    {                                                                                    \
        if(!rocprofsys::logger_t::is_suppressed())                                       \
            rocprofsys::logger_t::instance().log(                                        \
                spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::warn, \
                __VA_ARGS__);                                                            \
    } while(0)

#define LOG_INFO(...)                                                                    \
    do                                                                                   \
    {                                                                                    \
        if(!rocprofsys::logger_t::is_suppressed())                                       \
            rocprofsys::logger_t::instance().log(                                        \
                spdlog::source_loc{ __FILE__, __LINE__, __func__ }, spdlog::level::info, \
                __VA_ARGS__);                                                            \
    } while(0)

#define LOG_DEBUG(...)                                                                   \
    do                                                                                   \
    {                                                                                    \
        if(!rocprofsys::logger_t::is_suppressed())                                       \
            rocprofsys::logger_t::instance().log(                                        \
                spdlog::source_loc{ __FILE__, __LINE__, __func__ },                      \
                spdlog::level::debug, __VA_ARGS__);                                      \
    } while(0)

#define LOG_TRACE(...)                                                                   \
    do                                                                                   \
    {                                                                                    \
        if(!rocprofsys::logger_t::is_suppressed())                                       \
            rocprofsys::logger_t::instance().log(                                        \
                spdlog::source_loc{ __FILE__, __LINE__, __func__ },                      \
                spdlog::level::trace, __VA_ARGS__);                                      \
    } while(0)
