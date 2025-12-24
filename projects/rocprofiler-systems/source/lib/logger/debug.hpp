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
