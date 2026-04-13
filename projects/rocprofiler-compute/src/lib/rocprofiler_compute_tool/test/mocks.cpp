// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "mocks.h"

const char* MockInputParameters::get_output_path()
{
    return m_output_path.c_str();
}

const char* MockInputParameters::get_requested_counters()
{
    return m_requested_counters.c_str();
}

const char* MockInputParameters::get_iteration_multiplexing_mode()
{
    return m_iteration_multiplexing_mode.c_str();
}

const char* MockInputParameters::get_kernel_filter_include_regex()
{
    return m_kernel_filter_include_regex.c_str();
}

const char* MockInputParameters::get_kernel_filter_range()
{
    return m_kernel_filter_range.c_str();
}

void MockInputParameters::set_output_path(const std::string& output_path)
{
    m_output_path = output_path;
}

void MockInputParameters::set_requested_counters(const std::string& counters)
{
    m_requested_counters = counters;
}

void MockInputParameters::set_iteration_multiplexing_mode(const std::string& mode)
{
    m_iteration_multiplexing_mode = mode;
}

void MockInputParameters::set_kernel_filter_include_regex(const std::string& regex)
{
    m_kernel_filter_include_regex = regex;
}

void MockInputParameters::set_kernel_filter_range(const std::string& range)
{
    m_kernel_filter_range = range;
}
