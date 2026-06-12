// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "api.hpp"

#include "logger/debug.hpp"
#include <exception>
#include <stdexcept>

extern "C" void
rocprofsys_push_trace(const char* _name)
{
    rocprofsys_push_trace_hidden(_name);
}

extern "C" void
rocprofsys_push_trace_with_args(const char* _name, const char* _serialized_args)
{
    rocprofsys_push_trace_with_args_hidden(_name, _serialized_args);
}

extern "C" void
rocprofsys_pop_trace(const char* _name)
{
    rocprofsys_pop_trace_hidden(_name);
}

extern "C" int
rocprofsys_push_region(const char* _name)
{
    try
    {
        rocprofsys_push_region_hidden(_name);
    } catch(std::exception& _e)
    {
        LOG_WARNING("Exception caught: {}", _e.what());
        return -1;
    }
    return 0;
}

extern "C" int
rocprofsys_pop_region(const char* _name)
{
    try
    {
        rocprofsys_pop_region_hidden(_name);
    } catch(std::exception& _e)
    {
        LOG_WARNING("Exception caught: {}", _e.what());
        return -1;
    }
    return 0;
}

extern "C" int
rocprofsys_push_category_region(rocprofsys_category_t _category, const char* _name,
                                rocprofsys_annotation_t* _annotations,
                                size_t                   _annotation_count)
{
    try
    {
        rocprofsys_push_category_region_hidden(_category, _name, _annotations,
                                               _annotation_count);
    } catch(std::exception& _e)
    {
        LOG_WARNING("Exception caught: {}", _e.what());
        return -1;
    }
    return 0;
}

extern "C" int
rocprofsys_pop_category_region(rocprofsys_category_t _category, const char* _name,
                               rocprofsys_annotation_t* _annotations,
                               size_t                   _annotation_count)
{
    try
    {
        rocprofsys_pop_category_region_hidden(_category, _name, _annotations,
                                              _annotation_count);
    } catch(std::exception& _e)
    {
        LOG_WARNING("Exception caught: {}", _e.what());
        return -1;
    }
    return 0;
}

extern "C" void
rocprofsys_progress(const char* _name)
{
    rocprofsys_progress_hidden(_name);
}

extern "C" void
rocprofsys_annotated_progress(const char* _name, rocprofsys_annotation_t* _annotations,
                              size_t _annotation_count)
{
    rocprofsys_annotated_progress_hidden(_name, _annotations, _annotation_count);
}

extern "C" void
rocprofsys_init_library(void)
{
    rocprofsys_init_library_hidden();
}

extern "C" void
rocprofsys_init_tooling(void)
{
    rocprofsys_init_tooling_hidden();
}

extern "C" void
rocprofsys_init(const char* _mode, bool _rewrite, const char* _arg0)
{
    rocprofsys_init_hidden(_mode, _rewrite, _arg0);
}

extern "C" void
rocprofsys_finalize(void)
{
    rocprofsys_finalize_hidden();
}

extern "C" void
rocprofsys_reset_preload(void)
{
    rocprofsys_reset_preload_hidden();
}

extern "C" void
rocprofsys_set_env(const char* env_name, const char* env_val)
{
    rocprofsys_set_env_hidden(env_name, env_val);
}

extern "C" void
rocprofsys_set_mpi(bool use, bool attached)
{
    rocprofsys_set_mpi_hidden(use, attached);
}

extern "C" void
rocprofsys_register_source(const char* file, const char* func, size_t line,
                           size_t address, const char* source)
{
    rocprofsys_register_source_hidden(file, func, line, address, source);
}

extern "C" void
rocprofsys_register_coverage(const char* file, const char* func, size_t address)
{
    rocprofsys_register_coverage_hidden(file, func, address);
}
