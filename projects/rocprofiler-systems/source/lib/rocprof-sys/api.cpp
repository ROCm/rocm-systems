// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "api.hpp"

#include "common/diagnostic/format_exception.hpp"
#include "logger/debug.hpp"
#include <exception>
#include <stdexcept>

namespace
{
// Log helper for void-returning C entry points: we have no error-code channel
// back to the caller, so we just log and swallow rather than letting the throw
// escape the C ABI boundary (which is undefined behavior).
template <typename Fn>
void
guarded_void(const char* fn_name, Fn&& fn)
{
    try
    {
        fn();
    } catch(const std::exception& e)
    {
        LOG_ERROR("{}: {}\n{}", fn_name, e.what(),
                  ::rocprofsys::common::diagnostic::format_exception(e));
    } catch(...)
    {
        LOG_ERROR("{}: unknown exception", fn_name);
    }
}
}  // namespace

extern "C" void
rocprofsys_push_trace(const char* _name)
{
    guarded_void("rocprofsys_push_trace", [&]() { rocprofsys_push_trace_hidden(_name); });
}

extern "C" void
rocprofsys_pop_trace(const char* _name)
{
    guarded_void("rocprofsys_pop_trace", [&]() { rocprofsys_pop_trace_hidden(_name); });
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
    guarded_void("rocprofsys_progress", [&]() { rocprofsys_progress_hidden(_name); });
}

extern "C" void
rocprofsys_annotated_progress(const char* _name, rocprofsys_annotation_t* _annotations,
                              size_t _annotation_count)
{
    guarded_void("rocprofsys_annotated_progress", [&]() {
        rocprofsys_annotated_progress_hidden(_name, _annotations, _annotation_count);
    });
}

extern "C" void
rocprofsys_init_library(void)
{
    guarded_void("rocprofsys_init_library", [&]() { rocprofsys_init_library_hidden(); });
}

extern "C" void
rocprofsys_init_tooling(void)
{
    guarded_void("rocprofsys_init_tooling", [&]() { rocprofsys_init_tooling_hidden(); });
}

extern "C" void
rocprofsys_init(const char* _mode, bool _rewrite, const char* _arg0)
{
    guarded_void("rocprofsys_init",
                 [&]() { rocprofsys_init_hidden(_mode, _rewrite, _arg0); });
}

extern "C" void
rocprofsys_finalize(void)
{
    guarded_void("rocprofsys_finalize", [&]() { rocprofsys_finalize_hidden(); });
}

extern "C" void
rocprofsys_reset_preload(void)
{
    guarded_void("rocprofsys_reset_preload",
                 [&]() { rocprofsys_reset_preload_hidden(); });
}

extern "C" void
rocprofsys_set_env(const char* env_name, const char* env_val)
{
    guarded_void("rocprofsys_set_env",
                 [&]() { rocprofsys_set_env_hidden(env_name, env_val); });
}

extern "C" void
rocprofsys_set_mpi(bool use, bool attached)
{
    guarded_void("rocprofsys_set_mpi",
                 [&]() { rocprofsys_set_mpi_hidden(use, attached); });
}

extern "C" void
rocprofsys_register_source(const char* file, const char* func, size_t line,
                           size_t address, const char* source)
{
    guarded_void("rocprofsys_register_source", [&]() {
        rocprofsys_register_source_hidden(file, func, line, address, source);
    });
}

extern "C" void
rocprofsys_register_coverage(const char* file, const char* func, size_t address)
{
    guarded_void("rocprofsys_register_coverage",
                 [&]() { rocprofsys_register_coverage_hidden(file, func, address); });
}
