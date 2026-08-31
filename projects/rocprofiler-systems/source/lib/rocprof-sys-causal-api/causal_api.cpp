// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#if !defined(ROCPROFSYS_CAUSAL_API_SOURCE)
#    define ROCPROFSYS_CAUSAL_API_SOURCE 1
#endif

#include "rocprofiler-systems/causal_api.h"

#include "rocprofiler-systems/annotation.h"

#include <cstddef>

namespace
{
rocprofsys_causal_callbacks_t _callbacks = { .begin              = nullptr,
                                             .end                = nullptr,
                                             .progress           = nullptr,
                                             .annotated_progress = nullptr };

// -1 when no callback is registered (e.g. librocprof-sys-dl is not preloaded)
inline int
invoke(rocprofsys_causal_region_func_t _func, const char* _name)
{
    return _func ? (*_func)(_name) : -1;
}

inline int
invoke(rocprofsys_causal_annotated_func_t _func, const char* _name,
       rocprofsys_annotation_t* _annotations, size_t _annotation_count)
{
    return _func ? (*_func)(_name, _annotations, _annotation_count) : -1;
}
}  // namespace

extern "C"
{
    int rocprofsys_causal_begin(const char* name)
    {
        return invoke(_callbacks.begin, name);
    }

    int rocprofsys_causal_end(const char* name) { return invoke(_callbacks.end, name); }

    int rocprofsys_causal_progress(const char* name)
    {
        return invoke(_callbacks.progress, name);
    }

    int rocprofsys_causal_annotated_progress(const char*              name,
                                             rocprofsys_annotation_t* annotations,
                                             size_t                   annotation_count)
    {
        return invoke(_callbacks.annotated_progress, name, annotations, annotation_count);
    }

    void rocprofsys_causal_register_callbacks(rocprofsys_causal_callbacks_t callbacks)
    {
        _callbacks = callbacks;
    }
}
