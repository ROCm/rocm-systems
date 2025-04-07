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

#include "api.hpp"
#include "core/categories.hpp"
#include "core/config.hpp"
#include "library/components/category_region.hpp"
#include "library/tracing.hpp"

#if defined(__GNUC__) && (__GNUC__ == 7)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

namespace rocprofsys
{
namespace impl
{
namespace
{
template <size_t Idx, size_t... Tail>
void
invoke_category_region_start(rocprofsys_category_t _category, const char* name,
                             rocprofsys_annotation_t* _annotations,
                             size_t _annotation_count, std::index_sequence<Idx, Tail...>)
{
    static_assert(Idx > ROCPROFSYS_CATEGORY_NONE && Idx < ROCPROFSYS_CATEGORY_LAST,
                  "Error! index sequence should only contain values which are greater "
                  "than ROCPROFSYS_CATEGORY_NONE and less than ROCPROFSYS_CATEGORY_LAST");

    if(_category == Idx)
    {
        using category_type = category_type_id_t<Idx>;

        // skip if category is disabled
        if(!trait::runtime_enabled<category_type>::get()) return;

        component::category_region<category_type>::start(
            name, [&](::perfetto::EventContext ctx) {
                if(_annotations && config::get_perfetto_annotations())
                {
                    for(size_t i = 0; i < _annotation_count; ++i)
                        tracing::add_perfetto_annotation(ctx, _annotations[i]);
                }
            });
    }
    else
    {
        constexpr size_t remaining = sizeof...(Tail);
        if constexpr(remaining > 0)
            invoke_category_region_start(_category, name, _annotations, _annotation_count,
                                         std::index_sequence<Tail...>{});
    }
}

template <size_t Idx, size_t... Tail>
void
invoke_category_region_stop(rocprofsys_category_t _category, const char* name,
                            rocprofsys_annotation_t* _annotations,
                            size_t _annotation_count, std::index_sequence<Idx, Tail...>)
{
    static_assert(Idx > ROCPROFSYS_CATEGORY_NONE && Idx < ROCPROFSYS_CATEGORY_LAST,
                  "Error! index sequence should only contain values which are greater "
                  "than ROCPROFSYS_CATEGORY_NONE and less than ROCPROFSYS_CATEGORY_LAST");

    if(_category == Idx)
    {
        using category_type = category_type_id_t<Idx>;

        // skip if category is disabled
        if(!trait::runtime_enabled<category_type>::get()) return;

        component::category_region<category_type>::stop(
            name, [&](::perfetto::EventContext ctx) {
                if(_annotations && config::get_perfetto_annotations())
                {
                    for(size_t i = 0; i < _annotation_count; ++i)
                        tracing::add_perfetto_annotation(ctx, _annotations[i]);
                }
            });
    }
    else
    {
        constexpr size_t remaining = sizeof...(Tail);
        if constexpr(remaining > 0)
            invoke_category_region_stop(_category, name, _annotations, _annotation_count,
                                        std::index_sequence<Tail...>{});
    }
}
}  // namespace
}  // namespace impl
}  // namespace rocprofsys

//======================================================================================//

extern "C" void
rocprofsys_push_trace_hidden(const char* name)
{
    rocprofsys::component::category_region<rocprofsys::category::host>::start(name);
}

extern "C" void
rocprofsys_pop_trace_hidden(const char* name)
{
    rocprofsys::component::category_region<rocprofsys::category::host>::stop(name);
}

//======================================================================================//
///
///
///
//======================================================================================//

extern "C" void
rocprofsys_push_region_hidden(const char* name)
{
    rocprofsys::component::category_region<rocprofsys::category::user>::start(name);
}

extern "C" void
rocprofsys_pop_region_hidden(const char* name)
{
    rocprofsys::component::category_region<rocprofsys::category::user>::stop(name);
}

//======================================================================================//
///
///
///
//======================================================================================//

extern "C" void
rocprofsys_push_category_region_hidden(rocprofsys_category_t _category, const char* name,
                                       rocprofsys_annotation_t* _annotations,
                                       size_t                   _annotation_count)
{
    rocprofsys::impl::invoke_category_region_start(
        _category, name, _annotations, _annotation_count,
        rocprofsys::utility::make_index_sequence_range<1, ROCPROFSYS_CATEGORY_LAST>{});
}

extern "C" void
rocprofsys_pop_category_region_hidden(rocprofsys_category_t _category, const char* name,
                                      rocprofsys_annotation_t* _annotations,
                                      size_t                   _annotation_count)
{
    rocprofsys::impl::invoke_category_region_stop(
        _category, name, _annotations, _annotation_count,
        rocprofsys::utility::make_index_sequence_range<1, ROCPROFSYS_CATEGORY_LAST>{});
}

#if defined(__GNUC__) && (__GNUC__ == 7)
#    pragma GCC diagnostic pop
#endif
