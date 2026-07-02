// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

extern "C" void
rocprofsys_flush_pending_region_cache_hidden()
{
    rocprofsys::utility::category_region<>::instance().flush_pending_cached_entries();
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
rocprofsys_push_trace_with_args_hidden(const char* name, const char* serialized_args)
{
    rocprofsys::component::category_region<rocprofsys::category::host>::start_with_args(
        name, serialized_args ? serialized_args : "");
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
