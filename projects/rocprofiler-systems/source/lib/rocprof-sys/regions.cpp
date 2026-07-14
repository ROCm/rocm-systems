// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "api.hpp"
#include "core/categories.hpp"
#include "core/config.hpp"
#include "library/components/category_region.hpp"
#include "library/tracing.hpp"
#include <cstdint>

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
// The annotation type enum (ROCPROFSYS_ANNOTATION_TYPE) only ever resolves to the
// fixed scalar set registered via ROCPROFSYS_DEFINE_ANNOTATION_TYPE in
// annotation.hpp, so (unlike category_region's gotcha-arg serialization, which
// must handle arbitrary caller types) these can be dispatched explicitly instead
// of relying on a generic fmt::is_formattable/fmt::streamed fallback.
template <typename Tp>
[[nodiscard]] std::string
annotation_arg_type_name()
{
    using value_type = std::decay_t<Tp>;
    if constexpr(concepts::is_string_type<value_type>::value)
        return "string";
    else
        return utility::demangle<value_type>();
}

template <typename Tp>
[[nodiscard]] std::string
annotation_arg_value_string(Tp&& _val)
{
    using value_type = std::decay_t<Tp>;
    if constexpr(concepts::is_string_type<value_type>::value)
        return std::string{ std::string_view{ std::forward<Tp>(_val) } };
    else if constexpr(std::is_pointer<value_type>::value)
        return fmt::format("{:#x}", reinterpret_cast<std::uintptr_t>(_val));
    else if constexpr(std::is_integral<value_type>::value)
        return fmt::format_int{ _val }.str();
    else
        return fmt::format("{}", std::forward<Tp>(_val));
}

template <size_t Idx, size_t... Tail>
void
append_annotation_arg(function_args_t& _args, const rocprofsys_annotation_t& _annotation,
                      std::uint32_t _arg_number, std::index_sequence<Idx, Tail...>)
{
    static_assert(Idx > ROCPROFSYS_VALUE_NONE && Idx < ROCPROFSYS_VALUE_LAST,
                  "Error! index sequence should only contain values which are greater "
                  "than ROCPROFSYS_VALUE_NONE and less than ROCPROFSYS_VALUE_LAST");

    if(_annotation.type == Idx)
    {
        using type = tracing::annotation_value_type_t<Idx>;
        if constexpr(std::is_pointer<type>::value)
        {
            const auto _value = reinterpret_cast<type>(_annotation.value);
            _args.push_back({ _arg_number, annotation_arg_type_name<type>(),
                              _annotation.name, annotation_arg_value_string(_value) });
        }
        else
        {
            const auto* _value = reinterpret_cast<type*>(_annotation.value);
            _args.push_back({ _arg_number, annotation_arg_type_name<type>(),
                              _annotation.name, annotation_arg_value_string(*_value) });
        }
    }
    else if constexpr(sizeof...(Tail) > 0)
    {
        append_annotation_arg(_args, _annotation, _arg_number,
                              std::index_sequence<Tail...>{});
    }
}

// Converts a rocprofsys_annotation_t array (as passed to
// rocprofsys_push_category_region) into the trace-cache args wire format, so
// annotations attached to a region survive cache-replay (Perfetto + rocpd)
// instead of only appearing on the live-instrumentation path.
[[nodiscard]] function_args_t
annotations_to_function_args(const rocprofsys_annotation_t* _annotations, size_t _count)
{
    function_args_t _args;
    if(_annotations == nullptr || _count == 0) return _args;

    _args.reserve(_count);
    for(size_t i = 0; i < _count; ++i)
    {
        const auto& _annotation = _annotations[i];
        if(_annotation.name == nullptr || _annotation.type <= ROCPROFSYS_VALUE_NONE ||
           _annotation.type >= ROCPROFSYS_VALUE_LAST || _annotation.value == nullptr)
            continue;

        append_annotation_arg(
            _args, _annotation, static_cast<std::uint32_t>(_args.size()),
            utility::make_index_sequence_range<1, ROCPROFSYS_VALUE_LAST>{});
    }
    return _args;
}

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

        // Cache the annotations into the trace-cache args wire format so
        // cache-replay handlers (perfetto + rocpd) re-emit them. The lambda
        // above only reaches the live-instrumentation output, which the final
        // trace does not use when cache replay is active.
        if(_annotations != nullptr && _annotation_count > 0)
        {
            component::category_region<category_type>::append_cache_args(
                name, get_args_string(
                          annotations_to_function_args(_annotations, _annotation_count)));
        }
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
