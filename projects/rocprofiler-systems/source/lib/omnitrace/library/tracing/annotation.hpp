// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "core/common.hpp"
#include "core/concepts.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "omnitrace/categories.h"  // in omnitrace-user

#include <timemory/mpl/concepts.hpp>

#include <type_traits>

namespace omnitrace
{
namespace tracing
{
using perfetto_event_context_t = ::perfetto::EventContext;

template <size_t Idx>
struct annotation_value_type;

template <size_t Idx>
using annotation_value_type_t = typename annotation_value_type<Idx>::type;

#define OMNITRACE_DEFINE_ANNOTATION_TYPE(ENUM, TYPE)                                     \
    template <>                                                                          \
    struct annotation_value_type<ENUM>                                                   \
    {                                                                                    \
        using type = TYPE;                                                               \
    };

OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_CSTR, const char*)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_SIZE_T, size_t)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_INT16, int16_t)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_INT32, int32_t)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_INT64, int64_t)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_UINT16, uint16_t)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_UINT32, uint32_t)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_UINT64, uint64_t)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_FLOAT32, float)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_FLOAT64, double)
OMNITRACE_DEFINE_ANNOTATION_TYPE(OMNITRACE_VALUE_VOID_P, void*)

#undef OMNITRACE_DEFINE_ANNOTATION_TYPE

template <typename Np, typename Tp>
auto
add_perfetto_annotation(
    perfetto_event_context_t& ctx, Np&& _name, Tp&& _val, int64_t _idx = -1,
    std::enable_if_t<
        !std::is_same<std::remove_pointer_t<concepts::unqualified_type_t<Np>>,
                      omnitrace_annotation_t>::value,
        int> = 0)
{
    using named_type = std::remove_reference_t<std::remove_cv_t<std::decay_t<Np>>>;
    using value_type = std::remove_reference_t<std::remove_cv_t<std::decay_t<Tp>>>;

    static_assert(concepts::is_string_type<named_type>::value,
                  "Error! name is not a string type");

    auto _get_dbg = [&]() {
        auto* _dbg = ctx.event()->add_debug_annotations();
        if(_idx >= 0)
        {
            auto _arg_name = JOIN("", "arg", _idx, "-", std::forward<Np>(_name));
            _dbg->set_name(_arg_name);
        }
        else
        {
            _dbg->set_name(std::string_view{ std::forward<Np>(_name) }.data());
        }
        return _dbg;
    };

    if constexpr(std::is_same<value_type, std::string_view>::value)
    {
        _get_dbg()->set_string_value(_val.data());
    }
    else if constexpr(concepts::is_string_type<value_type>::value)
    {
        _get_dbg()->set_string_value(std::forward<Tp>(_val));
    }
    else if constexpr(std::is_same<value_type, bool>::value)
    {
        _get_dbg()->set_bool_value(std::forward<Tp>(_val));
    }
    else if constexpr(std::is_enum<value_type>::value)
    {
        _get_dbg()->set_int_value(static_cast<int64_t>(std::forward<Tp>(_val)));
    }
    else if constexpr(std::is_floating_point<value_type>::value)
    {
        _get_dbg()->set_double_value(std::forward<Tp>(_val));
    }
    else if constexpr(std::is_integral<value_type>::value)
    {
        if constexpr(std::is_unsigned<value_type>::value)
        {
            _get_dbg()->set_uint_value(std::forward<Tp>(_val));
        }
        else
        {
            _get_dbg()->set_int_value(std::forward<Tp>(_val));
        }
    }
    else if constexpr(std::is_pointer<value_type>::value)
    {
        _get_dbg()->set_pointer_value(reinterpret_cast<uint64_t>(std::forward<Tp>(_val)));
    }
    else if constexpr(concepts::can_stringify<value_type>::value)
    {
        _get_dbg()->set_string_value(JOIN("", std::forward<Tp>(_val)));
    }
    else
    {
        static_assert(std::is_empty<value_type>::value, "Error! unsupported data type");
    }
}

template <size_t Idx, size_t... Tail>
void
add_perfetto_annotation(perfetto_event_context_t&     ctx,
                        const omnitrace_annotation_t& _annotation,
                        std::index_sequence<Idx, Tail...>)
{
    static_assert(Idx > OMNITRACE_VALUE_NONE && Idx < OMNITRACE_VALUE_LAST,
                  "Error! index sequence should only contain values which are greater "
                  "than OMNITRACE_VALUE_NONE and less than OMNITRACE_VALUE_LAST");

    // in some situations the user might want to short circuit by setting
    // the name to a null pointer, type to none, or value to a null pointer
    if(_annotation.name == nullptr || _annotation.type == 0 ||
       _annotation.value == nullptr)
        return;

    if(_annotation.type == Idx)
    {
        using type = annotation_value_type_t<Idx>;
        // if the type is a pointer, pass the pointer. otherwise,
        // cast to pointer of that type and dereference it
        if constexpr(std::is_pointer<type>::value)
        {
            auto _value = reinterpret_cast<type>(_annotation.value);
            add_perfetto_annotation(ctx, _annotation.name, _value);
        }
        else
        {
            auto* _value = reinterpret_cast<type*>(_annotation.value);
            add_perfetto_annotation(ctx, _annotation.name, *_value);
        }
    }
    else
    {
        // the first "iteration": check whether annotation type has valid range
        if constexpr(Idx == OMNITRACE_VALUE_NONE + 1)
        {
            if(!(_annotation.type > OMNITRACE_VALUE_NONE &&
                 _annotation.type < OMNITRACE_VALUE_LAST))
            {
                OMNITRACE_FAIL_F("Error! annotation '%s' has an invalid type designation "
                                 "%lu which is outside of acceptable range [%i, %i]\n",
                                 _annotation.name, _annotation.type,
                                 OMNITRACE_VALUE_NONE + 1, OMNITRACE_VALUE_LAST - 1);
            }
        }

        if constexpr(sizeof...(Tail) > 0)
        {
            add_perfetto_annotation(ctx, _annotation, std::index_sequence<Tail...>{});
        }
        else
        {
            throw ::omnitrace::exception<std::runtime_error>(
                "invalid annotation value type");
        }
    }
}

void
add_perfetto_annotation(perfetto_event_context_t&     ctx,
                        const omnitrace_annotation_t& _annotation);
}  // namespace tracing
}  // namespace omnitrace
