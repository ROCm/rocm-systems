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

#pragma once

#include "core/common.hpp"
#include "core/concepts.hpp"
#include "core/debug.hpp"
#include "core/defines.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/utility.hpp"
#include "rocprofiler-systems/categories.h"  // in rocprof-sys-user

#include <timemory/mpl/concepts.hpp>
#include <timemory/operations/types/get.hpp>

#include <type_traits>

namespace rocprofsys
{
namespace tracing
{
using perfetto_event_context_t = ::perfetto::EventContext;

template <size_t Idx>
struct annotation_value_type;

template <size_t Idx>
using annotation_value_type_t = typename annotation_value_type<Idx>::type;

#define ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ENUM, TYPE)                                    \
    template <>                                                                          \
    struct annotation_value_type<ENUM>                                                   \
    {                                                                                    \
        using type = TYPE;                                                               \
    };

ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_CSTR, const char*)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_SIZE_T, size_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_INT16, int16_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_INT32, int32_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_INT64, int64_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_UINT16, uint16_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_UINT32, uint32_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_UINT64, uint64_t)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_FLOAT32, float)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_FLOAT64, double)
ROCPROFSYS_DEFINE_ANNOTATION_TYPE(ROCPROFSYS_VALUE_VOID_P, void*)

#undef ROCPROFSYS_DEFINE_ANNOTATION_TYPE

template <typename Np, typename Tp>
auto
add_perfetto_annotation(
    perfetto_event_context_t& ctx, Np&& _name, Tp&& _val, int64_t _idx = -1,
    std::enable_if_t<
        !std::is_same<std::remove_pointer_t<concepts::unqualified_type_t<Np>>,
                      rocprofsys_annotation_t>::value,
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
        _get_dbg()->set_bool_value(_val);
    }
    else if constexpr(std::is_enum<value_type>::value)
    {
        _get_dbg()->set_int_value(static_cast<int64_t>(_val));
    }
    else if constexpr(std::is_floating_point<value_type>::value)
    {
        _get_dbg()->set_double_value(static_cast<double>(_val));
    }
    else if constexpr(std::is_integral<value_type>::value)
    {
        if constexpr(std::is_unsigned<value_type>::value)
        {
            _get_dbg()->set_uint_value(_val);
        }
        else
        {
            _get_dbg()->set_int_value(_val);
        }
    }
    else if constexpr(std::is_pointer<value_type>::value)
    {
        _get_dbg()->set_pointer_value(reinterpret_cast<uint64_t>(_val));
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
add_perfetto_annotation(perfetto_event_context_t&      ctx,
                        const rocprofsys_annotation_t& _annotation,
                        std::index_sequence<Idx, Tail...>)
{
    static_assert(Idx > ROCPROFSYS_VALUE_NONE && Idx < ROCPROFSYS_VALUE_LAST,
                  "Error! index sequence should only contain values which are greater "
                  "than ROCPROFSYS_VALUE_NONE and less than ROCPROFSYS_VALUE_LAST");

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
        if constexpr(Idx == ROCPROFSYS_VALUE_NONE + 1)
        {
            if(!(_annotation.type > ROCPROFSYS_VALUE_NONE &&
                 _annotation.type < ROCPROFSYS_VALUE_LAST))
            {
                ROCPROFSYS_FAIL_F(
                    "Error! annotation '%s' has an invalid type designation "
                    "%lu which is outside of acceptable range [%i, %i]\n",
                    _annotation.name, _annotation.type, ROCPROFSYS_VALUE_NONE + 1,
                    ROCPROFSYS_VALUE_LAST - 1);
            }
        }

        if constexpr(sizeof...(Tail) > 0)
        {
            add_perfetto_annotation(ctx, _annotation, std::index_sequence<Tail...>{});
        }
        else
        {
            throw ::rocprofsys::exception<std::runtime_error>(
                "invalid annotation value type");
        }
    }
}

void
add_perfetto_annotation(perfetto_event_context_t&      ctx,
                        const rocprofsys_annotation_t& _annotation);
}  // namespace tracing
}  // namespace rocprofsys

#include <timemory/operations/types/annotate.hpp>

namespace tim
{
namespace operation
{
using perfetto_event_context_t = ::rocprofsys::tracing::perfetto_event_context_t;

template <typename Tp>
struct annotate<perfetto_event_context_t, Tp>
{
    TIMEMORY_DEFAULT_OBJECT(annotate)

    auto operator()(Tp& obj, perfetto_event_context_t& _ctx) const
    {
        return sfinae(obj, 0, _ctx);
    }

private:
    //  If the component has a annotate(...) member function
    template <typename T>
    static auto sfinae(T&                        obj, int,
                       perfetto_event_context_t& _ctx) -> decltype(obj.annotate(_ctx))
    {
        static_assert(std::is_same<T, Tp>::value, "Error T != Tp");
        return obj.annotate(_ctx);
    }

    //  If the component does not have a annotate(...) member function
    template <typename T>
    static void sfinae(T& obj, long, perfetto_event_context_t& _ctx)
    {
        static_assert(std::is_same<T, Tp>::value, "Error T != Tp");
        using value_type = typename T::value_type;
        if constexpr(!std::is_void<value_type>::value)
        {
            auto _obj_data = sfinae_data<Tp, decltype(obj.get())>(obj, 0);
            for(size_t i = 0; i < std::get<0>(_obj_data); ++i)
            {
                auto&& _label = std::get<1>(_obj_data).at(i);
                auto&& _value = std::get<2>(_obj_data).at(i);
                ::rocprofsys::tracing::add_perfetto_annotation(_ctx, _label, _value);
            }
        }
        (void) _ctx;
    }

    template <typename T, typename DataT>
    static auto sfinae_data(T& obj, int)
        -> decltype(std::tuple<size_t, std::vector<std::string>, DataT>(obj.get().size(),
                                                                        obj.label_array(),
                                                                        obj.get()))
    {
        static_assert(std::is_same<T, Tp>::value, "Error T != Tp");
        auto _labels = obj.label_array();
        auto _data   = obj.get();
        auto _size   = std::min<size_t>(_labels.size(), _data.size());
        return std::make_tuple(_size, _labels, _data);
    }

    template <typename T, typename DataT>
    static auto sfinae_data(T& obj, long)
    {
        using strvec_t    = std::vector<std::string>;
        using datavec_t   = std::vector<DataT>;
        size_t    _size   = 1;
        strvec_t  _labels = { obj.get_label() };
        datavec_t _data   = { obj.get() };
        return std::tuple<size_t, strvec_t, datavec_t>{ _size, _labels, _data };
    }
};

template <typename Tp>
struct perfetto_annotate : annotate<perfetto_event_context_t, Tp>
{
    using base_type = annotate<perfetto_event_context_t, Tp>;

    TIMEMORY_DEFAULT_OBJECT(perfetto_annotate)

    auto operator()(Tp& obj, perfetto_event_context_t& _ctx) const
    {
        return base_type::operator()(obj, _ctx);
    }
};
}  // namespace operation
}  // namespace tim
