//  MIT License
//
//  Copyright (c) 2020, The Regents of the University of California,
//  through Lawrence Berkeley National Laboratory (subject to receipt of any
//  required approvals from the U.S. Dept. of Energy).  All rights reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

/** \file mpl/types.hpp
 * \headerfile mpl/types.hpp "timemory/mpl/types.hpp"
 *
 * This is a declaration of all the operation structs.
 * Care should be taken to make sure that this includes a minimal
 * number of additional headers.
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/mpl/concepts.hpp"
#include "timemory/utility/types.hpp"

#include <cstdint>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

//======================================================================================//
//
namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_DEPRECATED)
//
template <typename TupleT, typename ListT>
class[[deprecated("Use component_bundle<T..., L*...>")]] component_hybrid;
//
template <typename TupleT, typename ListT>
class[[deprecated("Use auto_bundle<T..., L*...>")]] auto_hybrid;
//
#endif
//
//======================================================================================//
// type-traits for customization
//
namespace trait
{
//
using default_record_statistics_type = TIMEMORY_DEFAULT_STATISTICS_TYPE;
//
template <typename TraitT>
std::string
as_string();
//
/// \struct tim::trait::apply
/// \brief generic functions for setting/accessing static properties on types
template <template <typename...> class TraitT, typename... CommonT>
struct apply
{
    //
    TIMEMORY_DEFAULT_OBJECT(apply)
    //
    template <typename... Types, typename... Args>
    static inline void set(Args&&... args)
    {
        TIMEMORY_FOLD_EXPRESSION(
            TraitT<Types, CommonT...>::set(std::forward<Args>(args)...));
    }
    //
    template <typename... Types, typename... Args>
    static inline auto get(Args&&... args)
    {
        return std::make_tuple(
            TraitT<Types, CommonT...>::get(std::forward<Args>(args)...)...);
    }
    //
    template <typename... Types>
    inline bool operator()(type_list<Types...>)
    {
        bool _ret = true;
        TIMEMORY_FOLD_EXPRESSION(_ret = _ret && TraitT<Types>::get());
        return _ret;
    }
};  //
//
template <typename T>
struct base_has_accum;

template <typename T>
struct base_has_last;

template <typename T>
struct is_available;

template <typename T>
struct data;

template <typename T, bool>
struct component_value_type;

template <typename T>
struct collects_data;

template <typename T>
using runtime_configurable = concepts::is_runtime_configurable<T>;

template <typename T = void>
struct supports_runtime_enabled;

template <typename T = void>
struct default_runtime_enabled;

template <typename T = void>
struct runtime_enabled;

template <typename T>
struct record_max;

template <typename T>
struct array_serialization;

template <typename T>
struct requires_prefix;

template <typename T>
struct custom_label_printing;

template <typename T>
struct custom_unit_printing;

template <typename T>
struct start_priority;

template <typename T>
struct stop_priority;

template <typename T>
struct fini_priority;

template <typename T>
struct is_timing_category;

template <typename T>
struct is_memory_category;

template <typename T>
struct uses_timing_units;

template <typename T>
struct uses_memory_units;

template <typename T>
struct uses_percent_units;

template <typename T>
struct requires_json;

template <typename T>
struct is_component;

template <typename T, typename Tag = void>
struct api_components;

template <typename T>
struct component_apis;

template <typename T>
struct is_gotcha;

template <typename T>
struct is_user_bundle;

template <typename T, typename Tuple>
struct supports_args;

template <typename T>
struct supports_custom_record;

template <typename T>
struct iterable_measurement;

template <typename T>
struct secondary_data;

template <typename T>
struct thread_scope_only;

template <typename T>
struct custom_serialization;

template <typename T>
struct record_statistics;

template <typename T>
struct statistics;

template <typename T>
struct permissive_statistics;

template <typename T>
struct sampler;

template <typename T>
struct file_sampler;

template <typename T>
struct units;

template <typename T>
struct echo_enabled;

template <typename Api = TIMEMORY_API>
struct api_input_archive;

template <typename Api = TIMEMORY_API>
struct api_output_archive;

template <typename T, typename Api = TIMEMORY_API>
struct input_archive;

template <typename T, typename Api = TIMEMORY_API>
struct output_archive;

template <typename T>
struct pretty_archive;

template <typename T>
struct archive_extension;

template <typename T>
struct report;

template <typename T>
struct report_count;

template <typename T>
struct report_depth;

template <typename T>
struct report_metric_name;

template <typename T>
struct report_units;

template <typename T>
struct report_sum;

template <typename T>
struct report_mean;

template <typename T>
struct report_statistics;

template <typename T>
struct report_self;

template <typename T>
struct ompt_handle;

template <typename T>
struct supports_flamegraph;

template <typename T>
struct derivation_types;

template <int OpT, typename T>
struct python_args;

template <typename T>
struct cache;

template <typename T, typename V = trait::data<T>>
struct generates_output;

template <typename T>
struct uses_storage;

template <typename T>
struct tree_storage;

template <typename T>
struct flat_storage;

template <typename T>
struct timeline_storage;

template <typename T, typename V = trait::data<T>, typename A = trait::uses_storage<T>>
struct uses_value_storage;

//--------------------------------------------------------------------------------------//
//
//                              ALIASES
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
using input_archive_t = typename input_archive<T, TIMEMORY_API>::type;

template <typename T>
using output_archive_t = typename output_archive<T, TIMEMORY_API>::type;
//
//--------------------------------------------------------------------------------------//
//
}  // namespace trait

//======================================================================================//
// generic helpers that can/should be inherited from
//
namespace policy
{
/// \struct tim::policy::instance_tracker
/// \brief Inherit from this policy to add reference counting support. Useful if
/// you want to turn a global setting on/off when the number of components taking
/// measurements has hit zero (e.g. all instances of component have called stop).
/// Simply provides member functions and data values, increment/decrement is not
/// automatically performed.
/// In general, call instance_tracker::start or instance_tracker::stop inside
/// of the components constructor if a component collects data and is using
/// storage because instance(s) will be in the call-graph and thus, the instance
/// count will always be > 0. Set the second template parameter to true if thread-local
/// instance tracking is desired.
/// \code{.cpp}
/// struct foo
/// : public base<foo, void>
/// , private policy::instance_tracker<foo, false>
/// {
///     using value_type         = void;
///     using instance_tracker_t = policy::instance_tracker<foo, false>;
///
///     void start()
///     {
///         auto cnt = instance_tracker_t::start();
///         if(cnt == 0)
///             // start something
///     }
///
///     void stop()
///     {
///         auto cnt = instance_tracker_t::stop();
///         if(cnt == 0)
///             // stop something
///     }
/// };
/// \endcode
template <typename T, bool WithThreads = true>
struct instance_tracker;

/// \struct tim::policy::record_statistics
/// \brief Specification of how to accumulate statistics. This will not be used
/// unless \ref tim::trait::statistics has been assigned a type and \ref
/// tim::trait::record_statistics is true. Set \ref tim::trait::permissive_statistics to
/// allow implicit conversions, e.g. int -> size_t.
/// \code{.cpp}
/// template <typename CompT, typename Tp>
/// struct record_statistics
/// {
///     using type            = Tp;
///     using component_type  = CompT;
///     using statistics_type = statistics<type>;
///
///     static void apply(statistics_type& stats, const component_type& obj)
///     {
///         // example:
///         //      typeid(stats) is tim::statistics<double>
///         //      obj.get() returns a double precision value
///         stats += obj.get();
///     }
/// };
/// \endcode
template <typename CompT, typename T = typename trait::statistics<CompT>::type>
struct record_statistics;

/// \struct tim::policy::input_archive
/// \brief Provides a static get() function which returns a shared pointer to an instance
/// of the given archive format for input. Can also provides static functions for any
/// global configuration options, if necessary.
template <typename Archive, typename Api = TIMEMORY_API>
struct input_archive;

/// \struct tim::policy::output_archive
/// \brief Provides a static get() function which return a shared pointer to an instance
/// of the given archive format for output. Can also provide static functions for any
/// global configuration options for the archive format. For example, the (pretty) JSON
/// output archive supports specification of the precision, indentation length, and the
/// indentation character.
template <typename Archive, typename Api = TIMEMORY_API>
struct output_archive;

//--------------------------------------------------------------------------------------//
//
//                              ALIASES
//
//--------------------------------------------------------------------------------------//
//
template <typename T>
using input_archive_t = input_archive<trait::input_archive_t<T>, TIMEMORY_API>;

template <typename T>
using output_archive_t = output_archive<trait::output_archive_t<T>, TIMEMORY_API>;
//
//--------------------------------------------------------------------------------------//
//

}  // namespace policy
//
//--------------------------------------------------------------------------------------//
//
namespace impl
{
//--------------------------------------------------------------------------------------//
//
template <typename... Types>
struct tuple_concat
{
    using type = std::tuple<Types...>;
};

//--------------------------------------------------------------------------------------//

template <>
struct tuple_concat<>
{
    using type = std::tuple<>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts>
struct tuple_concat<std::tuple<Ts...>>
{
    using type = std::tuple<Ts...>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts0, typename... Ts1, typename... Rest>
struct tuple_concat<std::tuple<Ts0...>, std::tuple<Ts1...>, Rest...>
: tuple_concat<std::tuple<Ts0..., Ts1...>, Rest...>
{};

//--------------------------------------------------------------------------------------//
//
//          type_list concatenation
//
//--------------------------------------------------------------------------------------//

template <typename... Types>
struct type_concat
{
    using type = type_list<Types...>;
};

//--------------------------------------------------------------------------------------//

template <>
struct type_concat<>
{
    using type = type_list<>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts>
struct type_concat<type_list<Ts...>>
{
    using type = type_list<Ts...>;
};

//--------------------------------------------------------------------------------------//

template <typename... Ts0, typename... Ts1, typename... Rest>
struct type_concat<type_list<Ts0...>, type_list<Ts1...>, Rest...>
: type_concat<type_list<Ts0..., Ts1...>, Rest...>
{};

//--------------------------------------------------------------------------------------//
///
/// get the index of a type in expansion
///
template <typename Tp, typename Type>
struct index_of;

template <typename Tp, template <typename...> class Tuple, typename... Types>
struct index_of<Tp, Tuple<Tp, Types...>>
{
    static constexpr size_t value = 0;
};

template <typename Tp, typename Head, template <typename...> class Tuple,
          typename... Tail>
struct index_of<Tp, Tuple<Head, Tail...>>
{
    static constexpr size_t value = 1 + index_of<Tp, Tuple<Tail...>>::value;
};

//--------------------------------------------------------------------------------------//

}  // namespace impl

//======================================================================================//

template <typename... Ts>
using tuple_concat_t = typename impl::tuple_concat<Ts...>::type;

template <typename... Ts>
using type_concat_t = typename impl::type_concat<Ts...>::type;

template <typename U>
using remove_pointer_t = typename std::remove_pointer<U>::type;

template <typename U>
using add_pointer_t = conditional_t<(std::is_pointer<U>::value), U, U*>;

template <typename Tp, typename Type>
using index_of = impl::index_of<Tp, Type>;

//======================================================================================//

namespace impl
{
//--------------------------------------------------------------------------------------//

template <typename T>
struct unwrapper
{
    using type = T;
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class Tuple, typename... T>
struct unwrapper<Tuple<T...>>
{
    using type = Tuple<T...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class Tuple, typename... T>
struct unwrapper<Tuple<Tuple<T...>>>
{
    using type = conditional_t<(std::is_same<Tuple<>, std::tuple<>>::value ||
                                std::is_same<Tuple<>, type_list<>>::value),
                               typename unwrapper<Tuple<T...>>::type, Tuple<Tuple<T...>>>;
};

//--------------------------------------------------------------------------------------//

template <typename In, typename Out>
struct convert
{
    using type = Out;

    using input_type  = In;
    using output_type = Out;

    static output_type apply(const input_type& _in)
    {
        return static_cast<output_type>(_in);
    }
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class InTuple, typename... In,
          template <typename...> class OutTuple, typename... Out>
struct convert<InTuple<In...>, OutTuple<Out...>>
{
    using type = OutTuple<Out..., In...>;

    using input_type  = InTuple<In...>;
    using output_type = OutTuple<Out...>;

    static output_type apply(const input_type& _in)
    {
        output_type _out{};
        TIMEMORY_FOLD_EXPRESSION(
            std::get<index_of<Out, output_type>::value>(_out) =
                static_cast<Out>(std::get<index_of<In, input_type>::value>(_in)));
        return _out;
    }
};

//--------------------------------------------------------------------------------------//

template <typename ApiT, template <typename...> class InTuple,
          template <typename...> class OutTuple, typename... In>
struct convert<InTuple<ApiT, In...>, OutTuple<ApiT>>
: convert<type_list<In...>, OutTuple<ApiT>>
{};

//--------------------------------------------------------------------------------------//

template <template <typename...> class InTuple, typename... T>
struct convert_each;

template <template <typename...> class OutTuple, template <typename...> class InTuple,
          typename... In>
struct convert_each<OutTuple, InTuple<In...>>
{
    using type = std::tuple<OutTuple<In>...>;
};

//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_USE_DEPRECATED)
//
template <template <typename...> class LhsInT, typename... LhsIn,
          template <typename...> class RhsInT, typename... RhsIn,
          template <typename...> class LhsOutT, typename... LhsOut,
          template <typename...> class RhsOutT, typename... RhsOut>
struct convert<component_hybrid<LhsInT<LhsIn...>, RhsInT<RhsIn...>>,
               auto_hybrid<LhsOutT<LhsOut...>, RhsOutT<RhsOut...>>>
{
    using type = auto_hybrid<LhsInT<LhsIn...>, RhsInT<RhsIn...>>;
};

//--------------------------------------------------------------------------------------//

template <template <typename...> class LhsInT, typename... LhsIn,
          template <typename...> class RhsInT, typename... RhsIn,
          template <typename...> class LhsOutT, typename... LhsOut,
          template <typename...> class RhsOutT, typename... RhsOut>
struct convert<auto_hybrid<LhsInT<LhsIn...>, RhsInT<RhsIn...>>,
               component_hybrid<LhsOutT<LhsOut...>, RhsOutT<RhsOut...>>>
{
    using type = component_hybrid<LhsInT<LhsIn...>, RhsInT<RhsIn...>>;
};
//
#endif

//--------------------------------------------------------------------------------------//

template <template <typename> class CheckT, bool CheckV,
          template <typename> class TransformT, typename... T>
struct apply_transform;

//--------------------------------------------------------------------------------------//

/*template <typename In, typename Out>
struct apply_transform<identity, true, identity, In, Out>
{
    using type = Out;
};*/

//--------------------------------------------------------------------------------------//

template <template <typename> class CheckT, bool CheckV,
          template <typename> class TransformT, typename... In, typename... Out>
struct apply_transform<CheckT, CheckV, TransformT, type_list<In...>, type_list<Out...>>
{
    using type = type_list<Out..., conditional_t<CheckT<In>::value == CheckV,
                                                 typename TransformT<In>::type, In>...>;
};

//--------------------------------------------------------------------------------------//

template <template <typename> class CheckT, bool CheckV,
          template <typename> class TransformT, typename ApiT, typename... In>
struct apply_transform<CheckT, CheckV, TransformT, type_list<ApiT, In...>,
                       type_list<ApiT>>
: apply_transform<CheckT, CheckV, TransformT, type_list<In...>, type_list<ApiT>>
{};

//======================================================================================//
// check if type is in expansion
//
template <typename...>
struct is_one_of
{
    static constexpr bool value = false;
};

template <typename F, typename S, template <typename...> class Tuple, typename... T>
struct is_one_of<F, S, Tuple<T...>>
{
    static constexpr bool value =
        std::is_same<F, S>::value || is_one_of<F, Tuple<T...>>::value;
};

template <typename F, typename S, template <typename...> class Tuple, typename... T>
struct is_one_of<F, Tuple<S, T...>>
{
    static constexpr bool value = is_one_of<F, S, Tuple<T...>>::value;
};

//======================================================================================//
// check if trait is satisfied by at least one type in variadic sequence
//
template <template <typename> class Test, typename Sequence>
struct contains_one_of;

template <template <typename> class Test, template <typename...> class Tuple>
struct contains_one_of<Test, Tuple<>>
{
    static constexpr bool value = false;
    using type                  = Tuple<>;
};

template <template <typename> class Test, typename F, template <typename...> class Tuple,
          typename... T>
struct contains_one_of<Test, Tuple<F, T...>>
{
    static constexpr bool value =
        Test<F>::value || contains_one_of<Test, Tuple<T...>>::value;
    using type = conditional_t<Test<F>::value, F,
                               typename contains_one_of<Test, Tuple<T...>>::type>;
};

}  // namespace impl

//======================================================================================//
///
/// check if type is in expansion
///
template <typename Tp, typename Types>
using is_one_of = typename impl::is_one_of<Tp, Types>;

///
/// check if type is in expansion
///
template <template <typename> class Predicate, typename Types>
using contains_one_of = typename impl::contains_one_of<Predicate, Types>;

template <template <typename> class Predicate, typename Types>
using contains_one_of_t = typename contains_one_of<Predicate, Types>::type;

//======================================================================================//

namespace impl
{
template <typename T>
struct wrapper_index_sequence;
template <typename T>
struct nonwrapper_index_sequence;

template <template <typename...> class Tuple, typename... Types>
struct wrapper_index_sequence<Tuple<Types...>>
{
    static constexpr auto size  = sizeof...(Types);
    static constexpr auto value = make_index_sequence<size>{};
    using type                  = decltype(make_index_sequence<size>{});
};

template <template <typename...> class Tuple, typename... Types>
struct nonwrapper_index_sequence<Tuple<Types...>>
{
    static constexpr auto size  = sizeof...(Types);
    static constexpr auto value = type_list<>{};
    using type                  = type_list<>;
};
}  // namespace impl

template <typename Tp>
struct get_index_sequence
{
    static constexpr auto size  = 0;
    static constexpr auto value = type_list<>{};
    using type                  = type_list<>;
};

template <typename Lhs, typename Rhs>
struct get_index_sequence<std::pair<Lhs, Rhs>>
{
    static constexpr auto size  = 2;
    static constexpr auto value = index_sequence<0, 1>{};
    using type                  = index_sequence<0, 1>;
};

template <typename... Types>
struct get_index_sequence<std::tuple<Types...>>
{
    static constexpr auto size  = std::tuple_size<std::tuple<Types...>>::value;
    static constexpr auto value = make_index_sequence<size>{};
    using type                  = decltype(make_index_sequence<size>{});
};

template <template <typename...> class Tuple, typename... Types>
struct get_index_sequence<Tuple<Types...>>
{
    using base_type = conditional_t<concepts::is_variadic<Tuple<Types...>>::value,
                                    impl::wrapper_index_sequence<Tuple<Types...>>,
                                    impl::nonwrapper_index_sequence<Tuple<Types...>>>;
    static constexpr auto size  = base_type::size;
    static constexpr auto value = base_type::value;
    using type                  = typename base_type::type;
};

template <typename Tp>
using get_index_sequence_t = typename get_index_sequence<decay_t<Tp>>::type;

//======================================================================================//

template <typename T, typename U>
using convert_t = typename impl::convert<T, U>::type;

template <template <typename...> class T, typename... U>
using convert_each_t = typename impl::convert_each<T, U...>::type;

template <typename T>
using unwrap_t = typename impl::unwrapper<T>::type;

template <template <typename> class CheckT, bool CheckV,
          template <typename> class TransformT, typename T, typename U>
using apply_transform_t =
    typename impl::apply_transform<CheckT, CheckV, TransformT, T, U>::type;

//======================================================================================//

namespace mpl
{
//--------------------------------------------------------------------------------------//

template <typename Tp,
          typename std::enable_if<(std::is_arithmetic<Tp>::value), int>::type = 0>
constexpr auto
get_size(const Tp&, std::tuple<>) -> size_t
{
    return 1;
}

template <typename Tp>
auto
get_size(const Tp& _val, std::tuple<>) -> decltype(_val.size(), size_t())
{
    return _val.size();
}

template <typename Tp,
          typename std::enable_if<(std::is_arithmetic<Tp>::value), int>::type = 0>
constexpr auto
get_size(const Tp&, type_list<>) -> size_t
{
    return 1;
}

template <typename Tp>
auto
get_size(const Tp& _val, type_list<>) -> decltype(_val.size(), size_t())
{
    return _val.size();
}

template <typename Tp, size_t... Idx>
constexpr auto
get_size(const Tp& _val, index_sequence<Idx...>) -> decltype(std::get<0>(_val), size_t())
{
    return std::tuple_size<Tp>::value;
}

template <typename Tp>
auto
get_size(const Tp& _val)
    -> decltype(get_size(_val, get_index_sequence<decay_t<Tp>>::value))
{
    return get_size(_val, get_index_sequence<decay_t<Tp>>::value);
}

template <typename Tp>
struct get_tuple_size
{
    static constexpr size_t value = get_index_sequence<decay_t<Tp>>::size;
};

//--------------------------------------------------------------------------------------//

template <typename T>
auto
resize(T&, ...) -> void
{}

template <typename T>
auto
resize(T& _targ, size_t _n) -> decltype(_targ.resize(_n), void())
{
    _targ.resize(_n);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
void
assign(Tp& _targ, const Tp& _val, ...)
{
    _targ = _val;
}

template <typename Tp, typename Vp, typename ValueType = typename Tp::value_type>
auto
assign(Tp& _targ, const Vp& _val, type_list<>) -> decltype(_targ[0], void())
{
    auto _n = get_size(_val);
    resize(_targ, _n);
    for(decltype(_n) i = 0; i < _n; ++i)
    {
        assign(_targ[i], *(_val.begin() + i),
               get_index_sequence<decay_t<ValueType>>::value);
    }
}

template <typename Tp, size_t... Idx>
auto
assign(Tp& _targ, const Tp& _val, index_sequence<Idx...>)
    -> decltype(std::get<0>(_val), void())
{
    TIMEMORY_FOLD_EXPRESSION(
        assign(std::get<Idx>(_targ), std::get<Idx>(_val),
               get_index_sequence<decay_t<decltype(std::get<Idx>(_targ))>>::value));
}

template <typename Tp, typename Vp, size_t... Idx,
          enable_if_t<!std::is_same<Tp, Vp>::value, int> = 0>
auto
assign(Tp& _targ, const Vp& _val, index_sequence<Idx...>)
    -> decltype(std::get<0>(_targ) = *std::begin(_val), void())
{
    TIMEMORY_FOLD_EXPRESSION(
        assign(std::get<Idx>(_targ), *(std::begin(_val) + Idx),
               get_index_sequence<decay_t<decltype(std::get<Idx>(_targ))>>::value));
}

template <typename Tp, typename Vp>
void
assign(Tp& _targ, const Vp& _val)
{
    assign(_targ, _val, get_index_sequence<decay_t<Tp>>::value);
}

//--------------------------------------------------------------------------------------//

template <typename Tuple, typename T>
struct push_back;

template <template <typename...> class Tuple, typename... Types, typename T>
struct push_back<Tuple<Types...>, T>
{
    using type = Tuple<Types..., T>;
};

template <template <typename, typename> class Hybrid, template <typename...> class Lhs,
          typename... LhsTypes, template <typename...> class Rhs, typename... RhsTypes,
          typename T>
struct push_back<Hybrid<Lhs<LhsTypes...>, Rhs<RhsTypes...>>, T>
{
    using type = Hybrid<Lhs<LhsTypes..., T>, Rhs<RhsTypes...>>;
};

//--------------------------------------------------------------------------------------//

template <typename... T>
struct union_index_of;

template <typename T, template <typename...> class TupleT, typename... Types>
struct union_index_of<T, TupleT<Types...>>
{
    template <typename U = T>
    static constexpr auto value(
        int, enable_if_t<is_one_of<U, TupleT<Types...>>::value, int> = 0)
    {
        return index_of<T, TupleT<Types...>>::value;
    }

    template <typename U = T>
    static constexpr auto value(
        long, enable_if_t<is_one_of<U*, TupleT<Types...>>::value, int> = 0)
    {
        return index_of<T*, TupleT<Types...>>::value;
    }
};

template <typename T, template <typename...> class TupleT, typename... Types>
struct union_index_of<T*, TupleT<Types...>>
{
    template <typename U = T>
    static constexpr auto value(
        long, enable_if_t<is_one_of<U, TupleT<Types...>>::value, int> = 0)
    {
        return index_of<T, TupleT<Types...>>::value;
    }

    template <typename U = T>
    static constexpr auto value(
        int, enable_if_t<is_one_of<U*, TupleT<Types...>>::value, int> = 0)
    {
        return index_of<T*, TupleT<Types...>>::value;
    }
};

template <typename... T>
struct union_index_sequence;

template <template <typename...> class LhsT, typename... Lhs,
          template <typename...> class RhsT, typename... Rhs>
struct union_index_sequence<LhsT<Lhs...>, RhsT<Rhs...>>
{
    using type =
        index_sequence<union_index_of<decay_t<Lhs>, RhsT<decay_t<Rhs>...>>::value(0)...>;
};

template <typename... T>
using union_index_sequence_t = typename union_index_sequence<T...>::type;

namespace impl
{
template <typename Tp, size_t... Idx>
TIMEMORY_INLINE decltype(auto)
get_reference_tuple(Tp&& _tuple, index_sequence<Idx...>)
{
    return std::tie(std::get<Idx>(_tuple)...);
}
}  // namespace impl

template <typename Rp, typename Tp>
TIMEMORY_INLINE decltype(auto)
get_reference_tuple(Tp&& _tuple)
{
    // static_assert(concepts::is_variadic<Tp>::value, "Requires variadic type");
    using sequence_type = union_index_sequence_t<Rp, decay_t<Tp>>;
    return impl::get_reference_tuple(std::forward<Tp>(_tuple), sequence_type{});
}

//--------------------------------------------------------------------------------------//

}  // namespace mpl

template <typename Tuple, typename T>
using push_back_t = typename mpl::push_back<Tuple, T>::type;

}  // namespace tim
