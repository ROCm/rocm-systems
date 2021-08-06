// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>

#if __cplusplus >= 201703L  // C++17
#    include <string_view>
#endif

//--------------------------------------------------------------------------------------//

#define TIMEMORY_IMPL_IS_CONCEPT(CONCEPT)                                                \
    struct CONCEPT                                                                       \
    {};                                                                                  \
    template <typename Tp>                                                               \
    struct is_##CONCEPT                                                                  \
    {                                                                                    \
    private:                                                                             \
        template <typename, typename = std::true_type>                                   \
        struct have : std::false_type                                                    \
        {};                                                                              \
        template <typename U>                                                            \
        struct have<U, typename std::is_base_of<typename U::CONCEPT, U>::type>           \
        : std::true_type                                                                 \
        {};                                                                              \
        template <typename U>                                                            \
        struct have<U, typename std::is_base_of<typename U::CONCEPT##_type, U>::type>    \
        : std::true_type                                                                 \
        {};                                                                              \
                                                                                         \
    public:                                                                              \
        using type = typename is_##CONCEPT::template have<                               \
            typename std::remove_cv<Tp>::type>::type;                                    \
        static constexpr bool value =                                                    \
            is_##CONCEPT::template have<typename std::remove_cv<Tp>::type>::value;       \
    };
// constexpr operator bool() const noexcept { return value; }

//--------------------------------------------------------------------------------------//
namespace tim
{
namespace cereal
{
namespace detail
{
//
class OutputArchiveBase;
class InputArchiveBase;
//
}  // namespace detail
}  // namespace cereal
}  // namespace tim
//
namespace tim
{
//
using true_type  = std::true_type;
using false_type = std::false_type;
//
struct null_type;
//
struct quirk_type;
//
namespace audit
{
struct incoming;
struct outgoing;
}  // namespace audit
//
namespace trait
{
template <typename Tp>
struct is_available;
//
template <typename Tp>
struct is_component;
}  // namespace trait
//
namespace
{
template <typename Tp>
struct anonymous
{
    using type = Tp;
};
//
template <typename Tp>
using anonymous_t = typename anonymous<Tp>::type;
}  // namespace
//
namespace component
{
template <size_t Idx, typename Tag>
struct user_bundle;
//
template <size_t Nt, typename ComponentsT, typename DifferentiatorT = anonymous_t<void>>
struct gotcha;
//
template <typename... Types>
struct placeholder;
//
struct nothing;
//
}  // namespace component
//
template <typename... Types>
class lightweight_tuple;
//
namespace concepts
{
//
using input_archive_base  = cereal::detail::InputArchiveBase;
using output_archive_base = cereal::detail::OutputArchiveBase;
//
//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_empty
/// \brief concept that specifies that a variadic type is empty
///
TIMEMORY_IMPL_IS_CONCEPT(empty)

template <template <typename...> class Tuple>
struct is_empty<Tuple<>> : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_null_type
/// \brief concept that specifies that a type is not a useful type
///
TIMEMORY_IMPL_IS_CONCEPT(null_type)

template <template <typename...> class Tuple>
struct is_null_type<Tuple<>> : true_type
{};

template <>
struct is_null_type<void> : true_type
{};

template <>
struct is_null_type<false_type> : true_type
{};

template <>
struct is_null_type<::tim::null_type> : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_placeholder
/// \brief concept that specifies that a type is not necessarily marked as not available
/// but is still a dummy type
///
TIMEMORY_IMPL_IS_CONCEPT(placeholder)

template <typename... Types>
struct is_placeholder<component::placeholder<Types...>> : true_type
{};

template <>
struct is_placeholder<component::placeholder<component::nothing>> : true_type
{};

template <>
struct is_placeholder<component::nothing> : true_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_component
/// \brief concept that specifies that a type is a component. Components are used to
/// perform some measurement, capability, or logging implementation. Adding this
/// concept can be performs through inheriting from \ref tim::component::base,
/// inheriting from tim::concepts::component, or specializing either \ref
/// tim::concepts::is_component or \ref tim::trait::is_component (with the latter
/// being deprecated).
///
struct component
{};
//
template <typename Tp>
struct is_component
{
private:
    /// did not inherit
    template <typename, typename = std::true_type>
    struct have : std::false_type
    {};

    /// did inherit
    template <typename U>
    struct have<U, typename std::is_base_of<typename U::component, U>::type>
    : std::true_type
    {};

    /// this uses sfinae to see if U::is_component is provided within the type
    template <typename U>
    static constexpr decltype(U::is_component, bool{}) test(int)
    {
        return U::is_component;
    }

    /// this checks for the (deprecated) `tim::trait::is_component` specialization
    /// and checks for inheritance
    template <typename Up>
    static constexpr bool test(long)
    {
        return trait::is_component<Up>::value ||
               is_component::template have<std::remove_cv_t<Tp>>::value;
    }

public:
    static constexpr bool value = test<Tp>(int{});
    using type                  = std::conditional_t<value, true_type, false_type>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_quirk_type
/// \brief concept that specifies that a type is a quirk. Quirks are used to modify
/// the traditional behavior of component bundles slightly. E.g. disable calling
/// start in the constructor of an auto_tuple.
///
TIMEMORY_IMPL_IS_CONCEPT(quirk_type)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_api
/// \brief concept that specifies that a type is an API. APIs are used to designate
/// different project implementations, different external library tools, etc.
///
TIMEMORY_IMPL_IS_CONCEPT(api)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_variadic
/// \brief concept that specifies that a type is a generic variadic wrapper
///
TIMEMORY_IMPL_IS_CONCEPT(variadic)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
///
TIMEMORY_IMPL_IS_CONCEPT(wrapper)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_stack_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// and components are stack-allocated
///
TIMEMORY_IMPL_IS_CONCEPT(stack_wrapper)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_heap_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// and components are heap-allocated
///
TIMEMORY_IMPL_IS_CONCEPT(heap_wrapper)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_mixed_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// and variadic types are mix of stack- and heap- allocated
///
TIMEMORY_IMPL_IS_CONCEPT(mixed_wrapper)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_tagged
/// \brief concept that specifies that a type's template parameters include
/// a API specific tag as one of the template parameters (usually first)
///
TIMEMORY_IMPL_IS_CONCEPT(tagged)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_comp_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// that does not perform auto start/stop, e.g. component_{tuple,list,hybrid}
///
TIMEMORY_IMPL_IS_CONCEPT(comp_wrapper)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_auto_wrapper
/// \brief concept that specifies that a type is a timemory variadic wrapper
/// that performs auto start/stop, e.g. auto_{tuple,list,hybrid}
///
TIMEMORY_IMPL_IS_CONCEPT(auto_wrapper)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_runtime_configurable
/// \brief concept that specifies that a type is used to modify behavior at runtime.
/// For example, the \ref tim::component::user_bundle component is runtime configurable bc
/// it allows you insert components at runtime. The timing category
/// (`tim::category::timing`) is another example of a type that is runtime configurable --
/// setting `tim::trait::runtime_enabled<tim::category::timing>::set(false);` will disable
/// (at runtime) all the types which are part of the timing API. It should be noted that
/// types which satisfy `is_runtime_configurable<Tp>::value == true` (e.g. \ref
/// tim::component::user_bundle) are not eligible to be inserted into other runtime
/// configurable components; i.e. you cannot insert/add \ref
/// tim::component::user_trace_bundle into \ref tim::component::user_global_bundle, etc.
/// This restriction is primarily due to the significant increase in compile-time that
/// arises from allowing this behavior.
///
TIMEMORY_IMPL_IS_CONCEPT(runtime_configurable)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_external_function_wrapper
/// \brief concept that specifies that a component type wraps external functions
///
TIMEMORY_IMPL_IS_CONCEPT(external_function_wrapper)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_phase_id
/// \brief concept that specifies that a type is used for identifying a phase in some
/// measurement. For example, `tim::audit::incoming` and `tim::audit::outgoing`
/// can be added to overloads to distinguish whether the `double` type in `double
/// exp(double val)` is `val` or whether it is the return value.
///
/// \code{.cpp}
/// struct exp_wrapper_A
/// {
///     // unable to distingush whether "val" is input or output
///     void audit(double val) { ... }
/// };
///
/// struct exp_wrapper_B
/// {
///     // able to distingush whether "val" is input or output
///     void audit(audit::incoming, double val) { ... }
///     void audit(audit::outgoing, double val) { ... }
/// };
/// \endcode
TIMEMORY_IMPL_IS_CONCEPT(phase_id)

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_string_type
/// \brief concept that specifies that a component type wraps external functions
///
TIMEMORY_IMPL_IS_CONCEPT(string_type)

template <>
struct is_string_type<std::string> : true_type
{};

template <>
struct is_string_type<char*> : true_type
{};

template <>
struct is_string_type<const char*> : true_type
{};

#if __cplusplus >= 201703L  // C++17
template <>
struct is_string_type<std::string_view> : true_type
{};
#endif

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::has_gotcha
/// \brief determines if a variadic wrapper contains a gotcha component
///
template <typename T>
struct has_gotcha : std::false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::has_user_bundle
/// \brief concept that specifies that a type is a user_bundle type
///
template <typename T>
struct has_user_bundle : false_type
{};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_output_archive
/// \brief concept that specifies that a type is an output serialization archive
///
template <typename Tp>
struct is_output_archive
{
private:
    /// did not inherit
    template <typename, typename = std::true_type>
    struct have : std::false_type
    {};

    /// did inherit
    template <typename U>
    struct have<U, typename std::is_base_of<output_archive_base, U>::type>
    : std::true_type
    {};

public:
    static constexpr bool value =
        is_output_archive::template have<std::remove_cv_t<Tp>>::value;
    using type = std::conditional_t<value, true_type, false_type>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_input_archive
/// \brief concept that specifies that a type is an input serialization archive
///
template <typename Tp>
struct is_input_archive
{
private:
    /// did not inherit
    template <typename, typename = std::true_type>
    struct have : std::false_type
    {};

    /// did inherit
    template <typename U>
    struct have<U, typename std::is_base_of<input_archive_base, U>::type> : std::true_type
    {};

public:
    static constexpr bool value =
        is_input_archive::template have<std::remove_cv_t<Tp>>::value;
    using type = std::conditional_t<value, true_type, false_type>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_archive
/// \brief concept that specifies that a type is a serialization archive (input or output)
///
template <typename Tp>
struct is_archive
{
    static constexpr bool value =
        is_input_archive<Tp>::value || is_output_archive<Tp>::value;
    using type = std::conditional_t<value, true_type, false_type>;
};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::is_acceptable_conversion
/// \tparam Lhs the provided type
/// \tparam Rhs the target type
///
/// \brief This concept designates that is safe to perform a `static_cast<Lhs>(rhs)`
/// where needed. This is primarily used in the \ref tim::component::data_tracker
/// where `data_tracker<unsigned int, ...>` might be provided another integral type,
/// such as `int`.
///
template <typename Lhs, typename Rhs>
struct is_acceptable_conversion
{
    static constexpr bool value =
        (std::is_same<Lhs, Rhs>::value ||
         (std::is_integral<Lhs>::value && std::is_integral<Rhs>::value) ||
         (std::is_floating_point<Lhs>::value && std::is_floating_point<Rhs>::value));
};

//--------------------------------------------------------------------------------------//
/// \struct tim::concepts::tuple_type
/// \brief This concept is used to express how to convert a given type into a
/// `std::tuple`, e.g. `tim::component_tuple<T...>` to `std::tuple<T...>`. It
/// is necessary for types like \ref tim::component_bundle where certain template
/// parameters are tags.
///
template <typename T>
struct tuple_type
{
    using type = typename T::tuple_type;
};

/// \struct tim::concepts::auto_type
/// \brief This concept is used to express how to convert a component bundler into
/// another component bundler which performs automatic starting upon construction.
///
template <typename T>
struct auto_type
{
    using type = typename T::auto_type;
};

/// \struct tim::concepts::component_type
/// \brief This concept is used to express how to convert a component bundler which
/// automatically starts upon construction into a type that requires an explicit
/// call to start.
///
template <typename T>
struct component_type
{
    using type = typename T::component_type;
};

template <>
struct component_type<std::tuple<>>
{
    using type = lightweight_tuple<>;
};

//--------------------------------------------------------------------------------------//

}  // namespace concepts

template <typename T>
using is_empty_t = typename concepts::is_empty<T>::type;

template <typename T>
using is_variadic_t = typename concepts::is_variadic<T>::type;

template <typename T>
using is_wrapper_t = typename concepts::is_wrapper<T>::type;

template <typename T>
using is_stack_wrapper_t = typename concepts::is_stack_wrapper<T>::type;

template <typename T>
using is_heap_wrapper_t = typename concepts::is_heap_wrapper<T>::type;

//--------------------------------------------------------------------------------------//

}  // namespace tim

//======================================================================================//

#if !defined(TIMEMORY_CONCEPT_ALIAS)
#    define TIMEMORY_CONCEPT_ALIAS(ALIAS, TYPE)                                          \
        namespace tim                                                                    \
        {                                                                                \
        namespace concepts                                                               \
        {                                                                                \
        template <typename T>                                                            \
        using ALIAS = typename TYPE<T>::type;                                            \
        }                                                                                \
        }
#endif

//--------------------------------------------------------------------------------------//

TIMEMORY_CONCEPT_ALIAS(is_empty_t, is_empty)
TIMEMORY_CONCEPT_ALIAS(is_variadic_t, is_variadic)
TIMEMORY_CONCEPT_ALIAS(is_wrapper_t, is_wrapper)
TIMEMORY_CONCEPT_ALIAS(is_stack_wrapper_t, is_stack_wrapper)
TIMEMORY_CONCEPT_ALIAS(is_heap_wrapper_t, is_heap_wrapper)

TIMEMORY_CONCEPT_ALIAS(tuple_type_t, tuple_type)
TIMEMORY_CONCEPT_ALIAS(auto_type_t, auto_type)
TIMEMORY_CONCEPT_ALIAS(component_type_t, component_type)

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_CONCRETE_CONCEPT(CONCEPT, SPECIALIZED_TYPE, VALUE)               \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <>                                                                          \
    struct CONCEPT<SPECIALIZED_TYPE> : VALUE                                             \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_TEMPLATE_CONCEPT(CONCEPT, SPECIALIZED_TYPE, VALUE, TYPE)         \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE T>                                                                    \
    struct CONCEPT<SPECIALIZED_TYPE<T>> : VALUE                                          \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_VARIADIC_CONCEPT(CONCEPT, SPECIALIZED_TYPE, VALUE, TYPE)         \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE... T>                                                                 \
    struct CONCEPT<SPECIALIZED_TYPE<T...>> : VALUE                                       \
    {};                                                                                  \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_CONCRETE_CONCEPT_TYPE(CONCEPT, SPECIALIZED_TYPE, ...)            \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <>                                                                          \
    struct CONCEPT<SPECIALIZED_TYPE>                                                     \
    {                                                                                    \
        using type = __VA_ARGS__;                                                        \
    };                                                                                   \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_TEMPLATE_CONCEPT_TYPE(CONCEPT, SPECIALIZED_TYPE, TYPE, ...)      \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE T>                                                                    \
    struct CONCEPT<SPECIALIZED_TYPE<T>>                                                  \
    {                                                                                    \
        using type = __VA_ARGS__;                                                        \
    };                                                                                   \
    }                                                                                    \
    }

//--------------------------------------------------------------------------------------//

#define TIMEMORY_DEFINE_VARIADIC_CONCEPT_TYPE(CONCEPT, SPECIALIZED_TYPE, TYPE, ...)      \
    namespace tim                                                                        \
    {                                                                                    \
    namespace concepts                                                                   \
    {                                                                                    \
    template <TYPE... T>                                                                 \
    struct CONCEPT<SPECIALIZED_TYPE<T...>>                                               \
    {                                                                                    \
        using type = __VA_ARGS__;                                                        \
    };                                                                                   \
    }                                                                                    \
    }
