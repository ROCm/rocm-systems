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

// For std::decay
#include <type_traits>

namespace tim
{
namespace utility
{
template <typename>
struct transient_function;  // intentionally not defined

/// \struct tim::utility::transient_function
/// \brief A light-weight alternative to std::function. Pass any callback - including
/// capturing lambdas - cheaply and quickly as a function argument
///
///  - No instantiation of called function at each call site
///  - Simple to use - use transient_function<...> as the function argument
///  - Low cost, cheap setup, one indirect function call to invoke
///  - No risk of dynamic allocation (unlike std::function)
///  - Not persistent: synchronous calls only
///
template <typename RetT, typename... Args>
struct transient_function<RetT(Args...)>
{
    using dispatch_type             = RetT (*)(void*, Args...);
    using target_function_reference = RetT(Args...);

    // dispatch_type() is instantiated by the transient_function constructor,
    // which will store a pointer to the function in m_dispatch.
    template <typename Up>
    static RetT dispatcher(void* target, Args&&... args)
    {
        return (*static_cast<Up*>(target))(std::forward<Args>(args)...);
    }

    transient_function()                              = default;
    ~transient_function()                             = default;
    transient_function(transient_function&&) noexcept = default;
    transient_function& operator=(transient_function&&) noexcept = default;

    template <typename Tp>
    transient_function(Tp&& target)
    : m_dispatch(&dispatcher<std::decay_t<Tp>>)
    , m_target(&target)
    {}

    // Specialize for reference-to-function, to ensure that a valid pointer is stored.
    transient_function(target_function_reference target)
    : m_dispatch(dispatcher<target_function_reference>)
    {
        static_assert(
            sizeof(void*) == sizeof(target),
            "It will not be possible to pass functions by reference on this platform. "
            "Please use explicit function pointers i.e. foo(target) -> foo(&target)");
        m_target = static_cast<void*>(target);
    }

    // Specialize for reference-to-function, to ensure that a valid pointer is stored.
    transient_function& operator=(target_function_reference target)
    {
        if(this == &target)
            return *this;

        static_assert(
            sizeof(void*) == sizeof(target),
            "It will not be possible to pass functions by reference on this platform. "
            "Please use explicit function pointers i.e. foo(target) -> foo(&target)");
        m_dispatch = dispatcher<target_function_reference>;
        m_target   = static_cast<void*>(target);
        return *this;
    }

    RetT operator()(Args&&... args) const
    {
        return m_dispatch(m_target, std::forward<Args>(args)...);
    }

private:
    // A pointer to the static function that will call the wrapped invokable object
    dispatch_type m_dispatch = nullptr;
    void*         m_target   = nullptr;  // pointer to the invokable object
};
//
}  // namespace utility
//
namespace scope
{
//
//--------------------------------------------------------------------------------------//
//
/// \struct tim::scope::destructor
/// \brief provides an object which can be returned from functions that will execute
/// the lambda provided during construction when it is destroyed
///
struct transient_destructor
{
    template <typename FuncT>
    transient_destructor(FuncT&& _func)
    : m_functor(std::forward<FuncT>(_func))
    {}

    // delete copy operations
    transient_destructor(const transient_destructor&) = delete;
    transient_destructor& operator=(const transient_destructor&) = delete;

    // allow move operations
    transient_destructor(transient_destructor&& rhs) noexcept = default;
    transient_destructor& operator=(transient_destructor&& rhs) noexcept = default;

    ~transient_destructor() { m_functor(); }

private:
    utility::transient_function<void()> m_functor = []() {};
};
}  // namespace scope
//
}  // namespace tim
