// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/pthread_gotcha.hpp"
#include "core/config.hpp"
#include "core/utility.hpp"
#include "library/components/pthread_create_gotcha.hpp"
#include "library/components/pthread_mutex_gotcha.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/types.hpp>

#include <pthread.h>

#include <array>
#include <vector>

namespace tim
{
namespace operation
{
template <>
struct stop<rocprofsys::component::pthread_create_gotcha_t>
{
    using type = rocprofsys::component::pthread_create_gotcha_t;

    template <typename... Args>
    explicit stop(type&, Args&&...)
    {}

    template <typename... Args>
    void operator()(type&, Args&&...)
    {}
};
}  // namespace operation
}  // namespace tim

namespace rocprofsys
{
namespace
{
using bundle_t = tim::lightweight_tuple<component::pthread_create_gotcha_t,
                                        component::pthread_mutex_gotcha_t>;

auto&
get_bundle()
{
    static auto _v = std::unique_ptr<bundle_t>{};
    if(!_v) _v = std::make_unique<bundle_t>("pthread_gotcha");
    return _v;
}

bool is_configured = false;
}  // namespace

//--------------------------------------------------------------------------------------//

void
pthread_gotcha::configure()
{
    if(!is_configured)
    {
        ::rocprofsys::component::pthread_create_gotcha::configure();
        ::rocprofsys::component::pthread_mutex_gotcha::configure();
        is_configured = true;
    }
}

void
pthread_gotcha::shutdown()
{
    if(is_configured)
    {
        ::rocprofsys::component::pthread_mutex_gotcha::shutdown();
        ::rocprofsys::component::pthread_create_gotcha::shutdown();
        is_configured = false;
    }
}

void
pthread_gotcha::start()
{
    configure();
    get_bundle()->start();
}

void
pthread_gotcha::stop()
{
    get_bundle()->stop();
}

void
pthread_gotcha::pause()
{
    ::rocprofsys::component::pthread_mutex_gotcha::pause();
    ::rocprofsys::component::pthread_create_gotcha::pause();
}

void
pthread_gotcha::resume()
{
    ::rocprofsys::component::pthread_mutex_gotcha::resume();
    ::rocprofsys::component::pthread_create_gotcha::resume();
}

std::set<pthread_gotcha::native_handle_t>
pthread_gotcha::get_native_handles()
{
    return ::rocprofsys::component::pthread_create_gotcha::get_native_handles();
}
}  // namespace rocprofsys
