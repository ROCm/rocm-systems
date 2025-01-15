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

#include "library/components/pthread_gotcha.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
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

    ROCPROFSYS_DEFAULT_OBJECT(stop)

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

std::set<pthread_gotcha::native_handle_t>
pthread_gotcha::get_native_handles()
{
    return ::rocprofsys::component::pthread_create_gotcha::get_native_handles();
}
}  // namespace rocprofsys
