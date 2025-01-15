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

#include "library/causal/components/causal_gotcha.hpp"
#include "core/config.hpp"
#include "library/causal/components/blocking_gotcha.hpp"
#include "library/causal/components/unblocking_gotcha.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/signals/signal_mask.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/types.hpp>

#include <array>
#include <vector>

namespace rocprofsys
{
namespace causal
{
namespace component
{
namespace
{
using bundle_t = tim::lightweight_tuple<blocking_gotcha_t, unblocking_gotcha_t>;

auto&
get_bundle()
{
    static auto _v = std::unique_ptr<bundle_t>{};
    if(!_v) _v = std::make_unique<bundle_t>("causal_gotcha");
    return _v;
}

const auto&
sampling_signals()
{
    static auto _v = get_sampling_signals();
    return _v;
}

bool is_configured = false;
}  // namespace

//--------------------------------------------------------------------------------------//

void
causal_gotcha::configure()
{
    if(!is_configured)
    {
        blocking_gotcha::configure();
        unblocking_gotcha::configure();
        is_configured = true;
    }
}

void
causal_gotcha::shutdown()
{
    if(is_configured)
    {
        blocking_gotcha::shutdown();
        unblocking_gotcha::shutdown();
        is_configured = false;
    }
}

void
causal_gotcha::start()
{
    configure();
    get_bundle()->start();
}

void
causal_gotcha::stop()
{
    get_bundle()->stop();
    shutdown();
}

void
causal_gotcha::remove_signals(sigset_t* _set)
{
    for(auto _sig : sampling_signals())
    {
        if(sigismember(_set, _sig) != 0) sigdelset(_set, _sig);
    }

    if(sigismember(_set, SIGSEGV) != 0) sigdelset(_set, SIGSEGV);

    if(sigismember(_set, SIGABRT) != 0) sigdelset(_set, SIGABRT);
}
}  // namespace component
}  // namespace causal
}  // namespace rocprofsys
