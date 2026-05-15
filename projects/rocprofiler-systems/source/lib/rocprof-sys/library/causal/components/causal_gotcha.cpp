// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
