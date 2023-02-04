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

#include "library/causal/components/unblocking_gotcha.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "library/causal/delay.hpp"
#include "library/causal/experiment.hpp"
#include "library/runtime.hpp"

#include <timemory/components/macros.hpp>
#include <timemory/hash/types.hpp>

#include <csignal>
#include <cstdint>
#include <pthread.h>
#include <stdexcept>

namespace omnitrace
{
namespace causal
{
namespace component
{
std::string
unblocking_gotcha::label()
{
    return "causal_unblocking_gotcha";
}

std::string
unblocking_gotcha::description()
{
    return "Handles executing all necessary pauses before the thread performs some "
           "blocking function";
}

void
unblocking_gotcha::preinit()
{
    configure();
}

void
unblocking_gotcha::configure()
{
    unblocking_gotcha_t::get_initializer() = []() {
        if(!config::get_use_causal()) return;

        unblocking_gotcha_t::configure(
            comp::gotcha_config<0, int, pthread_mutex_t*>{ "pthread_mutex_unlock" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<1, int, pthread_rwlock_t*>{ "pthread_rwlock_unlock" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<2, int, pthread_spinlock_t*>{ "pthread_spin_unlock" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<3, int, pthread_barrier_t*>{ "pthread_barrier_wait" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<4, int, pthread_cond_t*>{ "pthread_cond_signal" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<5, int, pthread_cond_t*>{ "pthread_cond_broadcast" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<6, int, pthread_t, int>{ "pthread_kill" });

        unblocking_gotcha_t::configure(
            comp::gotcha_config<7, void, void*>{ "pthread_exit" });
    };
}

void
unblocking_gotcha::shutdown()
{
    unblocking_gotcha_t::disable();
}

void
unblocking_gotcha::start()
{
    if(causal::experiment::is_active() &&
       get_thread_state() == ::omnitrace::ThreadState::Enabled)
        causal::delay::process();
}

void
unblocking_gotcha::stop()
{
    if(causal::experiment::is_active() &&
       get_thread_state() == ::omnitrace::ThreadState::Enabled)
        causal::delay::credit();
}

void
unblocking_gotcha::set_data(const comp::gotcha_data& _data)
{
    OMNITRACE_SCOPED_THREAD_STATE(ThreadState::Internal);
    auto   _hash  = tim::add_hash_id(_data.tool_id);
    auto&& _ident = tim::get_hash_identifier(_hash);
    if(_ident != _data.tool_id)
        throw ::omnitrace::exception<std::runtime_error>(
            JOIN("", "Error! resolving hash for \"", _data.tool_id, "\" (", _hash,
                 ") returns ", _ident.c_str()));
#if defined(OMNITRACE_CI)
    OMNITRACE_VERBOSE_F(3, "data set for '%s'...\n", _data.tool_id.c_str());
#endif
}
}  // namespace component
}  // namespace causal
}  // namespace omnitrace

TIMEMORY_INVOKE_PREINIT(omnitrace::causal::component::unblocking_gotcha)
