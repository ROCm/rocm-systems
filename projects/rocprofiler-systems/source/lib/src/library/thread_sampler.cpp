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

#include "library/thread_sampler.hpp"
#include "library/components/rocm_smi.hpp"
#include "library/config.hpp"
#include "library/cpu_freq.hpp"
#include "library/debug.hpp"

#include <memory>
#include <vector>

namespace omnitrace
{
namespace thread_sampler
{
namespace
{
using tim::type_mutex;
using auto_lock_t                                       = tim::auto_lock_t;
using promise_t                                         = std::promise<void>;
std::unique_ptr<promise_t>             polling_finished = {};
std::vector<std::unique_ptr<instance>> instances        = {};

bool&
is_initialized()
{
    static bool _v = false;
    return _v;
}

std::unique_ptr<std::thread>&
get_thread()
{
    static std::unique_ptr<std::thread> _v;
    return _v;
}

std::atomic<State>&
get_sampler_state()
{
    static std::atomic<State> _v{ State::PreInit };
    return _v;
}
}  // namespace

void
sampler::poll(std::atomic<State>* _state, nsec_t _interval, promise_t* _ready)
{
    threading::set_thread_name("omni.sampler");

    // notify thread started
    if(_ready) _ready->set_value();

    for(auto& itr : instances)
        itr->config();

    OMNITRACE_CONDITIONAL_BASIC_PRINT(
        get_verbose() > 0 || get_debug(),
        "Thread sampler polling at an interval of %f seconds...\n",
        std::chrono::duration_cast<std::chrono::duration<double>>(_interval).count());

    auto _now = std::chrono::steady_clock::now();
    while(_state && _state->load() != State::Finalized && get_state() != State::Finalized)
    {
        std::this_thread::sleep_until(_now);
        if(_state->load() != State::Active) continue;
        for(auto& itr : instances)
            itr->sample();
        while(_now < std::chrono::steady_clock::now())
            _now += _interval;
    }

    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug(),
                                      "Thread sampler polling completed...\n");

    if(polling_finished) polling_finished->set_value();
}

void
sampler::setup()
{
    OMNITRACE_VERBOSE(1, "Setting up background sampler...\n");

    // shutdown if already running
    shutdown();

    auto _enable_samp = pthread_gotcha::enable_sampling_on_child_threads();
    pthread_gotcha::enable_sampling_on_child_threads() = false;

    if(get_use_rocm_smi())
    {
        auto& _rocm_smi         = instances.emplace_back(std::make_unique<instance>());
        _rocm_smi->setup        = []() { rocm_smi::setup(); };
        _rocm_smi->shutdown     = []() { rocm_smi::shutdown(); };
        _rocm_smi->post_process = []() { rocm_smi::post_process(); };
        _rocm_smi->config       = []() { rocm_smi::config(); };
        _rocm_smi->sample       = []() { rocm_smi::sample(); };
    }

    auto& _cpu_freq         = instances.emplace_back(std::make_unique<instance>());
    _cpu_freq->setup        = []() { cpu_freq::setup(); };
    _cpu_freq->shutdown     = []() { cpu_freq::shutdown(); };
    _cpu_freq->post_process = []() { cpu_freq::post_process(); };
    _cpu_freq->config       = []() { cpu_freq::config(); };
    _cpu_freq->sample       = []() { cpu_freq::sample(); };

    for(auto& itr : instances)
        itr->setup();

    polling_finished = std::make_unique<promise_t>();

    auto     _freq      = get_thread_sampling_freq();
    uint64_t _msec_freq = (1.0 / _freq) * 1.0e3;

    promise_t _prom{};
    auto      _fut   = _prom.get_future();
    polling_finished = std::make_unique<promise_t>();

    set_state(State::PreInit);
    get_thread() = std::make_unique<std::thread>(&poll<msec_t>, &get_sampler_state(),
                                                 msec_t{ _msec_freq }, &_prom);

    _fut.wait();

    pthread_gotcha::enable_sampling_on_child_threads() = _enable_samp;
    set_state(State::Active);
}

void
sampler::shutdown()
{
    for(auto& itr : instances)
        itr->shutdown();

    auto& _thread = get_thread();
    if(_thread)
    {
        OMNITRACE_VERBOSE(1, "Shutting down background sampler...\n");
        set_state(State::Finalized);
        if(polling_finished)
        {
            auto     _fut  = polling_finished->get_future();
            uint64_t _freq = (1.0 / get_thread_sampling_freq()) * 1.0e3;
            _fut.wait_for(msec_t{ 5 * _freq });
            _thread->join();
        }
        else
        {
            uint64_t _freq = (1.0 / get_thread_sampling_freq()) * 1.0e3;
            std::this_thread::sleep_for(msec_t{ 5 * _freq });
            pthread_cancel(_thread->native_handle());
            _thread->detach();
        }
        _thread          = std::unique_ptr<std::thread>{};
        polling_finished = std::unique_ptr<promise_t>{};
    }

    is_initialized() = false;
}

void
sampler::post_process()
{
    for(auto& itr : instances)
        itr->post_process();

    instances.clear();
}

void
sampler::set_state(state_t _state)
{
    get_sampler_state().store(_state);
}
}  // namespace thread_sampler
}  // namespace omnitrace
