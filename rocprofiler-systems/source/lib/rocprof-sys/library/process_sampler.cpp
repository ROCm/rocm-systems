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

#include "library/process_sampler.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "library/amd_smi.hpp"
#include "library/cpu_freq.hpp"
#include "library/runtime.hpp"

#include <memory>
#include <vector>

namespace rocprofsys
{
namespace process_sampler
{
namespace
{
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

std::atomic<bool>&
get_sampler_is_sampling()
{
    static std::atomic<bool> _v{ false };
    return _v;
}
}  // namespace

void
sampler::poll(std::atomic<State>* _state, nsec_t _interval, promise_t* _ready)
{
    threading::offset_this_id(true);
    threading::set_thread_name("omni.sampler");

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    // notify thread started
    if(_ready) _ready->set_value();

    for(auto& itr : instances)
        itr->config();

    ROCPROFSYS_VERBOSE(
        1, "Background process sampling polling at an interval of %f seconds...\n",
        std::chrono::duration_cast<std::chrono::duration<double>>(_interval).count());

    auto _duration = config::get_process_sampling_duration();
    if(_duration < 0.0) _duration = config::get_sampling_duration();
    bool _has_duration = (_duration > 0.0);

    auto _now = std::chrono::steady_clock::now();
    auto _end =
        _now + std::chrono::nanoseconds{ static_cast<uint64_t>(_duration * units::sec) };
    while(_state && _state->load() < State::Finalized && get_state() < State::Finalized)
    {
        std::this_thread::sleep_until(_now);
        if(_state->load() != State::Active) continue;
        if(get_state() >= State::Finalized) break;
        if(get_state() != State::Active) continue;
        get_sampler_is_sampling().store(true);
        for(auto& itr : instances)
            itr->sample();
        get_sampler_is_sampling().store(false);
        if(_has_duration && _now >= _end) break;
        _now = std::chrono::steady_clock::now() + _interval;
    }

    // ensure this is always false
    get_sampler_is_sampling().store(false);

    if(_has_duration && _now >= _end && get_state() < State::Finalized)
    {
        ROCPROFSYS_VERBOSE(
            1,
            "Background process sampling duration of %f seconds has elapsed. "
            "Shutting down process sampling...\n",
            _duration);
    }

    ROCPROFSYS_CONDITIONAL_BASIC_PRINT(get_debug(),
                                       "Thread sampler polling completed...\n");

    if(polling_finished) polling_finished->set_value();
}

void
sampler::setup()
{
    if(!get_use_process_sampling())
    {
        ROCPROFSYS_DEBUG("Background sampler is disabled...\n");
        return;
    }

    ROCPROFSYS_VERBOSE(1, "Setting up background sampler...\n");

    // shutdown if already running
    shutdown();

    if(get_use_amd_smi())
    {
        auto& _amd_smi         = instances.emplace_back(std::make_unique<instance>());
        _amd_smi->setup        = []() { amd_smi::setup(); };
        _amd_smi->shutdown     = []() { amd_smi::shutdown(); };
        _amd_smi->post_process = []() { amd_smi::post_process(); };
        _amd_smi->config       = []() { amd_smi::config(); };
        _amd_smi->sample       = []() { amd_smi::sample(); };
    }

    if(get_cpu_freq_enabled())
    {
        auto& _cpu_freq         = instances.emplace_back(std::make_unique<instance>());
        _cpu_freq->setup        = []() { cpu_freq::setup(); };
        _cpu_freq->shutdown     = []() { cpu_freq::shutdown(); };
        _cpu_freq->post_process = []() { cpu_freq::post_process(); };
        _cpu_freq->config       = []() { cpu_freq::config(); };
        _cpu_freq->sample       = []() { cpu_freq::sample(); };
    }

    for(auto& itr : instances)
        itr->setup();

    polling_finished = std::make_unique<promise_t>();

    auto     _freq      = get_process_sampling_freq();
    uint64_t _msec_freq = (1.0 / _freq) * 1.0e3;

    polling_finished = std::make_unique<promise_t>();

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    set_state(State::PreInit);
    get_thread() = std::make_unique<std::thread>(&poll<msec_t>, &get_sampler_state(),
                                                 msec_t{ _msec_freq }, nullptr);

    set_state(State::Active);
}

void
sampler::shutdown()
{
    // set the local sampler state to finalized
    set_state(State::Finalized);

    // shutdown all components
    for(auto& itr : instances)
        itr->shutdown();

    auto& _thread = get_thread();
    if(_thread)
    {
        size_t           _nitr     = 0;
        constexpr size_t _nitr_max = 100;
        uint64_t         _freq     = (1.0 / get_process_sampling_freq()) * 1.0e3;

        // wait until the sampler is no longer sampling
        std::this_thread::sleep_for(msec_t{ _freq });
        while(get_sampler_is_sampling().load())
        {
            if(_nitr++ > _nitr_max) break;
        }

        // during CI, throw an error if polling_finished is not valid
        ROCPROFSYS_CI_THROW(!polling_finished, "polling_finished is not valid\n");
        if(polling_finished)
        {
            // wait for the thread to finish
            auto _fut = polling_finished->get_future();
            _fut.wait_for(msec_t{ 10 * _freq });
            _thread->join();
        }
        else
        {
            // cancel the thread and detach
            std::this_thread::sleep_for(msec_t{ 10 * _freq });
            pthread_cancel(_thread->native_handle());
            _thread->detach();
        }
        _thread          = std::unique_ptr<std::thread>{ nullptr };
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
}  // namespace process_sampler
}  // namespace rocprofsys
