g / MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler-sdk/fwd.h>
#include <functional>
#include <unordered_map>
#include <vector>
#include "synchronized.hpp"

    namespace rocprofiler
{
    namespace common
    {
    template <typename StateT, typename TransitionT>
    class state_machine
    {
    public:
        using state_type      = StateT;
        using transition_type = TransitionT;
        using transitions_vec = std::vector<transition_type>;
        using transition_key  = std::pair<state_type, state_type>;

        struct transition_key_hash
        {
            std::size_t operator()(const transition_key& k) const
            {
                auto h1 = std::hash<state_type>{}(k.first);
                auto h2 = std::hash<state_type>{}(k.second);
                return h1 ^ (h2 << 1);
            }
        };

        using transition_map =
            std::unordered_map<transition_key, transition_type, transition_key_hash>;

        state_machine(state_type initial_state, transitions_vec transitions)
        : m_current_state(std::move(initial_state))
        {
            for(auto&& trans : transitions)
            {
                m_transitions[{trans.from_state, trans.to_state}] = std::move(trans);
            }
        }

        rocprofiler_status_t transition_to(state_type new_state)
        {
            return m_current_state.wlock(
                [this](state_type& current_state, state_type target_state) -> rocprofiler_status_t {
                    auto it = m_transitions.find({current_state, target_state});

                    if(it == m_transitions.end())
                    {
                        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
                    }

                    if(it->second.transition_func)
                    {
                        auto status = it->second.transition_func();
                        if(status != ROCPROFILER_STATUS_SUCCESS)
                        {
                            return status;
                        }
                    }

                    current_state = target_state;
                    return ROCPROFILER_STATUS_SUCCESS;
                },
                new_state);
        }

        state_type get_current_state() const
        {
            return m_current_state.rlock([](const state_type& state) { return state; });
        }

        bool is_valid_transition(state_type new_state) const
        {
            return m_current_state.rlock(
                [this](const state_type& current_state, state_type target_state) {
                    return m_transitions.find({current_state, target_state}) != m_transitions.end();
                },
                new_state);
        }

    private:
        Synchronized<state_type> m_current_state;
        transition_map           m_transitions;
    };

    }  // namespace common
}  // namespace rocprofiler
