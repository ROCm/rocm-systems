// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/aqlprofile/core/memorymanager.hpp"
#include <algorithm>
#include "lib/common/static_object.hpp"

std::atomic<size_t>&
MemoryManager::get_handle_counter()
{
    static auto _v = std::atomic<size_t>{1};
    return _v;
}

MemoryManager::memory_manager_synced_map_t*
MemoryManager::get_managers()
{
    static auto*& _v = rocprofiler::common::static_object<memory_manager_synced_map_t>::construct();
    return _v;
}

void
CounterMemoryManager::CopyEvents(const aqlprofile_pmc_event_t* _events, size_t count)
{
    events.reserve(count + 4);
    int num_flag_metrics = 0;
    int num_sp_events    = 0;
    int num_sq_events    = 0;
    for(size_t i = 0; i < count; i++)
    {
        events.push_back(EventRequest{_events[i], false});
        // This logic is specific to SQ accumulate
        bool bIsSq    = _events[i].block_name == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
        bool bIsSQG   = _events[i].block_name == static_cast<int>(AQLPROFILE_BLOCK_NAME_SQG);
        bool bIsAccum = _events[i].flags.sq_flags.accum != 0;

        if(bIsAccum && bIsSq) events.back().bShouldBeLast = true;

        num_flag_metrics += bIsAccum && (bIsSq || bIsSQG);
        num_sq_events += bIsSq && !bIsAccum;
        num_sp_events += _events[i].block_name == static_cast<int>(AQLPROFILE_BLOCK_NAME_SP);
    }

    if(!num_flag_metrics) return;

    EventRequest dummySqEvent{};
    dummySqEvent.bInternal  = true;
    dummySqEvent.block_name = HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;

    // Sorting will put these as the first SQ events, pushing back the level counters after them
    // This will make accumulate always be after all SP events in the register programming.
    int extra_dummy_sq = num_sp_events - num_sq_events - 1;
    for(int i = 0; i < extra_dummy_sq; i++)
        events.push_back(dummySqEvent);

    std::stable_sort(events.begin(), events.end());

    std::vector<EventRequest> acc_requests;
    for(auto it = events.begin(); it != events.end(); it++)
    {
        bool IsSq  = it->block_name == HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ;
        bool IsSQG = it->block_name == static_cast<int>(AQLPROFILE_BLOCK_NAME_SQG);

        if(it->flags.sq_flags.accum == 0 || !(IsSq || IsSQG)) continue;

        if(it != events.begin())
        {
            auto prev = std::prev(it);
            if(it->IsSameNoFlags(*prev) && (!prev->flags.raw || prev->bInternal)) continue;
        }

        EventRequest req = *it;
        req.bInternal    = true;
        req.flags.raw    = 0;
        acc_requests.push_back(req);
    }

    if(!acc_requests.size()) return;

    events.insert(events.end(), acc_requests.begin(), acc_requests.end());
    std::sort(events.begin(), events.end());
}
