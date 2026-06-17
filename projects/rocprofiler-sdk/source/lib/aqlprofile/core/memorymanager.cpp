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
    for(size_t i = 0; i < count; i++)
    {
        events.push_back(EventRequest{_events[i], false});
        num_flag_metrics += _events[i].flags.raw != 0;
    }

    if(!num_flag_metrics) return;

    std::sort(events.begin(), events.end());

    std::vector<EventRequest> acc_requests;
    for(auto it = events.begin(); it != events.end(); it++)
    {
        if(!it->flags.raw) continue;

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
