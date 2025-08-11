// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include "counter.hpp"
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>
#include <map>
#include <atomic>
#include <future>
#include "workload.hpp"
#include "counter.hpp"

Collection::Collection(AgentInfo& agent, const std::vector<std::string>& counters)
    : packet(std::make_unique<AQLPacket>(agent, counters)) {}

Collection::~Collection() {}

std::map<std::string, int64_t> Collection::iterate(Queue& queue, IWorkload& load)
{
    start(queue);
    load.run();
    stop(queue);
    return packet->get();
}

void Collection::start(Queue& queue)
{
    assert(packet);
    queue.flush();
    queue.Submit(&packet->packets.start_packet);
}

void Collection::stop(Queue& queue)
{
    assert(packet);
    queue.flush();

    queue.Submit(&packet->packets.read_packet);
    queue.Submit(&packet->packets.stop_packet);
    packet->iterate();
}
