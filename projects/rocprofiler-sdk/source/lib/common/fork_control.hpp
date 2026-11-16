// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#pragma once

#include <atomic>

namespace rocprofiler
{
namespace common
{
enum class fork_rebuild_phase
{
    core,
    auxiliary,
};

// Internal SDK/tool protocol, not an installed client API. Nodes have process lifetime and are
// published in ordinary parent execution. Child traversal cannot allocate or acquire a mutex.
struct fork_participant
{
    void (*prepare)(bool sdk_admitted);
    void (*parent)() noexcept;
    bool (*eligible)() noexcept;
    void (*abandon)() noexcept;
    void (*rebuild)();
    fork_participant* next           = nullptr;
    void (*begin_child)() noexcept   = nullptr;
    fork_rebuild_phase rebuild_phase = fork_rebuild_phase::auxiliary;
};

class fork_dispatcher
{
public:
    void add(fork_participant& node) noexcept
    {
        node.next = m_head.load(std::memory_order_acquire);
        while(!m_head.compare_exchange_weak(
            node.next, &node, std::memory_order_release, std::memory_order_acquire))
        {}
    }
    void prepare(bool sdk_admitted) const
    {
        for(auto* node = m_head.load(std::memory_order_acquire); node; node = node->next)
            node->prepare(sdk_admitted);
    }
    void parent() const noexcept
    {
        for(auto* node = m_head.load(std::memory_order_acquire); node; node = node->next)
            node->parent();
    }
    void begin_child() const noexcept
    {
        for(auto* node = m_head.load(std::memory_order_acquire); node; node = node->next)
            if(node->begin_child) node->begin_child();
    }
    bool eligible() const noexcept
    {
        bool valid = true;
        for(auto* node = m_head.load(std::memory_order_acquire); node; node = node->next)
            valid = node->eligible() && valid;
        return valid;
    }
    void abandon() const noexcept
    {
        for(auto* node = m_head.load(std::memory_order_acquire); node; node = node->next)
            node->abandon();
    }
    void rebuild(fork_rebuild_phase phase) const
    {
        for(auto* node = m_head.load(std::memory_order_acquire); node; node = node->next)
            if(node->rebuild_phase == phase) node->rebuild();
    }
    void rebuild() const
    {
        rebuild(fork_rebuild_phase::core);
        rebuild(fork_rebuild_phase::auxiliary);
    }

private:
    std::atomic<fork_participant*> m_head{nullptr};
    static_assert(std::atomic<fork_participant*>::is_always_lock_free);
};
}  // namespace common
}  // namespace rocprofiler
