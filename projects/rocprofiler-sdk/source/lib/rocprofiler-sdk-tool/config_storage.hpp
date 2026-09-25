// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace config_details
{
template <typename Tp>
class immutable_config_storage
{
public:
    explicit immutable_config_storage(Tp initial)
    {
        auto  value = std::make_unique<const Tp>(std::move(initial));
        auto* ptr   = value.get();

        m_generations.emplace_back(std::move(value));
        m_current.store(ptr, std::memory_order_relaxed);
    }

    immutable_config_storage(const immutable_config_storage&) = delete;
    immutable_config_storage(immutable_config_storage&&)      = delete;

    immutable_config_storage& operator=(const immutable_config_storage&) = delete;
    immutable_config_storage& operator=(immutable_config_storage&&) = delete;

    const Tp& get() const noexcept { return *m_current.load(std::memory_order_acquire); }

    const Tp& publish(Tp value)
    {
        auto  next = std::make_unique<const Tp>(std::move(value));
        auto* ptr  = next.get();

        auto lock = std::lock_guard<std::mutex>{m_mutex};
        m_generations.emplace_back(std::move(next));
        m_current.store(ptr, std::memory_order_release);
        return *ptr;
    }

    // Readers must be quiescent before this function is called. Published generations remain
    // alive until this explicit reclamation point so get() can return a reference without reader
    // locks or reference-counting overhead.
    void reclaim_retired()
    {
        auto lock = std::lock_guard<std::mutex>{m_mutex};
        if(m_generations.size() < 2) return;

        auto current = std::move(m_generations.back());
        m_generations.clear();
        m_generations.emplace_back(std::move(current));
    }

private:
    mutable std::mutex                     m_mutex       = {};
    std::vector<std::unique_ptr<const Tp>> m_generations = {};
    std::atomic<const Tp*>                 m_current     = {nullptr};
};
}  // namespace config_details
}  // namespace tool
}  // namespace rocprofiler
