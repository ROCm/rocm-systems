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

#include <mutex>

namespace rocprofiler
{
namespace common
{
// Parent-only ownership. Check fork abandonment before entering this object. Recursive ownership
// permits the SDK's terminal operation to dispatch its client finalizers on the same thread.
class session_lifecycle
{
public:
    class operation
    {
    public:
        operation(session_lifecycle& owner, bool terminal, bool internal, bool try_only = false)
        : m_lock{owner.m_mutex, std::defer_lock}
        , m_admitted{false}
        {
            if(try_only)
            {
                if(!m_lock.try_lock()) return;
            }
            else
                m_lock.lock();
            m_admitted = internal || !owner.m_terminal;
            if(terminal) owner.m_terminal = true;
        }
        explicit operator bool() const noexcept { return m_admitted; }

    private:
        std::unique_lock<std::recursive_mutex> m_lock;
        bool                                   m_admitted;
    };

    operation update() { return operation{*this, false, false}; }
    operation try_update() { return operation{*this, false, false, true}; }
    operation finalize() { return operation{*this, true, true}; }
    operation dispatch_finalizer() { return operation{*this, false, true}; }

private:
    std::recursive_mutex m_mutex{};
    bool                 m_terminal = false;
};
}  // namespace common
}  // namespace rocprofiler
