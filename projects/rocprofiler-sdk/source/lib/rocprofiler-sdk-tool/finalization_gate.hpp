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
//

#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>

namespace rocprofiler
{
namespace tool
{
// Runs rocprofv3's finalization exactly once, and makes every other caller wait until it has
// finished rather than merely started. Finalization is requested from several threads: the
// signal worker after SIGINT/SIGTERM, rocprofv3_main once main() returns, and atexit. A caller
// that returns as soon as another thread has started lets the process exit while that thread is
// still writing output (e.g. mid ATT decode), truncating it or crashing in static destructors.
class finalization_gate
{
public:
    enum class status
    {
        ran,        // this call ran the finalization
        completed,  // another thread ran it, and it has finished
        reentered,  // called from the finalizing thread itself, e.g. exit() during finalization
        timed_out,  // another thread was still finalizing when the wait timed out
    };

    // Runs `func` unless finalization has already started; otherwise waits for it to finish,
    // for up to `timeout_sec` seconds (<= 0 waits indefinitely).
    status run_or_wait(const std::function<void()>& func, int timeout_sec);

private:
    std::atomic_flag             m_started   = ATOMIC_FLAG_INIT;
    std::atomic<uint32_t>        m_complete  = {0};  // futex word
    std::atomic<std::thread::id> m_finalizer = {};
};
}  // namespace tool
}  // namespace rocprofiler
