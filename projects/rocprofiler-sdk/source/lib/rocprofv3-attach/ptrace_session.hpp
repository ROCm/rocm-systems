// MIT License
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

#include <rocprofiler-sdk/rocprofiler.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace attach
{
class PTraceSession
{
public:
    explicit PTraceSession(int);
    ~PTraceSession();

    bool attach();
    bool detach();
    bool simple_mmap(void*& addr, size_t length) const;
    bool simple_munmap(void*& addr, size_t length) const;

    bool write(size_t addr, const std::vector<uint8_t>& data, size_t size) const;
    bool read(size_t addr, std::vector<uint8_t>& data, size_t size) const;
    bool swap(size_t                      addr,
              const std::vector<uint8_t>& in_data,
              std::vector<uint8_t>&       out_data,
              size_t                      size) const;

    int get_pid() const { return m_pid; }

    bool call_function(const std::string& library, const std::string& symbol);
    bool call_function(const std::string& library, const std::string& symbol, void* first);
    bool call_function(const std::string& library,
                       const std::string& symbol,
                       void*              first,
                       void*              second);

    bool stop() const;
    bool cont() const;
    bool handle_signals() const;
    void detach_ptrace_session();

    std::atomic<rocprofiler_status_t> m_setup_status = ROCPROFILER_STATUS_SUCCESS;

private:
    static bool find_library(void*& addr, int inpid, const std::string& library);
    bool        find_symbol(void*& addr, const std::string& library, const std::string& symbol);

    std::unordered_map<std::string, void*> m_target_library_addrs = {};
    std::unordered_map<std::string, void*> m_target_symbol_addrs  = {};

    const int         m_pid                      = -1;
    bool              m_attached                 = false;
    std::atomic<bool> m_detaching_ptrace_session = false;
};

}  // namespace attach
}  // namespace rocprofiler
