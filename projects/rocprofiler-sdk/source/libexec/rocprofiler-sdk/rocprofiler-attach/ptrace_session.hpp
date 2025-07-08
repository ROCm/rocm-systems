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
    bool simple_mmap(void*& addr, size_t length);
    bool simple_munmap(void*& addr, size_t length);

    bool write(size_t addr, const std::vector<uint8_t>& data, size_t size);
    bool read(size_t addr, std::vector<uint8_t>& data, size_t size);
    bool swap(size_t                      addr,
              const std::vector<uint8_t>& in_data,
              std::vector<uint8_t>&       out_data,
              size_t                      size);

    int get_pid();

    bool call_function(const std::string& library, const std::string& symbol);
    bool call_function(const std::string& library, const std::string& symbol, void* first);

    bool stop();
    bool cont();
    bool handle_signals();
    void detach_ptrace_session();

private:
    bool find_library(void*& addr, int inpid, const std::string& library);
    bool find_symbol(void*& addr, const std::string& library, const std::string& symbol);

    std::unordered_map<std::string, void*> target_library_addrs;
    std::unordered_map<std::string, void*> target_symbol_addrs;

    const int         pid;
    bool              attached;
    std::atomic<bool> detaching_ptrace_session;
};

}  // namespace attach
}  // namespace rocprofiler
