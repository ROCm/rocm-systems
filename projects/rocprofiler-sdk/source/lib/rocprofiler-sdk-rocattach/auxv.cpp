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

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"

#include "auxv.hpp"

#include <vector>

namespace fs = rocprofiler::common::filesystem;

namespace rocprofiler
{
namespace rocattach
{
// When injecting assembly into a process, existing instructions are partially or completely
// overwritten. Other threads will continue running and may execute our injected code, resulting in
// illegal instructions, segmentation faults, or worse. A common technique to avoid this is to
// inject code at the program entry address, as this is extremely unlikely to be called again in a
// multithreaded process. To determine this address, we inspect the auxv file for the target
// process.

// Each entry in the auxv file will match this format.
typedef struct
{
    uint64_t type;
    void*    value;
} auxv_pair_t;

rocattach_status_t
get_auxv_entry(int pid, void*& entry_addr)
{
    constexpr int AT_ENTRY       = 9;  // Type number for program entry point
    constexpr int auxv_pair_size = sizeof(auxv_pair_t);

    auto          filename = fs::path{"/proc"} / std::to_string(pid) / "auxv";
    std::ifstream auxv(filename, std::ios::in | std::ios::binary);

    if(!auxv.is_open())
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Unable to open auxv file " << filename;
        return ROCATTACH_STATUS_ERROR;
    }

    std::vector<char> auxv_buffer(auxv_pair_size);

    entry_addr = nullptr;
    while(auxv.read(auxv_buffer.data(), auxv_pair_size) && entry_addr == nullptr)
    {
        auxv_pair_t* const aux = reinterpret_cast<auxv_pair_t*>(auxv_buffer.data());

        if(aux->type == AT_ENTRY)
        {
            entry_addr = aux->value;
        }
    }

    if(entry_addr == nullptr)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Unexpected mising AT_ENTRY for " << filename;
        return ROCATTACH_STATUS_ERROR;
    }
    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Entry address found to be " << entry_addr << " from "
               << filename;
    return ROCATTACH_STATUS_SUCCESS;
}
}  // namespace rocattach
}  // namespace rocprofiler
