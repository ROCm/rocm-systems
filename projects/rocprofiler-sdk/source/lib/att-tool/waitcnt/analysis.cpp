// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/att-tool/waitcnt/analysis.hpp"

#include <map>

namespace rocprofiler
{
namespace att_wrapper
{
int64_t
MemoryCounter::extract_waitcnt(const std::string& str) const
{
    // Find the position of the number in the string
    size_t pos  = str.find(name);
    size_t iter = pos + name.size() + 1;

    if(pos == std::string::npos || iter >= str.size()) return 0;

    // If no hex preamble found, assume it's decimal
    if(str.find("0x") == std::string::npos) return atoi(str.data() + iter);

    // Skip any additional spaces
    while(iter < str.size() && str.at(iter) == ' ')
        iter++;

    // Check if the number is in hexadecimal format (starts with "0x"), otherwise return decimal
    if(iter + 2 < str.size() && str.at(iter) == '0' && str.at(iter + 1) == 'x')
        return std::stoi(str.substr(iter + 2), nullptr, 16);
    else if(iter < str.size())
        return atoi(str.data() + iter);
    else
        return 0;
}

std::vector<int>
MemoryCounter::join_and_reset(int64_t offset, std::vector<int>& flats)
{
    std::vector<int> ret = std::move(flats);
    flats.clear();

    offset = std::max(std::min<int64_t>(offset, list.size()), 0l);
    ret.insert(ret.end(), list.begin(), list.begin() + offset);
    list.erase(list.begin(), list.begin() + offset);
    return ret;
}

std::optional<std::vector<int>>
MemoryCounter::handle_mem_op(const std::string& inst, std::vector<int>& flat_list)
{
    int64_t wait_n = extract_waitcnt(inst);

    if(wait_n == 0) order = Ordering::MEMORY_SEQUENTIAL;

    if(order == Ordering::MEMORY_SEQUENTIAL)
    {
        auto joined = join_and_reset(list.size() - wait_n, flat_list);
        if(!joined.empty()) return joined;
    }
    return std::nullopt;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
