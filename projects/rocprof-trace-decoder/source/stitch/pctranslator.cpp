// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <string>
#include <string_view>
#include <vector>

#include "stitch.hpp"
#include "trace_parser.hpp"
#include "trie.h"

PCTranslator::PCTranslator(std::vector<assemblyLinePtr>& _code, std::shared_ptr<ICodeServicer>& _service, int _gfxip) :
service(_service), code(_code), gfxip(_gfxip)
{
    std::unique_lock<SharedMutex> lk(jump_mut);
    {
        std::unique_lock<SharedMutex> lk(code_mut);
        for (auto& c : code) addrmap[c->addr] = c;
    }
}

assemblyLinePtr PCTranslator::jump(const assemblyLine& source)
{
    {
        std::shared_lock<SharedMutex> lk(jump_mut);
        if (jump_map.find(source.addr) != jump_map.end()) return jump_map.at(source.addr);
    }

    std::unique_lock<SharedMutex> lk(jump_mut);
    return jump_map.emplace(source.addr, getcode(getjump_loc(source))).first->second;
}

assemblyLinePtr PCTranslator::getcode(pcinfo_t addr)
{
    {
        std::shared_lock<SharedMutex> lk(code_mut);
        if (addrmap.find(addr) != addrmap.end()) return addrmap.at(addr);
    }

    std::unique_lock<SharedMutex> lk(code_mut);

    assemblyLine newline = service->GetInstruction(addr, gfxip);

    if (newline.cat == InstCategory::BRANCH) newline.to_line = getjump_loc(newline);

    auto line = std::make_shared<assemblyLine>(std::move(newline));
    code.push_back(line);
    addrmap[addr] = line;
    return line;
}

pcinfo_t PCTranslator::getjump_loc(const assemblyLine& line)
{
    int64_t delta = std::stoi(splitv(line.line, ' ').back().data());
    if (delta >= 32768) delta -= 65536;
    return {line.addr.address + 4 + 4 * delta, line.addr.code_object_id};
}
