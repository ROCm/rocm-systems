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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <fstream>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include <string_view>
#include <vector>

#ifndef CODEOBJ_BINARY_DIR
static_assert(false && "Please define CODEOBJ_BINARY_DIR to codeobj tests binary, "
                       "e.g. ../source/lib/tests/codeobj/");
#endif

namespace rocprofiler
{
namespace testing
{
namespace codeobjhelper
{
std::string
removeNull(std::string_view s)
{
    std::string u(s);
    while(u.find("null") != std::string::npos)
        u = u.substr(0, u.find("null")) + "0x0" + u.substr(u.find("null") + 4);
    return u;
}

static const std::vector<std::string>&
GetHipccOutput()
{
    static std::vector<std::string> result = []() {
        std::ifstream            file(CODEOBJ_BINARY_DIR "hipcc_output.s");
        std::vector<std::string> ret;

        while(file.good())
        {
            std::string s;
            getline(file, s);
            ret.push_back(removeNull(s));
        }
        return ret;
    }();
    return result;
}

static const std::vector<char>&
GetCodeobjContents()
{
    static std::vector<char> buffer = []() {
        std::string   filename = CODEOBJ_BINARY_DIR "smallkernel.bin";
        std::ifstream file(filename.data(), std::ios::binary);

        using iterator_t = std::istreambuf_iterator<char>;
        return std::vector<char>(iterator_t(file), iterator_t());
    }();
    return buffer;
}

}  // namespace codeobjhelper
}  // namespace testing
}  // namespace rocprofiler

TEST(codeobj_library, segment_test)
{
    using CodeobjTableTranslator = rocprofiler::sdk::codeobj::segment::CodeobjTableTranslator;

    CodeobjTableTranslator     table;
    std::unordered_set<size_t> used_addr{};

    for(size_t ITER = 0; ITER < 50; ITER++)
    {
        for(int j = 0; j < 2500; j++)
        {
            size_t addr = rand() % 10000000;
            size_t size = 1;
            if(used_addr.find(addr) != used_addr.end()) continue;
            used_addr.insert(addr);
            table.insert({addr, size, 0});
        }

        ASSERT_NE(table.begin(), table.end());
        {
            auto it = std::next(table.begin());
            while(it != table.end())
            {
                ASSERT_LT(*std::prev(it), *it);
                it++;
            }
        }

        std::vector<size_t> addr_leftover(used_addr.begin(), used_addr.end());
        for(size_t i = 0; i < 2400; i++)
        {
            size_t idx  = rand() % addr_leftover.size();
            auto   addr = addr_leftover.at(idx);
            ASSERT_EQ(table.remove(addr), true);
            addr_leftover.erase(addr_leftover.begin() + idx);
            used_addr.erase(addr);
        }
    }
}

namespace disassembly         = rocprofiler::sdk::codeobj::disassembly;
namespace codeobjhelper       = rocprofiler::testing::codeobjhelper;
using CodeobjDecoderComponent = rocprofiler::sdk::codeobj::disassembly::CodeobjDecoderComponent;
using LoadedCodeobjDecoder    = rocprofiler::sdk::codeobj::disassembly::LoadedCodeobjDecoder;

TEST(codeobj_library, file_opens)
{
    ASSERT_NE(codeobjhelper::GetHipccOutput().size(), 0);
    ASSERT_NE(codeobjhelper::GetCodeobjContents().size(), 0);
}

TEST(codeobj_library, decoder_component)
{
    const std::vector<std::string>& hiplines      = codeobjhelper::GetHipccOutput();
    const std::vector<char>&        objdata       = codeobjhelper::GetCodeobjContents();
    constexpr size_t                loaded_offset = 0x3000;

    CodeobjDecoderComponent component(objdata.data(), objdata.size());

    std::string          kernel_with_protocol = "file://" CODEOBJ_BINARY_DIR "smallkernel.bin";
    LoadedCodeobjDecoder loadecomp(kernel_with_protocol.data(), loaded_offset, objdata.size());

    ASSERT_EQ(component.m_symbol_map.size(), 1);

    for(auto& [kaddr, symbol] : component.m_symbol_map)
    {
        ASSERT_NE(symbol.name.find("reproducible_runtime"), std::string::npos);
        ASSERT_NE(symbol.mem_size, 0);

        size_t it    = 0;
        size_t vaddr = kaddr;
        while(vaddr < kaddr + symbol.mem_size)
        {
            if(!component.va2fo(vaddr))
            {
                ASSERT_NE(0, 0);
            }

            uint64_t faddr = *component.va2fo(vaddr);
            ASSERT_EQ(faddr - symbol.faddr, vaddr - kaddr);

            auto instruction        = component.disassemble_instruction(faddr, vaddr);
            auto loaded_instruction = loadecomp.get(vaddr + loaded_offset);

            ASSERT_NE(codeobjhelper::removeNull(instruction->inst).find(hiplines.at(it)),
                      std::string::npos);
            ASSERT_EQ(instruction->inst, loaded_instruction->inst);
            vaddr += instruction->size;
            it++;
        }
    }
}

TEST(codeobj_library, loaded_codeobj_component)
{
    const std::vector<char>& objdata = rocprofiler::testing::codeobjhelper::GetCodeobjContents();
    constexpr size_t         offset  = 0x1000;
    constexpr size_t         memsize = 0x1000;

    LoadedCodeobjDecoder decoder((const void*) objdata.data(), objdata.size(), offset, memsize);

    for(auto& [kaddr, symbol] : decoder.getSymbolMap())
    {
        ASSERT_NE(symbol.name.find("reproducible_runtime"), std::string::npos);
        ASSERT_NE(symbol.mem_size, 0);
    }
}

TEST(codeobj_library, codeobj_map_test)
{
    using marker_id_t = rocprofiler::sdk::codeobj::segment::marker_id_t;

    const std::vector<char>& objdata = rocprofiler::testing::codeobjhelper::GetCodeobjContents();
    constexpr size_t         laddr1  = 0x1000;
    constexpr size_t         laddr3  = 0x3000;

    uint64_t kaddr = [&objdata]() {
        CodeobjDecoderComponent comp(objdata.data(), objdata.size());
        for(auto& [addr, _] : comp.m_symbol_map)
            return addr;
        return 0ul;
    }();

    EXPECT_NE(kaddr, 0);

    disassembly::CodeobjMap map;
    const void*             objdataptr = (const void*) objdata.data();
    map.addDecoder(objdataptr, objdata.size(), marker_id_t{1}, laddr1, objdata.size());
    map.addDecoder(objdataptr, objdata.size(), marker_id_t{3}, laddr3, objdata.size());

    EXPECT_EQ(map.get(marker_id_t{1}, kaddr)->inst, map.get(marker_id_t{3}, kaddr)->inst);

    ASSERT_EQ(map.removeDecoderbyId(1), true);
    ASSERT_EQ(map.removeDecoderbyId(3), true);
    ASSERT_EQ(map.removeDecoderbyId(1), false);
}

TEST(codeobj_library, codeobj_table_test)
{
    using marker_id_t = rocprofiler::sdk::codeobj::segment::marker_id_t;

    const std::vector<std::string>& hiplines = codeobjhelper::GetHipccOutput();
    const std::vector<char>&        objdata  = codeobjhelper::GetCodeobjContents();
    constexpr size_t                laddr1   = 0x1000;
    constexpr size_t                laddr3   = 0x3000;

    disassembly::CodeobjAddressTranslate map;

    uint64_t kaddr = 0, memsize = 0;
    std::tie(kaddr, memsize) = [&objdata]() {
        CodeobjDecoderComponent comp(objdata.data(), objdata.size());
        for(auto& [addr, symbol] : comp.m_symbol_map)
            return std::pair<uint64_t, uint64_t>(addr, symbol.mem_size);
        return std::pair<uint64_t, uint64_t>(0, 0);
    }();
    ASSERT_NE(kaddr, 0);
    ASSERT_NE(memsize, 0);

    map.addDecoder((const void*) objdata.data(), objdata.size(), marker_id_t{1}, laddr1, 0x2000);
    map.addDecoder((const void*) objdata.data(), objdata.size(), marker_id_t{3}, laddr3, 0x2000);

    EXPECT_NE(map.get(laddr1 + kaddr).get(), nullptr);
    EXPECT_NE(map.get(laddr3 + kaddr).get(), nullptr);
    EXPECT_EQ(map.get(laddr1 + kaddr)->inst, map.get(laddr3 + kaddr)->inst);

    size_t it    = 0;
    size_t vaddr = kaddr;
    while(vaddr < kaddr + memsize)
    {
        auto instruction = map.get(laddr1 + vaddr);
        ASSERT_NE(codeobjhelper::removeNull(instruction->inst).find(hiplines.at(it)),
                  std::string::npos);
        vaddr += instruction->size;
        it++;
    }

    ASSERT_EQ(map.removeDecoderbyId(1), true);
    ASSERT_EQ(map.removeDecoderbyId(3), true);
    ASSERT_EQ(map.removeDecoderbyId(1), false);
}
