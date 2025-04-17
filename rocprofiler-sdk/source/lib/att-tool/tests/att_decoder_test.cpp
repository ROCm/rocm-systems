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

#include "lib/att-tool/att_lib_wrapper.hpp"
#include "lib/att-tool/code.hpp"
#include "lib/att-tool/outputfile.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <gtest/gtest.h>
#include <iostream>

namespace rocprofiler
{
namespace att_wrapper
{
class ATTDecoderTest : public ATTDecoder
{
public:
    ATTDecoderTest()
    : ATTDecoder(rocprofiler::att_wrapper::tool_att_capability_t::ATT_CAPABILITIES_TESTING1)
    {
        rocprofiler::att_wrapper::OutputFile::Enabled() = false;
        GlobalDefs::get().output_formats                = "json,csv";
        registration::init_logging();
    };

    void test_parse()
    {
        ATTFileMgr mgr("out/", dl, {});

        auto append_isa = [&](const char* line) {
            // matches addresses in dummy_decoder.cpp
            pcinfo_t pc{};
            pc.addr      = mgr.codefile->isa_map.size() * 8;
            pc.marker_id = 1;

            auto code             = std::make_unique<CodeLine>();
            code->code_line       = std::make_shared<CodeLine::Instruction>();
            code->code_line->inst = line;
            code->line_number     = mgr.codefile->isa_map.size();

            mgr.codefile->isa_map.emplace(pc, std::move(code));
        };

        mgr.codefile->kernel_names[pcinfo_t{}] = KernelName{"_Kernel", "Kernel"};

        append_isa("s_load_");
        append_isa("s_store_");
        append_isa("s_waitcnt vmcnt(0) lgkmcnt(0)");

        std::vector<char> dummy_data;
        dummy_data.resize(128);

        mgr.parseShader(0, dummy_data);
        mgr.parseShader(1, dummy_data);
    }
};

TEST(att_decoder_test, dlopen)
{
    registration::init_logging();
    auto query = query_att_decode_capability();
    ROCP_FATAL_IF(query.empty()) << "No decoder capability available!";
}

TEST(att_decoder_test, filewrite)
{
    ATTDecoderTest decoder;
    ROCP_FATAL_IF(!decoder.valid()) << "Failed to initialize decoder library!";

    decoder.test_parse();
}

TEST(att_decoder_test, warn_failures)
{
    std::vector<CodeobjLoadInfo> codeobjs;
    codeobjs.resize(5);
    codeobjs.at(0).name = "memory://unknown";
    codeobjs.at(1).name = "memory://unknown&offset=123&size=123";
    codeobjs.at(2).name = "file://nofile";
    codeobjs.at(3).name = "file://nofile&offset=123&size=123";
    codeobjs.at(4).name = "myfile123.out";

    std::vector<std::string> att_files;
    att_files.emplace_back("file123.att");

    ATTDecoderTest decoder;
    ROCP_FATAL_IF(!decoder.valid()) << "Failed to initialize decoder library!";

    decoder.parse(".", ".", att_files, codeobjs, {}, "csv,json");
}

TEST(att_decoder_test, code_write)
{
    registration::init_logging();
    rocprofiler::att_wrapper::OutputFile::Enabled() = false;
    GlobalDefs::get().output_formats                = "json,csv";

    CodeFile file{};

    pcinfo_t addr{};
    addr.marker_id          = 0;
    addr.addr               = 0x1000;
    file.kernel_names[addr] = KernelName{"_Kernel", "Kernel"};

    for(size_t i = 0; i < 4; i++)
    {
        auto line         = std::make_unique<CodeLine>();
        line->line_number = i;

        line->code_line       = std::make_shared<CodeLine::Instruction>();
        line->code_line->inst = "v_add";
        file.isa_map[addr]    = std::move(line);
    }
}

};  // namespace att_wrapper
};  // namespace rocprofiler
