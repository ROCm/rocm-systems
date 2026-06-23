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

#include <gtest/gtest.h>
#include <unistd.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include <sstream>
#include <string_view>
#include <vector>
#include "lib/common/logging.hpp"

#include "lib/common/filesystem.hpp"

#ifndef CODEOBJ_BINARY_DIR
static_assert(false && "Please define CODEOBJ_BINARY_DIR to codeobj tests binary, "
                       "e.g. ../source/lib/tests/codeobj/");
#endif

#ifndef CODEOBJ_INSTALL_DIR
static_assert(false && "Please define CODEOBJ_INSTALL_DIR to the installed tests bin directory "
                       "(e.g. <prefix>/share/rocprofiler-sdk/tests/unit-tests/bin/)");
#endif

namespace rocprofiler
{
namespace testing
{
namespace codeobjhelper
{
namespace fs = common::filesystem;

std::string
removeNull(std::string_view s)
{
    std::string u(s);
    while(u.find("null") != std::string::npos)
        u = u.substr(0, u.find("null")) + "0x0" + u.substr(u.find("null") + 4);
    return u;
}

// Helper function for path to a test assets
static std::string
get_data_file_path(const char* name)
{
    const auto try_path = [&](const fs::path& base) -> std::string {
        std::error_code ec;
        fs::path        p = base / name;
        if(fs::exists(p, ec) && fs::is_regular_file(p, ec)) return p.string();
        return {};
    };

    for(const char* base : {CODEOBJ_BINARY_DIR, CODEOBJ_INSTALL_DIR})
    {
        if(auto found = try_path(fs::path(base)); !found.empty()) return found;
    }

    if(std::error_code ec{}; true)
    {
        fs::path exe_dir = fs::read_symlink("/proc/self/exe", ec).parent_path();
        if(!ec)
        {
            if(auto found = try_path(exe_dir); !found.empty()) return found;
        }
    }
    return {};  // not found
}

static const std::vector<std::string>&
GetHipccOutput()
{
    static std::vector<std::string> result = []() {
        std::ifstream            file(get_data_file_path("hipcc_output.s"));
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
        std::string   filename = get_data_file_path("smallkernel.bin");
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

    std::string smallkernel_path =
        rocprofiler::testing::codeobjhelper::get_data_file_path("smallkernel.bin");
    std::string          kernel_with_protocol = "file://" + smallkernel_path;
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

/**
 * Verifies that DWARF inline annotation produces " -> " separators.
 * The test kernel calls a __device__ function that calls __syncthreads(),
 * which inlines through HIP headers.  This guarantees at least 3 call stack
 * levels (kernel -> device func -> HIP header), i.e. at least two " -> "
 * separators in the comment of the s_barrier instruction.
 */
TEST(codeobj_library, inline_annotation)
{
    std::string path = codeobjhelper::get_data_file_path("syncthreads_kernel.bin");
    ASSERT_FALSE(path.empty()) << "syncthreads_kernel.bin not found";

    std::ifstream file(path, std::ios::binary);
    using iterator_t = std::istreambuf_iterator<char>;
    std::vector<char> objdata{iterator_t(file), iterator_t{}};
    ASSERT_FALSE(objdata.empty());

    CodeobjDecoderComponent comp(objdata.data(), objdata.size());
    ASSERT_FALSE(comp.m_symbol_map.empty());

    constexpr size_t min_depth = 3;  // kernel -> barrier_wrapper -> HIP header(s)
    size_t           max_depth = 0;
    for(auto& [kaddr, sym] : comp.m_symbol_map)
    {
        size_t vaddr = kaddr;
        while(vaddr < kaddr + sym.mem_size)
        {
            auto faddr = comp.va2fo(vaddr);
            ASSERT_TRUE(faddr.has_value());

            auto inst = comp.disassemble_instruction(*faddr, vaddr);
            ASSERT_NE(inst, nullptr);
            ASSERT_NE(inst->size, 0u);

            // Count separators to determine call stack depth
            size_t depth = 1;
            size_t pos   = 0;
            while((pos = inst->comment.find(disassembly::Instruction::separator, pos)) !=
                  std::string::npos)
            {
                depth++;
                pos += disassembly::Instruction::separator.size();
            }
            max_depth = std::max(max_depth, depth);

            vaddr += inst->size;
        }
    }
    EXPECT_GE(max_depth, min_depth)
        << "Deepest inline call stack was " << max_depth << " levels (expected >= " << min_depth
        << "). DWARF inlined subroutine traversal may be broken.";
}

namespace
{
// Reduce a "file[:line[:col]]" frame to "basename:line", or "" for "no info".
// llvm-symbolizer renders "no info" as "??:0:0" / "??:?"; DWARF line 0 is also
// "no specific line".  Normalizing both sides this way makes path-prefix and
// column differences across machines/builds irrelevant.
std::string
canonicalize(std::string_view s)
{
    if(s.empty() || s == "??:0:0" || s == "??:?" || s == "??:0") return {};

    // Strip optional trailing ":col" (llvm-symbolizer is "file:line:col"; we are "file:line").
    auto last  = s.rfind(':');
    auto first = s.find(':');
    if(last != first && last != std::string_view::npos) s = s.substr(0, last);

    auto colon = s.rfind(':');
    if(colon == std::string_view::npos) return std::string{s};
    auto line = s.substr(colon + 1);
    if(line == "0" || line == "?") return {};
    auto path  = s.substr(0, colon);
    auto slash = path.rfind('/');
    auto base  = (slash == std::string_view::npos) ? path : path.substr(slash + 1);
    return std::string{base} + ':' + std::string{line};
}

// Read a FILE* to EOF into a string.
std::string
slurp(FILE* fp)
{
    std::string out;
    char        buf[4096];
    while(size_t n = std::fread(buf, 1, sizeof(buf), fp))
        out.append(buf, n);
    return out;
}

// Locate llvm-symbolizer at runtime: $ROCM_PATH, $ROCM_HOME, /opt/rocm, then $PATH.
// Resolved at runtime rather than configure time because some CI flows (e.g.
// TheRock) wipe the build dir before tests run, leaving any baked-in absolute
// path stale.
std::string
find_symbolizer()
{
    namespace fs = rocprofiler::common::filesystem;
    auto exists  = [](const std::string& p) {
        std::error_code ec;
        return !p.empty() && fs::exists(p, ec) && !ec;
    };
    for(const char* env : {"ROCM_PATH", "ROCM_HOME"})
        if(const char* v = std::getenv(env); v && *v)
            if(auto p = std::string{v} + "/llvm/bin/llvm-symbolizer"; exists(p)) return p;
    if(std::string p = "/opt/rocm/llvm/bin/llvm-symbolizer"; exists(p)) return p;

    FILE* pipe = ::popen("command -v llvm-symbolizer 2>/dev/null", "r");
    if(!pipe) return {};
    auto out = slurp(pipe);
    ::pclose(pipe);
    while(!out.empty() && (out.back() == '\n' || out.back() == '\r'))
        out.pop_back();
    return out;
}
}  // namespace

/**
 * Differential test: every instruction inside each kernel symbol must yield the
 * same source-line / inline-chain attribution from CodeobjDecoderComponent as
 * from llvm-symbolizer.  Both parse the same .debug_line in the same ELF --
 * disagreement is a bug, and llvm-symbolizer has had years more scrutiny.
 *
 * Guards against the bugs the recent line-attribution fix targets: ranges
 * spilling past end_sequence rows, and ranges spilling across DWARF gaps.
 * Skipped if llvm-symbolizer cannot be found at runtime.
 */
TEST(codeobj_library, dwarf_matches_llvm_symbolizer)
{
    namespace fs                   = rocprofiler::common::filesystem;
    constexpr std::string_view sep = disassembly::Instruction::separator;

    const std::string symbolizer = find_symbolizer();
    if(symbolizer.empty()) GTEST_SKIP() << "llvm-symbolizer not found";

    std::string obj_path = codeobjhelper::get_data_file_path("syncthreads_kernel.bin");
    ASSERT_FALSE(obj_path.empty()) << "syncthreads_kernel.bin not found";

    std::ifstream     in(obj_path, std::ios::binary);
    std::vector<char> obj{std::istreambuf_iterator<char>(in), {}};
    ASSERT_FALSE(obj.empty());

    CodeobjDecoderComponent comp(obj.data(), obj.size());
    ASSERT_FALSE(comp.m_symbol_map.empty());

    // Walk every instruction; step by inst->size so we never feed comgr a
    // mid-instruction address (which throws). Both sides see the same set.
    std::vector<uint64_t>    addrs;
    std::vector<std::string> ours;
    for(auto& [kaddr, sym] : comp.m_symbol_map)
    {
        for(uint64_t va = kaddr; va < kaddr + sym.mem_size;)
        {
            auto faddr = comp.va2fo(va);
            if(!faddr) break;
            std::unique_ptr<disassembly::Instruction> inst;
            try
            {
                inst = comp.disassemble_instruction(*faddr, va);
            } catch(...)
            {
                break;
            }
            if(!inst || inst->size == 0) break;
            addrs.push_back(va);
            ours.push_back(inst->comment);
            va += inst->size;
        }
    }
    ASSERT_FALSE(addrs.empty());

    // Feed addresses to llvm-symbolizer via stdin from a tmp file (the test's
    // build dir is read-only in some CI install layouts).
    std::error_code   ec;
    auto              tmp      = fs::temp_directory_path(ec);
    std::string       tmpl_str = (ec ? "/tmp" : tmp.string()) + "/codeobj_addrs.XXXXXX";
    std::vector<char> tmpl(tmpl_str.begin(), tmpl_str.end());
    tmpl.push_back('\0');
    int fd = ::mkstemp(tmpl.data());
    ASSERT_GE(fd, 0) << "mkstemp: " << ::strerror(errno);
    {
        FILE* fp = ::fdopen(fd, "w");
        ASSERT_NE(fp, nullptr);
        for(auto a : addrs)
            std::fprintf(fp, "0x%lx\n", static_cast<unsigned long>(a));
        std::fclose(fp);
    }
    std::string cmd = '"' + symbolizer + "\" --obj=\"" + obj_path +
                      "\" --inlines --output-style=LLVM < \"" + tmpl.data() + "\" 2>&1";
    FILE* pipe = ::popen(cmd.c_str(), "r");
    ASSERT_NE(pipe, nullptr) << "popen: " << ::strerror(errno);
    std::string raw = slurp(pipe);
    int         rc  = ::pclose(pipe);
    ::unlink(tmpl.data());
    ASSERT_EQ(rc, 0) << "llvm-symbolizer rc=" << rc << " cmd=" << cmd << "\n" << raw;

    // Parse: per-address record terminated by an empty line; each record is
    // (function_name, file:line:col) line pairs, innermost first.
    std::vector<std::vector<std::string>> theirs(addrs.size());
    {
        std::istringstream is(raw);
        std::string        line;
        size_t             i         = 0;
        bool               want_func = true;
        while(std::getline(is, line) && i < addrs.size())
        {
            if(line.empty())
            {
                ++i;
                want_func = true;
            }
            else if(want_func)
                want_func = false;
            else
            {
                theirs[i].emplace_back(canonicalize(line));
                want_func = true;
            }
        }
    }

    // Compare frame-by-frame after canonicalization. Trailing "no info" frames
    // are dropped on both sides so {""} (llvm's "??:0:0") matches our empty.
    auto trim = [](std::vector<std::string>& v) {
        while(!v.empty() && v.back().empty())
            v.pop_back();
    };

    size_t mismatches = 0;
    for(size_t i = 0; i < addrs.size(); ++i)
    {
        std::vector<std::string> mine;
        for(size_t pos = 0; pos <= ours[i].size();)
        {
            auto next = ours[i].find(sep, pos);
            mine.emplace_back(
                canonicalize(ours[i].substr(pos, next == std::string::npos ? next : next - pos)));
            if(next == std::string::npos) break;
            pos = next + sep.size();
        }
        trim(mine);
        trim(theirs[i]);
        if(mine == theirs[i]) continue;

        if(mismatches++ < 10)
        {
            auto fmt = [](const std::vector<std::string>& v) {
                std::string s;
                for(auto& f : v)
                    (s += '[') += f, s += ']';
                return s;
            };
            ADD_FAILURE() << "addr=0x" << std::hex << addrs[i] << std::dec
                          << "\n  ours  : " << fmt(mine) << "\n  theirs: " << fmt(theirs[i]);
        }
    }

    EXPECT_EQ(mismatches, 0u) << mismatches << " of " << addrs.size()
                              << " addresses disagreed with llvm-symbolizer";
}
