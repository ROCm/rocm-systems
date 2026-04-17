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
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>
#include <sstream>
#include <string_view>
#include <vector>

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
// Strip the trailing ":<col>" off "file:line:col" -> "file:line", and normalize
// llvm-symbolizer's "no info" rendering ("??:?", "??:0", "<file>:0") to the
// same convention used by code_printing.hpp ("<file>:?" or empty).
std::string
normalize_fileline(std::string_view s)
{
    if(s.empty()) return {};

    // strip trailing :col (if present and numeric/?). llvm-symbolizer emits
    // "file:line:col"; we only carry "file:line", so drop everything after the
    // last ':' if it is the third colon-delimited field.
    auto last_colon  = s.rfind(':');
    auto first_colon = s.find(':');
    if(last_colon != std::string_view::npos && last_colon != first_colon)
        s = s.substr(0, last_colon);

    // collapse "??:?" / "??:0" to empty (== our "no info")
    if(s == "??:?" || s == "??:0") return {};

    // collapse trailing ":0" to ":?" (line 0 in DWARF == "no specific line")
    if(s.size() >= 2 && s.substr(s.size() - 2) == ":0")
        return std::string(s.substr(0, s.size() - 2)) + ":?";

    return std::string(s);
}

// Compare two file:line strings by basename (path component after last '/'),
// ignoring path-prefix differences across machines/builds.
bool
fileline_basename_equal(std::string_view a, std::string_view b)
{
    if(a == b) return true;
    if(a.empty() || b.empty()) return a == b;

    auto basename = [](std::string_view s) {
        auto colon = s.rfind(':');
        if(colon == std::string_view::npos) return s;
        auto path  = s.substr(0, colon);
        auto slash = path.rfind('/');
        if(slash == std::string_view::npos) return s;
        return std::string_view{s.data() + slash + 1, s.size() - slash - 1};
    };

    return basename(a) == basename(b);
}

// Split our comment ("file:line -> file:line -> ...") into a list of frames.
std::vector<std::string>
split_our_comment(std::string_view comment)
{
    std::vector<std::string> out;
    if(comment.empty()) return out;

    constexpr std::string_view sep = rocprofiler::sdk::codeobj::disassembly::Instruction::separator;
    size_t                     pos = 0;
    while(pos <= comment.size())
    {
        auto next = comment.find(sep, pos);
        if(next == std::string_view::npos)
        {
            out.emplace_back(comment.substr(pos));
            break;
        }
        out.emplace_back(comment.substr(pos, next - pos));
        pos = next + sep.size();
    }
    return out;
}

// Run llvm-symbolizer once with all addresses fed on stdin via a tmp file, and
// return its parsed output: outer vector is one entry per address (in order),
// inner vector is the inline-chain frames as "file:line" (innermost first),
// already normalized for "no info" lines.
std::vector<std::vector<std::string>>
run_llvm_symbolizer(const std::string&           symbolizer,
                    const std::string&           obj_path,
                    const std::vector<uint64_t>& addrs,
                    const std::string&           tmp_dir)
{
    std::vector<std::vector<std::string>> result(addrs.size());
    if(addrs.empty()) return result;

    std::string addrs_path = tmp_dir + "/codeobj_dwarf_addrs.txt";
    {
        std::ofstream addrs_file(addrs_path);
        if(!addrs_file) return result;
        for(auto a : addrs)
            addrs_file << "0x" << std::hex << a << '\n';
    }

    // --inlines:        emit the inline chain (innermost first)
    // --output-style=LLVM: default; "func\nfile:line:col\n\n" per address
    // --demangle=1:     default; we ignore function names anyway
    std::ostringstream cmd;
    cmd << '"' << symbolizer << "\" --obj=\"" << obj_path << "\" --inlines --output-style=LLVM < \""
        << addrs_path << '"';

    std::unique_ptr<FILE, int (*)(FILE*)> pipe(::popen(cmd.str().c_str(), "r"), &::pclose);
    if(!pipe) return result;

    std::string raw;
    {
        char buf[4096];
        while(size_t n = std::fread(buf, 1, sizeof(buf), pipe.get()))
            raw.append(buf, n);
    }

    // Parse: per-address record terminated by an empty line; each record is a
    // sequence of (function_name, file:line:col) line pairs.
    std::istringstream is(raw);
    std::string        line;
    size_t             addr_idx    = 0;
    bool               expect_func = true;  // alternates between function-name and file:line:col

    while(std::getline(is, line))
    {
        if(addr_idx >= addrs.size()) break;
        if(line.empty())
        {
            ++addr_idx;
            expect_func = true;
            continue;
        }
        if(expect_func)
            expect_func = false;  // skip function name line
        else
        {
            result[addr_idx].emplace_back(normalize_fileline(line));
            expect_func = true;
        }
    }
    return result;
}
}  // namespace

/**
 * Differential test: every 4-byte address inside each kernel symbol must yield
 * the same source-line / inline-chain attribution from CodeobjDecoderComponent
 * as it does from llvm-symbolizer.  Both tools parse the same .debug_line in
 * the same ELF — disagreement is a bug on one side, and llvm-symbolizer has
 * had years more scrutiny.
 *
 * Specifically guards against the bugs the recent line-attribution fix exists
 * to prevent: ranges spilling past end_sequence rows, and ranges spilling
 * across gaps in DWARF coverage.
 *
 * Skipped at runtime if llvm-symbolizer was not available at configure time.
 */
TEST(codeobj_library, dwarf_matches_llvm_symbolizer)
{
#ifndef LLVM_SYMBOLIZER_PATH
    GTEST_SKIP() << "LLVM_SYMBOLIZER_PATH not defined";
#else
    const std::string symbolizer = LLVM_SYMBOLIZER_PATH;
    if(symbolizer.empty()) GTEST_SKIP() << "llvm-symbolizer not found at configure time";

    std::string obj_path = codeobjhelper::get_data_file_path("syncthreads_kernel.bin");
    ASSERT_FALSE(obj_path.empty()) << "syncthreads_kernel.bin not found";

    std::ifstream file(obj_path, std::ios::binary);
    using iterator_t = std::istreambuf_iterator<char>;
    std::vector<char> objdata{iterator_t(file), iterator_t{}};
    ASSERT_FALSE(objdata.empty());

    CodeobjDecoderComponent comp(objdata.data(), objdata.size());
    ASSERT_FALSE(comp.m_symbol_map.empty());

    // Walk every instruction in every kernel symbol and collect (vaddr, our-comment).
    // Stepping by inst->size (rather than the 4-byte AMDGPU minimum) avoids
    // calling amd_comgr_disassemble_instruction on mid-instruction addresses,
    // which throws.  Both sides still see the exact same address set, so any
    // disagreement is a real DWARF-attribution bug.
    std::vector<uint64_t>    addrs;
    std::vector<std::string> ours;
    for(auto& [kaddr, sym] : comp.m_symbol_map)
    {
        uint64_t va = kaddr;
        while(va < kaddr + sym.mem_size)
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

    auto theirs = run_llvm_symbolizer(symbolizer, obj_path, addrs, CODEOBJ_BINARY_DIR);
    ASSERT_EQ(theirs.size(), addrs.size())
        << "llvm-symbolizer returned a different number of address records";

    size_t mismatches = 0;
    for(size_t i = 0; i < addrs.size(); ++i)
    {
        auto  our_frames   = split_our_comment(ours[i]);
        auto& their_frames = theirs[i];

        // Both empty == both report "no info" for this address. OK.
        if(our_frames.empty() && their_frames.empty()) continue;

        bool ok = our_frames.size() == their_frames.size();
        if(ok)
        {
            for(size_t f = 0; f < our_frames.size(); ++f)
            {
                if(!fileline_basename_equal(normalize_fileline(our_frames[f]), their_frames[f]))
                {
                    ok = false;
                    break;
                }
            }
        }

        if(!ok && mismatches < 10)
        {
            std::ostringstream our_s, their_s;
            for(auto& f : our_frames)
                our_s << '[' << f << ']';
            for(auto& f : their_frames)
                their_s << '[' << f << ']';
            ADD_FAILURE() << "addr=0x" << std::hex << addrs[i] << std::dec
                          << "\n  ours  : " << our_s.str() << "\n  theirs: " << their_s.str();
            ++mismatches;
        }
        else if(!ok)
        {
            ++mismatches;
        }
    }

    EXPECT_EQ(mismatches, 0u) << mismatches << " of " << addrs.size()
                              << " addresses disagreed with llvm-symbolizer";
#endif
}
