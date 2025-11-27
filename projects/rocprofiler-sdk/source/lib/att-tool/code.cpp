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

#include "code.hpp"
#include <nlohmann/json.hpp>
#include "lib/output/csv.hpp"
#include "outputfile.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

#include <cxxabi.h>

namespace rocprofiler
{
namespace att_wrapper
{
using csv_encoder = rocprofiler::tool::csv::csv_encoder<10>;  // added Issued, Stalled

// Builds a json filetree by recursively inserting "path" into the json object.
void
navigate(nlohmann::json& json, std::vector<std::string>& path, const std::string& filename)
{
    if(path.size() == 1) json[path.at(0)] = filename;

    if(path.size() <= 1) return;

    auto& j = json[path.at(0)];
    path.erase(path.begin());
    navigate(j, path, filename);
}

CodeFile::CodeFile(Fspath _dir, std::shared_ptr<AddressTable> _table)
: dir(std::move(_dir))
, table(std::move(_table))
{}

CodeFile::~CodeFile()
{
    std::cout << "NUM SAMPLES: " << num_samples.load() << std::endl;
    std::vector<std::pair<pcinfo_t, std::unique_ptr<CodeLine>>> vec;
    vec.reserve(isa_map.size());

    for(auto& [pc, isa] : isa_map)
        if(isa && isa->code_line) vec.emplace_back(pc, std::move(isa));

    isa_map.clear();
    line_numbers.clear();

    if(GlobalDefs::get().has_format("csv"))
    {
        // Write CSV, ordered by id + vaddr
        std::sort(vec.begin(),
                  vec.end(),
                  [](const std::pair<pcinfo_t, std::unique_ptr<CodeLine>>& a,
                     const std::pair<pcinfo_t, std::unique_ptr<CodeLine>>& b) {
                      return a.first < b.first;
                  });

        std::stringstream ofs;
        csv_encoder::write_row(ofs,
                               "CodeObj",
                               "Vaddr",
                               "Instruction",
                               "Hitcount",
                               "Latency",
                               "Stall",
                               "Idle",
                               "Issued",
                               "Stalled",
                               "Source");

        for(auto& [pc, line] : vec)
        {
            if(kernel_names.find(pc) != kernel_names.end())
            {
                // kernel marker row (no instruction stats)
                csv_encoder::write_row(ofs,
                                       pc.code_object_id,
                                       pc.address,
                                       "; " + kernel_names.at(pc).name,
                                       0,
                                       0,
                                       0,
                                       0,
                                       0,  // Issued
                                       0,  // Stalled
                                       kernel_names.at(pc).demangled);
            }
            csv_encoder::write_row(ofs,
                                   pc.code_object_id,
                                   pc.address,
                                   line->code_line->inst,
                                   line->hitcount,
                                   line->latency,
                                   line->stall,
                                   line->idle,
                                   line->issued,
                                   line->stalled,
                                   line->code_line->comment);
        }

        OutputFile file(dir.parent_path() / ("stats_" + dir.filename().string() + ".csv"));
        file << ofs.str();
    }

    if(!GlobalDefs::get().has_format("json")) return;

    // Write JSON, ordered by exec line number
    std::sort(vec.begin(),
              vec.end(),
              [](const std::pair<pcinfo_t, std::unique_ptr<CodeLine>>& a,
                 const std::pair<pcinfo_t, std::unique_ptr<CodeLine>>& b) {
                  return a.second->line_number < b.second->line_number;
              });

    nlohmann::json jcode;

    std::unordered_set<std::string> snapshots{};

    for(auto& line : vec)
    {
        auto& isa = *line.second;

        if(kernel_names.find(line.first) != kernel_names.end())
        {
            std::stringstream code;
            code << "[\"; " << kernel_names.at(line.first).name << "\",0," << (isa.line_number - 1)
                 << ",\"" << kernel_names.at(line.first).demangled << "\","
                 << line.first.code_object_id << "," << line.first.address
                 << ",0,0,0,0,0,0";  // Hit, Latency, Stall, Idle, Issued, Stalled

            code << ",[0";
            for(size_t i = 1; i < CodeLine{}.stall_reasons.size(); ++i) code << ",0";
            code << "]]";
            jcode.push_back(nlohmann::json::parse(code.str()));
        }

        std::stringstream code;
        code << "[\"" << isa.code_line->inst << "\",0," << isa.line_number << ",\""
             << isa.code_line->comment << "\"," << line.first.code_object_id << ","
             << line.first.address << "," << isa.hitcount << "," << isa.latency << "," << isa.stall
             << "," << isa.idle << "," << isa.issued << "," << isa.stalled;

        code << ",[" << isa.stall_reasons.at(0);
        for(size_t i = 1; i < isa.stall_reasons.size(); ++i) code << ',' << isa.stall_reasons.at(i);
        code << "]]";
        jcode.push_back(nlohmann::json::parse(code.str()));

        auto&  comment  = isa.code_line->comment;
        size_t lineref  = comment.find(':');
        size_t previous = 0;

        // size() + 2 because we need at least ':' and one number after
        while(lineref != std::string::npos && lineref < comment.size() + 2)
        {
            auto source_ref = comment.substr(previous, lineref - previous);

            if(!source_ref.empty() && snapshots.find(source_ref) == snapshots.end())
                snapshots.insert(std::move(source_ref));

            previous = comment.find(CodeLine::Instruction::separator, lineref);
            if(previous == std::string::npos) break;

            previous += CodeLine::Instruction::separator.size();
            lineref = comment.find(':', previous);
        }
    }

    nlohmann::json json;
    json["code"]    = jcode;
    json["version"] = TOOL_VERSION;

    for (auto& entry : {"ISA", "_", "LineNumber", "Source", "Codeobj", "Vaddr", "Hit", "Latency", "Stall", "Idle", "PC_Issued", "PC_Stalled", "Stall_Reasons"})
        json["header"].push_back(std::string(entry));

    OutputFile(dir / "code.json") << json;

    nlohmann::json jsnapfiletree;
    size_t         num_snap = 0;

    for(const auto& source_ref : snapshots)
    {
        if(rocprofiler::common::filesystem::exists(source_ref))
        {
            Fspath            filepath(source_ref);
            std::stringstream newfile;
            newfile << "source_" << (num_snap++) << '_' << filepath.filename().string();

            std::vector<std::string> path_elements(filepath.begin(), filepath.end());
            navigate(jsnapfiletree, path_elements, newfile.str());

            constexpr auto opt = rocprofiler::common::filesystem::copy_options::overwrite_existing;
            try
            {
                rocprofiler::common::filesystem::copy(filepath, dir / newfile.str(), opt);
            } catch(std::exception& e)
            {
                ROCP_WARNING << "Missing source file " << filepath << ": " << e.what();
                ROCP_CI_LOG(ERROR) << "Unable to copy source files: " << (dir / newfile.str());
            }
        }
    }

    if(num_snap != 0) OutputFile(dir / "snapshots.json") << jsnapfiletree;
}

std::string
demangle(std::string_view line)
{
    int   status{0};
    char* c_name = abi::__cxa_demangle(line.data(), nullptr, nullptr, &status);

    if(c_name == nullptr) return "";

    std::string str = c_name;
    free(c_name);
    return str;
}

CodeLine&
CodeFile::get(pcinfo_t _pc)
{
    if(isa_map.find(_pc) != isa_map.end()) return *isa_map.at(_pc);

    // Attempt to disassemble full kernel
    if(_pc.code_object_id != 0u) try
        {
            rocprofiler::sdk::codeobj::segment::CodeobjTableTranslator symbol_table;
            for(auto& [vaddr, symbol] : table->getSymbolMap(_pc.code_object_id))
                symbol_table.insert({symbol.vaddr, symbol.mem_size, _pc.code_object_id});

            auto addr_range = symbol_table.find_codeobj_in_range(_pc.address);
            try
            {
                auto symbol = table->getSymbolMap(_pc.code_object_id).at(addr_range.addr);
                auto pair   = KernelName{symbol.name, demangle(symbol.name)};
                kernel_names.emplace(pcinfo_t{addr_range.addr, _pc.code_object_id}, pair);
            } catch(...)
            {
                ROCP_INFO << "Missing kernelSymbol at " << _pc.code_object_id << ':'
                          << addr_range.addr;
            }

            for(auto addr = addr_range.addr; addr < addr_range.addr + addr_range.size;)
            {
                pcinfo_t info{.address = addr, .code_object_id = addr_range.id};
                auto& cline = *(isa_map.emplace(info, std::make_unique<CodeLine>()).first->second);

                cline.line_number         = isa_map.size() + kernel_names.size() - 1;
                line_numbers[info] = cline.line_number;

                cline.code_line = table->get(addr_range.id, addr);
                addr += cline.code_line->size;
                if(cline.code_line->size == 0u) throw std::invalid_argument("Line has 0 bytes!");
            }

            if(isa_map.find(_pc) != isa_map.end()) return *isa_map.at(_pc);
        } catch(std::exception& e)
        {}

    auto& cline = *(isa_map.emplace(_pc, std::make_unique<CodeLine>()).first->second);

    cline.line_number = isa_map.size();
    line_numbers[_pc] = cline.line_number;

    cline.code_line = table->get(_pc.code_object_id, _pc.address);

    return cline;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
