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

#include "occupancy.hpp"
#include <nlohmann/json.hpp>
#include "outputfile.hpp"

#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>

#define OCCUPANCY_RES 8

namespace rocprofiler
{
namespace att_wrapper
{
namespace
{
std::map<pcinfo_t, int> kernel_ids{{pcinfo_t{0, 0}, 0}};
std::atomic<int>        current_id{1};

int
get_kernel_id(pcinfo_t pc)
{
    if(kernel_ids.find(pc) != kernel_ids.end()) return kernel_ids.at(pc);

    return kernel_ids.emplace(pc, current_id.fetch_add(1)).first->second;
}
}  // namespace

namespace OccupancyFile
{
void
OccupancyFile(const Fspath&                                     dir,
              std::shared_ptr<AddressTable>&                    table,
              const std::map<size_t, std::vector<occupancy_t>>& occ)
{
    if(!GlobalDefs::get().has_format("json")) return;
    nlohmann::json jocc;

    for(const auto& [se, eventlist] : occ)
    {
        nlohmann::json list;
        for(const auto& event : eventlist)
        {
            nlohmann::json json_event;
            json_event.push_back(event.time);
            json_event.push_back(event.cu);
            json_event.push_back(event.simd);
            json_event.push_back(event.wave_id);
            json_event.push_back(event.start);
            json_event.push_back(get_kernel_id(event.pc));
            list.push_back(json_event);
        }
        jocc[std::to_string(se)] = list;
    }

    for(auto& [pc, id] : kernel_ids)
    {
        std::stringstream ss;
        try
        {
            ss << table->getSymbolMap(pc.marker_id).at(pc.addr).name;
        } catch(std::exception& e)
        {
            ss << pc.marker_id << " / 0x" << std::hex << pc.addr << std::dec;
        }
        jocc["dispatches"][std::to_string(id)] = ss.str();
    }

    jocc["version"] = "3.0.0";

    OutputFile(dir / "occupancy.json") << jocc;
}
}  // namespace OccupancyFile
}  // namespace att_wrapper
}  // namespace rocprofiler
