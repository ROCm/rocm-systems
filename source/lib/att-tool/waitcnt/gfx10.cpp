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
#include <mutex>

#define CLASS_BITS (0xF)

namespace rocprofiler
{
namespace att_wrapper
{
enum MemoryInstType
{
    TYPE_UNCLASSIFIED = 0,
    TYPE_NOT_MEM,
    TYPE_SCALAR,
    TYPE_LDS_MSG,
    TYPE_MSG_RTN,
    TYPE_GLOBAL_LOAD,
    TYPE_GLOBAL_STOR,
    TYPE_FLAT_LOAD,
    TYPE_FLAT_STOR,
    TYPE_WAITCNT
};

union MemoryInst
{
    MemoryInst() = default;
    MemoryInst(int type)
    : raw(type){};

    struct
    {
        int inst : 8;  // One of MemoryInstType
        int lgkm : 1;
        int vmcn : 1;
        int vscn : 1;
    };
    int raw = 0;
};

static_assert(sizeof(MemoryInst) == sizeof(int));

namespace
{
MemoryInst
classify(const std::string& inst)
{
    constexpr size_t npos = std::string::npos;

    if(inst.find("s_wait") == 0)
    {
        if(inst.find("s_wait_alu") != npos) return MemoryInstType::TYPE_NOT_MEM;

        MemoryInst type = MemoryInstType::TYPE_WAITCNT;
        if(inst.find("lgkm") != npos) type.lgkm = true;
        if(inst.find("vscnt") != npos) type.vscn = true;
        if(inst.find("vmcnt") != npos) type.vmcn = true;

        return type;
    }

    if(inst.find("v_") == 0) return MemoryInstType::TYPE_NOT_MEM;

    if(inst.find("s_") == 0)
    {
        if(inst.find("s_load") == 0 || inst.find("s_store") == 0)
            return MemoryInstType::TYPE_SCALAR;

        if(inst.find("s_sendmsg") == 0)
        {
            if(inst.find("s_sendmsg_rtn") == 0)
                return MemoryInstType::TYPE_MSG_RTN;
            else
                return MemoryInstType::TYPE_LDS_MSG;
        }

        return MemoryInstType::TYPE_NOT_MEM;
    }

    bool bStore = inst.find("store") != npos;

    if(inst.find("global_") == 0 || inst.find("buffer_") <= 1 || inst.find("scratch_") == 0)
        return bStore ? MemoryInstType::TYPE_GLOBAL_STOR : MemoryInstType::TYPE_GLOBAL_LOAD;
    else if(inst.find("flat_") == 0)
        return bStore ? MemoryInstType::TYPE_FLAT_STOR : MemoryInstType::TYPE_FLAT_LOAD;
    else if(inst.find("ds_") == 0)
        return MemoryInstType::TYPE_LDS_MSG;
    else
        return MemoryInstType::TYPE_NOT_MEM;
}
}  // namespace

std::vector<LineWaitcnt>
WaitcntList::gfx10_construct(const wave_t& wave, isa_map_t& isa_map)
{
    std::vector<LineWaitcnt> mem_unroll;
    MemoryCounter            lgkm("lgkmcnt");
    MemoryCounter            vmcnt("vmcnt");
    MemoryCounter            vscnt("vscnt");

    std::vector<int> flat_load{};
    std::vector<int> flat_stor{};

    for(size_t i = 0; i < wave.instructions_size; i++)
    {
        auto& event = wave.instructions_array[i];
        if(event.pc.marker_id == 0 && event.pc.addr == 0) continue;

        auto it = isa_map.find(event.pc);
        if(it == isa_map.end() || !it->second->code_line || it->second->code_line->inst.empty())
        {
            static thread_local std::once_flag failed_flag{};
            std::call_once(failed_flag, []() { ROCP_WARNING << "Analysis failed!"; });
            break;
        }

        MemoryInst         type     = it->second->type;
        const std::string& inst_str = it->second->code_line->inst;

        if(type.inst == MemoryInstType::TYPE_UNCLASSIFIED)
        {
            type             = classify(inst_str);
            it->second->type = type.raw;
        }

        int line_number = it->second->line_number;

        if(type.inst == MemoryInstType::TYPE_NOT_MEM) continue;

        if(type.inst == MemoryInstType::TYPE_LDS_MSG)
        {
            lgkm.list.push_back(line_number);
        }
        else if(type.inst == MemoryInstType::TYPE_MSG_RTN)
        {
            lgkm.list.push_back(line_number);  // Increments by 2
            lgkm.list.push_back(line_number);
        }
        else if(type.inst == MemoryInstType::TYPE_SCALAR)
        {
            lgkm.order = MemoryCounter::Ordering::MEMORY_PARALLEL;
            lgkm.list.push_back(line_number);
        }
        else if(type.inst == MemoryInstType::TYPE_GLOBAL_LOAD)
        {
            vmcnt.list.push_back(line_number);
        }
        else if(type.inst == MemoryInstType::TYPE_GLOBAL_STOR)
        {
            vscnt.list.push_back(line_number);
        }
        else if(type.inst == MemoryInstType::TYPE_FLAT_LOAD)
        {
            lgkm.order  = MemoryCounter::Ordering::MEMORY_PARALLEL;
            vmcnt.order = MemoryCounter::Ordering::MEMORY_PARALLEL;
            flat_load.push_back(line_number);
        }
        else if(type.inst == MemoryInstType::TYPE_FLAT_STOR)
        {
            lgkm.order  = MemoryCounter::Ordering::MEMORY_PARALLEL;
            vscnt.order = MemoryCounter::Ordering::MEMORY_PARALLEL;
            flat_stor.push_back(line_number);
        }
        else if(type.inst == MemoryInstType::TYPE_WAITCNT)
        {
            if(type.vscn)
            {
                if(auto joined = vscnt.handle_mem_op(inst_str, flat_stor))
                    mem_unroll.emplace_back(LineWaitcnt{line_number, std::move(*joined)});
            }

            if(type.vmcn)
            {
                if(auto joined = vmcnt.handle_mem_op(inst_str, flat_load))
                    mem_unroll.emplace_back(LineWaitcnt{line_number, std::move(*joined)});
            }

            if(type.lgkm)
            {
                if(auto joined = lgkm.handle_mem_op(inst_str, flat_load))
                {
                    if(!flat_stor.empty())
                    {
                        if(auto stor = lgkm.handle_mem_op(inst_str, flat_stor))
                            joined->insert(joined->end(), stor->begin(), stor->end());
                    }

                    mem_unroll.emplace_back(LineWaitcnt{line_number, std::move(*joined)});
                }
            }
        }
    }

    return mem_unroll;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
