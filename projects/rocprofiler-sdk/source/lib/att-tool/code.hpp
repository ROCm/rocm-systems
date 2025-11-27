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

#pragma once

#include "att_lib_wrapper.hpp"

#include <rocprofiler-sdk/pc_sampling.h>

#include <map>
#include <unordered_map>
#include <vector>
#include "util.hpp"

namespace rocprofiler
{
namespace att_wrapper
{
struct CodeLine
{
    using Instruction = rocprofiler::sdk::codeobj::disassembly::Instruction;

    int                          line_number{0};
    int                          type{0};
    std::shared_ptr<Instruction> code_line{nullptr};

    size_t hitcount{0};
    size_t latency{0};
    size_t stall{0};
    size_t idle{0};

    size_t  issued{0};
    size_t  stalled{0};
    // No need to reserve slots for ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE
    // and ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST, so the total number
    // of elements in the array is ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST - 2.
    // To access the number of samples for a specific stall reason,
    // use the index (rocprofiler_pc_sampling_instruction_not_issued_reason_t enum value - 1).
    std::array<size_t, ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST> stall_reasons{};
};

class CodeFile
{
    using AddressTable = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

public:
    CodeFile() = default;
    CodeFile(Fspath dir, std::shared_ptr<AddressTable> table);
    ~CodeFile();

    CodeLine& get(pcinfo_t _pc);

    template <typename Generator, typename Record, bool isStochastic>
    void aggregate_pc_samples(const Generator& pc_sampling_gen)
    {
        for(auto pitr : pc_sampling_gen)
        {
            for(auto itr : pc_sampling_gen.get(pitr))
            {
                auto  record = itr.pc_sample_record;
                if (record.pc.code_object_id == 0) continue;

                auto& cline = get(pcinfo_t{record.pc.code_object_offset, record.pc.code_object_id});
                num_samples++;

                if constexpr(isStochastic)
                {
                    if (!record.wave_issued)
                    {
                        cline.stalled++;
                        cline.stall_reasons[record.snapshot.reason_not_issued]++;
                        continue;
                    }
                }
                cline.issued++;
            }
        }
    }

    const Fspath                                  dir{};
    std::unordered_map<pcinfo_t, int>             line_numbers{};
    std::map<pcinfo_t, std::unique_ptr<CodeLine>> isa_map{};
    std::map<pcinfo_t, KernelName>                kernel_names{};
    std::shared_ptr<AddressTable>                 table{};
    std::atomic<size_t>                           num_samples{};
};

}  // namespace att_wrapper
}  // namespace rocprofiler
