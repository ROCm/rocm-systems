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
typedef enum code_line_service_type_t
{
    code_line_service_type_att                    = 0,
    code_line_service_type_host_trap_pc_sampling  = 1,
    code_line_service_type_stochastic_pc_sampling = 2
} code_line_service_type_t;

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

    code_line_service_type_t service_type{code_line_service_type_att};
    size_t                   issued{0};
    size_t                   stalled{0};
    // No need to reserve slots for ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE
    // and ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST, so the total number
    // of elements in the array is ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST - 2.
    // To access the number of samples for a specific stall reason,
    // use the index (rocprofiler_pc_sampling_instruction_not_issued_reason_t enum value - 1).
    std::array<size_t, ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST - 2>
        not_issued_reasons{};
};

class CodeFile
{
    using AddressTable = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

public:
    CodeFile() = default;
    CodeFile(Fspath dir, std::shared_ptr<AddressTable> table);
    ~CodeFile();

    const Fspath                                  dir{};
    std::unordered_map<pcinfo_t, int>             line_numbers{};
    std::map<pcinfo_t, std::unique_ptr<CodeLine>> isa_map{};
    std::map<pcinfo_t, KernelName>                kernel_names{};

    std::shared_ptr<AddressTable> table;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
