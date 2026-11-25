// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/kernel_replay/blit-code-objects.hpp"

#include <cstdint>
#include <vector>

// CMake writes a ROCPROFILER_KERNEL_REPLAY_BLIT_TABLE_ENTRY(symbol, "gpu name", "path") line for
// each GPU the blit kernels were built for. The table is included twice: once to assemble the code
// objects, and once to list them.

// Each code object is assembled into read-only data, 4 KiB aligned as HIP aligns the code objects
// in its fat binaries, and followed by its size.
#define ROCPROFILER_KERNEL_REPLAY_BLIT_TABLE_ENTRY(SYMBOL, NAME, PATH)                             \
    __asm__(".pushsection .rodata.rocprofiler_kernel_replay_blit, \"a\", %progbits\n"              \
            ".p2align 12\n"                                                                        \
            ".globl rocprofiler_kernel_replay_blit_" #SYMBOL "\n"                                  \
            ".hidden rocprofiler_kernel_replay_blit_" #SYMBOL "\n"                                 \
            "rocprofiler_kernel_replay_blit_" #SYMBOL ":\n"                                        \
            ".incbin \"" PATH "\"\n"                                                               \
            ".Lrocprofiler_kernel_replay_blit_" #SYMBOL "_end:\n"                                  \
            ".p2align 3\n"                                                                         \
            ".globl rocprofiler_kernel_replay_blit_" #SYMBOL "_size\n"                             \
            ".hidden rocprofiler_kernel_replay_blit_" #SYMBOL "_size\n"                            \
            "rocprofiler_kernel_replay_blit_" #SYMBOL "_size:\n"                                   \
            ".quad .Lrocprofiler_kernel_replay_blit_" #SYMBOL "_end - "                            \
            "rocprofiler_kernel_replay_blit_" #SYMBOL "\n"                                         \
            ".popsection\n");                                                                      \
    extern "C" __attribute__((visibility("hidden")))                                               \
        const unsigned char rocprofiler_kernel_replay_blit_##SYMBOL[];                             \
    extern "C" __attribute__((visibility("hidden")))                                               \
        const uint64_t rocprofiler_kernel_replay_blit_##SYMBOL##_size;
#include ROCPROFILER_KERNEL_REPLAY_BLIT_CODE_OBJECTS
#undef ROCPROFILER_KERNEL_REPLAY_BLIT_TABLE_ENTRY

namespace rocprofiler
{
namespace kernel_replay
{
namespace blit
{
const std::vector<embedded_code_object_t>&
get_embedded_code_objects()
{
#define ROCPROFILER_KERNEL_REPLAY_BLIT_TABLE_ENTRY(SYMBOL, NAME, PATH)                             \
    embedded_code_object_t{NAME,                                                                   \
                           rocprofiler_kernel_replay_blit_##SYMBOL,                                \
                           rocprofiler_kernel_replay_blit_##SYMBOL##_size},
    static const auto* value = new std::vector<embedded_code_object_t>{
#include ROCPROFILER_KERNEL_REPLAY_BLIT_CODE_OBJECTS
    };
#undef ROCPROFILER_KERNEL_REPLAY_BLIT_TABLE_ENTRY
    return *value;
}
}  // namespace blit
}  // namespace kernel_replay
}  // namespace rocprofiler
