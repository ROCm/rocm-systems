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

#include "rocprof_trace_decoder/rocprof_trace_decoder.h"
#include "trie.h"

#include <cstdint>
#include <cstdlib>
#include <thread>

namespace
{
rocprof_trace_decoder_handle_t global_handle{};
int pre_main_status = 0;
bool main_completed = false;

uint64_t no_data(uint8_t** buffer, uint64_t* buffer_size, void*)
{
    *buffer = nullptr;
    *buffer_size = 0;
    return 0;
}

rocprofiler_thread_trace_decoder_status_t
accept_trace(rocprofiler_thread_trace_decoder_record_type_t, void*, uint64_t, void*)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

rocprofiler_thread_trace_decoder_status_t
empty_isa(char*, uint64_t* memory_size, uint64_t* isa_size, rocprofiler_thread_trace_decoder_pc_t, void*)
{
    *memory_size = 4;
    *isa_size = 0;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

bool exercise_handle(rocprof_trace_decoder_handle_t handle)
{
    if (rocprof_trace_decoder_set_isa_callback(handle, empty_isa, nullptr) !=
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        return false;

    if (rocprof_trace_decoder_set_se_data_callback(handle, no_data, nullptr) !=
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        return false;

    return rocprof_trace_decoder_parse(handle, nullptr, 0, accept_trace, nullptr) ==
           ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

bool exercise_v1()
{
    return rocprof_trace_decoder_parse_data(no_data, accept_trace, empty_isa, nullptr) ==
           ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

bool exercise_trie() { return Trie::inst_type("s_waitcnt vmcnt(0)", 10) == InstCategory::IMMED; }

[[noreturn]] void fail_after_main(int status) { std::_Exit(status); }

// Constructed before PreMainInitializer, so this destructor is registered
// before any function-local decoder state initialized by PreMainInitializer.
// It therefore runs after that ordinary static state would have been destroyed.
struct PostMainVerifier
{
    ~PostMainVerifier()
    {
        if (!main_completed) fail_after_main(20);
        if (!exercise_trie()) fail_after_main(21);
        if (!exercise_handle(global_handle)) fail_after_main(22);
        if (rocprof_trace_decoder_destroy_handle(global_handle) != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
            fail_after_main(23);

        if (!exercise_v1()) fail_after_main(24);
        if (!exercise_v1()) fail_after_main(25);

        int worker_status = 0;
        std::thread worker(
            [&]()
            {
                rocprof_trace_decoder_handle_t new_handle{};
                if (rocprof_trace_decoder_create_handle(&new_handle) != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
                {
                    worker_status = 26;
                    return;
                }
                if (!exercise_handle(new_handle))
                {
                    worker_status = 27;
                    return;
                }
                if (rocprof_trace_decoder_destroy_handle(new_handle) != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
                    worker_status = 28;
            }
        );
        worker.join();
        if (worker_status != 0) fail_after_main(worker_status);
    }
};

PostMainVerifier post_main_verifier{};

struct PreMainInitializer
{
    PreMainInitializer()
    {
        if (rocprof_trace_decoder_create_handle(&global_handle) != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS)
        {
            pre_main_status = 10;
            return;
        }
        if (!exercise_handle(global_handle))
        {
            pre_main_status = 11;
            return;
        }
        if (!exercise_v1()) pre_main_status = 12;
        if (!exercise_trie()) pre_main_status = 13;
    }
};

PreMainInitializer pre_main_initializer{};
} // namespace

int main()
{
    if (pre_main_status != 0) return pre_main_status;
    if (!exercise_handle(global_handle)) return 14;

    main_completed = true;
    return 0;
}
