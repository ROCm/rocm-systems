/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <gtest/gtest.h>

#include <mutex>
#include <vector>

#include "core/memory/generic_buffer.h"

void buffer_callback_fun(const rocprofiler_record_header_t* begin,
                         const rocprofiler_record_header_t* end, rocprofiler_session_id_t session_id,
                         rocprofiler_buffer_id_t buffer_id) {
  std::cout << "buffer callback" << std::endl;
}
// A lot have changed in the class, since this test was written
// Need to rewrite all the test cases again.
TEST(WhenAddingARecordToBuffer, DISABLED_RecordGetsAddedSuccefully) {
  Memory::GenericBuffer* buffer = new Memory::GenericBuffer(
      rocprofiler_session_id_t{0}, rocprofiler_buffer_id_t{0}, 0x8000, buffer_callback_fun);

  uint64_t start_time = 0;
  uint64_t end_time = 10;

  uint64_t kernel_object = 123456789;
  uint64_t gpu_name_descriptor = 1234565789;
  rocprofiler_record_profiler_t record = rocprofiler_record_profiler_t{
      rocprofiler_record_header_t{ROCPROFILER_PROFILER_RECORD, rocprofiler_record_id_t{0}},
      rocprofiler_kernel_id_t{kernel_object},
      rocprofiler_agent_id_t{gpu_name_descriptor},
      rocprofiler_queue_id_t{0},
      rocprofiler_record_header_timestamp_t{start_time, end_time},
      nullptr,
      0};

  EXPECT_TRUE(buffer->AddRecord(record));
  delete buffer;
}