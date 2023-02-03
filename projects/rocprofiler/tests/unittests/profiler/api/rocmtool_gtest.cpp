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

#include <vector>

#include "core/session/session.h"
#include "api/rocmtool.h"

void (*callback_fun)(const rocprofiler_record_header_t* begin, const rocprofiler_record_header_t* end,
                     rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id);

TEST(WhenTestingCounterCollectionMode, TestSucceeds) {
  rocprofiler_session_id_t session_id;

  rocmtools::rocmtool toolobj;
  session_id = toolobj.CreateSession(ROCPROFILER_NONE_REPLAY_MODE);
  rocprofiler_filter_id_t filter_id =
      toolobj.GetSession(session_id)
          ->CreateFilter(ROCPROFILER_COUNTERS_COLLECTION, rocprofiler_filter_data_t{}, 0,
                         rocprofiler_filter_property_t{});
  rocprofiler_buffer_id_t buffer_id =
      toolobj.GetSession(session_id)->CreateBuffer(callback_fun, 0x9999);
  toolobj.GetSession(session_id)->GetFilter(filter_id)->SetBufferId(buffer_id);


  rocmtools::Session* session = toolobj.GetSession(session_id);
  EXPECT_TRUE(session->FindFilterWithKind(ROCPROFILER_COUNTERS_COLLECTION));
  toolobj.DestroySession(session_id);
}

TEST(WhenTestingTimeStampCollectionMode, TestSucceeds) {
  rocprofiler_session_id_t session_id;

  rocmtools::rocmtool toolobj;
  session_id = toolobj.CreateSession(ROCPROFILER_NONE_REPLAY_MODE);
  rocprofiler_filter_id_t filter_id =
      toolobj.GetSession(session_id)
          ->CreateFilter(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION, rocprofiler_filter_data_t{}, 0,
                         rocprofiler_filter_property_t{});
  rocprofiler_buffer_id_t buffer_id =
      toolobj.GetSession(session_id)->CreateBuffer(callback_fun, 0x9999);
  toolobj.GetSession(session_id)->GetFilter(filter_id)->SetBufferId(buffer_id);


  rocmtools::Session* session = toolobj.GetSession(session_id);

  EXPECT_TRUE(session->FindFilterWithKind(ROCPROFILER_DISPATCH_TIMESTAMPS_COLLECTION));
  toolobj.DestroySession(session_id);
}

TEST(WhenTestingApplicationReplayMode, TestSucceeds) {
  std::vector<const char*> counters;
  counters.emplace_back("SQ_WAVES");
  rocprofiler_session_id_t session_id;

  rocmtools::rocmtool toolobj;
  session_id = toolobj.CreateSession(ROCPROFILER_APPLICATION_REPLAY_MODE);

  rocprofiler_filter_id_t filter_id =
      toolobj.GetSession(session_id)
          ->CreateFilter(ROCPROFILER_COUNTERS_COLLECTION,
                         rocprofiler_filter_data_t{.counters_names = &counters[0]}, counters.size(),
                         rocprofiler_filter_property_t{});
  rocprofiler_buffer_id_t buffer_id =
      toolobj.GetSession(session_id)->CreateBuffer(callback_fun, 0x8000);
  toolobj.GetSession(session_id)->GetFilter(filter_id)->SetBufferId(buffer_id);

  rocmtools::Session* session = toolobj.GetSession(session_id);

  EXPECT_TRUE(session->FindFilterWithKind(ROCPROFILER_COUNTERS_COLLECTION));
  toolobj.DestroySession(session_id);
}