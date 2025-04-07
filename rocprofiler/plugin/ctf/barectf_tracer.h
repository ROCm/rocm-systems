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

#ifndef PLUGIN_CTF_BARECTF_TRACER_H
#define PLUGIN_CTF_BARECTF_TRACER_H

#include <cstdlib>
#include <memory>
#include <vector>
#include <string>
#include "src/utils/filesystem.hpp"

#include "barectf_event_record.h"
#include "barectf_writer.h"

namespace rocm_ctf {

// A barectf tracer offers the AddEventRecord() method to add an event
// record which it will ultimately write to some CTF data stream file
// within some specified CTF trace directory.
//
// One important feature of such a tracer is that you don't need to add
// event records in order of time. A barectf tracer manages one or more
// barectf writers, each one managing a single barectf platform/context
// (CTF data stream file).
//
// All the CTF data stream files which a barectf tracer indirectly
// manages share a common specified prefix. You must not use the same
// prefix for two barectf tracers writing to the same CTF trace
// directory.
//
// `PlatformDescrT` is the specific barectf platform descriptor (see the
// documentation of the `BarectfPlatform` class template).
template <typename PlatformDescrT> class BarectfTracer final {
 public:
  // Specific barectf event record type.
  using EventRecord = typename BarectfWriter<PlatformDescrT>::EventRecord;

  // Builds a barectf tracer to write CTF packets of size `packet_size`
  // bytes to CTF data stream files having the prefix
  // `data_stream_file_name_prefix` within the CTF trace directory
  // `trace_dir`.
  //
  // The internal barectf writers manage event record queues having a
  // maximum size of `max_writer_queue_size`. Increasing
  // `max_writer_queue_size` increases the memory footprint of the
  // tracer, but may reduce the number of required CTF data stream files
  // to ensure time-ordered event records.
  explicit BarectfTracer(const std::size_t packet_size,
                         rocprofiler::common::filesystem::path trace_dir,
                         const char* const data_stream_file_name_prefix,
                         const std::size_t max_writer_queue_size = 200)
      : packet_size_{packet_size},
        trace_dir_{std::move(trace_dir)},
        data_stream_file_name_prefix_{data_stream_file_name_prefix},
        max_writer_queue_size_{max_writer_queue_size} {}

  // Disabled copy operations to make this class simpler.
  BarectfTracer(const BarectfTracer&) = delete;
  BarectfTracer& operator=(const BarectfTracer&) = delete;

  // Adds the event record `event_record` to this tracer.
  //
  // The clock value of `event_record` may be less than the clock value
  // of previously added event records.
  void AddEventRecord(typename EventRecord::SP event_record) {
    // Try to find a barectf writer to accept `event_record`.
    for (auto& writer : writers_) {
      if (writer->MayAddEventRecord(*event_record)) {
        // Found: add the event record to this writer and return.
        writer->AddEventRecord(std::move(event_record));
        return;
      }
    }

    // No barectf writer found: create a new one.
    std::ostringstream ss;

    ss << data_stream_file_name_prefix_ << writers_.size();
    writers_.emplace_back(new BarectfWriter<PlatformDescrT>{packet_size_, trace_dir_ / ss.str(),
                                                            max_writer_queue_size_});

    // Add the event record to this new barectf writer.
    assert(writers_.back()->MayAddEventRecord(*event_record));
    writers_.back()->AddEventRecord(std::move(event_record));
  }

 private:
  // CTF packet size.
  std::size_t packet_size_;

  // CTF trace directory.
  rocprofiler::common::filesystem::path trace_dir_;

  // CTF data stream file name prefix.
  std::string data_stream_file_name_prefix_;

  // Maximum event record queue size of a barectf writer.
  std::size_t max_writer_queue_size_;

  // barectf writers.
  std::vector<std::unique_ptr<BarectfWriter<PlatformDescrT>>> writers_;
};

}  // namespace rocm_ctf

#endif  // PLUGIN_CTF_BARECTF_TRACER_H
