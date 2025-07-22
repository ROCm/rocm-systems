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

#ifndef PLUGIN_CTF_BARECTF_WRITER_H
#define PLUGIN_CTF_BARECTF_WRITER_H

#include <cassert>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <queue>
#include <utility>
#include "src/utils/filesystem.hpp"

#include "barectf_platform.h"
#include "barectf_event_record.h"

namespace rocm_ctf {

template <typename> class BarectfTracer;

// A barectf writer manages a queue of event records, writing them
// through barectf when needed.
//
// Such an object makes it possible to add some event record with a
// clock value V and then some other event record of which the clock
// value is less than V. The barectf writer ensures that actual barectf
// tracing functions are called chronologically, a requirement of CTF.
//
// A barectf writer keeps event records in memory until its queue is
// full (you provide the maximum queue size at construction time), in
// which case it writes the oldest event record to some current CTF
// packet through a barectf tracing function.
//
// Call MayAddEventRecord() to check whether or not you may add an event
// record to the barectf writer, and then AddEventRecord() if you may.
//
// A barectf writer writes all its remaining event records on
// destruction.
//
// `PlatformDescrT` is the specific barectf platform descriptor (see the
// documentation of the `BarectfPlatform` class template).
template <typename PlatformDescrT> class BarectfWriter final {
  friend class BarectfTracer<PlatformDescrT>;

 public:
  // Specific barectf event record type.
  using EventRecord = BarectfEventRecord<typename PlatformDescrT::Ctx>;

 private:
  // Builds a barectf writer to write CTF packets of size `packet_size`
  // bytes to the CTF data stream file `data_stream_file_path`.
  //
  // The built barectf writer manages an event record queue having a
  // maximum size of `max_queue_size`.
  explicit BarectfWriter(const std::size_t packet_size,
                         const rocprofiler::common::filesystem::path& data_stream_file_path,
                         const std::size_t max_queue_size)
      : platform_{packet_size, data_stream_file_path, clock_val_},
        max_queue_size_{max_queue_size} {}

 public:
  // Writes all its remaining event records.
  ~BarectfWriter() {
    // Write all the remaining event records from the oldest to the
    // newest.
    while (!queue_.empty()) {
      WriteOldestEventRecord();
    }
  }

  // Disabled copy operations to make this class simpler.
  BarectfWriter(const BarectfWriter&) = delete;
  BarectfWriter& operator=(const BarectfWriter&) = delete;

  // Whether or not you may add the event record `event_record` to this
  // writer with AddEventRecord().
  bool MayAddEventRecord(const EventRecord& event_record) const noexcept {
    if (queue_.empty()) {
      return true;
    }

    // One may only add an event record if its clock value is greater
    // than or equal to the clock value of the most recently written
    // event record.
    return event_record.GetClockVal() >= clock_val_;
  }

  // Adds the event record `event_record` to this writer.
  //
  // `MayAddEventRecord(*event_record)` must return `true`.
  void AddEventRecord(typename EventRecord::SP event_record) {
    assert(MayAddEventRecord(*event_record) && "May add event record");

    // Add event record to queue.
    queue_.emplace(std::move(event_record));

    if (queue_.size() > max_queue_size_) {
      // Queue is too large: write the oldest event record now to
      // satisfy the requirement.
      WriteOldestEventRecord();
    }
  }

 private:
  // Comparison type for `queue_`.
  struct EventRecordQueueCompare final {
    bool operator()(const typename EventRecord::SP& left,
                    const typename EventRecord::SP& right) const noexcept {
      // "Greater than" so that the top element of the queue is the
      // oldest event record.
      return left->GetClockVal() > right->GetClockVal();
    }
  };

  // Oldest event record within `queue_`.
  //
  // `queue_` must not be empty.
  const EventRecord& GetOldestEventRecord() const noexcept {
    assert(!queue_.empty() && "Queue isn't empty");
    return *queue_.top();
  }

  // Writes the oldest event record through a barectf tracing function
  // and removes it from the event record queue.
  void WriteOldestEventRecord() {
    auto& oldest_event_record = GetOldestEventRecord();

    // When calling a barectf tracing function, it calls the clock value
    // accessor callback of the platform, which itself reads from
    // `clock_val_`.
    clock_val_ = oldest_event_record.GetClockVal();

    // Forward to a barectf tracing function.
    oldest_event_record.Write(platform_.GetCtx());

    // Remove from queue.
    queue_.pop();
  }

  // barectf platform (manages file I/O).
  BarectfPlatform<PlatformDescrT> platform_;

  // Current clock value for `platform_`.
  //
  // This is also the clock value of the most recently written event
  // record, therefore that MayAddEventRecord() can rely on this.
  std::uint64_t clock_val_ = 0;

  // Maximum size of `queue_` below.
  std::size_t max_queue_size_;

  // Event record queue.
  std::priority_queue<typename EventRecord::SP, std::vector<typename EventRecord::SP>,
                      EventRecordQueueCompare>
      queue_;
};

}  // namespace rocm_ctf

#endif  // PLUGIN_CTF_BARECTF_WRITER_H
