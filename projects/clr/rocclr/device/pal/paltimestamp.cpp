/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#include "os/os.hpp"
#include "platform/perfctr.hpp"
#include "device/pal/paldefs.hpp"
#include "device/pal/paltimestamp.hpp"
#include "device/pal/palvirtual.hpp"
#include "device/pal/palcounters.hpp"

namespace amd::pal {

TimeStamp::TimeStamp(const VirtualGPU& gpu, Pal::IGpuMemory* iMem, uint memOffset, address cpuAddr)
    : gpu_(gpu), iMem_(iMem), memOffset_(memOffset) {
  values_ = reinterpret_cast<volatile uint64_t*>(cpuAddr + memOffset);
}

TimeStamp::~TimeStamp() {}

void TimeStamp::begin() {
  if (!flags_.beginIssued_) {
    gpu().iCmd()->CmdWriteTimestamp(Pal::PipelineStageFlag::PipelineStageBottomOfPipe, *iMem_,
                                    memOffset_ + CommandStartTime * sizeof(uint64_t));
    flags_.beginIssued_ = true;
  }
}

void TimeStamp::end() {
  CondLog(!flags_.beginIssued_, "We didn't issue a begin operation!");
  gpu().iCmd()->CmdWriteTimestamp(Pal::PipelineStageFlag::PipelineStageBottomOfPipe, *iMem_,
                                  memOffset_ + CommandEndTime * sizeof(uint64_t));
  flags_.endIssued_ = true;
}

inline void SetValue(uint64_t* time, uint64_t val, double nanos) {
  *time = static_cast<uint64_t>(static_cast<double>(val) * nanos);
}

void TimeStamp::value(uint64_t* startTime, uint64_t* endTime) {
  CondLog(!flags_.endIssued_, "We didn't send the counter end operation!");
  //! @todo optimize!
  const double NanoSecondsPerTick = 1000000000.0 / (gpu_.dev().properties().timestampFrequency);

  SetValue(startTime, values_[CommandStartTime], NanoSecondsPerTick);
  SetValue(endTime, values_[CommandEndTime], NanoSecondsPerTick);
}

TimeStampCache::~TimeStampCache() {
  // Release all time stamp objects from the cache
  for (uint i = 0; i < freedTS_.size(); ++i) {
    delete freedTS_[i];
  }
  freedTS_.clear();

  // Release all memory objects
  for (uint i = 0; i < tsBuf_.size(); ++i) {
    tsBuf_[i]->unmap(&gpu_);
    gpu_.queue(MainEngine).removeMemRef(tsBuf_[i]->iMem());
    if (!gpu_.dev().settings().disableSdma_) {
      gpu_.queue(SdmaEngine).removeMemRef(tsBuf_[i]->iMem());
    }
    delete tsBuf_[i];
  }
  tsBuf_.clear();
}

TimeStamp* TimeStampCache::allocTimeStamp() {
  TimeStamp* ts = nullptr;
  if (0 != freedTS_.size()) {
    ts = freedTS_.back();
    freedTS_.pop_back();
  }

  if (nullptr == ts) {
    if ((tsBufCpu_ == nullptr) || ((tsOffset_ + TimerSlotSize) > TimerBufSize)) {
      Memory* buf = new Memory(gpu_.dev(), TimerBufSize);
      if (buf == nullptr || !buf->create(Resource::Remote)) {
        return nullptr;
      }
      gpu_.queue(MainEngine).addMemRef(buf->iMem());
      if (!gpu_.dev().settings().disableSdma_) {
        gpu_.queue(SdmaEngine).addMemRef(buf->iMem());
      }
      tsBufCpu_ = reinterpret_cast<address>(buf->map(&gpu_));
      memset(tsBufCpu_, 0, TimerBufSize);
      tsOffset_ = 0;
      tsBuf_.push_back(buf);
    }
    // Allocate a TimeStamp object
    ts = new TimeStamp(gpu_, tsBuf_[(tsBuf_.size() - 1)]->iMem(), tsOffset_, tsBufCpu_);
    // Create a timestamp
    if (ts == nullptr) {
      return nullptr;
    }
    tsOffset_ += TimerSlotSize;
  }

  // Set this timestamp into DRM profile mode if it was requested
  ts->clearStates();

  return ts;
}

}  // namespace amd::pal
