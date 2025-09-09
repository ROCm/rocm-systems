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

#include "device/pal/palthreadtrace.hpp"
#include "device/pal/palvirtual.hpp"

namespace amd::pal {

PalThreadTraceReference* PalThreadTraceReference::Create(VirtualGPU& gpu) {
  Pal::Result result;

  // Create performance experiment
  Pal::PerfExperimentCreateInfo createInfo = {};

  createInfo.optionFlags.sampleInternalOperations = 1;
  createInfo.optionFlags.cacheFlushOnCounterCollection = 1;
  createInfo.optionFlags.sqShaderMask = 1;
  createInfo.optionValues.sampleInternalOperations = true;
  createInfo.optionValues.cacheFlushOnCounterCollection = true;
  createInfo.optionValues.sqShaderMask = Pal::PerfShaderMaskCs;

  size_t palExperSize = gpu.dev().iDev()->GetPerfExperimentSize(createInfo, &result);
  if (result != Pal::Result::Success) {
    return nullptr;
  }

  PalThreadTraceReference* memRef = new (palExperSize) PalThreadTraceReference(gpu);
  if (memRef != nullptr) {
    result = gpu.dev().iDev()->CreatePerfExperiment(createInfo, &memRef[1], &memRef->perfExp_);
    if (result != Pal::Result::Success) {
      memRef->release();
      return nullptr;
    }
  }

  return memRef;
}

PalThreadTraceReference::~PalThreadTraceReference() {
  // The thread trace object is always associated with a particular queue,
  // so we have to lock just this queue
  amd::ScopedLock lock(gpu_.execution());

  delete layout_;
  delete memory_;

  if (nullptr != iPerf()) {
    iPerf()->Destroy();
  }
}

bool PalThreadTraceReference::finalize() {
  Pal::Result result;

  iPerf()->Finalize();

  // Acquire GPU memory for the query from the pool and bind it.
  Pal::GpuMemoryRequirements gpuMemReqs = {};
  iPerf()->GetGpuMemoryRequirements(&gpuMemReqs);
  memory_ = new Memory(gpu().dev(), amd::alignUp(gpuMemReqs.size, gpuMemReqs.alignment));

  if (nullptr == memory_) {
    return false;
  }

  if (!memory_->create(Resource::Local)) {
    return false;
  }

  gpu_.queue(gpu_.engineID_).addMemRef(memory_->iMem());

  result = iPerf()->BindGpuMemory(memory_->iMem(), memory_->offset());

  if (result != Pal::Result::Success) {
    return false;
  }

  Pal::ThreadTraceLayout layout = {};
  iPerf()->GetThreadTraceLayout(&layout);

  size_t size =
      sizeof(Pal::ThreadTraceLayout) + (sizeof(Pal::ThreadTraceSeLayout) * (layout.traceCount - 1));
  layout_ = reinterpret_cast<Pal::ThreadTraceLayout*>(new char[size]);
  if (layout_ == nullptr) {
    return false;
  }

  layout_->traceCount = layout.traceCount;
  iPerf()->GetThreadTraceLayout(layout_);

  return true;
}

void PalThreadTraceReference::copyToUserBuffer(Memory* dstMem, uint seIndex) {
  amd::Coord3D srcOrigin(layout_->traces[seIndex].dataOffset, 0, 0);
  amd::Coord3D dstOrigin(0, 0, 0);
  amd::Coord3D size(dstMem->size(), 0, 0);

  gpu_.blitMgr().copyBuffer(*memory_, *dstMem, srcOrigin, dstOrigin, size, true);
}

ThreadTrace::~ThreadTrace() {
  if (palRef_ == nullptr) {
    return;
  }

  // Release the thread trace reference object
  palRef_->release();
}

bool ThreadTrace::create() {
  palRef_->retain();

  size_t se = 0;
  for (auto itMemObj = memObj_.begin(); itMemObj != memObj_.end(); ++itMemObj, ++se) {
    // Initialize the thread trace
    Pal::ThreadTraceInfo sqttInfo = {};
    sqttInfo.traceType = Pal::PerfTraceType::ThreadTrace;
    sqttInfo.instance = se;

    sqttInfo.optionFlags.bufferSize = 1;
    // PAL requires ThreadTrace buffer aligned to 4KB
    sqttInfo.optionValues.bufferSize =
        amd::alignUp(dev().getGpuMemory(*itMemObj)->size(), (0x1 << 12));
    sqttInfo.optionFlags.threadTraceTokenConfig = 1;
    sqttInfo.optionValues.threadTraceTokenConfig.tokenMask = Pal::ThreadTraceTokenTypeFlags::All;

    Pal::Result result = iPerf()->AddThreadTrace(sqttInfo);
    if (result != Pal::Result::Success) {
      return false;
    }
  }

  return true;
}

void ThreadTrace::populateUserMemory() {
  uint se = 0;
  for (auto itMemObj = memObj_.begin(); itMemObj != memObj_.end(); ++itMemObj, ++se) {
    palRef_->copyToUserBuffer(dev().getGpuMemory(*itMemObj), se);
  }
}

bool ThreadTrace::info(uint infoType, uint* info, uint infoSize) const {
  switch (infoType) {
    case CL_THREAD_TRACE_BUFFERS_SIZE: {
      if (infoSize < numSe_) {
        LogError("The amount of buffers should be equal to the amount of Shader Engines");
        return false;
      } else {
        uint se = 0;
        for (auto itMemObj = memObj_.begin(); itMemObj != memObj_.end(); ++itMemObj, ++se) {
          info[se] = dev().getGpuMemory(*itMemObj)->size();
        }
      }
      break;
    }
    default:
      LogError("Wrong ThreadTrace::getInfo parameter");
      return false;
  }
  return true;
}

}  // namespace amd::pal
