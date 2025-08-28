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

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "platform/command_utils.hpp"

namespace hip {
hipError_t ihipBatchMemOperation(hipStream_t stream, cl_command_type cmdType, unsigned int count,
                                 hipStreamBatchMemOpParams* paramArray, unsigned int flags) {
  if (paramArray == nullptr || flags != 0 || count == 0 || count > 256) {
    return hipErrorInvalidValue;
  }

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  // Validate operations in paramArray
  for (unsigned int i = 0; i < count; i++) {
    // These operations are currently not supported
    if (paramArray[i].operation == hipStreamMemOpBarrier || hipStreamMemOpFlushRemoteWrites) {
      return hipErrorInvalidValue;
    }
  }

  hip::Stream* hip_stream = hip::getStream(stream);
  amd::Command::EventWaitList waitList;

  amd::BatchMemoryOperationCommand* command = new amd::BatchMemoryOperationCommand(
      *hip_stream, cmdType, count, flags, waitList, paramArray, sizeof(hipStreamBatchMemOpParams));

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }
  command->enqueue();
  command->release();
  HIP_RETURN(hipSuccess);
}


hipError_t ihipStreamOperation(hipStream_t stream, cl_command_type cmdType, void* ptr,
                               uint64_t value, uint64_t mask, unsigned int flags,
                               size_t sizeBytes) {
  size_t offset = 0;
  unsigned int outFlags = 0;

  if (ptr == nullptr) {
    return hipErrorInvalidValue;
  }

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  amd::Memory* memory = getMemoryObject(ptr, offset);
  if (!memory) {
    return hipErrorInvalidValue;
  }

  // NOTE: 'mask' is only used in Wait operation, 'sizeBytes' is only used in Write operation
  // 'flags' for now used only for Wait, but in future there will usecases for Write too.

  if (cmdType == ROCCLR_COMMAND_STREAM_WAIT_VALUE) {
    // Stream Wait on AQL barrier-value type packet is only supported on SignalMemory objects
    if (GPU_STREAMOPS_CP_WAIT && (!(memory->getMemFlags() & ROCCLR_MEM_HSA_SIGNAL_MEMORY))) {
      return hipErrorInvalidValue;
    }
    switch (flags) {
      case hipStreamWaitValueGte:
        outFlags = ROCCLR_STREAM_WAIT_VALUE_GTE;
        break;
      case hipStreamWaitValueEq:
        outFlags = ROCCLR_STREAM_WAIT_VALUE_EQ;
        break;
      case hipStreamWaitValueAnd:
        outFlags = ROCCLR_STREAM_WAIT_VALUE_AND;
        break;
      case hipStreamWaitValueNor:
        outFlags = ROCCLR_STREAM_WAIT_VALUE_NOR;
        break;
      default:
        return hipErrorInvalidValue;
        break;
    }
  } else if (cmdType != ROCCLR_COMMAND_STREAM_WRITE_VALUE) {
    return hipErrorInvalidValue;
  }

  hip::Stream* hip_stream = hip::getStream(stream);
  amd::Command::EventWaitList waitList;

  amd::StreamOperationCommand* command =
      new amd::StreamOperationCommand(*hip_stream, cmdType, waitList, *memory->asBuffer(), value,
                                      mask, outFlags, offset, sizeBytes);

  if (command == nullptr) {
    return hipErrorOutOfMemory;
  }
  command->enqueue();
  command->release();
  return hipSuccess;
}

hipError_t hipStreamWaitValue32(hipStream_t stream, void* ptr, uint32_t value, unsigned int flags,
                                uint32_t mask) {
  HIP_INIT_API(hipStreamWaitValue32, stream, ptr, value, mask, flags);
  // NOTE: ptr corresponds to a HSA Signal memeory which is 64 bits.
  // 32 bit value and mask are converted to 64-bit values.
  HIP_RETURN_DURATION(ihipStreamOperation(stream, ROCCLR_COMMAND_STREAM_WAIT_VALUE, ptr, value,
                                          mask, flags, sizeof(uint32_t)));
}

hipError_t hipStreamWaitValue64(hipStream_t stream, void* ptr, uint64_t value, unsigned int flags,
                                uint64_t mask) {
  HIP_INIT_API(hipStreamWaitValue64, stream, ptr, value, mask, flags);
  HIP_RETURN_DURATION(ihipStreamOperation(stream, ROCCLR_COMMAND_STREAM_WAIT_VALUE, ptr, value,
                                          mask, flags, sizeof(uint64_t)));
}

hipError_t hipStreamWriteValue32(hipStream_t stream, void* ptr, uint32_t value,
                                 unsigned int flags) {
  HIP_INIT_API(hipStreamWriteValue32, stream, ptr, value, flags);
  HIP_RETURN_DURATION(ihipStreamOperation(stream, ROCCLR_COMMAND_STREAM_WRITE_VALUE, ptr, value,
                                          0,  // mask un-used set it to 0
                                          0,  // flags un-used for now set it to 0
                                          sizeof(uint32_t)));
}

hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value,
                                 unsigned int flags) {
  HIP_INIT_API(hipStreamWriteValue64, stream, ptr, value, flags);
  HIP_RETURN_DURATION(ihipStreamOperation(stream, ROCCLR_COMMAND_STREAM_WRITE_VALUE, ptr, value,
                                          0,  // mask un-used set it to 0
                                          0,  // flags un-used for now set it to 0
                                          sizeof(uint64_t)));
}

hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams* paramArray, unsigned int flags) {
  HIP_INIT_API(hipStreamBatchMemOp, count, paramArray, flags);
  HIP_RETURN_DURATION(
      ihipBatchMemOperation(stream, ROCCLR_COMMAND_BATCH_STREAM, count, paramArray, flags));
}
}  // namespace hip
