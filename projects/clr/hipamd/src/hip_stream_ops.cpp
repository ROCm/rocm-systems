/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "platform/command_utils.hpp"
#include <unordered_set>

namespace hip {
hipError_t ihipBatchMemOperation(hipStream_t stream, cl_command_type cmdType, unsigned int count,
                                 hipStreamBatchMemOpParams* paramArray, unsigned int flags) {
  if (paramArray == nullptr || flags != 0 || count == 0 || count > 256) {
    return hipErrorInvalidValue;
  }

  if (!hip::isValid(stream)) {
    return hipErrorContextIsDestroyed;
  }

  // Reject work submission on a stream whose owning ctx has been destroyed.
  CHECK_STREAM_DETACHED(stream);

  // Validate operations in paramArray
  for (unsigned int i = 0; i < count; i++) {
    // These operations are currently not supported
    if (paramArray[i].operation == hipStreamMemOpBarrier ||
        paramArray[i].operation == hipStreamMemOpFlushRemoteWrites) {
      return hipErrorInvalidValue;
    }
  }

  hip::Stream* hip_stream = hip::getStream(stream);
  amd::Command::EventWaitList waitList;

  // The batch mem-op kernel runs every op in the batch concurrently (one work-item per op)
  // with no ordering between them. A write to an address that an earlier op in the SAME batch
  // waits on (e.g. a barrier that writes flag=1, waits flag==1, then resets flag=0) therefore
  // races that wait: the reset can clear the flag before the waiting work-item observes the
  // set value, so the wait spins forever and the stream deadlocks (AIRUNTIME-2528). Split the
  // batch at each such write-after-wait dependency and enqueue the pieces in stream order, so a
  // segment's waits complete before a later segment writes the waited addresses. Batches with
  // no write-after-wait dependency are still issued as a single op (no behavior change).
  auto op_is_write = [](const hipStreamBatchMemOpParams& p) {
    return p.operation == hipStreamMemOpWriteValue32 ||
           p.operation == hipStreamMemOpWriteValue64;
  };
  auto op_is_wait = [](const hipStreamBatchMemOpParams& p) {
    return p.operation == hipStreamMemOpWaitValue32 ||
           p.operation == hipStreamMemOpWaitValue64;
  };
  auto op_addr = [&](const hipStreamBatchMemOpParams& p) -> const void* {
    return op_is_write(p) ? p.writeValue.address : p.waitValue.address;
  };

  auto enqueue_segment = [&](unsigned int start, unsigned int n) {
    amd::BatchMemoryOperationCommand* seg = new amd::BatchMemoryOperationCommand(
        *hip_stream, cmdType, n, flags, waitList, &paramArray[start],
        sizeof(hipStreamBatchMemOpParams));
    seg->enqueue();
    seg->release();
  };

  unsigned int segStart = 0;
  std::unordered_set<const void*> waitedAddrs;
  for (unsigned int i = 0; i < count; i++) {
    if (op_is_write(paramArray[i]) && waitedAddrs.count(op_addr(paramArray[i])) != 0) {
      enqueue_segment(segStart, i - segStart);
      segStart = i;
      waitedAddrs.clear();
    }
    if (op_is_wait(paramArray[i])) {
      waitedAddrs.insert(op_addr(paramArray[i]));
    }
  }
  enqueue_segment(segStart, count - segStart);
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

  // Reject work submission on a stream whose owning ctx has been destroyed.
  CHECK_STREAM_DETACHED(stream);

  amd::Memory* memory = getMemoryObject(hip::getCurrentDevice(), ptr, offset);
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
  } else if (cmdType == ROCCLR_COMMAND_STREAM_WRITE_VALUE) {
    switch (flags) {
      case hipStreamWriteValueDefault:
        outFlags = ROCCLR_STREAM_WRITE_VALUE_DEFAULT;
        break;
      case hipExtStreamWriteValueIncrement:
        outFlags = ROCCLR_STREAM_WRITE_VALUE_INCREMENT;
        break;
      case hipExtStreamWriteValueDecrement:
        outFlags = ROCCLR_STREAM_WRITE_VALUE_DECREMENT;
        break;
      default:
        return hipErrorInvalidValue;
        break;
    }
  } else {
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
                                          0,      // mask un-used set it to 0
                                          flags,
                                          sizeof(uint32_t)));
}

hipError_t hipStreamWriteValue64(hipStream_t stream, void* ptr, uint64_t value,
                                 unsigned int flags) {
  HIP_INIT_API(hipStreamWriteValue64, stream, ptr, value, flags);
  HIP_RETURN_DURATION(ihipStreamOperation(stream, ROCCLR_COMMAND_STREAM_WRITE_VALUE, ptr, value,
                                          0,      // mask un-used set it to 0
                                          flags,
                                          sizeof(uint64_t)));
}

hipError_t hipStreamBatchMemOp(hipStream_t stream, unsigned int count,
                               hipStreamBatchMemOpParams* paramArray, unsigned int flags) {
  HIP_INIT_API(hipStreamBatchMemOp, count, paramArray, flags);
  HIP_RETURN_DURATION(
      ihipBatchMemOperation(stream, ROCCLR_COMMAND_BATCH_STREAM, count, paramArray, flags));
}
}  // namespace hip
