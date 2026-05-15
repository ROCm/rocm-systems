/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <cinttypes>
#include "impl/wddm/device.h"
#include "impl/wddm/queue.h"
#include "hsa-runtime/inc/amd_hsa_signal.h"

uint32_t get_vgpr_size_per_cu(HSA_ENGINE_ID id) {
  uint32_t vgpr_size = 0x40000;

  if (id.ui32.Major >= 11) {
    vgpr_size = 0x60000;
  }

  return vgpr_size;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCreateQueue(HSAuint32 NodeId,
					  HSA_QUEUE_TYPE Type,
					  HSAuint32 QueuePercentage,
					  HSA_QUEUE_PRIORITY Priority,
					  void *QueueAddress,
					  HSAuint64 QueueSizeInBytes,
					  HsaEvent *Event,
					  HsaQueueResource *QueueResource)
{
	if (Type == HSA_QUEUE_SDMA_BY_ENG_ID)
		return HSAKMT_STATUS_ERROR;

	return hsaKmtCreateQueueExt(NodeId, Type, QueuePercentage, Priority, 0,
				    QueueAddress, QueueSizeInBytes, Event,
				    QueueResource);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtCreateQueueExt(HSAuint32 NodeId,
					     HSA_QUEUE_TYPE Type,
					     HSAuint32 QueuePercentage,
					     HSA_QUEUE_PRIORITY Priority,
					     HSAuint32 SdmaEngineId,
					     void *QueueAddress,
					     HSAuint64 QueueSizeInBytes,
					     HsaEvent *Event,
					     HsaQueueResource *QueueResource) {
  HSAKMT_STATUS result;

  CHECK_DXG_OPEN();
  assert(Event == nullptr);

  if (Priority < HSA_QUEUE_PRIORITY_MINIMUM ||
      Priority > HSA_QUEUE_PRIORITY_MAXIMUM)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  wsl::thunk::WDDMDevice *device_ = get_wddmdev(NodeId);
  assert(device_);

  if (queue_acquire_buffer(QueueAddress) == false)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  switch (Type) {
  case HSA_QUEUE_COMPUTE_AQL: {
    assert(QueueResource->ErrorReason == nullptr);
    uint64_t pkg_num = QueueSizeInBytes / 64;
    uint32_t cmdbuf_size = device_->GetCmdbufSize();
    uint32_t queue_engine = device_->GetComputeEngine();
    bool use_hws = device_->IsHwsEnabled(queue_engine);
    auto queue_ = new wsl::thunk::ComputeQueue(
        device_, QueueAddress, pkg_num,
        reinterpret_cast<std::atomic<uint64_t> *>(
            QueueResource->Queue_write_ptr_aql),
        reinterpret_cast<std::atomic<uint64_t> *>(
            QueueResource->Queue_read_ptr_aql),
        QueueResource->ErrorReason, cmdbuf_size, queue_engine, use_hws);

    QueueResource->QueueId = reinterpret_cast<HSA_QUEUEID>(queue_);
    // for doorbell_signal.hardware_doorbell_ptr
    QueueResource->Queue_DoorBell_aql = queue_->GetDoorbellPtr();
  } break;
  case HSA_QUEUE_SDMA:
  case HSA_QUEUE_SDMA_BY_ENG_ID: {
    pr_debug("create sdma queue in engine %d\n", SdmaEngineId);
    uint32_t queue_engine = device_->GetSdmaEngine(0); // TODO: SdmaEngineId
    bool use_hws = device_->IsHwsEnabled(queue_engine);
    auto queue_ = new wsl::thunk::SDMAQueue(
		device_, QueueAddress, QueueSizeInBytes,
		queue_engine, use_hws);
    QueueResource->QueueId = reinterpret_cast<HSA_QUEUEID>(queue_);
    QueueResource->Queue_DoorBell_aql = queue_->GetDoorbellPtr();
    QueueResource->Queue_write_ptr_aql = queue_->GetRingWptr();
    QueueResource->Queue_read_ptr_aql = queue_->GetRingRptr();
  } break;
  default:
    assert(false);
    QueueResource->QueueId = 0;
    QueueResource->Queue_DoorBell = nullptr;
    break;
  }

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtUpdateQueue(
    HSA_QUEUEID QueueId, HSAuint32 QueuePercentage, HSA_QUEUE_PRIORITY Priority,
    void *QueueAddress, HSAuint64 QueueSize, HsaEvent *Event) {
  CHECK_DXG_OPEN();

  if (Priority < HSA_QUEUE_PRIORITY_MINIMUM ||
      Priority > HSA_QUEUE_PRIORITY_MAXIMUM)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  auto queue_ = reinterpret_cast<wsl::thunk::ComputeQueue *>(QueueId);
  if (!queue_)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtDestroyQueue(HSA_QUEUEID QueueId) {
  CHECK_DXG_OPEN();

  auto queue_ = reinterpret_cast<wsl::thunk::WDDMQueue *>(QueueId);
  void *QueueAddress = queue_->GetHsaQueueAddr();

  if (!queue_)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  delete queue_;
  queue_release_buffer(QueueAddress);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetQueueCUMask(HSA_QUEUEID QueueId,
                                             HSAuint32 CUMaskCount,
                                             HSAuint32 *QueueCUMask) {
  CHECK_DXG_OPEN();

  auto queue_ = reinterpret_cast<wsl::thunk::ComputeQueue *>(QueueId);
  if (!queue_)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  if (CUMaskCount == 0 || !QueueCUMask || ((CUMaskCount % 32) != 0))
    return HSAKMT_STATUS_INVALID_PARAMETER;

  pr_warn_once("not implemented\n");

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetQueueInfo(HSA_QUEUEID QueueId,
                                           HsaQueueInfo *QueueInfo) {
  CHECK_DXG_OPEN();

  if (QueueInfo == NULL)
    return HSAKMT_STATUS_INVALID_PARAMETER;
  memset(QueueInfo, 0, sizeof(*QueueInfo));

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSetTrapHandler(HSAuint32 Node,
                                             void *TrapHandlerBaseAddress,
                                             HSAuint64 TrapHandlerSizeInBytes,
                                             void *TrapBufferBaseAddress,
                                             HSAuint64 TrapBufferSizeInBytes) {
  CHECK_DXG_OPEN();
  pr_warn_once("not implemented\n");
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtAllocQueueGWS(HSA_QUEUEID QueueId, HSAuint32 nGWS,
                                            HSAuint32 *firstGWS) {
  CHECK_DXG_OPEN();

  auto queue_ = reinterpret_cast<wsl::thunk::ComputeQueue *>(QueueId);
  if (!queue_)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  assert(false);
  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtQueueRingDoorbell(HSA_QUEUEID QueueId, uint64_t value) {
  CHECK_DXG_OPEN();

  auto queue_ = reinterpret_cast<wsl::thunk::WDDMQueue *>(QueueId);
  if (!queue_)
    return HSAKMT_STATUS_INVALID_PARAMETER;

  queue_->RingDoorbell(value);
  return HSAKMT_STATUS_SUCCESS;
}
