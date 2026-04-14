/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "BaseQueue.hpp"
#include "SDMAQueue.hpp"
#include "PM4Queue.hpp"
#include "AqlQueue.hpp"
#include "hsakmt/hsakmt.h"
#include "KFDBaseComponentTest.hpp"

BaseQueue::BaseQueue()
    :m_QueueBuf(NULL),
    m_SkipWaitConsumption(true) {
}

BaseQueue::~BaseQueue(void) {
    Destroy();
}

HSAKMT_STATUS BaseQueue::Create(unsigned int NodeId, unsigned int size, HSAuint64 *pointers) {
    HSAKMT_STATUS status;
    HSA_QUEUE_TYPE type = GetQueueType();

    if (m_QueueBuf != NULL) {
        // Queue already exists, one queue per object
        Destroy();
    }

    memset(&m_Resources, 0, sizeof(m_Resources));

    m_QueueBuf = new HsaMemoryBuffer(size, NodeId, true/*zero*/, false/*local*/, true/*exec*/,
                        /*isScratch */ false, /* isReadOnly */false, /* isUncached */true);

    if (type == HSA_QUEUE_COMPUTE_AQL) {
        m_Resources.Queue_read_ptr_aql = &pointers[0];
        m_Resources.Queue_write_ptr_aql = &pointers[1];
    }

    if (type == HSA_QUEUE_SDMA_BY_ENG_ID)
        status = hsaKmtCreateQueueExt(NodeId,
                                      type,
                                      DEFAULT_QUEUE_PERCENTAGE,
                                      DEFAULT_PRIORITY,
                                      m_SdmaEngineId,
                                      m_QueueBuf->As<unsigned int*>(),
                                      m_QueueBuf->Size(),
                                      NULL,
                                      &m_Resources);
    else
        status = hsaKmtCreateQueue(NodeId,
                                   type,
                                   DEFAULT_QUEUE_PERCENTAGE,
                                   DEFAULT_PRIORITY,
                                   m_QueueBuf->As<unsigned int*>(),
                                   m_QueueBuf->Size(),
                                   NULL,
                                   &m_Resources);

    if (status != HSAKMT_STATUS_SUCCESS) {
        return status;
    }

    if (m_Resources.Queue_read_ptr  == NULL) {
        WARN() << "CreateQueue: read pointer value should be 0" << std::endl;
        status = HSAKMT_STATUS_ERROR;
    }

    if (m_Resources.Queue_write_ptr  == NULL) {
        WARN() << "CreateQueue: write pointer value should be 0" << std::endl;
        status = HSAKMT_STATUS_ERROR;
    }

    // Needs to match the queue write ptr
    m_pendingWptr = 0;
    m_pendingWptr64 = 0;
    m_Node = NodeId;
    m_FamilyId = g_baseTest->GetFamilyIdFromNodeId(NodeId);
    return status;
}

HSAKMT_STATUS BaseQueue::Update(unsigned int percent, HSA_QUEUE_PRIORITY priority, bool nullifyBuffer) {
    void* pNewBuffer = (nullifyBuffer ? NULL : m_QueueBuf->As<void*>());
    HSAuint64 newSize = (nullifyBuffer ? 0 : m_QueueBuf->Size());

    return hsaKmtUpdateQueue(m_Resources.QueueId, percent, priority, pNewBuffer, newSize, NULL);
}

HSAKMT_STATUS BaseQueue::SetCUMask(unsigned int *mask, unsigned int mask_count) {
    return hsaKmtSetQueueCUMask(m_Resources.QueueId, mask_count, mask);
}

HSAKMT_STATUS BaseQueue::Destroy() {
    HSAKMT_STATUS status =  HSAKMT_STATUS_SUCCESS;

    if (m_QueueBuf != NULL) {
        status = hsaKmtDestroyQueue(m_Resources.QueueId);

        if (status == HSAKMT_STATUS_SUCCESS) {
            delete m_QueueBuf;
            m_QueueBuf = NULL;
        }
    }

    return status;
}

void BaseQueue::PlaceAndSubmitPacket(const BasePacket &packet) {
    PlacePacket(packet);
    SubmitPacket();
}

void BaseQueue::Wait4PacketConsumption(HsaEvent *event, unsigned int timeOut) {
    ASSERT_TRUE(!event) << "Not supported!" << std::endl;
    ASSERT_TRUE(WaitOnValue(m_Resources.Queue_read_ptr, RptrWhenConsumed(), timeOut));
}

bool BaseQueue::AllPacketsSubmitted() {
    return Wptr() == Rptr();
}

void BaseQueue::PlacePacket(const BasePacket &packet) {
    ASSERT_EQ(packet.PacketType(), PacketTypeSupported())
        << "Cannot add a packet since packet type doesn't match queue";

    unsigned int readPtr = Rptr();
    unsigned int writePtr = m_pendingWptr;
    HSAuint64 writePtr64 = m_pendingWptr64;

    unsigned int packetSizeInDwords = packet.SizeInDWords();
    unsigned int dwordsRequired = packetSizeInDwords;
    unsigned int queueSizeInDWord = m_QueueBuf->Size() / sizeof(uint32_t);

    if (writePtr + packetSizeInDwords > queueSizeInDWord) {
        // Wraparound expected. We need enough room to also place NOPs to avoid crossing the buffer end.
        dwordsRequired +=  queueSizeInDWord - writePtr;
    }

    unsigned int dwordsAvailable = (readPtr - 1 - writePtr + queueSizeInDWord) % queueSizeInDWord;
    ASSERT_GE(dwordsAvailable, dwordsRequired) << "Cannot add a packet, buffer overrun";

    ASSERT_GE(queueSizeInDWord, packetSizeInDwords) << "Cannot add a packet, packet size too large";

    if (writePtr + packetSizeInDwords >= queueSizeInDWord) {
        // Wraparound
        while (writePtr + packetSizeInDwords > queueSizeInDWord) {
            m_QueueBuf->As<unsigned int *>()[writePtr] = CMD_NOP;
            writePtr = (writePtr + 1) % queueSizeInDWord;
            writePtr64++;
        }

        // Not updating Wptr since we might want to place the packet without submission
        m_pendingWptr = (writePtr % queueSizeInDWord);
        m_pendingWptr64 = writePtr64;
    }

    memcpy(m_pendingWptr + m_QueueBuf->As<unsigned int*>(), packet.GetPacket(), packetSizeInDwords * 4);

    m_pendingWptr = (m_pendingWptr + packetSizeInDwords) % queueSizeInDWord;
    m_pendingWptr64 += packetSizeInDwords;
}

BaseQueue* QueueArray::GetQueue(unsigned int Node) {
    // If a queue exists for that node then return, else create one
    for (unsigned int i = 0; i < m_QueueList.size(); i++) {
        if (Node == m_QueueList.at(i)->GetNodeId())
            return m_QueueList.at(i);
    }

    BaseQueue *pQueue = NULL;

    switch (m_QueueType) {
    case HSA_QUEUE_COMPUTE:
        pQueue = new PM4Queue();
        break;
    case HSA_QUEUE_SDMA:
        pQueue = new SDMAQueue();
        break;
    case HSA_QUEUE_COMPUTE_AQL:
        pQueue = new AqlQueue();
        break;
    default:
        return NULL;
    }

    if (pQueue) {
        pQueue->Create(Node);
        m_QueueList.push_back(pQueue);
    }
    return pQueue;
}

void QueueArray::Destroy() {
    for (unsigned int i = 0; i < m_QueueList.size(); i++)
        delete m_QueueList.at(i);

    m_QueueList.clear();
}
