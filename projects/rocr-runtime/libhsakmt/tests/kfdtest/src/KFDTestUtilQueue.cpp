/*
 * Copyright (C) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <algorithm>
#include <memory>
#include <vector>
#include <list>
#include "SDMAQueue.hpp"
#include "PM4Queue.hpp"
#include "SDMAPacket.hpp"
#include "PM4Packet.hpp"
#include "KFDTestUtil.hpp"
#include "KFDTestUtilQueue.hpp"
#include "KFDBaseComponentTest.hpp"

#define MB_PER_SEC(size, time) ((((size) * 1ULL) >> 20) * 1000ULL * 1000ULL * 1000ULL / (time))

class AsyncMPSQ;
class AsyncMPMQ;

typedef std::shared_ptr<AsyncMPSQ> sharedAsyncMPSQ;
typedef std::list<sharedAsyncMPSQ> AsyncMPSQList;

typedef std::shared_ptr<BasePacket> sharedPacket;
typedef std::list<sharedPacket> PacketList;

/* AsyncMPSQ is short for Async multiple packet single queue.
 * It is allowed to place a list of packets to run on one queue of the specified GPU node.
 */
class AsyncMPSQ {
    public:
        AsyncMPSQ() : m_queue(NULL), m_buf(NULL), m_event(NULL) { /*do nothing*/}

        virtual ~AsyncMPSQ(void) { Destroy(); }

        /* It is the main function to deal with the packet and queue.*/
        void PlacePacketOnNode(PacketList &packetList, int node, TSPattern tsp);

        /* Run the packets placed on nodes and return immediately.*/
        void Submit(void) { ASSERT_NE(m_queue, nullptr); m_queue->SubmitPacket(); }

        /* Return only when all packets are consumed.
         * If there is any packet issues some IO operations, wait these IO to complete too.
         */
        void Wait(void) {
            ASSERT_NE(m_queue, nullptr);
            m_queue->Wait4PacketConsumption(m_event, std::max((unsigned int)6000, g_TestTimeOut));
        }

        /* Report the time used between packet [begin, end) in Global Counter on success.
         * Return 0 on failure.
         */
        HSAuint64 Report(int indexOfPacketBegin = 0, int indexOfPacketEnd = 0);
        /* Report the timestamp around the packet.
         * Return the time used on success.
         * Return 0 on failure.
         */
        HSAuint64 Report(int indexOfPacket, HSAuint64 &tsBegin, HSAuint64 &tsEnd);

    private:
        BaseQueue *m_queue;
        HSA_QUEUE_TYPE m_queueType;
        HsaEvent *m_event;
        /* m_ts points to m_buf's memory.*/
        HsaMemoryBuffer *m_buf;
        TimeStamp *m_ts;
        unsigned m_ts_count;
        TSPattern m_ts_pattern;

        void AllocTimeStampBuf(int packetCount);
        void Destroy();

        /* It determines which queue will be created.*/
        void InitQueueType(PACKETTYPE packetType) {
            if (packetType == PACKETTYPE_SDMA)
                m_queueType = HSA_QUEUE_SDMA;
            else if (packetType == PACKETTYPE_PM4)
                m_queueType = HSA_QUEUE_COMPUTE;
            else
                WARN() << "Unsupported queue type!" << std::endl;
        }

        unsigned int TimePacketSize(void) {
            if (m_queueType == HSA_QUEUE_SDMA)
                return SDMATimePacket(0).SizeInBytes();
            else if (m_queueType == HSA_QUEUE_COMPUTE)
                return PM4ReleaseMemoryPacket(m_queue->GetFamilyId(), 0, 0, 0, 0, 0).SizeInBytes();
            return 0;
        }

        void CreateNewQueue(int node, unsigned int queueSize) {
            if (m_queueType == HSA_QUEUE_SDMA)
                m_queue = new SDMAQueue();
            else if (m_queueType == HSA_QUEUE_COMPUTE)
                m_queue = new PM4Queue();
            else {
                m_queue = NULL;
                WARN() << "Unsupported queue type!" << std::endl;
            }

            if (m_queue)
                ASSERT_SUCCESS(m_queue->Create(node, queueSize));
        }

        void PlaceTimestampPacket(void *addr) {
            if (m_queueType == HSA_QUEUE_SDMA)
                PlacePacket(SDMATimePacket(addr));
            else if (m_queueType == HSA_QUEUE_COMPUTE)
                PlacePacket(
                        PM4ReleaseMemoryPacket(m_queue->GetFamilyId(), true, (HSAuint64)addr, 0, true, true));
            else
                WARN() << "Unsupported queue type!" << std::endl;
        }

        void PlacePacket(const BasePacket &packet) {
            m_queue->PlacePacket(packet);
        }
};

void AsyncMPSQ::Destroy(void) {
    /* Delete queue first.*/
    if (m_queue) {
        delete m_queue;
    }

    if (m_buf)
        delete m_buf;

    if (m_event)
        hsaKmtDestroyEvent(m_event);
}

void AsyncMPSQ::AllocTimeStampBuf(int packetCount) {
    if (m_ts_pattern == NOTS) {
        m_buf = NULL;
        m_ts = NULL;
        m_ts_count = 0;
        return;
    }

    if (m_ts_pattern == ALLTS)
        /* One extra timestamp packet.*/
        m_ts_count = packetCount + 1;
    else
        m_ts_count = 2;

    /* One more timestamp space to fit with alignment.*/
    HSAuint64 size = ALIGN_UP(sizeof(TimeStamp) * (m_ts_count + 1), PAGE_SIZE);

    m_buf = new HsaMemoryBuffer(size, 0, true, false);

    TimeStamp *array = m_buf->As<TimeStamp*>();

    /* SDMATimePacket need 32bytes aligned boundary dst address*/
    m_ts = reinterpret_cast<TimeStamp *>ALIGN_UP(array, sizeof(TimeStamp));
}

void AsyncMPSQ::PlacePacketOnNode(PacketList &packets, int node, TSPattern tsp = ALLTS) {
    int nPacket = packets.size();

    if (nPacket == 0) {
        WARN() << "Empty packetList!" << std::endl;
        return;
    }

    /*1: All resources should be freed.*/
    Destroy();

    /*2: Must initialize queueType first.*/
    InitQueueType(packets.front()->PacketType());
    /*3: Initialize timestamp buf second with the pattern.*/
    m_ts_pattern = tsp;
    AllocTimeStampBuf(nPacket);
    /*4: Create a event for Wait().*/
    CreateQueueTypeEvent(false, false, node, &m_event);

    int i = -1;
    int packetSize = 0;
    /* Calculate the space to put all timestamp packet.*/
    int timePacketSize = TimePacketSize() * m_ts_count;
    /* Another one page space to put fence, trap, etc*/
    int extraPacketSize = PAGE_SIZE + timePacketSize;

    /* To calculate the total packet size we will need to create the queue.
     * As the packet in the vector might be different with each other,
     * we have no other way to calculate the queuesize.
     */
    for (auto &packet : packets)
        packetSize += packet->SizeInBytes();

    /* queueSize need be power of 2.*/
    const int queueSize = RoundToPowerOf2(packetSize + extraPacketSize);

    /*5: Create a new queue on node for the packets.*/
    CreateNewQueue(node, queueSize);

    if (tsp != NOTS) {
        i++;
        PlaceTimestampPacket(m_ts + i);
    }

    for (auto &packet : packets) {
        PlacePacket(*packet);
        if (tsp == ALLTS) {
            i++;
            PlaceTimestampPacket(m_ts + i);
        }
    }

    if (tsp == HEAD_TAIL) {
        i++;
        PlaceTimestampPacket(m_ts + i);
    }

    ASSERT_EQ(i + 1, m_ts_count);
}

HSAuint64 AsyncMPSQ::Report(int indexOfPacket, HSAuint64 &begin, HSAuint64 &end) {
    /* Should not get any timestamp if NOTS is specified.*/
    int error = 0;
    EXPECT_NE(m_ts_pattern, NOTS)
        << " Error " << ++error << ": No timestamp would be reported!" << std::endl;

    if (m_ts_pattern == HEAD_TAIL)
        indexOfPacket = 0;

    EXPECT_NE(m_ts, nullptr)
        << " Error " << ++error << ": No timestamp buf!" << std::endl;
    /* m_ts_count is equal to packets count + 1, see PlacePacketOnNode().
     * So the max index of a packet is m_ts_count - 2.
     * make it unsigned to defend any minus values.
     */
    EXPECT_GE(m_ts_count - 2, (unsigned)indexOfPacket)
        << " Error " << ++error << ": Index overflow!" << std::endl;

    if (error)
        return 0;

    begin = m_ts[indexOfPacket].timestamp;
    end = m_ts[indexOfPacket + 1].timestamp;
    return end - begin;
}

HSAuint64 AsyncMPSQ::Report(int indexOfPacketBegin, int indexOfPacketEnd) {
    HSAuint64 ts[4];
    int error = 0;

    if (indexOfPacketEnd == 0)
        indexOfPacketEnd = m_ts_count - 1;

    EXPECT_GT((unsigned)indexOfPacketEnd, (unsigned)indexOfPacketBegin)
        << " Error " << ++error << ": Index inverted!" << std::endl;

    if (error)
        return 0;
    /* Get the timestamps around the two packets.*/
    if (!Report(indexOfPacketBegin, ts[0], ts[1]))
        return 0;
    /* [begin, end)*/
    if (!Report(indexOfPacketEnd - 1, ts[2], ts[3]))
        return 0;

    EXPECT_GT(ts[3], ts[0])
        << " Waring: Might be wrong timestamp values!" << std::endl;

    return ts[3] - ts[0];
}

/* AsyncMPMQ is short for Async multiple packet multiple queue.
 * AsyncMPMQ manages a list of AsyncMPSQ.
 * So the packet can be running on multiple GPU nodes at same time.
 */

class AsyncMPMQ {
    public:
        AsyncMPMQ(void) { /* do nothing*/}

        virtual ~AsyncMPMQ(void) { /*do nothing*/}

        sharedAsyncMPSQ PlacePacketOnNode(PacketList &packetList, int node, TSPattern tsp = ALLTS) {
            /* Create a sharedAsyncMPSQ object and push it into the AsyncMPSQList.
             * As we might submit packet to same GPU nodes several times, AsyncMPSQ *
             * is returned to stand for the AsyncMPSQ it is created with
             */
            sharedAsyncMPSQ mpsq_ptr(new AsyncMPSQ);
            mpsq_ptr->PlacePacketOnNode(packetList, node, tsp);
            m_mpsqList.push_back(mpsq_ptr);
            return mpsq_ptr;
        }

        void Submit(void) {
            for (auto &mpsq : m_mpsqList)
                mpsq->Submit();
        }

        void Wait(void) {
            for (auto &mpsq : m_mpsqList)
                mpsq->Wait();
        }

    private:
        AsyncMPSQList m_mpsqList;
};


/*
 * SDMA queue helper functions.
 */

bool sort_SDMACopyParams(const SDMACopyParams &a1, const SDMACopyParams &a2) {
    if (a1.node != a2.node)
        return a1.node < a2.node;
    return a1.group < a2.group;
}

/*
 * Copy from src to dst with corresponding sDMA.
 * It will try to merge copy on same node into one queue unless
 * caller forbid it by setting mashup to 0 and SDMACopyParams::group to different values.
 * On condition of mashup is 1, it will re-sort array into mergeable state.
 * All mergeable copy will be placed together.
 * On condition os mashup is 0, it keeps array in original order.
 * It will merge nearby copy if they have same group and node anyway.
 */
void sdma_multicopy(std::vector<SDMACopyParams> &array, int mashup, TSPattern tsp) {
    int i, packet_index = 0, queue_index = 0;
    PacketList packetList;
    AsyncMPMQ obj;
    std::vector<sharedAsyncMPSQ> handle;

    /* Sort it and then reduce the amount of queues if caller permits.
     * We might change the order of array only here.
     */
    if (mashup)
        std::sort(array.begin(), array.end(), sort_SDMACopyParams);

    for (i = 0; i < array.size(); i++) {
        sharedPacket packet(new
                SDMACopyDataPacket(g_baseTest->GetFamilyIdFromNodeId(array[i].node), array[i].dst, array[i].src, array[i].size));
        packetList.push_back(packet);

        /* We put the real queue_id in local handle[] to reduce some assignment.*/
        array[i].queue_id = queue_index;
        /* Every queue has its packets with the index starts from 0.*/
        array[i].packet_id = packet_index++;

        /* If next copy is on same node and group, try to merge it into same queue.*/
        if (i + 1 < array.size() && array[i].node == array[i + 1].node
                                    && array[i].group == array[i + 1].group)
                continue;

        /* Now we have prepare one packetList, place packet into the queue on GPU node.*/
        queue_index++;
        handle.push_back(obj.PlacePacketOnNode(packetList, array[i].node, tsp));

        /* Prepare a new(empty) packetList.*/
        packetList.clear();

        /* Prepare a new(zero) packet index for the packets in the new queue.*/
        packet_index = 0;
    }

    obj.Submit();
    obj.Wait();

    if (tsp == NOTS)
        return;

    /* Get the time used by packet.*/
    for (i = 0; i < array.size(); i++)
        array[i].timeConsumption = (handle[array[i].queue_id])->Report(
                array[i].packet_id, array[i].timeBegin, array[i].timeEnd);
}

static
void sdma_multicopy_report(std::vector<SDMACopyParams> &array, HSAuint64 countPerGroup, std::stringstream *msg,
                                HSAuint64 &timeConsumptionMin, HSAuint64 &timeConsumptionMax,
                                HSAuint64 &totalSizeMin, HSAuint64 &totalSizeMax) {
    HSAuint64 begin, end;
    /* There can be different count of copies in different groups in the future.
     * But assume they are same now.
     */
    HSAuint64 group = array.size() / countPerGroup;
    HSAuint64 interval = -1;
    timeConsumptionMin = -1;
    timeConsumptionMax = 0;
    totalSizeMin = totalSizeMax = 0;

    /* Try to find out
     * 1) The max/min timeConsumption of one copy in all copies.
     * 2) The minimal average of timeConsumption of one packet in all copies.
     * And one char # or - stands for one interval, aka minimal average.
     * Say, one copy use 10ns with 10 copy packets. the other copy use 20ns
     * with 10 copy packets. So the interval is 1ns, the timeConsumption is 20ns.
     * So the ouput msg will be like
     * ########## //copy1 10ns
     * #---##----####### //copy2 20ns
     */
    for (int i = 0; i < group; i++) {
        HSAuint64 begin, end, base = i * countPerGroup;

        begin = array[base].timeBegin;
        end = array[base + countPerGroup - 1].timeEnd;

        if (begin == 0 && end == 0)
            continue;

        if (timeConsumptionMax < end - begin)
            timeConsumptionMax = end - begin;

        if (timeConsumptionMin > end - begin)
            timeConsumptionMin = end - begin;
    }

    interval = timeConsumptionMin / countPerGroup;

    /* Draw the timestamp event for each copy list.
     * - means still doing copy.
     * # means just finish one copy.
     */
    if (msg)
        for (int i = 0; i < group; i++) {
            HSAuint64 base = i * countPerGroup;
            HSAuint64 last = array[base].timeBegin;
            HSAuint64 timeConsumption;

            *msg << "[" << array[base].node << " : " << array[base].group << "] ";

            for (int j = 0; j < countPerGroup; j++) {
                timeConsumption = array[base + j].timeEnd - last;

                while (timeConsumption >= interval) {
                    timeConsumption -= interval;
                    last += interval;

                    if (timeConsumption >= interval)
                        *msg << "-";
                    else
                        *msg << "#";
                };
            }

            *msg << std::endl;
        }

    /* Try to find out
     * 1) The size of all copies in all queues.
     * 2) The size of the copies running within the same period in all queues.
     * We assume all packets begin to run at same time.
     */
    for (int i = 0; i < group; i++) {
        HSAuint64 base = i * countPerGroup;
        HSAuint64 time = 0;

        for (int j = 0; j < countPerGroup; j++) {
            totalSizeMax += array[base + j].size;

            if (time < timeConsumptionMin) {
                time += array[base + j].timeConsumption;
                totalSizeMin += array[base + j].size;
            }
        }
    }
}

/*
 * Do copy with corresponding sDMA.
 */
void
sdma_multicopy(SDMACopyParams *copyArray, int arrayCount,
                        HSAuint64 *minSpeed, HSAuint64 *maxSpeed, std::stringstream *msg) {
    const HSAuint64 countPerGroup = minSpeed || maxSpeed ? 100 : 1;
    std::vector<SDMACopyParams> array;
    HSAuint64 totalSizeMin, totalSizeMax, timeConsumptionMin, timeConsumptionMax;

    for (int i = 0; i < arrayCount; i++) {
        /* Each copy has its own queue.*/
        copyArray[i].group = i;
        for (int j = 0; j < countPerGroup; j++)
            array.push_back(copyArray[i]);
    }

    sdma_multicopy(array, 0, ALLTS);

    sdma_multicopy_report(array, countPerGroup, msg,
            timeConsumptionMin, timeConsumptionMax,
            totalSizeMin, totalSizeMax);

    if (minSpeed)
        *minSpeed = MB_PER_SEC(totalSizeMin, CounterToNanoSec(timeConsumptionMin));

    if (maxSpeed)
        *maxSpeed = MB_PER_SEC(totalSizeMax, CounterToNanoSec(timeConsumptionMax));
}

/*
 * PM4 queue helper functions.
 */
// TODO
