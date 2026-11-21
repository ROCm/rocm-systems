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
#include <map>
#include <memory>
#include <vector>
#include <list>
#include "SDMAQueue.hpp"
#include "PM4Queue.hpp"
#include "SDMAPacket.hpp"
#include "PM4Packet.hpp"
#include "Dispatch.hpp"
#include "ShaderStore.hpp"
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

        /* Takes the node rather than reading m_queue->GetFamilyId(): this is
         * called to size the queue, so it runs before CreateNewQueue().
         */
        unsigned int TimePacketSize(int node) {
            if (m_queueType == HSA_QUEUE_SDMA)
                return SDMATimePacket(0).SizeInBytes();
            else if (m_queueType == HSA_QUEUE_COMPUTE)
                return PM4ReleaseMemoryPacket(
                        g_baseTest->GetFamilyIdFromNodeId(node), 0, 0, 0, 0, 0).SizeInBytes();
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
        m_queue = NULL;
    }

    if (m_buf) {
        delete m_buf;
        m_buf = NULL;
    }

    if (m_event) {
        HSAKMT_CALL(hsaKmtDestroyEvent, g_baseTest->m_hsakmt_current_ctx, m_event);
        m_event = NULL;
    }
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
    int timePacketSize = TimePacketSize(node) * m_ts_count;
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
 * PM4 queue helper functions.
 *
 * A blit copy is a compute dispatch, so it reaches the queue as a single
 * PM4IndirectBufPacket and slots into the timestamp scheme above unchanged.
 * BuildIb() ends the IB with a ReleaseMem/WaitRegMem fence, so the CP does not
 * retire the IB packet until the dispatch has actually finished - which is what
 * makes the timestamp placed after it meaningful.
 */

/* One thread moves 16 bytes per iteration. See BlitCopyIsa. */
#define BLIT_CHUNK_BYTES        16
/* Threads per workgroup. Large enough to amortize workgroup launch, small
 * enough that a partition with few CUs still gets several workgroups each.
 */
#define BLIT_WORKGROUP_SIZE     256
/* Workgroups we aim to have resident per CU. One 256-thread group puts a
 * single wave on each of the CU's four SIMDs, which sounds like too few until
 * you look at what a wider grid does to the access pattern.
 *
 * The shader is grid-strided, so the stride is the whole grid: at four groups
 * per CU on a 228-CU node that is a 3.7MB step, and every one of the resulting
 * 3648 waves walks memory in strides that large. Nothing stays in an open DRAM
 * row and the streams thrash each other. Measured on MI300A, going from four
 * groups per CU to one takes a peer copy from 13.5 to 20 GB/s and a local one
 * from 14.4 to 35 GB/s.
 *
 * Occupancy is not what covers the memory latency here - the unrolled loop is,
 * by keeping four loads in flight per wave. See BlitCopyIsa.
 */
#define BLIT_WORKGROUPS_PER_CU  1
/* Fallback for topologies that do not report a usable CU count. */
#define BLIT_DEFAULT_CUS        16
/* Enough for BlitCopyIsa's highest register, v39: 40 VGPRs on gfx9, 80 on
 * gfx10+ in wave32. Costs no occupancy on CDNA, which allocates 512 VGPRs per
 * SIMD and caps at 8 waves either way.
 */
#define BLIT_VGPR_GRANULES      9

struct BlitGeometry {
    unsigned int wgSize;
    unsigned int log2WgSize;
    unsigned int gridThreads;
};

/* Size the grid to the node rather than to a fixed number, so the geometry
 * follows the partition: under DPX/QPX each logical node reports its own share
 * of the CUs and gets a correspondingly smaller grid.
 *
 * Grid size is a pure performance knob. The shader is grid-strided, so any grid
 * copies the whole buffer correctly; a bad one just costs GB/s.
 */
static BlitGeometry GetBlitGeometry(HSAuint32 node, HSAuint64 chunks) {
    HsaNodeProperties props = {0};
    BlitGeometry geo;
    HSAuint64 useful;
    unsigned int cus = BLIT_DEFAULT_CUS;

    geo.wgSize = BLIT_WORKGROUP_SIZE;
    for (geo.log2WgSize = 0; (1u << geo.log2WgSize) < geo.wgSize; geo.log2WgSize++)
        {}
    /* The shader derives the global id by shifting, so this has to be exact.*/
    EXPECT_EQ(1u << geo.log2WgSize, geo.wgSize)
        << "BLIT_WORKGROUP_SIZE must be a power of 2" << std::endl;

    if (HSAKMT_CALL(hsaKmtGetNodeProperties, g_baseTest->m_hsakmt_current_ctx, node, &props)
                == HSAKMT_STATUS_SUCCESS
            && props.NumSIMDPerCU && props.NumFComputeCores)
        cus = props.NumFComputeCores / props.NumSIMDPerCU;

    geo.gridThreads = cus * BLIT_WORKGROUPS_PER_CU * geo.wgSize;

    /* Launching threads with nothing to copy only costs launch time. Round up
     * to keep the grid a whole number of workgroups.
     */
    useful = ALIGN_UP(chunks, geo.wgSize);
    if (useful && useful < geo.gridThreads)
        geo.gridThreads = useful;

    return geo;
}

/* One blit copy: the arg buffers, the dispatch, and the IB it built.
 *
 * The IB cannot be replayed - BuildIb()'s fence works by having a ReleaseMem
 * write 0xdeadbeef into a NOP payload dword inside the IB itself, so a second
 * run would find the dword already set and skip the wait. Each placed packet
 * therefore gets its own BlitCopy.
 */
class BlitCopy {
    public:
        BlitCopy(HSAuint32 node, void *dst, void *src, HSAuint64 size,
                        const HsaMemoryBuffer &isaBuf)
                :m_args(PAGE_SIZE, node), m_params(PAGE_SIZE, node), m_dispatch(isaBuf) {
            HSAuint64 chunks = size / BLIT_CHUNK_BYTES;
            void **args = m_args.As<void **>();
            HSAuint32 *params = m_params.As<HSAuint32 *>();

            EXPECT_EQ(0ULL, size % BLIT_CHUNK_BYTES)
                << "Blit copy needs a size that is a multiple of "
                << BLIT_CHUNK_BYTES << " bytes" << std::endl;
            EXPECT_EQ(0ULL, (HSAuint64)src % BLIT_CHUNK_BYTES);
            EXPECT_EQ(0ULL, (HSAuint64)dst % BLIT_CHUNK_BYTES);
            /* The shader keeps its offsets in 32 bits. */
            EXPECT_LT(size, 1ULL << 32)
                << "Blit copy is limited to buffers under 4GB" << std::endl;

            BlitGeometry geo = GetBlitGeometry(node, chunks);

            args[0] = src;
            args[1] = dst;

            /* Precompute the loop geometry so the shader needs no division.
             * Every thread runs the same itersPerThread, which keeps the main
             * loop uniform; whatever does not divide evenly is left to the
             * guarded remainder pass.
             */
            params[0] = chunks / geo.gridThreads;                   // itersPerThread
            params[1] = geo.gridThreads * BLIT_CHUNK_BYTES;         // strideBytes
            params[2] = chunks % geo.gridThreads;                   // remChunks
            params[3] = params[0] * params[1];                      // remBaseByte
            params[4] = geo.log2WgSize;

            m_dispatch.SetArgs(args, params);
            /* BlitCopyIsa reaches v39, past the 20 VGPRs a dispatch hands out
             * by default. Its unrolled loop needs the registers: they are what
             * lets it keep four loads in flight per wave instead of one.
             */
            m_dispatch.SetVgprGranules(BLIT_VGPR_GRANULES);
            m_dispatch.SetWorkgroupSize(geo.wgSize, 1, 1);
            /* SetDim() counts threads, not workgroups - DISPATCH_INITIATOR
             * sets USE_THREAD_DIMENSIONS.
             */
            m_dispatch.SetDim(geo.gridThreads, 1, 1);

            m_ib = m_dispatch.PrepareIb();
        }

        sharedPacket Packet(void) { return sharedPacket(new PM4IndirectBufPacket(m_ib)); }

    private:
        HsaMemoryBuffer m_args;
        HsaMemoryBuffer m_params;
        Dispatch m_dispatch;
        IndirectBuffer *m_ib;
};

/* The shader text is the same for every copy on a node, so assemble it once.
 * The cache is per call rather than static: the buffers are GPU mappings, and
 * they must not outlive the KFD context the test opened them under.
 */
typedef std::map<HSAuint32, std::shared_ptr<HsaMemoryBuffer>> BlitIsaCache;

static const HsaMemoryBuffer &GetBlitIsa(HSAuint32 node, BlitIsaCache &cache) {
    BlitIsaCache::iterator it = cache.find(node);

    if (it == cache.end()) {
        std::shared_ptr<HsaMemoryBuffer> buf(new HsaMemoryBuffer(PAGE_SIZE, node, true, false, true));
        Assembler *asmb = g_baseTest->GetAssemblerFromNodeId(node);

        EXPECT_NE(asmb, nullptr) << "No assembler for node " << node << std::endl;
        if (asmb)
            EXPECT_SUCCESS(asmb->RunAssembleBuf(BlitCopyIsa, buf->As<char *>()));

        it = cache.insert(std::make_pair(node, buf)).first;
    }

    return *it->second;
}

/*
 * Copy queue helper functions, shared by both engines.
 */

bool sort_GpuCopyParams(const GpuCopyParams &a1, const GpuCopyParams &a2) {
    if (a1.node != a2.node)
        return a1.node < a2.node;
    return a1.group < a2.group;
}

/*
 * Copy from src to dst with the sDMA engines or a CU (blit) copy kernel.
 * It will try to merge copy on same node into one queue unless
 * caller forbid it by setting mashup to 0 and GpuCopyParams::group to different values.
 * On condition of mashup is 1, it will re-sort array into mergeable state.
 * All mergeable copy will be placed together.
 * On condition os mashup is 0, it keeps array in original order.
 * It will merge nearby copy if they have same group and node anyway.
 *
 * Only packet construction depends on the engine. Queue placement, submission,
 * waiting and reporting are shared.
 */
void gpu_multicopy(std::vector<GpuCopyParams> &array, CopyEngine engine, int mashup, TSPattern tsp) {
    int i, packet_index = 0, queue_index = 0;
    PacketList packetList;
    /* Declared before the dispatches so it outlives them: Dispatch keeps a
     * reference to its ISA buffer.
     */
    BlitIsaCache isaCache;
    /* The blit resources back the IBs the queues are still reading, so they
     * have to stay alive until after Wait().
     */
    std::vector<std::shared_ptr<BlitCopy>> blits;
    AsyncMPMQ obj;
    std::vector<sharedAsyncMPSQ> handle;

    /* Sort it and then reduce the amount of queues if caller permits.
     * We might change the order of array only here.
     */
    if (mashup)
        std::sort(array.begin(), array.end(), sort_GpuCopyParams);

    for (i = 0; i < array.size(); i++) {
        sharedPacket packet;

        if (engine == COPY_BLIT) {
            std::shared_ptr<BlitCopy> blit(new BlitCopy(array[i].node, array[i].dst, array[i].src,
                        array[i].size, GetBlitIsa(array[i].node, isaCache)));
            packet = blit->Packet();
            blits.push_back(blit);
        } else {
            packet = sharedPacket(new SDMACopyDataPacket(
                        g_baseTest->GetFamilyIdFromNodeId(array[i].node),
                        array[i].dst, array[i].src, array[i].size));
        }

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
void copy_report(std::vector<GpuCopyParams> &array, HSAuint64 countPerGroup, std::stringstream *msg,
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
 * Do copy with the requested engine and report the bandwidth achieved.
 */
void
gpu_multicopy(GpuCopyParams *copyArray, int arrayCount, CopyEngine engine,
                        HSAuint64 *minSpeed, HSAuint64 *maxSpeed, std::stringstream *msg) {
    /* A blit copy costs an IB and an HsaEvent per repeat, where an SDMA copy is
     * just a packet, so it repeats fewer times. Even 10 repeats of a 32MB copy
     * per queue is already 320MB of traffic.
     */
    const HSAuint64 repeats = engine == COPY_BLIT ? 10 : 100;
    const HSAuint64 countPerGroup = minSpeed || maxSpeed ? repeats : 1;
    std::vector<GpuCopyParams> array;
    HSAuint64 totalSizeMin, totalSizeMax, timeConsumptionMin, timeConsumptionMax;

    for (int i = 0; i < arrayCount; i++) {
        /* Each copy has its own queue.*/
        copyArray[i].group = i;
        for (int j = 0; j < countPerGroup; j++)
            array.push_back(copyArray[i]);
    }

    gpu_multicopy(array, engine, 0, ALLTS);

    copy_report(array, countPerGroup, msg,
            timeConsumptionMin, timeConsumptionMax,
            totalSizeMin, totalSizeMax);

    if (minSpeed)
        *minSpeed = MB_PER_SEC(totalSizeMin, CounterToNanoSec(timeConsumptionMin));

    if (maxSpeed)
        *maxSpeed = MB_PER_SEC(totalSizeMax, CounterToNanoSec(timeConsumptionMax));
}
