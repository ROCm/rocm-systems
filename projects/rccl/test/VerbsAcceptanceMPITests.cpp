/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file VerbsAcceptanceMPITests.cpp
 * @brief Hardware-agnostic RDMA/verbs acceptance (regression) tests.
 *
 * Each test targets a specific low-level NIC/driver code path that RCCL's
 * network transport depends on. A test that flips from PASS to FAIL after a
 * firmware or driver update is an unambiguous regression signal.
 *
 * The suite is MPI-first: MPI is both the execution model and the OOB channel
 * (MPI_Sendrecv carries the RDMA metadata that the standalone reference would
 * have exchanged over TCP). It is vendor-agnostic: it auto-selects the first
 * active IB device and SKIPs (coordinated across ranks) whenever a capability
 * is unavailable, rather than failing.
 *
 * Scaling model: ranks are paired lower-half <-> upper-half. With world size N,
 * rank r's peer is r +/- N/2; the lower half acts as receiver ("server"), the
 * upper half as sender ("client"). This yields cross-node pairs for multi-node
 * launches (e.g. 16 ranks @ 8/node -> 0..7 <-> 8..15) and intra-node loopback
 * pairs for single-node launches.
 *
 * This file currently covers the baseline connectivity and DMABUF data-path
 * tests (UD/RC support, RDMA Write and Read over DMABUF-registered GPU memory);
 * further acceptance tests build on the same fixture.
 */

#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#ifdef MPI_TESTS_ENABLED

#include <infiniband/verbs.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <hip/hip_runtime.h>

#include <arpa/inet.h>
#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace
{

// ---- Test parameters -------------------------------------------------------
constexpr size_t   kMsgSize       = 4096;
constexpr int      kCqDepth       = 32;
constexpr int      kQpDepth       = 16;
constexpr int      kPollTimeoutMs = 5000;
constexpr int      kMinRanks      = 2;

// OOB tags for MPI exchanges.
constexpr int kTagQpInfo    = 7001;
constexpr int kTagRemoteBuf = 7002;
constexpr int kTagResult    = 7003;
constexpr int kTagFlag      = 7004;

// ---- HIP allocation modes (7) ---------------------------------------------
enum AllocMode
{
    ALLOC_DEVICE = 0,
    ALLOC_HOST_DEFAULT,
    ALLOC_HOST_WC,
    ALLOC_HOST_COHERENT,
    ALLOC_HOST_NONCOHERENT,
    ALLOC_FINEGRAIN,
    ALLOC_MANAGED,
    ALLOC_COUNT
};

// ---- IB registration modes (3) --------------------------------------------
enum RegMode
{
    REG_STD = 0,
    REG_DMABUF_V1,
    REG_DMABUF_V2_PCIE,
    REG_COUNT
};

// ---- Optional symbols resolved at runtime (graceful skip if absent) --------
typedef hsa_status_t (*fn_export_dmabuf_v1_t)(const void*, size_t, int*, uint64_t*);
typedef hsa_status_t (*fn_export_dmabuf_v2_t)(const void*, size_t, int*, uint64_t*, uint64_t);
typedef struct ibv_mr* (*fn_reg_dmabuf_mr_t)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int);

fn_export_dmabuf_v1_t g_dmabuf_v1        = nullptr;
fn_export_dmabuf_v2_t g_dmabuf_v2        = nullptr;
fn_reg_dmabuf_mr_t    g_reg_dmabuf       = nullptr;
bool                  g_symbols_resolved = false;

void resolveOptionalSymbols()
{
    if(g_symbols_resolved)
        return;
    g_symbols_resolved = true;

    void* hsa = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if(hsa)
    {
        g_dmabuf_v1 = reinterpret_cast<fn_export_dmabuf_v1_t>(
            dlsym(hsa, "hsa_amd_portable_export_dmabuf"));
        g_dmabuf_v2 = reinterpret_cast<fn_export_dmabuf_v2_t>(
            dlsym(hsa, "hsa_amd_portable_export_dmabuf_v2"));
    }
    void* ibv = dlopen("libibverbs.so.1", RTLD_LAZY | RTLD_GLOBAL);
    if(ibv)
        g_reg_dmabuf = reinterpret_cast<fn_reg_dmabuf_mr_t>(dlsym(ibv, "ibv_reg_dmabuf_mr"));
}

// Wire-format structs exchanged over MPI OOB.
struct QpInfo
{
    uint32_t qpn;
    uint32_t psn;
    uint16_t lid;
    uint8_t  gid[16];
};

struct RemoteBuf
{
    uint64_t addr;
    uint32_t rkey;
    uint32_t pad;
};

struct RegResult
{
    struct ibv_mr* mr     = nullptr;
    int            dma_fd = -1;
};

// Reduce a local boolean "supported/pass" flag across all ranks so every rank
// makes the same skip/continue decision and no collective deadlocks.
bool allRanksAgree(bool localOk)
{
    int local  = localOk ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return global != 0;
}

} // namespace

// ===========================================================================
// Fixture
// ===========================================================================
class VerbsAcceptanceMPITest : public MPITestBase
{
protected:
    struct ibv_context* ctx_     = nullptr;
    struct ibv_pd*      pd_      = nullptr;
    struct ibv_cq*      cq_      = nullptr;
    struct ibv_qp*      qp_      = nullptr;
    int                 ibPort_  = 1;
    int                 gidIdx_  = 3; // RoCE v2 default; override via NCCL_IB_GID_INDEX
    QpInfo              localQp_ = {};

    int  worldRank_   = 0;
    int  worldSize_   = 0;
    int  peerRank_    = -1;
    bool isLowerHalf_ = false; // lower half == receiver/server role

    struct ibv_device_attr devAttr_ = {};

    void SetUp() override
    {
        MPITestBase::SetUp();
        worldRank_ = MPIEnvironment::world_rank;
        worldSize_ = MPIEnvironment::world_size;

        if(const char* g = std::getenv("NCCL_IB_GID_INDEX"))
        {
            int v = std::atoi(g);
            if(v >= 0)
                gidIdx_ = v;
        }

        // Pairing: lower half <-> upper half.
        isLowerHalf_ = (worldRank_ < worldSize_ / 2);
        peerRank_    = isLowerHalf_ ? worldRank_ + worldSize_ / 2 : worldRank_ - worldSize_ / 2;
    }

    void TearDown() override
    {
        (void)hipDeviceSynchronize();
        destroyQp();
        if(cq_)
        {
            ibv_destroy_cq(cq_);
            cq_ = nullptr;
        }
        if(pd_)
        {
            ibv_dealloc_pd(pd_);
            pd_ = nullptr;
        }
        if(ctx_)
        {
            ibv_close_device(ctx_);
            ctx_ = nullptr;
        }
        MPITestBase::TearDown();
    }

    // Even world size and >= kMinRanks; coordinated skip otherwise.
    bool requireEvenPairs()
    {
        if(!validateTestPrerequisites(kMinRanks))
            return false;
        if((worldSize_ % 2) != 0)
            return false;
        return true;
    }

    // Open the first IB device with an ACTIVE port. Vendor-agnostic. Allocates
    // pd_ and cq_. Returns false (locally) if nothing usable; callers reduce.
    bool openIbDevice()
    {
        int          nDevs = 0;
        ibv_device** devs  = ibv_get_device_list(&nDevs);
        if(!devs || nDevs == 0)
            return false;
        SCOPE_EXIT(ibv_free_device_list(devs));

        for(int i = 0; i < nDevs && !ctx_; ++i)
        {
            ibv_context* c = ibv_open_device(devs[i]);
            if(!c)
                continue;
            ibv_port_attr pa{};
            if(ibv_query_port(c, ibPort_, &pa) == 0 && pa.state == IBV_PORT_ACTIVE)
            {
                ctx_ = c;
            }
            else
            {
                ibv_close_device(c);
            }
        }
        if(!ctx_)
            return false;

        if(ibv_query_device(ctx_, &devAttr_) != 0)
            return false;

        pd_ = ibv_alloc_pd(ctx_);
        if(!pd_)
            return false;
        cq_ = ibv_create_cq(ctx_, kCqDepth, nullptr, nullptr, 0);
        if(!cq_)
            return false;
        return true;
    }

    // Create a QP of the given type in INIT state, capture local QpInfo.
    bool createQp(enum ibv_qp_type type)
    {
        ibv_qp_init_attr qia{};
        qia.send_cq          = cq_;
        qia.recv_cq          = cq_;
        qia.qp_type          = type;
        qia.cap.max_send_wr  = kQpDepth;
        qia.cap.max_recv_wr  = kQpDepth;
        qia.cap.max_send_sge = 1;
        qia.cap.max_recv_sge = 1;
        qp_                  = ibv_create_qp(pd_, &qia);
        if(!qp_)
            return false;

        ibv_qp_attr qa{};
        qa.qp_state   = IBV_QPS_INIT;
        qa.pkey_index = 0;
        qa.port_num   = static_cast<uint8_t>(ibPort_);
        int flags     = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT;
        if(type == IBV_QPT_RC)
        {
            qa.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
                                 | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
            flags |= IBV_QP_ACCESS_FLAGS;
        }
        else // UD
        {
            qa.qkey = 0x11111111U;
            flags |= IBV_QP_QKEY;
        }
        if(ibv_modify_qp(qp_, &qa, flags) != 0)
            return false;

        ibv_port_attr pa{};
        ibv_query_port(ctx_, ibPort_, &pa);
        localQp_.qpn = qp_->qp_num;
        localQp_.psn = static_cast<uint32_t>(lrand48() & 0xFFFFFF);
        localQp_.lid = pa.lid;
        ibv_gid gid{};
        ibv_query_gid(ctx_, ibPort_, gidIdx_, &gid);
        std::memcpy(localQp_.gid, gid.raw, 16);
        return true;
    }

    void destroyQp()
    {
        if(qp_)
        {
            ibv_destroy_qp(qp_);
            qp_ = nullptr;
        }
    }

    QpInfo exchangeQpInfo()
    {
        QpInfo remote{};
        MPI_Sendrecv(&localQp_, sizeof(QpInfo), MPI_BYTE, peerRank_, kTagQpInfo, &remote,
                     sizeof(QpInfo), MPI_BYTE, peerRank_, kTagQpInfo, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        return remote;
    }

    // Transition an RC QP INIT->RTR->RTS given the remote QpInfo.
    bool rcToRtrRts(const QpInfo& remote)
    {
        ibv_qp_attr qa{};
        qa.qp_state           = IBV_QPS_RTR;
        qa.path_mtu           = IBV_MTU_1024;
        qa.dest_qp_num        = remote.qpn;
        qa.rq_psn             = remote.psn;
        qa.max_dest_rd_atomic = 1;
        qa.min_rnr_timer      = 12;
        qa.ah_attr.is_global  = 1;
        qa.ah_attr.dlid       = remote.lid;
        qa.ah_attr.port_num   = static_cast<uint8_t>(ibPort_);
        std::memcpy(qa.ah_attr.grh.dgid.raw, remote.gid, 16);
        qa.ah_attr.grh.sgid_index = static_cast<uint8_t>(gidIdx_);
        qa.ah_attr.grh.hop_limit  = 64;
        if(ibv_modify_qp(qp_, &qa,
                         IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN
                             | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)
           != 0)
            return false;

        std::memset(&qa, 0, sizeof(qa));
        qa.qp_state      = IBV_QPS_RTS;
        qa.sq_psn        = localQp_.psn;
        qa.timeout       = 14;
        qa.retry_cnt     = 7;
        qa.rnr_retry     = 7;
        qa.max_rd_atomic = 1;
        if(ibv_modify_qp(qp_, &qa,
                         IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT
                             | IBV_QP_RNR_RETRY | IBV_QP_MAX_QP_RD_ATOMIC)
           != 0)
            return false;
        return true;
    }

    // Bring a UD QP to RTS (no remote address needed for state machine).
    bool udToRts()
    {
        ibv_qp_attr qa{};
        qa.qp_state = IBV_QPS_RTR;
        if(ibv_modify_qp(qp_, &qa, IBV_QP_STATE) != 0)
            return false;
        std::memset(&qa, 0, sizeof(qa));
        qa.qp_state = IBV_QPS_RTS;
        qa.sq_psn   = localQp_.psn;
        if(ibv_modify_qp(qp_, &qa, IBV_QP_STATE | IBV_QP_SQ_PSN) != 0)
            return false;
        return true;
    }

    // Allocate a buffer using the given HIP allocation mode.
    void* allocBuf(AllocMode mode, size_t size)
    {
        void*      ptr = nullptr;
        hipError_t e   = hipSuccess;
        switch(mode)
        {
            case ALLOC_DEVICE: e = hipMalloc(&ptr, size); break;
            case ALLOC_HOST_DEFAULT: e = hipHostMalloc(&ptr, size, hipHostMallocDefault); break;
            case ALLOC_HOST_WC: e = hipHostMalloc(&ptr, size, hipHostMallocWriteCombined); break;
            case ALLOC_HOST_COHERENT: e = hipHostMalloc(&ptr, size, hipHostMallocCoherent); break;
            case ALLOC_HOST_NONCOHERENT:
                e = hipHostMalloc(&ptr, size, hipHostMallocNonCoherent);
                break;
            case ALLOC_FINEGRAIN:
                e = hipExtMallocWithFlags(&ptr, size, hipDeviceMallocFinegrained);
                break;
            case ALLOC_MANAGED: e = hipMallocManaged(&ptr, size, hipMemAttachGlobal); break;
            default: return nullptr;
        }
        return (e == hipSuccess) ? ptr : nullptr;
    }

    void freeBuf(AllocMode mode, void* ptr)
    {
        if(!ptr)
            return;
        switch(mode)
        {
            case ALLOC_DEVICE:
            case ALLOC_FINEGRAIN:
            case ALLOC_MANAGED: (void)hipFree(ptr); break;
            default: (void)hipHostFree(ptr); break;
        }
    }

    RegResult regBuf(void* ptr, size_t size, RegMode mode, int access)
    {
        RegResult r;
        if(mode == REG_STD)
        {
            r.mr = ibv_reg_mr(pd_, ptr, size, access);
            return r;
        }
        if(!g_reg_dmabuf)
            return r;
        uint64_t     offset = 0;
        hsa_status_t hs;
        if(mode == REG_DMABUF_V1)
        {
            if(!g_dmabuf_v1)
                return r;
            hs = g_dmabuf_v1(ptr, size, &r.dma_fd, &offset);
        }
        else
        {
            if(!g_dmabuf_v2)
                return r;
            hs = g_dmabuf_v2(ptr, size, &r.dma_fd, &offset, HSA_AMD_DMABUF_MAPPING_TYPE_PCIE);
        }
        if(hs != HSA_STATUS_SUCCESS)
        {
            r.dma_fd = -1;
            return r;
        }
        r.mr = g_reg_dmabuf(pd_, offset, size, reinterpret_cast<uint64_t>(ptr), r.dma_fd, access);
        if(!r.mr)
        {
            close(r.dma_fd);
            r.dma_fd = -1;
        }
        return r;
    }

    void deregBuf(RegResult& r)
    {
        if(r.mr)
            ibv_dereg_mr(r.mr);
        if(r.dma_fd >= 0)
            close(r.dma_fd);
        r = {};
    }

    // Poll the CQ until a completion with wr_id arrives, or timeout. On a
    // non-success completion, fills wc and returns false.
    bool pollCq(uint64_t wrId, ibv_wc& wc, int timeoutMs)
    {
        for(int elapsed = 0; elapsed < timeoutMs; ++elapsed)
        {
            int n = ibv_poll_cq(cq_, 1, &wc);
            if(n < 0)
                return false;
            if(n > 0)
            {
                if(wc.status != IBV_WC_SUCCESS)
                    return false;
                if(wc.wr_id == wrId)
                    return true;
            }
            usleep(1000);
        }
        return false;
    }

    void fillPatternGpu(void* ptr, size_t size, uint8_t seed)
    {
        std::vector<uint8_t> tmp(size);
        for(size_t i = 0; i < size; ++i)
            tmp[i] = static_cast<uint8_t>((seed + i) % 256);
        (void)hipMemcpy(ptr, tmp.data(), size, hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
    }

    bool verifyPatternGpu(const void* ptr, size_t size, uint8_t seed)
    {
        std::vector<uint8_t> tmp(size);
        (void)hipMemcpy(tmp.data(), ptr, size, hipMemcpyDeviceToHost);
        (void)hipDeviceSynchronize();
        for(size_t i = 0; i < size; ++i)
            if(tmp[i] != static_cast<uint8_t>((seed + i) % 256))
                return false;
        return true;
    }

    // Post a single signaled RDMA Write from local buf to a remote address and
    // wait for local completion.
    bool doRdmaWrite(void*          localBuf,
                     struct ibv_mr* localMr,
                     uint64_t       remoteAddr,
                     uint32_t       remoteRkey,
                     size_t         len,
                     uint64_t       wrId)
    {
        ibv_sge sge{};
        sge.addr   = reinterpret_cast<uint64_t>(localBuf);
        sge.length = static_cast<uint32_t>(len);
        sge.lkey   = localMr->lkey;
        ibv_send_wr wr{};
        wr.wr_id               = wrId;
        wr.opcode              = IBV_WR_RDMA_WRITE;
        wr.send_flags          = IBV_SEND_SIGNALED;
        wr.sg_list             = &sge;
        wr.num_sge             = 1;
        wr.wr.rdma.remote_addr = remoteAddr;
        wr.wr.rdma.rkey        = remoteRkey;
        ibv_send_wr* bad       = nullptr;
        if(ibv_post_send(qp_, &wr, &bad) != 0)
            return false;
        ibv_wc wc{};
        return pollCq(wrId, wc, kPollTimeoutMs);
    }
};

// ===========================================================================
// Test 1: UD_RC_Support -- baseline. Can the NIC create both RC and UD QPs and
// bring each to RTS?
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, UD_RC_Support)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";

    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";

    // RC -- reduce the create result before any OOB exchange so paired ranks
    // never block in MPI_Sendrecv when one side fails to create a QP.
    bool rc = createQp(IBV_QPT_RC);
    if(!allRanksAgree(rc))
    {
        destroyQp();
        GTEST_SKIP() << "RC QP creation not supported on all ranks";
    }
    QpInfo remote = exchangeQpInfo();
    rc            = rcToRtrRts(remote);
    destroyQp();
    ASSERT_MPI_TRUE(rc);

    // UD
    bool ud = createQp(IBV_QPT_UD);
    if(!allRanksAgree(ud))
    {
        destroyQp();
        GTEST_SKIP() << "UD QP not supported on all ranks";
    }
    (void)exchangeQpInfo(); // keep OOB symmetric across ranks
    ud = udToRts();
    destroyQp();
    ASSERT_MPI_TRUE(ud);
}

// ===========================================================================
// Test 2: RdmaWrite_DMABUF -- RC write into peer GPU memory registered via
// DMABUF. Runs once per available dmabuf version. Lower half = target.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, RdmaWrite_DMABUF)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(g_reg_dmabuf != nullptr && (g_dmabuf_v1 || g_dmabuf_v2)))
        GTEST_SKIP() << "DMABUF registration not available";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    bool      ranAny = false;

    for(RegMode reg : {REG_DMABUF_V1, REG_DMABUF_V2_PCIE})
    {
        bool haveSym = (reg == REG_DMABUF_V1) ? (g_dmabuf_v1 != nullptr) : (g_dmabuf_v2 != nullptr);
        if(!allRanksAgree(haveSym))
            continue;

        ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
        QpInfo remote = exchangeQpInfo();
        ASSERT_MPI_TRUE(rcToRtrRts(remote));

        void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
        ASSERT_MPI_TRUE(buf != nullptr);
        SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
        RegResult rr = regBuf(buf, kMsgSize, reg, access);
        if(!allRanksAgree(rr.mr != nullptr))
        {
            deregBuf(rr);
            destroyQp();
            continue;
        }

        const uint8_t seed = static_cast<uint8_t>(0xA0 + static_cast<int>(reg));
        bool          pass = false;
        if(isLowerHalf_)
        {
            // Target: clear buffer, advertise it, wait for the data to land.
            (void)hipMemset(buf, 0, kMsgSize);
            (void)hipDeviceSynchronize();
            RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
            MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
            uint8_t go = 0;
            MPI_Recv(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass        = verifyPatternGpu(buf, kMsgSize, seed);
            uint8_t res = pass ? 1 : 0;
            MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
        }
        else
        {
            RemoteBuf rb{};
            MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            fillPatternGpu(buf, kMsgSize, seed);
            bool    wrote = doRdmaWrite(buf, rr.mr, rb.addr, rb.rkey, kMsgSize, 1);
            uint8_t go    = 1;
            MPI_Send(&go, 1, MPI_BYTE, peerRank_, kTagFlag, MPI_COMM_WORLD);
            uint8_t res = 0;
            MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass = wrote && (res == 1);
        }
        deregBuf(rr);
        destroyQp();
        ranAny = true;
        ASSERT_MPI_TRUE(pass);
    }

    if(!allRanksAgree(ranAny))
        GTEST_SKIP() << "No DMABUF version exercised";
}

// ===========================================================================
// Test 3: RdmaRead_DMABUF -- RC read from peer GPU memory via DMABUF. A
// separate code path from Write. Runs once per available dmabuf version.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, RdmaRead_DMABUF)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";
    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";
    if(!allRanksAgree(g_reg_dmabuf != nullptr && (g_dmabuf_v1 || g_dmabuf_v2)))
        GTEST_SKIP() << "DMABUF registration not available";

    const int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
    bool      ranAny = false;

    for(RegMode reg : {REG_DMABUF_V1, REG_DMABUF_V2_PCIE})
    {
        bool haveSym = (reg == REG_DMABUF_V1) ? (g_dmabuf_v1 != nullptr) : (g_dmabuf_v2 != nullptr);
        if(!allRanksAgree(haveSym))
            continue;

        ASSERT_MPI_TRUE(createQp(IBV_QPT_RC));
        QpInfo remote = exchangeQpInfo();
        ASSERT_MPI_TRUE(rcToRtrRts(remote));

        void* buf = allocBuf(ALLOC_DEVICE, kMsgSize);
        ASSERT_MPI_TRUE(buf != nullptr);
        SCOPE_EXIT(freeBuf(ALLOC_DEVICE, buf));
        RegResult rr = regBuf(buf, kMsgSize, reg, access);
        if(!allRanksAgree(rr.mr != nullptr))
        {
            deregBuf(rr);
            destroyQp();
            continue;
        }

        const uint8_t seed = static_cast<uint8_t>(0xB0 + static_cast<int>(reg));
        bool          pass = false;
        if(isLowerHalf_)
        {
            // Target holds the source data; initiator reads it.
            fillPatternGpu(buf, kMsgSize, seed);
            RemoteBuf rb{reinterpret_cast<uint64_t>(buf), rr.mr->rkey, 0};
            MPI_Send(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD);
            uint8_t res = 0;
            MPI_Recv(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            pass = (res == 1);
        }
        else
        {
            RemoteBuf rb{};
            MPI_Recv(&rb, sizeof(rb), MPI_BYTE, peerRank_, kTagRemoteBuf, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            (void)hipMemset(buf, 0, kMsgSize);
            (void)hipDeviceSynchronize();
            ibv_sge sge{};
            sge.addr   = reinterpret_cast<uint64_t>(buf);
            sge.length = static_cast<uint32_t>(kMsgSize);
            sge.lkey   = rr.mr->lkey;
            ibv_send_wr wr{};
            wr.wr_id               = 2;
            wr.opcode              = IBV_WR_RDMA_READ;
            wr.send_flags          = IBV_SEND_SIGNALED;
            wr.sg_list             = &sge;
            wr.num_sge             = 1;
            wr.wr.rdma.remote_addr = rb.addr;
            wr.wr.rdma.rkey        = rb.rkey;
            ibv_send_wr* bad       = nullptr;
            bool         ok        = (ibv_post_send(qp_, &wr, &bad) == 0);
            if(ok)
            {
                ibv_wc wc{};
                ok = pollCq(2, wc, kPollTimeoutMs);
            }
            pass        = ok && verifyPatternGpu(buf, kMsgSize, seed);
            uint8_t res = pass ? 1 : 0;
            MPI_Send(&res, 1, MPI_BYTE, peerRank_, kTagResult, MPI_COMM_WORLD);
        }
        deregBuf(rr);
        destroyQp();
        ranAny = true;
        ASSERT_MPI_TRUE(pass);
    }

    if(!allRanksAgree(ranAny))
        GTEST_SKIP() << "No DMABUF version exercised";
}

#endif // MPI_TESTS_ENABLED
