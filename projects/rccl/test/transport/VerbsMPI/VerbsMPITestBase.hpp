/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file VerbsMPITestBase.hpp
 * @brief Shared fixture and helpers for the Verbs acceptance test suite.
 *
 * Hardware-agnostic RDMA/verbs acceptance (regression) tests. Each test targets
 * a specific low-level NIC/driver code path that RCCL's network transport
 * depends on; a PASS->FAIL flip after a firmware or driver update is an
 * unambiguous regression signal.
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
 * Test files in this folder are organized by area:
 *   - ConnectionTests.cpp    : QP creation and connection establishment
 *   - RdmaTransferTests.cpp   : RDMA data path (Write/Read over DMABUF)
 *
 * This header carries only what those files use; later areas (atomics,
 * write-with-immediate, etc.) extend it as they are added.
 */

#ifndef RCCL_TEST_VERBS_MPI_TEST_BASE_HPP_
#define RCCL_TEST_VERBS_MPI_TEST_BASE_HPP_

#include "MPIHelpers.hpp"
#include "MPITestBase.hpp"
#include "ResourceGuards.hpp"
#include "TestChecks.hpp"

#ifdef MPI_TESTS_ENABLED

#include <infiniband/verbs.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <vector>

using namespace MPITestConstants;
using namespace RCCLTestGuards;

namespace VerbsAcceptance
{

// ---- Test parameters -------------------------------------------------------
inline constexpr size_t kMsgSize       = 4096;
inline constexpr int    kCqDepth       = 32;
inline constexpr int    kQpDepth       = 16;
inline constexpr int    kPollTimeoutMs = 5000;
inline constexpr int    kMinRanks      = 2;

// OOB tags for MPI exchanges.
inline constexpr int kTagQpInfo    = 7001;
inline constexpr int kTagRemoteBuf = 7002;
inline constexpr int kTagResult    = 7003;
inline constexpr int kTagFlag      = 7004;

// ---- IB registration modes -------------------------------------------------
enum RegMode
{
    REG_STD = 0,     // plain ibv_reg_mr
    REG_DMABUF_V1,   // hsa_amd_portable_export_dmabuf      + ibv_reg_dmabuf_mr
    REG_DMABUF_V2_PCIE // hsa_amd_portable_export_dmabuf_v2 + ibv_reg_dmabuf_mr
};

// ---- Optional symbols resolved at runtime (graceful skip if absent) --------
typedef hsa_status_t (*fn_export_dmabuf_v1_t)(const void*, size_t, int*, uint64_t*);
typedef hsa_status_t (*fn_export_dmabuf_v2_t)(const void*, size_t, int*, uint64_t*, uint64_t);
typedef struct ibv_mr* (*fn_reg_dmabuf_mr_t)(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int);

inline fn_export_dmabuf_v1_t g_dmabuf_v1        = nullptr;
inline fn_export_dmabuf_v2_t g_dmabuf_v2        = nullptr;
inline fn_reg_dmabuf_mr_t    g_reg_dmabuf       = nullptr;
inline bool                  g_symbols_resolved = false;

inline void resolveOptionalSymbols()
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
inline bool allRanksAgree(bool localOk)
{
    int local  = localOk ? 1 : 0;
    int global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return global != 0;
}

} // namespace VerbsAcceptance

// ===========================================================================
// Fixture shared by all Verbs acceptance test files.
// ===========================================================================
class VerbsAcceptanceMPITest : public MPITestBase
{
protected:
    using RegMode   = VerbsAcceptance::RegMode;
    using QpInfo    = VerbsAcceptance::QpInfo;
    using RemoteBuf = VerbsAcceptance::RemoteBuf;
    using RegResult = VerbsAcceptance::RegResult;

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
        if(!validateTestPrerequisites(VerbsAcceptance::kMinRanks))
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

        pd_ = ibv_alloc_pd(ctx_);
        if(!pd_)
            return false;
        cq_ = ibv_create_cq(ctx_, VerbsAcceptance::kCqDepth, nullptr, nullptr, 0);
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
        qia.cap.max_send_wr  = VerbsAcceptance::kQpDepth;
        qia.cap.max_recv_wr  = VerbsAcceptance::kQpDepth;
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
                                 | IBV_ACCESS_REMOTE_READ;
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
        MPI_Sendrecv(&localQp_, sizeof(QpInfo), MPI_BYTE, peerRank_, VerbsAcceptance::kTagQpInfo,
                     &remote, sizeof(QpInfo), MPI_BYTE, peerRank_, VerbsAcceptance::kTagQpInfo,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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

    // Bring a UD QP to RTS (no remote address needed for the state machine).
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

    void* allocDeviceBuf(size_t size)
    {
        void* ptr = nullptr;
        return (hipMalloc(&ptr, size) == hipSuccess) ? ptr : nullptr;
    }

    void freeDeviceBuf(void* ptr)
    {
        if(ptr)
            (void)hipFree(ptr);
    }

    // Register a buffer with the NIC, either via plain ibv_reg_mr (REG_STD) or
    // by exporting it as a DMABUF and registering that (REG_DMABUF_*).
    RegResult regBuf(void* ptr, size_t size, RegMode mode, int access)
    {
        RegResult r;
        if(mode == VerbsAcceptance::REG_STD)
        {
            r.mr = ibv_reg_mr(pd_, ptr, size, access);
            return r;
        }
        if(!VerbsAcceptance::g_reg_dmabuf)
            return r;
        uint64_t     offset = 0;
        hsa_status_t hs;
        if(mode == VerbsAcceptance::REG_DMABUF_V1)
        {
            if(!VerbsAcceptance::g_dmabuf_v1)
                return r;
            hs = VerbsAcceptance::g_dmabuf_v1(ptr, size, &r.dma_fd, &offset);
        }
        else
        {
            if(!VerbsAcceptance::g_dmabuf_v2)
                return r;
            hs = VerbsAcceptance::g_dmabuf_v2(ptr, size, &r.dma_fd, &offset,
                                              HSA_AMD_DMABUF_MAPPING_TYPE_PCIE);
        }
        if(hs != HSA_STATUS_SUCCESS)
        {
            r.dma_fd = -1;
            return r;
        }
        r.mr = VerbsAcceptance::g_reg_dmabuf(pd_, offset, size, reinterpret_cast<uint64_t>(ptr),
                                             r.dma_fd, access);
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

    // Post a single signaled one-sided RDMA op (Write or Read) against the
    // remote buffer and wait for the local completion.
    bool postRdmaAndWait(void*          localBuf,
                         struct ibv_mr* localMr,
                         uint64_t       remoteAddr,
                         uint32_t       remoteRkey,
                         size_t         len,
                         ibv_wr_opcode  opcode,
                         uint64_t       wrId)
    {
        ibv_sge sge{};
        sge.addr   = reinterpret_cast<uint64_t>(localBuf);
        sge.length = static_cast<uint32_t>(len);
        sge.lkey   = localMr->lkey;
        ibv_send_wr wr{};
        wr.wr_id               = wrId;
        wr.opcode              = opcode;
        wr.send_flags          = IBV_SEND_SIGNALED;
        wr.sg_list             = &sge;
        wr.num_sge             = 1;
        wr.wr.rdma.remote_addr = remoteAddr;
        wr.wr.rdma.rkey        = remoteRkey;
        ibv_send_wr* bad       = nullptr;
        if(ibv_post_send(qp_, &wr, &bad) != 0)
            return false;
        ibv_wc wc{};
        return pollCq(wrId, wc, VerbsAcceptance::kPollTimeoutMs);
    }
};

#endif // MPI_TESTS_ENABLED

#endif // RCCL_TEST_VERBS_MPI_TEST_BASE_HPP_
