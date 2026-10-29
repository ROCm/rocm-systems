/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collectives.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace RcclUnitTesting
{

namespace
{

// Host model of one-rank-per-node PAT: each rank's step stream applied as patReduce/patCopy apply it.

using Elem = uint64_t;

// One direction of a connection: the sender writes, the receiver reads.
struct PatConn
{
    std::vector<Elem> fifo;
    uint64_t          tail = 0;  // steps the sender posted
    uint64_t          head = 0;  // steps the receiver released
};

struct PatModelConfig
{
    int    nranks;
    size_t count;       // elements per rank block
    int    chunkCount;  // also the connection step size in elements
    int    nChannels;
    int    maxParallelFactor;
};

// Rank q owns hex digit q (1..15, varying with j), so a lost, doubled or moved block changes the value.
Elem patInput(int q, size_t j, int /*nranks*/)
{
    return static_cast<Elem>(1 + (j + 7 * q) % 15) << (4 * q);
}

class PatModel
{
public:
    PatModel(const PatModelConfig& cfg, bool reduceScatter, bool shared)
        : cfg_(cfg), rs_(reduceScatter), shared_(shared)
    {
        while((1 << nDims_) < cfg.nranks)
        {
            nDims_++;
        }
        const size_t inLen  = rs_ ? cfg.nranks * cfg.count : cfg.count;
        const size_t outLen = rs_ ? cfg.count : cfg.nranks * cfg.count;
        in_.assign(cfg.nranks, std::vector<Elem>(inLen));
        out_.assign(cfg.nranks, std::vector<Elem>(outLen, ~Elem{0}));
        for(int q = 0; q < cfg.nranks; ++q)
        {
            for(size_t j = 0; j < inLen; ++j)
            {
                in_[q][j] = patInput(q, j, cfg.nranks);
            }
        }
    }

    // Empty when every channel ran to completion without a model violation.
    std::string run()
    {
        // Even split; the device aligns channel bounds to chunks, which moves starts, not the schedule.
        const size_t perCh = (cfg_.count + cfg_.nChannels - 1) / cfg_.nChannels;
        for(int c = 0; c < cfg_.nChannels; ++c)
        {
            size_t begin = std::min(cfg_.count, c * perCh);
            size_t end   = std::min(cfg_.count, begin + perCh);
            if(begin == end)
            {
                continue;
            }
            runChannel(begin, end);
            if(!error_.empty())
            {
                return "channel " + std::to_string(c) + ": " + error_;
            }
        }
        return {};
    }

    const std::vector<Elem>& output(int rank) const
    {
        return out_[rank];
    }

private:
    struct RankState
    {
        std::vector<ncclPatStep> steps;
        size_t                   next    = 0;  // first step of the next batch
        int                      nGroups = 1;
        std::vector<bool>        groupDone;
        std::vector<uint64_t>    sendStep, recvStep;
        std::vector<long long>   sendAcc, recvAcc;
        long long                localAcc = 0;
    };

    // One step resolved against the state at the start of its batch.
    struct Op
    {
        std::vector<Elem>  vals;
        std::vector<Elem*> dsts;
    };

    int recvPeer(int r, int d) const
    {
        return patRecvPeer(r, cfg_.nranks, d, /*mirrored=*/!rs_ && !shared_);
    }

    void fail(const std::string& why)
    {
        if(error_.empty())
        {
            error_ = why;
        }
    }

    // Precompute each rank's step stream: getNextOp does not depend on timing.
    void schedule(RankState& st, int r, size_t begin, size_t end)
    {
        const int stepSize = cfg_.chunkCount * sizeof(Elem);
        int       pf       = 0;
        auto      push     = [&](auto& algo)
        {
            pf             = algo.getParallelFactor();
            ncclPatStep ps = {};
            do
            {
                algo.getNextOp(&ps);
                st.steps.push_back(ps);
            } while(ps.last != 2 && st.steps.size() < kMaxSteps);
        };
        if(rs_)
        {
            PatRSAlgorithm<Elem> algo(
                stepSize, NCCL_STEPS, cfg_.maxParallelFactor, begin, end,
                cfg_.count, cfg_.chunkCount, r, cfg_.nranks);
            push(algo);
        }
        else
        {
            PatAGAlgorithm<Elem> algo(
                stepSize, NCCL_STEPS, cfg_.maxParallelFactor, begin, end,
                cfg_.count, cfg_.chunkCount, r, cfg_.nranks, shared_);
            push(algo);
        }
        if(st.steps.back().last != 2)
        {
            fail("getNextOp did not terminate");
        }
        // As in reduce_scatter.h: group g runs steps g, g+nGroups, ... in lockstep via patBarrier.
        const int nworkers  = NCCL_PAT_NWORKERS;
        const int warp      = nworkers / cfg_.maxParallelFactor;
        const int groupSize = patWorkerGroupSize(nworkers, warp, pf);
        st.nGroups          = nworkers / groupSize;
        st.groupDone.assign(st.nGroups, false);
        st.sendStep.assign(nDims_, 0);
        st.recvStep.assign(nDims_, 0);
        st.sendAcc.assign(nDims_, 0);
        st.recvAcc.assign(nDims_, 0);
    }

    // patReduce writes at step + stepOffset; patCopy reads there.
    uint64_t sendSlot(uint64_t step, const ncclPatStep& ps) const
    {
        return rs_ ? step + ps.stepOffset : step;
    }

    uint64_t recvSlot(uint64_t step, const ncclPatStep& ps) const
    {
        return rs_ ? step : step + ps.stepOffset;
    }

    // Connection accSize once `nelem` elements at `offset` in `slot` land.
    long long accEnd(int offset, int nelem, uint64_t slot) const
    {
        return static_cast<long long>(offset) + nelem +
               static_cast<long long>(slot) * cfg_.chunkCount;
    }

    bool inSlot(int offset, int nelem) const
    {
        return offset >= 0 && offset + nelem <= cfg_.chunkCount;
    }

    // Resolves one step as patReduce/patCopy do before their first barrier; false if a wait is unmet.
    bool plan(int r, const RankState& st, const ncclPatStep& ps, Op* op)
    {
        const int  nelem = ps.nelem < 0 ? 0 : ps.nelem;
        const int  cs    = cfg_.chunkCount;
        const bool recv  = ps.recvDim >= 0;
        const bool send  = ps.sendDim >= 0;
        if((recv && ps.recvDim >= nDims_) || (send && ps.sendDim >= nDims_))
        {
            fail("step uses a dimension beyond log2(nranks)");
            return true;
        }
        PatConn* rc =
            recv ? &conns_[recvPeer(r, ps.recvDim)][ps.recvDim] : nullptr;
        PatConn*       sc    = send ? &conns_[r][ps.sendDim] : nullptr;
        const uint64_t rStep = recv ? st.recvStep[ps.recvDim] : 0;
        const uint64_t sStep = send ? st.sendStep[ps.sendDim] : 0;
        const uint64_t rSlot = recvSlot(rStep, ps);
        const uint64_t sSlot = sendSlot(sStep, ps);
        if((recv && rc->tail < rSlot + 1) ||
           (send && sc->head + NCCL_STEPS < sSlot + 1))
        {
            return false;
        }
        if((recv && !inSlot(ps.recvOffset, nelem)) ||
           (send && !inSlot(ps.sendOffset, nelem)))
        {
            fail("FIFO offset outside its slot");
            return true;
        }
        Elem* rBuf =
            recv ? rc->fifo.data() + (rSlot % NCCL_STEPS) * cs + ps.recvOffset
                 : nullptr;
        Elem* sBuf =
            send ? sc->fifo.data() + (sSlot % NCCL_STEPS) * cs + ps.sendOffset
                 : nullptr;
        std::vector<Elem>& in  = in_[r];
        std::vector<Elem>& out = out_[r];
        // fresh: the destination holds no partial result yet.
        const Elem* src   = nullptr;
        Elem*       dst   = nullptr;
        bool        fresh = false;
        if(rs_)
        {
            dst = sBuf;
            if(send)
            {
                fresh = st.sendAcc[ps.sendDim] <
                        accEnd(ps.sendOffset, nelem, sSlot);
            }
            else
            {
                dst   = out.data() + ps.outIx;
                fresh = st.localAcc < static_cast<long long>(ps.outIx + nelem);
            }
            src = fresh ? in.data() + ps.inpIx : dst;
            if((!send && ps.outIx + nelem > out.size()) ||
               (fresh && ps.inpIx + nelem > in.size()))
            {
                fail("inpIx or outIx outside its buffer");
                return true;
            }
            op->dsts.push_back(dst);
        }
        else
        {
            src = rBuf;
            if(recv)
            {
                fresh = st.recvAcc[ps.recvDim] <
                        accEnd(ps.recvOffset, nelem, rSlot);
            }
            else
            {
                src   = in.data() + ps.inpIx;
                fresh = st.localAcc < static_cast<long long>(ps.inpIx + nelem);
            }
            if((!recv && ps.inpIx + nelem > in.size()) ||
               (fresh && ps.outIx + nelem > out.size()))
            {
                fail("inpIx or outIx outside its buffer");
                return true;
            }
            if(send)
            {
                op->dsts.push_back(sBuf);
            }
            if(fresh)
            {
                op->dsts.push_back(out.data() + ps.outIx);
            }
        }
        op->vals.resize(nelem);
        for(int i = 0; i < nelem; ++i)
        {
            op->vals[i] = src[i] + ((rs_ && recv) ? rBuf[i] : 0);
        }
        return true;
    }

    // Runs rank r's next batch, one step per group, all from the batch-start state as at patBarrier.
    bool runBatch(int r, RankState& st)
    {
        std::vector<const ncclPatStep*> batch;
        for(int g = 0; g < st.nGroups; ++g)
        {
            size_t k = st.next + g;
            if(!st.groupDone[g] && k < st.steps.size())
            {
                batch.push_back(&st.steps[k]);
            }
        }
        if(batch.empty())
        {
            return false;
        }
        std::vector<Op> ops(batch.size());
        for(size_t i = 0; i < batch.size(); ++i)
        {
            bool skipped = batch[i]->flags & PatSkipped;
            if(!skipped && !plan(r, st, *batch[i], &ops[i]))
            {
                return false;
            }
        }
        RankState prev = st;
        for(size_t i = 0; i < batch.size(); ++i)
        {
            const ncclPatStep& ps = *batch[i];
            for(Elem* dst : ops[i].dsts)
            {
                std::copy(ops[i].vals.begin(), ops[i].vals.end(), dst);
            }
            if(ps.flags & PatSkipped)
            {
                continue;
            }
            const int  nelem = ps.nelem < 0 ? 0 : ps.nelem;
            const bool recv  = ps.recvDim >= 0;
            const bool send  = ps.sendDim >= 0;
            if(send)
            {
                uint64_t  s   = prev.sendStep[ps.sendDim];
                long long acc = accEnd(ps.sendOffset, nelem, sendSlot(s, ps));
                if(rs_)
                {
                    st.sendAcc[ps.sendDim] =
                        std::max(st.sendAcc[ps.sendDim], acc);
                }
                if(ps.postSend)
                {
                    st.sendStep[ps.sendDim] = s + 1;
                }
            }
            else if(rs_)
            {
                st.localAcc = std::max(
                    st.localAcc, static_cast<long long>(ps.outIx + nelem));
            }
            if(recv)
            {
                uint64_t  s   = prev.recvStep[ps.recvDim];
                long long acc = accEnd(ps.recvOffset, nelem, recvSlot(s, ps));
                if(!rs_)
                {
                    st.recvAcc[ps.recvDim] =
                        std::max(st.recvAcc[ps.recvDim], acc);
                }
                if(ps.postRecv)
                {
                    st.recvStep[ps.recvDim] = s + 1;
                }
            }
            else if(!rs_)
            {
                st.localAcc = std::max(
                    st.localAcc, static_cast<long long>(ps.inpIx + nelem));
            }
        }
        // Publish tail and head after the second barrier.
        for(int d = 0; d < nDims_; ++d)
        {
            conns_[r][d].tail              = st.sendStep[d];
            conns_[recvPeer(r, d)][d].head = st.recvStep[d];
        }
        int ended = 0;
        for(int g = 0; g < st.nGroups; ++g)
        {
            size_t k = st.next + g;
            if(k < st.steps.size() && st.steps[k].last != 0)
            {
                st.groupDone[g] = true;
                ++ended;
            }
        }
        // Each group leaves its loop on `last`, and patBarrier waits for every worker.
        if(ended != 0 && ended != st.nGroups)
        {
            fail("rank " + std::to_string(r) + " batch at step " +
                 std::to_string(st.next) + ": " + std::to_string(ended) +
                 " of " + std::to_string(st.nGroups) +
                 " worker groups end on different batches; patBarrier would hang");
            return false;
        }
        st.next += st.nGroups;
        return true;
    }

    void runChannel(size_t begin, size_t end)
    {
        const int n = cfg_.nranks;
        conns_.assign(n, std::vector<PatConn>(nDims_));
        for(auto& rank : conns_)
        {
            for(auto& c : rank)
            {
                c.fifo.assign(NCCL_STEPS * cfg_.chunkCount, 0);
            }
        }
        std::vector<RankState> ranks(n);
        for(int r = 0; r < n && error_.empty(); ++r)
        {
            schedule(ranks[r], r, begin, end);
        }
        bool progress = true;
        while(progress && error_.empty())
        {
            progress = false;
            for(int r = 0; r < n && error_.empty(); ++r)
            {
                while(error_.empty() && runBatch(r, ranks[r]))
                {
                    progress = true;
                }
            }
        }
        for(int r = 0; r < n && error_.empty(); ++r)
        {
            const RankState& st = ranks[r];
            for(int g = 0; g < st.nGroups; ++g)
            {
                if(!st.groupDone[g])
                {
                    fail("rank " + std::to_string(r) + " stalled at step " +
                         std::to_string(st.next + g));
                    break;
                }
            }
            // The device never runs steps past a group's last, so they must be skips.
            for(size_t k = st.next; k < st.steps.size(); ++k)
            {
                if(!(st.steps[k].flags & PatSkipped))
                {
                    fail("rank " + std::to_string(r) + " step " +
                         std::to_string(k) + " follows its group's last step");
                }
            }
        }
    }

    static constexpr size_t kMaxSteps = 1 << 20;

    PatModelConfig                    cfg_;
    bool                              rs_;
    bool                              shared_;
    int                               nDims_ = 0;
    std::vector<std::vector<Elem>>    in_;
    std::vector<std::vector<Elem>>    out_;
    std::vector<std::vector<PatConn>> conns_;
    std::string                       error_;
};

// Covers aggregation, several chunks per channel, 1 and 2 channels, and wave64 and wave32 factors.
std::vector<PatModelConfig> patModelConfigs()
{
    std::vector<PatModelConfig> cfgs;
    for(int nranks = 3; nranks <= 16; ++nranks)
    {
        for(size_t count : {size_t{1}, size_t{5}, size_t{64}, size_t{129}})
        {
            for(int chunk : {4, 1024})
            {
                for(int nChannels : {1, 2})
                {
                    for(int pf :
                        {NCCL_PAT_NWORKERS / 64, NCCL_PAT_NWORKERS / 32})
                    {
                        cfgs.push_back({nranks, count, chunk, nChannels, pf});
                    }
                }
            }
        }
    }
    return cfgs;
}

std::string describe(const PatModelConfig& c)
{
    std::ostringstream os;
    os << "nranks=" << c.nranks << " count=" << c.count
       << " chunk=" << c.chunkCount << " channels=" << c.nChannels
       << " maxParallelFactor=" << c.maxParallelFactor;
    return os.str();
}

}  // namespace

// Every rank's PatRSAlgorithm stream leaves each rank holding the sum of its block.
TEST(PatRSAlgorithmTests, StepStreamsReduceScatterEveryBlock)
{
    for(const PatModelConfig& cfg : patModelConfigs())
    {
        PatModel    model(cfg, /*reduceScatter=*/true, /*shared=*/true);
        std::string err = model.run();
        ASSERT_TRUE(err.empty()) << describe(cfg) << ": " << err;
        for(int r = 0; r < cfg.nranks; ++r)
        {
            const std::vector<Elem>& out = model.output(r);
            for(size_t i = 0; i < cfg.count; ++i)
            {
                Elem want = 0;
                for(int q = 0; q < cfg.nranks; ++q)
                {
                    want += patInput(q, r * cfg.count + i, cfg.nranks);
                }
                ASSERT_EQ(want, out[i])
                    << describe(cfg) << " rank=" << r << " i=" << i;
            }
        }
    }
}

// Every rank's PatAGAlgorithm stream leaves every rank holding every block.
TEST(PatAGAlgorithmTests, StepStreamsAllGatherEveryBlock)
{
    for(int shared = 0; shared <= 1; ++shared)
    {
        for(const PatModelConfig& cfg : patModelConfigs())
        {
            PatModel    model(cfg, /*reduceScatter=*/false, shared != 0);
            std::string err = model.run();
            ASSERT_TRUE(err.empty())
                << describe(cfg) << " shared=" << shared << ": " << err;
            for(int r = 0; r < cfg.nranks; ++r)
            {
                const std::vector<Elem>& out = model.output(r);
                for(size_t j = 0; j < out.size(); ++j)
                {
                    Elem want =
                        patInput(j / cfg.count, j % cfg.count, cfg.nranks);
                    ASSERT_EQ(want, out[j])
                        << describe(cfg) << " shared=" << shared
                        << " rank=" << r << " j=" << j;
                }
            }
        }
    }
}

// PatAGAlgorithm::getNextOp is __host__ as well as __device__, so the shared-conn
// data-block index can be checked here without MPI or a GPU. The skip=1 case
// (s >= nranks) is what used to feed C++ a negative dividend.
TEST(PatAGAlgorithmTests, SharedDataRankStaysInRangeOnSkippedSteps)
{
    constexpr size_t kCount = 1024;
    constexpr int    kChunk = 1024;
    const int        ranks[] = {3, 4, 5, 6, 7, 8, 9, 15, 16, 17};

    for(int nranks : ranks)
    {
        for(int rank = 0; rank < nranks; ++rank)
        {
            for(int shared = 0; shared <= 1; ++shared)
            {
                PatAGAlgorithm<char> algo(/*stepSize=*/4096,
                                          NCCL_STEPS,
                                          /*maxParallelFactor=*/16,
                                          /*offset=*/0,
                                          /*end=*/kCount,
                                          kCount,
                                          kChunk,
                                          rank,
                                          nranks,
                                          shared);
                struct ncclPatStep ps    = {};
                int                steps = 0;
                do
                {
                    algo.getNextOp(&ps);
                    // outIx is size_t: a negative recvDataRank wraps to a huge offset.
                    EXPECT_LE(ps.outIx, static_cast<size_t>(nranks - 1) * kCount)
                        << "nranks=" << nranks << " rank=" << rank << " shared=" << shared
                        << " step=" << steps;
                    ASSERT_LT(++steps, 100000) << "getNextOp did not terminate";
                } while(ps.last != 2);
            }
        }
    }
}

}  // namespace RcclUnitTesting
