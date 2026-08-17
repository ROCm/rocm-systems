/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_QP_SHARE_REF_MODEL_HPP_
#define RCCL_TEST_QP_SHARE_REF_MODEL_HPP_

#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>

#ifdef MPI_TESTS_ENABLED

// ─────────────────────────────────────────────────────────────────────────────
// Environment readers
//
// RCCL_IB_COMM_NGROUPS and RCCL_IB_QP_DEPTH_MULTIPLIER are plain RCCL_PARAMs
// (qp_sharing.cc) -- only the RCCL_ prefix is honoured, there is no NCCL_ alias
// for either despite what the feature commit message says. RCCL_PARAM caches
// both for the process lifetime, so a test may read them once and assume they
// hold for its whole run. That also means a sweep over either value needs one
// process (one mpirun) per point; see tools/scripts/qpshare/qpshare_validate.sh.
// ─────────────────────────────────────────────────────────────────────────────

static inline int QpShareEnvInt(const char* name, int fallback) {
    const char* v = getenv(name);
    if (!v || v[0] == '\0') return fallback;
    return static_cast<int>(std::atoll(v));
}

static inline int QpShareEnvNGroups() {
    return QpShareEnvInt("RCCL_IB_COMM_NGROUPS", 0);
}

// connect.cc clamps the multiplier to >= 1 before scaling CQ/WR depth, so 0 or
// a negative value behaves exactly like the default of 1.
static inline int QpShareEnvDepthMultiplier() {
    return std::max(1, QpShareEnvInt("RCCL_IB_QP_DEPTH_MULTIPLIER", 1));
}

// ─────────────────────────────────────────────────────────────────────────────
// QpShareRefModel -- expected group layout, derived from RCCL_IB_COMM_NGROUPS
//
// Encodes the assignment rule the transport implements today (connect.cc):
//
//     groupIdx = IbCastCountPeerTotalRefcount(...) % ngroups
//
// i.e. occupancy-modulo -- the next comm joins the group selected by the total
// number of *live* refcounts to that peer, not by a monotonic counter and not
// by picking the least-loaded group. It does not balance under churn, and that
// is deliberately reproduced here rather than corrected: this model exists to
// lock current behaviour so a net_ib_cast change cannot move it silently. A
// divergence between model and implementation is a regression signal, not a
// verdict on whether the rule is the right one. Changing the rule means
// changing this header in the same commit, on purpose.
//
// This is not a tautology despite mirroring the rule. The model tracks group
// loads from the create/destroy sequence the test issued; the implementation
// rederives the index by scanning the live shared-QP pool. A refcount that
// survives teardown, or a freed pool slot that is still counted, makes the two
// disagree -- which is precisely the defect class QP sharing has today.
// ─────────────────────────────────────────────────────────────────────────────
class QpShareRefModel {
public:
    // Where one comm landed. Both fields are fixed at connect()/accept() time,
    // exactly like ncclIbQpSharingState::sharedGroupIdx / isSharedQpPrimary:
    // releasing the PRIMARY of a group does not promote a SECONDARY.
    struct Placement {
        int  group   = -1;
        bool primary = false;
    };

    explicit QpShareRefModel(int ngroups)
        : ngroups_(ngroups > 0 ? ngroups : 1), loads_(ngroups_, 0) {}

    // Model the next successful connect(). Call this immediately before the
    // real connection is established, in the same order the test issues them.
    Placement AssignOnCreate() {
        Placement p;
        p.group   = TotalRefs() % ngroups_;
        p.primary = (loads_[p.group] == 0);
        loads_[p.group]++;
        peakLoad_ = std::max(peakLoad_, loads_[p.group]);
        return p;
    }

    // Model a comm teardown. Pass the Placement returned by AssignOnCreate().
    void Release(const Placement& p) {
        if (p.group < 0 || p.group >= ngroups_) return;
        if (loads_[p.group] > 0) loads_[p.group]--;
    }

    int NGroups() const { return ngroups_; }

    // Live comms in `group` -- the expected ncclIbQpSharingState::refcount for
    // every comm currently in it.
    int Load(int group) const {
        if (group < 0 || group >= ngroups_) return -1;
        return loads_[group];
    }

    int TotalRefs() const {
        int sum = 0;
        for (int l : loads_) sum += l;
        return sum;
    }

    // Busiest group -- the achieved sharing depth, and the value
    // RCCL_IB_QP_DEPTH_MULTIPLIER must currently cover. Simulated rather than
    // computed as ceil(nconns/ngroups), because occupancy-modulo does not
    // balance once teardowns are interleaved with creates.
    int MaxLoad() const {
        int m = 0;
        for (int l : loads_) m = std::max(m, l);
        return m;
    }

    // Busiest any group ever got, over the whole create/destroy sequence.
    //
    // This, not MaxLoad(), is the sharing depth a run achieved. MaxLoad() is
    // instantaneous: a churn test that stacked three comms on one group and
    // then released two reports MaxLoad()==1, and the coverage line then prints
    // "NO QP WAS SHARED" for a run that plainly did share. Measured: churn at
    // ngroups=2 reported max_comms_per_group=1 beside group_spread=3, which
    // cannot both be true of the same instant.
    int PeakLoad() const { return peakLoad_; }

    int MinLoad() const {
        int m = loads_.empty() ? 0 : loads_[0];
        for (int l : loads_) m = std::min(m, l);
        return m;
    }

    // Imbalance across groups. Reported by the tests, never asserted: how
    // evenly occupancy-modulo spreads comms is a property worth measuring (it
    // drives the depth multiplier a given ngroups demands) but not one this
    // suite is entitled to dictate.
    int Spread() const { return MaxLoad() - MinLoad(); }

private:
    int              ngroups_;
    std::vector<int> loads_;
    int              peakLoad_ = 0;
};

#endif /* MPI_TESTS_ENABLED */

#endif /* RCCL_TEST_QP_SHARE_REF_MODEL_HPP_ */
