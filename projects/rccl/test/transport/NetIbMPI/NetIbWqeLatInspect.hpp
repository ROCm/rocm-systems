/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_NET_IB_WQE_LAT_INSPECT_HPP_
#define RCCL_TEST_NET_IB_WQE_LAT_INSPECT_HPP_

#ifdef MPI_TESTS_ENABLED

#include <cstddef>
#include <cstdint>

#include "net_ib_wqe_lat_inspect.h"
#include "NetIbMsegInspect.hpp"

// net_ib_multiseg's comm state is net_ib's under fork names, so one reader serves both.
static_assert(sizeof(IbMsegWqeLatCommState) == sizeof(ncclIbWqeLatCommState),
              "IbMsegWqeLatCommState must mirror ncclIbWqeLatCommState");
static_assert(offsetof(IbMsegWqeLatCommState, qps) == offsetof(ncclIbWqeLatCommState, qps),
              "IbMsegWqeLatCommState.qps offset");
static_assert(sizeof(IbMsegWqeLatQpState) == sizeof(ncclIbWqeLatQpState),
              "IbMsegWqeLatQpState must mirror ncclIbWqeLatQpState");
static_assert(offsetof(IbMsegWqeLatQpState, stats) == offsetof(ncclIbWqeLatQpState, stats),
              "IbMsegWqeLatQpState.stats offset");

/*
 * One plugin's WQE latency monitor: its process globals, the one-time init
 * that sets them from the environment, and the live comm-state reader.
 */
struct WqeLatMonitor {
    bool* enabled;
    uint64_t* thresholdNs;
    bool* report;
    void (*init)();
    ncclResult_t (*getCommState)(void* comm, struct ncclIbWqeLatCommState* out);
};

inline const WqeLatMonitor& WqeLatNetIb() {
    static const WqeLatMonitor m = {
        &ncclIbWqeLatEnabled, &ncclIbWqeLatThresholdNs, &ncclIbWqeLatReportEnabled,
        [] {
            struct ncclIbWqeLatMon dummy;
            ncclIbWqeLatMonInit(&dummy);
        },
        ncclIbWqeLatGetCommState,
    };
    return m;
}

inline const WqeLatMonitor& WqeLatIbMseg() {
    static const WqeLatMonitor m = {
        &IbMsegWqeLatEnabled, &IbMsegWqeLatThresholdNs, &IbMsegWqeLatReportEnabled,
        [] {
            struct IbMsegWqeLatMon dummy;
            IbMsegWqeLatMonInit(&dummy);
        },
        [](void* comm, struct ncclIbWqeLatCommState* out) {
            return IbMsegWqeLatGetCommState(comm, reinterpret_cast<struct IbMsegWqeLatCommState*>(out));
        },
    };
    return m;
}

/*
 * Overrides the net_ib and net_ib_multiseg WQE latency monitors' process
 * globals for a scope and restores them on exit. The globals are set once from
 * the environment via std::call_once (ensureInitialized, triggered by the
 * monitor init), so that is fired first: otherwise the first QP created under
 * the override would run it and overwrite the override. When used around live
 * connections the guard must outlive every connection created under it. report
 * only matters when enabled: the library never reports with the monitor off.
 */
struct ScopedWqeLatGlobals {
    struct Saved {
        const WqeLatMonitor* monitor;
        bool enabled;
        uint64_t thresholdNs;
        bool report;
    } saved[2];
    ScopedWqeLatGlobals(bool enabled, uint64_t thresholdNs, bool report = true) {
        const WqeLatMonitor* monitors[] = {&WqeLatNetIb(), &WqeLatIbMseg()};
        for (int i = 0; i < 2; i++) {
            const WqeLatMonitor* m = monitors[i];
            m->init();
            saved[i] = {m, *m->enabled, *m->thresholdNs, *m->report};
            *m->enabled = enabled;
            *m->thresholdNs = enabled ? thresholdNs : 0;
            *m->report = enabled && report;
        }
    }
    ~ScopedWqeLatGlobals() {
        for (const Saved& s : saved) {
            *s.monitor->enabled = s.enabled;
            *s.monitor->thresholdNs = s.thresholdNs;
            *s.monitor->report = s.report;
        }
    }
    ScopedWqeLatGlobals(const ScopedWqeLatGlobals&) = delete;
    ScopedWqeLatGlobals& operator=(const ScopedWqeLatGlobals&) = delete;
};

#endif /* MPI_TESTS_ENABLED */

#endif /* RCCL_TEST_NET_IB_WQE_LAT_INSPECT_HPP_ */
