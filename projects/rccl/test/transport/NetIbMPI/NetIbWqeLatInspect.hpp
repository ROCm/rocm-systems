/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TEST_NET_IB_WQE_LAT_INSPECT_HPP_
#define RCCL_TEST_NET_IB_WQE_LAT_INSPECT_HPP_

#ifdef MPI_TESTS_ENABLED

#include <cstdint>

#include "net_ib_wqe_lat_inspect.h"

/*
 * Overrides the net_ib WQE latency monitor's process globals for a scope and
 * restores them on exit. The globals are set once from the environment via
 * std::call_once (ensureInitialized, triggered by ncclIbWqeLatMonInit), so that
 * is fired first: otherwise the first QP created under the override would run
 * it and overwrite the override. When used around live connections the guard
 * must outlive every connection created under it. report only matters when
 * enabled: the library never reports with the monitor off.
 */
struct ScopedWqeLatGlobals {
    bool savedEnabled;
    uint64_t savedThresholdNs;
    bool savedReport;
    ScopedWqeLatGlobals(bool enabled, uint64_t thresholdNs, bool report = true) {
        struct ncclIbWqeLatMon dummy;
        ncclIbWqeLatMonInit(&dummy);
        savedEnabled = ncclIbWqeLatEnabled;
        savedThresholdNs = ncclIbWqeLatThresholdNs;
        savedReport = ncclIbWqeLatReportEnabled;
        ncclIbWqeLatEnabled = enabled;
        ncclIbWqeLatThresholdNs = enabled ? thresholdNs : 0;
        ncclIbWqeLatReportEnabled = enabled && report;
    }
    ~ScopedWqeLatGlobals() {
        ncclIbWqeLatEnabled = savedEnabled;
        ncclIbWqeLatThresholdNs = savedThresholdNs;
        ncclIbWqeLatReportEnabled = savedReport;
    }
    ScopedWqeLatGlobals(const ScopedWqeLatGlobals&) = delete;
    ScopedWqeLatGlobals& operator=(const ScopedWqeLatGlobals&) = delete;
};

#endif /* MPI_TESTS_ENABLED */

#endif /* RCCL_TEST_NET_IB_WQE_LAT_INSPECT_HPP_ */
