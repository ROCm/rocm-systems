/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#include <string>
#include <sys/types.h>

namespace hipFile::test {

// Check AIS capability for tests that attempt to force fast path.
// Reimplements logic from hipfile/tools/ais-check/ais-check.
class AisCapability {
public:
    // allow_skip_on_unavailable: if true, an unavailable fastpath yields Skip
    // instead of Fail.
    explicit AisCapability(bool allow_skip_on_unavailable) : allow_skip{allow_skip_on_unavailable}
    {
    }

    // Described if fastpath tests should fail with a warning message, skip, or run.
    enum class GateDecision {
        Run,
        Skip,
        Fail,
    };

    // Try 0 size I/O on ROCm >= 7.14.
    GateDecision populate(hipFileHandle_t handle, void *device_buffer);

    std::string report() const;
    std::string skipHint() const;

private:
    enum class IoProbeResult {
        NotRun,        // zero-sized I/O probe unavailable (ROCm < 7.14)
        Ok,            // zero-sized read returned 0
        NoDevice,      // -1/ENODEV: one of buffer not on NVMe device (e.g. lvm volume or md device), AIS not
                       // initialized on device, or p2pdma not ready
        InternalError, // -hipFileInternalError: usually a fastpath score() rejection
        OtherError,    // any other result
    };

    void detectKernelAis();
    void detectHipRuntime();
    void detectAmdgpu();
    void detectIoProbe(hipFileHandle_t handle, void *device_buffer);

    static IoProbeResult classifyIoProbe(ssize_t ret, int err);
    static const char   *ioProbeReason(IoProbeResult result);

    bool fastpathAvailable() const
    {
        const bool static_ok = kernel_ais && hip_runtime && amdgpu;
        if (io_probe == IoProbeResult::NotRun) {
            return static_ok;
        }
        return static_ok && io_probe == IoProbeResult::Ok;
    }

    bool allow_skip = false;

    bool          kernel_ais     = false; // AIS-init bit set on all GPU nodes in KFD topology
    bool          hip_runtime    = false; // hipAmdFileRead + hipAmdFileWrite resolvable
    bool          amdgpu         = false; // kfd_ais_rw_file present in /proc/kallsyms
    IoProbeResult io_probe       = IoProbeResult::NotRun;
    ssize_t       io_probe_ret   = 0;
    int           io_probe_errno = 0;
};

}
