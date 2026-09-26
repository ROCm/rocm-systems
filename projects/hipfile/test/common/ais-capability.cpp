/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "ais-capability.h"

#include "hip.h"
#include "hipfile.h"

#include <charconv>
#include <cerrno>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace hipFile::test {

namespace {

    // BIT6 of the KFD topology node "capability" field signals that amdgpu has
    // initialized AIS on that node, implying the kernel supports P2PDMA.
    constexpr uint64_t KFD_AIS_CAPABILITY_BIT = 0x40;

}

void
AisCapability::detectKernelAis()
{
    const std::string topology_nodes = "/sys/class/kfd/kfd/topology/nodes";

    bool any_gpu = false;

    std::error_code ec;
    auto            node = std::filesystem::directory_iterator{topology_nodes, ec};
    const auto      end  = std::filesystem::directory_iterator{};
    for (; !ec && node != end; node.increment(ec)) {
        const auto    properties_path = node->path() / "properties";
        std::ifstream in{properties_path};
        if (!in.is_open()) {
            continue;
        }

        uint64_t    capability = 0;
        uint32_t    simd_count = 0;
        std::string line;
        while (std::getline(in, line)) {
            std::istringstream fields{line};
            std::string        key;
            std::string        value_text;
            if (!(fields >> key >> value_text) || (key != "capability" && key != "simd_count")) {
                continue;
            }

            uint64_t value = 0;
            auto [last, parse_error] =
                std::from_chars(value_text.data(), value_text.data() + value_text.size(), value);
            if (parse_error != std::errc{} || last != value_text.data() + value_text.size()) {
                continue;
            }

            if (key == "capability") {
                capability = value;
            }
            else if (value <= std::numeric_limits<uint32_t>::max()) {
                simd_count = static_cast<uint32_t>(value);
            }
        }

        if (simd_count == 0) {
            continue; // Not a GPU, disregard
        }
        any_gpu = true;
        if ((capability & KFD_AIS_CAPABILITY_BIT) == 0) {
            kernel_ais = false;
            return;
        }
    }

    kernel_ais = any_gpu;
}

void
AisCapability::detectHipRuntime()
{
    hip_runtime = hipFile::getHipAmdFileReadPtr() != nullptr && hipFile::getHipAmdFileWritePtr() != nullptr;
}

void
AisCapability::detectAmdgpu()
{
    std::ifstream kallsyms{"/proc/kallsyms"};
    if (!kallsyms.is_open()) {
        std::cerr << "Unable to open /proc/kallsyms\n";
        amdgpu = false;
        return;
    }

    std::string line;
    while (std::getline(kallsyms, line)) {
        if (line.find("kfd_ais_rw_file") != std::string::npos) {
            amdgpu = true;
            return;
        }
    }
    amdgpu = false;
}

void
AisCapability::detectIoProbe(hipFileHandle_t handle, void *device_buffer)
{
#if HIP_VERSION_MAJOR > 7 || (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR >= 14)
    errno          = 0;
    ssize_t ret    = hipFileRead(handle, device_buffer, /*size=*/0, /*file_offset=*/0, /*buffer_offset=*/0);
    io_probe_ret   = ret;
    io_probe_errno = errno;
    io_probe       = classifyIoProbe(ret, errno);
#else
    // Avoid unused argument compiler warning.
    static_cast<void>(handle);
    static_cast<void>(device_buffer);
    io_probe = IoProbeResult::NotRun;
#endif
}

AisCapability::IoProbeResult
AisCapability::classifyIoProbe(ssize_t ret, int err)
{
    if (ret == 0) {
        return IoProbeResult::Ok;
    }
    if (ret == -1 && err == ENODEV) {
        return IoProbeResult::NoDevice;
    }
    if (ret == -static_cast<ssize_t>(hipFileInternalError)) {
        return IoProbeResult::InternalError;
    }
    return IoProbeResult::OtherError;
}

const char *
AisCapability::ioProbeReason(IoProbeResult result)
{
    switch (result) {
        case IoProbeResult::NotRun:
            return "not run (zero-sized I/O probe unavailable)";
        case IoProbeResult::Ok:
            return "ok";
        case IoProbeResult::NoDevice:
            return "ENODEV: storage not on a P2PDMA-capable NVMe device (e.g. lvm/md "
                   "volume), AIS not initialized on the device, or pcie_p2pdma_distance < 0";
        case IoProbeResult::InternalError:
            return "fastpath rejected the request (unsupported filesystem, misaligned I/O, "
                   "or non-device buffer)";
        case IoProbeResult::OtherError:
            return "unexpected error (zero-sized I/O may be unsupported)";
        default:
            return "unknown";
    }
}

AisCapability::GateDecision
AisCapability::populate(hipFileHandle_t handle, void *device_buffer)
{
    static const AisCapability static_capability = [] {
        AisCapability capability{false};
        capability.detectKernelAis();
        capability.detectHipRuntime();
        capability.detectAmdgpu();
        return capability;
    }();

    kernel_ais  = static_capability.kernel_ais;
    hip_runtime = static_capability.hip_runtime;
    amdgpu      = static_capability.amdgpu;
    detectIoProbe(handle, device_buffer);

    if (fastpathAvailable()) {
        return GateDecision::Run;
    }
    return allow_skip ? GateDecision::Skip : GateDecision::Fail;
}

std::string
AisCapability::report() const
{
    auto pass_fail = [](bool ok) { return ok ? "Pass" : "Fail"; };

    std::ostringstream os;
    os << "Fastpath Validation:\n";
    os << "  HIP Runtime Check:  " << pass_fail(hip_runtime) << "\n";
    os << "  amdgpu Check:       " << pass_fail(amdgpu) << "\n";
    os << "  AIS init check:     " << pass_fail(kernel_ais) << "\n";

    const char *io_status =
        io_probe == IoProbeResult::Ok ? "Pass" : (io_probe == IoProbeResult::NotRun ? "Skipped" : "Fail");
    os << "  Test I/O check:     " << io_status;
    if (io_probe != IoProbeResult::Ok && io_probe != IoProbeResult::NotRun) {
        os << " (" << ioProbeReason(io_probe) << "; ret=" << io_probe_ret << ", errno=" << io_probe_errno
           << ")";
    }

    return os.str();
}

std::string
AisCapability::skipHint() const
{
    return "To skip these tests instead of failing, configure the build with "
           "-DHIPFILE_ALLOW_SKIP_FASTPATH_TESTS=ON "
           "(passes --allow-skip-fastpath to the system test binary).";
}

}
