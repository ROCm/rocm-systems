/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <thread>

namespace
{

constexpr int kAmdSmiSuccess      = 0;
constexpr int kAmdSmiUnknownError = -1;

struct AmdSmiVersion
{
    uint32_t    major;
    uint32_t    minor;
    uint32_t    release;
    const char* build;
};

std::atomic<unsigned> initCount{0};
std::atomic<unsigned> versionCount{0};
std::atomic<unsigned> shutdownCount{0};

bool envEnabled(const char* name)
{
    const char* value = std::getenv(name);
    return value != nullptr && value[0] == '1';
}

} // namespace

extern "C" int amdsmi_init(uint64_t)
{
    ++initCount;
    if(envEnabled("RCCL_AMDSMI_TEST_DELAY_INIT"))
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return envEnabled("RCCL_AMDSMI_TEST_FAIL_INIT") ? kAmdSmiUnknownError : kAmdSmiSuccess;
}

extern "C" int amdsmi_get_lib_version(AmdSmiVersion* version)
{
    ++versionCount;
    if(envEnabled("RCCL_AMDSMI_TEST_FAIL_VERSION"))
        return kAmdSmiUnknownError;

    version->major   = 0;
    version->minor   = 0;
    version->release = 0;
    version->build   = "test";
    return kAmdSmiSuccess;
}

extern "C" int amdsmi_shut_down()
{
    ++shutdownCount;
    return kAmdSmiSuccess;
}

extern "C" int amdsmi_status_code_to_string(int, const char** statusString)
{
    *statusString = "injected amd_smi failure";
    return kAmdSmiSuccess;
}

extern "C" unsigned amdsmi_test_init_count()
{
    return initCount.load();
}

extern "C" unsigned amdsmi_test_version_count()
{
    return versionCount.load();
}

extern "C" unsigned amdsmi_test_shutdown_count()
{
    return shutdownCount.load();
}
