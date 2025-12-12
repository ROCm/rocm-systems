// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "attach.h"
#include "code_object_registration.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "queue_registration.hpp"
#include "table.hpp"

#include "lib/common/logging.hpp"

#include <rocprofiler-register/rocprofiler-register.h>
#include <rocprofiler-sdk/version.h>

#include <signal.h>
#include <unistd.h>

#include <condition_variable>
#include <mutex>
#include <thread>

#define ROCPROFILER_ATTACH_VERSION_MAJOR ROCPROFILER_VERSION_MAJOR
#define ROCPROFILER_ATTACH_VERSION_MINOR ROCPROFILER_VERSION_MINOR
#define ROCPROFILER_ATTACH_VERSION_PATCH ROCPROFILER_VERSION_PATCH
#define ROCPROFILER_ATTACH_VERSION                                                                 \
    ROCPROFILER_COMPUTE_VERSION(ROCPROFILER_ATTACH_VERSION_MAJOR,                                  \
                                ROCPROFILER_ATTACH_VERSION_MINOR,                                  \
                                ROCPROFILER_ATTACH_VERSION_PATCH)

using rocprofiler_register_library_api_table_func_t =
    decltype(::rocprofiler_register_library_api_table)*;

namespace
{
// Dedicated idle background thread used as a safe ptrace injection target during
// attach/detach. Previously, ptrace-based code injection targeted the main
// application thread, which could deadlock if that thread held internal mutexes.
//
// This thread names itself "rocp-bg-attach" via pthread_setname_np so the attacher
// can locate it by scanning /proc/<pid>/task/*/comm. It blocks all signals, holds
// no application-owned locks, and waits on a condition variable indefinitely until
// shutdown.
struct BackgroundThread
{
    std::thread             thread;
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    should_stop = false;
    pid_t                   tid         = 0;

    void start()
    {
        if(thread.joinable()) return;

        thread = std::thread([this]() {
            pthread_setname_np(pthread_self(), "rocp-bg-attach");

            // Prevent signals from interrupting this thread -- it exists solely as a
            // safe ptrace injection target and must never run application signal handlers.
            sigset_t all_signals;
            sigfillset(&all_signals);
            pthread_sigmask(SIG_BLOCK, &all_signals, nullptr);

            tid = static_cast<pid_t>(rocprofiler::common::get_tid());

            ROCP_INFO << "[rocprofiler-sdk-attach] Background thread started, TID=" << tid
                      << " PID=" << getpid();

            std::unique_lock<std::mutex> lock(mtx);
            cv.wait(lock, [this] { return should_stop; });
        });
    }

    ~BackgroundThread()
    {
        {
            std::lock_guard<std::mutex> lock(mtx);
            should_stop = true;
        }
        cv.notify_one();
        if(thread.joinable()) thread.join();
    }
};

BackgroundThread*
get_background_thread()
{
    static auto*& _v = rocprofiler::common::static_object<BackgroundThread>::construct();
    return _v;
}
}  // namespace

ROCPROFILER_EXTERN_C_INIT

int
rocprofiler_attach_set_api_table(const char* name,
                                 uint64_t /*lib_version*/,
                                 uint64_t /*lib_instance*/,
                                 void**                                        tables,
                                 uint64_t                                      num_tables,
                                 rocprofiler_register_library_api_table_func_t register_functor)
    ROCPROFILER_PUBLIC_API;

int
rocprofiler_attach_set_api_table(const char* name,
                                 uint64_t /*lib_version*/,
                                 uint64_t /*lib_instance*/,
                                 void**                                        tables,
                                 uint64_t                                      num_tables,
                                 rocprofiler_register_library_api_table_func_t register_functor)
{
    rocprofiler::common::init_logging("ROCPROFILER_ATTACH");

    get_background_thread()->start();

    ROCP_TRACE << "rocprofiler_attach_set_api_table called for api " << name;

    if(std::string_view{name} != "hsa")
    {
        ROCP_ERROR << "rocprofiler_attach_set_api_table was called with a table other than HSA";
        return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    ROCP_ERROR_IF(num_tables > 1) << "rocprofiler expected HSA library to pass 1 API table, not "
                                  << num_tables;

    auto* hsa_api_table = static_cast<HsaApiTable*>(tables[0]);

    rocprofiler::attach::dispatch_table_init();

    if(register_functor)
    {
        auto library_id    = rocprofiler_register_library_indentifier_t{};
        auto attach_tables = std::array<void*, 1>{rocprofiler::attach::get_dispatch_table()};
        register_functor("rocattach",
                         nullptr,
                         ROCPROFILER_ATTACH_VERSION,
                         attach_tables.data(),
                         attach_tables.size(),
                         &library_id);
    }

    // Initialize all registration services in attach
    rocprofiler::attach::queue_registration_init(hsa_api_table);
    rocprofiler::attach::code_object_registration_init(hsa_api_table);

    return ROCPROFILER_STATUS_SUCCESS;
}

int
rocprofiler_attach_get_version()
{
    return ROCPROFILER_VERSION;
}

ROCPROFILER_EXTERN_C_FINI
