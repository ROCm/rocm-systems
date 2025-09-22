#include "hsa-runtime.hpp"

#include <rocprofiler-register/rocprofiler-register.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <string_view>

#define ROCP_REG_VERSION                                                                 \
    ROCPROFILER_REGISTER_COMPUTE_VERSION_3(                                              \
        HSA_VERSION_MAJOR, HSA_VERSION_MINOR, HSA_VERSION_PATCH)

ROCPROFILER_REGISTER_DEFINE_IMPORT(hsa, ROCP_REG_VERSION)

#ifndef ROCP_REG_FILE_NAME
#    define ROCP_REG_FILE_NAME                                                           \
        ::std::string{ __FILE__ }                                                        \
            .substr(::std::string_view{ __FILE__ }.find_last_of('/') + 1)                \
            .c_str()
#endif

namespace hsa
{
namespace
{
auto&
get_hsa_api_table_impl()
{
    static auto _table = std::atomic<HsaApiTable*>{ nullptr };
    return _table;
}

void
register_profiler_impl()
{
    static auto _const_api_table = HsaApiTable{};
    initialize_hsa_api_table(&_const_api_table);

    // set this before any recursive opportunity arises
    get_hsa_api_table_impl().exchange(&_const_api_table);

    // create a copy of the api table for modification by registration
    static auto _profiler_api_table = HsaApiTable{};
    copy_hsa_api_table(&_profiler_api_table, &_const_api_table);

    void* _profiler_api_table_v = static_cast<void*>(&_profiler_api_table);

    auto lib_id = rocprofiler_register_library_indentifier_t{};
    auto success =
        rocprofiler_register_library_api_table("hsa",
                                               &ROCPROFILER_REGISTER_IMPORT_FUNC(hsa),
                                               ROCP_REG_VERSION,
                                               &_profiler_api_table_v,
                                               1,
                                               &lib_id);

    if(success == 0)
    {
        printf("[%s] hsa identifier %lu\n", ROCP_REG_FILE_NAME, lib_id.handle);
        auto* _api_table = &_const_api_table;
        if(!get_hsa_api_table_impl().compare_exchange_strong(_api_table,
                                                             &_profiler_api_table))
        {
            // with the current impl, if we ever get here, someone is calling one the
            // functions in this anonymous namespace that shouldn't
            std::cerr
                << "register_profiler_impl expected the API table to be the internal "
                   "implementation and yet it is not. something went wrong.\n";
            abort();
        }
    }
    else if(success != ROCP_REG_NO_TOOLS)
    {
        std::cerr << "HSA library failed to register with rocprofiler-register: "
                  << rocprofiler_register_error_string(success) << "\n";
        exit(EXIT_FAILURE);
    }
}

void
register_profiler()
{
    // this registration scheme is designed to minimize overhead once
    // registered (only pay cost of checking atomic boolean)
    // once the profiler is registered. If the library has not
    // been registered and two or more threads try to register concurrently
    // the first thread to acquire the lock below, will block the
    // threads until registration is complete. However,
    // if the same thread performing the registration re-enters this function
    // i.e. this library's API is called during registration, this function
    // will prevent a deadlock by not attempting to re-enter the
    // the call-once and not releasing any waiting threads by flipping
    // the _is_registered field to true.
    static auto _is_registered = std::atomic<bool>{ false };

    if(!_is_registered.load(std::memory_order_acquire))
    {
        using mutex_t      = std::recursive_mutex;
        using auto_lock_t  = std::unique_lock<mutex_t>;
        static auto _once  = std::once_flag{};
        static auto _mutex = mutex_t{};

        // defer the lock so we can check for recursion
        auto _lk = auto_lock_t{ _mutex, std::defer_lock };

        // this will be true if the same thread currently executing the call_once invokes
        // the library's API while registering the profiler (e.g. tool which wants to
        // instrument HSA API invokes a HSA function while registering with the profiler)
        // we allow this thread to proceed and access the "const" API table but
        // return so it does not flip _is_registered to true, which would result
        // in any subsequent threads not waiting until the library is fully registered,
        // resulting in missed callbacks for the tools
        if(_lk.owns_lock()) return;

        // ensures any subsequent threads wait until the first thread
        // finishes registration
        _lk.lock();
        // call_once to ensure that we only register once
        std::call_once(_once, register_profiler_impl);
        // the first thread has completed registration and all
        // threads waiting on lock will be released and this
        // block will not be entered again
        _is_registered.exchange(true, std::memory_order_release);
    }
}
}  // namespace

HsaApiTable*
get_hsa_api_table()
{
    register_profiler();
    return get_hsa_api_table_impl().load(std::memory_order_relaxed);
}

void
hsa_init()
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __FUNCTION__);
}
}  // namespace hsa

extern "C" {
void
hsa_init(void)
{
    hsa::get_hsa_api_table()->hsa_init_fn();
}
}
