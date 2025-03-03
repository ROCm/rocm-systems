// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * This test is supposed to check if rocprofiler is wrapping all threads inside calls to:
 * rocprofiler_internal_thread_library_cb_t precreate
 * rocprofiler_internal_thread_library_cb_t postcreate
 * Currently, Runtime threads are not reported, so a tolerance "MAX_ALLOWED_THREADS" was added.
 * Some memory needs to be leaked due to global destructor issues: Test disabled for sanitizers.
 */

#include <dlfcn.h>
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <mutex>

#define PUBLIC_API __attribute__((visibility("default")))

#define ASSERT(x, msg)                                                                             \
    if(!(x))                                                                                       \
    {                                                                                              \
        std::cerr << __FILE__ << ':' << __LINE__ << " - " << msg << std::endl;                     \
        abort();                                                                                   \
    }

// Tolerance for amount of threads not inside a pre/post callback
// "3" is the number seen for the given application
#define MAX_ALLOWED_THREADS 3

// We should not link to librocprofiler-sdk.so, so we need to define the callback types
typedef void (*rocprofiler_internal_thread_library_cb_t)(int, void*);
typedef int (*rocprofiler_at_internal_thread_create_t)(
    rocprofiler_internal_thread_library_cb_t precreate,
    rocprofiler_internal_thread_library_cb_t postcreate,
    int                                      libs,
    void*                                    data);

// Used in pthread_create
typedef void*(routine_t)(void*);

namespace
{
// Number of calls to pthread_create sucessfuly wrapped in pre/post callbacks
std::atomic<size_t> wrapped_threads{0};

// Bitmask for each library that is inside a pre callback
size_t&
get_library_bitmask()
{
    // Pre-post callbacks are supposed to be called from the same thread as pthread_create
    thread_local auto* bitmask = new size_t{0};
    return *bitmask;
}

void
pre_callback(int bitmask, void* /* arg */)
{
    get_library_bitmask() |= bitmask;
}

void
post_callback(int bitmask, void* /* arg */)
{
    get_library_bitmask() &= ~bitmask;
}

class RegisterProfiler
{
public:
    RegisterProfiler() = default;

    // Returns true if rocprofiler has already been registered before this call
    bool try_register()
    {
        if(init.load()) return true;

        auto lk = std::unique_lock{mut};
        // Case for which two threads were waiting on the lock. No need to initialize again.
        if(init.load()) return false;

        auto* handle = dlopen("librocprofiler-sdk.so", RTLD_LAZY | RTLD_NOLOAD);
        // If handle is nullptr, that means pthread_create was called before
        // librocprofiler-sdk.so initialized. We try again later.
        if(!handle) return false;

        auto* register_fn = (rocprofiler_at_internal_thread_create_t) dlsym(
            handle, "rocprofiler_at_internal_thread_create");
        ASSERT(register_fn, "Could not dlsym rocprofiler library");

        register_fn(pre_callback, post_callback, ~0, nullptr);
        dlclose(handle);
        init.store(true);
        return false;
    }

    // Indicates we have registered rocprofiler_at_internal_thread_create_t
    std::atomic<bool> init{false};
    std::mutex        mut{};
};

class DL
{
    using PthreadFn = decltype(pthread_create);

public:
    DL()
    {
        handle = dlopen("libpthread.so.0", RTLD_LAZY | RTLD_LOCAL);
        ASSERT(handle, "Could not load pthread library");

        pthread_create_fn = (PthreadFn*) dlsym(handle, "pthread_create");
        ASSERT(pthread_create_fn, "Could not dlsym pthread library");
    }
    ~DL()
    {
        pthread_create_fn = nullptr;
        if(handle) dlclose(handle);
    }
    void*      handle            = nullptr;
    PthreadFn* pthread_create_fn = nullptr;
};

__attribute__((destructor)) void
check_wrapped_threads()
{
    // Ensures the test has actually run
    ASSERT(wrapped_threads.load() > 0, "No thread was created inside a pre/post callback");
}
}  // namespace

// This function wraps pthread_create defined in pthread.h
PUBLIC_API
int
pthread_create(pthread_t* thread, const pthread_attr_t* attr, routine_t* start_routine, void* arg)
{
    static auto* reg = new RegisterProfiler();
    static auto* dl  = new DL();

    // Number of threads not wrapped inside a pre/post callback
    static auto unwrapped_threads = std::atomic<size_t>{0};

    // If try_register returns false, SDK was not loaded yet or has just been loaded.
    // We have to ignore the first initialization because we may have just missed a pre-callback.
    if(reg->try_register())
    {
        // Comparing to zero checks if any library has called pre-callback.
        if(get_library_bitmask() == 0)
        {
            size_t count = unwrapped_threads.fetch_add(1);

            // This will fail if either rocprofiler does not wrap a thread,
            // or if the runtime decides to create new threads. TODO: HIP/HSA callbacks
            ASSERT(count < MAX_ALLOWED_THREADS,
                   "Limit reached for number of threads not inside a pre/post callback!");
        }
        else
        {
            // We have sucessfuly created a thread wrapped in pre/post callbacks
            wrapped_threads.fetch_add(1);
        }
    }

    return dl->pthread_create_fn(thread, attr, start_routine, arg);
}
