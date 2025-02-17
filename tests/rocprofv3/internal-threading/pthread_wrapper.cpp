#include <dlfcn.h>
#include <pthread.h>
#include <atomic>
#include <iostream>
#include <mutex>

#if defined(_MSC_VER)
#    define ROCPROFILER_PUBLIC_API __declspec(dllexport)
#else
#    define ROCPROFILER_PUBLIC_API __attribute__((visibility("default")))
#endif

#define ASSERT(x, msg)                                                                             \
    if(!(x))                                                                                       \
    {                                                                                              \
        std::cerr << __FILE__ << ':' << __LINE__ << " - " << msg << std::endl;                     \
        abort();                                                                                   \
    }

#define MAX_ALLOWED_THREADS 3

typedef void (*rocprofiler_internal_thread_library_cb_t)(int, void*);
typedef int (*rocprofiler_at_internal_thread_create_t)(
    rocprofiler_internal_thread_library_cb_t precreate,
    rocprofiler_internal_thread_library_cb_t postcreate,
    int                                      libs,
    void*                                    data);

typedef void*(routine_t)(void*);

class RegisterProfiler
{
public:
    RegisterProfiler() = default;

    bool try_register()
    {
        if(init.load()) return true;

        auto lk = std::unique_lock{mut};
        if(init.load()) return false;

        auto* handle = dlopen("librocprofiler-sdk.so", RTLD_LAZY | RTLD_NOLOAD);
        if(!handle) return false;

        auto* register_fn = (rocprofiler_at_internal_thread_create_t) dlsym(
            handle, "rocprofiler_at_internal_thread_create");
        ASSERT(register_fn, "Could not dlsym rocprofiler library");

        register_fn(pre_callback, post_callback, ~0, this);
        dlclose(handle);
        init.store(true);
        return false;
    }

    static void pre_callback(int bitmask, void* arg)
    {
        static_cast<RegisterProfiler*>(arg)->pre_library_list.fetch_or(bitmask);
    }

    static void post_callback(int bitmask, void* arg)
    {
        static_cast<RegisterProfiler*>(arg)->pre_library_list.fetch_and(~bitmask);
    }

    std::atomic<size_t> pre_library_list{0};
    std::atomic<bool>   init{false};
    std::mutex          mut;
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

ROCPROFILER_PUBLIC_API
int
pthread_create(pthread_t* thread, const pthread_attr_t* attr, routine_t* start_routine, void* arg)
{
    static auto* reg = new RegisterProfiler();
    static auto* dl  = new DL();

    static std::atomic<size_t> unwrapped_threads{0};

    if(reg->try_register() && reg->pre_library_list.load() == 0) unwrapped_threads++;

    ASSERT(unwrapped_threads <= MAX_ALLOWED_THREADS,
           "Limit reached for number of thread not inside a pre/post callback!");

    return dl->pthread_create_fn(thread, attr, start_routine, arg);
}
