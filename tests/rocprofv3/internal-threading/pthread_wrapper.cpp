#include <pthread.h>
#include <dlfcn.h>
#include <iostream>
#include <atomic>
#include <mutex>

#if defined(_MSC_VER)
#    define ROCPROFILER_PUBLIC_API __declspec(dllexport)
#else
#    define ROCPROFILER_PUBLIC_API __attribute__((visibility("default")))
#endif

typedef void (*rocprofiler_internal_thread_library_cb_t)(int, void*);
typedef int (*rocprofiler_at_internal_thread_create_t)(rocprofiler_internal_thread_library_cb_t precreate, rocprofiler_internal_thread_library_cb_t postcreate, int libs, void* data);

typedef void*(routine_t)(void *);

class RegisterProfiler
{
public:
    RegisterProfiler() = default;

    void try_register()
    {
        if (init.load()) return;

        auto lk = std::unique_lock{mut};
        if (init.load()) return;

        auto* handle = dlopen("librocprofiler-sdk.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!handle) return;

        auto* register_fn = (rocprofiler_at_internal_thread_create_t) dlsym(handle, "rocprofiler_at_internal_thread_create");
        if (!register_fn) throw std::runtime_error("Could not dlsym rocprofiler library");

        register_fn(pre_callback, post_callback, ~0, this);
        dlclose(handle);
        init.store(true);
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
        if (!handle) throw std::runtime_error("Could not load pthread library");

        pthread_create_fn = (PthreadFn*) dlsym(handle, "pthread_create");
        if (!pthread_create_fn) throw std::runtime_error("Could not dlsym pthread library");
    }
    void* handle = nullptr;
    PthreadFn* pthread_create_fn = nullptr;
};

RegisterProfiler* get_reg()
{
    static auto* reg = new RegisterProfiler();
    return reg;
}

ROCPROFILER_PUBLIC_API
int pthread_create(
    pthread_t* thread,
    const pthread_attr_t* attr,
    routine_t* start_routine,
    void* arg
) {
    static auto* dl = new DL();
    auto* reg = get_reg();

    reg->try_register();

    if (reg->pre_library_list.load() == 0) throw std::runtime_error("Thread not registered!");

    std::cout << "Creating thread: " << std::hex << reg->pre_library_list << std::dec << std::endl;
    return dl->pthread_create_fn(thread, attr, start_routine, arg);
}

class ConstructCheck
{
public:
    ConstructCheck() = default;
    ~ConstructCheck()
    {
        if (!get_reg()->init.load()) abort();
    }
};

ConstructCheck const_check();
