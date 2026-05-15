/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include "nccl.h"
#include "net.h"
#include "transport.h"
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <utility>

/**
 * @file ResourceGuards.hpp
 * @brief Comprehensive RAII resource guards for automatic cleanup in tests
 *
 * Provides all RAII guard types for automatic resource management:
 * - ScopeGuard: Generic cleanup for any action (with lambdas)
 * - AutoGuard: Typed guards for resources with simple cleanup functions
 * - ResourceGuard: Typed guards for resources with stateful deleters
 * - Specialized guards: NcclRegHandleGuard, etc.
 *
 * Guards ensure cleanup even when ASSERT_* fails in tests.
 * See MPITestRunner.md for detailed usage documentation.
 */

namespace RCCLTestGuards
{

// ============================================================================
// ScopeGuard - Generic cleanup for arbitrary actions
// ============================================================================

/**
 * @class ScopeGuard
 * @brief Generic RAII scope guard for custom cleanup logic
 *
 * Executes a cleanup function on scope exit (normal return, early return, or exception).
 * Useful for resources that don't have dedicated RAII guards or for one-off cleanup needs.
 *
 * @par Example:
 * @code
 * void* buffer = nullptr;
 * hipMalloc(&buffer, size);
 * auto guard = makeScopeGuard([&]() { if(buffer) hipFree(buffer); });
 * // Automatic cleanup on scope exit
 * @endcode
 *
 * @tparam Func Callable type (lambda, function pointer, functor)
 */
template<typename Func>
class ScopeGuard
{
    Func cleanup_;     ///< Cleanup function to execute on scope exit
    bool dismissed_;   ///< If true, skip cleanup (for ownership transfer)

public:
    explicit ScopeGuard(Func f) noexcept : cleanup_(std::move(f)), dismissed_(false) {}

    ~ScopeGuard() noexcept
    {
        if(!dismissed_)
        {
            cleanup_();
        }
    }

    void dismiss() noexcept { dismissed_ = true; }
    void restore() noexcept { dismissed_ = false; }

    ScopeGuard(ScopeGuard&& other) noexcept
        : cleanup_(std::move(other.cleanup_)), dismissed_(other.dismissed_)
    {
        other.dismissed_ = true;
    }

    ScopeGuard& operator=(ScopeGuard&& other) noexcept
    {
        if(this != &other)
        {
            if(!dismissed_)
            {
                cleanup_();
            }
            cleanup_   = std::move(other.cleanup_);
            dismissed_ = other.dismissed_;
            other.dismissed_ = true;
        }
        return *this;
    }

    ScopeGuard(const ScopeGuard&)            = delete;
    ScopeGuard& operator=(const ScopeGuard&) = delete;
};

/**
 * @brief Factory function to create ScopeGuard with type deduction
 *
 * @par Example:
 * @code
 * auto guard = makeScopeGuard([&]() { cleanup(); });
 * @endcode
 */
template<typename Func>
ScopeGuard<Func> makeScopeGuard(Func f)
{
    return ScopeGuard<Func>(std::move(f));
}

/**
 * @def SCOPE_EXIT
 * @brief Convenience macro for creating anonymous scope guards
 *
 * @par Example:
 * @code
 * void* buffer = nullptr;
 * hipMalloc(&buffer, size);
 * SCOPE_EXIT(if(buffer) hipFree(buffer));
 * @endcode
 */
#define SCOPE_EXIT_CONCAT_IMPL(a, b) a##b
#define SCOPE_EXIT_CONCAT(a, b) SCOPE_EXIT_CONCAT_IMPL(a, b)
#define SCOPE_EXIT(code) \
    auto SCOPE_EXIT_CONCAT(scope_guard_, __LINE__) = RCCLTestGuards::makeScopeGuard([&]() { code; })

// ============================================================================
// AutoGuard & ResourceGuard - Typed resource management
// ============================================================================

/**
 * @class AutoGuard
 * @brief Modern RAII guard using non-type template parameter for deleter
 *
 * Uses C++17's auto template parameters to directly reference cleanup functions,
 * eliminating the need for deleter functors in simple cases.
 *
 * @tparam T Resource handle type
 * @tparam DeleterFunc Function pointer for cleanup (auto-deduced)
 */
template<typename T, auto DeleterFunc>
class AutoGuard
{
private:
    T    resource_;
    bool dismissed_;

public:
    explicit AutoGuard(T resource = T{}) : resource_(resource), dismissed_(false) {}

    ~AutoGuard()
    {
        if(!dismissed_ && resource_)
        {
            DeleterFunc(resource_);
        }
    }

    // Get the resource handle
    T get() const
    {
        return resource_;
    }
    // Get pointer to resource handle (for API calls)
    T* ptr()
    {
        return &resource_;
    }
    // Set the resource handle
    void set(T resource)
    {
        resource_ = resource;
    }
    // Dismiss the guard (prevent cleanup)
    void dismiss()
    {
        dismissed_ = true;
    }

    // Release ownership (prevent cleanup)
    T release()
    {
        dismissed_ = true;
        return resource_;
    }

    AutoGuard(const AutoGuard&)            = delete;
    AutoGuard& operator=(const AutoGuard&) = delete;

    AutoGuard(AutoGuard&& other) noexcept : resource_(other.resource_), dismissed_(other.dismissed_)
    {
        other.dismissed_ = true;
    }

    AutoGuard& operator=(AutoGuard&& other) noexcept
    {
        if(this != &other)
        {
            if(!dismissed_ && resource_)
            {
                DeleterFunc(resource_);
            }
            resource_        = other.resource_;
            dismissed_       = other.dismissed_;
            other.dismissed_ = true;
        }
        return *this;
    }
};

/**
 * @class ResourceGuard
 * @brief Generic RAII guard template for resources with complex cleanup
 *
 * Uses a functor-based deleter for stateful deleters requiring additional context.
 * For simple cleanup functions, prefer AutoGuard<T, func> instead.
 *
 * @tparam T Resource handle type
 * @tparam Deleter Functor type for cleanup
 */
template<typename T, typename Deleter>
class ResourceGuard
{
private:
    T       resource_;
    Deleter deleter_;
    bool    owns_;

public:
    // Construct a resource guard
    // @param resource Resource handle (can be nullptr/0)
    // @param deleter Cleanup function/functor
    explicit ResourceGuard(T resource = T{}, Deleter deleter = Deleter{})
        : resource_(resource), deleter_(std::move(deleter)), owns_(true)
    {}

    // Destructor - automatically cleans up resource
    ~ResourceGuard()
    {
        if(owns_ && resource_)
        {
            deleter_(resource_);
        }
    }

    // Get the resource handle
    T get() const
    {
        return resource_;
    }
    // Get pointer to resource handle (for API calls)
    T* ptr()
    {
        return &resource_;
    }
    // Set the resource handle
    void set(T resource)
    {
        resource_ = resource;
    }

    // Reset the resource handle
    // @param resource New resource handle (can be nullptr/0)
    void reset(T resource = T{})
    {
        if(owns_ && resource_ && resource_ != resource)
        {
            deleter_(resource_);
        }
        resource_ = resource;
        owns_     = true;
    }

    T release()
    {
        owns_ = false;
        return resource_;
    }

    ResourceGuard(const ResourceGuard&)            = delete;
    ResourceGuard& operator=(const ResourceGuard&) = delete;

    ResourceGuard(ResourceGuard&& other) noexcept
        : resource_(other.resource_), deleter_(std::move(other.deleter_)), owns_(other.owns_)
    {
        other.owns_ = false;
    }

    ResourceGuard& operator=(ResourceGuard&& other) noexcept
    {
        if(this != &other)
        {
            // Clean up current resource
            if(owns_ && resource_)
            {
                deleter_(resource_);
            }
            // Take ownership of other's resource
            resource_   = other.resource_;
            deleter_    = std::move(other.deleter_);
            owns_       = other.owns_;
            other.owns_ = false;
        }
        return *this;
    }
};

// Note: Simple stateless deleters are replaced by wrapper functions + AutoGuard.
// Only stateful deleters that need additional context are kept here.
// Common deleters (NCCL-specific, used across many tests)
struct NcclRegHandleDeleter
{
    ncclComm_t comm;
    explicit NcclRegHandleDeleter(ncclComm_t c = nullptr) : comm(c) {}
    void operator()(void* reg_handle) const
    {
        if(reg_handle && comm)
        {
            ncclCommDeregister(comm, reg_handle);
        }
    }
};

// Wrapper functions for AutoGuard (void-returning cleanup functions)
inline void hipFreeWrapper(void* ptr)
{
    if(ptr)
    {
        hipError_t err = hipFree(ptr);
        if(err != hipSuccess)
        {
            fprintf(stderr,
                    "WARNING: hipFree failed in destructor: %s (ptr=%p)\n",
                    hipGetErrorString(err),
                    ptr);
        }
    }
}

inline void hipStreamDestroyWrapper(hipStream_t stream)
{
    if(stream)
    {
        hipError_t err = hipStreamDestroy(stream);
        if(err != hipSuccess)
        {
            fprintf(stderr,
                    "WARNING: hipStreamDestroy failed in destructor: %s (stream=%p)\n",
                    hipGetErrorString(err),
                    static_cast<void*>(stream));
        }
    }
}

inline void hipEventDestroyWrapper(hipEvent_t event)
{
    if(event)
    {
        hipError_t err = hipEventDestroy(event);
        if(err != hipSuccess)
        {
            fprintf(stderr,
                    "WARNING: hipEventDestroy failed in destructor: %s (event=%p)\n",
                    hipGetErrorString(err),
                    static_cast<void*>(event));
        }
    }
}

inline void ncclCommDestroyWrapper(ncclComm_t comm)
{
    if(comm)
    {
        ncclResult_t result = ncclCommDestroy(comm);
        if(result != ncclSuccess)
        {
            fprintf(stderr,
                    "WARNING: ncclCommDestroy failed in destructor: %s (comm=%p)\n",
                    ncclGetErrorString(result),
                    static_cast<void*>(comm));
        }
    }
}

inline void freeWrapper(void* ptr)
{
    if(ptr)
        free(ptr);
}

// Type aliases for AutoGuard-based guards
using HostBufferAutoGuard   = AutoGuard<void*, freeWrapper>;
using DeviceBufferAutoGuard = AutoGuard<void*, hipFreeWrapper>;
using HipStreamAutoGuard    = AutoGuard<hipStream_t, hipStreamDestroyWrapper>;
using HipEventAutoGuard     = AutoGuard<hipEvent_t, hipEventDestroyWrapper>;
using NcclCommAutoGuard     = AutoGuard<ncclComm_t, ncclCommDestroyWrapper>;

// Type aliases for ResourceGuard-based guards (common/NCCL-specific)
using NcclRegHandleGuard = ResourceGuard<void*, NcclRegHandleDeleter>;

// Factory methods for ResourceGuard
template<typename T, typename Deleter>
inline auto makeGuard(T resource, Deleter deleter) -> ResourceGuard<T, Deleter>
{
    return ResourceGuard<T, Deleter>(resource, std::move(deleter));
}

inline NcclRegHandleGuard makeRegHandleGuard(void* handle, ncclComm_t comm)
{
    return NcclRegHandleGuard(handle, NcclRegHandleDeleter(comm));
}

template<typename T, typename Deleter>
inline auto makeCustomGuard(T resource, Deleter deleter) -> ResourceGuard<T, Deleter>
{
    return ResourceGuard<T, Deleter>(resource, std::move(deleter));
}

// Factory methods for AutoGuard
template<typename T, auto DeleterFunc>
inline AutoGuard<T, DeleterFunc> makeAutoGuard(T resource)
{
    return AutoGuard<T, DeleterFunc>(resource);
}

inline HostBufferAutoGuard makeHostBufferAutoGuard(void* buffer)
{
    return HostBufferAutoGuard(buffer);
}

inline DeviceBufferAutoGuard makeDeviceBufferAutoGuard(void* buffer)
{
    return DeviceBufferAutoGuard(buffer);
}

inline HipStreamAutoGuard makeStreamAutoGuard(hipStream_t stream)
{
    return HipStreamAutoGuard(stream);
}

inline HipEventAutoGuard makeEventAutoGuard(hipEvent_t event)
{
    return HipEventAutoGuard(event);
}

inline NcclCommAutoGuard makeCommAutoGuard(ncclComm_t comm)
{
    return NcclCommAutoGuard(comm);
}

} // namespace RCCLTestGuards

