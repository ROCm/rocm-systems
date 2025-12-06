// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"
#include "shared_mutex"

#define LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US 5000
#define LOCK_HELD_TOO_LONG_WARNING_TIME_LIMIT_US 500000


#if defined(LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US) || defined(LOCK_HELD_TOO_LONG_WARNING_TIME_LIMIT_US)
#include "spdlog/spdlog.h"
#endif

namespace RocProfVis
{
namespace DataModel
{
// base class for all classes that implement property access interface methods
class DmBase {
public:
    // virtual destructor must be defined in order to free derived classes resources  
    virtual ~DmBase(){}
    // Virtual method to read object property as uint64
    // @param property - property enumeration 
    // @param index - index of any indexed property
    // @param value - pointer reference to uint64_t return value
    // @return status of operation
    virtual rocprofvis_dm_result_t GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value);
    // Virtual method to read object property as int64
    // @param property - property enumeration 
    // @param index - index of any indexed property
    // @param value - pointer reference to int64_t return value
    // @return status of operation
    virtual rocprofvis_dm_result_t GetPropertyAsInt64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, int64_t* value);
    // Virtual method to read object property as char*
    // @param property - property enumeration 
    // @param index - index of any indexed property
    // @param value - pointer reference to char* return value
    // @return status of operation
    virtual rocprofvis_dm_result_t GetPropertyAsCharPtr(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, char** value);
    // Virtual method to read object property as double
    // @param property - property enumeration 
    // @param index - index of any indexed property
    // @param value - pointer reference to double return value
    // @return status of operation
    virtual rocprofvis_dm_result_t GetPropertyAsDouble(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, double* value);
    // Virtual method to read object property as handle (aka void*)
    // @param property - property enumeration 
    // @param index - index of any indexed property
    // @param value - pointer reference to handle (aka void*) return value
    // @return status of operation
    virtual rocprofvis_dm_result_t GetPropertyAsHandle(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, rocprofvis_dm_handle_t* value);
    // Returns class mutex
    virtual std::shared_mutex* Mutex() { return nullptr; }
#ifdef TEST
    // Virtual method to get property symbol for testing/debugging
    // @param property - property enumeration rocprofvis_dm_extdata_property_t
    // @return pointer to property symbol string 
    virtual const char*            GetPropertySymbol(rocprofvis_dm_property_t property);
#endif
};


template <typename LockType>
class TimedLock
{
public:
    using clock = std::chrono::high_resolution_clock;
    TimedLock() = default;
    template <typename MutexType>
    TimedLock(MutexType& mtx, const char* func_name, DmBase* object)
#if defined(LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US) || defined(LOCK_HELD_TOO_LONG_WARNING_TIME_LIMIT_US)
    : m_start(std::chrono::high_resolution_clock::now())
    , m_func_name(func_name)
    , m_owner_object(object)
#endif
    , m_lock(mtx)
    {
#ifdef LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US
        uint64_t us =
            std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - m_start)
                .count();
        if(us >= LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US)
        {
            spdlog::debug("Lock waited {}us for {}, class {}", us, m_func_name, typeid(*m_owner_object).name());
        }
#endif
    }

    ~TimedLock() {
#ifdef LOCK_HELD_TOO_LONG_WARNING_TIME_LIMIT_US
        uint64_t us =
            std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - m_start)
                .count();
        if(us >= LOCK_HELD_TOO_LONG_WARNING_TIME_LIMIT_US)
        {
            spdlog::debug("Lock was held for {}us by {}, class {}", us,
                          m_func_name, typeid(*m_owner_object).name());
        }

#endif
    }

    void lock()
    {
#ifdef LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US
        m_start = clock::now();
#endif
        m_lock.lock();
#ifdef LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US
        uint64_t us =
            std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - m_start)
                .count();
        if(us >= LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US)
        {
            spdlog::debug("Manual lock waited {}us for {}, class {}", us, m_func_name, typeid(*m_owner_object).name());
        }

#endif
    }

    void unlock() { 
#ifdef LOCK_HELD_TOO_LONG_WARNING_TIME_LIMIT_US
        uint64_t us =
            std::chrono::duration_cast<std::chrono::microseconds>(clock::now() - m_start)
                .count();
        if(us >= LOCK_HELD_TOO_LONG_WARNING_TIME_LIMIT_US)
        {
            spdlog::debug("Manual lock was held for {}us by {}, class {}", us, m_func_name,
                          typeid(*m_owner_object).name());
        }

#endif
        m_lock.unlock(); 
    }

    bool owns_lock() const { return m_lock.owns_lock(); }

    TimedLock(const TimedLock&)            = delete;
    TimedLock& operator=(const TimedLock&) = delete;
    TimedLock(TimedLock&&)                 = default;
    TimedLock& operator=(TimedLock&&)      = default;

private:
#if defined(LOCK_WAITED_TOO_LONG_WARNING_TIME_LIMIT_US) || defined(LOCK_HELD_TOO_LONG_WARNING_TIME_LIMIT_US)
    std::chrono::high_resolution_clock::time_point m_start;
    const char*                                    m_func_name;
    DmBase*                                        m_owner_object;
#endif
    LockType m_lock;
};


}  // namespace DataModel
}  // namespace RocProfVis