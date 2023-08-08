
#pragma once

#include <rocprofiler/rocprofiler.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <utility>

namespace
{
inline size_t  // NOLINTNEXTLINE
get_domain_max_op(rocprofiler_tracer_activity_domain_t _domain)
{
    switch(_domain)
    {
        case ACTIVITY_DOMAIN_NONE: return -1;
        case ACTIVITY_DOMAIN_HSA_API: return 0;
        case ACTIVITY_DOMAIN_HSA_OPS: return 0;
        case ACTIVITY_DOMAIN_HIP_OPS: return 0;
        case ACTIVITY_DOMAIN_HIP_API: return 0;
        case ACTIVITY_DOMAIN_KFD_API: return -1;
        case ACTIVITY_DOMAIN_EXT_API: return -1;
        case ACTIVITY_DOMAIN_ROCTX: return 0;
        case ACTIVITY_DOMAIN_HSA_EVT: return 0;
        case ACTIVITY_DOMAIN_NUMBER: return -1;
    }
    return -1;
}

template <typename Tp, size_t N = 8>
struct allocator
{
    void construct(Tp* const _p, const Tp& _v) const { ::new((void*) _p) Tp{_v}; }
    void construct(Tp* const _p, Tp&& _v) const { ::new((void*) _p) Tp{std::move(_v)}; }

    void destroy(Tp* const _p) const { _p->~Tp(); }

    static constexpr auto size = sizeof(Tp);
    using buffer_value_t       = char[size];

    struct buffer_entry
    {
        std::atomic_flag flag  = ATOMIC_FLAG_INIT;
        buffer_value_t   value = {};

        void* get()
        {
            if(flag.test_and_set())
            {
                return &value[0];
            }
            return nullptr;
        }

        bool reset(void* p)
        {
            if(static_cast<void*>(&value[0]) == p)
            {
                flag.clear();
                return true;
            }
            return false;
        }
    };

    static auto& get_buffer()
    {
        static auto _v = std::array<buffer_entry, N>{};
        return _v;
    }

    Tp* allocate(const size_t n) const
    {
        if(n == 0) return nullptr;

        if(n == 1)
        {
            // try an find in buffer for data locality
            for(auto& itr : get_buffer())
            {
                auto* _p = itr.get();
                if(_p) return static_cast<Tp*>(_p);
            }
        }

        auto* _p = new char[n * size];
        return reinterpret_cast<Tp*>(_p);
    }

    void deallocate(Tp* const ptr, const size_t /*unused*/) const
    {
        for(auto& itr : get_buffer())
        {
            if(itr.reset(ptr)) return;
        }

        delete ptr;
    }

    Tp* allocate(const size_t n, const void* const /* hint */) const { return allocate(n); }

    void reserve(const size_t) {}
};

}  // namespace
