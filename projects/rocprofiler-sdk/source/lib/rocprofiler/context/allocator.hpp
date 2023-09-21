// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <utility>

namespace rocprofiler
{
namespace context
{
template <typename Tp, size_t N = 8>
struct locality_allocator
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
}  // namespace context
}  // namespace rocprofiler
