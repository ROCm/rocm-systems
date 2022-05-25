// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/config.hpp"
#include "library/defines.hpp"
#include "library/thread_data.hpp"

#include <timemory/hash/types.hpp>
#include <timemory/tpls/cereal/cereal.hpp>
#include <timemory/utility/demangle.hpp>

#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <string>
#include <vector>

namespace omnitrace
{
namespace critical_trace
{
enum class Device : uint8_t
{
    NONE = 0,
    CPU,
    GPU,
    ANY,
};

enum class Phase : uint8_t
{
    NONE = 0,
    BEGIN,
    END,
    DELTA,
};

struct OMNITRACE_ATTRIBUTE(packed) entry
{
    entry()                 = default;
    ~entry()                = default;
    entry(const entry&)     = default;
    entry(entry&&) noexcept = default;
    entry& operator=(const entry&) = default;
    entry& operator=(entry&&) noexcept = default;

    Device    device     = Device::CPU;  /// which device it executed on
    Phase     phase      = Phase::NONE;  /// start / stop / unspecified
    uint16_t  priority   = 0;            /// priority value (for sorting)
    uint32_t  depth      = 0;            /// call-stack depth
    int32_t   devid      = 0;            /// device id
    int32_t   pid        = 0;            /// process id
    int32_t   tid        = 0;            /// thread id it was registered on
    uint64_t  cpu_cid    = 0;            /// CPU correlation id
    uint64_t  gpu_cid    = 0;            /// GPU correlation id
    uint64_t  parent_cid = 0;            /// parent CPU correlation id
    int64_t   begin_ns   = 0;            /// timestamp of start
    int64_t   end_ns     = 0;            /// timestamp of end
    uintptr_t queue_id   = 0;            /// stream id (GPU) or mutex id
    size_t    hash       = 0;            /// hash for name

    bool operator==(const entry& rhs) const;
    bool operator!=(const entry& rhs) const { return !(*this == rhs); }
    bool operator<(const entry& rhs) const;
    bool operator>(const entry& rhs) const;
    bool operator<=(const entry& rhs) const { return !(*this > rhs); }
    bool operator>=(const entry& rhs) const { return !(*this < rhs); }

    entry& operator+=(const entry& rhs);

    size_t  get_hash() const;
    int64_t get_timestamp() const;

    int64_t get_cost() const;

    bool    is_bounded(const entry& rhs) const;
    int64_t get_overlap(const entry& rhs) const;
    int64_t get_independent(const entry& rhs) const;

    int64_t get_overlap(const entry& rhs, int32_t _devid, int32_t _pid,
                        int64_t _tid) const;
    int64_t get_independent(const entry& rhs, int32_t _devid, int32_t _pid,
                            int64_t _tid) const;
    bool is_bounded(const entry& rhs, int32_t _devid, int32_t _pid, int64_t _tid) const;

    void write(std::ostream& _os) const;

    static bool is_delta(const entry&, const std::string_view&);

    friend std::ostream& operator<<(std::ostream& _os, const entry& _v)
    {
        _v.write(_os);
        return _os;
    }
    template <typename Archive>
    void save(Archive& ar, unsigned int) const;

    template <typename Archive>
    void load(Archive& ar, unsigned int);
};

template <typename Archive>
void
entry::save(Archive& ar, unsigned int) const
{
    namespace cereal = tim::cereal;

#define SAVE_PACKED_ENTRY_FIELD(VAR)                                                     \
    {                                                                                    \
        auto _val = VAR;                                                                 \
        ar(cereal::make_nvp(#VAR, _val));                                                \
    }
    SAVE_PACKED_ENTRY_FIELD(priority);
    SAVE_PACKED_ENTRY_FIELD(device);
    SAVE_PACKED_ENTRY_FIELD(phase);
    SAVE_PACKED_ENTRY_FIELD(depth);
    SAVE_PACKED_ENTRY_FIELD(devid);
    SAVE_PACKED_ENTRY_FIELD(pid);
    SAVE_PACKED_ENTRY_FIELD(tid);
    SAVE_PACKED_ENTRY_FIELD(cpu_cid);
    SAVE_PACKED_ENTRY_FIELD(gpu_cid);
    SAVE_PACKED_ENTRY_FIELD(parent_cid);
    SAVE_PACKED_ENTRY_FIELD(begin_ns);
    SAVE_PACKED_ENTRY_FIELD(end_ns);
    SAVE_PACKED_ENTRY_FIELD(queue_id);
    SAVE_PACKED_ENTRY_FIELD(hash);
#undef SAVE_PACKED_ENTRY_FIELD

    std::string _name{};
    auto        _hash = hash;
    if(_hash > 0) _name = tim::get_hash_identifier(_hash);

    ar(cereal::make_nvp("name", _name),
       cereal::make_nvp("demangled_name", tim::demangle(_name)));
}

template <typename Archive>
void
entry::load(Archive& ar, unsigned int)
{
    namespace cereal = tim::cereal;

#define LOAD_PACKED_ENTRY_FIELD(VAR)                                                     \
    {                                                                                    \
        auto _val = VAR;                                                                 \
        ar(cereal::make_nvp(#VAR, _val));                                                \
        VAR = _val;                                                                      \
    }
    LOAD_PACKED_ENTRY_FIELD(priority);
    LOAD_PACKED_ENTRY_FIELD(device);
    LOAD_PACKED_ENTRY_FIELD(phase);
    LOAD_PACKED_ENTRY_FIELD(depth);
    LOAD_PACKED_ENTRY_FIELD(devid);
    LOAD_PACKED_ENTRY_FIELD(pid);
    LOAD_PACKED_ENTRY_FIELD(tid);
    LOAD_PACKED_ENTRY_FIELD(cpu_cid);
    LOAD_PACKED_ENTRY_FIELD(gpu_cid);
    LOAD_PACKED_ENTRY_FIELD(parent_cid);
    LOAD_PACKED_ENTRY_FIELD(begin_ns);
    LOAD_PACKED_ENTRY_FIELD(end_ns);
    LOAD_PACKED_ENTRY_FIELD(queue_id);
    LOAD_PACKED_ENTRY_FIELD(hash);
#undef LOAD_PACKED_ENTRY_FIELD

    std::string _name{};
    std::string _demangled_name{};
    ar(cereal::make_nvp("name", _name),
       cereal::make_nvp("demangled_name", _demangled_name));

    auto _hash = hash;
    tim::get_hash_ids()->emplace(_hash, _name);
}

struct call_chain : private std::vector<entry>
{
    using base_type = std::vector<entry>;

    using base_type::at;
    using base_type::back;
    using base_type::begin;
    using base_type::cbegin;
    using base_type::cend;
    using base_type::clear;
    using base_type::emplace_back;
    using base_type::empty;
    using base_type::end;
    using base_type::erase;
    using base_type::front;
    using base_type::pop_back;
    using base_type::push_back;
    using base_type::rbegin;
    using base_type::rend;
    using base_type::reserve;
    using base_type::size;

    size_t  get_hash() const;
    int64_t get_cost(int64_t _tid = -1) const;
    int64_t get_overlap(int32_t _devid, int32_t _pid, int64_t _tid = -1) const;
    int64_t get_independent(int32_t _devid, int32_t _pid, int64_t _tid = -1) const;
    static std::vector<call_chain>& get_top_chains();

    bool operator==(const call_chain& rhs) const;
    bool operator!=(const call_chain& rhs) const { return !(*this == rhs); }
    friend std::ostream& operator<<(std::ostream& _os, const call_chain& _v)
    {
        size_t _n = 0;
        for(const auto& itr : _v)
            _os << "    [" << _n++ << "] " << itr << "\n";
        return _os;
    }

    template <typename Archive>
    void serialize(Archive& ar, unsigned int)
    {
        namespace cereal = tim::cereal;
        ar(cereal::make_nvp("call_chain", static_cast<base_type&>(*this)));
    }

    template <Device DevT>
    void generate_perfetto(std::set<entry>& _used) const;

    template <bool BoolV = true, typename FuncT>
    bool query(FuncT&&) const;
};

template <bool BoolV, typename FuncT>
bool
call_chain::query(FuncT&& _func) const
{
    for(const auto& itr : *this)
    {
        if(std::forward<FuncT>(_func)(itr)) return BoolV;
    }
    return !BoolV;
}

using hash_ids = std::unordered_set<std::string>;

uint64_t
get_update_frequency();

unique_ptr_t<call_chain>&
get(int64_t _tid = threading::get_id());

size_t
add_hash_id(const std::string& _label);

void
add_hash_id(const hash_ids&);

void
update(int64_t _tid = threading::get_id());

void
compute(int64_t _tid = threading::get_id());

std::vector<std::pair<std::string, entry>>
get_entries(
    int64_t                                  _ts,
    const std::function<bool(const entry&)>& _eval = [](const entry&) { return true; });

struct id
{};

}  // namespace critical_trace
}  // namespace omnitrace
