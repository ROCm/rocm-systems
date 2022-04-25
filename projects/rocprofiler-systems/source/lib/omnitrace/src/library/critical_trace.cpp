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

#include "library/critical_trace.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/perfetto.hpp"
#include "library/ptl.hpp"
#include "library/thread_data.hpp"

#include <PTL/ThreadPool.hh>
#include <timemory/backends/dmp.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/tpls/cereal/cereal/archives/json.hpp>
#include <timemory/tpls/cereal/cereal/cereal.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/utility/utility.hpp>

#include <cctype>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace omnitrace
{
namespace critical_trace
{
namespace
{
using call_graph_t              = tim::graph<entry>;
using call_graph_itr_t          = typename call_graph_t::iterator;
using call_graph_sibling_itr_t  = typename call_graph_t::sibling_iterator;
using call_graph_preorder_itr_t = typename call_graph_t::pre_order_iterator;

hash_ids   complete_hash_ids{};
call_chain complete_call_chain{};
std::mutex complete_call_mutex{};

void
update_critical_path(call_chain _chain, int64_t _tid);

void
compute_critical_trace();

void
copy_hash_ids()
{
    // make copy to avoid parallel iteration issues
    auto _hash_ids = complete_hash_ids;
    // ensure all hash ids exist
    for(const auto& itr : _hash_ids)
        tim::hash::add_hash_id(itr);
}
}  // namespace
}  // namespace critical_trace

namespace critical_trace
{
namespace
{
template <typename Arg0, typename Arg1, typename... Args>
size_t
get_combined_hash(Arg0&& _zero, Arg1&& _one, Args&&... _args)
{
    size_t _hash = tim::hash::get_combined_hash_id(std::forward<Arg0>(_zero),
                                                   std::forward<Arg1>(_one));
    if constexpr(sizeof...(_args) == 0)
    {
        return _hash;
    }
    else
    {
        return get_combined_hash(_hash, std::forward<Args>(_args)...);
    }
}
}  // namespace

//--------------------------------------------------------------------------------------//
//
//                          ENTRY
//
//--------------------------------------------------------------------------------------//

bool
entry::operator==(const entry& rhs) const
{
    return (device == rhs.device && depth == rhs.depth && priority == rhs.priority &&
            tid == rhs.tid && cpu_cid == rhs.cpu_cid && gpu_cid == rhs.gpu_cid &&
            hash == rhs.hash);
}

bool
entry::operator<(const entry& rhs) const
{
    // sort by cpu ids
    auto _cpu_eq = (cpu_cid == rhs.cpu_cid);
    if(!_cpu_eq) return (cpu_cid < rhs.cpu_cid);

    // sort by gpu ids
    if(gpu_cid > 0 && rhs.gpu_cid > 0)
    {
        auto _gpu_eq = (gpu_cid == rhs.gpu_cid);
        if(!_gpu_eq) return (gpu_cid < rhs.gpu_cid);
    }

    // sort by parent ids
    auto _par_eq = (parent_cid == rhs.parent_cid);
    if(!_par_eq) return (parent_cid < rhs.parent_cid);

    // sort by priority
    auto _prio_eq = (priority == rhs.priority);
    if(!_prio_eq) return (priority < rhs.priority);

    // sort by timestamp (last resort)
    return (begin_ns < rhs.begin_ns);
}

bool
entry::operator>(const entry& rhs) const
{
    return (!(*this < rhs) && begin_ns != rhs.begin_ns && cpu_cid != rhs.cpu_cid &&
            gpu_cid != rhs.gpu_cid);
}

entry&
entry::operator+=(const entry& rhs)
{
    if(phase == Phase::BEGIN && rhs.phase == Phase::END)
    {
        assert(rhs.end_ns >= begin_ns);
        end_ns = rhs.end_ns;
        phase  = Phase::DELTA;
        return *this;
    }
    else
    {
        OMNITRACE_CONDITIONAL_PRINT(
            get_verbose() > 1,
            "Warning! Incorrect phase. entry::operator+=(entry) is only valid for "
            "Phase::BEGIN += Phase::END\n");
    }
    return *this;
}

size_t
entry::get_hash() const
{
    return get_combined_hash(hash, static_cast<short>(device), static_cast<short>(phase),
                             tid, cpu_cid, gpu_cid, priority);
}

int64_t
entry::get_timestamp() const
{
    switch(phase)
    {
        case Phase::BEGIN: return begin_ns;
        case Phase::END: return end_ns;
        case Phase::DELTA: return (end_ns - begin_ns);
        case Phase::NONE: break;
    }
    return 0;
}

int64_t
entry::get_cost() const
{
    switch(phase)
    {
        case Phase::DELTA: return (end_ns - begin_ns);
        default: break;
    }
    return 0;
}

int64_t
entry::get_overlap(const entry& rhs) const
{
    if(begin_ns >= rhs.end_ns || end_ns >= rhs.begin_ns)  // no overlap
        return 0;
    else if(begin_ns >= rhs.begin_ns && end_ns <= rhs.end_ns)  // inclusive to rhs
        return get_cost();
    else if(begin_ns <= rhs.begin_ns && end_ns >= rhs.end_ns)  // rhs is inclusive
        return rhs.get_cost();
    else if(begin_ns <= rhs.begin_ns && end_ns <= rhs.end_ns)  // at beginning
        return (end_ns - rhs.begin_ns);
    else if(begin_ns >= rhs.begin_ns && end_ns >= rhs.end_ns)  // at end
        return (rhs.end_ns - begin_ns);
    else
    {
        OMNITRACE_PRINT("Warning! entry::get_overlap(entry, tid) "
                        "could not determine the overlap :: %s\n",
                        JOIN("", *this).c_str());
    }
    return 0;
}

int64_t
entry::get_overlap(const entry& rhs, int64_t _tid) const
{
    if(!is_delta(*this, __FUNCTION__)) return 0;
    if(!is_delta(rhs, __FUNCTION__)) return 0;

    if(_tid < 0 || (this->tid == _tid && rhs.tid == _tid))  // all threads or same thread
        return get_overlap(rhs);

    return 0;
}

int64_t
entry::get_independent(const entry& rhs) const
{
    if(begin_ns >= rhs.end_ns || end_ns >= rhs.begin_ns)  // no overlap
        return get_cost();
    else if(begin_ns >= rhs.begin_ns && end_ns <= rhs.end_ns)  // inclusive to rhs
        return 0;
    else if(begin_ns <= rhs.begin_ns && end_ns >= rhs.end_ns)  // rhs is inclusive
        return get_cost() - rhs.get_cost();
    else if(begin_ns <= rhs.begin_ns && end_ns <= rhs.end_ns)  // at beginning
        return (rhs.begin_ns - begin_ns);
    else if(begin_ns >= rhs.begin_ns && end_ns >= rhs.end_ns)  // at end
        return (end_ns - rhs.end_ns);
    else
    {
        OMNITRACE_PRINT("Warning! entry::get_independent(entry, tid) "
                        "could not determine the overlap :: %s\n",
                        JOIN("", *this).c_str());
    }
    return 0;
}

int64_t
entry::get_independent(const entry& rhs, int64_t _tid) const
{
    if(!is_delta(*this, __FUNCTION__)) return 0;
    if(!is_delta(rhs, __FUNCTION__)) return 0;

    if(_tid < 0 || (this->tid == _tid && rhs.tid == _tid))  // all threads or same thread
        return get_independent(rhs);
    else if(this->tid == _tid && rhs.tid != _tid)  // rhs is on different thread
        return get_cost();
    return 0;
}

bool
entry::is_bounded(const entry& rhs) const
{
    // ignores thread
    return !(begin_ns < rhs.begin_ns || end_ns > rhs.end_ns);
}

bool
entry::is_bounded(const entry& rhs, int64_t _tid) const
{
    if(this->tid == _tid && rhs.tid == _tid)  // all threads or same thread
        return !(begin_ns < rhs.begin_ns || end_ns > rhs.end_ns);

    return false;
}

void
entry::write(std::ostream& _os) const
{
    if(device == Device::GPU)
        _os << "[GPU][" << cpu_cid << "][" << gpu_cid << "]";
    else
        _os << "[CPU][" << cpu_cid << "]";
    _os << " parent: " << static_cast<int64_t>(parent_cid);
    _os << ", tid: " << tid;
    _os << ", depth: " << depth;
    _os << ", priority: " << priority;
    if(phase == Phase::DELTA)
    {
        std::stringstream _cost{};
        _cost << std::setprecision(4) << std::scientific << (get_timestamp() / 1.0e9);
        _os << ", cost: [" << std::setw(8) << _cost.str() << " sec]";
    }
    else
    {
        _os << ", phase: ";
        if(phase == Phase::BEGIN)
            _os << "begin ";
        else if(phase == Phase::END)
            _os << "end ";
        _os << "[" << begin_ns << ":" << end_ns << "]";
    }
    _os << ", hash: " << hash << " :: " << tim::demangle(tim::get_hash_identifier(hash));
}

bool
entry::is_delta(const entry& _v, const std::string_view& _ctx)
{
    if(_v.phase != Phase::DELTA)
    {
        OMNITRACE_CT_DEBUG(
            "Warning! Invalid phase for entry. entry::%s requires Phase::DELTA :: %s\n",
            _ctx.data(), JOIN("", _v).c_str());
        return true;
    }
    return false;
}

//--------------------------------------------------------------------------------------//
//
//                          CALL CHAIN
//
//--------------------------------------------------------------------------------------//

bool
call_chain::operator==(const call_chain& rhs) const
{
    if(size() != rhs.size()) return false;
    for(size_t i = 0; i < size(); ++i)
        if(at(i) != rhs.at(i)) return false;
    return true;
}

size_t
call_chain::get_hash() const
{
    if(empty()) return 0;
    int64_t _hash = this->at(0).get_hash();
    for(size_t i = 1; i < this->size(); ++i)
        _hash = get_combined_hash(_hash, at(i).get_hash());
    return _hash;
}

int64_t
call_chain::get_cost(int64_t _tid) const
{
    int64_t _cost = 0;
    if(_tid < 0)
    {
        for(const auto& itr : *this)
            _cost += itr.get_cost();
    }
    else
    {
        for(const auto& itr : *this)
        {
            if(itr.tid == _tid) _cost += itr.get_cost();
        }
    }
    return _cost;
}

int64_t
call_chain::get_overlap(int64_t _tid) const
{
    int64_t _cost = 0;
    auto    itr   = this->begin();
    auto    nitr  = ++this->begin();
    for(; nitr != this->end(); ++nitr, ++itr)
        _cost += nitr->get_overlap(*itr, _tid);
    return _cost;
}

int64_t
call_chain::get_independent(int64_t _tid) const
{
    int64_t _cost = 0;
    auto    itr   = this->begin();
    auto    nitr  = ++this->begin();
    for(; nitr != this->end(); ++nitr, ++itr)
        _cost += itr->get_independent(*nitr, _tid);
    return _cost;
}

std::vector<call_chain>&
call_chain::get_top_chains()
{
    static std::vector<call_chain> _v{};
    return _v;
}

template <>
void
call_chain::generate_perfetto<Device::NONE>(std::set<entry>& _used) const
{
    for(const auto& itr : *this)
    {
        if(!_used.emplace(itr).second) continue;
        if(itr.device == Device::CPU)
        {
            TRACE_EVENT_BEGIN("device-critical-trace", "CPU",
                              static_cast<uint64_t>(itr.begin_ns));
        }
        else if(itr.device == Device::GPU)
        {
            TRACE_EVENT_BEGIN("device-critical-trace", "GPU",
                              static_cast<uint64_t>(itr.begin_ns));
        }
        TRACE_EVENT_END("device-critical-trace", static_cast<uint64_t>(itr.end_ns));
    }
}

template <>
void
call_chain::generate_perfetto<Device::CPU>(std::set<entry>& _used) const
{
    static std::set<std::string> _static_strings{};
    static std::mutex            _static_mutex{};
    for(const auto& itr : *this)
    {
        if(!_used.emplace(itr).second) continue;
        if(itr.device != Device::CPU) continue;
        std::string _name = tim::demangle(tim::get_hash_identifier(itr.hash));
        _static_mutex.lock();
        auto sitr = _static_strings.emplace(_name);
        _static_mutex.unlock();
        TRACE_EVENT_BEGIN("host-critical-trace",
                          perfetto::StaticString{ sitr.first->c_str() },
                          static_cast<uint64_t>(itr.begin_ns));
        TRACE_EVENT_END("host-critical-trace", static_cast<uint64_t>(itr.end_ns));
    }
}

template <>
void
call_chain::generate_perfetto<Device::GPU>(std::set<entry>& _used) const
{
    static std::set<std::string> _static_strings{};
    static std::mutex            _static_mutex{};
    for(const auto& itr : *this)
    {
        if(!_used.emplace(itr).second) continue;
        if(itr.device != Device::GPU) continue;
        std::string _name = tim::demangle(tim::get_hash_identifier(itr.hash));
        _static_mutex.lock();
        auto sitr = _static_strings.emplace(_name);
        _static_mutex.unlock();
        TRACE_EVENT_BEGIN("device-critical-trace",
                          perfetto::StaticString{ sitr.first->c_str() },
                          static_cast<uint64_t>(itr.begin_ns));
        TRACE_EVENT_END("device-critical-trace", static_cast<uint64_t>(itr.end_ns));
    }
}

template <>
void
call_chain::generate_perfetto<Device::ANY>(std::set<entry>& _used) const
{
    static std::set<std::string> _static_strings{};
    static std::mutex            _static_mutex{};
    for(const auto& itr : *this)
    {
        if(!_used.emplace(itr).second) continue;
        std::string _name = tim::demangle(tim::get_hash_identifier(itr.hash));
        _static_mutex.lock();
        auto sitr = _static_strings.emplace(_name);
        _static_mutex.unlock();
        TRACE_EVENT_BEGIN("critical-trace", perfetto::StaticString{ sitr.first->c_str() },
                          static_cast<uint64_t>(itr.begin_ns));
        TRACE_EVENT_END("critical-trace", static_cast<uint64_t>(itr.end_ns));
    }
}

//--------------------------------------------------------------------------------------//
//
//                          FREE FUNCTIONS
//
//--------------------------------------------------------------------------------------//

uint64_t
get_update_frequency()
{
    return get_critical_trace_update_freq();
}

unique_ptr_t<call_chain>&
get(int64_t _tid)
{
    static auto&             _v    = thread_data<call_chain>::instances();
    static thread_local auto _once = [_tid]() {
        if(!_v.at(0)) _v.at(0) = unique_ptr_t<call_chain>{ new call_chain{} };
        if(!_v.at(_tid)) _v.at(_tid) = unique_ptr_t<call_chain>{ new call_chain{} };
        if(_tid > 0) *_v.at(_tid) = *_v.at(0);
        return true;
    }();
    (void) _once;
    return _v.at(_tid);
}

void
add_hash_id(const hash_ids& _labels)
{
    std::unique_lock<std::mutex> _lk{ tasking::get_critical_trace_mutex() };
    if(!tasking::get_critical_trace_task_group().pool()) return;
    tasking::get_critical_trace_task_group().exec([_labels]() {
        static std::mutex _mtx{};
        _mtx.lock();
        for(auto itr : _labels)
            complete_hash_ids.emplace(std::move(itr));
        _mtx.unlock();
    });
}

size_t
add_hash_id(const std::string& _label)
{
    using critical_trace_hash_data =
        thread_data<critical_trace::hash_ids, critical_trace::id>;

    auto _hash = tim::hash::add_hash_id(_label);
    if(get_use_critical_trace() || get_use_rocm_smi())
    {
        critical_trace_hash_data::construct();
        critical_trace_hash_data::instance()->emplace(_label);
    }
    return _hash;
}

void
update(int64_t _tid)
{
    if(!get_use_critical_trace() && !get_use_rocm_smi()) return;
    std::unique_lock<std::mutex> _lk{ tasking::get_critical_trace_mutex() };
    if(!tasking::get_critical_trace_task_group().pool()) return;
    call_chain _data{};
    std::swap(_data, *critical_trace::get(_tid));
    tasking::get_critical_trace_task_group().exec(update_critical_path, _data, _tid);
}

void
compute(int64_t _tid)
{
    update(_tid);
    std::unique_lock<std::mutex> _lk{ tasking::get_critical_trace_mutex() };
    if(!tasking::get_critical_trace_task_group().pool()) return;
    tasking::get_critical_trace_task_group().exec(compute_critical_trace);
}

//--------------------------------------------------------------------------------------//
//
//                          HELPER FUNCTIONS
//
//--------------------------------------------------------------------------------------//

namespace
{
std::string
get_perf_name(std::string _func)
{
    const auto _npos = std::string::npos;
    auto       _pos  = std::string::npos;
    while((_pos = _func.find('_')) != _npos)
        _func = _func.replace(_pos, 1, " ");
    if(_func.length() > 0) _func.at(0) = std::toupper(_func.at(0));
    return _func;
}

void
save_call_chain_json(const std::string& _fname, const std::string& _label,
                     const call_chain& _call_chain, bool _msg = false,
                     std::string _func = {})
{
    OMNITRACE_CT_DEBUG("[%s]\n", __FUNCTION__);

    using perfstats_t =
        tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::page_rss>;
    perfstats_t _perf{ get_perf_name(__FUNCTION__) };
    _perf.start();

    auto _save = [&](std::ostream& _os) {
        namespace cereal = tim::cereal;
        auto ar = tim::policy::output_archive<cereal::MinimalJSONOutputArchive>::get(_os);

        auto _hash_map = *tim::hash::get_hash_ids();
        for(auto& itr : _hash_map)
            itr.second = tim::demangle(itr.second);
        ar->setNextName("omnitrace");
        ar->startNode();
        (*ar)(cereal::make_nvp("hash_map", _hash_map),
              cereal::make_nvp(_label.c_str(), _call_chain));
        ar->finishNode();
    };

    std::ofstream ofs{};
    if(tim::filepath::open(ofs, _fname))
    {
        if(_msg)
        {
            if(_func.empty()) _func = __FUNCTION__;
            OMNITRACE_CONDITIONAL_PRINT(get_verbose() >= 0, "[%s] Outputting '%s'...\n",
                                        _func.c_str(), _fname.c_str());
        }
        std::stringstream oss{};
        if(_call_chain.size() > 100000)
        {
            _save(ofs);
        }
        else
        {
            _save(oss);
            ofs << oss.str() << std::endl;
        }
    }

    _perf.stop();
    if(_msg)
    {
        OMNITRACE_CT_DEBUG("%s\n", JOIN("", _perf).c_str());
    }
}

template <typename Tp, template <typename...> class ContainerT, typename... Args,
          typename FuncT = bool (*)(const Tp&, const Tp&)>
inline Tp*
find(
    const Tp& _v, ContainerT<Tp, Args...>& _vec,
    FuncT&& _func = [](const Tp& _lhs, const Tp& _rhs) { return (_lhs == _rhs); })
{
    for(auto& itr : _vec)
    {
        if(std::forward<FuncT>(_func)(_v, itr)) return &itr;
    }
    return nullptr;
};

template <typename FuncT = bool (*)(const entry&, const entry&)>
inline entry*
find(
    const entry& _v, call_chain& _vec,
    FuncT&& _func = [](const entry& _lhs, const entry& _rhs) { return (_lhs == _rhs); })
{
    return find(_v, reinterpret_cast<std::vector<entry>&>(_vec),
                std::forward<FuncT>(_func));
}

void
squash_critical_path(call_chain& _targ)
{
    OMNITRACE_CT_DEBUG("[%s]\n", __FUNCTION__);
    static auto _strict_equal = [](const entry& _lhs, const entry& _rhs) {
        auto _same_phase  = (_lhs.phase == _rhs.phase);
        bool _phase_check = true;
        if(_same_phase) _phase_check = (_lhs.get_timestamp() == _rhs.get_timestamp());
        return (_lhs == _rhs && _lhs.parent_cid == _rhs.parent_cid && _phase_check);
    };

    std::sort(_targ.begin(), _targ.end());

    call_chain _squashed{};
    for(auto& itr : _targ)
    {
        if(itr.phase == Phase::DELTA)
        {
            _squashed.emplace_back(itr);
        }
        else if(itr.phase == Phase::BEGIN)
        {
            if(!find(itr, _squashed, _strict_equal)) _squashed.emplace_back(itr);
        }
        else
        {
            entry* _match = nullptr;
            if((_match = find(itr, _squashed)) != nullptr)
                *_match += itr;
            else
                _squashed.emplace_back(itr);
        }
    }

    std::swap(_targ, _squashed);
    std::sort(_targ.begin(), _targ.end());
}

void
combine_critical_path(call_chain& _targ, call_chain _chain)
{
    OMNITRACE_CT_DEBUG("[%s]\n", __FUNCTION__);
    call_chain _delta{};
    call_chain _begin{};
    call_chain _end{};
    for(auto& itr : _chain)
    {
        if(itr.phase == Phase::DELTA)
            _delta.emplace_back(itr);
        else if(itr.phase == Phase::BEGIN)
            _begin.emplace_back(itr);
        else if(itr.phase == Phase::END)
        {
            entry* _match = nullptr;
            if((_match = find(itr, _begin)) != nullptr)
                *_match += itr;
            else
                _end.emplace_back(itr);
        }
    }

    call_chain _combined{};
    _combined.reserve(_delta.size() + _begin.size() + _end.size());
    for(auto& itr : _delta)
        _combined.emplace_back(itr);
    for(auto& itr : _begin)
        _combined.emplace_back(itr);
    for(auto& itr : _end)
        _combined.emplace_back(itr);
    std::sort(_combined.begin(), _combined.end());

    std::unique_lock<std::mutex> _lk{ complete_call_mutex };
    for(auto& itr : _combined)
        _targ.emplace_back(itr);

    // squash_critical_path(_targ);
}

void
update_critical_path(call_chain _chain, int64_t)
{
    OMNITRACE_CT_DEBUG("[%s]\n", __FUNCTION__);
    try
    {
        // remove any data not
        // auto _diff_tid = [_tid](const entry& _v) { return _v.tid != _tid; };
        //_chain.erase(std::remove_if(_chain.begin(), _chain.end(), _diff_tid),
        //             _chain.end());
        combine_critical_path(complete_call_chain, std::move(_chain));
    } catch(const std::exception& e)
    {
        std::cerr << "Thread exited with exception: " << e.what() << std::endl;
        TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(true, 32);
    }
}

void
compute_critical_trace()
{
    static bool                  _computed = false;
    std::unique_lock<std::mutex> _lk{ complete_call_mutex };

    if(_computed) return;

    OMNITRACE_CONDITIONAL_PRINT(get_critical_trace_debug() || get_verbose() >= 0,
                                "[%s] Generating critical trace...\n", __FUNCTION__);

    // ensure all hash ids exist
    copy_hash_ids();

    using perfstats_t =
        tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::page_rss>;

    perfstats_t _ct_perf{ JOIN("", "[", __FUNCTION__, "]") };
    _ct_perf.start();

    try
    {
        OMNITRACE_CT_DEBUG("[%s] initial call chain: %zu entries\n", __FUNCTION__,
                           complete_call_chain.size());

        perfstats_t _perf{ get_perf_name(__FUNCTION__) };
        _perf.start();

        std::sort(complete_call_chain.begin(), complete_call_chain.end());

        _perf.stop().rekey("Sorting critical trace");
        OMNITRACE_CT_DEBUG("%s\n", JOIN("", _perf).c_str());

        _perf.reset().start();
        save_call_chain_json(
            tim::settings::compose_output_filename("call-chain", ".json"), "call_chain",
            complete_call_chain, true, __FUNCTION__);

        _perf.stop().rekey("Save call-chain");
        OMNITRACE_CT_DEBUG("%s\n", JOIN("", _perf).c_str());

    } catch(std::exception& e)
    {
        OMNITRACE_PRINT("Thread exited '%s' with exception: %s\n", __FUNCTION__,
                        e.what());
        TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(true, 32);
    }

    _ct_perf.stop();
    auto _ct_msg = JOIN("", _ct_perf);
    auto _ct_pos = _ct_msg.find(">>>  ");
    if(_ct_pos != std::string::npos) _ct_msg = _ct_msg.substr(_ct_pos + 5);
    OMNITRACE_PRINT("%s\n", _ct_msg.c_str());
}
}  // namespace

std::vector<std::pair<std::string, entry>>
get_entries(int64_t _ts, const std::function<bool(const entry&)>& _eval)
{
    copy_hash_ids();
    auto _func = [_eval, _ts](std::vector<std::pair<std::string, entry>>* _targ,
                              size_t*                                     _avail) {
        copy_hash_ids();
        squash_critical_path(complete_call_chain);
        *_avail = complete_call_chain.size();
        std::vector<std::pair<std::string, entry>> _v{};
        std::sort(complete_call_chain.begin(), complete_call_chain.end());
        for(const auto& itr : complete_call_chain)
        {
            if(itr.phase != Phase::DELTA) continue;
            if(itr.begin_ns <= _ts && itr.end_ns >= _ts)
            {
                if(_eval(itr)) _v.emplace_back(tim::get_hash_identifier(itr.hash), itr);
            }
        }
        *_targ = _v;
    };
    std::unique_lock<std::mutex>               _lk{ tasking::get_critical_trace_mutex() };
    size_t                                     _n = 0;
    std::vector<std::pair<std::string, entry>> _v{};
    if(!tasking::get_critical_trace_task_group().pool()) return _v;
    tasking::get_critical_trace_task_group().exec(_func, &_v, &_n);
    tasking::get_critical_trace_task_group().join();
    OMNITRACE_DEBUG("critical_trace::%s :: found %zu out of %zu entries at %li...\n",
                    __FUNCTION__, _v.size(), _n, _ts);
    return _v;
}

}  // namespace critical_trace
}  // namespace omnitrace
