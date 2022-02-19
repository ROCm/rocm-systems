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

#include "critical-trace.hpp"

#include "library/api.hpp"
#include "library/perfetto.hpp"

#include <timemory/hash/types.hpp>

#include <cstdlib>

namespace config         = omnitrace::config;
namespace critical_trace = omnitrace::critical_trace;

int
main(int argc, char** argv)
{
    omnitrace_init_library();

    config::set_setting_value("OMNITRACE_CRITICAL_TRACE", true);
    // config::set_setting_value("OMNITRACE_CRITICAL_TRACE_DEBUG", true);
    config::set_setting_value<int64_t>("OMNITRACE_CRITICAL_TRACE_COUNT", 500);
    config::set_setting_value<int64_t>("OMNITRACE_CRITICAL_TRACE_PER_ROW", 100);
    config::set_setting_value<int64_t>("OMNITRACE_CRITICAL_TRACE_NUM_THREADS",
                                       std::thread::hardware_concurrency());
    config::set_setting_value("OMNITRACE_CRITICAL_TRACE_SERIALIZE_NAMES", true);

    perfetto::TracingInitArgs               args{};
    perfetto::TraceConfig                   cfg{};
    perfetto::protos::gen::TrackEventConfig track_event_cfg{};

    auto  shmem_size_hint = config::get_perfetto_shmem_size_hint();
    auto  buffer_size     = config::get_perfetto_buffer_size();
    auto* buffer_config   = cfg.add_buffers();

    buffer_config->set_size_kb(buffer_size);
    buffer_config->set_fill_policy(
        perfetto::protos::gen::TraceConfig_BufferConfig_FillPolicy_DISCARD);

    auto* ds_cfg = cfg.add_data_sources()->mutable_config();
    ds_cfg->set_name("track_event");
    ds_cfg->set_track_event_config_raw(track_event_cfg.SerializeAsString());

    args.backends |= perfetto::kInProcessBackend;
    args.shmem_size_hint_kb = shmem_size_hint;

    perfetto::Tracing::Initialize(args);
    perfetto::TrackEvent::Register();

    auto tracing_session = perfetto::Tracing::NewTrace();
    tracing_session->Setup(cfg);
    tracing_session->StartBlocking();

    for(int i = 1; i < argc; ++i)
    {
        critical_trace::complete_call_chain = {};
        OMNITRACE_BASIC_PRINT_F("Loading call-chain %s...\n", argv[i]);
        critical_trace::load_call_chain(argv[i], "call_chain",
                                        critical_trace::complete_call_chain);
        for(const auto& itr : *tim::get_hash_ids())
            critical_trace::complete_hash_ids.emplace(itr.second);
        OMNITRACE_BASIC_PRINT_F("Computing critical trace for %s...\n", argv[i]);
        critical_trace::compute_critical_trace();
    }

    // Make sure the last event is closed for this example.
    perfetto::TrackEvent::Flush();

    OMNITRACE_DEBUG_F("Stopping the blocking perfetto trace sessions...\n");
    tracing_session->StopBlocking();

    OMNITRACE_DEBUG_F("Getting the trace data...\n");
    std::vector<char> trace_data{ tracing_session->ReadTraceBlocking() };

    if(trace_data.empty())
    {
        OMNITRACE_BASIC_PRINT_F(
            "> trace data is empty. File '%s' will not be written...\n",
            config::get_perfetto_output_filename().c_str());
    }
    else
    {
        // Write the trace into a file.
        OMNITRACE_CONDITIONAL_BASIC_PRINT_F(
            config::get_verbose() >= 0,
            "> Outputting '%s' (%.2f KB / %.2f MB / %.2f GB)... ",
            config::get_perfetto_output_filename().c_str(),
            static_cast<double>(trace_data.size()) / tim::units::KB,
            static_cast<double>(trace_data.size()) / tim::units::MB,
            static_cast<double>(trace_data.size()) / tim::units::GB);

        std::ofstream ofs{};
        if(!tim::filepath::open(ofs, config::get_perfetto_output_filename(),
                                std::ios::out | std::ios::binary))
        {
            OMNITRACE_BASIC_PRINT_F("> Error opening '%s'...\n",
                                    config::get_perfetto_output_filename().c_str());
            return EXIT_FAILURE;
        }
        else
        {
            // Write the trace into a file.
            if(config::get_verbose() >= 0) fprintf(stderr, "Done\n");
            ofs.write(&trace_data[0], trace_data.size());
        }
        ofs.close();
    }
}

namespace omnitrace
{
namespace critical_trace
{
namespace
{
//--------------------------------------------------------------------------------------//

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

//--------------------------------------------------------------------------------------//

void
save_call_graph(const std::string& _fname, const std::string& _label,
                const call_graph_t& _call_graph, bool _msg = false,
                std::string _func = {})
{
    OMNITRACE_CT_DEBUG("\n");

    using perfstats_t =
        tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::page_rss>;
    perfstats_t _perf{ get_perf_name(__FUNCTION__) };
    _perf.start();

    std::stringstream oss{};
    {
        namespace cereal = tim::cereal;
        auto ar = tim::policy::output_archive<cereal::MinimalJSONOutputArchive>::get(oss);

        auto _hash_map = *tim::hash::get_hash_ids();
        for(auto& itr : _hash_map)
            itr.second = tim::demangle(itr.second);
        ar->setNextName("omnitrace");
        ar->startNode();
        (*ar)(cereal::make_nvp("hash_map", _hash_map));
        ar->setNextName(_label.c_str());
        ar->startNode();
        serialize_graph(*ar, _call_graph);
        ar->finishNode();
        ar->finishNode();
    }

    std::ofstream ofs{};
    if(tim::filepath::open(ofs, _fname))
    {
        if(_msg)
        {
            if(_func.empty()) _func = __FUNCTION__;
            OMNITRACE_CONDITIONAL_BASIC_PRINT(get_verbose() >= 0,
                                              "[%s] Outputting '%s'...\n", _func.c_str(),
                                              _fname.c_str());
        }
        ofs << oss.str() << std::endl;
    }

    _perf.stop();
    if(_msg)
    {
        OMNITRACE_CT_DEBUG("%s\n", JOIN("", _perf).substr(4).c_str());
    }
}

void
save_critical_trace(const std::string& _fname, const std::string& _label,
                    const std::vector<call_chain>& _cchain, bool _msg = false,
                    std::string _func = {})
{
    OMNITRACE_CT_DEBUG("\n");

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
              cereal::make_nvp(_label.c_str(), _cchain));
        ar->finishNode();
    };

    std::ofstream ofs{};
    if(tim::filepath::open(ofs, _fname))
    {
        if(_msg)
        {
            if(_func.empty()) _func = __FUNCTION__;
            OMNITRACE_CONDITIONAL_BASIC_PRINT(get_verbose() >= 0,
                                              "[%s] Outputting '%s'...\n", _func.c_str(),
                                              _fname.c_str());
        }
        std::stringstream oss{};
        if(_cchain.size() > 1000)
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
        OMNITRACE_CT_DEBUG("%s\n", JOIN("", _perf).substr(4).c_str());
    }
}

void
save_call_chain_text(const std::string& _fname, const call_chain& _call_chain,
                     bool _msg = false, std::string _func = {})
{
    OMNITRACE_CT_DEBUG("\n");

    using perfstats_t =
        tim::lightweight_tuple<comp::wall_clock, comp::peak_rss, comp::page_rss>;
    perfstats_t _perf{ get_perf_name(__FUNCTION__) };
    _perf.start();

    std::ofstream ofs{};
    if(tim::filepath::open(ofs, _fname))
    {
        if(_msg)
        {
            if(_func.empty()) _func = __FUNCTION__;
            OMNITRACE_CONDITIONAL_BASIC_PRINT(get_verbose() >= 0,
                                              "[%s] Outputting '%s'...\n", _func.c_str(),
                                              _fname.c_str());
        }
        ofs << _call_chain << "\n";
    }

    _perf.stop();
    if(_msg)
    {
        OMNITRACE_CT_DEBUG("%s\n", JOIN("", _perf).substr(4).c_str());
    }
}

void
save_call_chain_json(const std::string& _fname, const std::string& _label,
                     const call_chain& _call_chain, bool _msg = false,
                     std::string _func = {})
{
    OMNITRACE_CT_DEBUG("\n");

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
            OMNITRACE_CONDITIONAL_BASIC_PRINT(get_verbose() >= 0,
                                              "[%s] Outputting '%s'...\n", _func.c_str(),
                                              _fname.c_str());
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
        OMNITRACE_CT_DEBUG("%s\n", JOIN("", _perf).substr(4).c_str());
    }
}

void
load_call_chain(const std::string& _fname, const std::string& _label,
                call_chain& _call_chain)
{
    std::ifstream ifs{};
    ifs.open(_fname);
    if(ifs && ifs.is_open())
    {
        namespace cereal = tim::cereal;
        auto ar          = tim::policy::input_archive<cereal::JSONInputArchive>::get(ifs);

        ar->setNextName("omnitrace");
        ar->startNode();
        (*ar)(cereal::make_nvp(_label.c_str(), _call_chain));
        ar->finishNode();
    }
}

auto
get_indexed(const call_chain& _chain)
{
    OMNITRACE_CT_DEBUG("\n");
    std::map<int64_t, std::vector<entry>> _indexed = {};

    // allocate for all cpu correlation ids
    for(const auto& itr : _chain)
    {
        _indexed.emplace(static_cast<int64_t>(itr.cpu_cid), std::vector<entry>{});
        _indexed.emplace(static_cast<int64_t>(itr.parent_cid), std::vector<entry>{});
    }

    // index based on parent correlation id
    for(const auto& itr : _chain)
    {
        if(itr.depth < 1 && itr.phase == Phase::BEGIN) continue;
        _indexed[static_cast<int64_t>(itr.parent_cid)].emplace_back(itr);
    }

    for(auto& itr : _indexed)
        std::sort(itr.second.begin(), itr.second.end(),
                  [](const entry& lhs, const entry& rhs) {
                      // return lhs.cpu_cid < rhs.cpu_cid;
                      return lhs.begin_ns < rhs.begin_ns;
                  });

    return _indexed;
}

void
find_children(PTL::ThreadPool& _tp, call_graph_t& _graph, const call_chain& _chain)
{
    OMNITRACE_CT_DEBUG("\n");

    using iterator_t      = call_graph_sibling_itr_t;
    using itr_entry_vec_t = std::vector<std::pair<iterator_t, entry>>;
    using task_group_t    = PTL::TaskGroup<void>;

    auto                                _indexed = get_indexed(_chain);
    std::map<entry, std::vector<entry>> _entry_map{};

    // allocate all entries
    OMNITRACE_CT_DEBUG_F("Allocating...\n");
    for(const auto& itr : _chain)
    {
        auto _ins = _entry_map.emplace(itr, std::vector<entry>{});
        if(!_ins.second)
        {
            auto _existing = _ins.first->first;
            OMNITRACE_BASIC_PRINT("Warning! Duplicate entry for [%s] :: [%s]\n",
                                  JOIN("", _existing).c_str(), JOIN("", itr).c_str());
        }
    }

    task_group_t _tg{ &_tp };
    OMNITRACE_CT_DEBUG_F("Parallel mapping...\n");
    for(const auto& itr : _chain)
    {
        _tg.run([&]() { _entry_map[itr] = _indexed.at(itr.cpu_cid); });
    }
    _tg.join();

    std::function<void(iterator_t, const entry&)> _recursive_func;
    _recursive_func = [&](iterator_t itr, const entry& _v) {
        auto _child    = _graph.append_child(itr, _v);
        auto _children = std::move(_entry_map[_v]);
        _entry_map[_v].clear();
        for(auto&& vitr : _children)
        {
            _recursive_func(_child, vitr);
        }
    };

    // the recursive version of _func + _loop_func has a tendency to overflow the stack
    auto _func = [&](iterator_t itr, const entry& _v) {
        auto _child    = _graph.append_child(itr, _v);
        auto _children = std::move(_entry_map[_v]);
        _entry_map[_v].clear();
        itr_entry_vec_t _data{};
        for(auto&& vitr : _children)
            _data.emplace_back(_child, vitr);
        return _data;
    };

    auto _loop_func = [&_func](itr_entry_vec_t& _data) {
        auto _inp = _data;
        _data.clear();
        for(auto itr : _inp)
        {
            for(auto&& fitr : _func(itr.first, itr.second))
                _data.emplace_back(std::move(fitr));
        }
        // if data is empty return false so we can break out of while loop
        return !_data.empty();
    };

    if(!_indexed.at(-1).empty())
    {
        OMNITRACE_CT_DEBUG_F("Setting root (line %i)...\n", __LINE__);
        _graph.set_head(_indexed.at(-1).front());
    }
    else
    {
        OMNITRACE_CT_DEBUG_F("Setting root (line %i)...\n", __LINE__);
        auto  _depth = static_cast<uint16_t>(-1);
        entry _root{ 0, Device::NONE, Phase::NONE, _depth, 0, 0, 0, 0, 0, 0, 0 };
        _graph.set_head(_root);
    }

    iterator_t _root = _graph.begin();
    for(auto&& itr : _entry_map)
    {
        if(itr.first.depth == _root->depth + 1)
        {
            OMNITRACE_CT_DEBUG_F("Generating call-graph...\n");
            // _recursive_func(_root, itr.first);
            itr_entry_vec_t _data = _func(_root, itr.first);
            while(_loop_func(_data))
            {}
        }
    }
}

void
find_sequences(PTL::ThreadPool& _tp, call_graph_t& _graph,
               std::vector<call_chain>& _chain)
{
    OMNITRACE_CT_DEBUG("\n");
    /*
    using sibling_itr_t = call_graph_sibling_itr_t;
    using sibling_vec_t = std::vector<sibling_itr_t>;
    using sibling_map_t = std::map<int64_t, sibling_vec_t>;

    std::function<void(sibling_map_t & _v, sibling_itr_t root)> _no_overlap{};
    _no_overlap = [&](sibling_map_t& _v, sibling_itr_t root) {
        sibling_map_t _l{};
        int64_t       n = _graph.number_of_children(root);
        if(n == 0) return;

        //_graph.sort(sibling_itr_t{ root },
        //            [](auto lhs, auto rhs) { return lhs.get_cost() > rhs.get_cost(); });

        for(int64_t i = 0; i < n; ++i)
        {
            if(_l.empty())
            {
                auto itr = _graph.child(root, i);
                _l[itr->tid].emplace_back(itr);
            }
            else
            {
                auto itr       = _graph.child(root, i);
                bool _overlaps = false;
                for(auto& litr : _l[itr->tid])
                {
                    if(litr->device == itr->device && litr->get_overlap(*itr) > 0)
                    {
                        _overlaps = true;
                        break;
                    }
                }
                if(!_overlaps) _l[itr->tid].emplace_back(itr);
            }
        }
        for(auto& iitr : _l)
        {
            for(auto itr : iitr.second)
            {
                _v[iitr.first].emplace_back(itr);
                _no_overlap(_v, itr);
            }
        }
    };

    std::map<int64_t, sibling_vec_t> _tot{};
    for(sibling_itr_t itr = _graph.begin(); itr != _graph.end(); ++itr)
    {
        _no_overlap(_tot, itr);
    }

    for(const auto& iitr : _tot)
    {
        call_chain _cc{};
        _cc.emplace_back(*_graph.begin());
        for(const auto& itr : iitr.second)
            _cc.emplace_back(*itr);
        _chain.emplace_back(_cc);
    }

    (void) _tp;
    */

    using iterator_t = call_graph_preorder_itr_t;
    std::vector<iterator_t> _end_nodes{};
    size_t                  _n = 0;
    for(iterator_t itr = _graph.begin(); itr != _graph.end(); ++itr, ++_n)
    {
        auto _nchild = _graph.number_of_children(itr);
        if(_nchild > 0)
        {
            OMNITRACE_CT_DEBUG("Skipping node #%zu with %u children :: %s\n", _n, _nchild,
                               JOIN("", *itr).c_str());
            continue;
        }
        _end_nodes.emplace_back(itr);
    }
    OMNITRACE_CT_DEBUG("Number of end nodes: %zu\n", _end_nodes.size());
    _chain.resize(_end_nodes.size());

    auto _construct = [&](size_t i) {
        auto itr = _end_nodes.at(i);
        while(itr != nullptr && _graph.is_valid(itr))
        {
            _chain.at(i).emplace_back(*itr);
            itr = _graph.parent(itr);
        }
        std::reverse(_chain.at(i).begin(), _chain.at(i).end());
        std::sort(
            _chain.at(i).begin(), _chain.at(i).end(),
            [](const entry& lhs, const entry& rhs) { return lhs.begin_ns > rhs.end_ns; });
    };

    PTL::TaskGroup<void> _tg{ &_tp };
    for(size_t i = 0; i < _end_nodes.size(); ++i)
        _tg.run(_construct, i);
    _tg.join();

    std::sort(_chain.begin(), _chain.end(),
              [](const call_chain& lhs, const call_chain& rhs) {
                  return lhs.get_cost() > rhs.get_cost();
              });

    /*
    std::vector<call_chain> _new_chain{};
    for(auto& itr : _chain)
    {
        if(itr.empty()) continue;
        if(_new_chain.empty())
        {
            _new_chain.emplace_back(std::move(itr));
            continue;
        }
        std::sort(itr.begin(), itr.end(), [](const entry& lhs, const entry& rhs) {
            return lhs.get_cost() > rhs.get_cost();
        });

        call_chain* _append_chain = nullptr;
        for(auto& nitr : _new_chain)
        {
            if(nitr.at(0).tid == itr.at(0).tid && nitr.at(0).get_overlap(itr.at(0)) <= 0)
            {
                _append_chain = &nitr;
                break;
            }
        }

        if(_append_chain)
        {
            for(auto& oitr : itr)
                _append_chain->emplace_back(oitr);
            std::sort(_append_chain->begin(), _append_chain->end(),
                      [](const entry& lhs, const entry& rhs) {
                          return lhs.get_cost() > rhs.get_cost();
                      });
        }
        else
        {
            _new_chain.emplace_back(std::move(itr));
        }
        itr.clear();
    }

    _chain = _new_chain;*/
}

template <typename ArchiveT, typename T, typename AllocatorT>
void
serialize_graph(ArchiveT& ar, const tim::graph<T, AllocatorT>& t)
{
    OMNITRACE_CT_DEBUG("\n");

    namespace cereal = tim::cereal;
    using iterator_t = typename tim::graph<T, AllocatorT>::sibling_iterator;

    ar(cereal::make_nvp("graph_nodes", t.size()));
    ar.setNextName("graph");
    ar.startNode();
    ar.makeArray();
    for(iterator_t itr = t.begin(); itr != t.end(); ++itr)
        serialize_subgraph(ar, t, itr);
    ar.finishNode();
}

template <typename ArchiveT, typename T, typename AllocatorT>
void
serialize_subgraph(ArchiveT& ar, const tim::graph<T, AllocatorT>& _graph,
                   typename tim::graph<T, AllocatorT>::iterator _root)
{
    using iterator_t = typename tim::graph<T, AllocatorT>::sibling_iterator;

    if(_graph.empty()) return;

    ar.setNextName("node");
    ar.startNode();
    ar(*_root);
    {
        ar.setNextName("children");
        ar.startNode();
        ar.makeArray();
        for(iterator_t itr = _graph.begin(_root); itr != _graph.end(_root); ++itr)
            serialize_subgraph(ar, _graph, itr);
        ar.finishNode();
    }
    ar.finishNode();
}

template <Device DevT>
std::vector<call_chain>
get_top(const std::vector<call_chain>& _chain, size_t _count)
{
    OMNITRACE_CT_DEBUG("\n");
    std::vector<call_chain> _data{};
    _data.reserve(_count);
    for(const auto& itr : _chain)
    {
        if(_data.size() >= _count) break;
        if(itr.query<>([](const entry& _v) {
               return (DevT == Device::ANY) ? true : (_v.device == DevT);
           }))
        {
            _data.emplace_back(itr);
        }
    }
    return _data;
}

template <Device DevT>
void
generate_perfetto(const std::vector<call_chain>& _data)
{
    OMNITRACE_CT_DEBUG("\n");

    auto _nrows = std::min<size_t>(get_critical_trace_per_row(), _data.size());

    // run in separate thread(s) so that it ends up in unique row
    if(_nrows < 1) _nrows = _data.size();

    std::string _dev    = (DevT == Device::NONE)  ? ""
                          : (DevT == Device::ANY) ? "CPU + GPU "
                          : (DevT == Device::CPU) ? "CPU "
                                                  : "GPU ";
    std::string _cpname = _dev + "CritPath";
    auto        _func   = [&](size_t _idx, size_t _beg, size_t _end) {
        if(DevT != Device::NONE)
        {
            if(_nrows != 1)
                threading::set_thread_name(TIMEMORY_JOIN(" ", _cpname, _idx).c_str());
            else
                threading::set_thread_name(_cpname.c_str());
        }
        // ensure all hash ids exist
        copy_hash_ids();
        std::set<entry> _used{};
        for(size_t i = _beg; i < _end; ++i)
        {
            if(i >= _data.size()) break;
            _data.at(i).generate_perfetto<DevT>(_used);
        }
    };

    for(size_t i = 0; i < _data.size(); i += _nrows)
    {
        if(DevT == Device::NONE)
            _func(i, i, i + _nrows);
        else
            std::thread{ _func, i, i, i + _nrows }.join();
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
    OMNITRACE_CT_DEBUG("\n");
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
compute_critical_trace()
{
    OMNITRACE_CT_DEBUG_F("Generating critical trace...\n");

    // ensure all hash ids exist
    copy_hash_ids();

    using perfstats_t =
        tim::lightweight_tuple<comp::wall_clock, comp::cpu_clock, comp::cpu_util,
                               comp::peak_rss, comp::page_rss>;

    perfstats_t _ct_perf{};
    _ct_perf.start();

    auto _report_perf = [](auto& _perf, const char* _func, const std::string& _label) {
        _perf.stop().rekey(_label);
        OMNITRACE_BASIC_PRINT("[%s] %s\n", _func, JOIN("", _perf).substr(5).c_str());
        OMNITRACE_BASIC_PRINT("\n");
        _perf.reset().start();
    };

    OMNITRACE_BASIC_PRINT("\n");

    try
    {
        PTL::ThreadPool _tp{ get_critical_trace_num_threads(), []() { copy_hash_ids(); },
                             []() {} };
        _tp.set_verbose(-1);
        PTL::TaskGroup<void> _tg{ &_tp };

        perfstats_t _perf{};
        _perf.start();

        OMNITRACE_BASIC_PRINT_F("sorting %zu call chain entries\n",
                                complete_call_chain.size());

        // sort the complete call chain
        std::sort(complete_call_chain.begin(), complete_call_chain.end());
        _report_perf(_perf, __FUNCTION__, "sorting call chain");

        OMNITRACE_BASIC_PRINT_F("squashing call chain...\n");

        // squash the critical path (combine start/stop into delta)
        squash_critical_path(complete_call_chain);
        _report_perf(_perf, __FUNCTION__, "squashing critical path");

        // generate the perfetto
        if(config::get_use_perfetto())
        {
            OMNITRACE_BASIC_PRINT_F("generating perfetto for call chain...\n");
            generate_perfetto<Device::NONE>({ complete_call_chain });
            generate_perfetto<Device::CPU>({ complete_call_chain });
            generate_perfetto<Device::GPU>({ complete_call_chain });
            _report_perf(_perf, __FUNCTION__, "perfetto generation");
        }

        OMNITRACE_BASIC_PRINT_F("finding children...\n");
        call_graph_t _graph{};
        find_children(_tp, _graph, complete_call_chain);
        _report_perf(_perf, __FUNCTION__, "finding children");

        // sort the call-graph based on cost
        OMNITRACE_BASIC_PRINT_F("sorting %zu call-graph entries...\n", _graph.size() - 1);
        _graph.sort([](auto lhs, auto rhs) { return lhs.get_cost() > rhs.get_cost(); },
                    [&_tg](auto _f) { _tg.run(_f); }, [&_tg]() { _tg.join(); });
        _report_perf(_perf, __FUNCTION__, "call-graph sort");

        OMNITRACE_BASIC_PRINT_F("saving call-graph...\n");
        save_call_graph(tim::settings::compose_output_filename("call-graph", ".json"),
                        "call_graph", _graph, true, __FUNCTION__);
        _report_perf(_perf, __FUNCTION__, "saving call-graph");

        OMNITRACE_BASIC_PRINT_F("finding sequences...\n");
        std::vector<call_chain> _top{};
        find_sequences(_tp, _graph, _top);
        _report_perf(_perf, __FUNCTION__, "call-graph sequence search");

        OMNITRACE_BASIC_PRINT_F("number of sequences found: %zu (%zu)...\n", _top.size(),
                                (_top.empty()) ? 0 : _top.at(0).size());

        if(get_critical_trace_count() == 0)
        {
            OMNITRACE_CT_DEBUG_F("saving critical trace...\n");
            save_critical_trace(
                tim::settings::compose_output_filename("critical-trace", ".json"),
                "critical_trace", _top, true, __FUNCTION__);
        }
        else
        {
            // get the top CPU critical traces
            OMNITRACE_BASIC_PRINT_F("getting top CPU functions...\n");
            auto _top_cpu = get_top<Device::CPU>(_top, get_critical_trace_count());

            // get the top GPU critical traces
            OMNITRACE_BASIC_PRINT_F("getting top GPU functions...\n");
            auto _top_gpu = get_top<Device::GPU>(_top, get_critical_trace_count());

            // get the top CPU + GPU critical traces
            OMNITRACE_BASIC_PRINT_F("getting top CPU + GPU functions...\n");
            auto _top_any = get_top<Device::ANY>(_top, get_critical_trace_count());

            if(!_top_cpu.empty())
            {
                OMNITRACE_BASIC_PRINT_F(
                    "generating %zu perfetto CPU critical traces...\n", _top_cpu.size());
                if(config::get_use_perfetto()) generate_perfetto<Device::CPU>(_top_cpu);
                OMNITRACE_CT_DEBUG_F("saving CPU critical traces...\n");
                save_critical_trace(
                    tim::settings::compose_output_filename("critical-trace-cpu", ".json"),
                    "critical_trace", _top_cpu, true, __FUNCTION__);
            }

            if(!_top_gpu.empty())
            {
                OMNITRACE_BASIC_PRINT_F(
                    "generating %zu perfetto GPU critical traces...\n", _top_gpu.size());
                if(config::get_use_perfetto()) generate_perfetto<Device::GPU>(_top_gpu);
                OMNITRACE_CT_DEBUG_F("saving GPU critical traces...\n");
                save_critical_trace(
                    tim::settings::compose_output_filename("critical-trace-gpu", ".json"),
                    "critical_trace", _top_gpu, true, __FUNCTION__);
            }

            if(!_top_any.empty())
            {
                OMNITRACE_BASIC_PRINT_F(
                    "generating %zu perfetto CPU + GPU critical traces...\n",
                    _top_gpu.size());
                if(config::get_use_perfetto()) generate_perfetto<Device::ANY>(_top_gpu);
                OMNITRACE_CT_DEBUG_F("saving CPU + GPU critical traces...\n");
                save_critical_trace(
                    tim::settings::compose_output_filename("critical-trace-any", ".json"),
                    "critical_trace", _top_any, true, __FUNCTION__);
            }
        }

        _tg.join();
        _tp.destroy_threadpool();
    } catch(std::exception& e)
    {
        OMNITRACE_BASIC_PRINT("Thread exited '%s' with exception: %s\n", __FUNCTION__,
                              e.what());
        TIMEMORY_CONDITIONAL_DEMANGLED_BACKTRACE(true, 32);
    }

    _report_perf(_ct_perf, __FUNCTION__, "critical trace computation");
}
}  // namespace
}  // namespace critical_trace
}  // namespace omnitrace
