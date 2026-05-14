// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/tracing.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/state.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include <cstdint>

#include <timemory/hash/types.hpp>
#include <timemory/process/threading.hpp>

#include "logger/debug.hpp"

namespace rocprofsys::tracing
{
namespace
{
tim::hash_map_ptr_t&
get_timemory_hash_ids(std::int64_t _tid = threading::get_id());

tim::hash_alias_ptr_t&
get_timemory_hash_aliases(std::int64_t _tid = threading::get_id());

tim::hash_map_ptr_t&
get_timemory_hash_ids(std::int64_t _tid)
{
    return thread_data<identity<tim::hash_map_ptr_t>>::instance(
        construct_on_thread{ _tid });
}

tim::hash_alias_ptr_t&
get_timemory_hash_aliases(std::int64_t _tid)
{
    return thread_data<identity<tim::hash_alias_ptr_t>>::instance(
        construct_on_thread{ _tid });
}
}  // namespace

bool debug_push = tim::get_env("ROCPROFSYS_DEBUG_PUSH", false) || get_debug_env();
bool debug_pop  = tim::get_env("ROCPROFSYS_DEBUG_POP", false) || get_debug_env();
bool debug_mark = tim::get_env("ROCPROFSYS_DEBUG_MARK", false) || get_debug_env();
bool debug_user = tim::get_env("ROCPROFSYS_DEBUG_USER_REGIONS", false) || get_debug_env();

std::unordered_map<hash_value_t, std::string>&
get_perfetto_track_uuids()
{
    static auto _v = std::unordered_map<hash_value_t, std::string>{};
    return _v;
}

std::mutex&
get_perfetto_track_uuids_mutex()
{
    static auto _mtx = std::mutex{};
    return _mtx;
}

void
copy_timemory_hash_ids()
{
    auto_lock_t _ilk{ type_mutex<tim::hash_map_t>(), std::defer_lock };
    auto_lock_t _alk{ type_mutex<tim::hash_alias_map_t>(), std::defer_lock };

    if(!_ilk.owns_lock()) _ilk.lock();
    if(!_alk.owns_lock()) _alk.lock();

    // copy these over so that all hashes are known
    auto& _hmain = tim::hash::get_main_hash_ids();
    auto& _amain = tim::hash::get_main_hash_aliases();
    if(_hmain == nullptr)
    {
        LOG_CRITICAL("no main timemory hash ids");
        std::exit(1);
    }
    if(_amain == nullptr)
    {
        LOG_CRITICAL("no main timemory hash aliases");
        std::exit(1);
    }

    // Access underlying storage directly to avoid construct_on_thread which
    // throws when peak_num_threads grew beyond the thread_data capacity
    // (e.g., during re-attach where grow functors weren't registered yet)
    using hash_thread_data_t  = thread_data<identity<tim::hash_map_ptr_t>>;
    using alias_thread_data_t = thread_data<identity<tim::hash_alias_ptr_t>>;

    const auto& hash_storage  = hash_thread_data_t::instance();
    const auto& alias_storage = alias_thread_data_t::instance();

    const auto peak_threads = thread_info::get_peak_num_threads();

    if(hash_storage)
    {
        const auto num_entries = std::min(peak_threads, hash_storage->size());
        for(size_t i = 0; i < num_entries; ++i)
        {
            const auto& _hitr = (*hash_storage)[i];
            if(_hitr)
            {
                for(const auto& itr : *_hitr)
                    _hmain->emplace(itr.first, itr.second);
            }
        }
    }

    if(alias_storage)
    {
        const auto num_entries = std::min(peak_threads, alias_storage->size());
        for(size_t i = 0; i < num_entries; ++i)
        {
            const auto& _aitr = (*alias_storage)[i];
            if(_aitr)
            {
                for(const auto& itr : *_aitr)
                    _amain->emplace(itr.first, itr.second);
            }
        }
    }

    // distribute the contents of that combined container to each thread-specific
    // container before finalizing
    if(get_state() == State::Finalized)
    {
        if(hash_storage)
        {
            const auto num_entries = std::min(peak_threads, hash_storage->size());
            for(size_t i = 0; i < num_entries; ++i)
            {
                auto& _hitr = (*hash_storage)[i];
                if(_hitr) *_hitr = *_hmain;
            }
        }

        if(alias_storage)
        {
            const auto num_entries = std::min(peak_threads, alias_storage->size());
            for(size_t i = 0; i < num_entries; ++i)
            {
                auto& _aitr = (*alias_storage)[i];
                if(_aitr) *_aitr = *_amain;
            }
        }
    }
}

std::vector<std::function<void()>>&
get_finalization_functions()
{
    static auto _v = std::vector<std::function<void()>>{};
    return _v;
}

void
record_thread_start_time()
{
    static thread_local std::once_flag _once{};
    std::call_once(_once, []() {
        thread_info::set_start(comp::wall_clock::record(), get_mode() != Mode::Sampling);
    });
}

void
thread_init()
{
    if(get_thread_state() == ThreadState::Disabled) return;

    static thread_local auto _thread_dtor = scope::destructor{ []() {
        if(get_state() != State::Finalized)
        {
            if(get_use_causal())
                causal::sampling::shutdown();
            else if(get_use_sampling())
                sampling::shutdown();
            auto& _thr_bundle = thread_data<thread_bundle_t>::instance();
            if(_thr_bundle && _thr_bundle->get<comp::wall_clock>() &&
               _thr_bundle->get<comp::wall_clock>()->get_is_running())
                _thr_bundle->stop();
        }
    } };

    if(get_thread_state() == ThreadState::Disabled) return;

    static thread_local auto _thread_setup = []() {
        const auto& _tinfo = thread_info::init();
        auto _tidx = (_tinfo && _tinfo->index_data) ? _tinfo->index_data->sequent_value
                                                    : threading::get_id();

        if(_tidx < 0)
        {
            LOG_CRITICAL("thread setup failed. thread info not initialized: {}",
                         [&_tinfo]() {
                             if(_tinfo) return _tinfo->as_string();
                             return std::string{ "no thread_info" };
                         }());
            std::exit(1);
        }

        if(_tidx > 0) threading::set_thread_name(fmt::format("Thread {}", _tidx).c_str());
        thread_data<thread_bundle_t>::construct(
            fmt::format("rocprofsys/process/{}/thread/{}", process::get_id(), _tidx),
            quirk::config<quirk::auto_start>{});
        // save the hash maps
        get_timemory_hash_ids(_tidx)     = tim::get_hash_ids();
        get_timemory_hash_aliases(_tidx) = tim::get_hash_aliases();

        if(get_timemory_hash_ids(_tidx) == nullptr)
        {
            LOG_CRITICAL("no timemory hash ids pointer for thread {}", _tidx);
            std::exit(1);
        }
        if(get_timemory_hash_aliases(_tidx) == nullptr)
        {
            LOG_CRITICAL("no timemory hash aliases pointer for thread {}", _tidx);
            std::exit(1);
        }

        record_thread_start_time();
        return true;
    }();

    if(get_thread_state() == ThreadState::Disabled) return;

    static thread_local auto _sample_setup = []() {
        auto _idx = utility::get_thread_index();
        // the main thread will initialize sampling when it initializes the tooling
        if(_idx > 0)
        {
            auto _use_causal   = get_use_causal();
            auto _use_sampling = get_use_sampling();
            if(_use_causal || _use_sampling)
            {
                ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);
                if(_use_causal)
                    causal::sampling::setup();
                else if(_use_sampling)
                    sampling::setup();
            }
            return (_use_causal || _use_sampling);
        }
        return false;
    }();

    (void) _thread_dtor;
    (void) _thread_setup;
    (void) _sample_setup;
}
}  // namespace rocprofsys::tracing
