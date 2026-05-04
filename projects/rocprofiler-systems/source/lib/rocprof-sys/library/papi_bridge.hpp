// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#if defined(ROCPROFSYS_USE_PAPI)

#include "logger/debug.hpp"

#include <papi.h>
#include <pthread.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

namespace rocprofsys::papi_bridge
{

namespace detail
{

inline std::once_flag              g_init_flag;
inline std::vector<int>            g_event_codes;
inline std::vector<std::string>    g_labels;
inline bool                        g_available = false;
inline thread_local int            tl_event_set = PAPI_NULL;

inline void
ensure_initialized()
{
    std::call_once(g_init_flag, [] {
        int ver = PAPI_library_init(PAPI_VER_CURRENT);
        if(ver != PAPI_VER_CURRENT)
        {
            LOG_WARNING("PAPI_library_init failed (returned {}), hw counters disabled",
                        ver);
            return;
        }

        int ret = PAPI_thread_init(
            reinterpret_cast<unsigned long (*)(void)>(pthread_self));
        if(ret != PAPI_OK)
        {
            LOG_WARNING("PAPI_thread_init failed (returned {}), hw counters disabled",
                        ret);
            return;
        }

        const char* env_val = std::getenv("ROCPROFSYS_PAPI_EVENTS");
        if(!env_val || env_val[0] == '\0') return;

        std::string events_str{ env_val };

        std::vector<std::string> names;
        std::string              token;
        for(char c : events_str)
        {
            if(c == ',' || c == ' ' || c == ';' || c == '\"' || c == '\'')
            {
                if(!token.empty())
                {
                    names.push_back(token);
                    token.clear();
                }
            }
            else
            {
                token += c;
            }
        }
        if(!token.empty()) names.push_back(token);

        for(auto const& name : names)
        {
            int code = 0;
            if(PAPI_event_name_to_code(name.c_str(), &code) == PAPI_OK)
            {
                g_event_codes.push_back(code);
                g_labels.push_back(name);
            }
            else
            {
                LOG_WARNING("PAPI event '{}' not available, skipping", name);
            }
        }

        g_available = !g_event_codes.empty();

        if(g_available)
        {
            LOG_INFO("PAPI bridge: {} events configured", g_event_codes.size());
        }
    });
}

}  // namespace detail

inline void
setup(int64_t tid)
{
    detail::ensure_initialized();
    if(!detail::g_available) return;
    if(detail::tl_event_set != PAPI_NULL) return;

    PAPI_register_thread();

    detail::tl_event_set = PAPI_NULL;
    int ret = PAPI_create_eventset(&detail::tl_event_set);
    if(ret != PAPI_OK)
    {
        LOG_WARNING("PAPI_create_eventset failed for thread {} (ret={})", tid, ret);
        detail::tl_event_set = PAPI_NULL;
        return;
    }

    ret = PAPI_add_events(detail::tl_event_set, detail::g_event_codes.data(),
                          static_cast<int>(detail::g_event_codes.size()));
    if(ret != PAPI_OK)
    {
        LOG_WARNING("PAPI_add_events failed for thread {} (ret={})", tid, ret);
        PAPI_destroy_eventset(&detail::tl_event_set);
        detail::tl_event_set = PAPI_NULL;
        return;
    }

    ret = PAPI_start(detail::tl_event_set);
    if(ret != PAPI_OK)
    {
        LOG_WARNING("PAPI_start failed for thread {} (ret={})", tid, ret);
        PAPI_cleanup_eventset(detail::tl_event_set);
        PAPI_destroy_eventset(&detail::tl_event_set);
        detail::tl_event_set = PAPI_NULL;
        return;
    }

    LOG_DEBUG("PAPI bridge: started {} events on thread {}", detail::g_event_codes.size(),
              tid);
}

inline void
teardown(int64_t tid)
{
    if(detail::tl_event_set == PAPI_NULL) return;

    std::vector<long long> tmp(detail::g_event_codes.size(), 0);
    PAPI_stop(detail::tl_event_set, tmp.data());
    PAPI_cleanup_eventset(detail::tl_event_set);
    PAPI_destroy_eventset(&detail::tl_event_set);
    detail::tl_event_set = PAPI_NULL;

    LOG_DEBUG("PAPI bridge: stopped events on thread {}", tid);
}

inline std::size_t
read(int64_t /*tid*/, long long* out, std::size_t max_count)
{
    if(detail::tl_event_set == PAPI_NULL) return 0;

    std::size_t n = std::min(max_count, detail::g_event_codes.size());
    if(n == 0) return 0;

    if(PAPI_read(detail::tl_event_set, out) != PAPI_OK) return 0;

    return n;
}

inline std::vector<std::string> const&
get_labels()
{
    detail::ensure_initialized();
    return detail::g_labels;
}

}  // namespace rocprofsys::papi_bridge

#endif  // ROCPROFSYS_USE_PAPI
