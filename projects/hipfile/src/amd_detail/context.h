/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile-warnings.h"
#include "stats.h"

#include <stdexcept>
#ifdef AIS_TESTING
#include <mutex>
#endif

namespace hipFile {

template <typename T> struct ContextOverride;

template <typename T> struct ContextDefaultImpl {
    using type = T;
};

template <> struct ContextDefaultImpl<IStatsServer> {
    using type = StatsServer;
};

template <typename T> struct Context {
    using DefaultImpl = typename ContextDefaultImpl<T>::type;
    static_assert(std::is_base_of_v<T, DefaultImpl>, "ContextDefaultImpl<T>::type must derive from T");
    Context()                           = delete;
    Context(const Context &)            = delete;
    Context(Context &&)                 = delete;
    Context &operator=(const Context &) = delete;
    Context &operator=(Context &&)      = delete;
    ~Context()                          = delete;
#ifdef AIS_TESTING
    inline static T         *replacement = nullptr;
    inline static std::mutex m;

    /// @brief Get the context
    /// @return A pointer to the context
    static T *get()
    {
        std::lock_guard<std::mutex> lock{m};
        if (replacement)
            return replacement;
        HIPFILE_WARN_NO_EXIT_DTOR_OFF
        static DefaultImpl standard{};
        HIPFILE_WARN_NO_EXIT_DTOR_ON
        return &standard;
    }

private:
    /// @brief Set a context to temporarily override the default
    /// @param _replacement A pointer to the replacement context
    static void override(T *_replacement)
    {
        std::lock_guard<std::mutex> lock{m};
        if (replacement) {
            throw std::runtime_error("A replacement is already set.");
        }
        replacement = _replacement;
    }
    /// @brief Restore the default context
    static void restore()
    {
        std::lock_guard<std::mutex> lock{m};
        if (!replacement) {
            throw std::runtime_error("No context is available to restore.");
        }
        replacement = nullptr;
    }

    friend struct ContextOverride<T>;
#else
    /// @brief Get the context
    /// @return A pointer to the context
    static T *get()
    {
        HIPFILE_WARN_NO_EXIT_DTOR_OFF
        static DefaultImpl context{};
        HIPFILE_WARN_NO_EXIT_DTOR_ON
        return &context;
    }
#endif
};

#ifdef AIS_TESTING
template <typename T> struct ContextOverride {
    /// @brief Override a context.
    ///        The context will be restored upon destruction.
    /// @param val A pointer to the new context
    template <typename U> ContextOverride(U *val)
    {
        Context<T>::override(val);
    }
    ~ContextOverride()
    {
        Context<T>::restore();
    }
    ContextOverride(const ContextOverride &)            = delete;
    ContextOverride &operator=(const ContextOverride &) = delete;
};
#endif

void hipFileInit();

}
