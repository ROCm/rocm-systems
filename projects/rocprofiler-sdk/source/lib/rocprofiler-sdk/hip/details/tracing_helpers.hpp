// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/tracing/tracing.hpp"

#include <rocprofiler-sdk/fwd.h>

#include <cassert>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace hip
{
namespace tracing_helpers
{
// Sentinel: pass as the BufferKind template argument to
// create_traced_wrapper_none_phase / _enter_exit_phase to subscribe only via
// the callback domain (no companion buffer kind).
inline constexpr auto buffer_kind_none = ROCPROFILER_BUFFER_TRACING_NONE;

// Bundles per-call subscriber state populated by populate_*. Reused across
// paired ENTER/EXIT fires (execute_phase_exit_callbacks reads per-context
// thread_id / correlation_id that execute_phase_enter_callbacks wrote into
// itr.record, so the same callback_contexts vector must flow through both
// phases).
struct subscriber_state
{
    tracing::callback_context_data_vec_t   callback_contexts;
    tracing::buffered_context_data_vec_t   buffered_contexts;
    tracing::external_correlation_id_map_t external_corr_ids;
    rocprofiler_thread_id_t                thread_id = 0;
};

// ---------------------------------------------------------------------------
// Wrapper factories
//
// Each factory returns a plain-function-pointer-typed lambda suitable for
// installation into a HIP dispatch table slot. The lambda:
//   1. Populates the subscriber state for the given callback + buffer kinds.
//   2. Updates external correlation ids.
//   3. Invokes next_func with the forwarded args (for ENTER/EXIT, between
//      the enter and exit fires).
//   4. If any callback context is subscribed, builds the payload via
//      build_payload(...) and fires the appropriate callback phase(s).
//
// The build_payload callable owns the wrapper-specific work: it decides
// which input args / return values / runtime state populate the payload.
// It MUST return a PayloadT by value.
//
// For NONE-phase: build_payload(args..., RetT) is invoked AFTER next_func
//   (so the payload may use the return value, e.g., a freshly-created
//   handle returned through an out-parameter).
// For ENTER/EXIT phase: build_payload(args...) is invoked BEFORE next_func
//   and the SAME payload instance is used for both fires. Wrappers that
//   need EXIT-specific payload state should not use this factory.
//
// For void-returning APIs, build_payload(args...) is the only form.
// ---------------------------------------------------------------------------

namespace detail
{
inline bool
populate_callback(rocprofiler_callback_tracing_kind_t callback_kind,
                  rocprofiler_buffer_tracing_kind_t   buffer_kind,
                  rocprofiler_tracing_operation_t     op,
                  subscriber_state&                   state)
{
    state.thread_id = common::get_tid();
    if(buffer_kind == buffer_kind_none)
    {
        tracing::populate_contexts(
            callback_kind, op, state.callback_contexts, state.external_corr_ids);
    }
    else
    {
        tracing::populate_contexts(callback_kind,
                                   buffer_kind,
                                   state.callback_contexts,
                                   state.buffered_contexts,
                                   state.external_corr_ids);
    }
    return !state.callback_contexts.empty();
}

inline void
update_external_corr_ids(subscriber_state&                                  state,
                         rocprofiler_external_correlation_id_request_kind_t request_kind)
{
    tracing::update_external_correlation_ids(
        state.external_corr_ids, state.thread_id, request_kind);
}
}  // namespace detail

// One-shot (phase NONE) wrapper factory. Suitable for CREATE / DESTROY style
// wrappers where the callback fires once after the wrapped function returns.
//
// The lambda has no captures and decays to a plain function pointer (FuncT).
// next_func is held in static storage; the static lives at function scope and
// is unique per template instantiation. Callers pass distinct
// (TableIdx, OpIdx) for each wrapped API to guarantee a separate
// instantiation per call site -- otherwise two wrappers for different APIs
// with identical signatures would share storage and the second call would
// overwrite the first.
//
// PayloadT is the callback domain's payload struct (e.g.,
// rocprofiler_callback_tracing_hip_stream_data_t).
//
// build_payload is invoked AFTER next_func returns and is passed
// (args..., RetT&) for non-void return types, or just (args...) for void.
// It must return PayloadT by value.
template <size_t TableIdx,
          size_t OpIdx,
          typename PayloadT,
          typename BuildFn,
          typename RetT,
          typename... Args,
          typename FuncT = RetT (*)(Args...)>
FuncT
create_traced_wrapper_none_phase(
    RetT (*next)(Args...),
    rocprofiler_callback_tracing_kind_t                callback_kind,
    rocprofiler_buffer_tracing_kind_t                  buffer_kind,
    rocprofiler_tracing_operation_t                    op,
    rocprofiler_external_correlation_id_request_kind_t corr_request_kind,
    BuildFn                                            build_payload)
{
    // Capture the configuration in template-instantiation-local static
    // storage. We rely on the fact that each unique call site (different
    // template parameter pack OR different non-type domain arguments via
    // separate template instantiations) gets its own static slot.
    static auto next_func    = next;
    static auto fn_callback  = callback_kind;
    static auto fn_buffer    = buffer_kind;
    static auto fn_op        = op;
    static auto fn_corr_kind = corr_request_kind;
    static auto fn_build     = build_payload;

    return +[](Args... args) -> RetT {
        auto state = subscriber_state{};
        bool any   = detail::populate_callback(fn_callback, fn_buffer, fn_op, state);

        detail::update_external_corr_ids(state, fn_corr_kind);

        if constexpr(std::is_void<RetT>::value)
        {
            next_func(std::forward<Args>(args)...);
            if(any)
            {
                auto payload = fn_build(std::forward<Args>(args)...);
                tracing::execute_phase_none_callbacks(state.callback_contexts,
                                                      state.thread_id,
                                                      /*internal_corr_id*/ 0,
                                                      state.external_corr_ids,
                                                      /*ancestor_corr_id*/ 0,
                                                      fn_callback,
                                                      fn_op,
                                                      payload);
            }
        }
        else
        {
            auto _ret = next_func(std::forward<Args>(args)...);
            if(any)
            {
                auto payload = fn_build(std::forward<Args>(args)..., _ret);
                tracing::execute_phase_none_callbacks(state.callback_contexts,
                                                      state.thread_id,
                                                      /*internal_corr_id*/ 0,
                                                      state.external_corr_ids,
                                                      /*ancestor_corr_id*/ 0,
                                                      fn_callback,
                                                      fn_op,
                                                      payload);
            }
            return _ret;
        }
    };
}

// Paired ENTER/EXIT wrapper factory. Suitable for wrappers that need to
// bracket the wrapped function with a callback fire on each side (e.g.,
// HIP_STREAM_SET). The same payload instance flows through both fires;
// wrappers requiring different EXIT-time payload state should not use
// this factory.
//
// build_payload is invoked BEFORE next_func and is passed (args...). It
// must return PayloadT by value.
//
// See create_traced_wrapper_none_phase for the (TableIdx, OpIdx)
// instantiation-uniqueness rationale.
template <size_t TableIdx,
          size_t OpIdx,
          typename PayloadT,
          typename BuildFn,
          typename RetT,
          typename... Args,
          typename FuncT = RetT (*)(Args...)>
FuncT
create_traced_wrapper_enter_exit_phase(
    RetT (*next)(Args...),
    rocprofiler_callback_tracing_kind_t                callback_kind,
    rocprofiler_buffer_tracing_kind_t                  buffer_kind,
    rocprofiler_tracing_operation_t                    op,
    rocprofiler_external_correlation_id_request_kind_t corr_request_kind,
    BuildFn                                            build_payload)
{
    static auto next_func    = next;
    static auto fn_callback  = callback_kind;
    static auto fn_buffer    = buffer_kind;
    static auto fn_op        = op;
    static auto fn_corr_kind = corr_request_kind;
    static auto fn_build     = build_payload;

    return +[](Args... args) -> RetT {
        auto state = subscriber_state{};
        bool any   = detail::populate_callback(fn_callback, fn_buffer, fn_op, state);

        auto payload = fn_build(args...);

        if(any)
        {
            tracing::execute_phase_enter_callbacks(state.callback_contexts,
                                                   state.thread_id,
                                                   /*internal_corr_id*/ 0,
                                                   state.external_corr_ids,
                                                   /*ancestor_corr_id*/ 0,
                                                   fn_callback,
                                                   fn_op,
                                                   payload);
        }

        detail::update_external_corr_ids(state, fn_corr_kind);

        if constexpr(std::is_void<RetT>::value)
        {
            next_func(std::forward<Args>(args)...);
            if(any)
            {
                tracing::execute_phase_exit_callbacks(state.callback_contexts,
                                                      state.external_corr_ids,
                                                      fn_callback,
                                                      fn_op,
                                                      payload);
            }
        }
        else
        {
            auto _ret = next_func(std::forward<Args>(args)...);
            if(any)
            {
                tracing::execute_phase_exit_callbacks(state.callback_contexts,
                                                      state.external_corr_ids,
                                                      fn_callback,
                                                      fn_op,
                                                      payload);
            }
            return _ret;
        }
    };
}
}  // namespace tracing_helpers
}  // namespace hip
}  // namespace rocprofiler
