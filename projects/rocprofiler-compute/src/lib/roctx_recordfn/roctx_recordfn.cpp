// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// ROCTX bridge for PyTorch's RecordFunction callback. Subscribes to the
// FUNCTION and BACKWARD_FUNCTION scopes and propagates the main-thread
// USER_SCOPE chain into autograd workers via RecordFunction::seqNr()
// and c10::ThreadLocalDebugInfo.

#include "leaf_context.h"

#include <ATen/record_function.h>
#include <c10/util/ThreadLocalDebugInfo.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

namespace
{

struct StackEntry
{
    std::string marker;
    std::string context;
};

// Records what start_cb pushed so end_cb can unwind exactly that.
struct RoctxObserverContext : public at::ObserverContext
{
    bool        pushed_roctx_range     = false;
    bool        pushed_leaf            = false;
    std::size_t pushed_snapshot_frames = 0;
};

// Per-thread marker stack. Autograd workers start empty and are
// re-seeded from the seqNr snapshot and the TLS USER_SCOPE chain.
thread_local std::vector<StackEntry> g_stack;

// Carries the main-thread USER_SCOPE chain to autograd workers.
class RoctxUserScopeChain : public c10::DebugInfoBase
{
public:
    explicit RoctxUserScopeChain(std::vector<StackEntry> c)
        : chain(std::move(c))
    {
    }

    std::vector<StackEntry> chain;
};

// Use a private DebugInfoKind slot keyed by string_view identity when
// the PyTorch ABI supports it; otherwise reuse the TEST_INFO_2 slot.
#ifdef ROCPROF_TORCHTRACE_HAS_CUSTOM_DBGINFOKIND
inline constexpr std::string_view kRoctxUserScopeName = "ROCPROF_TORCHTRACE_INFO";
inline const c10::DebugInfoKind   kRoctxDbgKind(&kRoctxUserScopeName);
#else
constexpr c10::DebugInfoKind kRoctxDbgKind = c10::DebugInfoKind::TEST_INFO_2;
#endif

// LIFO of DebugInfoGuards mirroring push_user_scope/pop_user_scope.
thread_local std::vector<std::unique_ptr<c10::DebugInfoGuard>> g_dbg_guards;

// Sharded seqNr -> forward-stack snapshot store.
constexpr std::size_t NUM_SHARDS = 64;

struct Shard
{
    std::mutex                                                          mutex;
    std::unordered_map<std::int64_t, std::vector<StackEntry>>           snapshots;
    std::list<std::int64_t>                                             lru_order;
    std::unordered_map<std::int64_t, std::list<std::int64_t>::iterator> lru_idx;
};

std::array<Shard, NUM_SHARDS> g_shards;

// Per-shard cap. Snapshots whose backward never runs are evicted in LRU order.
constexpr std::size_t SHARD_SOFT_CAP = 10000;

std::atomic<at::CallbackHandle> g_handle{at::INVALID_CALLBACK_HANDLE};
std::atomic<bool>               g_installed{false};
std::mutex                      g_install_mutex;

std::atomic<std::uint64_t> g_n_pushes{0};
std::atomic<std::uint64_t> g_n_pops{0};
std::atomic<std::uint64_t> g_n_snapshots_saved{0};
std::atomic<std::uint64_t> g_n_snapshots_consumed{0};
std::atomic<std::uint64_t> g_n_snapshots_dropped{0};
std::atomic<std::uint64_t> g_n_callback_errors{0};
std::atomic<std::uint64_t> g_n_user_scope_pushes{0};
std::atomic<std::uint64_t> g_n_user_scope_pops{0};
std::atomic<std::uint64_t> g_n_user_scope_inherits{0};

// Opt-in capture buffer used by the test hook.
std::atomic<bool>        g_capturing{false};
std::mutex               g_capture_mutex;
std::vector<std::string> g_captured;
constexpr std::size_t    CAPTURE_CAP = 4096;

// The RecordFunction tier instruments PyTorch ATen operators.
constexpr const char* kRecordFnBackend = "torch";

void maybe_capture(const std::string& wire_string)
{
    if (!g_capturing.load(std::memory_order_relaxed))
        return;
    std::lock_guard<std::mutex> guard(g_capture_mutex);
    if (g_captured.size() < CAPTURE_CAP)
    {
        g_captured.push_back(wire_string);
    }
}

// Percent-encoding of the two characters that would otherwise collide with the
// marker-path grammar. The inverse decode lives with the Python readers
// (utils/inject_roctx/core.py decode_marker_name, utils/utils_analysis.py); the
// C++ round-trip test reuses these same constants so the escape table has a
// single definition.
constexpr const char* kEncodedPercent = "%25";
constexpr const char* kEncodedSlash   = "%2F";

// Appends name to out with '%' and '/' percent-encoded so an embedded '/' is
// not read as the frame separator in build_marker_string.
void encode_marker_segment(const std::string& name, std::string& out)
{
    for (char c : name)
    {
        if (c == '%')
            out += kEncodedPercent;
        else if (c == '/')
            out += kEncodedSlash;
        else
            out += c;
    }
}

// Appends the frames to out as a '/'-separated list, using select_field to
// render each frame's chosen field.
template<typename SelectField>
void append_joined_frames(const std::vector<StackEntry>& stack, std::string& out, SelectField select_field)
{
    bool first = true;
    for (const auto& entry : stack)
    {
        if (!first)
            out += '/';
        select_field(entry, out);
        first = false;
    }
}

// Renders the stack as "marker1/.../markerN:context1/.../contextN". Marker names
// are percent-encoded so an embedded '/' is not read as the frame separator.
std::string build_marker_string(const std::vector<StackEntry>& stack)
{
    std::size_t marker_len = 0;
    std::size_t ctx_len    = 0;
    for (const auto& entry : stack)
    {
        marker_len += entry.marker.size() + 1;
        // Each '%' or '/' expands from one char to three when encoded.
        for (char c : entry.marker)
            if (c == '%' || c == '/')
                marker_len += 2;
        ctx_len += entry.context.size() + 1;
    }
    std::string out;
    out.reserve(marker_len + ctx_len + 1);

    append_joined_frames(stack,
                         out,
                         [](const StackEntry& entry, std::string& dst)
                         { encode_marker_segment(entry.marker, dst); });
    out += ':';
    append_joined_frames(stack,
                         out,
                         [](const StackEntry& entry, std::string& dst) { dst += entry.context; });
    return out;
}

void lru_remove(Shard& shard, std::int64_t seq_nr)
{
    auto it = shard.lru_idx.find(seq_nr);
    if (it == shard.lru_idx.end())
        return;
    shard.lru_order.erase(it->second);
    shard.lru_idx.erase(it);
}

void lru_touch(Shard& shard, std::int64_t seq_nr)
{
    lru_remove(shard, seq_nr);
    shard.lru_order.push_back(seq_nr);
    auto tail = shard.lru_order.end();
    --tail;
    shard.lru_idx.emplace(seq_nr, tail);
}

void evict_oldest_snapshot(Shard& shard)
{
    if (shard.lru_order.empty())
        return;
    const std::int64_t oldest = shard.lru_order.front();
    shard.lru_order.pop_front();
    shard.lru_idx.erase(oldest);
    shard.snapshots.erase(oldest);
    g_n_snapshots_dropped.fetch_add(1, std::memory_order_relaxed);
}

void save_snapshot(std::int64_t seq_nr, const std::vector<StackEntry>& stack)
{
    auto&                       shard = g_shards[static_cast<std::size_t>(seq_nr) % NUM_SHARDS];
    std::lock_guard<std::mutex> guard(shard.mutex);
    auto                        it = shard.snapshots.find(seq_nr);
    if (it != shard.snapshots.end())
    {
        it->second = stack;
        lru_touch(shard, seq_nr);
        g_n_snapshots_saved.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    while (shard.snapshots.size() >= SHARD_SOFT_CAP)
    {
        evict_oldest_snapshot(shard);
    }
    shard.snapshots.emplace(seq_nr, stack);
    lru_touch(shard, seq_nr);
    g_n_snapshots_saved.fetch_add(1, std::memory_order_relaxed);
}

bool consume_snapshot(std::int64_t seq_nr, std::vector<StackEntry>* out_stack)
{
    auto&                       shard = g_shards[static_cast<std::size_t>(seq_nr) % NUM_SHARDS];
    std::lock_guard<std::mutex> guard(shard.mutex);
    auto                        it = shard.snapshots.find(seq_nr);
    if (it == shard.snapshots.end())
        return false;
    *out_stack = std::move(it->second);
    shard.snapshots.erase(it);
    lru_remove(shard, seq_nr);
    g_n_snapshots_consumed.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Pushes `chain` onto g_stack, skipping any leading prefix that is
// already present. Returns the number of frames pushed.
std::size_t push_with_prefix_dedup(const std::vector<StackEntry>& chain)
{
    const std::size_t maxc   = std::min(chain.size(), g_stack.size());
    std::size_t       common = 0;
    for (; common < maxc; ++common)
    {
        if (chain[common].marker != g_stack[common].marker ||
            chain[common].context != g_stack[common].context)
        {
            break;
        }
    }
    std::size_t pushed = 0;
    for (std::size_t i = common; i < chain.size(); ++i)
    {
        g_stack.push_back(chain[i]);
        ++pushed;
    }
    return pushed;
}

// Overlays the published USER_SCOPE chain onto g_stack.
std::size_t apply_userscope_overlay()
{
    auto* base       = c10::ThreadLocalDebugInfo::get(kRoctxDbgKind);
    auto* chain_info = dynamic_cast<const RoctxUserScopeChain*>(base);
    if (chain_info == nullptr || chain_info->chain.empty())
    {
        return 0;
    }
    const std::vector<StackEntry> chain_copy = chain_info->chain;
    const std::size_t             pushed     = push_with_prefix_dedup(chain_copy);
    if (pushed > 0)
    {
        g_n_user_scope_inherits.fetch_add(1, std::memory_order_relaxed);
    }
    return pushed;
}

// Pops the ROCTX range, leaf frame, and snapshot frames recorded in
// observer_ctx. When count_pop is true, the ROCTX pop is added to g_n_pops.
void unwind_observer_context(const RoctxObserverContext& observer_ctx, bool count_pop)
{
    if (observer_ctx.pushed_roctx_range)
    {
        roctxRangePop();
        if (count_pop)
        {
            g_n_pops.fetch_add(1, std::memory_order_relaxed);
        }
    }
    if (observer_ctx.pushed_leaf && !g_stack.empty())
    {
        g_stack.pop_back();
    }
    for (std::size_t i = 0; i < observer_ctx.pushed_snapshot_frames && !g_stack.empty(); ++i)
    {
        g_stack.pop_back();
    }
}

std::unique_ptr<at::ObserverContext> start_cb(const at::RecordFunction& record_fn)
{
    std::unique_ptr<RoctxObserverContext> observer_ctx;
    try
    {
        observer_ctx = std::make_unique<RoctxObserverContext>();

        const at::RecordScope scope  = record_fn.scope();
        const std::int64_t    seq_nr = record_fn.seqNr();
        const char*           name   = record_fn.name();
        if (name == nullptr || name[0] == '\0')
        {
            name = "<anonymous>";
        }

        const bool stack_was_empty          = g_stack.empty();
        bool       stack_was_empty_for_leaf = stack_was_empty;

        // Apply the TLS overlay on the first record observed on this
        // thread; this re-seeds autograd workers from the main-thread
        // chain and is a no-op on the main thread.
        if (stack_was_empty)
        {
            const std::size_t overlay_frames = apply_userscope_overlay();
            observer_ctx->pushed_snapshot_frames += overlay_frames;
            if (overlay_frames > 0)
            {
                stack_was_empty_for_leaf = false;
            }
        }

        if (scope == at::RecordScope::BACKWARD_FUNCTION && seq_nr >= 0)
        {
            std::vector<StackEntry> snapshot;
            if (consume_snapshot(seq_nr, &snapshot))
            {
                observer_ctx->pushed_snapshot_frames += push_with_prefix_dedup(snapshot);
            }
        }

        StackEntry leaf;
        leaf.marker                  = name;
        const bool is_backward_scope = (scope == at::RecordScope::BACKWARD_FUNCTION);
        leaf.context = roctx_recordfn::default_leaf_context(is_backward_scope,
                                                            seq_nr,
                                                            stack_was_empty_for_leaf);
        g_stack.push_back(std::move(leaf));
        observer_ctx->pushed_leaf = true;

        if (scope == at::RecordScope::FUNCTION && seq_nr >= 0)
        {
            save_snapshot(seq_nr, g_stack);
        }

        // Emit the ROCTX range last. RecordFunction ops are torch-backed.
        std::string wire_string = build_marker_string(g_stack);
        wire_string += '|';
        wire_string += kRecordFnBackend;
        roctxRangePushA(wire_string.c_str());
        observer_ctx->pushed_roctx_range = true;
        maybe_capture(wire_string);
        g_n_pushes.fetch_add(1, std::memory_order_relaxed);
        return observer_ctx;
    }
    catch (...)
    {
        g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
        try
        {
            if (observer_ctx)
            {
                unwind_observer_context(*observer_ctx, /*count_pop=*/false);
            }
        }
        catch (...)
        {
            g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
        }
        return nullptr;
    }
}

void end_cb(const at::RecordFunction& /*record_fn*/, at::ObserverContext* obs_ctx)
{
    if (obs_ctx == nullptr)
    {
        return;
    }
    auto* observer_ctx = static_cast<RoctxObserverContext*>(obs_ctx);
    try
    {
        unwind_observer_context(*observer_ctx, /*count_pop=*/true);
    }
    catch (...)
    {
        g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

// Main-thread USER_SCOPE push. On partial failure it rolls back and
// rethrows. When non-empty, backend is appended to the range as "|<backend>".
void push_user_scope(const std::string& marker, const std::string& context, const std::string& backend)
{
    bool pushed_to_stack  = false;
    bool pushed_to_guards = false;
    bool pushed_roctx     = false;
    try
    {
        StackEntry entry;
        entry.marker  = marker;
        entry.context = context;
        g_stack.push_back(std::move(entry));
        pushed_to_stack = true;

        // Always push a guard slot (real or null) to keep g_dbg_guards
        // balanced with g_stack for pop_user_scope().
        std::unique_ptr<c10::DebugInfoGuard> guard;
        try
        {
            auto info = std::make_shared<RoctxUserScopeChain>(g_stack);
            guard     = std::make_unique<c10::DebugInfoGuard>(kRoctxDbgKind, std::move(info));
        }
        catch (...)
        {
        }
        g_dbg_guards.push_back(std::move(guard));
        pushed_to_guards = true;

        std::string wire_string = build_marker_string(g_stack);
        if (!backend.empty())
        {
            wire_string += '|';
            wire_string += backend;
        }
        roctxRangePushA(wire_string.c_str());
        pushed_roctx = true;

        maybe_capture(wire_string);
        g_n_user_scope_pushes.fetch_add(1, std::memory_order_relaxed);
        g_n_pushes.fetch_add(1, std::memory_order_relaxed);
    }
    catch (...)
    {
        g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
        try
        {
            if (pushed_roctx)
            {
                roctxRangePop();
            }
            if (pushed_to_guards && !g_dbg_guards.empty())
            {
                g_dbg_guards.pop_back();
            }
            if (pushed_to_stack && !g_stack.empty())
            {
                g_stack.pop_back();
            }
        }
        catch (...)
        {
            g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
        }
        throw;
    }
}

void pop_user_scope()
{
    try
    {
        if (g_stack.empty() || g_dbg_guards.empty())
        {
            g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        roctxRangePop();
        g_n_user_scope_pops.fetch_add(1, std::memory_order_relaxed);
        g_n_pops.fetch_add(1, std::memory_order_relaxed);
        g_stack.pop_back();
        g_dbg_guards.pop_back();
    }
    catch (...)
    {
        g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

std::int64_t install()
{
    std::lock_guard<std::mutex> lock(g_install_mutex);
    const auto                  existing = g_handle.load();
    if (existing != at::INVALID_CALLBACK_HANDLE)
    {
        return static_cast<std::int64_t>(existing);
    }
    const auto handle = at::addGlobalCallback(
        at::RecordFunctionCallback(start_cb, end_cb)
            .scopes({at::RecordScope::FUNCTION, at::RecordScope::BACKWARD_FUNCTION}));
    g_handle.store(handle);
    g_installed.store(true);
    return static_cast<std::int64_t>(handle);
}

void uninstall()
{
    std::lock_guard<std::mutex> lock(g_install_mutex);
    const auto                  handle = g_handle.exchange(at::INVALID_CALLBACK_HANDLE);
    g_installed.store(false);
    if (handle != at::INVALID_CALLBACK_HANDLE)
    {
        at::removeCallback(handle);
    }
}

bool is_installed()
{
    return g_installed.load();
}

pybind11::dict dump_stats()
{
    pybind11::dict stats_dict;
    stats_dict["installed"]           = g_installed.load();
    stats_dict["pushes"]              = g_n_pushes.load();
    stats_dict["pops"]                = g_n_pops.load();
    stats_dict["user_scope_pushes"]   = g_n_user_scope_pushes.load();
    stats_dict["user_scope_pops"]     = g_n_user_scope_pops.load();
    stats_dict["user_scope_inherits"] = g_n_user_scope_inherits.load();
    stats_dict["snapshots_saved"]     = g_n_snapshots_saved.load();
    stats_dict["snapshots_consumed"]  = g_n_snapshots_consumed.load();
    stats_dict["snapshots_dropped"]   = g_n_snapshots_dropped.load();
    stats_dict["callback_errors"]     = g_n_callback_errors.load();

    std::size_t pending = 0;
    for (auto& shard : g_shards)
    {
        std::lock_guard<std::mutex> guard(shard.mutex);
        pending += shard.snapshots.size();
    }
    stats_dict["snapshots_pending"] = pending;
    return stats_dict;
}

void start_capture()
{
    std::lock_guard<std::mutex> guard(g_capture_mutex);
    g_captured.clear();
    g_capturing.store(true, std::memory_order_release);
}

std::vector<std::string> stop_capture()
{
    g_capturing.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> guard(g_capture_mutex);
    auto                        out = g_captured;
    g_captured.clear();
    return out;
}

}  // namespace

PYBIND11_MODULE(roctx_recordfn, m)
{
    m.doc() = "ROCTX bridge for PyTorch's RecordFunction callback.";

    m.def("install", &install, "Install the global RecordFunction callback. Idempotent.");
    m.def("uninstall", &uninstall, "Remove the registered callback.");
    m.def("is_installed", &is_installed, "Return True if the callback is installed.");
    m.def("push_user_scope",
          &push_user_scope,
          pybind11::arg("marker"),
          pybind11::arg("context"),
          pybind11::arg("backend") = std::string(""),
          "Push a USER_SCOPE frame, emit a ROCTX range, publish chain into TLS DebugInfo.");
    m.def("pop_user_scope", &pop_user_scope, "Pop the most recent push_user_scope() frame on this thread.");
    m.def("dump_stats", &dump_stats, "Internal counters for tests/debugging.");
    m.def("start_capture", &start_capture, "Begin recording wire strings (test hook).");
    m.def("stop_capture", &stop_capture, "Stop and return captured wire strings.");
}
