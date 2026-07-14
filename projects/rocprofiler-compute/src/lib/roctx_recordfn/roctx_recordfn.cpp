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

// Per-thread marker state. Autograd workers start empty and are re-seeded
// from the seqNr snapshot and the TLS USER_SCOPE chain. guards is the LIFO
// of DebugInfoGuards mirroring push_user_scope/pop_user_scope.
struct ThreadState
{
    std::vector<StackEntry>                           stack;
    std::vector<std::unique_ptr<c10::DebugInfoGuard>> guards;
};

thread_local ThreadState g_thread;

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

// Runtime counters exposed through dump_stats().
struct Stats
{
    std::atomic<std::uint64_t> pushes{0};
    std::atomic<std::uint64_t> pops{0};
    std::atomic<std::uint64_t> snapshots_saved{0};
    std::atomic<std::uint64_t> snapshots_consumed{0};
    std::atomic<std::uint64_t> snapshots_dropped{0};
    std::atomic<std::uint64_t> callback_errors{0};
    std::atomic<std::uint64_t> user_scope_pushes{0};
    std::atomic<std::uint64_t> user_scope_pops{0};
    std::atomic<std::uint64_t> user_scope_inherits{0};
};

Stats g_stats;

void inc(std::atomic<std::uint64_t>& counter)
{
    counter.fetch_add(1, std::memory_order_relaxed);
}

// Sharded seqNr -> forward-stack snapshot store. Snapshots whose backward
// never runs are evicted per shard in LRU order.
class SnapshotStore
{
public:
    static constexpr std::size_t kNumShards    = 64;
    static constexpr std::size_t kShardSoftCap = 10000;

    void save(std::int64_t seq_nr, const std::vector<StackEntry>& stack)
    {
        Shard&                      shard = shard_for(seq_nr);
        std::lock_guard<std::mutex> guard(shard.mutex);
        auto                        it = shard.snapshots.find(seq_nr);
        if (it != shard.snapshots.end())
        {
            it->second = stack;
            lru_touch(shard, seq_nr);
            inc(g_stats.snapshots_saved);
            return;
        }
        while (shard.snapshots.size() >= kShardSoftCap)
        {
            evict_oldest(shard);
        }
        shard.snapshots.emplace(seq_nr, stack);
        lru_touch(shard, seq_nr);
        inc(g_stats.snapshots_saved);
    }

    bool consume(std::int64_t seq_nr, std::vector<StackEntry>* out_stack)
    {
        Shard&                      shard = shard_for(seq_nr);
        std::lock_guard<std::mutex> guard(shard.mutex);
        auto                        it = shard.snapshots.find(seq_nr);
        if (it == shard.snapshots.end())
            return false;
        *out_stack = std::move(it->second);
        shard.snapshots.erase(it);
        lru_remove(shard, seq_nr);
        inc(g_stats.snapshots_consumed);
        return true;
    }

    std::size_t pending()
    {
        std::size_t total = 0;
        for (auto& shard : shards_)
        {
            std::lock_guard<std::mutex> guard(shard.mutex);
            total += shard.snapshots.size();
        }
        return total;
    }

    void clear()
    {
        for (auto& shard : shards_)
        {
            std::lock_guard<std::mutex> guard(shard.mutex);
            shard.snapshots.clear();
            shard.lru_order.clear();
            shard.lru_idx.clear();
        }
    }

private:
    struct Shard
    {
        std::mutex                                                          mutex;
        std::unordered_map<std::int64_t, std::vector<StackEntry>>           snapshots;
        std::list<std::int64_t>                                             lru_order;
        std::unordered_map<std::int64_t, std::list<std::int64_t>::iterator> lru_idx;
    };

    Shard& shard_for(std::int64_t seq_nr)
    {
        return shards_[static_cast<std::size_t>(seq_nr) % kNumShards];
    }

    static void lru_remove(Shard& shard, std::int64_t seq_nr)
    {
        auto it = shard.lru_idx.find(seq_nr);
        if (it == shard.lru_idx.end())
            return;
        shard.lru_order.erase(it->second);
        shard.lru_idx.erase(it);
    }

    static void lru_touch(Shard& shard, std::int64_t seq_nr)
    {
        lru_remove(shard, seq_nr);
        shard.lru_order.push_back(seq_nr);
        auto tail = shard.lru_order.end();
        --tail;
        shard.lru_idx.emplace(seq_nr, tail);
    }

    static void evict_oldest(Shard& shard)
    {
        if (shard.lru_order.empty())
            return;
        const std::int64_t oldest = shard.lru_order.front();
        shard.lru_order.pop_front();
        shard.lru_idx.erase(oldest);
        shard.snapshots.erase(oldest);
        inc(g_stats.snapshots_dropped);
    }

    std::array<Shard, kNumShards> shards_;
};

SnapshotStore g_snapshots;

// Opt-in buffer of emitted wire strings, used by the test capture hook.
class CaptureBuffer
{
public:
    void capture(const std::string& wire_string)
    {
        if (!capturing_.load(std::memory_order_relaxed))
            return;
        std::lock_guard<std::mutex> guard(mutex_);
        if (captured_.size() < kCap)
        {
            captured_.push_back(wire_string);
        }
    }

    void start()
    {
        std::lock_guard<std::mutex> guard(mutex_);
        captured_.clear();
        capturing_.store(true, std::memory_order_release);
    }

    std::vector<std::string> stop()
    {
        capturing_.store(false, std::memory_order_release);
        std::lock_guard<std::mutex> guard(mutex_);
        std::vector<std::string>    out = captured_;
        captured_.clear();
        return out;
    }

private:
    static constexpr std::size_t kCap = 4096;

    std::atomic<bool>        capturing_{false};
    std::mutex               mutex_;
    std::vector<std::string> captured_;
};

CaptureBuffer g_capture;

// Global RecordFunction callback registration state.
struct InstallState
{
    std::atomic<at::CallbackHandle> handle{at::INVALID_CALLBACK_HANDLE};
    std::atomic<bool>               installed{false};
    std::mutex                      mutex;
};

InstallState g_install;

// The RecordFunction tier instruments PyTorch ATen operators.
constexpr const char* kRecordFnBackend = "torch";

// Percent-encoding of the two characters that would otherwise collide with the
// marker-path grammar. The inverse decode lives with the Python readers
// (utils/inject_roctx/core.py decode_marker_name, utils/utils_analysis.py).
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

// Pushes `chain` onto the thread stack, skipping any leading prefix that is
// already present. Returns the number of frames pushed.
std::size_t push_with_prefix_dedup(const std::vector<StackEntry>& chain)
{
    const std::size_t maxc   = std::min(chain.size(), g_thread.stack.size());
    std::size_t       common = 0;
    for (; common < maxc; ++common)
    {
        if (chain[common].marker != g_thread.stack[common].marker ||
            chain[common].context != g_thread.stack[common].context)
        {
            break;
        }
    }
    std::size_t pushed = 0;
    for (std::size_t i = common; i < chain.size(); ++i)
    {
        g_thread.stack.push_back(chain[i]);
        ++pushed;
    }
    return pushed;
}

// Overlays the published USER_SCOPE chain onto the thread stack.
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
        inc(g_stats.user_scope_inherits);
    }
    return pushed;
}

// Pops the ROCTX range, leaf frame, and snapshot frames recorded in
// observer_ctx. When count_pop is true, the ROCTX pop is added to g_stats.pops.
void unwind_observer_context(const RoctxObserverContext& observer_ctx, bool count_pop)
{
    if (observer_ctx.pushed_roctx_range)
    {
        roctxRangePop();
        if (count_pop)
        {
            inc(g_stats.pops);
        }
    }
    if (observer_ctx.pushed_leaf && !g_thread.stack.empty())
    {
        g_thread.stack.pop_back();
    }
    for (std::size_t i = 0; i < observer_ctx.pushed_snapshot_frames && !g_thread.stack.empty(); ++i)
    {
        g_thread.stack.pop_back();
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

        const bool stack_was_empty          = g_thread.stack.empty();
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
            if (g_snapshots.consume(seq_nr, &snapshot))
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
        g_thread.stack.push_back(std::move(leaf));
        observer_ctx->pushed_leaf = true;

        if (scope == at::RecordScope::FUNCTION && seq_nr >= 0)
        {
            g_snapshots.save(seq_nr, g_thread.stack);
        }

        // Emit the ROCTX range last. RecordFunction ops are torch-backed.
        std::string wire_string = build_marker_string(g_thread.stack);
        wire_string += '|';
        wire_string += kRecordFnBackend;
        roctxRangePushA(wire_string.c_str());
        observer_ctx->pushed_roctx_range = true;
        g_capture.capture(wire_string);
        inc(g_stats.pushes);
        return observer_ctx;
    }
    catch (...)
    {
        inc(g_stats.callback_errors);
        try
        {
            if (observer_ctx)
            {
                unwind_observer_context(*observer_ctx, /*count_pop=*/false);
            }
        }
        catch (...)
        {
            inc(g_stats.callback_errors);
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
        inc(g_stats.callback_errors);
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
        g_thread.stack.push_back(std::move(entry));
        pushed_to_stack = true;

        // Always push a guard slot (real or null) to keep guards
        // balanced with the stack for pop_user_scope().
        std::unique_ptr<c10::DebugInfoGuard> guard;
        try
        {
            auto info = std::make_shared<RoctxUserScopeChain>(g_thread.stack);
            guard     = std::make_unique<c10::DebugInfoGuard>(kRoctxDbgKind, std::move(info));
        }
        catch (...)
        {
        }
        g_thread.guards.push_back(std::move(guard));
        pushed_to_guards = true;

        std::string wire_string = build_marker_string(g_thread.stack);
        if (!backend.empty())
        {
            wire_string += '|';
            wire_string += backend;
        }
        roctxRangePushA(wire_string.c_str());
        pushed_roctx = true;

        g_capture.capture(wire_string);
        inc(g_stats.user_scope_pushes);
        inc(g_stats.pushes);
    }
    catch (...)
    {
        inc(g_stats.callback_errors);
        try
        {
            if (pushed_roctx)
            {
                roctxRangePop();
            }
            if (pushed_to_guards && !g_thread.guards.empty())
            {
                g_thread.guards.pop_back();
            }
            if (pushed_to_stack && !g_thread.stack.empty())
            {
                g_thread.stack.pop_back();
            }
        }
        catch (...)
        {
            inc(g_stats.callback_errors);
        }
        throw;
    }
}

void pop_user_scope()
{
    try
    {
        if (g_thread.stack.empty() || g_thread.guards.empty())
        {
            inc(g_stats.callback_errors);
            return;
        }
        roctxRangePop();
        inc(g_stats.user_scope_pops);
        inc(g_stats.pops);
        g_thread.stack.pop_back();
        g_thread.guards.pop_back();
    }
    catch (...)
    {
        inc(g_stats.callback_errors);
    }
}

std::int64_t install()
{
    std::lock_guard<std::mutex> lock(g_install.mutex);
    const auto                  existing = g_install.handle.load();
    if (existing != at::INVALID_CALLBACK_HANDLE)
    {
        return static_cast<std::int64_t>(existing);
    }
    const auto handle = at::addGlobalCallback(
        at::RecordFunctionCallback(start_cb, end_cb)
            .scopes({at::RecordScope::FUNCTION, at::RecordScope::BACKWARD_FUNCTION}));
    g_install.handle.store(handle);
    g_install.installed.store(true);
    return static_cast<std::int64_t>(handle);
}

void uninstall()
{
    std::lock_guard<std::mutex> lock(g_install.mutex);
    const auto                  handle = g_install.handle.exchange(at::INVALID_CALLBACK_HANDLE);
    g_install.installed.store(false);
    if (handle != at::INVALID_CALLBACK_HANDLE)
    {
        at::removeCallback(handle);
    }
}

bool is_installed()
{
    return g_install.installed.load();
}

pybind11::dict dump_stats()
{
    pybind11::dict stats_dict;
    stats_dict["installed"]           = g_install.installed.load();
    stats_dict["pushes"]              = g_stats.pushes.load();
    stats_dict["pops"]                = g_stats.pops.load();
    stats_dict["user_scope_pushes"]   = g_stats.user_scope_pushes.load();
    stats_dict["user_scope_pops"]     = g_stats.user_scope_pops.load();
    stats_dict["user_scope_inherits"] = g_stats.user_scope_inherits.load();
    stats_dict["snapshots_saved"]     = g_stats.snapshots_saved.load();
    stats_dict["snapshots_consumed"]  = g_stats.snapshots_consumed.load();
    stats_dict["snapshots_dropped"]   = g_stats.snapshots_dropped.load();
    stats_dict["callback_errors"]     = g_stats.callback_errors.load();
    stats_dict["snapshots_pending"]   = g_snapshots.pending();
    return stats_dict;
}

void start_capture()
{
    g_capture.start();
}

std::vector<std::string> stop_capture()
{
    return g_capture.stop();
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
