// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
//
// ROCTX bridge for PyTorch's RecordFunction callback. Subscribes to
// FUNCTION + BACKWARD_FUNCTION and propagates the main-thread
// USER_SCOPE chain into the autograd worker via two correlation
// channels: RecordFunction::seqNr() (forward op -> backward Node)
// and c10::ThreadLocalDebugInfo (main thread -> worker scope chain).
//
// Wire format is defined by build_marker_string() below; the Python
// API is exposed by the PYBIND11_MODULE block at the bottom. Leaf
// context sentinels emitted here: aten:0, aten.nested:0,
// autograd.engine:0, autograd.bwd:0.

// torch/extension.h is intentionally avoided -- it pulls torch/all.h
// which is stripped from some ROCm nightly wheels. The narrower
// headers below cover everything we need; this contract is pinned by
// test_roctx_recordfn_avoids_torch_umbrella_header.
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
#include <unordered_map>
#include <utility>
#include <vector>

extern "C"
{
#include <rocprofiler-sdk-roctx/roctx.h>
}

namespace
{

// One stack entry: marker segment + matching context segment.
struct StackEntry
{
    std::string marker;
    std::string context;
};

// Per-RecordFunction-call state carried from start_cb to end_cb so
// end_cb only undoes what start_cb actually committed. Returning
// nullptr from start_cb on a hard failure means end_cb sees a null
// ctx and short-circuits without touching g_stack or roctx ranges
// (which is what kept push/pop counts and the marker stack drifting
// before this struct was introduced).
struct RoctxObsCtx : public at::ObserverContext
{
    bool        pushed_roctx_range     = false;
    bool        pushed_leaf            = false;
    std::size_t pushed_snapshot_frames = 0;
};

// Per-OS-thread marker stack. The autograd worker starts empty and is
// re-seeded by start_cb via the seqNr snapshot map (forward chain) and
// the TLS DebugInfo USER_SCOPE chain.
thread_local std::vector<StackEntry> g_stack;

// TLS DebugInfo subclass carrying the main-thread USER_SCOPE chain.
// Set by push_user_scope and read by start_cb on the worker so
// backward Nodes inherit their launching scope.
class RoctxUserScopeChain : public c10::DebugInfoBase
{
public:
    explicit RoctxUserScopeChain(std::vector<StackEntry> c)
        : chain(std::move(c))
    {
    }

    std::vector<StackEntry> chain;
};

// PRODUCER_INFO is universally present and unused by non-JIT scripts,
// which is the only workflow --torch-trace targets.
constexpr c10::DebugInfoKind kRoctxDbgKind = c10::DebugInfoKind::PRODUCER_INFO;

// LIFO of DebugInfoGuards driving non-RAII push/pop via RAII. Strict
// LIFO is preserved by the Python-side push/pop nesting contract.
thread_local std::vector<std::unique_ptr<c10::DebugInfoGuard>> g_dbg_guards;

// Sharded seqNr -> stack snapshot. 64 shards is past the contention
// point on parallel data-loader / worker setups.
constexpr std::size_t NUM_SHARDS = 64;

struct Shard
{
    std::mutex                                                mu;
    std::unordered_map<std::int64_t, std::vector<StackEntry>> snapshots;
    // Front = oldest seqNr; evict one entry at a time when at SHARD_SOFT_CAP.
    std::list<std::int64_t> lru_order;
};

std::array<Shard, NUM_SHARDS> g_shards;

// Soft cap per shard. Detached forwards leak entries; at the cap we
// LRU-evict one snapshot rather than clear the whole shard.
constexpr std::size_t SHARD_SOFT_CAP = 10000;

std::atomic<at::CallbackHandle> g_handle{at::INVALID_CALLBACK_HANDLE};
std::atomic<bool>               g_installed{false};
// Serializes install()/uninstall(): without it a concurrent install()
// could briefly observe (g_installed=true, g_handle=INVALID).
std::mutex g_install_mu;

// dump_stats() counters.
std::atomic<std::uint64_t> g_n_pushes{0};
std::atomic<std::uint64_t> g_n_pops{0};
std::atomic<std::uint64_t> g_n_snapshots_saved{0};
std::atomic<std::uint64_t> g_n_snapshots_consumed{0};
std::atomic<std::uint64_t> g_n_snapshots_dropped{0};
std::atomic<std::uint64_t> g_n_callback_errors{0};
std::atomic<std::uint64_t> g_n_user_scope_pushes{0};
std::atomic<std::uint64_t> g_n_user_scope_pops{0};
std::atomic<std::uint64_t> g_n_userscope_inherits{0};

// Opt-in marker capture for tests. Off by default; zero overhead when off
// (one relaxed atomic load per emitted marker).
std::atomic<bool>        g_capturing{false};
std::mutex               g_capture_mu;
std::vector<std::string> g_captured;
constexpr std::size_t    CAPTURE_CAP = 4096;

void maybe_capture(const std::string& s)
{
    if (!g_capturing.load(std::memory_order_relaxed))
        return;
    std::lock_guard<std::mutex> guard(g_capture_mu);
    if (g_captured.size() < CAPTURE_CAP)
    {
        g_captured.push_back(s);
    }
}

// Leaf context label classifier. See file header for the label catalogue.
const char* default_leaf_context(at::RecordScope scope, std::int64_t seq, bool stack_was_empty)
{
    if (scope == at::RecordScope::BACKWARD_FUNCTION)
    {
        return (seq >= 0) ? "#1@autograd.bwd:0" : "#1@autograd.engine:0";
    }
    // at::RecordScope::FUNCTION.
    return stack_was_empty ? "#1@aten:0" : "#1@aten.nested:0";
}

// Assemble the wire string consumed by utils_analysis.py:
//   "<marker1>/<marker2>/.../<markerN>:<context1>/.../<contextN>"
// The two halves are split on the literal ":#" by the analyzer; each
// context segment is "#<call_count>@<location>" where <location> is
// either a user "file:line" or a leaf sentinel (see default_leaf_context).
std::string build_marker_string(const std::vector<StackEntry>& stack)
{
    std::size_t marker_len = 0;
    std::size_t ctx_len    = 0;
    for (const auto& e : stack)
    {
        marker_len += e.marker.size() + 1;
        ctx_len += e.context.size() + 1;
    }
    std::string out;
    out.reserve(marker_len + ctx_len + 1);

    bool first = true;
    for (const auto& e : stack)
    {
        if (!first)
            out += '/';
        out += e.marker;
        first = false;
    }
    out += ':';
    first = true;
    for (const auto& e : stack)
    {
        if (!first)
            out += '/';
        out += e.context;
        first = false;
    }
    return out;
}

void lru_remove(Shard& shard, std::int64_t seq)
{
    shard.lru_order.remove(seq);
}

void lru_touch(Shard& shard, std::int64_t seq)
{
    lru_remove(shard, seq);
    shard.lru_order.push_back(seq);
}

void evict_oldest_snapshot(Shard& shard)
{
    if (shard.lru_order.empty())
        return;
    const std::int64_t oldest = shard.lru_order.front();
    shard.lru_order.pop_front();
    shard.snapshots.erase(oldest);
    g_n_snapshots_dropped.fetch_add(1, std::memory_order_relaxed);
}

void save_snapshot(std::int64_t seq, const std::vector<StackEntry>& stack)
{
    auto&                       shard = g_shards[static_cast<std::size_t>(seq) % NUM_SHARDS];
    std::lock_guard<std::mutex> guard(shard.mu);
    auto                        it = shard.snapshots.find(seq);
    if (it != shard.snapshots.end())
    {
        it->second = stack;
        lru_touch(shard, seq);
        g_n_snapshots_saved.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    while (shard.snapshots.size() >= SHARD_SOFT_CAP)
    {
        evict_oldest_snapshot(shard);
    }
    shard.snapshots.emplace(seq, stack);
    shard.lru_order.push_back(seq);
    g_n_snapshots_saved.fetch_add(1, std::memory_order_relaxed);
}

bool consume_snapshot(std::int64_t seq, std::vector<StackEntry>* out)
{
    auto&                       shard = g_shards[static_cast<std::size_t>(seq) % NUM_SHARDS];
    std::lock_guard<std::mutex> guard(shard.mu);
    auto                        it = shard.snapshots.find(seq);
    if (it == shard.snapshots.end())
        return false;
    *out = std::move(it->second);
    shard.snapshots.erase(it);
    lru_remove(shard, seq);
    g_n_snapshots_consumed.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// Push entries from `chain` onto g_stack, skipping any leading entries
// that are byte-identical (marker + context) to g_stack's current
// prefix. Matching on BOTH fields avoids spurious dedup when the user
// happens to push two distinct scopes that share a marker name but
// differ in call-site context. Returns the number of entries pushed.
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

// Layer the backward-time USER_SCOPE chain (read from TLS DebugInfo) on
// top of whatever is currently in g_stack. Called only at the start of
// a backward subtree on this worker (stack_was_empty == true), so the
// dedup inside push_with_prefix_dedup is defensive: g_stack is empty
// in current usage.
std::size_t apply_userscope_overlay()
{
    auto* base       = c10::ThreadLocalDebugInfo::get(kRoctxDbgKind);
    auto* chain_info = dynamic_cast<const RoctxUserScopeChain*>(base);
    if (chain_info == nullptr || chain_info->chain.empty())
    {
        return 0;
    }
    const std::size_t pushed = push_with_prefix_dedup(chain_info->chain);
    if (pushed > 0)
    {
        g_n_userscope_inherits.fetch_add(1, std::memory_order_relaxed);
    }
    return pushed;
}

std::unique_ptr<at::ObserverContext> start_cb(const at::RecordFunction& fn)
{
    // Allocated before the try so the catch can roll back any partial
    // commits captured in ctx->pushed_* flags. If allocation itself
    // throws, ctx is null and the catch only bumps callback_errors.
    std::unique_ptr<RoctxObsCtx> ctx;
    try
    {
        ctx = std::make_unique<RoctxObsCtx>();

        const at::RecordScope scope = fn.scope();
        const std::int64_t    seq   = fn.seqNr();
        const char*           name  = fn.name();
        if (name == nullptr || name[0] == '\0')
        {
            name = "<anonymous>";
        }

        const bool stack_was_empty = g_stack.empty();
        // Tracks g_stack-emptiness at leaf-context-computation time
        // (post-overlay/snapshot). Drives the aten:0 vs aten.nested:0
        // choice in default_leaf_context().
        bool stack_was_empty_for_leaf = stack_was_empty;

        // Apply the TLS USER_SCOPE overlay on the FIRST record on this
        // thread, regardless of scope. This covers the outer
        // FUNCTION-scope autograd evaluator (which wraps each backward
        // Node body) as well as the inner BACKWARD_FUNCTION record;
        // gating on BACKWARD_FUNCTION only would orphan the evaluator
        // and lose the loss.backward() launching scope. Idempotent
        // within a subtree (dedup against non-empty g_stack is a no-op).
        // No-op on the main thread by construction: push_user_scope
        // pushes to BOTH g_stack and TLS, so reaching this branch with
        // a published TLS overlay implies a worker thread.
        if (stack_was_empty)
        {
            const std::size_t overlay_frames = apply_userscope_overlay();
            ctx->pushed_snapshot_frames += overlay_frames;
            if (overlay_frames > 0)
            {
                stack_was_empty_for_leaf = false;
            }
        }

        // Backward Node body: restore the forward chain via the
        // seqNr-keyed snapshot. Dedup avoids duplicating frames shared
        // with the TLS overlay or the outer evaluator's leaf.
        if (scope == at::RecordScope::BACKWARD_FUNCTION && seq >= 0)
        {
            std::vector<StackEntry> snapshot;
            if (consume_snapshot(seq, &snapshot))
            {
                ctx->pushed_snapshot_frames += push_with_prefix_dedup(snapshot);
            }
        }

        // Push the leaf. USER_SCOPE pushes use push_user_scope().
        StackEntry leaf;
        leaf.marker  = name;
        leaf.context = default_leaf_context(scope, seq, stack_was_empty_for_leaf);
        g_stack.push_back(std::move(leaf));
        ctx->pushed_leaf = true;

        // Forward op with seqNr -> save the whole stack (including
        // USER_SCOPE parents) for the matching backward Node.
        if (scope == at::RecordScope::FUNCTION && seq >= 0)
        {
            save_snapshot(seq, g_stack);
        }

        // ROCTX emit is the last side effect so any earlier failure
        // rolls back without an unmatched roctxRangePop in end_cb.
        const std::string full = build_marker_string(g_stack);
        roctxRangePushA(full.c_str());
        ctx->pushed_roctx_range = true;
        maybe_capture(full);
        g_n_pushes.fetch_add(1, std::memory_order_relaxed);
        return ctx;
    }
    catch (...)
    {
        g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
        // Best-effort rollback of whatever start_cb committed before
        // the throw so end_cb (which still runs with a null obs ctx)
        // doesn't compound the damage.
        try
        {
            if (ctx)
            {
                if (ctx->pushed_roctx_range)
                {
                    roctxRangePop();
                }
                if (ctx->pushed_leaf && !g_stack.empty())
                {
                    g_stack.pop_back();
                }
                for (std::size_t i = 0; i < ctx->pushed_snapshot_frames && !g_stack.empty(); ++i)
                {
                    g_stack.pop_back();
                }
            }
        }
        catch (...)
        {
            g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
        }
        // Null obs ctx signals end_cb to skip every pop.
        return nullptr;
    }
}

void end_cb(const at::RecordFunction& /*fn*/, at::ObserverContext* obs_ctx)
{
    // start_cb returned nullptr on a hard failure (or never started a
    // matching push); skip every pop so counters and g_stack stay
    // paired with what was actually committed.
    if (obs_ctx == nullptr)
    {
        return;
    }
    auto* ctx = static_cast<RoctxObsCtx*>(obs_ctx);
    try
    {
        if (ctx->pushed_roctx_range)
        {
            roctxRangePop();
            g_n_pops.fetch_add(1, std::memory_order_relaxed);
        }
        if (ctx->pushed_leaf && !g_stack.empty())
        {
            g_stack.pop_back();
        }
        for (std::size_t i = 0; i < ctx->pushed_snapshot_frames && !g_stack.empty(); ++i)
        {
            g_stack.pop_back();
        }
    }
    catch (...)
    {
        g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

// Python-side USER_SCOPE push from main-thread structural wraps. A
// dedicated entry is needed because RecordFunction::name() can't carry
// a context segment, and the autograd worker needs the chain via TLS.
//
// All-or-nothing: on any partial failure we roll back every committed
// side effect (g_stack, g_dbg_guards, roctx range) and rethrow so the
// Python caller can fall through to the Python tier without
// double-pushing into a half-mutated C++ stack.
void push_user_scope(const std::string& marker, const std::string& context)
{
    bool pushed_to_stack  = false;
    bool pushed_to_guards = false;
    bool pushed_roctx     = false;
    try
    {
        StackEntry e;
        e.marker  = marker;
        e.context = context;
        g_stack.push_back(std::move(e));
        pushed_to_stack = true;

        // Publish into TLS DebugInfo; the guard restores on pop. We
        // push a unique_ptr onto g_dbg_guards in either case (real
        // guard or null sentinel) so pop_user_scope's strict pair-wise
        // pop with g_stack stays balanced.
        std::unique_ptr<c10::DebugInfoGuard> guard;
        try
        {
            auto info = std::make_shared<RoctxUserScopeChain>(g_stack);
            guard     = std::make_unique<c10::DebugInfoGuard>(kRoctxDbgKind, std::move(info));
        }
        catch (...)
        {
            // Swallowed: missing TLS publication only degrades worker
            // overlay; the roctx range itself still emits.
        }
        g_dbg_guards.push_back(std::move(guard));
        pushed_to_guards = true;

        const std::string full = build_marker_string(g_stack);
        roctxRangePushA(full.c_str());
        pushed_roctx = true;

        maybe_capture(full);
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
        // Rethrow so Python's _push_scope falls through to the Python
        // tier instead of treating the failed push as a success.
        throw;
    }
}

void pop_user_scope()
{
    try
    {
        // Strict pair-wise pop: g_stack and g_dbg_guards are pushed
        // together by push_user_scope, so an imbalance here is a
        // programmer error (unmatched push/pop). Silently no-op rather
        // than throw, to avoid masking user exceptions in Python
        // finally blocks; bump callback_errors so the imbalance is
        // observable in dump_stats and the end-of-run warning.
        if (g_stack.empty() || g_dbg_guards.empty())
        {
            g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
            return;
        }
        roctxRangePop();
        g_n_user_scope_pops.fetch_add(1, std::memory_order_relaxed);
        g_n_pops.fetch_add(1, std::memory_order_relaxed);
        g_stack.pop_back();
        // unique_ptr dtor restores the previous TLS DebugInfo head
        // (no-op for null sentinels pushed when TLS publish failed).
        g_dbg_guards.pop_back();
    }
    catch (...)
    {
        g_n_callback_errors.fetch_add(1, std::memory_order_relaxed);
    }
}

std::int64_t install()
{
    // Serialize on g_install_mu so a second install() observes the
    // existing valid handle and skips re-registration. g_handle is
    // only published after addThreadLocalCallback returns.
    std::lock_guard<std::mutex> lock(g_install_mu);
    const auto                  existing = g_handle.load();
    if (existing != at::INVALID_CALLBACK_HANDLE)
    {
        return static_cast<std::int64_t>(existing);
    }
    const auto handle = at::addThreadLocalCallback(
        at::RecordFunctionCallback(start_cb, end_cb)
            .scopes({at::RecordScope::FUNCTION, at::RecordScope::BACKWARD_FUNCTION}));
    g_handle.store(handle);
    g_installed.store(true);
    return static_cast<std::int64_t>(handle);
}

void uninstall()
{
    // exchange()+store under g_install_mu so a concurrent install()
    // either sees the cleared handle (and re-registers) or runs
    // strictly after this critical section.
    std::lock_guard<std::mutex> lock(g_install_mu);
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
    pybind11::dict d;
    d["installed"]           = g_installed.load();
    d["pushes"]              = g_n_pushes.load();
    d["pops"]                = g_n_pops.load();
    d["user_scope_pushes"]   = g_n_user_scope_pushes.load();
    d["user_scope_pops"]     = g_n_user_scope_pops.load();
    d["user_scope_inherits"] = g_n_userscope_inherits.load();
    d["snapshots_saved"]     = g_n_snapshots_saved.load();
    d["snapshots_consumed"]  = g_n_snapshots_consumed.load();
    d["snapshots_dropped"]   = g_n_snapshots_dropped.load();
    d["callback_errors"]     = g_n_callback_errors.load();

    std::size_t pending = 0;
    for (auto& shard : g_shards)
    {
        std::lock_guard<std::mutex> guard(shard.mu);
        pending += shard.snapshots.size();
    }
    d["snapshots_pending"] = pending;
    return d;
}

void start_capture()
{
    std::lock_guard<std::mutex> guard(g_capture_mu);
    g_captured.clear();
    g_capturing.store(true, std::memory_order_release);
}

std::vector<std::string> stop_capture()
{
    g_capturing.store(false, std::memory_order_release);
    std::lock_guard<std::mutex> guard(g_capture_mu);
    auto                        out = g_captured;
    g_captured.clear();
    return out;
}

}  // namespace

PYBIND11_MODULE(roctx_recordfn, m)
{
    m.doc() = "ROCTX bridge for PyTorch's RecordFunction callback. "
              "See build_marker_string() in roctx_recordfn.cpp for the "
              "wire-format contract and the file header for the two-channel "
              "correlation (seqNr + TLS DebugInfo) it implements.";

    m.def("install",
          &install,
          "Install the thread-local RecordFunction callback. Returns the "
          "CallbackHandle; on a second call returns the existing handle "
          "(never 0) without re-registering.");
    m.def("uninstall", &uninstall, "Remove the registered callback.");
    m.def("is_installed", &is_installed, "True if the callback is installed.");
    m.def("push_user_scope",
          &push_user_scope,
          pybind11::arg("marker"),
          pybind11::arg("context"),
          "Push a USER_SCOPE-equivalent frame, emit a ROCTX range, and "
          "publish the current chain into TLS DebugInfo so the autograd "
          "worker can inherit it.");
    m.def("pop_user_scope",
          &pop_user_scope,
          "Pop the most recent push_user_scope() frame on this thread "
          "(also restores the previous TLS DebugInfo head).");
    m.def("dump_stats", &dump_stats, "Return internal counters for tests and debugging.");
    m.def("start_capture", &start_capture, "Begin recording every assembled wire string (test hook).");
    m.def("stop_capture", &stop_capture, "Stop recording and return the captured wire strings.");
}
