#pragma once

/**
 * @file temporal_diff_filter.hpp
 * @brief Per-tid running stack tracker that sits between kernel_drain
 *        and the frame_trie on the v1 streaming hot path.
 *
 * Detects top-down divergence between successive samples on the same
 * tid and walks the trie only for the changed suffix. Steady-state
 * cost on a sample whose stack matches the previous one is zero trie
 * ops + zero allocations; this is the dominant case at high sampling
 * frequency and the reason the ~100 ns per-sample userspace budget is
 * reachable.
 *
 * Header-only: the per-sample diff() is small and hot, inlining buys
 * real cycles.
 */

#include <cstddef>
#include <cstdint>
#include <span>
#include <sys/types.h>
#include <vector>

#include "cpu_snap/sampling/flat_tid_map.hpp"
#include "cpu_snap/sampling/frame_trie.hpp"

namespace cpu_snap::sampling::internal
{

/**
 * @brief Per-tid divergence detector over a shared frame_trie.
 *
 * For each (tid, frames) sample, computes the longest common prefix
 * (root-ward) against the prior path on the same tid and only interns
 * the divergent suffix into the trie. The returned diff_result reports
 * the leaf node id plus a span of trie nodes that were newly allocated
 * on this call. The caller MUST emit those new nodes as NEW_NODE
 * records BEFORE emitting the SAMPLE that references leaf_id, so that
 * a downstream reader rebuilding the trie by forward replay sees
 * parents before children.
 *
 * @note Instances are NOT thread-safe; the streaming pipeline owns one
 *       instance accessed only from its single worker thread.
 */
class temporal_diff_filter
{
public:
    using node_id = frame_trie::node_id;

    /**
     * @brief One sample's worth of output.
     *
     * `new_nodes` is a view into scratch storage owned by the filter
     * and is valid only until the next call to diff() on the same
     * instance. Callers that need to retain the new-node list across
     * diff() calls must copy it out.
     */
    struct diff_result
    {
        node_id                                leaf_id = frame_trie::k_root;
        std::span<const frame_trie::node_view> new_nodes;
    };

    /**
     * @brief Construct a filter that interns into @p trie.
     * @param trie Frame trie to intern into. Must outlive the filter.
     */
    explicit temporal_diff_filter(frame_trie& trie) noexcept
    : m_trie{ &trie }
    {}

    ~temporal_diff_filter() = default;

    temporal_diff_filter(const temporal_diff_filter&)            = delete;
    temporal_diff_filter& operator=(const temporal_diff_filter&) = delete;
    temporal_diff_filter(temporal_diff_filter&&)                 = default;
    temporal_diff_filter& operator=(temporal_diff_filter&&)      = default;

    /**
     * @brief Diff @p frames against the prior path on @p tid and intern
     *        the divergent suffix into the trie.
     *
     * @param tid    Linux thread id; per-tid state is keyed on this.
     * @param frames Sample frames, innermost-first (frames[0] is the
     *               leaf, frames[n-1] is the outermost frame closest
     *               to main).
     * @return       leaf_id for this sample + span of newly allocated
     *               trie nodes in root-ward order (parents first). On
     *               empty @p frames, returns { k_root, {} } and leaves
     *               the per-tid state untouched.
     *
     * @note The returned span is invalidated by the next diff() call.
     * @note Steady-state (frames identical to prior sample on same tid):
     *       zero trie ops, zero allocations, empty new_nodes span.
     */
    diff_result diff(pid_t tid, std::span<const std::uint64_t> frames)
    {
        m_scratch_new_nodes.clear();

        if(frames.empty()) return { frame_trie::k_root, {} };

        per_tid_state& state = m_per_tid[tid];

        const std::size_t new_depth   = frames.size();
        const std::size_t prior_depth = state.path_ips.size();
        const std::size_t comparable =
            (new_depth < prior_depth) ? new_depth : prior_depth;

        // Walk outermost -> innermost on both sides. The new sample's
        // outermost frame is frames[new_depth - 1]; the prior path's
        // outermost frame is path_ips[0].
        std::size_t common_prefix = 0;
        while(common_prefix < comparable &&
              frames[new_depth - 1 - common_prefix] == state.path_ips[common_prefix])
        {
            ++common_prefix;
        }

        state.path_nodes.resize(common_prefix);
        state.path_ips.resize(common_prefix);

        node_id parent_id =
            common_prefix == 0 ? frame_trie::k_root : state.path_nodes.back();

        for(std::size_t depth = common_prefix; depth < new_depth; ++depth)
        {
            const std::uint64_t ip     = frames[new_depth - 1 - depth];
            const auto          result = m_trie->intern(parent_id, ip);
            if(result.was_new) m_scratch_new_nodes.push_back(m_trie->node_at(result.id));
            state.path_nodes.push_back(result.id);
            state.path_ips.push_back(ip);
            parent_id = result.id;
        }

        return { parent_id,
                 std::span<const frame_trie::node_view>{ m_scratch_new_nodes.data(),
                                                         m_scratch_new_nodes.size() } };
    }

    /**
     * @brief Drop per-tid state for @p tid.
     *
     * Use when a tid exits to avoid stale state polluting a recycled tid.
     * @param tid Linux thread id whose state should be dropped.
     */
    void forget_tid(pid_t tid) { m_per_tid.erase(tid); }

    /**
     * @brief Number of tids currently tracked.
     * @return Tracked-tid count.
     */
    [[nodiscard]] std::size_t tracked_tids() const noexcept { return m_per_tid.size(); }

    /**
     * @brief Trie leaf last observed for @p tid.
     * @param tid Linux thread id to query.
     * @return The trie leaf id last observed for @p tid, or k_root if
     *         the tid has not been seen.
     */
    [[nodiscard]] node_id current_leaf_for_tid(pid_t tid) const noexcept
    {
        const auto* state = m_per_tid.find(tid);
        if(state == nullptr || state->path_nodes.empty())
        {
            return frame_trie::k_root;
        }
        return state->path_nodes.back();
    }

private:
    struct per_tid_state
    {
        std::vector<node_id>       path_nodes;  // outermost -> innermost
        std::vector<std::uint64_t> path_ips;    // parallel to path_nodes
    };

    frame_trie*                                      m_trie = nullptr;
    cpu_snap::sampling::flat_tid_map<per_tid_state> m_per_tid;
    std::vector<frame_trie::node_view>               m_scratch_new_nodes;
};

}  // namespace cpu_snap::sampling::internal
