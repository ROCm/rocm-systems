#pragma once

/**
 * @file frame_trie.hpp
 * @brief Grow-only Flyweight trie of callstack frames for the v1
 *        streaming pipeline.
 *
 * Each unique (parent_node_id, ip) edge maps to a single node_id;
 * intern() either returns an existing id or allocates a new one.
 * Heart of L4 compression: 99%+ of samples reuse an existing leaf so
 * the per-sample wire cost collapses to a single u32.
 *
 * Hash-table choice: std::unordered_map keyed on a 64-bit mix of
 * (parent_id, ip). Stdlib hash is robust and the projected hot-path
 * rate (~1.28 M samples/s x ~2 ops amortized) is well within reach;
 * if bench in PR-12 flags this as the bottleneck, swap to a flat
 * open-addressed map then.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "cpu_snap/error.hpp"

namespace cpu_snap::sampling::internal
{

/**
 * @brief Persistent grow-only trie of callstack frames, keyed on
 *        (parent_node_id, instruction_pointer) edges.
 *
 * Used by temporal_diff_filter to compress recurrent stack paths
 * down to a single leaf id per sample.
 *
 * @note Thread compatibility: instances are NOT thread-safe; the
 *       streaming pipeline owns one instance accessed only from its
 *       single worker thread.
 * @note Node id 0 is reserved for the implicit trie root. The first
 *       allocated node has id 1.
 */
class frame_trie
{
public:
    using node_id = std::uint32_t;

    // NOLINTNEXTLINE(readability-identifier-naming): codebase k_*-prefix convention
    static constexpr node_id k_root = 0;

    /** @brief Result of intern(): the (possibly newly allocated) id and a flag. */
    struct intern_result
    {
        /** Stable id of the (possibly newly allocated) node. */
        node_id id      = k_root;
        /** True iff intern() allocated a new node; false on a cache hit. */
        bool    was_new = false;
    };

    /** @brief Reflection view of an allocated node. parent_id == k_root means root child. */
    struct node_view
    {
        /** Stable id of this node. */
        node_id       id        = k_root;
        /** Parent node id; k_root for direct children of the implicit root. */
        node_id       parent_id = k_root;
        /** Instruction pointer stored at this node. */
        std::uint64_t ip        = 0;
    };

    frame_trie()  = default;
    ~frame_trie() = default;

    frame_trie(const frame_trie&)            = delete;
    frame_trie& operator=(const frame_trie&) = delete;
    frame_trie(frame_trie&&)                 = default;
    frame_trie& operator=(frame_trie&&)      = default;

    /**
     * @brief Return the node id for the (parent_id, ip) edge, allocating
     *        a new node if the edge has not been seen before.
     *
     * Ordering contract: when `was_new == true` the caller MUST emit a
     * NEW_NODE record for the returned id BEFORE emitting any SAMPLE
     * that references the id downstream. Readers rebuild the trie by
     * forward replay and will reject a SAMPLE that names an unknown
     * node.
     *
     * @param parent_id Parent node id, or k_root for a top-of-stack frame.
     * @param ip        Instruction pointer that labels this edge.
     * @return Pair { id, was_new }. Returned id is always >= 1.
     *
     * @throws cpu_snap::resource_error iff a non-zero cap set via
     *         set_max_nodes() would be exceeded by the allocation.
     */
    [[nodiscard]] intern_result intern(node_id parent_id, std::uint64_t ip)
    {
        const edge_key key{ parent_id, ip };

        if(auto it = m_edges.find(key); it != m_edges.end())
        {
            return { it->second, false };
        }

        if((m_max_nodes != 0 && m_nodes.size() >= m_max_nodes) ||
           m_nodes.size() >= std::numeric_limits<node_id>::max())
        {
            throw resource_error("frame_trie: max_nodes exceeded");
        }

        const auto new_id = static_cast<node_id>(m_nodes.size() + 1);
        m_nodes.push_back({ new_id, parent_id, ip });
        m_edges.emplace(key, new_id);
        return { new_id, true };
    }

    /**
     * @brief Look up an existing node without allocating.
     * @param parent_id Parent node id, or k_root for a top-of-stack frame.
     * @param ip        Instruction pointer that labels the edge.
     * @return The node id, or k_root if the edge has not been interned.
     */
    [[nodiscard]] node_id find(node_id parent_id, std::uint64_t ip) const noexcept
    {
        const auto it = m_edges.find(edge_key{ parent_id, ip });
        return it == m_edges.end() ? k_root : it->second;
    }

    /**
     * @brief Number of allocated nodes (excluding the implicit root).
     * @return Allocated node count.
     */
    [[nodiscard]] std::size_t size() const noexcept { return m_nodes.size(); }

    /**
     * @brief The last id allocated by intern(), or k_root if none.
     * @return Last-assigned node id.
     */
    [[nodiscard]] node_id last_assigned_id() const noexcept
    {
        return static_cast<node_id>(m_nodes.size());
    }

    /**
     * @brief Reflection of node @p id.
     * @param id Node id to inspect. Caller must ensure
     *           1 <= id <= last_assigned_id(); calling with k_root
     *           or an out-of-range id is undefined.
     * @return Snapshot of the requested node.
     */
    [[nodiscard]] node_view node_at(node_id id) const noexcept { return m_nodes[id - 1]; }

    /**
     * @brief Set a hard ceiling on the number of allocated nodes.
     *
     * v1 throws on overflow; a fallback raw-IP record type is
     * deferred to v2 per the architecture plan.
     * @param cap Maximum number of allocated nodes, or 0 to disable
     *            the cap (the default).
     */
    void set_max_nodes(std::size_t cap) noexcept { m_max_nodes = cap; }

private:
    // NOLINTNEXTLINE(readability-identifier-naming): codebase k_*-prefix convention
    static constexpr std::uint64_t k_mix = 0x9E3779B97F4A7C15ULL;

    struct edge_key
    {
        node_id       parent_id;
        std::uint64_t ip;

        [[nodiscard]] bool operator==(const edge_key&) const noexcept = default;
    };

    struct edge_hash
    {
        [[nodiscard]] std::size_t operator()(const edge_key& key) const noexcept
        {
            return static_cast<std::size_t>(
                static_cast<std::uint64_t>(key.parent_id) * k_mix + key.ip);
        }
    };

    std::vector<node_view>                          m_nodes;
    std::unordered_map<edge_key, node_id, edge_hash> m_edges;
    std::size_t                                     m_max_nodes = 0;
};

}  // namespace cpu_snap::sampling::internal
