/*
************************************************************************************************************************
*
*  Copyright (C) 2007-2022 Advanced Micro Devices, Inc.  All rights reserved.
*  SPDX-License-Identifier: MIT
*
***********************************************************************************************************************/

#ifndef HSA_RUNTIME_CORE_UTIL_MPSC_QUEUE_HPP_
#define HSA_RUNTIME_CORE_UTIL_MPSC_QUEUE_HPP_

#include <atomic>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <vector>
#include <cassert>

/*
 * Lock-free Multi-Producer Single-Consumer (MPSC) queue.
 *
 * Algorithm:
 *  - Based on a classic singly-linked list with a dummy stub node.
 *  - Producers:
 *      1. Allocate & construct a node (value constructed in-place).
 *      2. Set node->next = nullptr.
 *      3. prev = tail_.exchange(node, acq_rel).
 *      4. prev->next.store(node, release).  (Publishes the node to the consumer.)
 *  - Consumer (single thread):
 *      1. Reads head_->next (acquire).
 *      2. If null -> empty.
 *      3. Else claim next, move its value out, advance head_, delete old dummy.
 *
 * Progress:
 *  - Enqueue is wait-free for producers (bounded steps, no loops).
 *  - Dequeue is lock-free (O(1); waits only if empty).
 *
 * Thread-safety:
 *  - Multiple threads may call enqueue() concurrently.
 *  - Exactly ONE thread may call any of: dequeue(), dequeue_batch(), clear(), destructor.
 *
 * Memory ordering rationale:
 *  - tail_.exchange(..., acq_rel) pairs with consumer's acquire load of next pointer.
 *  - The prev->next.store(..., release) publishes the new node after its value is fully constructed.
 *  - Consumer's next.load(acquire) ensures it sees a fully initialized node.
 *
 * Custom Allocator:
 *  - Alloc template parameter (defaults std::allocator<T>).
 *  - Rebound to internal Node type.
 *
 */

namespace rocr {

template<typename T, typename Alloc = std::allocator<T>>
class MPSCQueue {
private:
    struct Node {
        std::atomic<Node*> next;
        T value;

        template<typename... Args>
        explicit Node(Args&&... args)
          : next(nullptr), value(std::forward<Args>(args)...) {}
        // Dummy node ctor (no value)
        Node() : next(nullptr), value() {}
    };

    using NodeAlloc = typename std::allocator_traits<Alloc>::template rebind_alloc<Node>;
    using NodeAllocTraits = std::allocator_traits<NodeAlloc>;

    // Cache line padding to reduce false-sharing between head & tail hot fields.
    struct alignas(64) PaddedAtomicPtr {
        std::atomic<Node*> ptr;
        char pad[64 - sizeof(std::atomic<Node*>) > 0 ? 64 - sizeof(std::atomic<Node*>) : 1];
    };

public:
    MPSCQueue()
      : alloc_()
    {
        Node* stub = create_node_stub();
        head_.ptr.store(stub, std::memory_order_relaxed);
        tail_.ptr.store(stub, std::memory_order_relaxed);
        qsize_.store(0, std::memory_order_relaxed);
    }

    explicit MPSCQueue(const Alloc& alloc)
      : alloc_(alloc)
    {
        Node* stub = create_node_stub();
        head_.ptr.store(stub, std::memory_order_relaxed);
        tail_.ptr.store(stub, std::memory_order_relaxed);
        qsize_.store(0, std::memory_order_relaxed);
    }

    // Non-copyable / non-movable (simplify invariants)
    MPSCQueue(const MPSCQueue&) = delete;
    MPSCQueue& operator=(const MPSCQueue&) = delete;
    MPSCQueue(MPSCQueue&&) = delete;
    MPSCQueue& operator=(MPSCQueue&&) = delete;

    ~MPSCQueue() {
        clear(); // drains and deletes any remaining nodes
        Node* stub = head_.ptr.load(std::memory_order_relaxed);
        if (stub) {
            destroy_node(stub);
        }
        qsize_.store(0, std::memory_order_relaxed);
    }

    // Enqueue by const reference
    inline void enqueue(const T& value) {
        emplace(value);
    }

    // Enqueue by rvalue
    inline void enqueue(T&& value) {
        emplace(std::move(value));
    }

    // Perfect-forwarding construction
    template<typename... Args>
    inline void emplace(Args&&... args) {
        Node* n = allocate_node(std::forward<Args>(args)...);
        // Publish:
        Node* prev = tail_.ptr.exchange(n, std::memory_order_acq_rel);
        // Link from previous tail
        prev->next.store(n, std::memory_order_release);
        qsize_.fetch_add(1, std::memory_order_release);
    }

    // Try dequeue single element; returns false if empty
    inline T dequeue() {
        Node* head = head_.ptr.load(std::memory_order_relaxed);
        Node* next = head->next.load(std::memory_order_acquire);
        if (!next) {
            return nullptr; // empty
        }
        // Move value out
        T out = std::move(next->value);
        // Advance head
        head_.ptr.store(next, std::memory_order_relaxed);
        // Old head was dummy
        destroy_node(head);
        qsize_.fetch_sub(1, std::memory_order_acq_rel);
        return out;
    }

    // Non-blocking peek (observes front without removing). Not linearizable with concurrent enqueues,
    // but safe for single-consumer inspection.
    inline bool peek(T& out) const {
        Node* head = head_.ptr.load(std::memory_order_relaxed);
        Node* next = head->next.load(std::memory_order_acquire);
        if (!next) return false;
        out = next->value; 
        return true;
    }

    // Batch dequeue up to max_items. Returns actual count.
    inline size_t dequeue_batch(std::vector<T>& out) {
        size_t count = 0;
        while (1) {
            Node* head = head_.ptr.load(std::memory_order_relaxed);
            Node* next = head->next.load(std::memory_order_acquire);
            if (!next) break;
            out.emplace_back(std::move(next->value));
            head_.ptr.store(next, std::memory_order_relaxed);
            destroy_node(head);
            ++count;
        }
        if (count) {
            qsize_.fetch_sub(count, std::memory_order_acq_rel);
        }
        return count;
    }

    // empty check (safe for single consumer scenario)
    inline bool empty() const {
        return qsize_.load(std::memory_order_acquire) == 0;
    }

    // Drain everything
    inline void clear() {
        T dummy;
        do  {
            dummy = dequeue();
        } while (dummy != nullptr);
    }

    inline size_t size() const {
        return qsize_.load(std::memory_order_acquire);
    }

private:
    NodeAlloc alloc_;
    PaddedAtomicPtr head_;
    PaddedAtomicPtr tail_;
    alignas(64) std::atomic<size_t> qsize_;

    Node* create_node_stub() {
        Node* n = NodeAllocTraits::allocate(alloc_, 1);
        try {
            NodeAllocTraits::construct(alloc_, n);
        } catch (...) {
            NodeAllocTraits::deallocate(alloc_, n, 1);
            throw;
        }
        return n;
    }

    template<typename... Args>
    Node* allocate_node(Args&&... args) {
        Node* n = NodeAllocTraits::allocate(alloc_, 1);
        try {
            NodeAllocTraits::construct(alloc_, n, std::forward<Args>(args)...);
        } catch (...) {
            NodeAllocTraits::deallocate(alloc_, n, 1);
            throw;
        }
        n->next.store(nullptr, std::memory_order_relaxed);
        return n;
    }

    void destroy_node(Node* n) {
        NodeAllocTraits::destroy(alloc_, n);
        NodeAllocTraits::deallocate(alloc_, n, 1);
    }
};

}
#endif // HSA_RUNTIME_CORE_UTIL_MPSC_QUEUE_HPP_ 