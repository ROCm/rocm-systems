/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _HSAKMT_INTERVAL_TREE_H_
#define _HSAKMT_INTERVAL_TREE_H_

#include "rbtree.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NOTE: despite the name, this is not an augmented interval tree (no
 * subtree-max invariant, no O(log n) overlap search guarantee). It is
 * built directly on top of the existing hsakmt rbtree (rbtree.c), ordered
 * by start address; overlap queries walk forward in ascending order and
 * stop once a node's start passes the query end (O(n) worst case). The
 * public API/name is kept stable so callers, and any future upgrade to a
 * real augmented interval tree, don't need to change.
 */
typedef rbtree_t interval_tree_t;

typedef struct interval_tree_node {
  rbtree_node_t rb;
  unsigned long start;
  unsigned long last;
} interval_tree_node_t;

static inline void interval_tree_init(interval_tree_t* tree) {
  rbtree_sentinel_init(&tree->sentinel);
  tree->root = &tree->sentinel;
}

static inline void interval_tree_node_init(interval_tree_node_t* node, unsigned long start,
                                           unsigned long last) {
  node->start = start;
  node->last = last;
  /* Duplicate start addresses are fine: inserts never overwrite, and the scan below walks every node, not just key matches. */
  node->rb.key = rbtree_key(start, 0);
}

void hsakmt_interval_tree_insert(interval_tree_t* tree, interval_tree_node_t* node);
void hsakmt_interval_tree_remove(interval_tree_t* tree, interval_tree_node_t* node);

/* Find the first (lowest-start) node overlapping [start, last], or NULL. */
interval_tree_node_t* hsakmt_interval_tree_iter_first(interval_tree_t* tree, unsigned long start,
                                                      unsigned long last);
/* Continue a search from a previously returned node. */
interval_tree_node_t* hsakmt_interval_tree_iter_next(interval_tree_t* tree,
                                                     interval_tree_node_t* node,
                                                     unsigned long start, unsigned long last);

#ifdef __cplusplus
}
#endif

#endif /* _HSAKMT_INTERVAL_TREE_H_ */
