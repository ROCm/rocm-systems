/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hsakmt_interval_tree.h"
#include <stddef.h>

static inline interval_tree_node_t* interval_tree_entry(rbtree_node_t* rb) {
  return (interval_tree_node_t*)((char*)rb - offsetof(interval_tree_node_t, rb));
}

void hsakmt_interval_tree_insert(interval_tree_t* tree, interval_tree_node_t* node) {
  hsakmt_rbtree_insert(tree, &node->rb);
}

void hsakmt_interval_tree_remove(interval_tree_t* tree, interval_tree_node_t* node) {
  hsakmt_rbtree_delete(tree, &node->rb);
}

static interval_tree_node_t* interval_tree_scan(interval_tree_t* tree, rbtree_node_t* rb,
                                                unsigned long start, unsigned long last) {
  while (rb) {
    interval_tree_node_t* node = interval_tree_entry(rb);
    if (node->start > last) return NULL;
    if (node->last >= start) return node;
    rb = hsakmt_rbtree_next(tree, rb);
  }
  return NULL;
}

interval_tree_node_t* hsakmt_interval_tree_iter_first(interval_tree_t* tree, unsigned long start,
                                                      unsigned long last) {
  return interval_tree_scan(tree, rbtree_min_max(tree, LEFT), start, last);
}

interval_tree_node_t* hsakmt_interval_tree_iter_next(interval_tree_t* tree,
                                                     interval_tree_node_t* node,
                                                     unsigned long start, unsigned long last) {
  return interval_tree_scan(tree, hsakmt_rbtree_next(tree, &node->rb), start, last);
}
