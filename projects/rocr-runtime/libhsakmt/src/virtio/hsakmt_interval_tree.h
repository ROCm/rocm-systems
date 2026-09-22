/*
 * HSAKMT Interval Tree Implementation
 * Based on Linux kernel's augmented red-black tree implementation
 *
 * This implementation is derived from the Linux kernel source code
 * (lib/rbtree.c, lib/rbtree_augmented.h, lib/interval_tree_generic.h)
 * and simplified for use in this project.
 *
 * Copyright (C) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#ifndef _HSAKMT_INTERVAL_TREE_H_
#define _HSAKMT_INTERVAL_TREE_H_

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

struct rb_node {
  unsigned long __rb_parent_color;
  struct rb_node* rb_right;
  struct rb_node* rb_left;
} __attribute__((aligned(sizeof(long))));

struct rb_root {
  struct rb_node* rb_node;
};

#define RB_ROOT                                                                                    \
  (struct rb_root) { NULL }

typedef struct interval_tree_node {
  struct rb_node rb;
  unsigned long start;          /* Start of interval */
  unsigned long last;           /* Last location in interval */
  unsigned long __subtree_last; /* Max 'last' in this subtree */
} interval_tree_node_t;

typedef struct rb_root interval_tree_t;

#define rb_parent(r) ((struct rb_node*)((r)->__rb_parent_color & ~3UL))

#define __rb_parent(pc) ((struct rb_node*)((pc) & ~3UL))

#define RB_RED 0
#define RB_BLACK 1

#define __rb_color(pc) ((pc)&1)
#define __rb_is_black(pc) __rb_color(pc)
#define __rb_is_red(pc) (!__rb_color(pc))

#define rb_color(rb) ((rb)->__rb_parent_color & 1)
#define rb_is_red(rb) (!rb_color(rb))
#define rb_is_black(rb) rb_color(rb)

#define rb_entry(ptr, type, member) ((type*)((char*)(ptr)-offsetof(type, member)))

#define rb_entry_safe(ptr, type, member)                                                           \
  ({                                                                                               \
    typeof(ptr) ____ptr = (ptr);                                                                   \
    ____ptr ? rb_entry(____ptr, type, member) : NULL;                                              \
  })

#define RB_EMPTY_ROOT(root) ((root)->rb_node == NULL)

#define RB_EMPTY_NODE(node) (rb_parent(node) == (node))

#define RB_CLEAR_NODE(node) ((node)->__rb_parent_color = (unsigned long)(node))

#define interval_start(node) ((node)->start)
#define interval_last(node) ((node)->last)

#define interval_tree_init(itree) ((itree)->rb_node = NULL)


/**
 * rb_link_node - Link a new node into the tree
 * @node: new node to link
 * @parent: parent node
 * @rb_link: pointer to the parent's link (left or right child pointer)
 */
static inline void rb_link_node(struct rb_node* node, struct rb_node* parent,
                                struct rb_node** rb_link) {
  node->__rb_parent_color = (unsigned long)parent;
  node->rb_left = node->rb_right = NULL;
  *rb_link = node;
}

void __rb_insert_color(struct rb_node* node, struct rb_root* root,
                       void (*augment_rotate)(struct rb_node* old, struct rb_node* new));

struct rb_node* __rb_erase_node(struct rb_node* node, struct rb_root* root,
                                void (*augment_rotate)(struct rb_node* old, struct rb_node* new));

void __rb_erase_color(struct rb_node* parent, struct rb_root* root,
                      void (*augment_rotate)(struct rb_node* old, struct rb_node* new));

struct rb_node* rb_first(const struct rb_root* root);
struct rb_node* rb_next(const struct rb_node* node);

/**
 * hsakmt_interval_tree_insert - Insert an interval node into the tree
 * @itree: root of the interval tree
 * @node: interval node to insert
 */
void hsakmt_interval_tree_insert(interval_tree_t* itree, interval_tree_node_t* node);

/**
 * hsakmt_interval_tree_remove - Remove an interval node from the tree
 * @itree: root of the interval tree
 * @node: interval node to remove
 */
void hsakmt_interval_tree_remove(interval_tree_t* itree, interval_tree_node_t* node);

/**
 * hsakmt_interval_tree_iter_first - Find first interval overlapping [start, last]
 * @itree: root of the interval tree
 * @start: start of the query interval
 * @last: last location of the query interval
 *
 * Returns: First overlapping interval node, or NULL if none found
 */
interval_tree_node_t* hsakmt_interval_tree_iter_first(interval_tree_t* itree, unsigned long start,
                                                      unsigned long last);

/**
 * hsakmt_interval_tree_iter_next - Find next interval overlapping [start, last]
 * @itree: root of the interval tree (unused, kept for API consistency)
 * @node: previous interval node returned by iter_first or iter_next
 * @start: start of the query interval
 * @last: last location of the query interval
 *
 * Returns: Next overlapping interval node, or NULL if no more found
 */
interval_tree_node_t* hsakmt_interval_tree_iter_next(interval_tree_t* itree,
                                                     interval_tree_node_t* node,
                                                     unsigned long start, unsigned long last);

/**
 * interval_tree_node_init - Initialize an interval tree node
 * @node: node to initialize
 * @start: start of interval
 * @last: last location of interval
 */
static inline void interval_tree_node_init(interval_tree_node_t* node, unsigned long start,
                                           unsigned long last) {
  node->start = start;
  node->last = last;
  node->__subtree_last = last;
  node->rb.rb_left = NULL;
  node->rb.rb_right = NULL;
  node->rb.__rb_parent_color = 0;
}

/**
 * interval_tree_overlap - Check if two intervals overlap
 * @start1: start of first interval
 * @last1: last of first interval
 * @start2: start of second interval
 * @last2: last of second interval
 *
 * Returns: non-zero if intervals overlap, 0 otherwise
 */
static inline int interval_tree_overlap(unsigned long start1, unsigned long last1,
                                        unsigned long start2, unsigned long last2) {
  return start1 <= last2 && start2 <= last1;
}

#ifdef __cplusplus
}
#endif

#endif /* _HSAKMT_INTERVAL_TREE_H_ */
