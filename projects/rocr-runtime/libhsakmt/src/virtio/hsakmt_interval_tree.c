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

#include "hsakmt_interval_tree.h"
#include <stddef.h>

static inline void rb_set_parent(struct rb_node* rb, struct rb_node* p) {
  rb->__rb_parent_color = rb_color(rb) | (unsigned long)p;
}

static inline void rb_set_parent_color(struct rb_node* rb, struct rb_node* p, int color) {
  rb->__rb_parent_color = (unsigned long)p | color;
}

static inline void rb_set_black(struct rb_node* rb) { rb->__rb_parent_color |= RB_BLACK; }

static inline struct rb_node* rb_red_parent(struct rb_node* red) {
  return (struct rb_node*)red->__rb_parent_color;
}

static inline void __rb_change_child(struct rb_node* old, struct rb_node* new,
                                     struct rb_node* parent, interval_tree_t* root) {
  if (parent) {
    if (parent->rb_left == old)
      parent->rb_left = new;
    else
      parent->rb_right = new;
  } else {
    root->rb_node = new;
  }
}

static inline void __rb_rotate_set_parents(struct rb_node* old, struct rb_node* new,
                                           interval_tree_t* root, int color) {
  struct rb_node* parent = rb_parent(old);
  new->__rb_parent_color = old->__rb_parent_color;
  rb_set_parent_color(old, new, color);
  __rb_change_child(old, new, parent, root);
}

static inline unsigned long compute_subtree_last(interval_tree_node_t* node);
static void interval_tree_propagate(struct rb_node* rb, struct rb_node* stop);
static void interval_tree_copy(struct rb_node* rb_old, struct rb_node* rb_new);
static void interval_tree_rotate(struct rb_node* rb_old, struct rb_node* rb_new);

void __rb_insert_color(struct rb_node* node, interval_tree_t* root,
                       void (*augment_rotate)(struct rb_node* old, struct rb_node* new)) {
  struct rb_node *parent = rb_red_parent(node), *gparent, *tmp;

  while (true) {
    if (!parent) {
      rb_set_parent_color(node, NULL, RB_BLACK);
      break;
    } else if (rb_is_black(parent)) {
      break;
    }

    gparent = rb_red_parent(parent);

    tmp = gparent->rb_right;
    if (parent != tmp) {
      if (tmp && rb_is_red(tmp)) {
        rb_set_parent_color(tmp, gparent, RB_BLACK);
        rb_set_parent_color(parent, gparent, RB_BLACK);
        node = gparent;
        parent = rb_parent(node);
        rb_set_parent_color(node, parent, RB_RED);
        continue;
      }

      tmp = parent->rb_right;
      if (node == tmp) {
        parent->rb_right = tmp = node->rb_left;
        node->rb_left = parent;
        if (tmp) rb_set_parent_color(tmp, parent, RB_BLACK);
        rb_set_parent_color(parent, node, RB_RED);
        augment_rotate(parent, node);
        parent = node;
        tmp = node->rb_right;
      }

      gparent->rb_left = tmp;
      parent->rb_right = gparent;
      if (tmp) rb_set_parent_color(tmp, gparent, RB_BLACK);
      __rb_rotate_set_parents(gparent, parent, root, RB_RED);
      augment_rotate(gparent, parent);
      break;
    } else {
      tmp = gparent->rb_left;
      if (tmp && rb_is_red(tmp)) {
        rb_set_parent_color(tmp, gparent, RB_BLACK);
        rb_set_parent_color(parent, gparent, RB_BLACK);
        node = gparent;
        parent = rb_parent(node);
        rb_set_parent_color(node, parent, RB_RED);
        continue;
      }

      tmp = parent->rb_left;
      if (node == tmp) {
        parent->rb_left = tmp = node->rb_right;
        node->rb_right = parent;
        if (tmp) rb_set_parent_color(tmp, parent, RB_BLACK);
        rb_set_parent_color(parent, node, RB_RED);
        augment_rotate(parent, node);
        parent = node;
        tmp = node->rb_left;
      }

      gparent->rb_right = tmp;
      parent->rb_left = gparent;
      if (tmp) rb_set_parent_color(tmp, gparent, RB_BLACK);
      __rb_rotate_set_parents(gparent, parent, root, RB_RED);
      augment_rotate(gparent, parent);
      break;
    }
  }
}

void __rb_erase_color(struct rb_node* parent, interval_tree_t* root,
                      void (*augment_rotate)(struct rb_node* old, struct rb_node* new)) {
  struct rb_node *node = NULL, *sibling, *tmp1, *tmp2;

  while (true) {
    sibling = parent->rb_right;
    if (node != sibling) {
      if (rb_is_red(sibling)) {
        parent->rb_right = tmp1 = sibling->rb_left;
        sibling->rb_left = parent;
        rb_set_parent_color(tmp1, parent, RB_BLACK);
        __rb_rotate_set_parents(parent, sibling, root, RB_RED);
        augment_rotate(parent, sibling);
        sibling = tmp1;
      }
      tmp1 = sibling->rb_right;
      if (!tmp1 || rb_is_black(tmp1)) {
        tmp2 = sibling->rb_left;
        if (!tmp2 || rb_is_black(tmp2)) {
          rb_set_parent_color(sibling, parent, RB_RED);
          if (rb_is_red(parent))
            rb_set_black(parent);
          else {
            node = parent;
            parent = rb_parent(node);
            if (parent) continue;
          }
          break;
        }
        sibling->rb_left = tmp1 = tmp2->rb_right;
        tmp2->rb_right = sibling;
        parent->rb_right = tmp2;
        if (tmp1) rb_set_parent_color(tmp1, sibling, RB_BLACK);
        augment_rotate(sibling, tmp2);
        tmp1 = sibling;
        sibling = tmp2;
      }
      parent->rb_right = tmp2 = sibling->rb_left;
      sibling->rb_left = parent;
      rb_set_parent_color(tmp1, sibling, RB_BLACK);
      if (tmp2) rb_set_parent(tmp2, parent);
      __rb_rotate_set_parents(parent, sibling, root, RB_BLACK);
      augment_rotate(parent, sibling);
      break;
    } else {
      sibling = parent->rb_left;
      if (rb_is_red(sibling)) {
        parent->rb_left = tmp1 = sibling->rb_right;
        sibling->rb_right = parent;
        rb_set_parent_color(tmp1, parent, RB_BLACK);
        __rb_rotate_set_parents(parent, sibling, root, RB_RED);
        augment_rotate(parent, sibling);
        sibling = tmp1;
      }
      tmp1 = sibling->rb_left;
      if (!tmp1 || rb_is_black(tmp1)) {
        tmp2 = sibling->rb_right;
        if (!tmp2 || rb_is_black(tmp2)) {
          rb_set_parent_color(sibling, parent, RB_RED);
          if (rb_is_red(parent))
            rb_set_black(parent);
          else {
            node = parent;
            parent = rb_parent(node);
            if (parent) continue;
          }
          break;
        }
        sibling->rb_right = tmp1 = tmp2->rb_left;
        tmp2->rb_left = sibling;
        parent->rb_left = tmp2;
        if (tmp1) rb_set_parent_color(tmp1, sibling, RB_BLACK);
        augment_rotate(sibling, tmp2);
        tmp1 = sibling;
        sibling = tmp2;
      }
      parent->rb_left = tmp2 = sibling->rb_right;
      sibling->rb_right = parent;
      rb_set_parent_color(tmp1, sibling, RB_BLACK);
      if (tmp2) rb_set_parent(tmp2, parent);
      __rb_rotate_set_parents(parent, sibling, root, RB_BLACK);
      augment_rotate(parent, sibling);
      break;
    }
  }
}

struct rb_node* __rb_erase_node(struct rb_node* node, interval_tree_t* root,
                                void (*augment_rotate)(struct rb_node* old, struct rb_node* new)) {
  struct rb_node *child = node->rb_right, *tmp = node->rb_left;
  struct rb_node *parent, *rebalance;
  unsigned long pc;

  if (!tmp) {
    /*
     * Case 1: node to erase has no more than 1 child (easy!)
     *
     * Note that if there is one child it must be red due to 5)
     * and node must be black due to 4). We adjust colors locally
     * so as to bypass __rb_erase_color() later on.
     */
    pc = node->__rb_parent_color;
    parent = __rb_parent(pc);
    __rb_change_child(node, child, parent, root);
    if (child) {
      child->__rb_parent_color = pc;
      rebalance = NULL;
    } else {
      rebalance = __rb_is_black(pc) ? parent : NULL;
    }
    tmp = parent;
  } else if (!child) {
    /* Still case 1, but this time the child is node->rb_left */
    tmp->__rb_parent_color = pc = node->__rb_parent_color;
    parent = __rb_parent(pc);
    __rb_change_child(node, tmp, parent, root);
    rebalance = NULL;
    tmp = parent;
  } else {
    struct rb_node *successor = child, *child2;
    tmp = child->rb_left;
    if (!tmp) {
      /*
       * Case 2: node's successor is its right child
       *
       *    (n)          (s)
       *    / \          / \
       *  (x) (s)  ->  (x) (c)
       *        \
       *        (c)
       */
      parent = successor;
      child2 = successor->rb_right;
      /* Copy augmented data: successor takes node's place */
      interval_tree_copy(node, successor);
    } else {
      /*
       * Case 3: node's successor is leftmost under
       * node's right child subtree
       *
       *    (n)          (s)
       *    / \          / \
       *  (x) (y)  ->  (x) (y)
       *      /            /
       *    (p)          (p)
       *    /            /
       *  (s)          (c)
       *    \
       *    (c)
       */
      do {
        parent = successor;
        successor = tmp;
        tmp = tmp->rb_left;
      } while (tmp);
      parent->rb_left = child2 = successor->rb_right;
      successor->rb_right = child;
      rb_set_parent(child, successor);
      /* Copy augmented data */
      interval_tree_copy(node, successor);
      /* Propagate changes up from parent */
      interval_tree_propagate(parent, successor);
    }

    successor->rb_left = tmp = node->rb_left;
    rb_set_parent(tmp, successor);

    pc = node->__rb_parent_color;
    tmp = __rb_parent(pc);
    __rb_change_child(node, successor, tmp, root);
    if (child2) {
      successor->__rb_parent_color = pc;
      rb_set_parent_color(child2, parent, RB_BLACK);
      rebalance = NULL;
    } else {
      unsigned long pc2 = successor->__rb_parent_color;
      successor->__rb_parent_color = pc;
      rebalance = __rb_is_black(pc2) ? parent : NULL;
    }
    tmp = successor;
  }

  /* Propagate augmented data changes */
  interval_tree_propagate(tmp, NULL);
  return rebalance;
}

struct rb_node* rb_first(const interval_tree_t* root) {
  struct rb_node* n;

  n = root->rb_node;
  if (!n) return NULL;
  while (n->rb_left) n = n->rb_left;
  return n;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-qual"
struct rb_node* rb_next(const struct rb_node* node) {
  struct rb_node* parent;

  if (RB_EMPTY_NODE(node)) return NULL;

  if (node->rb_right) {
    node = node->rb_right;
    while (node->rb_left) node = node->rb_left;
    return (struct rb_node*)node;
  }

  while ((parent = rb_parent(node)) && node == parent->rb_right) node = parent;

  return parent;
}
#pragma GCC diagnostic pop

static inline unsigned long compute_subtree_last(interval_tree_node_t* node) {
  unsigned long max = node->last;
  unsigned long subtree_last;

  if (node->rb.rb_left) {
    interval_tree_node_t* left = rb_entry(node->rb.rb_left, interval_tree_node_t, rb);
    subtree_last = left->__subtree_last;
    if (max < subtree_last) max = subtree_last;
  }

  if (node->rb.rb_right) {
    interval_tree_node_t* right = rb_entry(node->rb.rb_right, interval_tree_node_t, rb);
    subtree_last = right->__subtree_last;
    if (max < subtree_last) max = subtree_last;
  }

  return max;
}

static void interval_tree_propagate(struct rb_node* rb, struct rb_node* stop) {
  while (rb != stop) {
    interval_tree_node_t* node = rb_entry(rb, interval_tree_node_t, rb);
    unsigned long augmented = compute_subtree_last(node);

    if (node->__subtree_last == augmented) break;

    node->__subtree_last = augmented;
    rb = rb_parent(&node->rb);
  }
}

static void interval_tree_copy(struct rb_node* rb_old, struct rb_node* rb_new) {
  interval_tree_node_t* old = rb_entry(rb_old, interval_tree_node_t, rb);
  interval_tree_node_t* new = rb_entry(rb_new, interval_tree_node_t, rb);

  new->__subtree_last = old->__subtree_last;
}

static void interval_tree_rotate(struct rb_node* rb_old, struct rb_node* rb_new) {
  interval_tree_node_t* old = rb_entry(rb_old, interval_tree_node_t, rb);
  interval_tree_node_t* new = rb_entry(rb_new, interval_tree_node_t, rb);

  new->__subtree_last = old->__subtree_last;
  old->__subtree_last = compute_subtree_last(old);
}

static void interval_tree_augment_rotate_wrapper(struct rb_node* rb_old, struct rb_node* rb_new) {
  if (rb_old && rb_new) {
    interval_tree_rotate(rb_old, rb_new);
  } else if (rb_old) {
    /* Called from __rb_erase_node with rb_new == NULL */
    interval_tree_propagate(rb_old, NULL);
  }
}

void hsakmt_interval_tree_insert(interval_tree_t* root, interval_tree_node_t* node) {
  struct rb_node** link = &root->rb_node;
  struct rb_node* rb_parent = NULL;
  unsigned long start = node->start;
  unsigned long last = node->last;
  interval_tree_node_t* parent;

  while (*link) {
    rb_parent = *link;
    parent = rb_entry(rb_parent, interval_tree_node_t, rb);

    if (parent->__subtree_last < last) parent->__subtree_last = last;

    if (start < parent->start)
      link = &parent->rb.rb_left;
    else
      link = &parent->rb.rb_right;
  }

  node->__subtree_last = last;
  rb_link_node(&node->rb, rb_parent, link);
  __rb_insert_color(&node->rb, root, interval_tree_augment_rotate_wrapper);
}

void hsakmt_interval_tree_remove(interval_tree_t* root, interval_tree_node_t* node) {
  struct rb_node* rebalance;

  rebalance = __rb_erase_node(&node->rb, root, interval_tree_augment_rotate_wrapper);
  if (rebalance) __rb_erase_color(rebalance, root, interval_tree_augment_rotate_wrapper);
}

static interval_tree_node_t* interval_tree_subtree_search(interval_tree_node_t* node,
                                                          unsigned long start, unsigned long last) {
  while (true) {
    /*
     * Loop invariant: start <= node->__subtree_last
     * (Cond2 is satisfied by one of the subtree nodes)
     */
    if (node->rb.rb_left) {
      interval_tree_node_t* left = rb_entry(node->rb.rb_left, interval_tree_node_t, rb);
      if (start <= left->__subtree_last) {
        /*
         * Some nodes in left subtree satisfy Cond2.
         * Iterate to find the leftmost such node N.
         * If it also satisfies Cond1, that's the match
         * we are looking for. Otherwise, there is no
         * matching interval as nodes to the right of N
         * can't satisfy Cond1 either.
         */
        node = left;
        continue;
      }
    }

    if (node->start <= last) { /* Cond1 */
      if (start <= node->last) /* Cond2 */
        return node;           /* Match */
      if (node->rb.rb_right) {
        node = rb_entry(node->rb.rb_right, interval_tree_node_t, rb);
        if (start <= node->__subtree_last) continue;
      }
    }
    return NULL; /* No match */
  }
}

interval_tree_node_t* hsakmt_interval_tree_iter_first(interval_tree_t* root, unsigned long start,
                                                      unsigned long last) {
  interval_tree_node_t* node;

  if (!root->rb_node) return NULL;

  node = rb_entry(root->rb_node, interval_tree_node_t, rb);
  if (node->__subtree_last < start) return NULL;

  return interval_tree_subtree_search(node, start, last);
}

interval_tree_node_t* hsakmt_interval_tree_iter_next(interval_tree_t* root,
                                                     interval_tree_node_t* node,
                                                     unsigned long start, unsigned long last) {
  struct rb_node* rb = node->rb.rb_right;
  struct rb_node* prev;

  /* Note: root parameter is unused but kept for API consistency */
  (void)root;

  while (true) {
    /*
     * Loop invariants:
     *   Cond1: node->start <= last
     *   rb == node->rb.rb_right
     *
     * First, search right subtree if suitable
     */
    if (rb) {
      interval_tree_node_t* right = rb_entry(rb, interval_tree_node_t, rb);
      if (start <= right->__subtree_last) return interval_tree_subtree_search(right, start, last);
    }

    /* Move up the tree until we come from a node's left child */
    do {
      rb = rb_parent(&node->rb);
      if (!rb) return NULL;
      prev = &node->rb;
      node = rb_entry(rb, interval_tree_node_t, rb);
      rb = node->rb.rb_right;
    } while (prev == rb);

    /* Check if the node intersects [start;last] */
    if (last < node->start) /* !Cond1 */
      return NULL;
    else if (start <= node->last) /* Cond2 */
      return node;
  }
}
