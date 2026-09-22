/*
 * Copyright (C) 2002-2018 Igor Sysoev
 * Copyright (C) 2011-2018 Nginx, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#include "rbtree.h"

static inline void rbtree_left_rotate(rbtree_node_t **root,
		rbtree_node_t *sentinel, rbtree_node_t *node);
static inline void rbtree_right_rotate(rbtree_node_t **root,
		rbtree_node_t *sentinel, rbtree_node_t *node);

static void
hsakmt_rbtree_insert_value(rbtree_node_t *temp, rbtree_node_t *node,
		rbtree_node_t *sentinel)
{
	rbtree_node_t  **p;

	for ( ;; ) {

		p = rbtree_key_compare(LKP_ALL, &node->key, &temp->key) < 0 ?
			&temp->left : &temp->right;

		if (*p == sentinel) {
			break;
		}

		temp = *p;
	}

	*p = node;
	node->parent = temp;
	node->left = sentinel;
	node->right = sentinel;
	rbt_red(node);
}


void
hsakmt_rbtree_insert(rbtree_t *tree, rbtree_node_t *node)
{
	rbtree_node_t  **root, *temp, *sentinel;

	/* a binary tree insert */

	root = &tree->root;
	sentinel = &tree->sentinel;

	if (*root == sentinel) {
		node->parent = NULL;
		node->left = sentinel;
		node->right = sentinel;
		rbt_black(node);
		*root = node;

		return;
	}

	hsakmt_rbtree_insert_value(*root, node, sentinel);

	/* re-balance tree */

	while (node != *root && rbt_is_red(node->parent)) {

		if (node->parent == node->parent->parent->left) {
			temp = node->parent->parent->right;

			if (rbt_is_red(temp)) {
				rbt_black(node->parent);
				rbt_black(temp);
				rbt_red(node->parent->parent);
				node = node->parent->parent;

			} else {
				if (node == node->parent->right) {
					node = node->parent;
					rbtree_left_rotate(root, sentinel, node);
				}

				rbt_black(node->parent);
				rbt_red(node->parent->parent);
				rbtree_right_rotate(root, sentinel, node->parent->parent);
			}

		} else {
			temp = node->parent->parent->left;

			if (rbt_is_red(temp)) {
				rbt_black(node->parent);
				rbt_black(temp);
				rbt_red(node->parent->parent);
				node = node->parent->parent;

			} else {
				if (node == node->parent->left) {
					node = node->parent;
					rbtree_right_rotate(root, sentinel, node);
				}

				rbt_black(node->parent);
				rbt_red(node->parent->parent);
				rbtree_left_rotate(root, sentinel, node->parent->parent);
			}
		}
	}

	rbt_black(*root);
}


void
hsakmt_rbtree_delete(rbtree_t *tree, rbtree_node_t *node)
{
	unsigned int red;
	rbtree_node_t  **root, *sentinel, *subst, *temp, *w;

	/* a binary tree delete */

	root = &tree->root;
	sentinel = &tree->sentinel;

	if (node->left == sentinel) {
		temp = node->right;
		subst = node;

	} else if (node->right == sentinel) {
		temp = node->left;
		subst = node;

	} else {
		subst = rbtree_min(node->right, sentinel);

		if (subst->left != sentinel) {
			temp = subst->left;
		} else {
			temp = subst->right;
		}
	}

	if (subst == *root) {
		*root = temp;
		rbt_black(temp);

		return;
	}

	red = rbt_is_red(subst);

	if (subst == subst->parent->left) {
		subst->parent->left = temp;

	} else {
		subst->parent->right = temp;
	}

	if (subst == node) {

		temp->parent = subst->parent;

	} else {

		if (subst->parent == node) {
			temp->parent = subst;

		} else {
			temp->parent = subst->parent;
		}

		subst->left = node->left;
		subst->right = node->right;
		subst->parent = node->parent;
		rbt_copy_color(subst, node);

		if (node == *root) {
			*root = subst;

		} else {
			if (node == node->parent->left) {
				node->parent->left = subst;
			} else {
				node->parent->right = subst;
			}
		}

		if (subst->left != sentinel) {
			subst->left->parent = subst;
		}

		if (subst->right != sentinel) {
			subst->right->parent = subst;
		}
	}

	if (red) {
		return;
	}

	/* a delete fixup */

	while (temp != *root && rbt_is_black(temp)) {

		if (temp == temp->parent->left) {
			w = temp->parent->right;

			if (rbt_is_red(w)) {
				rbt_black(w);
				rbt_red(temp->parent);
				rbtree_left_rotate(root, sentinel, temp->parent);
				w = temp->parent->right;
			}

			if (rbt_is_black(w->left) && rbt_is_black(w->right)) {
				rbt_red(w);
				temp = temp->parent;

			} else {
				if (rbt_is_black(w->right)) {
					rbt_black(w->left);
					rbt_red(w);
					rbtree_right_rotate(root, sentinel, w);
					w = temp->parent->right;
				}

				rbt_copy_color(w, temp->parent);
				rbt_black(temp->parent);
				rbt_black(w->right);
				rbtree_left_rotate(root, sentinel, temp->parent);
				temp = *root;
			}

		} else {
			w = temp->parent->left;

			if (rbt_is_red(w)) {
				rbt_black(w);
				rbt_red(temp->parent);
				rbtree_right_rotate(root, sentinel, temp->parent);
				w = temp->parent->left;
			}

			if (rbt_is_black(w->left) && rbt_is_black(w->right)) {
				rbt_red(w);
				temp = temp->parent;

			} else {
				if (rbt_is_black(w->left)) {
					rbt_black(w->right);
					rbt_red(w);
					rbtree_left_rotate(root, sentinel, w);
					w = temp->parent->left;
				}

				rbt_copy_color(w, temp->parent);
				rbt_black(temp->parent);
				rbt_black(w->left);
				rbtree_right_rotate(root, sentinel, temp->parent);
				temp = *root;
			}
		}
	}

	rbt_black(temp);
}


static inline void
rbtree_left_rotate(rbtree_node_t **root, rbtree_node_t *sentinel,
		rbtree_node_t *node)
{
	rbtree_node_t  *temp;

	temp = node->right;
	node->right = temp->left;

	if (temp->left != sentinel) {
		temp->left->parent = node;
	}

	temp->parent = node->parent;

	if (node == *root) {
		*root = temp;

	} else if (node == node->parent->left) {
		node->parent->left = temp;

	} else {
		node->parent->right = temp;
	}

	temp->left = node;
	node->parent = temp;
}


static inline void
rbtree_right_rotate(rbtree_node_t **root, rbtree_node_t *sentinel,
		rbtree_node_t *node)
{
	rbtree_node_t  *temp;

	temp = node->left;
	node->left = temp->right;

	if (temp->right != sentinel) {
		temp->right->parent = node;
	}

	temp->parent = node->parent;

	if (node == *root) {
		*root = temp;

	} else if (node == node->parent->right) {
		node->parent->right = temp;

	} else {
		node->parent->left = temp;
	}

	temp->right = node;
	node->parent = temp;
}


rbtree_node_t *
hsakmt_rbtree_next(rbtree_t *tree, rbtree_node_t *node)
{
	rbtree_node_t  *root, *sentinel, *parent;

	sentinel = &tree->sentinel;

	if (node->right != sentinel) {
		return rbtree_min(node->right, sentinel);
	}

	root = tree->root;

	for ( ;; ) {
		parent = node->parent;

		if (node == root) {
			return NULL;
		}

		if (node == parent->left) {
			return parent;
		}

		node = parent;
	}
}

rbtree_node_t *
hsakmt_rbtree_prev(rbtree_t *tree, rbtree_node_t *node)
{
	rbtree_node_t  *root, *sentinel, *parent;

	sentinel = &tree->sentinel;

	if (node->left != sentinel) {
		return rbtree_max(node->left, sentinel);
	}

	root = tree->root;

	for ( ;; ) {
		parent = node->parent;

		if (node == root) {
			return NULL;
		}

		if (node == parent->right) {
			return parent;
		}

		node = parent;
	}
}
