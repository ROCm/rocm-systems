/*
 * Copyright © 2018 Advanced Micro Devices, Inc.
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

#ifndef _RBTREE_AMD_H_
#define _RBTREE_AMD_H_

typedef struct rbtree_key_s rbtree_key_t;
struct rbtree_key_s {
#define ADDR_BIT 0
#define SIZE_BIT 1
	unsigned long addr;
	unsigned long size;
};
#define BIT(x) (1<<(x))
#define LKP_ALL (BIT(ADDR_BIT) | BIT(SIZE_BIT))
#define LKP_ADDR (BIT(ADDR_BIT))
#define LKP_ADDR_SIZE (BIT(ADDR_BIT) | BIT(SIZE_BIT))

static inline rbtree_key_t
rbtree_key(unsigned long addr, unsigned long size)
{
	return (rbtree_key_t){addr, size};
}

/*
 * compare addr, size one by one
 */
static inline int
rbtree_key_compare(unsigned int type, rbtree_key_t *key1, rbtree_key_t *key2)
{
	if ((type & 1 << ADDR_BIT) && (key1->addr != key2->addr))
		return key1->addr > key2->addr ? 1 : -1;

	if ((type & 1 << SIZE_BIT) && (key1->size != key2->size))
		return key1->size > key2->size ? 1 : -1;

	return 0;
}
#endif /*_RBTREE_AMD_H_*/

/*inlcude this file again with RBTREE_HELPER defined*/
#ifndef RBTREE_HELPER
#define RBTREE_HELPER
#else
#ifndef _RBTREE_AMD_H_HELPER_
#define _RBTREE_AMD_H_HELPER_
static inline rbtree_node_t *
rbtree_max(rbtree_node_t *node, rbtree_node_t *sentinel)
{
	while (node->right != sentinel)
		node = node->right;

	return node;
}

#define LEFT 0
#define RIGHT 1
#define MID 2
static inline rbtree_node_t *
rbtree_min_max(rbtree_t *tree, int lr)
{
	rbtree_node_t *sentinel = &tree->sentinel;
	rbtree_node_t *node = tree->root;

	if (node == sentinel)
		return NULL;

	if (lr == LEFT)
		node = rbtree_min(node, sentinel);
	else if (lr == RIGHT)
		node = rbtree_max(node, sentinel);

	return node;
}

static inline rbtree_node_t *
rbtree_node_any(rbtree_t *tree, int lmr)
{
	rbtree_node_t *sentinel = &tree->sentinel;
	rbtree_node_t *node = tree->root;

	if (node == sentinel)
		return NULL;

	if (lmr == MID)
		return node;

	return rbtree_min_max(tree, lmr);
}

static inline rbtree_node_t *
rbtree_lookup_nearest(rbtree_t *rbtree, rbtree_key_t *key,
		unsigned int type, int lr)
{
	int rc;
	rbtree_node_t *node, *sentinel, *n = NULL;

	node = rbtree->root;
	sentinel = &rbtree->sentinel;

	while (node != sentinel) {
		rc = rbtree_key_compare(type, key, &node->key);

		if (rc < 0) {
			if (lr == RIGHT)
				n = node;
			node = node->left;
			continue;
		}

		if (rc > 0) {
			if (lr == LEFT)
				n = node;
			node = node->right;
			continue;
		}

		return node;
	}

	return n;
}

static inline rbtree_node_t *
rbtree_lookup(rbtree_t *rbtree, rbtree_key_t *key,
		unsigned int type)
{
	return rbtree_lookup_nearest(rbtree, key, type, -1);
}
#endif /*_RBTREE_AMD_H_HELPER_*/

#endif /*RBTREE_HELPER*/

