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

#ifndef _RBTREE_H_
#define _RBTREE_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <inttypes.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <errno.h>
#include "rbtree_amd.h"

typedef struct rbtree_node_s rbtree_node_t;

struct rbtree_node_s {
	rbtree_key_t    key;
	rbtree_node_t   *left;
	rbtree_node_t   *right;
	rbtree_node_t   *parent;
	unsigned char   color;
	unsigned char   data;
};

typedef struct rbtree_s rbtree_t;

struct rbtree_s {
	rbtree_node_t   *root;
	rbtree_node_t   sentinel;
};

#define rbtree_init(tree)				\
	rbtree_sentinel_init(&(tree)->sentinel);	\
	(tree)->root = &(tree)->sentinel;

void hsakmt_rbtree_insert(rbtree_t *tree, rbtree_node_t *node);
void hsakmt_rbtree_delete(rbtree_t *tree, rbtree_node_t *node);
rbtree_node_t *hsakmt_rbtree_prev(rbtree_t *tree,
		rbtree_node_t *node);
rbtree_node_t *hsakmt_rbtree_next(rbtree_t *tree,
		rbtree_node_t *node);

#define rbt_red(node)			((node)->color = 1)
#define rbt_black(node)			((node)->color = 0)
#define rbt_is_red(node)		((node)->color)
#define rbt_is_black(node)		(!rbt_is_red(node))
#define rbt_copy_color(n1, n2)		(n1->color = n2->color)

/* a sentinel must be black */

#define rbtree_sentinel_init(node)	rbt_black(node)

static inline rbtree_node_t *
rbtree_min(rbtree_node_t *node, rbtree_node_t *sentinel)
{
	while (node->left != sentinel) {
		node = node->left;
	}

	return node;
}

#include "rbtree_amd.h"

#endif
