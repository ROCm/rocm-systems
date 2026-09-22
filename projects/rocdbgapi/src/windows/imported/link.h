/* Copyright (c) 2023 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

/* This file reproduces the definition of "struct link_map" and
   "struct r_debug" used by the ROCr runtime.  */

#ifndef LINK_H
#define LINK_H

#include <cstdint>

struct link_map
{
  uint64_t l_addr;
  char *l_name;
  uint64_t *l_ld;
  link_map *l_next;
};

struct r_debug
{
  uint32_t r_version;

  link_map *r_map;
  uint64_t r_brk;
  enum
  {
    RT_CONSISTENT,
    RT_ADD,
    RT_DELETE
  } r_state;

  uint64_t r_ldbase;
};

#endif
