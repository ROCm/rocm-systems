// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <cstdio>
#include <cstdlib>

// The post-call `+ 1` keeps the recursive call out of tail position so the
// compiler cannot apply sibling-call optimization (which would collapse all
// recursive frames into a single function entry at runtime).
__attribute__((noinline)) long
recurse(int depth, long acc)
{
    if(depth <= 0) return acc;
    return recurse(depth - 1, acc) + 1;
}

int
main(int argc, char** argv)
{
    int  depth  = (argc > 1) ? atoi(argv[1]) : 100;
    long result = recurse(depth, 0);
    printf("recurse(%d) = %ld (expected %d)\n", depth, result, depth);
    return 0;
}
