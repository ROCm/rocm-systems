#pragma once
#include <cstdio>

#define NCCL_PROXY 0
#define INFO(...) do { (void)0; } while (0)
#define WARN(...)                                                                                \
  do {                                                                                           \
    fprintf(stderr, "[proxytrace] ");                                                          \
    fprintf(stderr, __VA_ARGS__);                                                                \
    fprintf(stderr, "\n");                                                                      \
  } while (0)
