#include <stdio.h>
#include <infiniband/verbs.h>

int main(void) {
  enum ibv_access_flags has_ibv_access_relaxed_ordering = IBV_ACCESS_RELAXED_ORDERING;
  printf("IBV_ACCESS_RELAXED_ORDERING: %d\n", has_ibv_access_relaxed_ordering);
  return 0;
}
