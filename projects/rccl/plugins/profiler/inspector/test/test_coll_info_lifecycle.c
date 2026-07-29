/*************************************************************************
 * Regression tests for NCCL inspector collInfo lifetime / lock ordering.
 *
 * Covers NVIDIA/nccl#2000:
 *   - Issue 1: rwlock must not be destroyed while still held.
 *   - Issue 2: collInfo must not be released while proxy paths may still access it.
 *
 * The fixed lifecycle mirrors inspector_plugin.cc (unlock before cleanup).
 ************************************************************************/

#define _GNU_SOURCE

#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef enum {
  inspectorSuccess = 0,
  inspectorReturn = 1,
} inspectorResult_t;

struct testCollInfo {
  int refCount;
  pthread_rwlock_t guard;
  int destroyed;
  int generation;
};

#define TEST_ASSERT(cond, msg)                                                 \
  do {                                                                         \
    if (!(cond)) {                                                             \
      printf("FAIL: %s:%d - %s\n", __func__, __LINE__, msg);                   \
      return 0;                                                                \
    }                                                                          \
  } while (0)

#define TEST_PASS()                                                            \
  do {                                                                         \
    printf("PASS: %s\n", __func__);                                            \
    return 1;                                                                  \
  } while (0)

static int timespec_elapsed_ms(const struct timespec* start,
                               const struct timespec* end) {
  return (int)((end->tv_sec - start->tv_sec) * 1000 +
               (end->tv_nsec - start->tv_nsec) / 1000000);
}

static int collInfoInit(struct testCollInfo* collInfo, int refCount) {
  memset(collInfo, 0, sizeof(*collInfo));
  collInfo->refCount = refCount;
  collInfo->generation = 1;
  if (pthread_rwlock_init(&collInfo->guard, NULL) != 0) {
    return 0;
  }
  return 1;
}

static inspectorResult_t collInfoDeRef(struct testCollInfo* collInfo) {
  collInfo->refCount -= 1;
  if (collInfo->refCount == 0) {
    return inspectorReturn;
  }
  return inspectorSuccess;
}

static int collInfoCleanup(struct testCollInfo* collInfo) {
  if (pthread_rwlock_destroy(&collInfo->guard) != 0) {
    return 0;
  }
  collInfo->destroyed = 1;
  return 1;
}

static int stopEventCollFixed(struct testCollInfo* collInfo) {
  int needsCleanup = 0;
  if (pthread_rwlock_wrlock(&collInfo->guard) != 0) {
    return 0;
  }
  inspectorResult_t res = collInfoDeRef(collInfo);
  if (res == inspectorReturn) {
    needsCleanup = 1;
  }
  if (pthread_rwlock_unlock(&collInfo->guard) != 0) {
    return 0;
  }
  if (needsCleanup && !collInfoCleanup(collInfo)) {
    return 0;
  }
  return 1;
}

static int stopEventKernelChFixed(struct testCollInfo* collInfo) {
  int needsCleanup = 0;
  if (pthread_rwlock_wrlock(&collInfo->guard) != 0) {
    return 0;
  }
  inspectorResult_t res = collInfoDeRef(collInfo);
  if (res == inspectorReturn) {
    needsCleanup = 1;
  }
  if (pthread_rwlock_unlock(&collInfo->guard) != 0) {
    return 0;
  }
  if (needsCleanup && !collInfoCleanup(collInfo)) {
    return 0;
  }
  return 1;
}

struct proxyThreadArg {
  struct testCollInfo* collInfo;
  volatile int stop;
  volatile int acquired;
  volatile int lastGeneration;
};

static void* proxyProgressThread(void* arg) {
  struct proxyThreadArg* ctx = (struct proxyThreadArg*)arg;
  while (!ctx->stop) {
    if (pthread_rwlock_trywrlock(&ctx->collInfo->guard) == 0) {
      ctx->lastGeneration = ctx->collInfo->generation;
      ctx->acquired = 1;
      pthread_rwlock_unlock(&ctx->collInfo->guard);
      break;
    }
    sched_yield();
  }
  return NULL;
}

static int wait_for(volatile int* flag, int timeoutMs) {
  struct timespec start, now;
  clock_gettime(CLOCK_MONOTONIC, &start);
  while (!*flag) {
    sched_yield();
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (timespec_elapsed_ms(&start, &now) > timeoutMs) {
      return 0;
    }
  }
  return 1;
}

static int test_unlock_before_destroy(void) {
  struct testCollInfo collInfo;
  TEST_ASSERT(collInfoInit(&collInfo, 1), "collInfoInit failed");
  TEST_ASSERT(stopEventCollFixed(&collInfo), "stopEventCollFixed failed");
  TEST_ASSERT(collInfo.destroyed == 1, "collInfo should be cleaned up");
  TEST_ASSERT(collInfo.refCount == 0, "refCount should be zero after cleanup");
  TEST_PASS();
}

static int test_proxy_unblocked_after_fixed_teardown(void) {
  struct testCollInfo collInfo;
  TEST_ASSERT(collInfoInit(&collInfo, 2), "collInfoInit failed");

  struct proxyThreadArg arg = {
      .collInfo = &collInfo,
      .stop = 0,
      .acquired = 0,
      .lastGeneration = 0,
  };

  pthread_t proxy;
  TEST_ASSERT(pthread_create(&proxy, NULL, proxyProgressThread, &arg) == 0,
              "pthread_create failed");

  TEST_ASSERT(stopEventKernelChFixed(&collInfo), "first kernel stop failed");
  TEST_ASSERT(wait_for(&arg.acquired, 2000),
              "proxy thread should acquire lock after fixed teardown");
  TEST_ASSERT(arg.lastGeneration == 1, "proxy should observe live generation");

  arg.stop = 1;
  pthread_join(proxy, NULL);

  TEST_ASSERT(stopEventKernelChFixed(&collInfo), "final kernel stop failed");
  TEST_ASSERT(collInfo.destroyed == 1, "collInfo should be destroyed");
  TEST_PASS();
}

static int test_repeated_teardown_cycles(void) {
  for (int cycle = 0; cycle < 100; ++cycle) {
    struct testCollInfo collInfo;
    TEST_ASSERT(collInfoInit(&collInfo, 3), "collInfoInit failed");
    TEST_ASSERT(stopEventKernelChFixed(&collInfo), "kernel stop failed");
    TEST_ASSERT(stopEventKernelChFixed(&collInfo), "kernel stop failed");
    TEST_ASSERT(stopEventCollFixed(&collInfo), "coll stop failed");
    TEST_ASSERT(collInfo.destroyed == 1, "cycle cleanup failed");
  }
  TEST_PASS();
}

typedef struct {
  const char* name;
  int (*func)(void);
  const char* description;
} TestCase;

static TestCase test_cases[] = {
    {"unlock-before-destroy", test_unlock_before_destroy,
     "Destroy rwlock only after unlock (NCCL #2000 issue 1)"},
    {"proxy-after-teardown", test_proxy_unblocked_after_fixed_teardown,
     "Proxy thread can acquire lock during fixed teardown path"},
    {"repeated-cycles", test_repeated_teardown_cycles,
     "Repeated collInfo refcount/cleanup cycles"},
    {NULL, NULL, NULL},
};

static void show_help(const char* program) {
  printf("Usage: %s [test_name ...]\n\n", program);
  printf("NCCL inspector collInfo lifecycle regression tests (issue #2000)\n\n");
  for (int i = 0; test_cases[i].name != NULL; ++i) {
    printf("  %-22s - %s\n", test_cases[i].name, test_cases[i].description);
  }
}

int main(int argc, char* argv[]) {
  if (argc > 1 &&
      (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
    show_help(argv[0]);
    return 0;
  }

  int passed = 0;
  int total = 0;

  if (argc == 1) {
    for (int i = 0; test_cases[i].name != NULL; ++i) {
      total++;
      passed += test_cases[i].func();
    }
  } else {
    for (int arg = 1; arg < argc; ++arg) {
      int found = 0;
      for (int i = 0; test_cases[i].name != NULL; ++i) {
        if (strcmp(argv[arg], test_cases[i].name) == 0) {
          total++;
          passed += test_cases[i].func();
          found = 1;
          break;
        }
      }
      if (!found) {
        printf("ERROR: unknown test '%s'\n", argv[arg]);
        show_help(argv[0]);
        return 1;
      }
    }
  }

  printf("\nResults: %d/%d passed\n", passed, total);
  return passed == total ? 0 : 1;
}
