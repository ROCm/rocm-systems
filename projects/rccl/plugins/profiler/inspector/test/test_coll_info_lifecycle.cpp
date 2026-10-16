/*************************************************************************
 * Regression tests for NCCL inspector collInfo lifetime / lock ordering.
 *
 * Covers NVIDIA/nccl#2000:
 *   - Issue 1: rwlock must not be destroyed while still held.
 *   - Issue 2: collInfo must not be released while proxy paths may still access it.
 *
 * The fixed lifecycle mirrors inspector_plugin.cc (unlock before cleanup).
 ************************************************************************/

#include <gtest/gtest.h>

#include <pthread.h>
#include <sched.h>
#include <time.h>
#include <unistd.h>

#include <cstring>

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

static int timespec_elapsed_ms(const struct timespec* start,
                               const struct timespec* end) {
  return static_cast<int>((end->tv_sec - start->tv_sec) * 1000 +
                          (end->tv_nsec - start->tv_nsec) / 1000000);
}

static bool collInfoInit(struct testCollInfo* collInfo, int refCount) {
  std::memset(collInfo, 0, sizeof(*collInfo));
  collInfo->refCount = refCount;
  collInfo->generation = 1;
  return pthread_rwlock_init(&collInfo->guard, NULL) == 0;
}

static inspectorResult_t collInfoDeRef(struct testCollInfo* collInfo) {
  collInfo->refCount -= 1;
  if (collInfo->refCount == 0) {
    return inspectorReturn;
  }
  return inspectorSuccess;
}

static bool collInfoCleanup(struct testCollInfo* collInfo) {
  if (pthread_rwlock_destroy(&collInfo->guard) != 0) {
    return false;
  }
  collInfo->destroyed = 1;
  return true;
}

static bool stopEventFixed(struct testCollInfo* collInfo) {
  int needsCleanup = 0;
  if (pthread_rwlock_wrlock(&collInfo->guard) != 0) {
    return false;
  }
  inspectorResult_t res = collInfoDeRef(collInfo);
  if (res == inspectorReturn) {
    needsCleanup = 1;
  }
  if (pthread_rwlock_unlock(&collInfo->guard) != 0) {
    return false;
  }
  if (needsCleanup && !collInfoCleanup(collInfo)) {
    return false;
  }
  return true;
}

struct proxyThreadArg {
  struct testCollInfo* collInfo;
  volatile int stop;
  volatile int acquired;
  volatile int lastGeneration;
};

static void* proxyProgressThread(void* arg) {
  struct proxyThreadArg* ctx = static_cast<struct proxyThreadArg*>(arg);
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

static bool wait_for(volatile int* flag, int timeoutMs) {
  struct timespec start, now;
  clock_gettime(CLOCK_MONOTONIC, &start);
  while (!*flag) {
    sched_yield();
    clock_gettime(CLOCK_MONOTONIC, &now);
    if (timespec_elapsed_ms(&start, &now) > timeoutMs) {
      return false;
    }
  }
  return true;
}

TEST(CollInfoLifecycle, UnlockBeforeDestroy) {
  struct testCollInfo collInfo;
  ASSERT_TRUE(collInfoInit(&collInfo, 1));
  ASSERT_TRUE(stopEventFixed(&collInfo));
  EXPECT_EQ(collInfo.destroyed, 1);
  EXPECT_EQ(collInfo.refCount, 0);
}

TEST(CollInfoLifecycle, ProxyUnblockedAfterFixedTeardown) {
  struct testCollInfo collInfo;
  ASSERT_TRUE(collInfoInit(&collInfo, 2));

  struct proxyThreadArg arg = {
      .collInfo = &collInfo,
      .stop = 0,
      .acquired = 0,
      .lastGeneration = 0,
  };

  pthread_t proxy;
  ASSERT_EQ(pthread_create(&proxy, NULL, proxyProgressThread, &arg), 0);

  ASSERT_TRUE(stopEventFixed(&collInfo));
  ASSERT_TRUE(wait_for(&arg.acquired, 2000));
  EXPECT_EQ(arg.lastGeneration, 1);

  arg.stop = 1;
  pthread_join(proxy, NULL);

  ASSERT_TRUE(stopEventFixed(&collInfo));
  EXPECT_EQ(collInfo.destroyed, 1);
}

TEST(CollInfoLifecycle, RepeatedTeardownCycles) {
  for (int cycle = 0; cycle < 100; ++cycle) {
    struct testCollInfo collInfo;
    ASSERT_TRUE(collInfoInit(&collInfo, 3)) << "cycle " << cycle;
    ASSERT_TRUE(stopEventFixed(&collInfo)) << "cycle " << cycle;
    ASSERT_TRUE(stopEventFixed(&collInfo)) << "cycle " << cycle;
    ASSERT_TRUE(stopEventFixed(&collInfo)) << "cycle " << cycle;
    EXPECT_EQ(collInfo.destroyed, 1) << "cycle " << cycle;
  }
}
