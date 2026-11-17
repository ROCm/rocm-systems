// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/native_cleanup.h"

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#if defined(EXPECT_NO_EXCEPTIONS) && defined(__EXCEPTIONS)
#error "The no-exceptions test must compile with -fno-exceptions"
#elif !defined(EXPECT_NO_EXCEPTIONS) && !defined(__EXCEPTIONS)
#error "The exceptions test must compile with -fexceptions"
#endif

static void require(bool condition, const char *message) {
  if (!condition) {
    std::fprintf(stderr, "%s\n", message);
    std::abort();
  }
}

class MoveOnly {
public:
  explicit MoveOnly(int value) : value(value) {}
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly(MoveOnly &&) = default;

  int value;
};

static void test_normal_return() {
  int cleanup_count = 0;
  auto cleanup = [&] { ++cleanup_count; };
  bool body_ran = false;
  util::with_native_cleanup(cleanup, [&] {
    body_ran = true;
    errno = EDOM;
  });
  require(body_ran, "void body did not run");
  require(errno == EDOM, "void return changed errno");

  int result = util::with_native_cleanup(cleanup, [] {
    errno = ERANGE;
    return 42;
  });
  require(result == 42, "scalar return changed");
  require(errno == ERANGE, "scalar return changed errno");

  MoveOnly moved = util::with_native_cleanup(cleanup, [] {
    errno = EOVERFLOW;
    return MoveOnly(73);
  });
  require(moved.value == 73, "move-only return changed");
  require(errno == EOVERFLOW, "move-only return changed errno");
  require(cleanup_count == 0, "normal return ran cleanup");
}

struct ThreadState {
  std::atomic<bool> ready{false};
  bool cancel;
  int descriptors[2];
  int cleanup_order[2] = {};
  int cleanup_count = 0;
  bool returned = false;
};

static void *run_thread(void *argument) {
  auto &state = *static_cast<ThreadState *>(argument);
  auto outer_cleanup = [&] {
    require(state.cleanup_count < 2, "outer cleanup repeated");
    state.cleanup_order[state.cleanup_count++] = 1;
    require(close(state.descriptors[0]) == 0, "outer resource close failed");
  };
  auto inner_cleanup = [&] {
    require(state.cleanup_count < 2, "inner cleanup repeated");
    state.cleanup_order[state.cleanup_count++] = 2;
    require(close(state.descriptors[1]) == 0, "inner resource close failed");
  };
  util::with_native_cleanup(outer_cleanup, [&] {
    util::with_native_cleanup(inner_cleanup, [&] {
      if (!state.cancel) {
        pthread_exit(&state);
      }
      state.ready.store(true, std::memory_order_release);
      for (;;) {
        pthread_testcancel();
        sched_yield();
      }
    });
  });
  state.returned = true;
  return nullptr;
}

static void test_native_exit(bool cancel) {
  ThreadState state{.cancel = cancel,
                    .descriptors = {open("/dev/null", O_RDONLY), open("/dev/null", O_RDONLY)}};
  require(state.descriptors[0] >= 0 && state.descriptors[1] >= 0,
          "opening cleanup resources failed");
  pthread_t thread;
  require(pthread_create(&thread, nullptr, run_thread, &state) == 0, "pthread_create failed");
  if (cancel) {
    while (!state.ready.load(std::memory_order_acquire)) {
      sched_yield();
    }
    require(pthread_cancel(thread) == 0, "pthread_cancel failed");
  }
  void *status = nullptr;
  require(pthread_join(thread, &status) == 0, "pthread_join failed");
  require(status == (cancel ? PTHREAD_CANCELED : &state), "native thread exit status changed");
  require(!state.returned, "native exit returned through helper");
  require(state.cleanup_count == 2, "cleanup did not run exactly once");
  require(state.cleanup_order[0] == 2 && state.cleanup_order[1] == 1,
          "nested cleanup did not run in LIFO order");
  for (int descriptor : state.descriptors) {
    errno = 0;
    require(fcntl(descriptor, F_GETFD) == -1 && errno == EBADF, "native exit leaked a resource");
  }
}

int main() {
  alarm(10);
  test_normal_return();
  test_native_exit(false);
  test_native_exit(true);
}
