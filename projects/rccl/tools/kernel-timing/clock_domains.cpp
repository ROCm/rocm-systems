// Which host clock is the HSA system clock domain, the one dispatch timestamps
// live in? rocprof stamps with CLOCK_BOOTTIME, so if those differ, kernel
// intervals cannot be placed on a rocprof timeline without a conversion.

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdio>
#include <cstdint>
#include <ctime>

static double secs(clockid_t c) {
  timespec ts;
  clock_gettime(c, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
  if (hsa_init() != HSA_STATUS_SUCCESS) {
    printf("hsa_init failed\n");
    return 1;
  }
  uint64_t freq = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq);

  // Sample the HSA clock sandwiched between host clock reads.
  double bootBefore = secs(CLOCK_BOOTTIME);
  double monoBefore = secs(CLOCK_MONOTONIC);
  double realBefore = secs(CLOCK_REALTIME);
  uint64_t hsaTick = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &hsaTick);
  double bootAfter = secs(CLOCK_BOOTTIME);

  double hsaSec = freq ? (double)hsaTick / (double)freq : 0.0;
  printf("HSA timestamp frequency : %lu Hz\n", (unsigned long)freq);
  printf("HSA timestamp           : %.6f s\n", hsaSec);
  printf("CLOCK_BOOTTIME          : %.6f s   (delta vs HSA %+.6f)\n", bootBefore,
         bootBefore - hsaSec);
  printf("CLOCK_MONOTONIC         : %.6f s   (delta vs HSA %+.6f)\n", monoBefore,
         monoBefore - hsaSec);
  printf("CLOCK_REALTIME          : %.6f s   (delta vs HSA %+.6f)\n", realBefore,
         realBefore - hsaSec);
  printf("host read window        : %.6f s\n", bootAfter - bootBefore);

  // Rate check over a fixed interval: do the clocks advance together?
  uint64_t t0 = 0, t1 = 0;
  hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &t0);
  double b0 = secs(CLOCK_BOOTTIME);
  timespec nap{0, 200 * 1000 * 1000};
  nanosleep(&nap, nullptr);
  hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &t1);
  double b1 = secs(CLOCK_BOOTTIME);
  double hsaElapsed = freq ? (double)(t1 - t0) / (double)freq : 0.0;
  printf("\nover a ~0.2 s nap: HSA %.6f s, BOOTTIME %.6f s, ratio %.6f\n", hsaElapsed, b1 - b0,
         (b1 - b0) ? hsaElapsed / (b1 - b0) : 0.0);

  hsa_shut_down();
  return 0;
}
