// SPDX-License-Identifier: MIT

// The version-query contract used by libhsakmt's KFDBaseComponentTest setup.
// This tests the ioctl directly; it does not claim thunk or device-work
// coverage.
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <linux/kfd_ioctl.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/utsname.h>
#include <unistd.h>

int main(void) {
  struct utsname system;
  if (uname(&system) != 0) {
    perror("uname");
    return 1;
  }
  printf("kernel=%s %s %s\nexpected_uapi=%u.%u\n", system.sysname,
         system.release, system.machine, KFD_IOCTL_MAJOR_VERSION,
         KFD_IOCTL_MINOR_VERSION);
  int descriptor = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (descriptor < 0) {
    fprintf(stderr, "Required KFD access unavailable: /dev/kfd: %s\n",
            strerror(errno));
    return 1;
  }
  int result = 1;
  struct kfd_ioctl_get_version_args version = {0};
  if (ioctl(descriptor, AMDKFD_IOC_GET_VERSION, &version) != 0) {
    perror("AMDKFD_IOC_GET_VERSION");
    goto cleanup;
  }
  printf("driver_uapi=%u.%u\n", version.major_version, version.minor_version);
  // GET_VERSION requires the original interface, not every capability added
  // through the snapshot's minor version. New compatible minors are accepted.
  if (version.major_version != KFD_IOCTL_MAJOR_VERSION ||
      version.minor_version < 1) {
    fprintf(stderr,
            "Required KFD interface 1.1 or newer compatible minor is absent\n");
    goto cleanup;
  }
  struct kfd_ioctl_get_version_args again = {UINT32_MAX, UINT32_MAX};
  if (ioctl(descriptor, AMDKFD_IOC_GET_VERSION, &again) != 0) {
    perror("AMDKFD_IOC_GET_VERSION repeat");
    goto cleanup;
  }
  if (again.major_version != version.major_version ||
      again.minor_version != version.minor_version) {
    fprintf(stderr, "GET_VERSION did not overwrite both fields consistently\n");
    goto cleanup;
  }
  result = 0;
cleanup:
  if (close(descriptor) != 0) {
    perror("close /dev/kfd");
    result = 1;
  }
  if (result == 0)
    puts("PASS kfd.version-contract");
  return result;
}
