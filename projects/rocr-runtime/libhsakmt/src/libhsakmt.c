/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates. 
 * 
 * SPDX-License-Identifier: MIT
 */
 
#include <errno.h>
#include <sys/ioctl.h>

#include "libhsakmt.h"
#include "hsakmt/hsakmtmodel.h"

int hsakmt_safe_env_to_int(const char* envvar, int default_val) {
  if (envvar == NULL) return default_val;
  char* endptr;
  errno = 0;
  long val = strtol(envvar, &endptr, 10);
  if (errno == ERANGE) return default_val;
  if (endptr == envvar) return default_val;
  // Allow trailing whitespace from shell/.env files; reject other trailing content.
  while (*endptr == ' ' || *endptr == '\t' || *endptr == '\n' || *endptr == '\r') {
    ++endptr;
  }
  if (*endptr != '\0') return default_val;
  if (val < INT_MIN || val > INT_MAX) return default_val;
  return (int)val;
}

/* Call ioctl, restarting if it is interrupted */
int hsakmt_ioctl(int fd, unsigned long request, void *arg)
{
	if (hsakmt_use_model)
		return model_kfd_ioctl(request, arg);

	int ret;

	do {
		ret = ioctl(fd, request, arg);
	} while (ret == -1 && (errno == EINTR || errno == EAGAIN));

	if (ret == -1 && errno == EBADF) {
		/* In case pthread_atfork didn't catch it, this will
		 * make any subsequent hsaKmt calls fail in CHECK_KFD_OPEN.
		 */
		pr_err("KFD file descriptor not valid in this process\n");
		hsakmt_is_forked_child();
	}

	return ret;
}
