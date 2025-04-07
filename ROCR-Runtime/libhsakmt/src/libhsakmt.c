#include <stdio.h>
#include <errno.h>
#include <sys/ioctl.h>

#include "libhsakmt.h"
#include "hsakmt/hsakmtmodel.h"

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
