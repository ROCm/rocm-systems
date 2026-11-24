/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * FreeBSD stand-in for <linux/ioctl.h>, so that the shared Linux UAPI copies
 * in include/hsakmt/linux/ (kfd_ioctl.h, udmabuf.h) build unmodified.  amdkfd
 * runs on FreeBSD through the LinuxKPI layer with the same ioctl ABI.
 *
 * <sys/ioccom.h> provides _IO/_IOR/_IOW/_IOWR.  libhsakmt also uses Linux's
 * decoding helpers, defined here for FreeBSD's encoding, which keeps the
 * argument size at bit 16 in a 13-bit field (IOCPARM_MASK).
 */
#ifndef HSAKMT_FREEBSD_LINUX_IOCTL_H
#define HSAKMT_FREEBSD_LINUX_IOCTL_H

#include <sys/ioccom.h>

#define _IOC_SIZESHIFT	16
#define _IOC_SIZEBITS	13
#define _IOC_TYPE(cmd)	IOCGROUP(cmd)
#define _IOC_NR(cmd)	((cmd) & 0xff)
#define _IOC_SIZE(cmd)	IOCPARM_LEN(cmd)

#endif /* HSAKMT_FREEBSD_LINUX_IOCTL_H */
