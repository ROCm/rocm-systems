/* SPDX-License-Identifier: GPL-2.0 WITH Linux-syscall-note */
/*
 * UAPI for /dev/sdxi<N> — SDXI userspace char device.
 *
 * Shared between the kernel module (sdxi_char.c) and userspace tools.
 * Userspace: include <linux/types.h> or define __u32/__u64 from <stdint.h>.
 */
#ifndef _UAPI_LINUX_SDXI_USER_H
#define _UAPI_LINUX_SDXI_USER_H

#ifdef __KERNEL__
#include <linux/ioctl.h>
#include <linux/types.h>
#else
#include <sys/ioctl.h>
#include <stdint.h>
typedef uint32_t __u32;
typedef uint64_t __u64;
#endif

/*
 * SDXI_IOC_MEMCPY — copy len bytes from src to dst.
 *
 * Both src and dst are userspace virtual addresses of buffers at least
 * len bytes in size.
 */
struct sdxi_ioc_memcpy {
	__u64 src;    /* source userspace virtual address */
	__u64 dst;    /* destination userspace virtual address */
	__u64 len;    /* byte count */
};

/* Compression algorithm selector passed in ioc structs. */
#define SDXI_ALGO_ZLIB  0U   /* zlib/deflate (default) */
#define SDXI_ALGO_LZ4   1U   /* LZ4 lossless */
#define SDXI_ALGO_ZSTD  2U   /* Zstandard lossless */

/*
 * SDXI_IOC_COMPRESS — compress src_len bytes from src into dst.
 *
 * dst_cap must be large enough for the worst-case compressed output;
 * src_len + src_len/100 + 64 is sufficient for all supported algorithms.
 * On success out_len is filled with the actual compressed size.
 * level 0 = default, 1–9 = algorithm-specific level.
 * algo: SDXI_ALGO_ZLIB / SDXI_ALGO_LZ4 / SDXI_ALGO_ZSTD.
 */
struct sdxi_ioc_compress {
	__u64 src;      /* source userspace virtual address */
	__u64 src_len;  /* bytes to compress */
	__u64 dst;      /* destination userspace virtual address */
	__u64 dst_cap;  /* capacity of dst buffer */
	__u32 level;    /* 0 = default; 1–9 = algo-specific level */
	__u32 algo;     /* SDXI_ALGO_* */
	__u64 out_len;  /* [out] actual compressed bytes */
};

/*
 * SDXI_IOC_DECOMPRESS — decompress src_len bytes from src into dst.
 *
 * dst_cap must be >= the original uncompressed size.
 * algo must match the value used to compress the data.
 * On success out_len is filled with the actual decompressed size.
 */
struct sdxi_ioc_decompress {
	__u64 src;      /* compressed source, userspace virtual address */
	__u64 src_len;  /* compressed byte count */
	__u64 dst;      /* destination userspace virtual address */
	__u64 dst_cap;  /* capacity of dst buffer */
	__u32 algo;     /* SDXI_ALGO_* — must match compression algo */
	__u32 _pad;
	__u64 out_len;  /* [out] actual decompressed bytes */
};

/* Element data type selector. */
#define SDXI_DTYPE_FP32      0U
#define SDXI_DTYPE_FP16      1U
#define SDXI_DTYPE_BF16      2U
#define SDXI_DTYPE_INT8      3U
#define SDXI_DTYPE_INT32     4U
#define SDXI_DTYPE_FP8_E4M3  5U
#define SDXI_DTYPE_FP8_E5M2  6U
#define SDXI_DTYPE_INT4      7U

/* Element operation selector. */
#define SDXI_ELEMOP_ADD         0U
#define SDXI_ELEMOP_MIN         1U
#define SDXI_ELEMOP_MAX         2U
#define SDXI_ELEMOP_REDUCE_MIN  3U
#define SDXI_ELEMOP_REDUCE_MAX  4U

struct sdxi_ioc_elemwise {
	__u64 src;        /* source userspace virtual address */
	__u64 src_len;    /* source byte count */
	__u64 dst;        /* destination userspace virtual address */
	__u32 dtype;      /* SDXI_DTYPE_* */
	__u32 op;         /* SDXI_ELEMOP_* */
	__u64 scalar;     /* scalar as dtype-native bits (ignored for reduce ops) */
	__u64 out_len;    /* [out] dst byte count written */
};

#define SDXI_IOC_MAGIC       'X'
#define SDXI_IOC_MEMCPY      _IOW (SDXI_IOC_MAGIC, 1, struct sdxi_ioc_memcpy)
#define SDXI_IOC_COMPRESS    _IOWR(SDXI_IOC_MAGIC, 2, struct sdxi_ioc_compress)
#define SDXI_IOC_DECOMPRESS  _IOWR(SDXI_IOC_MAGIC, 3, struct sdxi_ioc_decompress)
#define SDXI_IOC_ELEMWISE    _IOWR(SDXI_IOC_MAGIC, 4, struct sdxi_ioc_elemwise)

#endif /* _UAPI_LINUX_SDXI_USER_H */
