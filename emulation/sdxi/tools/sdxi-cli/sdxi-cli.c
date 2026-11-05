/*
 * sdxi-cli.c — command-line tool for the SDXI char device
 *
 * Subcommands:
 *   copy        <src> <dst>                   DMA copy a file
 *   compress    <src> <dst> [level]           zlib-compress via SDXI
 *   decompress  <src> <dst>                   zlib-decompress via SDXI
 *   roundtrip   <src>                         compress + decompress, verify
 *   bench       <copy|compress|decompress> <size_bytes> [iterations]
 *   stress      [iterations]                  compress/decompress across levels and patterns
 *   info                                      print device version, PCI IDs, and capabilities
 *   stats       [cid]                         print all 14 performance counters for context cid
 *
 * Usage: sdxi-cli [-d /dev/sdxi] [-v] <subcommand> [args...]
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <sys/ioctl.h>

#include "sdxi_user.h"

static int        g_fd      = -1;
static int        g_verbose = 0;
static const char *g_dev    = "/dev/sdxi";
static uint32_t   g_algo    = SDXI_ALGO_ZLIB;

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static void die(const char *msg)
{
	perror(msg);
	exit(1);
}

static void open_device(void)
{
	g_fd = open(g_dev, O_RDWR);
	if (g_fd < 0) {
		fprintf(stderr, "sdxi-cli: cannot open %s: %s\n",
			g_dev, strerror(errno));
		exit(1);
	}
}

static void *read_file(const char *path, size_t *len_out)
{
	FILE *f = fopen(path, "rb");
	if (!f) {
		fprintf(stderr, "sdxi-cli: cannot open %s: %s\n",
			path, strerror(errno));
		exit(1);
	}
	fseek(f, 0, SEEK_END);
	long sz = ftell(f);
	rewind(f);
	if (sz <= 0) {
		fclose(f);
		fprintf(stderr, "sdxi-cli: %s is empty or unreadable\n", path);
		exit(1);
	}
	void *buf = malloc((size_t)sz);
	if (!buf)
		die("malloc");
	if (fread(buf, 1, (size_t)sz, f) != (size_t)sz)
		die("fread");
	fclose(f);
	*len_out = (size_t)sz;
	return buf;
}

static void write_file(const char *path, const void *buf, size_t len)
{
	FILE *f = fopen(path, "wb");
	if (!f)
		die("fopen for write");
	if (fwrite(buf, 1, len, f) != len)
		die("fwrite");
	fclose(f);
}

static double elapsed_ms(struct timespec *start, struct timespec *end)
{
	return (double)(end->tv_sec  - start->tv_sec)  * 1000.0 +
	       (double)(end->tv_nsec - start->tv_nsec) / 1e6;
}

static double throughput_mbs(size_t bytes, double ms)
{
	return (ms > 0) ? ((double)bytes / (1024.0 * 1024.0)) / (ms / 1000.0) : 0.0;
}

/* ------------------------------------------------------------------ */
/* copy                                                                */
/* ------------------------------------------------------------------ */

static int cmd_copy(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: copy <src> <dst>\n");
		return 1;
	}

	size_t len;
	void *src = read_file(argv[0], &len);
	void *dst = malloc(len);
	if (!dst)
		die("malloc");

	struct sdxi_ioc_memcpy ioc = {
		.src = (uint64_t)(uintptr_t)src,
		.dst = (uint64_t)(uintptr_t)dst,
		.len = len,
	};

	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	if (ioctl(g_fd, SDXI_IOC_MEMCPY, &ioc) < 0)
		die("SDXI_IOC_MEMCPY");
	clock_gettime(CLOCK_MONOTONIC, &t1);

	write_file(argv[1], dst, len);

	if (g_verbose)
		printf("copy: %zu bytes in %.2f ms (%.1f MB/s)\n",
		       len, elapsed_ms(&t0, &t1),
		       throughput_mbs(len, elapsed_ms(&t0, &t1)));

	free(src);
	free(dst);
	return 0;
}

/* ------------------------------------------------------------------ */
/* compress                                                            */
/* ------------------------------------------------------------------ */

static int cmd_compress(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: compress <src> <dst> [level]\n");
		return 1;
	}

	uint32_t level = (argc >= 3) ? (uint32_t)atoi(argv[2]) : 0;

	size_t src_len;
	void *src = read_file(argv[0], &src_len);

	/* worst-case zlib bound */
	size_t dst_cap = src_len + src_len / 100 + 64;
	void *dst = malloc(dst_cap);
	if (!dst)
		die("malloc");

	struct sdxi_ioc_compress ioc = {
		.src     = (uint64_t)(uintptr_t)src,
		.src_len = src_len,
		.dst     = (uint64_t)(uintptr_t)dst,
		.dst_cap = dst_cap,
		.level   = level,
		.algo    = g_algo,
	};

	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	if (ioctl(g_fd, SDXI_IOC_COMPRESS, &ioc) < 0)
		die("SDXI_IOC_COMPRESS");
	clock_gettime(CLOCK_MONOTONIC, &t1);

	write_file(argv[1], dst, (size_t)ioc.out_len);

	if (g_verbose)
		printf("compress: %zu -> %llu bytes (%.1f%%) in %.2f ms (%.1f MB/s)\n",
		       src_len, (unsigned long long)ioc.out_len,
		       100.0 * (double)ioc.out_len / (double)src_len,
		       elapsed_ms(&t0, &t1),
		       throughput_mbs(src_len, elapsed_ms(&t0, &t1)));
	else
		printf("compressed: %zu -> %llu bytes\n",
		       src_len, (unsigned long long)ioc.out_len);

	free(src);
	free(dst);
	return 0;
}

/* ------------------------------------------------------------------ */
/* decompress                                                          */
/* ------------------------------------------------------------------ */

static int cmd_decompress(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: decompress <src> <dst>\n");
		return 1;
	}

	size_t src_len;
	void *src = read_file(argv[0], &src_len);

	/* Allocate generously; zlib can expand to at most ~1032x for pathological
	 * input.  For typical compressed data 32× is more than enough. */
	size_t dst_cap = src_len * 32;
	if (dst_cap < 65536)
		dst_cap = 65536;

	void *dst = malloc(dst_cap);
	if (!dst)
		die("malloc");

	struct sdxi_ioc_decompress ioc = {
		.src     = (uint64_t)(uintptr_t)src,
		.src_len = src_len,
		.dst     = (uint64_t)(uintptr_t)dst,
		.dst_cap = dst_cap,
		.algo    = g_algo,
	};

	struct timespec t0, t1;
	clock_gettime(CLOCK_MONOTONIC, &t0);
	if (ioctl(g_fd, SDXI_IOC_DECOMPRESS, &ioc) < 0) {
		int err = errno;
		fprintf(stderr, "sdxi-cli: SDXI_IOC_DECOMPRESS: %s\n",
			strerror(err));
		free(src);
		free(dst);
		return 1;
	}
	clock_gettime(CLOCK_MONOTONIC, &t1);

	write_file(argv[1], dst, (size_t)ioc.out_len);

	if (g_verbose)
		printf("decompress: %zu -> %llu bytes in %.2f ms (%.1f MB/s)\n",
		       src_len, (unsigned long long)ioc.out_len,
		       elapsed_ms(&t0, &t1),
		       throughput_mbs(ioc.out_len, elapsed_ms(&t0, &t1)));
	else
		printf("decompressed: %zu -> %llu bytes\n",
		       src_len, (unsigned long long)ioc.out_len);

	free(src);
	free(dst);
	return 0;
}

/* ------------------------------------------------------------------ */
/* roundtrip                                                           */
/* ------------------------------------------------------------------ */

static int cmd_roundtrip(int argc, char **argv)
{
	if (argc < 1) {
		fprintf(stderr, "usage: roundtrip <src>\n");
		return 1;
	}

	size_t src_len;
	void *src = read_file(argv[0], &src_len);

	size_t cmp_cap = src_len + src_len / 100 + 64;
	void *cmp_buf = malloc(cmp_cap);
	if (!cmp_buf)
		die("malloc");

	struct sdxi_ioc_compress cioc = {
		.src     = (uint64_t)(uintptr_t)src,
		.src_len = src_len,
		.dst     = (uint64_t)(uintptr_t)cmp_buf,
		.dst_cap = cmp_cap,
		.level   = 0,
		.algo    = g_algo,
	};

	if (ioctl(g_fd, SDXI_IOC_COMPRESS, &cioc) < 0)
		die("SDXI_IOC_COMPRESS");

	void *dec_buf = malloc(src_len);
	if (!dec_buf)
		die("malloc");

	struct sdxi_ioc_decompress dioc = {
		.src     = (uint64_t)(uintptr_t)cmp_buf,
		.src_len = cioc.out_len,
		.dst     = (uint64_t)(uintptr_t)dec_buf,
		.dst_cap = src_len,
		.algo    = g_algo,
	};

	if (ioctl(g_fd, SDXI_IOC_DECOMPRESS, &dioc) < 0)
		die("SDXI_IOC_DECOMPRESS");

	int ok = (dioc.out_len == src_len) &&
		 (memcmp(src, dec_buf, src_len) == 0);

	if (g_verbose || !ok) {
		printf("roundtrip: %zu bytes, ratio %.2fx, data %s\n",
		       src_len,
		       (double)src_len / (double)cioc.out_len,
		       ok ? "MATCH" : "MISMATCH");
	}

	free(src);
	free(cmp_buf);
	free(dec_buf);
	return ok ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* bench                                                               */
/* ------------------------------------------------------------------ */

static int cmd_bench(int argc, char **argv)
{
	if (argc < 2) {
		fprintf(stderr, "usage: bench <copy|compress|decompress> <size_bytes> [iterations]\n");
		return 1;
	}

	const char *op   = argv[0];
	size_t size      = (size_t)strtoull(argv[1], NULL, 0);
	int    iters     = (argc >= 3) ? atoi(argv[2]) : 10;

	if (size == 0 || iters <= 0) {
		fprintf(stderr, "bench: invalid size or iteration count\n");
		return 1;
	}

	void *src = malloc(size);
	void *dst = malloc(size + size / 100 + 64);
	if (!src || !dst)
		die("malloc");

	/* Fill with compressible pattern */
	for (size_t i = 0; i < size; i++)
		((uint8_t *)src)[i] = (uint8_t)(i & 0x7f);

	struct timespec t0, t1;
	double total_ms = 0.0;

	if (strcmp(op, "copy") == 0) {
		struct sdxi_ioc_memcpy ioc = {
			.src = (uint64_t)(uintptr_t)src,
			.dst = (uint64_t)(uintptr_t)dst,
			.len = size,
		};
		for (int i = 0; i < iters; i++) {
			clock_gettime(CLOCK_MONOTONIC, &t0);
			if (ioctl(g_fd, SDXI_IOC_MEMCPY, &ioc) < 0)
				die("SDXI_IOC_MEMCPY");
			clock_gettime(CLOCK_MONOTONIC, &t1);
			total_ms += elapsed_ms(&t0, &t1);
		}

	} else if (strcmp(op, "compress") == 0) {
		struct sdxi_ioc_compress ioc = {
			.src     = (uint64_t)(uintptr_t)src,
			.src_len = size,
			.dst     = (uint64_t)(uintptr_t)dst,
			.dst_cap = size + size / 100 + 64,
			.level   = 0,
			.algo    = g_algo,
		};
		for (int i = 0; i < iters; i++) {
			clock_gettime(CLOCK_MONOTONIC, &t0);
			if (ioctl(g_fd, SDXI_IOC_COMPRESS, &ioc) < 0)
				die("SDXI_IOC_COMPRESS");
			clock_gettime(CLOCK_MONOTONIC, &t1);
			total_ms += elapsed_ms(&t0, &t1);
		}

	} else if (strcmp(op, "decompress") == 0) {
		/* Pre-compress the data once */
		size_t cmp_cap = size + size / 100 + 64;
		void *cmp_buf  = malloc(cmp_cap);
		if (!cmp_buf)
			die("malloc");

		struct sdxi_ioc_compress cioc = {
			.src     = (uint64_t)(uintptr_t)src,
			.src_len = size,
			.dst     = (uint64_t)(uintptr_t)cmp_buf,
			.dst_cap = cmp_cap,
			.level   = 0,
			.algo    = g_algo,
		};
		if (ioctl(g_fd, SDXI_IOC_COMPRESS, &cioc) < 0)
			die("SDXI_IOC_COMPRESS (pre-compress for bench)");

		struct sdxi_ioc_decompress dioc = {
			.src     = (uint64_t)(uintptr_t)cmp_buf,
			.src_len = cioc.out_len,
			.dst     = (uint64_t)(uintptr_t)dst,
			.dst_cap = size,
			.algo    = g_algo,
		};
		for (int i = 0; i < iters; i++) {
			clock_gettime(CLOCK_MONOTONIC, &t0);
			if (ioctl(g_fd, SDXI_IOC_DECOMPRESS, &dioc) < 0)
				die("SDXI_IOC_DECOMPRESS");
			clock_gettime(CLOCK_MONOTONIC, &t1);
			total_ms += elapsed_ms(&t0, &t1);
		}
		free(cmp_buf);

	} else {
		fprintf(stderr, "bench: unknown op '%s' (copy|compress|decompress)\n", op);
		free(src);
		free(dst);
		return 1;
	}

	double avg_ms  = total_ms / iters;
	double avg_mbs = throughput_mbs(size, avg_ms);
	printf("bench %s: %zu bytes × %d iters, avg %.2f ms, %.1f MB/s\n",
	       op, size, iters, avg_ms, avg_mbs);

	free(src);
	free(dst);
	return 0;
}

/* ------------------------------------------------------------------ */
/* stress                                                              */
/* ------------------------------------------------------------------ */

/*
 * Exercise compress+decompress roundtrip across all algorithms, compression
 * levels (0,1,6,9) and two data patterns (compressible and pseudo-random) at
 * several buffer sizes.  Intended for in-VM validation of the full
 * compress/decompress path.
 */
static int cmd_stress(int argc, char **argv)
{
	static const size_t    sizes[]      = {1, 64, 4096, 65536, 1 << 20};
	static const uint32_t  levels[]     = {0, 1, 6, 9};
	static const uint32_t  algos[]      = {SDXI_ALGO_ZLIB, SDXI_ALGO_LZ4, SDXI_ALGO_ZSTD};
	static const char     *algo_names[] = {"zlib", "lz4", "zstd"};
	int iters = (argc >= 1) ? atoi(argv[0]) : 1;
	int pass = 0, fail = 0;

	if (iters <= 0) {
		fprintf(stderr, "stress: iterations must be > 0\n");
		return 1;
	}

	for (int it = 0; it < iters; it++) {
		for (size_t ai = 0; ai < sizeof(algos)/sizeof(algos[0]); ai++) {
			uint32_t algo = algos[ai];
			const char *algo_name = algo_names[ai];

			for (int pat = 0; pat < 2; pat++) {
				const char *pat_name = (pat == 0) ? "seq" : "rand";

				for (size_t si = 0; si < sizeof(sizes)/sizeof(sizes[0]); si++) {
					size_t sz = sizes[si];

					uint8_t *orig = malloc(sz);
					if (!orig)
						die("malloc");

					if (pat == 0) {
						for (size_t i = 0; i < sz; i++)
							orig[i] = (uint8_t)(i & 0x7f);
					} else {
						/* xorshift32 — reproducible pseudo-random */
						uint32_t s = 0xDEADBEEF;
						for (size_t i = 0; i < sz; i++) {
							s ^= s << 13;
							s ^= s >> 17;
							s ^= s << 5;
							orig[i] = (uint8_t)(s & 0xFF);
						}
					}

					for (size_t li = 0; li < sizeof(levels)/sizeof(levels[0]); li++) {
						uint32_t level = levels[li];

						/* worst-case compressed bound */
						size_t cmp_cap = sz + sz / 100 + 128;
						uint8_t *cmp_buf = malloc(cmp_cap);
						uint8_t *dec_buf = malloc(sz + 1);
						if (!cmp_buf || !dec_buf)
							die("malloc");

						struct sdxi_ioc_compress cioc = {
							.src     = (uint64_t)(uintptr_t)orig,
							.src_len = sz,
							.dst     = (uint64_t)(uintptr_t)cmp_buf,
							.dst_cap = cmp_cap,
							.level   = level,
							.algo    = algo,
						};

						int ok = 1;
						if (ioctl(g_fd, SDXI_IOC_COMPRESS, &cioc) < 0) {
							if (g_verbose)
								fprintf(stderr,
									"FAIL compress algo=%s pat=%s sz=%zu level=%u: %s\n",
									algo_name, pat_name, sz, level, strerror(errno));
							ok = 0;
						} else {
							struct sdxi_ioc_decompress dioc = {
								.src     = (uint64_t)(uintptr_t)cmp_buf,
								.src_len = cioc.out_len,
								.dst     = (uint64_t)(uintptr_t)dec_buf,
								.dst_cap = sz + 1,
								.algo    = algo,
							};
							if (ioctl(g_fd, SDXI_IOC_DECOMPRESS, &dioc) < 0) {
								if (g_verbose)
									fprintf(stderr,
										"FAIL decompress algo=%s pat=%s sz=%zu level=%u: %s\n",
										algo_name, pat_name, sz, level, strerror(errno));
								ok = 0;
							} else if (dioc.out_len != sz ||
								   memcmp(orig, dec_buf, sz) != 0) {
								if (g_verbose)
									fprintf(stderr,
										"FAIL data mismatch algo=%s pat=%s sz=%zu level=%u "
										"(out_len=%llu)\n",
										algo_name, pat_name, sz, level,
										(unsigned long long)dioc.out_len);
								ok = 0;
							} else if (g_verbose) {
								printf("PASS algo=%4s pat=%4s sz=%7zu level=%u: "
								       "%zu -> %llu bytes (%.2fx)\n",
								       algo_name, pat_name, sz, level, sz,
								       (unsigned long long)cioc.out_len,
								       sz > 0 ? (double)sz / (double)cioc.out_len
									     : 0.0);
							}
						}

						if (ok)
							pass++;
						else
							fail++;

						free(cmp_buf);
						free(dec_buf);
					}
					free(orig);
				}
			}
		}
	}

	int total = pass + fail;
	printf("stress: %d/%d passed", pass, total);
	if (iters > 1)
		printf(" (%d iteration%s)", iters, iters == 1 ? "" : "s");
	printf("\n");
	return (fail == 0) ? 0 : 1;
}

/* ------------------------------------------------------------------ */
/* elemwise                                                            */
/* ------------------------------------------------------------------ */

static uint32_t parse_dtype(const char *s)
{
	if (strcmp(s, "fp32")    == 0) return SDXI_DTYPE_FP32;
	if (strcmp(s, "fp16")    == 0) return SDXI_DTYPE_FP16;
	if (strcmp(s, "bf16")    == 0) return SDXI_DTYPE_BF16;
	if (strcmp(s, "int8")    == 0) return SDXI_DTYPE_INT8;
	if (strcmp(s, "int32")   == 0) return SDXI_DTYPE_INT32;
	if (strcmp(s, "fp8e4m3") == 0) return SDXI_DTYPE_FP8_E4M3;
	if (strcmp(s, "fp8e5m2") == 0) return SDXI_DTYPE_FP8_E5M2;
	if (strcmp(s, "int4")    == 0) return SDXI_DTYPE_INT4;
	fprintf(stderr, "sdxi-cli: unknown dtype '%s'\n", s);
	exit(1);
}

static uint64_t scalar_to_bits(const char *s, uint32_t dtype)
{
	switch (dtype) {
	case SDXI_DTYPE_FP32: {
		float f = strtof(s, NULL);
		uint32_t b; memcpy(&b, &f, 4); return b;
	}
	case SDXI_DTYPE_FP16: case SDXI_DTYPE_BF16:
		/* Accept as float, store raw bits from strtoul so user can pass hex. */
		return (uint64_t)strtoul(s, NULL, 0);
	case SDXI_DTYPE_INT8:  return (uint64_t)(int64_t)(int8_t)strtol(s, NULL, 0);
	case SDXI_DTYPE_INT32: return (uint64_t)(int64_t)(int32_t)strtol(s, NULL, 0);
	case SDXI_DTYPE_FP8_E4M3: case SDXI_DTYPE_FP8_E5M2:
		return (uint64_t)strtoul(s, NULL, 0);
	case SDXI_DTYPE_INT4:  return (uint64_t)(int64_t)strtol(s, NULL, 0);
	}
	return 0;
}

static int cmd_elemwise(int argc, char **argv)
{
	/* elemwise <op> <dtype> <scalar> <src-file> <dst-file> */
	if (argc < 5) {
		fprintf(stderr,
			"usage: elemwise <op> <dtype> <scalar> <src-file> <dst-file>\n"
			"  op:    add | min | max | reduce_min | reduce_max\n"
			"  dtype: fp32 | fp16 | bf16 | int8 | int32 | fp8e4m3 | fp8e5m2 | int4\n");
		return 1;
	}

	const char *op_str    = argv[0];
	const char *dtype_str = argv[1];
	const char *scalar_s  = argv[2];
	const char *src_path  = argv[3];
	const char *dst_path  = argv[4];

	uint32_t op;
	if      (strcmp(op_str, "add")        == 0) op = SDXI_ELEMOP_ADD;
	else if (strcmp(op_str, "min")        == 0) op = SDXI_ELEMOP_MIN;
	else if (strcmp(op_str, "max")        == 0) op = SDXI_ELEMOP_MAX;
	else if (strcmp(op_str, "reduce_min") == 0) op = SDXI_ELEMOP_REDUCE_MIN;
	else if (strcmp(op_str, "reduce_max") == 0) op = SDXI_ELEMOP_REDUCE_MAX;
	else {
		fprintf(stderr, "sdxi-cli: unknown op '%s'\n", op_str);
		return 1;
	}

	uint32_t dtype    = parse_dtype(dtype_str);
	uint64_t scalar   = scalar_to_bits(scalar_s, dtype);

	size_t src_len;
	uint8_t *src_buf = read_file(src_path, &src_len);
	if (!src_buf) return 1;

	/* dst buffer: for reduce ops one element (at most 4 bytes), else src_len. */
	bool is_reduce = (op == SDXI_ELEMOP_REDUCE_MIN || op == SDXI_ELEMOP_REDUCE_MAX);
	size_t dst_cap = is_reduce ? 8 : src_len;
	uint8_t *dst_buf = calloc(1, dst_cap);
	if (!dst_buf) { free(src_buf); perror("calloc"); return 1; }

	struct sdxi_ioc_elemwise ioc = {
		.src     = (uint64_t)(uintptr_t)src_buf,
		.src_len = src_len,
		.dst     = (uint64_t)(uintptr_t)dst_buf,
		.dtype   = dtype,
		.op      = op,
		.scalar  = scalar,
		.out_len = 0,
	};

	if (ioctl(g_fd, SDXI_IOC_ELEMWISE, &ioc) < 0) {
		perror("SDXI_IOC_ELEMWISE");
		free(src_buf); free(dst_buf);
		return 1;
	}

	if (g_verbose)
		fprintf(stderr, "elemwise: op=%s dtype=%s src=%zu dst=%llu bytes\n",
			op_str, dtype_str, src_len,
			(unsigned long long)ioc.out_len);

	write_file(dst_path, dst_buf, (size_t)ioc.out_len);

	if (is_reduce)
		printf("reduce result written to %s (%llu bytes)\n",
		       dst_path, (unsigned long long)ioc.out_len);

	free(src_buf);
	free(dst_buf);
	return 0;
}

/* ------------------------------------------------------------------ */
/* info                                                                */
/* ------------------------------------------------------------------ */

static int cmd_info(int argc, char **argv)
{
	(void)argc; (void)argv;
	struct sdxi_ioc_info inf = {0};
	if (ioctl(g_fd, SDXI_IOC_INFO, &inf) < 0) {
		perror("SDXI_IOC_INFO");
		return 1;
	}
	printf("SDXI Device Information\n");
	printf("  Version     : %u.%u\n",
	       (inf.version >> 8) & 0xffU, inf.version & 0xffU);
	printf("  Vendor ID   : 0x%04x\n", inf.vendor_id);
	printf("  Device ID   : 0x%04x\n", inf.device_id);
	printf("  Max contexts: %u\n", inf.max_cxt);
	printf("  CAP0        : 0x%016llx\n", (unsigned long long)inf.cap0);
	printf("  CAP1        : 0x%016llx\n", (unsigned long long)inf.cap1);
	return 0;
}

/* ------------------------------------------------------------------ */
/* stats                                                               */
/* ------------------------------------------------------------------ */

static const char *stat_names[SDXI_NUM_STATS] = {
	"bytes_moved",
	"num_copy",
	"num_errors",
	"num_intr",
	"num_cxt_start",
	"num_cxt_stop",
	"num_nop",
	"num_sync",
	"bytes_compressed",
	"bytes_decompressed",
	"num_compress",
	"num_decompress",
	"num_elemwise",
	"bytes_elemwise",
};

static int cmd_stats(int argc, char **argv)
{
	uint32_t cid = 0;
	if (argc >= 1)
		cid = (uint32_t)strtoul(argv[0], NULL, 0);

	struct sdxi_ioc_stats st = { .cid = cid };
	if (ioctl(g_fd, SDXI_IOC_STATS, &st) < 0) {
		perror("SDXI_IOC_STATS");
		return 1;
	}
	printf("SDXI Statistics (context %u)\n", cid);
	for (int i = 0; i < SDXI_NUM_STATS; i++)
		printf("  %-24s: %llu\n",
		       stat_names[i], (unsigned long long)st.val[i]);
	return 0;
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

static void usage(void)
{
	fprintf(stderr,
		"usage: sdxi-cli [-a <algo>] [-d <device>] [-v] <subcommand> [args]\n"
		"\n"
		"Subcommands:\n"
		"  copy        <src> <dst>\n"
		"  compress    <src> <dst> [level]\n"
		"  decompress  <src> <dst>\n"
		"  roundtrip   <src>\n"
		"  bench       <copy|compress|decompress> <size> [iters]\n"
		"  stress      [iters]   compress/decompress roundtrip across all algos, levels and patterns\n"
		"  elemwise    <op> <dtype> <scalar> <src-file> <dst-file>\n"
		"  info                  print device version, PCI IDs, and capabilities\n"
		"  stats       [cid]     print per-context performance counters (default cid=0)\n"
		"\n"
		"Options:\n"
		"  -a <algo>     compression algorithm: zlib (default), lz4, zstd\n"
		"  -d <device>   SDXI device path (default: /dev/sdxi)\n"
		"  -v            verbose output\n");
}

int main(int argc, char **argv)
{
	int opt;
	while ((opt = getopt(argc, argv, "+a:d:vh")) != -1) {
		switch (opt) {
		case 'a':
			if (strcmp(optarg, "zlib") == 0)       g_algo = SDXI_ALGO_ZLIB;
			else if (strcmp(optarg, "lz4") == 0)   g_algo = SDXI_ALGO_LZ4;
			else if (strcmp(optarg, "zstd") == 0)  g_algo = SDXI_ALGO_ZSTD;
			else {
				fprintf(stderr, "sdxi-cli: unknown algo '%s' (zlib|lz4|zstd)\n", optarg);
				return 1;
			}
			break;
		case 'd': g_dev     = optarg; break;
		case 'v': g_verbose = 1;      break;
		case 'h': usage(); return 0;
		default:  usage(); return 1;
		}
	}

	if (optind >= argc) {
		usage();
		return 1;
	}

	const char *cmd  = argv[optind];
	int         rem  = argc - optind - 1;
	char      **rest = argv + optind + 1;

	open_device();

	int rc;
	if      (strcmp(cmd, "copy")       == 0) rc = cmd_copy(rem, rest);
	else if (strcmp(cmd, "compress")   == 0) rc = cmd_compress(rem, rest);
	else if (strcmp(cmd, "decompress") == 0) rc = cmd_decompress(rem, rest);
	else if (strcmp(cmd, "roundtrip")  == 0) rc = cmd_roundtrip(rem, rest);
	else if (strcmp(cmd, "bench")      == 0) rc = cmd_bench(rem, rest);
	else if (strcmp(cmd, "stress")     == 0) rc = cmd_stress(rem, rest);
	else if (strcmp(cmd, "elemwise")   == 0) rc = cmd_elemwise(rem, rest);
	else if (strcmp(cmd, "info")       == 0) rc = cmd_info(rem, rest);
	else if (strcmp(cmd, "stats")      == 0) rc = cmd_stats(rem, rest);
	else {
		fprintf(stderr, "sdxi-cli: unknown subcommand '%s'\n", cmd);
		usage();
		rc = 1;
	}

	close(g_fd);
	return rc;
}
