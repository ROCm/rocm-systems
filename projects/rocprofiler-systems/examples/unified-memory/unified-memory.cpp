// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Unified Memory KFD Events Test
//
// Comprehensive HIP program designed to trigger all categories of KFD events:
//   1. Page Faults        – read/write faults (migrated & updated)
//   2. Page Migrations    – prefetch, GPU pagefault, CPU pagefault
//   3. Queue Evictions    – SVM evictions under memory pressure
//   4. Unmap from GPU     – unmapping ranges previously resident on GPU
//
// Usage: unified-memory [OPTIONS]
//   -s, --size <MB>       Per-allocation size in MB (default: 64)
//   -p, --pressure <MB>   Total managed memory for pressure test (default: 512)
//   -d, --device <ID>     GPU device ID (default: 0)
//   -i, --iterations <N>  Iterations for ping-pong tests (default: 8)
//   -a, --all             Run all tests (default: first NUM_DEFAULT_TESTS only)
//   -h, --help            Show this help message

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <linux/types.h>
#include <numeric>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Minimal KFD SVM ioctl bindings. These mirror fields from
// <linux/kfd_ioctl.h>; the UAPI is stable across recent kernels w.r.t. SVM op
// 0x20. Used by Test 7 to set KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED on a user
// range so the QUEUE_EVICT_SVM emission path is reachable under HSA_XNACK=1.
// ---------------------------------------------------------------------------
#ifndef KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED
#    define KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED 0x00000040
#endif

enum kfd_svm_op_local
{
    KFD_SVM_OP_SET_ATTR_LOCAL = 0
};

enum kfd_svm_attr_type_local
{
    KFD_SVM_ATTR_SET_FLAGS_LOCAL = 5,
    KFD_SVM_ATTR_CLR_FLAGS_LOCAL = 6
};

struct kfd_svm_attribute_local
{
    __u32 type;
    __u32 value;
};

struct kfd_svm_args_header
{
    __u64 start_addr;
    __u64 size;
    __u32 op;
    __u32 nattr;
    // attrs[] follows
};

#define AMDKFD_IOCTL_BASE_LOCAL 'K'
#define AMDKFD_IOC_SVM_LOCAL                                                             \
    _IOWR(AMDKFD_IOCTL_BASE_LOCAL, 0x20, struct kfd_svm_args_header)

#define HIP_CHECK(cmd)                                                                   \
    do                                                                                   \
    {                                                                                    \
        hipError_t error = (cmd);                                                        \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            fprintf(stderr, "HIP error: %s (%d) at %s:%d\n", hipGetErrorString(error),   \
                    error, __FILE__, __LINE__);                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while(0)

// ---------------------------------------------------------------------------
// GPU Kernels
// ---------------------------------------------------------------------------

__global__ void
kern_write_pattern(uint64_t* data, size_t n, uint64_t pattern)
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = idx; i < n; i += stride)
        data[i] = pattern + i;
}

__global__ void
kern_read_reduce(const uint64_t* data, size_t n, uint64_t* result)
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    uint64_t local_sum = 0;
    for(size_t i = idx; i < n; i += stride)
        local_sum += data[i];

    atomicAdd(reinterpret_cast<unsigned long long*>(result), local_sum);
}

__global__ void
kern_read_write_stencil(uint64_t* dst, const uint64_t* src, size_t n)
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = idx; i < n; i += stride)
        dst[i] = src[i] * 3 + 1;
}

__global__ void
kern_saxpy(float* y, const float* x, float a, size_t n)
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = idx; i < n; i += stride)
        y[i] = a * x[i] + y[i];
}

__global__ void
kern_touch_pages(uint64_t* data, size_t n, size_t page_stride)
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = idx * page_stride; i < n; i += stride * page_stride)
        data[i] += 1;
}

// Long-running kernel that does many LCG iterations per element. Used by
// Test 7 to keep the GPU queue actively executing on a GPU_ALWAYS_MAPPED
// SVM range long enough for racing mprotect() invalidations to force
// kfd_smi_event_queue_eviction(KFD_QUEUE_EVICTION_TRIGGER_SVM).
__global__ void
kern_compute_heavy(uint64_t* data, size_t n, uint64_t inner_iters)
{
    size_t idx    = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = idx; i < n; i += stride)
    {
        uint64_t v = data[i];
        for(uint64_t j = 0; j < inner_iters; j++)
            v = v * 6364136223846793005ULL + 1442695040888963407ULL;
        data[i] = v;
    }
}

// ---------------------------------------------------------------------------
static constexpr int NUM_DEFAULT_TESTS = 9;
static constexpr int NUM_ALL_TESTS     = 19;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct Config
{
    size_t alloc_size_mb = 64;
    size_t pressure_mb   = 512;
    int    device_id     = 0;
    int    iterations    = 8;
    bool   run_all       = false;
};

static void
print_usage(const char* prog)
{
    printf("Unified Memory KFD Events Test\n\n");
    printf("Usage: %s [OPTIONS]\n\n", prog);
    printf("Options:\n");
    printf("  -s, --size <MB>       Per-allocation size in MB (default: 64)\n");
    printf("  -p, --pressure <MB>   Managed memory for pressure test (default: 512)\n");
    printf("  -d, --device <ID>     GPU device ID (default: 0)\n");
    printf("  -i, --iterations <N>  Ping-pong iterations (default: 8)\n");
    printf("  -a, --all             Run all %d tests (default: first %d only)\n",
           NUM_ALL_TESTS, NUM_DEFAULT_TESTS);
    printf("  -h, --help            Show this help message\n");
}

static Config
parse_args(int argc, char** argv)
{
    Config cfg;
    for(int i = 1; i < argc; i++)
    {
        if(strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            print_usage(argv[0]);
            exit(0);
        }
        else if((strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--size") == 0) &&
                i + 1 < argc)
            cfg.alloc_size_mb = strtoull(argv[++i], nullptr, 10);
        else if((strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--pressure") == 0) &&
                i + 1 < argc)
            cfg.pressure_mb = strtoull(argv[++i], nullptr, 10);
        else if((strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--device") == 0) &&
                i + 1 < argc)
            cfg.device_id = atoi(argv[++i]);
        else if((strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) &&
                i + 1 < argc)
            cfg.iterations = atoi(argv[++i]);
        else if(strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--all") == 0)
            cfg.run_all = true;
        else
        {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            exit(1);
        }
    }
    return cfg;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static constexpr int BLOCK_SIZE = 256;

static int
grid_size(size_t n)
{
    return static_cast<int>(std::min<size_t>((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 65535));
}

static void
banner(const char* title)
{
    printf("\n========== %s ==========\n", title);
}

// ---------------------------------------------------------------------------
// Test 1: GPU Read Fault – CPU writes, GPU reads managed memory
//
// KFD events expected:
//   PAGE_FAULT_READ_FAULT_MIGRATED  (GPU reads pages owned by CPU)
//   PAGE_MIGRATE_PAGEFAULT_GPU      (pages migrate CPU→GPU on demand)
// ---------------------------------------------------------------------------

static void
test_gpu_read_fault(size_t bytes, int device)
{
    banner("Test 1: GPU Read Fault (CPU write → GPU read)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    for(size_t i = 0; i < n; i++)
        managed[i] = i;
    printf("  CPU initialized %zu MB of managed memory\n", bytes >> 20);

    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    printf("  Prefetched to GPU (avoids demand-fault avalanche under XNACK)\n");

    uint64_t* d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint64_t)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(uint64_t)));

    kern_read_reduce<<<grid_size(n), BLOCK_SIZE>>>(managed, n, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t result = 0;
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(uint64_t), hipMemcpyDeviceToHost));
    printf("  GPU reduction result: %lu\n", result);

    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 2: GPU Write Fault – GPU writes to fresh managed memory
//
// KFD events expected:
//   PAGE_FAULT_WRITE_FAULT_MIGRATED (GPU writes to pages never touched)
// ---------------------------------------------------------------------------

static void
test_gpu_write_fault(size_t bytes, int device)
{
    banner("Test 2: GPU Write Fault (GPU writes fresh managed memory)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    kern_write_pattern<<<grid_size(n), BLOCK_SIZE>>>(managed, n, 0xDEAD);
    HIP_CHECK(hipDeviceSynchronize());
    printf("  GPU wrote pattern to %zu MB of managed memory\n", bytes >> 20);

    uint64_t check = managed[0];
    printf("  CPU verification: managed[0] = 0x%lx (expected 0x%lx)\n", check,
           (uint64_t) 0xDEAD);

    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 3: CPU Read Fault – GPU writes, CPU reads back
//
// KFD events expected:
//   PAGE_FAULT_READ_FAULT_MIGRATED  (CPU reads GPU-resident pages)
//   PAGE_MIGRATE_PAGEFAULT_CPU      (pages migrate GPU→CPU on demand)
//   UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE (GPU MMU unmaps during migration)
// ---------------------------------------------------------------------------

static void
test_cpu_read_fault(size_t bytes, int device)
{
    banner("Test 3: CPU Read Fault (GPU write → CPU read)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    kern_write_pattern<<<grid_size(n), BLOCK_SIZE>>>(managed, n, 0xBEEF);
    HIP_CHECK(hipDeviceSynchronize());
    printf("  GPU wrote pattern to %zu MB\n", bytes >> 20);

    uint64_t sum = 0;
    for(size_t i = 0; i < std::min<size_t>(n, 4096); i++)
        sum += managed[i];
    printf("  CPU read-back sum (first 4096 elements): %lu\n", sum);

    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 4: CPU Write Fault – GPU writes, then CPU modifies
//
// KFD events expected:
//   PAGE_FAULT_WRITE_FAULT_MIGRATED (CPU writes to GPU-resident pages)
//   PAGE_MIGRATE_PAGEFAULT_CPU      (pages migrate GPU→CPU)
//   UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE
// ---------------------------------------------------------------------------

static void
test_cpu_write_fault(size_t bytes, int /*device*/)
{
    banner("Test 4: CPU Write Fault (GPU write → CPU write)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    kern_write_pattern<<<grid_size(n), BLOCK_SIZE>>>(managed, n, 0xCAFE);
    HIP_CHECK(hipDeviceSynchronize());
    printf("  GPU wrote pattern to %zu MB\n", bytes >> 20);

    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, hipCpuDeviceId, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    printf("  Prefetched to CPU (avoids demand-fault avalanche under XNACK)\n");

    for(size_t i = 0; i < n; i++)
        managed[i] = i * 2;
    printf("  CPU overwrote all %zu MB\n", bytes >> 20);

    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 5: Explicit Prefetch – hipMemPrefetchAsync
//
// KFD events expected:
//   PAGE_MIGRATE_PREFETCH (explicit migration via prefetch API)
// ---------------------------------------------------------------------------

static void
test_prefetch(size_t bytes, int device)
{
    banner("Test 5: Explicit Prefetch (hipMemPrefetchAsync)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    for(size_t i = 0; i < n; i++)
        managed[i] = i;
    printf("  CPU initialized %zu MB\n", bytes >> 20);

    printf("  Prefetching CPU → GPU %d ...\n", device);
    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t* d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint64_t)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(uint64_t)));

    kern_read_reduce<<<grid_size(n), BLOCK_SIZE>>>(managed, n, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t result = 0;
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(uint64_t), hipMemcpyDeviceToHost));
    printf("  GPU read after prefetch, sum = %lu\n", result);

    printf("  Prefetching GPU → CPU ...\n");
    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, hipCpuDeviceId, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t host_check = managed[n / 2];
    printf("  CPU read after prefetch-back, managed[%zu] = %lu\n", n / 2, host_check);

    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 6: Ping-Pong Migration – alternate CPU/GPU access
//
// KFD events expected:
//   PAGE_FAULT_READ_FAULT_MIGRATED / WRITE_FAULT_MIGRATED (both directions)
//   PAGE_MIGRATE_PAGEFAULT_GPU / PAGEFAULT_CPU (alternating)
//   UNMAP_FROM_GPU_MMU_NOTIFY_MIGRATE (each GPU→CPU transition)
// ---------------------------------------------------------------------------

static void
test_pingpong(size_t bytes, int device, int iterations)
{
    banner("Test 6: Ping-Pong Migration (alternating CPU/GPU access)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    for(int iter = 0; iter < iterations; iter++)
    {
        if(iter % 2 == 0)
        {
            HIP_CHECK(hipMemPrefetchAsync(managed, bytes, hipCpuDeviceId, nullptr));
            HIP_CHECK(hipDeviceSynchronize());
            for(size_t i = 0; i < n; i++)
                managed[i] = iter + i;
            printf("  [iter %d] CPU wrote %zu MB\n", iter, bytes >> 20);
        }
        else
        {
            HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
            HIP_CHECK(hipDeviceSynchronize());
            kern_write_pattern<<<grid_size(n), BLOCK_SIZE>>>(managed, n, iter * 1000ULL);
            HIP_CHECK(hipDeviceSynchronize());
            printf("  [iter %d] GPU wrote %zu MB\n", iter, bytes >> 20);
        }
    }

    HIP_CHECK(hipFree(managed));
    printf("  DONE (%d ping-pong iterations)\n", iterations);
}

// ---------------------------------------------------------------------------
// Test 7: Emits QUEUE_EVICT_SVM by forcing GPU_ALWAYS_MAPPED via raw KFD SVM ioctl
//         Deterministically fires QUEUE_EVICT_SVM under HSA_XNACK=1
// ---------------------------------------------------------------------------

static int
svm_set_flags(int kfd_fd, void* addr, size_t size, unsigned set, unsigned clr)
{
    constexpr int            NATTR         = 2;
    constexpr size_t         ATTRS_LEN     = NATTR * sizeof(kfd_svm_attribute_local);
    constexpr size_t         ARGS_LEN      = sizeof(kfd_svm_args_header) + ATTRS_LEN;
    alignas(8) unsigned char buf[ARGS_LEN] = { 0 };
    auto*                    hdr           = reinterpret_cast<kfd_svm_args_header*>(buf);
    auto*                    atts =
        reinterpret_cast<kfd_svm_attribute_local*>(buf + sizeof(kfd_svm_args_header));

    hdr->start_addr = reinterpret_cast<__u64>(addr);
    hdr->size       = static_cast<__u64>(size);
    hdr->op         = KFD_SVM_OP_SET_ATTR_LOCAL;
    hdr->nattr      = NATTR;
    atts[0].type    = KFD_SVM_ATTR_SET_FLAGS_LOCAL;
    atts[0].value   = set;
    atts[1].type    = KFD_SVM_ATTR_CLR_FLAGS_LOCAL;
    atts[1].value   = clr;

    // The KFD SVM ioctl uses a variable-length payload (header + attrs[]).
    // The kernel does ONE copy_from_user using _IOC_SIZE(cmd), so we must
    // *inflate* the ioctl number's encoded size by the attrs payload length.
    // This is how libhsakmt issues it (libhsakmt/src/svm.c:96).
    unsigned long cmd =
        AMDKFD_IOC_SVM_LOCAL + (static_cast<unsigned long>(ATTRS_LEN) << _IOC_SIZESHIFT);

    int r = ioctl(kfd_fd, cmd, buf);
    return (r < 0) ? -errno : 0;
}

static void
test_force_gpu_always_mapped(size_t bytes, int device, int iterations)
{
    banner("Test 7: Force GPU_ALWAYS_MAPPED via raw KFD SVM ioctl "
           "(QUEUE_EVICT_SVM under XNACK=1 on discrete GPUs)");

    HIP_CHECK(hipSetDevice(device));

    int kfd_fd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    if(kfd_fd < 0)
    {
        printf("  Skipping: cannot open /dev/kfd: %s\n", strerror(errno));
        return;
    }

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    for(size_t i = 0; i < n; i++)
        managed[i] = i;

    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    printf("  Allocated %zu MB managed memory, prefetched to GPU %d\n", bytes >> 20,
           device);

    int r =
        svm_set_flags(kfd_fd, managed, bytes, KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED, 0);
    if(r != 0)
    {
        printf("  KFD SVM SET_ATTR(GPU_ALWAYS_MAPPED) failed: %s\n", strerror(-r));
        printf("  This means the OR-arm of svm_range_evict() cannot be hit.\n"
               "  Falling back to skip.\n");
        HIP_CHECK(hipFree(managed));
        close(kfd_fd);
        return;
    }
    printf("  KFD SET_ATTR succeeded: GPU_ALWAYS_MAPPED is now set on the "
           "range.\n");

    long   pgsize_l = sysconf(_SC_PAGESIZE);
    size_t pgsize   = (pgsize_l > 0) ? static_cast<size_t>(pgsize_l) : 4096UL;
    auto   base     = reinterpret_cast<uintptr_t>(managed);
    auto   pg_start = base & ~(pgsize - 1);                         // align start down
    auto   pg_end   = (base + bytes + pgsize - 1) & ~(pgsize - 1);  // align end up
    void*  mp_addr  = reinterpret_cast<void*>(pg_start);
    size_t mp_len   = pg_end - pg_start;

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));
    constexpr uint64_t INNER_ITERS = 5000ULL;
    int                grid        = grid_size(n);

    int total_cycles = 0;
    for(int iter = 0; iter < iterations; iter++)
    {
        kern_compute_heavy<<<grid, BLOCK_SIZE, 0, stream>>>(managed, n, INNER_ITERS);

        // Each mprotect cycle triggers an MMU_NOTIFY_PROTECTION_VMA invalidation
        // on the live SVM range. Because the range now has GPU_ALWAYS_MAPPED,
        // svm_range_evict() takes the quiesce branch and emits
        // KFD_QUEUE_EVICTION_TRIGGER_SVM (one QUEUE_EVICT_SVM event in rocpd).
        constexpr int CYCLES_PER_IT    = 8;
        int           cycles_this_iter = 0;
        for(int c = 0; c < CYCLES_PER_IT; c++)
        {
            if(mprotect(mp_addr, mp_len, PROT_READ) != 0)
            {
                printf("  mprotect(PROT_READ) failed: %s\n", strerror(errno));
                break;
            }
            if(mprotect(mp_addr, mp_len, PROT_READ | PROT_WRITE) != 0)
            {
                printf("  mprotect(PROT_READ|PROT_WRITE) failed: %s\n", strerror(errno));
                break;
            }
            cycles_this_iter++;
            total_cycles++;
        }

        HIP_CHECK(hipStreamSynchronize(stream));
        printf("  [iter %d] %d/%d mprotect cycles completed on always-mapped range "
               "while kernel in flight\n",
               iter, cycles_this_iter, CYCLES_PER_IT);
    }

    // Clear the flag before freeing so the runtime can release the range
    // through its normal codepath.
    r = svm_set_flags(kfd_fd, managed, bytes, 0, KFD_IOCTL_SVM_FLAG_GPU_ALWAYS_MAPPED);
    if(r != 0) printf("  Warning: KFD CLR_FLAGS returned %s\n", strerror(-r));

    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(managed));
    close(kfd_fd);

    printf("  Total mprotect cycles: %d. Expect ~%d QUEUE_EVICT_SVM events "
           "from this test.\n",
           total_cycles, total_cycles);
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 8: UNMAP_FROM_GPU_MMU_NOTIFY event is triggered by madvise(MADV_DONTNEED) on an
// SVM range that is currently mapped on the GPU.
// ---------------------------------------------------------------------------

static void
test_madvise_unmap_from_gpu(size_t bytes, int device)
{
    banner("Test 8: madvise(MADV_DONTNEED) -> UNMAP_FROM_GPU_MMU_NOTIFY");
    long   pgsize_l = sysconf(_SC_PAGESIZE);
    size_t pgsize   = (pgsize_l > 0) ? static_cast<size_t>(pgsize_l) : 4096;
    bytes           = (bytes + pgsize - 1) & ~(pgsize - 1);

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));
    if(reinterpret_cast<uintptr_t>(managed) % pgsize != 0)
    {
        printf("  SKIP: hipMallocManaged returned non-page-aligned pointer %p "
               "(alignment %zu required for madvise).\n",
               static_cast<void*>(managed), pgsize);
        HIP_CHECK(hipFree(managed));
        return;
    }

    for(size_t i = 0; i < n; i++)
        managed[i] = i;

    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    printf("  Prefetched %zu MB to GPU %d (GPU PTE now populated)\n", bytes >> 20,
           device);

    constexpr int CYCLES         = 8;
    size_t        elems_per_page = pgsize / sizeof(uint64_t);
    int           success        = 0;
    for(int c = 0; c < CYCLES; c++)
    {
        if(madvise(managed, bytes, MADV_DONTNEED) != 0)
        {
            printf("  madvise(MADV_DONTNEED) failed: %s\n", strerror(errno));
            break;
        }

        for(size_t i = 0; i < n; i += elems_per_page)
            managed[i] = i;
        HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
        HIP_CHECK(hipDeviceSynchronize());
        success++;
    }

    HIP_CHECK(hipFree(managed));
    printf("  Total madvise cycles: %d. Expect ~%d UNMAP_FROM_GPU_MMU_NOTIFY "
           "events.\n",
           success, success);
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 9: Simulates a GPU read fault that drives KFD's recoverable-fault path to
// migrate the page CPU to VRAM.
// ---------------------------------------------------------------------------

static void
test_xnack_read_fault_migrated(size_t bytes, int device)
{
    banner("Test 9: XNACK read fault -> PAGE_FAULT_READ_FAULT_MIGRATED");

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

    const char* xnack_env    = getenv("HSA_XNACK");
    bool        xnack_env_on = (xnack_env && strcmp(xnack_env, "1") == 0);
    bool        xnack_arch   = (strstr(props.gcnArchName, "xnack+") != nullptr);

    printf("  GPU arch: %s\n", props.gcnArchName);
    printf("  HSA_XNACK: %s\n", xnack_env ? xnack_env : "(not set)");

    if(!xnack_arch || !xnack_env_on)
    {
        printf("  SKIP: needs xnack+ ISA AND HSA_XNACK=1.\n"
               "        (current arch is %s, HSA_XNACK=%s)\n"
               "        On xnack- hardware, GPU loads against unmapped SVM "
               "are UB,\n"
               "        so no recoverable fault reaches the KFD handler.\n",
               xnack_arch ? "xnack+" : "xnack-/unknown", xnack_env_on ? "1" : "0");
        printf("  DONE\n");
        return;
    }

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    for(size_t i = 0; i < n; i++)
        managed[i] = i;
    printf("  CPU initialized %zu MB of managed memory (no prefetch)\n", bytes >> 20);

    uint64_t* d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint64_t)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(uint64_t)));

    printf("  Launching read-only kernel; each wave's first global load on a\n"
           "  4 KiB page triggers a recoverable read fault.\n");
    kern_read_reduce<<<grid_size(n), BLOCK_SIZE>>>(managed, n, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t result = 0;
    HIP_CHECK(hipMemcpy(&result, d_result, sizeof(uint64_t), hipMemcpyDeviceToHost));
    printf("  GPU reduction result: %lu\n", static_cast<unsigned long>(result));

    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(managed));

    long   pgsize_l       = sysconf(_SC_PAGESIZE);
    size_t pgsize         = (pgsize_l > 0) ? static_cast<size_t>(pgsize_l) : 4096;
    size_t expected_pages = bytes / pgsize;
    printf("  %zu pages backed in sysmem; expect up to %zu "
           "READ_FAULT_MIGRATED events.\n",
           expected_pages, expected_pages);
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 10: Memory Advise Hints
//
// Exercises hipMemAdvise with various hint flags before access to influence
// migration behavior. Expected KFD events vary by hint and access pattern.
// ---------------------------------------------------------------------------

static void
test_mem_advise(size_t bytes, int device)
{
    banner("Test 10: Memory Advise Hints (hipMemAdvise)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    printf("  Setting hipMemAdviseSetPreferredLocation → GPU %d\n", device);
    HIP_CHECK(hipMemAdvise(managed, bytes, hipMemAdviseSetPreferredLocation, device));

    for(size_t i = 0; i < n; i++)
        managed[i] = i;

    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    kern_write_pattern<<<grid_size(n), BLOCK_SIZE>>>(managed, n, 0xA1);
    HIP_CHECK(hipDeviceSynchronize());
    printf("  GPU accessed memory with preferred-location hint\n");

    printf("  Setting hipMemAdviseSetAccessedBy → CPU\n");
    HIP_CHECK(hipMemAdvise(managed, bytes, hipMemAdviseSetAccessedBy, hipCpuDeviceId));

    uint64_t sum = 0;
    for(size_t i = 0; i < std::min<size_t>(n, 4096); i++)
        sum += managed[i];
    printf("  CPU read under accessed-by hint, partial sum = %lu\n", sum);

    printf("  Unsetting hints\n");
    HIP_CHECK(hipMemAdvise(managed, bytes, hipMemAdviseUnsetPreferredLocation, device));
    HIP_CHECK(hipMemAdvise(managed, bytes, hipMemAdviseUnsetAccessedBy, hipCpuDeviceId));

    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 11: Read-Only Access Pattern
//
// Sets hipMemAdviseSetReadMostly and has both CPU and GPU read concurrently.
// This should cause read-fault events with "updated" semantics instead of
// "migrated" because both can keep read-only copies.
//
// KFD events expected:
//   PAGE_FAULT_READ_FAULT_UPDATED (read-replicated pages)
// ---------------------------------------------------------------------------

static void
test_read_mostly(size_t bytes, int device)
{
    banner("Test 11: Read-Mostly Pattern (concurrent read access)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    for(size_t i = 0; i < n; i++)
        managed[i] = i * 7;

    printf("  Setting hipMemAdviseSetReadMostly\n");
    HIP_CHECK(hipMemAdvise(managed, bytes, hipMemAdviseSetReadMostly, device));

    HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t* d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint64_t)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(uint64_t)));

    kern_read_reduce<<<grid_size(n), BLOCK_SIZE>>>(managed, n, d_result);
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t gpu_sum = 0;
    HIP_CHECK(hipMemcpy(&gpu_sum, d_result, sizeof(uint64_t), hipMemcpyDeviceToHost));

    uint64_t cpu_sum = 0;
    for(size_t i = 0; i < n; i++)
        cpu_sum += managed[i];

    printf("  GPU sum = %lu, CPU sum = %lu, match = %s\n", gpu_sum, cpu_sum,
           gpu_sum == cpu_sum ? "YES" : "NO");

    HIP_CHECK(hipMemAdvise(managed, bytes, hipMemAdviseUnsetReadMostly, device));
    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 12: Multi-Kernel Stencil Pipeline
//
// Chains multiple kernels that read/write different managed buffers.
// This creates a sustained stream of page migration events as data flows
// through the pipeline.
//
// KFD events expected:
//   Multiple PAGE_FAULT and PAGE_MIGRATE events
//   UNMAP_FROM_GPU events as buffers rotate
// ---------------------------------------------------------------------------

static void
test_stencil_pipeline(size_t bytes, int device, int iterations)
{
    banner("Test 12: Multi-Kernel Stencil Pipeline");

    size_t    n     = bytes / sizeof(uint64_t);
    uint64_t* buf_a = nullptr;
    uint64_t* buf_b = nullptr;
    uint64_t* buf_c = nullptr;
    HIP_CHECK(hipMallocManaged(&buf_a, bytes));
    HIP_CHECK(hipMallocManaged(&buf_b, bytes));
    HIP_CHECK(hipMallocManaged(&buf_c, bytes));

    for(size_t i = 0; i < n; i++)
    {
        buf_a[i] = i;
        buf_b[i] = 0;
        buf_c[i] = 0;
    }

    HIP_CHECK(hipMemPrefetchAsync(buf_a, bytes, device, nullptr));
    HIP_CHECK(hipMemPrefetchAsync(buf_b, bytes, device, nullptr));
    HIP_CHECK(hipMemPrefetchAsync(buf_c, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    for(int iter = 0; iter < iterations; iter++)
    {
        kern_read_write_stencil<<<grid_size(n), BLOCK_SIZE>>>(buf_b, buf_a, n);
        kern_read_write_stencil<<<grid_size(n), BLOCK_SIZE>>>(buf_c, buf_b, n);
        kern_read_write_stencil<<<grid_size(n), BLOCK_SIZE>>>(buf_a, buf_c, n);
        HIP_CHECK(hipDeviceSynchronize());
        printf("  [iter %d] kernels completed\n", iter);
    }

    HIP_CHECK(hipMemPrefetchAsync(buf_a, bytes, hipCpuDeviceId, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    uint64_t spot_check = buf_a[0];
    printf("  buf_a[0] = %lu\n", spot_check);

    HIP_CHECK(hipFree(buf_a));
    HIP_CHECK(hipFree(buf_b));
    HIP_CHECK(hipFree(buf_c));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 13: SAXPY with Managed Memory
//
// Classic y = a*x + y using managed memory for both x and y.
// Initializes on CPU, computes on GPU, verifies on CPU.
//
// KFD events expected:
//   PAGE_MIGRATE_PAGEFAULT_GPU (CPU→GPU when kernel starts)
//   PAGE_MIGRATE_PAGEFAULT_CPU (GPU→CPU for verification)
// ---------------------------------------------------------------------------

static void
test_saxpy_managed(size_t bytes, int device, int iterations)
{
    banner("Test 13: SAXPY with Managed Memory");

    size_t n = bytes / sizeof(float);
    float* x = nullptr;
    float* y = nullptr;
    HIP_CHECK(hipMallocManaged(&x, bytes));
    HIP_CHECK(hipMallocManaged(&y, bytes));

    for(size_t i = 0; i < n; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    printf("  CPU initialized x and y (%zu MB each)\n", bytes >> 20);

    HIP_CHECK(hipMemPrefetchAsync(x, bytes, device, nullptr));
    HIP_CHECK(hipMemPrefetchAsync(y, bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    float a = 3.0f;
    for(int iter = 0; iter < iterations; iter++)
    {
        kern_saxpy<<<grid_size(n), BLOCK_SIZE>>>(y, x, a, n);
        HIP_CHECK(hipDeviceSynchronize());
    }
    printf("  GPU ran %d SAXPY iterations\n", iterations);

    float expected = 2.0f + iterations * 3.0f;
    float actual   = y[0];
    printf("  Verification: y[0] = %.1f (expected %.1f) → %s\n", actual, expected,
           (fabsf(actual - expected) < 0.01f) ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(x));
    HIP_CHECK(hipFree(y));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 14: Memory Pressure – many concurrent managed allocations
//
// Allocates a large number of managed buffers, touches them from both CPU
// and GPU to create memory pressure that can trigger:
//   QUEUE_EVICT_SVM        (SVM range eviction)
//   PAGE_MIGRATE_TTM_EVICTION (TTM-level eviction under pressure)
//   UNMAP_FROM_GPU_MMU_NOTIFY (bulk unmapping)
// ---------------------------------------------------------------------------

static void
test_memory_pressure(size_t total_bytes, int device)
{
    banner("Test 14: Memory Pressure (oversubscription)");

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    size_t gpu_mem = props.totalGlobalMem;
    printf("  GPU %d total memory: %zu MB\n", device, gpu_mem >> 20);

    // Auto-scale: use at least 5% of GPU VRAM (capped at 1 GB) to create
    // meaningful pressure that can trigger queue eviction and unmap events
    // without dominating CI wall-time.
    constexpr size_t MAX_AUTO_PRESSURE = 1ULL * 1024 * 1024 * 1024;
    size_t           min_pressure      = std::min(gpu_mem / 20, MAX_AUTO_PRESSURE);
    if(total_bytes < min_pressure)
    {
        printf("  Requested pressure (%zu MB) is < 5%% of GPU VRAM (%zu MB)\n",
               total_bytes >> 20, min_pressure >> 20);
        printf("  Auto-scaling pressure to %zu MB\n", min_pressure >> 20);
        total_bytes = min_pressure;
    }

    printf("  Allocating %zu MB of managed memory to create pressure\n",
           total_bytes >> 20);

    constexpr size_t CHUNK_SIZE = 16ULL * 1024 * 1024;
    size_t           n_chunks   = total_bytes / CHUNK_SIZE;
    if(n_chunks < 2) n_chunks = 2;

    std::vector<uint64_t*> chunks(n_chunks, nullptr);
    size_t                 elems_per_chunk = CHUNK_SIZE / sizeof(uint64_t);

    for(size_t c = 0; c < n_chunks; c++)
    {
        HIP_CHECK(hipMallocManaged(&chunks[c], CHUNK_SIZE));
    }
    printf("  Allocated %zu chunks of %zu MB each\n", n_chunks, CHUNK_SIZE >> 20);

    printf("  CPU touching all chunks...\n");
    for(size_t c = 0; c < n_chunks; c++)
    {
        for(size_t i = 0; i < elems_per_chunk; i += 512)
            chunks[c][i] = c * 1000 + i;
    }

    printf("  Prefetching all chunks to GPU...\n");
    for(size_t c = 0; c < n_chunks; c++)
        HIP_CHECK(hipMemPrefetchAsync(chunks[c], CHUNK_SIZE, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    printf("  GPU touching all chunks (creating migration pressure)...\n");
    for(size_t c = 0; c < n_chunks; c++)
    {
        constexpr size_t PAGE_ELEMS = 4096 / sizeof(uint64_t);
        kern_touch_pages<<<grid_size(elems_per_chunk / PAGE_ELEMS), BLOCK_SIZE>>>(
            chunks[c], elems_per_chunk, PAGE_ELEMS);
    }
    HIP_CHECK(hipDeviceSynchronize());

    printf("  Prefetching all chunks back to CPU...\n");
    for(size_t c = 0; c < n_chunks; c++)
        HIP_CHECK(hipMemPrefetchAsync(chunks[c], CHUNK_SIZE, hipCpuDeviceId, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    printf("  CPU reading back all chunks...\n");
    uint64_t total_sum = 0;
    for(size_t c = 0; c < n_chunks; c++)
    {
        for(size_t i = 0; i < elems_per_chunk; i += 512)
            total_sum += chunks[c][i];
    }
    printf("  Aggregate checksum: %lu\n", total_sum);

    printf("  Freeing all managed memory (triggers unmap events)...\n");
    for(size_t c = 0; c < n_chunks; c++)
        HIP_CHECK(hipFree(chunks[c]));

    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 15: Rapid Alloc/Free Cycling
//
// Rapidly allocates and frees managed memory while accessing from GPU.
// Stresses the VM subsystem and triggers unmap-from-GPU events.
//
// KFD events expected:
//   UNMAP_FROM_GPU_UNMAP_FROM_CPU (freeing mapped ranges)
//   UNMAP_FROM_GPU_MMU_NOTIFY
// ---------------------------------------------------------------------------

static void
test_alloc_free_cycle(size_t bytes, int device, int iterations)
{
    banner("Test 15: Rapid Alloc/Free Cycling");

    size_t n = bytes / sizeof(uint64_t);

    for(int iter = 0; iter < iterations; iter++)
    {
        uint64_t* managed = nullptr;
        HIP_CHECK(hipMallocManaged(&managed, bytes));

        // Prefetch to GPU so pages become GPU-resident, then free.
        // This ensures hipFree triggers UNMAP_FROM_GPU events.
        HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        kern_write_pattern<<<grid_size(n), BLOCK_SIZE>>>(managed, n, iter * 100ULL);
        HIP_CHECK(hipDeviceSynchronize());

        uint64_t spot = managed[0];
        HIP_CHECK(hipFree(managed));

        if(iter % 4 == 0)
            printf("  [cycle %d/%d] spot check = %lu\n", iter + 1, iterations, spot);
    }
    printf("  DONE (%d alloc/free cycles)\n", iterations);
}

// ---------------------------------------------------------------------------
// Test 16: Prefetch Ping-Pong
//
// Uses hipMemPrefetchAsync to explicitly bounce data between CPU and GPU.
// Each prefetch triggers PAGE_MIGRATE_PREFETCH events.
// ---------------------------------------------------------------------------

static void
test_prefetch_pingpong(size_t bytes, int device, int iterations)
{
    banner("Test 16: Prefetch Ping-Pong");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    for(size_t i = 0; i < n; i++)
        managed[i] = i;

    for(int iter = 0; iter < iterations; iter++)
    {
        HIP_CHECK(hipMemPrefetchAsync(managed, bytes, device, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        kern_write_pattern<<<grid_size(n), BLOCK_SIZE>>>(managed, n, iter * 7ULL);
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemPrefetchAsync(managed, bytes, hipCpuDeviceId, nullptr));
        HIP_CHECK(hipDeviceSynchronize());

        managed[0] += iter;

        if(iter % 2 == 0)
            printf("  [iter %d] prefetch ping-pong, managed[0] = %lu\n", iter,
                   managed[0]);
    }

    HIP_CHECK(hipFree(managed));
    printf("  DONE (%d prefetch round-trips)\n", iterations);
}

// ---------------------------------------------------------------------------
// Test 17: Multi-Stream Concurrent Access
//
// Multiple HIP streams access different managed buffers simultaneously.
// This maximizes concurrent page fault and migration traffic.
// ---------------------------------------------------------------------------

static void
test_multi_stream(size_t bytes, int device)
{
    banner("Test 17: Multi-Stream Concurrent Access");

    constexpr int NUM_STREAMS = 4;
    hipStream_t   streams[NUM_STREAMS];
    uint64_t*     buffers[NUM_STREAMS];
    size_t        n = bytes / sizeof(uint64_t);

    for(int s = 0; s < NUM_STREAMS; s++)
    {
        HIP_CHECK(hipStreamCreate(&streams[s]));
        HIP_CHECK(hipMallocManaged(&buffers[s], bytes));
        for(size_t i = 0; i < n; i++)
            buffers[s][i] = s * 1000 + i;
    }
    printf("  Created %d streams with %zu MB managed buffers each\n", NUM_STREAMS,
           bytes >> 20);

    for(int s = 0; s < NUM_STREAMS; s++)
        HIP_CHECK(hipMemPrefetchAsync(buffers[s], bytes, device, streams[s]));
    for(int s = 0; s < NUM_STREAMS; s++)
        HIP_CHECK(hipStreamSynchronize(streams[s]));

    printf("  Launching concurrent kernels on all streams...\n");
    for(int s = 0; s < NUM_STREAMS; s++)
    {
        kern_write_pattern<<<grid_size(n), BLOCK_SIZE, 0, streams[s]>>>(buffers[s], n,
                                                                        s * 0xF0F0ULL);
    }

    for(int s = 0; s < NUM_STREAMS; s++)
        HIP_CHECK(hipStreamSynchronize(streams[s]));

    printf("  CPU reading back all stream buffers...\n");
    for(int s = 0; s < NUM_STREAMS; s++)
    {
        uint64_t val = buffers[s][0];
        printf("    stream %d: buffers[%d][0] = %lu\n", s, s, val);
    }

    for(int s = 0; s < NUM_STREAMS; s++)
    {
        HIP_CHECK(hipFree(buffers[s]));
        HIP_CHECK(hipStreamDestroy(streams[s]));
    }
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 18: Partial Range Access
//
// Allocates a large managed buffer but only touches specific sub-ranges
// from CPU and GPU, causing page-granularity faults and migrations.
// ---------------------------------------------------------------------------

static void
test_partial_range(size_t bytes, int device)
{
    banner("Test 18: Partial Range Access (page-level faults)");

    size_t    n       = bytes / sizeof(uint64_t);
    uint64_t* managed = nullptr;
    HIP_CHECK(hipMallocManaged(&managed, bytes));

    size_t quarter       = n / 4;
    size_t quarter_bytes = quarter * sizeof(uint64_t);

    printf("  CPU writing first quarter (%zu MB)...\n", quarter_bytes >> 20);
    for(size_t i = 0; i < quarter; i++)
        managed[i] = i;

    printf("  GPU writing second quarter...\n");
    HIP_CHECK(hipMemPrefetchAsync(managed + quarter, quarter_bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    kern_write_pattern<<<grid_size(quarter), BLOCK_SIZE>>>(managed + quarter, quarter,
                                                           0xAA);
    HIP_CHECK(hipDeviceSynchronize());

    printf("  CPU writing third quarter...\n");
    for(size_t i = 2 * quarter; i < 3 * quarter; i++)
        managed[i] = i * 3;

    printf("  GPU writing fourth quarter...\n");
    HIP_CHECK(hipMemPrefetchAsync(managed + 3 * quarter, quarter_bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    kern_write_pattern<<<grid_size(quarter), BLOCK_SIZE>>>(managed + 3 * quarter, quarter,
                                                           0xBB);
    HIP_CHECK(hipDeviceSynchronize());

    printf("  Cross-reading: CPU reads GPU quarters, GPU reads CPU quarters...\n");
    HIP_CHECK(
        hipMemPrefetchAsync(managed + quarter, quarter_bytes, hipCpuDeviceId, nullptr));
    HIP_CHECK(hipMemPrefetchAsync(managed + 3 * quarter, quarter_bytes, hipCpuDeviceId,
                                  nullptr));
    HIP_CHECK(hipDeviceSynchronize());
    uint64_t cpu_sum = 0;
    for(size_t i = quarter; i < 2 * quarter; i += 64)
        cpu_sum += managed[i];
    for(size_t i = 3 * quarter; i < n; i += 64)
        cpu_sum += managed[i];
    printf("  CPU sum of GPU quarters: %lu\n", cpu_sum);

    HIP_CHECK(hipMemPrefetchAsync(managed, quarter_bytes, device, nullptr));
    HIP_CHECK(hipMemPrefetchAsync(managed + 2 * quarter, quarter_bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t* d_result = nullptr;
    HIP_CHECK(hipMalloc(&d_result, sizeof(uint64_t)));
    HIP_CHECK(hipMemset(d_result, 0, sizeof(uint64_t)));
    kern_read_reduce<<<grid_size(quarter), BLOCK_SIZE>>>(managed, quarter, d_result);
    kern_read_reduce<<<grid_size(quarter), BLOCK_SIZE>>>(managed + 2 * quarter, quarter,
                                                         d_result);
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t gpu_sum = 0;
    HIP_CHECK(hipMemcpy(&gpu_sum, d_result, sizeof(uint64_t), hipMemcpyDeviceToHost));
    printf("  GPU sum of CPU quarters: %lu\n", gpu_sum);

    HIP_CHECK(hipFree(d_result));
    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// Test 19: Large Single Allocation
//
// A single very large managed allocation. Exercises the system's ability
// to handle large-scale page table operations.
// ---------------------------------------------------------------------------

static void
test_large_allocation(size_t bytes, int device)
{
    banner("Test 19: Large Single Allocation");

    size_t large_bytes = bytes * 4;
    size_t n           = large_bytes / sizeof(uint64_t);

    uint64_t*  managed = nullptr;
    hipError_t err     = hipMallocManaged(&managed, large_bytes);
    if(err != hipSuccess)
    {
        printf("  Skipping: could not allocate %zu MB (%s)\n", large_bytes >> 20,
               hipGetErrorString(err));
        return;
    }
    printf("  Allocated %zu MB managed buffer\n", large_bytes >> 20);

    printf("  Prefetching to GPU...\n");
    HIP_CHECK(hipMemPrefetchAsync(managed, large_bytes, device, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    printf("  GPU writing...\n");
    kern_write_pattern<<<grid_size(n), BLOCK_SIZE>>>(managed, n, 0x1234);
    HIP_CHECK(hipDeviceSynchronize());

    printf("  Prefetching back to CPU...\n");
    HIP_CHECK(hipMemPrefetchAsync(managed, large_bytes, hipCpuDeviceId, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    uint64_t spot = managed[n / 2];
    printf("  CPU spot check: managed[%zu] = %lu\n", n / 2, spot);

    HIP_CHECK(hipFree(managed));
    printf("  DONE\n");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int
main(int argc, char** argv)
{
    Config cfg = parse_args(argc, argv);

    HIP_CHECK(hipSetDevice(cfg.device_id));

    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, cfg.device_id));
    printf("Unified Memory KFD Events Test\n");
    printf("  GPU %d: %s\n", cfg.device_id, props.name);
    printf("  GPU memory: %zu MB\n", props.totalGlobalMem >> 20);
    printf("  Managed memory support: %s\n", props.managedMemory ? "YES" : "NO");
    printf("  Concurrent managed access: %s\n",
           props.concurrentManagedAccess ? "YES" : "NO");
    printf("  HSA_XNACK: %s\n", getenv("HSA_XNACK") ? getenv("HSA_XNACK") : "(not set)");
    printf("  Per-allocation size: %zu MB\n", cfg.alloc_size_mb);
    printf("  Pressure test size: %zu MB\n", cfg.pressure_mb);
    printf("  Iterations: %d\n", cfg.iterations);
    printf("  Test suite: %s\n", cfg.run_all ? "all (1-19)" : "basic (1-9)");

    if(!props.managedMemory)
    {
        const char* xnack = getenv("HSA_XNACK");
        if(!xnack || strcmp(xnack, "1") != 0)
        {
            fprintf(stderr,
                    "\nERROR: GPU %d does not report managed memory support and "
                    "HSA_XNACK is not set to 1. Aborting.\n"
                    "  Set HSA_XNACK=1 to enable managed memory on XNACK-capable GPUs.\n",
                    cfg.device_id);
            return 1;
        }
        printf("  Note: GPU reports no managed memory support but HSA_XNACK=1 is set; "
               "proceeding.\n");
    }

    size_t alloc_bytes    = cfg.alloc_size_mb * 1024ULL * 1024ULL;
    size_t pressure_bytes = cfg.pressure_mb * 1024ULL * 1024ULL;

    auto t0 = std::chrono::steady_clock::now();

    int n_tests = NUM_DEFAULT_TESTS;

    test_gpu_read_fault(alloc_bytes, cfg.device_id);
    test_gpu_write_fault(alloc_bytes, cfg.device_id);
    test_cpu_read_fault(alloc_bytes, cfg.device_id);
    test_cpu_write_fault(alloc_bytes, cfg.device_id);
    test_prefetch(alloc_bytes, cfg.device_id);
    test_pingpong(alloc_bytes, cfg.device_id, cfg.iterations);
    test_force_gpu_always_mapped(alloc_bytes, cfg.device_id, cfg.iterations);
    test_madvise_unmap_from_gpu(alloc_bytes, cfg.device_id);
    test_xnack_read_fault_migrated(alloc_bytes, cfg.device_id);

    if(cfg.run_all)
    {
        n_tests = NUM_ALL_TESTS;
        test_mem_advise(alloc_bytes, cfg.device_id);
        test_read_mostly(alloc_bytes, cfg.device_id);
        test_stencil_pipeline(alloc_bytes, cfg.device_id, cfg.iterations);
        test_saxpy_managed(alloc_bytes, cfg.device_id, cfg.iterations);
        test_memory_pressure(pressure_bytes, cfg.device_id);
        test_alloc_free_cycle(alloc_bytes, cfg.device_id, cfg.iterations * 2);
        test_prefetch_pingpong(alloc_bytes, cfg.device_id, cfg.iterations);
        test_multi_stream(alloc_bytes, cfg.device_id);
        test_partial_range(alloc_bytes, cfg.device_id);
        test_large_allocation(alloc_bytes, cfg.device_id);
    }

    auto   t1 = std::chrono::steady_clock::now();
    double elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;

    banner("SUMMARY");
    printf("  %d tests completed in %.2f seconds\n", n_tests, elapsed);
    printf("  KFD event categories exercised:\n");
    printf("    - Page Faults (read/write, migrated/updated)\n");
    printf("    - Page Migrations (prefetch, GPU pagefault, CPU pagefault)\n");
    printf("    - Queue Evictions (SVM under memory pressure)\n");
    printf("    - Unmap from GPU (MMU notify, migrate, unmap from CPU)\n");

    return 0;
}
