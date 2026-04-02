/*
 * late_attach_test.cpp - Integration test for HSA queue intercept late attach.
 *
 * Simulates the customer scenario: a running HIP application launches kernels,
 * then a profiler library (libtrace_injector.so) is dlopen'd at runtime to
 * attach and start tracing, then later detached.
 *
 * Flow:
 *   1. Launch HIP kernels for WARMUP_SECONDS
 *   2. dlopen libtrace_injector.so and call trace_injector_start()
 *   3. Continue launching HIP kernels for TRACE_SECONDS (intercept active)
 *   4. Call trace_injector_stop() and dlclose
 *   5. Continue launching HIP kernels for COOLDOWN_SECONDS (verify stability)
 *   6. Verify functional correctness and that traces were captured
 *
 * The profiler library is loaded ONLY via dlopen - it is NOT linked at build time.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <dlfcn.h>

#include <hip/hip_runtime.h>

// HSA headers for queue creation and intercept types
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

// Timing configuration (seconds)
static constexpr int WARMUP_SECONDS   = 2;
static constexpr int TRACE_SECONDS    = 5;
static constexpr int COOLDOWN_SECONDS = 2;

// Kernel configuration
static constexpr int N = 1024;
static constexpr float SCALE = 2.0f;

#define LOG(fmt, ...) do { fprintf(stderr, "[test] " fmt "\n", ##__VA_ARGS__); } while(0)

#define HIP_CHECK(cmd) do { \
    hipError_t err = (cmd); \
    if (err != hipSuccess) { \
        LOG("FAIL: %s at %s:%d - %s", #cmd, __FILE__, __LINE__, hipGetErrorString(err)); \
        return 1; \
    } \
} while(0)

// Simple SAXPY kernel: y[i] = a * x[i] + y[i]
__global__ void saxpy_kernel(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

// HSA agent discovery
static hsa_agent_t g_gpu_agent = {0};

static hsa_status_t find_gpu_agent(hsa_agent_t agent, void* data) {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
        *(hsa_agent_t*)data = agent;
        return HSA_STATUS_INFO_BREAK;
    }
    return HSA_STATUS_SUCCESS;
}

// Run HIP kernels for a given duration
// Returns: number of kernel launches, or -1 on error
static int run_kernels(float* d_x, float* d_y,
                       int duration_seconds, const char* phase_name) {
    int launches = 0;
    auto start = std::chrono::steady_clock::now();
    auto end_time = start + std::chrono::seconds(duration_seconds);

    while (std::chrono::steady_clock::now() < end_time) {
        int blockSize = 256;
        int gridSize = (N + blockSize - 1) / blockSize;
        saxpy_kernel<<<gridSize, blockSize>>>(SCALE, d_x, d_y, N);

        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            LOG("FAIL: Kernel launch error in %s: %s", phase_name, hipGetErrorString(err));
            return -1;
        }
        launches++;

        // Sync periodically
        if (launches % 100 == 0) {
            err = hipDeviceSynchronize();
            if (err != hipSuccess) {
                LOG("FAIL: Sync error in %s after %d launches: %s",
                    phase_name, launches, hipGetErrorString(err));
                return -1;
            }
        }
    }

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        LOG("FAIL: Final sync in %s: %s", phase_name, hipGetErrorString(err));
        return -1;
    }

    LOG("%s: %d kernel launches completed", phase_name, launches);
    return launches;
}

// Submit barrier packets to an HSA queue
static int submit_barriers(hsa_queue_t* queue, int duration_seconds,
                           const char* phase_name) {
    int submissions = 0;
    auto start = std::chrono::steady_clock::now();
    auto end_time = start + std::chrono::seconds(duration_seconds);

    while (std::chrono::steady_clock::now() < end_time) {
        uint64_t write_idx = hsa_queue_add_write_index_relaxed(queue, 1);
        uint64_t mask = queue->size - 1;

        hsa_barrier_and_packet_t* barrier =
            (hsa_barrier_and_packet_t*)((char*)queue->base_address +
                                        (write_idx & mask) * 64);
        memset(barrier, 0, sizeof(*barrier));
        barrier->header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        barrier->header |= (1 << HSA_PACKET_HEADER_BARRIER);

        hsa_signal_store_screlease(queue->doorbell_signal, write_idx);
        submissions++;

        // Pace: ~100 submissions/sec
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    LOG("%s: %d barrier submissions", phase_name, submissions);
    return submissions;
}

int main() {
    int failures = 0;

    LOG("=== Late-Attach Integration Test ===");

    // --- Phase 0: Setup ---
    LOG("Phase 0: Setup");

    HIP_CHECK(hipSetDevice(0));

    // Allocate buffers
    float* h_x = (float*)malloc(N * sizeof(float));
    float* h_y = (float*)malloc(N * sizeof(float));
    float* d_x = nullptr;
    float* d_y = nullptr;

    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 0.0f;
    }

    HIP_CHECK(hipMalloc(&d_x, N * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_x, h_x, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice));

    // Find GPU agent
    hsa_status_t hsa_status = hsa_iterate_agents(find_gpu_agent, &g_gpu_agent);
    if (hsa_status != HSA_STATUS_INFO_BREAK || g_gpu_agent.handle == 0) {
        LOG("FAIL: Could not find GPU agent");
        return 1;
    }

    // Create a dedicated HSA queue for intercept testing
    hsa_queue_t* hsa_queue = nullptr;
    hsa_status = hsa_queue_create(g_gpu_agent, 256, HSA_QUEUE_TYPE_SINGLE,
                                  nullptr, nullptr, UINT32_MAX, UINT32_MAX,
                                  &hsa_queue);
    if (hsa_status != HSA_STATUS_SUCCESS) {
        LOG("FAIL: Could not create HSA queue");
        return 1;
    }
    LOG("Created HSA queue: %p (base=%p, doorbell=%lx, size=%u)",
        (void*)hsa_queue, hsa_queue->base_address,
        hsa_queue->doorbell_signal.handle, hsa_queue->size);

    // --- Phase 1: Warmup ---
    LOG("Phase 1: Warmup (%d seconds)", WARMUP_SECONDS);
    std::atomic<int> barrier_count_warmup{0};
    std::thread barrier_thread_warmup([&]() {
        barrier_count_warmup = submit_barriers(hsa_queue, WARMUP_SECONDS, "warmup-barriers");
    });
    int warmup_launches = run_kernels(d_x, d_y, WARMUP_SECONDS, "warmup-kernels");
    barrier_thread_warmup.join();
    if (warmup_launches < 0) failures++;

    // --- Phase 2: Attach profiler ---
    LOG("Phase 2: Attaching profiler via dlopen");

    void* injector_lib = dlopen("./libtrace_injector.so", RTLD_NOW);
    if (!injector_lib) {
        LOG("FAIL: dlopen failed: %s", dlerror());
        failures++;
        goto cleanup;
    }
    LOG("dlopen succeeded: %p", injector_lib);

    {
        typedef int (*start_fn_t)(hsa_queue_t**, int);
        typedef int (*stop_fn_t)(int*);

        start_fn_t injector_start = (start_fn_t)dlsym(injector_lib, "trace_injector_start");
        stop_fn_t injector_stop = (stop_fn_t)dlsym(injector_lib, "trace_injector_stop");

        if (!injector_start || !injector_stop) {
            LOG("FAIL: dlsym failed: %s", dlerror());
            failures++;
            dlclose(injector_lib);
            goto cleanup;
        }
        LOG("dlsym: start=%p stop=%p", (void*)injector_start, (void*)injector_stop);

        // Record queue state before attach
        LOG("Queue before attach: base=%p doorbell=%lx",
            hsa_queue->base_address, hsa_queue->doorbell_signal.handle);

        // Attach
        hsa_queue_t* queues[] = { hsa_queue };
        int start_result = injector_start(queues, 1);
        if (start_result != 0) {
            LOG("FAIL: trace_injector_start returned %d", start_result);
            failures++;
            dlclose(injector_lib);
            goto cleanup;
        }

        // Check if queue was modified by attach
        LOG("Queue after attach: base=%p doorbell=%lx",
            hsa_queue->base_address, hsa_queue->doorbell_signal.handle);

        // --- Phase 3: Tracing active ---
        LOG("Phase 3: Tracing active (%d seconds)", TRACE_SECONDS);
        HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice));

        std::atomic<int> barrier_count_trace{0};
        std::thread barrier_thread_trace([&]() {
            barrier_count_trace = submit_barriers(hsa_queue, TRACE_SECONDS, "trace-barriers");
        });
        int trace_launches = run_kernels(d_x, d_y, TRACE_SECONDS, "trace-kernels");
        barrier_thread_trace.join();
        if (trace_launches < 0) failures++;

        // --- Phase 4: Stop tracing ---
        LOG("Phase 4: Detaching profiler");

        int trace_count = 0;
        int stop_result = injector_stop(&trace_count);
        if (stop_result != 0) {
            LOG("FAIL: trace_injector_stop returned %d", stop_result);
            failures++;
        }

        LOG("Profiler reported %d intercepted packets", trace_count);

        // Check if queue was restored after detach
        LOG("Queue after detach: base=%p doorbell=%lx",
            hsa_queue->base_address, hsa_queue->doorbell_signal.handle);

        // Verify traces were captured
        if (trace_count > 0) {
            LOG("PASS: Profiler captured %d packet traces", trace_count);
        } else {
            LOG("FAIL: No traces captured during active phase (expected > 0)");
            failures++;
        }

        // Verify barrier count roughly matches trace count
        int expected_min = barrier_count_trace.load() / 2;
        if (trace_count >= expected_min) {
            LOG("PASS: Trace count (%d) >= expected minimum (%d)", trace_count, expected_min);
        } else {
            LOG("FAIL: Trace count (%d) < expected minimum (%d)", trace_count, expected_min);
            failures++;
        }

        dlclose(injector_lib);
        LOG("Profiler library unloaded");
    }

    // --- Phase 5: Cooldown ---
    LOG("Phase 5: Cooldown (%d seconds) - stability after detach", COOLDOWN_SECONDS);
    HIP_CHECK(hipMemcpy(d_y, h_y, N * sizeof(float), hipMemcpyHostToDevice));

    {
        std::atomic<int> barrier_count_cool{0};
        std::thread barrier_thread_cool([&]() {
            barrier_count_cool = submit_barriers(hsa_queue, COOLDOWN_SECONDS, "cooldown-barriers");
        });
        int cooldown_launches = run_kernels(d_x, d_y, COOLDOWN_SECONDS, "cooldown-kernels");
        barrier_thread_cool.join();
        if (cooldown_launches < 0) {
            failures++;
        } else {
            LOG("PASS: %d kernel launches after detach (no crash)", cooldown_launches);
        }
    }

    // --- Phase 6: Verify correctness ---
    LOG("Phase 6: Verifying functional correctness");
    {
        float* h_result = (float*)malloc(N * sizeof(float));
        HIP_CHECK(hipMemcpy(h_result, d_y, N * sizeof(float), hipMemcpyDeviceToHost));

        bool correct = true;
        float expected = h_result[0];
        for (int i = 1; i < N; i++) {
            if (h_result[i] != expected) {
                LOG("FAIL: h_result[%d]=%f != h_result[0]=%f", i, h_result[i], expected);
                correct = false;
                break;
            }
        }
        if (correct && expected > 0.0f) {
            LOG("PASS: Functional correctness verified (all elements = %f)", expected);
        } else if (expected == 0.0f) {
            LOG("FAIL: All results are zero");
            failures++;
        } else {
            failures++;
        }
        free(h_result);
    }

cleanup:
    hsa_queue_destroy(hsa_queue);
    (void)hipFree(d_x);
    (void)hipFree(d_y);
    free(h_x);
    free(h_y);

    LOG("=== Results: %d failure(s) ===", failures);
    return failures > 0 ? 1 : 0;
}
