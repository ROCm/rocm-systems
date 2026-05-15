// Copied from https://github.com/benrichard-amd/rocflop/tree/82f197e12314bab694fc70451a2b495b4f51bf90

#include <iostream>
#include <cstring>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <unistd.h>
#include <type_traits>
#include <vector>
#include <sys/wait.h>
#include <fcntl.h>

using float16 = _Float16;

// Vector types. Useful for packed math (where supported) and MFMA inputs.
template<typename T, uint32_t Rank>
using vecT = T __attribute__((ext_vector_type(Rank)));

template<typename T> using vec4 = vecT<T, 4>;
template<typename T> using vec8 = vecT<T, 8>;


// Kernels


template<typename T> __global__ void fma_throughput(vec4<T>* buffer, int count)
{
    const T k = 1.0;

    const int grid_size = gridDim.x * blockDim.x;
    const int tid = blockDim.x * blockIdx.x + threadIdx.x;

    vec4<T>* ptr = buffer;

    vec4<T> value0 = ptr[0 * grid_size + tid];
    vec4<T> value1 = ptr[1 * grid_size + tid];
    vec4<T> value2 = ptr[2 * grid_size + tid];
    vec4<T> value3 = ptr[3 * grid_size + tid];

    for(int j = 0; j < count; j++) {
        for(int j = 0; j < 64; j++) {

            // 16 FMA ops
            value0 = value0 * value0 + k;
            value1 = value1 * value1 + k;
            value2 = value2 * value2 + k;
            value3 = value3 * value3 + k;
        }
    }

    ptr[tid] = value0 + value1 + value2 + value3;
}

// MFMA instructions are only available on gfx908 and later (not supported on gfx906)
#if !defined(__gfx906__)
__global__ void matmul_fp16_throughput(vec4<float16>* inputs, vec4<float>* outputs, int count)
{
    int grid_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    vec4<float16>* ptr = inputs;

    vec4<float16> value0 = ptr[0 * grid_size + tid];
    vec4<float16> value1 = ptr[1 * grid_size + tid];
    vec4<float16> value2 = ptr[2 * grid_size + tid];
    vec4<float16> value3 = ptr[3 * grid_size + tid];

    vec4<float> accum0;
    vec4<float> accum1;
    vec4<float> accum2;
    vec4<float> accum3;
    for(int i = 0; i < count; i++) {
        for(int j = 0; j < 64; j++) {
            // 4 MFMA ops
            accum0 = __builtin_amdgcn_mfma_f32_16x16x16f16(value0, value0, accum0, 0, 0, 0);
            accum1 = __builtin_amdgcn_mfma_f32_16x16x16f16(value1, value1, accum1, 0, 0, 0);
            accum2 = __builtin_amdgcn_mfma_f32_16x16x16f16(value2, value2, accum2, 0, 0, 0);
            accum3 = __builtin_amdgcn_mfma_f32_16x16x16f16(value3, value3, accum3, 0, 0, 0);
        }
    }

    outputs[tid] = accum0 + accum1 + accum2 + accum3;
}

__global__ void matmul_fp32_throughput(float* inputs, vec4<float>* outputs, int count)
{
    int grid_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float* ptr = inputs;

    float value0 = ptr[0 * grid_size + tid];
    float value1 = ptr[1 * grid_size + tid];
    float value2 = ptr[2 * grid_size + tid];
    float value3 = ptr[2 * grid_size + tid];

    vec4<float> accum0;
    vec4<float> accum1;
    vec4<float> accum2;
    vec4<float> accum3;
    for(int i = 0; i < count; i++) {
        for(int j = 0; j < 64; j++) {
            // 4 MFMA ops
            accum0 = __builtin_amdgcn_mfma_f32_16x16x4f32(value0, value0, accum0, 0, 0, 0);
            accum1 = __builtin_amdgcn_mfma_f32_16x16x4f32(value1, value1, accum1, 0, 0, 0);
            accum2 = __builtin_amdgcn_mfma_f32_16x16x4f32(value2, value2, accum2, 0, 0, 0);
            accum3 = __builtin_amdgcn_mfma_f32_16x16x4f32(value3, value3, accum3, 0, 0, 0);
        }
    }

    outputs[tid] = accum0 + accum1 + accum2 + accum3;
}
#endif // !defined(__gfx906__)

// SMFMAC (Sparse MFMA) instructions are only available on gfx940 and later (not on gfx906, gfx908, or gfx90a)
#if !defined(__gfx906__) && !defined(__gfx908__) && !defined(__gfx90a__)
__global__ void sparse_matmul_fp16_throughput(vec4<float16>* input0, vec8<float16>* input1, vec4<float>* outputs, int count)
{
    int grid_size = gridDim.x * blockDim.x;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    vec4<float16>* x_ptr = input0;
    vec8<float16>* y_ptr = input1;

    vec4<float16> x0 = x_ptr[0 * grid_size + tid];
    vec4<float16> x1 = x_ptr[1 * grid_size + tid];
    vec4<float16> x2 = x_ptr[2 * grid_size + tid];
    vec4<float16> x3 = x_ptr[3 * grid_size + tid];
    
    vec8<float16> y0 = y_ptr[0 * grid_size + tid];
    vec8<float16> y1 = y_ptr[1 * grid_size + tid];
    vec8<float16> y2 = y_ptr[2 * grid_size + tid];
    vec8<float16> y3 = y_ptr[3 * grid_size + tid];
    
    vec4<float> accum0;
    vec4<float> accum1;
    vec4<float> accum2;
    vec4<float> accum3;
   
    for(int i = 0; i < count; i++) {
        for(int j = 0; j < 64; j++) {
            // 4 SMFMAC ops
            accum0 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(x0, y0, accum0, 0, 0, 0);
            accum1 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(x1, y1, accum1, 0, 0, 0);
            accum2 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(x2, y2, accum2, 0, 0, 0);
            accum3 = __builtin_amdgcn_smfmac_f32_16x16x32_f16(x3, y3, accum3, 0, 0, 0);
        }
    }

    outputs[tid] = accum0 + accum1 + accum2 + accum3;
}
#endif // !defined(__gfx906__) && !defined(__gfx908__) && !defined(__gfx90a__)

void HIP_CALL(hipError_t err)
{
    if(err != hipSuccess) {
        std::cout << "HIP Error: " << (int)err << " " << hipGetErrorString(err) << std::endl;
        exit(1);
    }
}

struct GCNArch {
    int major;
    int minor;
    int rev;
};

GCNArch get_gcn_arch(int device)
{
    hipDeviceProp_t props;

    HIP_CALL(hipGetDeviceProperties(&props, device));

    // Example: gfx908:sramecc+:xnack-
    std::string arch_full(props.gcnArchName);

    // Extract number e.g. "908" 
    std::string gfx_str = arch_full.substr(3, arch_full.find_first_of(':'));

    int gfx_num = std::stoi(gfx_str, nullptr, 16);

    GCNArch arch;
    arch.major = (gfx_num & 0xff00) >> 8;
    arch.minor = (gfx_num & 0x00f0) >> 4;
    arch.rev   = (gfx_num & 0x000f);

    return arch;
}

enum : uint32_t {
    VALU_FP32   = 1 << 0,
    VALU_FP16   = 1 << 1,
    VALU_FP64   = 1 << 2,
    MATRIX_FP16 = 1 << 3,
    MATRIX_FP32 = 1 << 4,
    SMATRIX_FP16 = 1 << 5,
    VALU_INT32  = 1 << 6,

    ALL         = (uint32_t)-1
};

// Timer for measuring kernel duration
class HIPTimer {

private:
    hipEvent_t m_start;
    hipEvent_t m_stop;

public:
    HIPTimer()
    {
        HIP_CALL(hipEventCreate(&m_start));
        HIP_CALL(hipEventCreate(&m_stop));
    }

    void start()
    {
        HIP_CALL(hipEventRecord(m_start));
    }

    void stop()
    {
        HIP_CALL(hipEventRecord(m_stop));
    }

    double elapsed()
    {
        float ms;
        HIP_CALL(hipEventElapsedTime(&ms, m_start, m_stop));

        return (double)ms / 1000.0;
    }
};

// Host code

template<typename T> double fma_throughput_test(int device, int count, int runs = 1)
{
    vec4<T>* buffer = nullptr;

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, device));
    
    int blocks = props.multiProcessorCount * 512;
    int threads_per_block = 64;
    int total_threads = blocks * threads_per_block;

    HIP_CALL(hipMalloc(&buffer, sizeof(vec4<T>) * total_threads * 4));
    
    HIPTimer t;
    t.start();
    for(int i = 0; i < runs; i++) {
        fma_throughput<T><<<blocks, threads_per_block>>>(buffer, count);
    }
    t.stop();
    HIP_CALL(hipDeviceSynchronize());

    double elapsed = t.elapsed();
    double ops = (double)total_threads * count * 64 * 16 * runs;
    double flops = (double)ops * 2.0 / elapsed;

    HIP_CALL(hipFree(buffer));

    return flops;
}

#if !defined(__gfx906__)
template<typename matT, typename accumT> double matmul_throughput_test(int device, int count, int runs = 1)
{
    const int wave_size = 64;
    int k;
    int m;
    int n;

    if(std::is_same<matT, float16>::value) {
        m = 16;
        n = 16;
        k = 16;
    } else if(std::is_same<matT, float>::value) {
        m = 16;
        n = 16;
        k = 4;
    } else {
        assert(false);
    }
    
    int ops_per_matmul = k * m * n * 2;

    void* buffer = nullptr;
    void* accum = nullptr;

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, device));

    int blocks = props.multiProcessorCount * 512;
    int threads_per_block = wave_size;
    int total_threads = blocks * threads_per_block;

    HIP_CALL(hipMalloc(&buffer, 4 * sizeof(matT) * m * k * total_threads));
    HIP_CALL(hipMalloc(&accum, sizeof(accumT) * m * n * total_threads));

    HIPTimer t;
    t.start();
    for(int i = 0; i < runs; i++) {
        if(std::is_same<matT, float16>::value && std::is_same<accumT, float>::value) {
            matmul_fp16_throughput<<<blocks, threads_per_block>>>((vec4<float16>*)buffer, (vec4<float>*)accum, count);
        } else if(std::is_same<matT,float>::value && std::is_same<accumT, float>::value) {
            matmul_fp32_throughput<<<blocks, threads_per_block>>>((float*)buffer, (vec4<float>*)accum, count);
        }
    }
    t.stop();
    HIP_CALL(hipDeviceSynchronize());

    double elapsed = t.elapsed();
    double ops = (double)blocks * count * 64 * 4 * runs;
    double flops = (double)ops * ops_per_matmul / elapsed;

    HIP_CALL(hipFree(buffer));
    HIP_CALL(hipFree(accum));

    return flops;
}
#endif // !defined(__gfx906__)

#if !defined(__gfx906__) && !defined(__gfx908__) && !defined(__gfx90a__)
template<typename matT, typename accumT> double sparse_matmul_throughput_test(int device, int count, int runs = 1)
{
    const int wave_size = 64;
    int k;
    int m;
    int n;

    if(std::is_same<matT, float16>::value) {
        m = 16;
        n = 16;
        k = 32;
    } else {
        assert(false);
    }

    int ops_per_matmul = k * m * n * 2;

    void* buffer1 = nullptr;
    void* buffer2 = nullptr;
    void* accum = nullptr;

    hipDeviceProp_t props;
    HIP_CALL(hipGetDeviceProperties(&props, device));

    int blocks = props.multiProcessorCount * 512;
    int threads_per_block = wave_size;
    int total_threads = blocks * threads_per_block;

    HIP_CALL(hipMalloc(&buffer1, 4 * sizeof(matT) * m * k * total_threads));
    HIP_CALL(hipMalloc(&buffer2, 8 * sizeof(matT) * n * k * total_threads));
    HIP_CALL(hipMalloc(&accum, sizeof(accumT) * m * n * total_threads));

    HIPTimer t;
    t.start();
    for(int i = 0; i < runs; i++) {
        if(std::is_same<matT, float16>::value && std::is_same<accumT, float>::value) {
            sparse_matmul_fp16_throughput<<<blocks, threads_per_block>>>((vec4<float16>*)buffer1,
            (vec8<float16>*)buffer2, (vec4<float>*)accum, count);
        }
    }
    t.stop();
    HIP_CALL(hipDeviceSynchronize());

    double elapsed = t.elapsed();
    double ops = (double)blocks * count * 64 * 4 * runs;
    double flops = (double)ops * ops_per_matmul / elapsed;

    HIP_CALL(hipFree(buffer1));
    HIP_CALL(hipFree(buffer2));
    HIP_CALL(hipFree(accum));

    return flops;
}
#endif // !defined(__gfx906__) && !defined(__gfx908__) && !defined(__gfx90a__)

struct Result {
    int device = -1;
    double valu_fp16 = 0;
    double valu_fp32 = 0;
    double valu_fp64 = 0;
    double valu_int32 = 0;
    double mfma_fp16 = 0;
    double mfma_fp32 = 0;
    double smfmac_fp16 = 0;

    // Used for sorting
    bool operator<(const Result& other) {
        return device < other.device;
    }
};

void print_result(const Result& res, uint32_t mask)
{
    if(mask & VALU_FP16) {
        printf("VALU FP16: %8.2f TFLOPS\n", res.valu_fp16 / 1e12);
    }
    if(mask & VALU_FP32) {
        printf("VALU FP32: %8.2f TFLOPS\n", res.valu_fp32 / 1e12);
    }
    if(mask & VALU_FP64) {
        printf("VALU FP64: %8.2f TFLOPS\n", res.valu_fp64 / 1e12);
    }
    if(mask & VALU_INT32) {
        printf("VALU INT32: %8.2f TIOPS\n", res.valu_int32 / 1e12);
    }
    if(mask & MATRIX_FP16) {
        printf("MFMA FP16: %8.2f TFLOPS\n", res.mfma_fp16 / 1e12);
    }
    if(mask & MATRIX_FP32) {
        printf("MFMA FP32: %8.2f TFLOPS\n", res.mfma_fp32 / 1e12);
    }
    if(mask & SMATRIX_FP16) {
        printf("SMFMAC FP16: %8.2f TFLOPS\n", res.smfmac_fp16 / 1e12);
    }
}

Result run_tests(int device, int runs, uint32_t mask)
{
    int device_count;

    HIP_CALL(hipGetDeviceCount(&device_count));

    if(device >= device_count) {
        std::cout << "Device " << device << " does not exist. Skipping..." << std::endl;
        exit(1);
    }

    HIP_CALL(hipSetDevice(device));
    GCNArch arch = get_gcn_arch(device);

    Result res = {.device = device};

    if(mask & VALU_FP16) {
        res.valu_fp16 = fma_throughput_test<float16>(device, 4096, runs);
    }

    if(mask & VALU_FP32) {
        res.valu_fp32 = fma_throughput_test<float>(device, 4096, runs);
    }

    if(mask & VALU_FP64) {
        res.valu_fp64 = fma_throughput_test<double>(device, 4096, runs);
    }

    if(mask & VALU_INT32) {
        res.valu_int32 = fma_throughput_test<int>(device, 4096, runs);
    }

#if !defined(__gfx906__)
    // MFMA available on gfx908+ (excludes gfx906 with rev=6)
    bool has_mfma = arch.major == 0x9 && (arch.minor >= 0x4 || (arch.minor == 0 && arch.rev >= 8));
    
    if(mask & MATRIX_FP16) {
        if(has_mfma) {
            res.mfma_fp16 = matmul_throughput_test<float16, float>(device, 4096, runs);
        } else {
            res.mfma_fp16 = 0;
        }
    }
    
    if(mask & MATRIX_FP32) {
        if(has_mfma) {
            res.mfma_fp32 = matmul_throughput_test<float, float>(device, 4096, runs);
        } else {
            res.mfma_fp32 = 0;
        }
    }
#else
    // MFMA not available when compiling for gfx906
    if(mask & MATRIX_FP16) {
        res.mfma_fp16 = 0;
    }
    if(mask & MATRIX_FP32) {
        res.mfma_fp32 = 0;
    }
#endif

#if !defined(__gfx906__) && !defined(__gfx908__) && !defined(__gfx90a__)
    if(mask & SMATRIX_FP16) {
        // SMFMAC only available on gfx940 (MI300) and later, not on gfx906, gfx908, or gfx90a
        if(arch.major == 0x9 && arch.minor >= 0x4) {
            res.smfmac_fp16 = sparse_matmul_throughput_test<float16, float>(device, 4096, runs);
        } else {
            res.smfmac_fp16 = 0;
        }
    }
#else
    // SMFMAC not available when compiling for gfx906, gfx908, or gfx90a
    if(mask & SMATRIX_FP16) {
        res.smfmac_fp16 = 0;
    }
#endif

    return res;
}

// Use fork() followed by exec() to run child process. For some reason
// rocprof does not pick up the child processes when only fork() is
// used.
pid_t fork_process(int device, int runs, uint32_t mask, int fd)
{
    pid_t pid = fork();

    if(pid != 0) {
        return pid;
    }

    std::string str_device = std::to_string(device);
    std::string str_runs = std::to_string(runs);
    std::string str_mask = std::to_string(mask);
    std::string str_fd = std::to_string(fd);

    char* const args[] = {
        (char*)"CHILD",
        (char*)str_device.c_str(),
        (char*)str_runs.c_str(),
        (char*)str_mask.c_str(),
        (char*)str_fd.c_str(),
        NULL
    };

    execv("/proc/self/exe", args);
    std::cout << "execv() failed: " << std::strerror(errno) << std::endl;
    exit(1);
}

void run(std::vector<int>& devices, int runs, uint32_t mask)
{
    std::vector<pid_t> pids;

    // We will receive results from the child processes using a pipe
    int fd[2];

    if(pipe(fd)) {
        std::cout << std::strerror(errno) << std::endl;
        exit(1);
    }

    // Start a new process for each GPU
    for(auto d : devices) {
        pid_t pid = fork_process(d, runs, mask, fd[1]);
        
        pids.push_back(pid);
    }

    // Wait for all processes to finish
    for(auto pid : pids) {
        int status;
        waitpid(pid, &status, 0);
    }

    // Set the read to non-blocking
    int flags = fcntl(fd[0], F_GETFL, 0);
    fcntl(fd[0], F_SETFL, flags | O_NONBLOCK);

    // Read records from pipe
    std::vector<Result> results(pids.size());
    int count = read(fd[0], results.data(), results.size() * sizeof(Result)) / sizeof(Result);

    results.resize(count);

    // Sort results by GPU id
    std::sort(results.begin(), results.end());
 
    // Print results
    for(auto r : results) {
        std::cout << std::endl << "GPU " << r.device << std::endl;
        print_result(r, mask);
    }

    Result total;
    for(auto r : results) {
        total.valu_fp16 += r.valu_fp16;
        total.valu_fp32 += r.valu_fp32;
        total.valu_fp64 += r.valu_fp64;
        total.valu_int32 += r.valu_int32;
        total.mfma_fp16 += r.mfma_fp16;
        total.mfma_fp32 += r.mfma_fp32;
        total.smfmac_fp16 += r.smfmac_fp16;
    }
    std::cout << std::endl << "System total" << std::endl;
    print_result(total, mask);
}


void usage()
{
    std::cout << "--device  ID          Use device with the given numerical ID" << std::endl;
    std::cout << "--devices IDS | ALL   Comma-separated list of device Ids (e.g., 1,2,3)" << std::endl;
    std::cout << "                      ALL for all devices" << std::endl;                                  
    std::cout << "--runs    RUNS        Number of times each kernel is dispatched" << std::endl;

    std::cout << "--fp16                Run FP16 (VALU) test" << std::endl;
    std::cout << "--fp32                Run FP32 (VALU) test" << std::endl;
    std::cout << "--fp64                Run FP64 (VALU) test" << std::endl;
    std::cout << "--int32               Run INT32 (VALU) test" << std::endl;
    std::cout << "--matfp16             Run FP16 (MFMA) test" << std::endl;
    std::cout << "--matfp32             Run FP32 (MFMA) test" << std::endl;
    std::cout << "--smatfp16            Run FP16 (SMFMAC) test" << std::endl;
}

int main(int argc, char** argv)
{
    if(std::string(argv[0]) == "CHILD") {
        int device = atoi(argv[1]);
        int runs = atoi(argv[2]);
        uint32_t mask = atoi(argv[3]);
        int fd = atoi(argv[4]);

        Result res = run_tests(device, runs, mask);

        write(fd, &res, sizeof(res));
        return 0;
    }

    int runs = 1;

    uint32_t mask = 0;
    bool all_devices = false;
    std::vector<int> devices;
    int device_count;
    int device = 0;

    HIP_CALL(hipGetDeviceCount(&device_count));

    int i = 1;
    while(i < argc) {
        std::string arg = std::string(argv[i]);

        if(arg == "--help") {
            usage();
            return 0;
        } else if(arg == "--device") {
            devices.push_back(atoi(argv[i + 1]));
            // Skip next 
            i++;
        } else if(arg == "--devices") {
            // Parse comma-separated string of numbers
            std::string s(argv[i + 1]);

            if(s == "all" || s == "ALL") {
                all_devices = true;
            } else {
                std::stringstream ss(s);
                std::string r;
                while(getline(ss, r, ',')) {
                    devices.push_back(std::stoi(r));
                }
            }
            // Skip next 
            i++;
        } else if(arg == "--runs") {
            runs = atoi(argv[i + 1]);

            // Skip next
            i++;
        } else if(arg == "--fp32") {
            mask |= VALU_FP32;
        } else if(arg == "--fp64") {
            mask |= VALU_FP64;
        } else if(arg == "--fp16") {
            mask |= VALU_FP16;
        } else if(arg == "--int32") {
            mask |= VALU_INT32;
        } else if(arg == "--matfp16") {
            mask |= MATRIX_FP16;
        } else if(arg == "--matfp32") {
            mask |= MATRIX_FP32;
        } else if(arg == "--smatfp16") {
            mask |= SMATRIX_FP16;
        } else {
            std::cout << "Invalid argument '" << arg << "'" << std::endl;
            std::cout << std::endl;
            usage();
            return 1;
        }

        i++;
    }

    if(all_devices) {
        for(int i = 0; i < device_count; i++ ){
            devices.push_back(i);
        }
    }

    // Verify device ID's
    for(auto d : devices) {
        if(d >= device_count) {
            std::cout << "Invalid device ordinal: " << d << std::endl;
            return 1;
        }
    }

    if(devices.size() == 0) {
        devices.push_back(0);
    }

    if(mask == 0) {
        mask = ALL;
    }

    run(devices, runs, mask);

    return 0;
}


