#include <hip/hip_runtime.h>
#include <iostream>
#include <iomanip>

// Simple memory copy kernel - one thread per element
__global__ void memoryCopyKernel(const double* __restrict__ X,
                                  double* __restrict__ Y,
                                  size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        Y[idx] = X[idx];
    }
}

int main() {
    const size_t N = 268435456;  // Number of elements
    const size_t bytes = N * sizeof(double);

    // Print array size in GiB
    double sizeGiB = bytes / (1024.0 * 1024.0 * 1024.0);
    std::cout << "Array size: " << std::fixed << std::setprecision(2)
              << sizeGiB << " GiB (" << bytes << " bytes)" << std::endl;
    std::cout << "Number of elements: " << N << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Allocate device memory
    double *d_X, *d_Y;
    hipMalloc(&d_X, bytes);
    hipMalloc(&d_Y, bytes);

    // Allocate and initialize host memory
    double *h_X = new double[N];
    for (size_t i = 0; i < N; i++) {
        h_X[i] = static_cast<double>(i);
    }

    // Copy data to device
    hipMemcpy(d_X, h_X, bytes, hipMemcpyHostToDevice);

    // Launch configuration
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    std::cout << "Launch configuration:" << std::endl;
    std::cout << "  Block size: " << blockSize << std::endl;
    std::cout << "  Grid size: " << gridSize << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Warmup run
    memoryCopyKernel<<<gridSize, blockSize>>>(d_X, d_Y, N);
    hipDeviceSynchronize();

    // Timed runs
    const int numRuns = 1000;
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);

    float totalTime = 0.0f;

    for (int run = 0; run < numRuns; run++) {
        hipEventRecord(start);

        memoryCopyKernel<<<gridSize, blockSize>>>(d_X, d_Y, N);

        hipEventRecord(stop);
        hipEventSynchronize(stop);

        float milliseconds = 0;
        hipEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;
    }

    float avgTime = totalTime / numRuns;

    // Calculate bandwidth
    // Memory copy reads N elements and writes N elements = 2*N*sizeof(double) bytes
    double totalBytesTransferred = 2.0 * N * sizeof(double);
    double bandwidthGBps = (totalBytesTransferred / (avgTime / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Performance Results (averaged over " << numRuns << " runs):" << std::endl;
    std::cout << "  Kernel execution time: " << std::fixed << std::setprecision(3)
              << avgTime << " ms" << std::endl;
    std::cout << "  Data transferred: " << std::setprecision(2)
              << (totalBytesTransferred / (1024.0 * 1024.0 * 1024.0)) << " GiB" << std::endl;
    std::cout << "  Achieved bandwidth: " << std::setprecision(2)
              << bandwidthGBps << " GiB/s" << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Verify correctness
    double *h_Y = new double[N];
    hipMemcpy(h_Y, d_Y, bytes, hipMemcpyDeviceToHost);

    bool correct = true;
    for (size_t i = 0; i < N; i++) {
        if (h_Y[i] != h_X[i]) {
            correct = false;
            std::cout << "Mismatch at index " << i << ": "
                      << h_Y[i] << " != " << h_X[i] << std::endl;
            break;
        }
    }

    if (correct) {
        std::cout << "✓ Verification PASSED - all elements copied correctly" << std::endl;
    } else {
        std::cout << "✗ Verification FAILED" << std::endl;
    }

    std::cout << "\nFirst 5 elements:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  Y[" << i << "] = " << h_Y[i] << std::endl;
    }

    // Cleanup
    hipEventDestroy(start);
    hipEventDestroy(stop);
    delete[] h_X;
    delete[] h_Y;
    hipFree(d_X);
    hipFree(d_Y);

    return 0;
}
