#ifndef TEST_FRAMEWORK_HPP
#define TEST_FRAMEWORK_HPP

#include "hip_utils.hpp"
#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

namespace test_framework {

class TestBase {
public:
    TestBase(const std::string& name) : test_name_(name) {}

    virtual ~TestBase() = default;

    // Run the test
    int run() {
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Test: " << test_name_ << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        hip_utils::printDeviceInfo();

        std::cout << "\nInitializing test..." << std::endl;
        if (!setup()) {
            std::cerr << "Test setup failed!" << std::endl;
            return 1;
        }

        std::cout << "Running test..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();

        bool success = execute();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Cleaning up..." << std::endl;
        cleanup();

        std::cout << "\n" << std::string(80, '-') << std::endl;
        if (success) {
            std::cout << "Result: PASSED" << std::endl;
        } else {
            std::cout << "Result: FAILED" << std::endl;
        }
        std::cout << "Execution time: " << std::fixed << std::setprecision(3)
                  << (duration.count() / 1000.0) << " ms" << std::endl;
        std::cout << std::string(80, '=') << std::endl;

        return success ? 0 : 1;
    }

protected:
    // Override these in derived classes
    virtual bool setup() = 0;
    virtual bool execute() = 0;
    virtual void cleanup() = 0;

    std::string test_name_;
};

// Simple kernel execution helper
class KernelRunner {
public:
    KernelRunner(const std::string& kernel_name,
                 dim3 grid_size,
                 dim3 block_size)
        : kernel_name_(kernel_name)
        , grid_size_(grid_size)
        , block_size_(block_size) {}

    void printConfig() const {
        std::cout << "Kernel: " << kernel_name_ << std::endl;
        std::cout << "  Grid size:  (" << grid_size_.x << ", "
                  << grid_size_.y << ", " << grid_size_.z << ")" << std::endl;
        std::cout << "  Block size: (" << block_size_.x << ", "
                  << block_size_.y << ", " << block_size_.z << ")" << std::endl;

        size_t total_threads = static_cast<size_t>(grid_size_.x) * grid_size_.y * grid_size_.z *
                               block_size_.x * block_size_.y * block_size_.z;
        std::cout << "  Total threads: " << total_threads << std::endl;
    }

    template<typename KernelFunc, typename... Args>
    void launch(KernelFunc kernel, Args... args) {
        printConfig();

        hipLaunchKernelGGL(kernel, grid_size_, block_size_, 0, 0, args...);
        HIP_CHECK_LAST();
        HIP_CHECK(hipDeviceSynchronize());

        std::cout << "Kernel execution completed successfully" << std::endl;
    }

private:
    std::string kernel_name_;
    dim3 grid_size_;
    dim3 block_size_;
};

// Helper to verify results
template<typename T>
bool verifyResults(const T* results, size_t count, T expected, const std::string& name) {
    bool success = true;
    size_t errors = 0;
    const size_t max_errors_to_print = 10;

    for (size_t i = 0; i < count; i++) {
        if (results[i] != expected) {
            if (errors < max_errors_to_print) {
                std::cerr << "Verification error in " << name
                          << " at index " << i << ": expected "
                          << expected << ", got " << results[i] << std::endl;
            }
            errors++;
            success = false;
        }
    }

    if (errors > 0) {
        std::cerr << "Total verification errors: " << errors << " out of " << count << std::endl;
    } else {
        std::cout << "Verification passed for " << name << " (" << count << " elements)" << std::endl;
    }

    return success;
}

} // namespace test_framework

#endif // TEST_FRAMEWORK_HPP
