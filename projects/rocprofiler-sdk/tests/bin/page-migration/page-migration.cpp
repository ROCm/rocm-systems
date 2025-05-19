// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <hip/hip_runtime.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            auto _hip_api_print_lk = auto_lock_t{print_lock};                                      \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

using auto_lock_t = std::unique_lock<std::mutex>;
auto print_lock   = std::mutex{};

__global__ void
kernel(size_t* __restrict__ data, int size)
{
    int x      = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int stride = hipBlockDim_x * hipGridDim_x;

    for(int i = x; i < size; i += stride)
    {
        data[i] *= 2;
    }
}

struct mmap_allocator
{
    explicit mmap_allocator(size_t num_pages)
    {
        m_size    = num_pages * sysconf(_SC_PAGE_SIZE);
        void* ret = mmap(nullptr,                 // addr: null. Kernel gives us page-aligned memory
                         m_size,                  // length: num bytes to allocate
                         PROT_WRITE | PROT_READ,  // mem_prot: Allow read/write
                         MAP_ANONYMOUS | MAP_PRIVATE,  // flags: No file handle
                         -1,                           // no fd, use memory "MAP_ANONYMOUS"
                         0);                           // offset into fd
        if(ret == ((void*) -1))                        // NOLINT(performance-no-int-to-ptr)
        {
            auto ecode = errno;
            fprintf(stderr, "mmap error %d: %s", ecode, strerror(ecode));
            throw std::runtime_error("mmap failed");
        }
        else
        {
            m_addr = ret;
            ::memset(m_addr, 0, m_size);
        }
    }

    ~mmap_allocator()
    {
        auto ret = munmap(m_addr, m_size);
        if(ret != 0) perror("munmap failed");
    }

    mmap_allocator(const mmap_allocator&)     = delete;
    mmap_allocator(mmap_allocator&&) noexcept = default;
    mmap_allocator& operator=(const mmap_allocator&) = delete;
    mmap_allocator& operator=(mmap_allocator&&) noexcept = default;

    template <typename Up>
    Up* get() const
    {
        static_assert(!std::is_pointer<Up>::value, "must not be pointer type");
        return static_cast<Up*>(m_addr);
    }

private:
    size_t m_size = 0;
    void*  m_addr = nullptr;
};

int
run_test(int dev_idx, int num_iter)
{
    using namespace std::chrono_literals;

    constexpr size_t NUM_PAGES       = 512;
    const size_t     PAGE_SIZE_BYTES = ::sysconf(_SC_PAGE_SIZE);

    const size_t elem_count = (NUM_PAGES * PAGE_SIZE_BYTES) / sizeof(size_t);

    auto  alloc  = mmap_allocator(NUM_PAGES);
    void* data_v = alloc.get<void>();
    auto* data   = alloc.get<size_t>();

    for(size_t i = 0; i < elem_count; ++i)
        if(data[i] != 0) throw std::runtime_error{"bad init"};

    printf("Allocated size: %lu bytes (%lu KB), (%lu MB), %zu elements @ %p\n",
           sizeof(size_t) * elem_count,
           sizeof(size_t) * elem_count / 1024,
           sizeof(size_t) * elem_count / 1024 / 1024,
           elem_count,
           data_v);

    HIP_API_CALL(hipSetDevice(dev_idx));
    HIP_API_CALL(hipHostRegister(data, elem_count * sizeof(size_t), hipHostRegisterDefault));

    constexpr size_t MAPS_BUFFER_SIZE = 1024 * 1024;

    char maps[MAPS_BUFFER_SIZE];
    std::memset(maps, '\0', MAPS_BUFFER_SIZE);
    auto fd = open("/proc/self/maps", O_RDONLY | O_CLOEXEC);

    if(fd == -1)
    {
        auto ecode = errno;
        fprintf(stderr, "mmap error %d: %s", ecode, strerror(ecode));
        exit(-1);
    }

    auto bytes = read(fd, maps, MAPS_BUFFER_SIZE - 1);
    if(bytes == -1)
    {
        auto ecode = errno;
        fprintf(stderr, "mmap error %d: %s", ecode, strerror(ecode));
        exit(-1);
    }
    close(fd);

    std::string_view maps_data{maps, static_cast<size_t>(bytes)};
    std::cout << "------------\n";
    std::cout << maps_data;
    std::cout << "------------\n";
    std::istringstream maps_stream{maps_data.data()};
    std::string        line(1024, '\0');

    while(std::getline(maps_stream, line))
    {
        char __[1024];
        int  _{};

        void* start{};
        void* end{};

        auto ret =
            std::sscanf(line.data(), "%p-%p %s %d %d:%d %d\n", &start, &end, __, &_, &_, &_, &_);
        if(ret > 0 && (start == data_v))
        {
            size_t ptr_diff = ((size_t) end - (size_t) start);
            printf("Found match: %zu %zu KB, %zu 4K > %s\n",
                   ptr_diff,
                   ptr_diff / 1024,
                   ptr_diff / 4096,
                   line.data());
        }
    }

    hipStream_t stream{};
    HIP_API_CALL(hipStreamCreate(&stream));

    for(int iter = 0; iter < num_iter; ++iter)
    {
        for(size_t i = 0; i < elem_count; ++i)
        {
            data[i] = i;
        }

        // std::cout << "launching..." << std::endl;
        hipLaunchKernelGGL(kernel, 1024, 1024, 0, stream, data, elem_count);

        HIP_API_CALL(hipStreamSynchronize(stream));

        for(size_t i = 0; i < elem_count; ++i)
        {
            const auto data_i = data[i];
            if(data_i != (i * 2))
            {
                auto msg = std::stringstream{};
                msg << "GPU computed value at " << i << " in iteration " << iter
                    << " is incorrect. Expected " << (i * 2) << ", found " << data_i;
                throw std::runtime_error{msg.str()};
            }
        }
    }

    HIP_API_CALL(hipStreamSynchronize(stream));
    HIP_API_CALL(hipStreamDestroy(stream));
    HIP_API_CALL(hipHostUnregister(data));

    return 0;
}

int
main(int argc, const char** argv)
{
    using namespace std::chrono_literals;

    const auto usage_msg = [](const char** _argv) {
        fprintf(stderr, "usage: %s <NUMBER OF THREADS> <ITERATIONS PER THREAD>\n", _argv[0]);
    };

    if(argc != 3)
    {
        usage_msg(argv);
        exit(EXIT_FAILURE);
    }

    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            usage_msg(argv);
            exit(EXIT_SUCCESS);
        }
    }

    const auto num_threads = std::atoi(argv[1]);
    if(num_threads < 1)
    {
        fprintf(stderr, "Error: Invalid value %d for num_threads (min 1)\n", num_threads);
        exit(EXIT_FAILURE);
    }

    const auto num_iter = std::atoi(argv[2]);
    if(num_iter < 1)
    {
        fprintf(stderr, "Error: Invalid value %d for num_iter (min 1)\n", num_iter);
        exit(EXIT_FAILURE);
    }

    int ndevice = 0;
    HIP_API_CALL(hipGetDeviceCount(&ndevice));

    run_test(0, num_iter);

    std::vector<std::thread> threads;
    threads.reserve(num_threads);

    std::cerr << "Running " << num_iter << " iterations/thread on " << num_threads << " threads\n";
    for(auto i = 0; i < num_threads; ++i)
    {
        auto _dev_idx = i % ndevice;
        threads.emplace_back([_dev_idx, _num_iter = num_iter]() { run_test(_dev_idx, _num_iter); });
    }

    std::cerr << "Waiting for threads\n";
    for(auto& t : threads)
    {
        t.join();
    }

    for(int i = 0; i < ndevice; ++i)
    {
        HIP_API_CALL(hipSetDevice(i));
        HIP_API_CALL(hipDeviceSynchronize());
        HIP_API_CALL(hipDeviceReset());
    }

    return 0;
}
