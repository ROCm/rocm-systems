// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Two processes run the same two conjugate-gradient kernels, each process
// loading them in the opposite order, so a given kernel ends up with a
// different process-local code-object ID in each one.
//
// The kernels live in two shared libraries rather than in this binary. Kernels
// compiled into one binary share a single fat binary, so the runtime would load
// them as one code object holding both symbols and there would be nothing to
// tell apart.
//
// HIP defers loading a code object until the first launch out of it, so the
// launch order is what assigns the IDs. The libraries are opened in that same
// order as well, which keeps the split intact when deferred loading is off.
//
// The child re-execs itself so the profiler's runtime hooks initialize inside
// it; a plain fork() is not enough for the child to be picked up.

#include "cg_args.hpp"

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
// CMake supplies the file names, so renaming a library target cannot desync it
// from this source.
constexpr std::array<const char*, 2> cg_kernel_libraries = {CG_SPMV_LIBRARY, CG_UPDATE_LIBRARY};

// SpMV does far more work per round than the update/reduce kernel, so the two
// counts are tuned separately. Both land near 15 ms on MI350X, which is a few
// dozen intervals at the 512 us host-trap default, so neither kernel can slip
// between samples.
constexpr std::array<std::uint32_t, 2> cg_kernel_rounds = {200, 15000};

constexpr const char* cg_launch_symbol = "cg_launch";

// execv() passes this as argv[0], which is how the re-execed child knows its
// role without needing a command-line option.
constexpr const char* cg_child_marker = "cg-child";

constexpr std::uint32_t cg_seed = 12345;

void check_hip(hipError_t error, const char* action)
{
    if (error != hipSuccess)
    {
        throw std::runtime_error(std::string(action) + ": " + hipGetErrorString(error));
    }
}

std::string executable_directory()
{
    std::string path(4096, '\0');
    const ssize_t length = ::readlink("/proc/self/exe", path.data(), path.size());
    if (length <= 0 || static_cast<std::size_t>(length) >= path.size())
    {
        throw std::runtime_error("cannot resolve /proc/self/exe: " +
                                 std::string(std::strerror(errno)));
    }
    path.resize(static_cast<std::size_t>(length));
    return path.substr(0, path.find_last_of('/'));
}

// Owns a dlopen handle and the launch entry point resolved out of it.
class KernelLibrary
{
public:
    explicit KernelLibrary(const std::string& path)
    {
        handle_ = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle_ == nullptr)
        {
            throw std::runtime_error("dlopen(" + path + "): " + std::string(::dlerror()));
        }
        launch_ = reinterpret_cast<CgLaunch>(::dlsym(handle_, cg_launch_symbol));
        if (launch_ == nullptr)
        {
            throw std::runtime_error("dlsym(" + std::string(cg_launch_symbol) + ") in " + path);
        }
    }

    ~KernelLibrary()
    {
        if (handle_ != nullptr)
        {
            (void)::dlclose(handle_);
        }
    }

    KernelLibrary(const KernelLibrary&)            = delete;
    KernelLibrary& operator=(const KernelLibrary&) = delete;

    void launch(const CgArgs& arguments) const { launch_(arguments); }

private:
    void*    handle_ = nullptr;
    CgLaunch launch_ = nullptr;
};

template<typename Value>
class DeviceBuffer
{
public:
    explicit DeviceBuffer(const std::vector<Value>& host_data)
    {
        const std::size_t bytes = host_data.size() * sizeof(Value);
        check_hip(hipMalloc(reinterpret_cast<void**>(&pointer_), bytes), "hipMalloc");
        check_hip(hipMemcpy(pointer_, host_data.data(), bytes, hipMemcpyHostToDevice),
                  "hipMemcpy to device");
    }

    ~DeviceBuffer()
    {
        if (pointer_ != nullptr)
        {
            (void)hipFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&)            = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Value* get() const { return pointer_; }

private:
    Value* pointer_ = nullptr;
};

struct HostWorkload
{
    std::vector<std::uint32_t> row_offsets;
    std::vector<std::uint32_t> column_indices;
    std::vector<float>         values;
    std::vector<float>         p;
    std::vector<float>         q;
    std::vector<float>         x;
    std::vector<float>         r;
    std::vector<float>         partials;
};

// A CSR matrix with power-law row lengths, which gives the SpMV kernel uneven
// per-row work and so a more interesting sample distribution than a dense band.
HostWorkload make_host_workload(std::uint32_t partial_count)
{
    HostWorkload workload;
    workload.row_offsets.resize(static_cast<std::size_t>(cg_rows) + 1);
    workload.column_indices.reserve(static_cast<std::size_t>(cg_rows) * 8);
    workload.values.reserve(static_cast<std::size_t>(cg_rows) * 8);
    workload.p.resize(cg_rows);
    workload.q.resize(cg_rows);
    workload.x.assign(cg_rows, 0.0F);
    workload.r.resize(cg_rows);
    workload.partials.assign(partial_count, 0.0F);

    std::mt19937                           random_generator(cg_seed);
    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
    for (std::uint32_t row = 0; row < cg_rows; ++row)
    {
        const double uniform_value = uniform_distribution(random_generator);
        const int    row_nonzeros =
            std::clamp(static_cast<int>(4.0 / std::pow(1.0 - 0.999 * uniform_value, 0.7)), 1, 512);
        for (int entry = 0; entry < row_nonzeros; ++entry)
        {
            workload.column_indices.push_back(random_generator() % cg_rows);
            workload.values.push_back(0.5F + static_cast<float>(entry % 7) * 0.1F);
        }
        workload.row_offsets[row + 1] = static_cast<std::uint32_t>(workload.column_indices.size());
    }

    for (std::uint32_t row = 0; row < cg_rows; ++row)
    {
        workload.p[row] = 1.0F / static_cast<float>(1 + row % 13);
        workload.q[row] = 0.5F + static_cast<float>(row % 7) * 0.1F;
        workload.r[row] = 0.25F + static_cast<float>(row % 5) * 0.1F;
    }
    return workload;
}

// Role 0 runs SpMV first, role 1 runs update/reduce first. Both roles run both
// kernels; only the order differs.
std::array<std::size_t, 2> library_order(int role)
{
    if (role == 0)
    {
        return {0, 1};
    }
    return {1, 0};
}

int run_role(int role)
{
    const std::string directory = executable_directory();
    const auto        order     = library_order(role);

    std::array<std::unique_ptr<KernelLibrary>, 2> libraries;
    for (std::size_t position = 0; position < order.size(); ++position)
    {
        libraries[position] =
            std::make_unique<KernelLibrary>(directory + "/" + cg_kernel_libraries[order[position]]);
    }

    const std::uint32_t grid_size    = (cg_rows + cg_block_size - 1) / cg_block_size;
    const HostWorkload  host         = make_host_workload(grid_size);
    DeviceBuffer        row_offsets(host.row_offsets);
    DeviceBuffer        column_indices(host.column_indices);
    DeviceBuffer        values(host.values);
    DeviceBuffer        p(host.p);
    DeviceBuffer        q(host.q);
    DeviceBuffer        x(host.x);
    DeviceBuffer        r(host.r);
    DeviceBuffer        partials(host.partials);

    CgArgs arguments{
        row_offsets.get(),
        column_indices.get(),
        values.get(),
        p.get(),
        q.get(),
        x.get(),
        r.get(),
        partials.get(),
        cg_rows,
        0,
    };

    std::string launched;
    for (std::size_t position = 0; position < order.size(); ++position)
    {
        const std::size_t library_index = order[position];
        arguments.rounds                = cg_kernel_rounds[library_index];
        libraries[position]->launch(arguments);
        launched += (position == 0 ? "" : ",");
        launched += cg_kernel_libraries[library_index];
    }
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    std::cout << "pid=" << ::getpid() << " role=" << role << " order=" << launched << '\n'
              << std::flush;
    return EXIT_SUCCESS;
}

int run_parent()
{
    const pid_t child_process = ::fork();
    if (child_process < 0)
    {
        std::cerr << "conjugate_gradient: fork failed: " << std::strerror(errno) << '\n';
        return EXIT_FAILURE;
    }
    if (child_process == 0)
    {
        char* const child_arguments[] = {const_cast<char*>(cg_child_marker), nullptr};
        ::execv("/proc/self/exe", child_arguments);
        std::cerr << "conjugate_gradient: execv failed: " << std::strerror(errno) << '\n';
        std::cerr.flush();
        ::_exit(127);
    }

    // The parent makes no HIP call before fork(), so the child inherits a clean
    // runtime state.
    const int parent_status = run_role(0);

    int   status      = 0;
    pid_t wait_result = 0;
    do
    {
        wait_result = ::waitpid(child_process, &status, 0);
    }
    while (wait_result < 0 && errno == EINTR);

    if (wait_result < 0)
    {
        std::cerr << "conjugate_gradient: waitpid failed: " << std::strerror(errno) << '\n';
        return EXIT_FAILURE;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
    {
        std::cerr << "conjugate_gradient: child " << child_process << " did not exit cleanly\n";
        return EXIT_FAILURE;
    }
    return parent_status;
}
}  // namespace

int main(int argument_count, char* arguments[])
{
    try
    {
        if (argument_count > 0 && std::strcmp(arguments[0], cg_child_marker) == 0)
        {
            return run_role(1);
        }
        return run_parent();
    }
    catch (const std::exception& error)
    {
        std::cerr << "conjugate_gradient: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
