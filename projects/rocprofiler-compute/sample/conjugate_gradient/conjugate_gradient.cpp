// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "cg_args.hpp"

#include <hip/hip_runtime.h>
#include <sysexits.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace
{
constexpr std::uint32_t default_processes = 2;
constexpr std::uint32_t default_rounds    = 400;
constexpr std::uint32_t default_rows      = 65536;
constexpr std::uint32_t default_seed      = 12345;
constexpr std::uint32_t default_device    = 0;

constexpr std::string_view module_a_filename = "cg_module_a.hsaco";
constexpr std::string_view module_b_filename = "cg_module_b.hsaco";

enum class KernelKind
{
    spmv,
    update,
};

struct Options
{
    std::uint32_t                processes           = default_processes;
    std::vector<KernelKind>      kernels             = {KernelKind::spmv, KernelKind::update};
    std::uint32_t                rounds              = default_rounds;
    std::uint32_t                rows                = default_rows;
    std::uint32_t                seed                = default_seed;
    std::uint32_t                device              = default_device;
    bool                         rotate_code_objects = false;
    bool                         show_help           = false;
    std::filesystem::path        module_directory;
    std::optional<std::uint32_t> child_index;
};

struct ModulePaths
{
    std::filesystem::path module_a;
    std::filesystem::path module_b;
};

struct Workload
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

std::string_view kernel_name(KernelKind kernel)
{
    switch (kernel)
    {
    case KernelKind::spmv:
        return "spmv";
    case KernelKind::update:
        return "update";
    }
    throw std::logic_error("unrecognized kernel kind");
}

std::string_view kernel_symbol(KernelKind kernel)
{
    switch (kernel)
    {
    case KernelKind::spmv:
        return "kernel_spmv_csr";
    case KernelKind::update:
        return "kernel_cg_update_reduce";
    }
    throw std::logic_error("unrecognized kernel kind");
}

void print_usage(std::ostream& stream, const char* program)
{
    stream << "Usage: " << program
           << " [OPTIONS]\n"
              "\n"
              "Run conjugate-gradient iteration fragments in independent GPU processes.\n"
              "\n"
              "Options:\n"
              "  -p, --processes <N>       Number of child processes (default: "
           << default_processes
           << ")\n"
              "  -k, --kernels <LIST>      Comma-separated spmv/update assignments\n"
              "                            (default: spmv,update)\n"
              "  -r, --rounds <N>          Kernel repetitions (default: "
           << default_rounds
           << ")\n"
              "  -n, --rows <N>            Matrix rows (default: "
           << default_rows
           << ")\n"
              "  -s, --seed <N>            Matrix RNG seed (default: "
           << default_seed
           << ")\n"
              "  -d, --device <N>          Base device ordinal (default: "
           << default_device
           << ")\n"
              "      --rotate-code-objects Reverse A/B load order for odd children\n"
              "      --module-dir <PATH>   Directory containing the two HSACO modules\n"
              "                            (default: executable directory)\n"
              "  -h, --help                Show this help message\n";
}

void reject_duplicate(bool& seen, std::string_view option)
{
    if (seen)
    {
        throw std::invalid_argument("option specified more than once: " + std::string(option));
    }
    seen = true;
}

std::optional<std::string_view> take_option_value(std::string_view argument,
                                                  std::string_view short_option,
                                                  std::string_view long_option,
                                                  int&             argument_index,
                                                  int              argument_count,
                                                  char*            arguments[],
                                                  bool&            seen)
{
    const bool matches_short  = !short_option.empty() && argument == short_option;
    const bool matches_long   = argument == long_option;
    const bool matches_inline = argument.size() > long_option.size() &&
                                argument.compare(0, long_option.size(), long_option) == 0 &&
                                argument[long_option.size()] == '=';
    if (!matches_short && !matches_long && !matches_inline)
    {
        return std::nullopt;
    }

    reject_duplicate(seen, long_option);
    if (matches_inline)
    {
        const std::string_view value = argument.substr(long_option.size() + 1);
        if (value.empty())
        {
            throw std::invalid_argument("option requires a value: " + std::string(long_option));
        }
        return value;
    }

    if (argument_index + 1 >= argument_count)
    {
        throw std::invalid_argument("option requires a value: " + std::string(long_option));
    }
    return std::string_view(arguments[++argument_index]);
}

std::uint32_t parse_unsigned(std::string_view value, std::string_view option, bool allow_zero)
{
    if (value.empty() || value.front() == '-')
    {
        throw std::invalid_argument("invalid value for " + std::string(option) + ": " +
                                    std::string(value));
    }

    std::uint32_t parsed_value = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed_value);
    if (error != std::errc{} || end != value.data() + value.size() || (!allow_zero && parsed_value == 0))
    {
        throw std::invalid_argument("invalid value for " + std::string(option) + ": " +
                                    std::string(value));
    }
    return parsed_value;
}

std::vector<KernelKind> parse_kernels(std::string_view value)
{
    std::vector<KernelKind> kernels;
    std::size_t             start = 0;
    while (start <= value.size())
    {
        const std::size_t end = value.find(',', start);
        const std::size_t count = end == std::string_view::npos ? value.size() - start : end - start;
        const std::string_view name = value.substr(start, count);
        if (name == "spmv")
        {
            kernels.push_back(KernelKind::spmv);
        }
        else if (name == "update")
        {
            kernels.push_back(KernelKind::update);
        }
        else
        {
            throw std::invalid_argument("invalid kernel name: " + std::string(name));
        }

        if (end == std::string_view::npos)
        {
            break;
        }
        start = end + 1;
    }
    return kernels;
}

Options parse_arguments(int argument_count, char* arguments[])
{
    Options options;
    bool    processes_seen        = false;
    bool    kernels_seen          = false;
    bool    rounds_seen           = false;
    bool    rows_seen             = false;
    bool    seed_seen             = false;
    bool    device_seen           = false;
    bool    rotation_seen         = false;
    bool    module_directory_seen = false;
    bool    child_seen            = false;
    bool    help_seen             = false;

    for (int argument_index = 1; argument_index < argument_count; ++argument_index)
    {
        const std::string_view argument = arguments[argument_index];
        if (const auto value = take_option_value(argument,
                                                 "-p",
                                                 "--processes",
                                                 argument_index,
                                                 argument_count,
                                                 arguments,
                                                 processes_seen))
        {
            options.processes = parse_unsigned(*value, "--processes", false);
        }
        else if (const auto value = take_option_value(argument,
                                                      "-k",
                                                      "--kernels",
                                                      argument_index,
                                                      argument_count,
                                                      arguments,
                                                      kernels_seen))
        {
            options.kernels = parse_kernels(*value);
        }
        else if (const auto value = take_option_value(argument,
                                                      "-r",
                                                      "--rounds",
                                                      argument_index,
                                                      argument_count,
                                                      arguments,
                                                      rounds_seen))
        {
            options.rounds = parse_unsigned(*value, "--rounds", false);
        }
        else if (const auto value =
                     take_option_value(argument, "-n", "--rows", argument_index, argument_count, arguments, rows_seen))
        {
            options.rows = parse_unsigned(*value, "--rows", false);
        }
        else if (const auto value =
                     take_option_value(argument, "-s", "--seed", argument_index, argument_count, arguments, seed_seen))
        {
            options.seed = parse_unsigned(*value, "--seed", true);
        }
        else if (const auto value = take_option_value(argument,
                                                      "-d",
                                                      "--device",
                                                      argument_index,
                                                      argument_count,
                                                      arguments,
                                                      device_seen))
        {
            options.device = parse_unsigned(*value, "--device", true);
        }
        else if (const auto value = take_option_value(argument,
                                                      "",
                                                      "--module-dir",
                                                      argument_index,
                                                      argument_count,
                                                      arguments,
                                                      module_directory_seen))
        {
            options.module_directory = std::filesystem::path(*value);
        }
        else if (const auto value =
                     take_option_value(argument, "", "--child", argument_index, argument_count, arguments, child_seen))
        {
            options.child_index = parse_unsigned(*value, "--child", true);
        }
        else if (argument == "--rotate-code-objects")
        {
            reject_duplicate(rotation_seen, "--rotate-code-objects");
            options.rotate_code_objects = true;
        }
        else if (argument == "-h" || argument == "--help")
        {
            reject_duplicate(help_seen, "--help");
            options.show_help = true;
        }
        else
        {
            throw std::invalid_argument("unknown argument: " + std::string(argument));
        }
    }

    if (options.child_index && *options.child_index >= options.processes)
    {
        throw std::invalid_argument("--child must be less than --processes");
    }
    return options;
}

std::filesystem::path running_executable()
{
    std::vector<char> path_buffer(256);
    while (path_buffer.size() <= 1024 * 1024)
    {
        const ssize_t path_length = ::readlink("/proc/self/exe", path_buffer.data(), path_buffer.size());
        if (path_length < 0)
        {
            throw std::runtime_error("cannot resolve /proc/self/exe: " +
                                     std::string(std::strerror(errno)));
        }
        if (static_cast<std::size_t>(path_length) < path_buffer.size())
        {
            return std::filesystem::path(
                std::string(path_buffer.data(), static_cast<std::size_t>(path_length)));
        }
        path_buffer.resize(path_buffer.size() * 2);
    }
    throw std::runtime_error("resolved executable path is unexpectedly long");
}

std::filesystem::path resolve_module_directory(const std::filesystem::path& configured_directory)
{
    const std::filesystem::path directory = configured_directory.empty()
                                                ? running_executable().parent_path()
                                                : configured_directory;
    std::error_code             error;
    const std::filesystem::path absolute_directory = std::filesystem::absolute(directory, error);
    if (error)
    {
        throw std::runtime_error("cannot resolve module directory " + directory.string() + ": " +
                                 error.message());
    }
    return absolute_directory.lexically_normal();
}

void validate_module(const std::filesystem::path& module_path)
{
    std::error_code error;
    const bool      is_regular_file = std::filesystem::is_regular_file(module_path, error);
    if (error || !is_regular_file)
    {
        const std::string detail = error ? ": " + error.message() : "";
        throw std::runtime_error("module is not a regular file: " + module_path.string() + detail);
    }
    if (::access(module_path.c_str(), R_OK) != 0)
    {
        throw std::runtime_error("module is not readable: " + module_path.string() + ": " +
                                 std::string(std::strerror(errno)));
    }
}

ModulePaths preflight_modules(const std::filesystem::path& module_directory)
{
    ModulePaths modules{
        module_directory / module_a_filename,
        module_directory / module_b_filename,
    };
    validate_module(modules.module_a);
    validate_module(modules.module_b);
    return modules;
}

void check_hip(hipError_t error, std::string_view action)
{
    if (error != hipSuccess)
    {
        throw std::runtime_error(std::string(action) + ": " + hipGetErrorString(error));
    }
}

template<typename Value>
class DeviceAllocation
{
public:
    explicit DeviceAllocation(std::size_t element_count)
    {
        if (element_count > std::numeric_limits<std::size_t>::max() / sizeof(Value))
        {
            throw std::overflow_error("device allocation size overflow");
        }
        check_hip(hipMalloc(reinterpret_cast<void**>(&pointer_), element_count * sizeof(Value)),
                  "hipMalloc");
    }

    ~DeviceAllocation()
    {
        if (pointer_ != nullptr)
        {
            (void)hipFree(pointer_);
        }
    }

    DeviceAllocation(const DeviceAllocation&)            = delete;
    DeviceAllocation& operator=(const DeviceAllocation&) = delete;

    Value* get() const { return pointer_; }

private:
    Value* pointer_ = nullptr;
};

class LoadedModule
{
public:
    ~LoadedModule()
    {
        if (module_ != nullptr)
        {
            (void)hipModuleUnload(module_);
        }
    }

    LoadedModule(const LoadedModule&)            = delete;
    LoadedModule& operator=(const LoadedModule&) = delete;
    LoadedModule()                               = default;

    void load(const std::filesystem::path& path)
    {
        check_hip(hipModuleLoad(&module_, path.c_str()), "hipModuleLoad(" + path.string() + ")");
    }

    hipModule_t get() const { return module_; }

private:
    hipModule_t module_ = nullptr;
};

template<typename Value>
void copy_to_device(DeviceAllocation<Value>&  destination,
                    const std::vector<Value>& source,
                    std::string_view          name)
{
    check_hip(hipMemcpy(destination.get(), source.data(), source.size() * sizeof(Value), hipMemcpyHostToDevice),
              "copy " + std::string(name) + " to device");
}

Workload make_workload(std::uint32_t rows, std::uint32_t seed, std::uint32_t partial_count)
{
    Workload workload;
    workload.row_offsets.resize(static_cast<std::size_t>(rows) + 1);
    workload.column_indices.reserve(static_cast<std::size_t>(rows) * 8);
    workload.values.reserve(static_cast<std::size_t>(rows) * 8);
    workload.p.resize(rows);
    workload.q.resize(rows);
    workload.x.assign(rows, 0.0F);
    workload.r.resize(rows);
    workload.partials.assign(partial_count, 0.0F);

    std::mt19937                           random_generator(seed);
    std::uniform_real_distribution<double> uniform_distribution(0.0, 1.0);
    for (std::uint32_t row = 0; row < rows; ++row)
    {
        const double uniform_value = uniform_distribution(random_generator);
        const int    row_nonzeros =
            std::clamp(static_cast<int>(4.0 / std::pow(1.0 - 0.999 * uniform_value, 0.7)), 1, 512);
        if (workload.column_indices.size() >
            std::numeric_limits<std::uint32_t>::max() - static_cast<std::uint32_t>(row_nonzeros))
        {
            throw std::overflow_error("CSR nonzero count exceeds the 32-bit index range");
        }

        for (int entry = 0; entry < row_nonzeros; ++entry)
        {
            workload.column_indices.push_back(random_generator() % rows);
            workload.values.push_back(0.5F + static_cast<float>(entry % 7) * 0.1F);
        }
        workload.row_offsets[row + 1] = static_cast<std::uint32_t>(workload.column_indices.size());
    }

    for (std::uint32_t row = 0; row < rows; ++row)
    {
        workload.p[row] = 1.0F / static_cast<float>(1 + row % 13);
        workload.q[row] = 0.5F + static_cast<float>(row % 7) * 0.1F;
        workload.r[row] = 0.25F + static_cast<float>(row % 5) * 0.1F;
    }
    return workload;
}

std::string join_kernel_names(const std::vector<KernelKind>& kernels)
{
    std::string joined;
    for (std::size_t index = 0; index < kernels.size(); ++index)
    {
        if (index != 0)
        {
            joined += ',';
        }
        joined += kernel_name(kernels[index]);
    }
    return joined;
}

std::vector<std::string> make_child_arguments(const Options& options, std::uint32_t child_index)
{
    std::vector<std::string> arguments{
        "/proc/self/exe",
        "--child",
        std::to_string(child_index),
        "--processes",
        std::to_string(options.processes),
        "--kernels",
        join_kernel_names(options.kernels),
        "--rounds",
        std::to_string(options.rounds),
        "--rows",
        std::to_string(options.rows),
        "--seed",
        std::to_string(options.seed),
        "--device",
        std::to_string(options.device),
        "--module-dir",
        options.module_directory.string(),
    };
    if (options.rotate_code_objects)
    {
        arguments.emplace_back("--rotate-code-objects");
    }
    return arguments;
}

int run_parent(const Options& options)
{
    std::vector<pid_t> child_processes;
    child_processes.reserve(options.processes);
    bool child_failed = false;

    for (std::uint32_t child_index = 0; child_index < options.processes; ++child_index)
    {
        std::vector<std::string> child_arguments = make_child_arguments(options, child_index);
        std::vector<char*>       argument_pointers;
        argument_pointers.reserve(child_arguments.size() + 1);
        for (std::string& argument : child_arguments)
        {
            argument_pointers.push_back(argument.data());
        }
        argument_pointers.push_back(nullptr);

        const pid_t child_process = ::fork();
        if (child_process < 0)
        {
            std::cerr << "conjugate_gradient: fork for child " << child_index
                      << " failed: " << std::strerror(errno) << '\n';
            child_failed = true;
            break;
        }
        if (child_process == 0)
        {
            ::execv("/proc/self/exe", argument_pointers.data());
            const int exec_error = errno;
            std::cerr << "conjugate_gradient: execv for child " << child_index
                      << " failed: " << std::strerror(exec_error) << '\n';
            std::cerr.flush();
            ::_exit(127);
        }
        child_processes.push_back(child_process);
    }

    for (const pid_t child_process : child_processes)
    {
        int   status      = 0;
        pid_t wait_result = 0;
        do
        {
            wait_result = ::waitpid(child_process, &status, 0);
        }
        while (wait_result < 0 && errno == EINTR);

        if (wait_result < 0)
        {
            std::cerr << "conjugate_gradient: waitpid(" << child_process
                      << ") failed: " << std::strerror(errno) << '\n';
            child_failed = true;
        }
        else if (WIFEXITED(status))
        {
            if (WEXITSTATUS(status) != 0)
            {
                std::cerr << "conjugate_gradient: child " << child_process << " exited with status "
                          << WEXITSTATUS(status) << '\n';
                child_failed = true;
            }
        }
        else if (WIFSIGNALED(status))
        {
            std::cerr << "conjugate_gradient: child " << child_process << " terminated by signal "
                      << WTERMSIG(status) << '\n';
            child_failed = true;
        }
        else
        {
            std::cerr << "conjugate_gradient: child " << child_process
                      << " ended in an unexpected state\n";
            child_failed = true;
        }
    }

    return child_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}

void write_summary(const std::string& summary)
{
    std::cout << summary << std::flush;
    if (!std::cout)
    {
        throw std::runtime_error("cannot write child summary");
    }
}

int run_child(const Options& options, const ModulePaths& module_paths)
{
    const std::uint32_t child_index = *options.child_index;

    int device_count = 0;
    check_hip(hipGetDeviceCount(&device_count), "hipGetDeviceCount");
    if (device_count <= 0)
    {
        throw std::runtime_error("no HIP devices are available");
    }
    const std::uint32_t selected_device = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(options.device) + child_index) %
        static_cast<std::uint32_t>(device_count));
    check_hip(hipSetDevice(static_cast<int>(selected_device)), "hipSetDevice");

    const bool   reverse_module_order = options.rotate_code_objects && child_index % 2 != 0;
    LoadedModule module_a;
    LoadedModule module_b;
    if (reverse_module_order)
    {
        module_b.load(module_paths.module_b);
        module_a.load(module_paths.module_a);
    }
    else
    {
        module_a.load(module_paths.module_a);
        module_b.load(module_paths.module_b);
    }

    const std::uint32_t grid_size = static_cast<std::uint32_t>(
        (static_cast<std::uint64_t>(options.rows) + cg_block_size - 1) / cg_block_size);
    Workload workload = make_workload(options.rows, options.seed, grid_size);

    DeviceAllocation<std::uint32_t> device_row_offsets(workload.row_offsets.size());
    DeviceAllocation<std::uint32_t> device_column_indices(workload.column_indices.size());
    DeviceAllocation<float>         device_values(workload.values.size());
    DeviceAllocation<float>         device_p(workload.p.size());
    DeviceAllocation<float>         device_q(workload.q.size());
    DeviceAllocation<float>         device_x(workload.x.size());
    DeviceAllocation<float>         device_r(workload.r.size());
    DeviceAllocation<float>         device_partials(workload.partials.size());

    copy_to_device(device_row_offsets, workload.row_offsets, "row offsets");
    copy_to_device(device_column_indices, workload.column_indices, "column indices");
    copy_to_device(device_values, workload.values, "values");
    copy_to_device(device_p, workload.p, "p");
    copy_to_device(device_q, workload.q, "q");
    copy_to_device(device_x, workload.x, "x");
    copy_to_device(device_r, workload.r, "r");
    copy_to_device(device_partials, workload.partials, "partials");

    const KernelKind  selected_kernel = options.kernels[child_index % options.kernels.size()];
    hipFunction_t     kernel          = nullptr;
    const std::string symbol(kernel_symbol(selected_kernel));
    check_hip(hipModuleGetFunction(&kernel, module_a.get(), symbol.c_str()),
              "hipModuleGetFunction(" + symbol + ")");

    CgArgs kernel_arguments{
        device_row_offsets.get(),
        device_column_indices.get(),
        device_values.get(),
        device_p.get(),
        device_q.get(),
        device_x.get(),
        device_r.get(),
        device_partials.get(),
        options.rows,
        options.rounds,
    };
    std::size_t argument_size          = sizeof(kernel_arguments);
    void*       launch_configuration[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        &kernel_arguments,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        &argument_size,
        HIP_LAUNCH_PARAM_END,
    };
    check_hip(hipModuleLaunchKernel(kernel, grid_size, 1, 1, cg_block_size, 1, 1, 0, nullptr, nullptr, launch_configuration),
              "hipModuleLaunchKernel");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    const std::string module_order = reverse_module_order ? "cg_module_b.hsaco,cg_module_a.hsaco"
                                                          : "cg_module_a.hsaco,cg_module_b.hsaco";
    const std::string summary      = "pid=" + std::to_string(static_cast<long>(::getpid())) +
                                     " child=" + std::to_string(child_index) +
                                     " device=" + std::to_string(selected_device) +
                                     " kernel=" + std::string(kernel_name(selected_kernel)) +
                                     " target=cg_module_a" + " modules=" + module_order +
                                     " rows=" + std::to_string(options.rows) +
                                     " rounds=" + std::to_string(options.rounds) + "\n";
    write_summary(summary);
    return EXIT_SUCCESS;
}
}  // namespace

int main(int argument_count, char* arguments[])
{
    try
    {
        Options options = parse_arguments(argument_count, arguments);
        if (options.show_help)
        {
            print_usage(std::cout, arguments[0]);
            return EXIT_SUCCESS;
        }

        options.module_directory       = resolve_module_directory(options.module_directory);
        const ModulePaths module_paths = preflight_modules(options.module_directory);
        if (options.child_index)
        {
            return run_child(options, module_paths);
        }
        return run_parent(options);
    }
    catch (const std::invalid_argument& error)
    {
        std::cerr << "conjugate_gradient: " << error.what() << "\n\n";
        print_usage(std::cerr, arguments[0]);
        return EX_USAGE;
    }
    catch (const std::exception& error)
    {
        std::cerr << "conjugate_gradient: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
