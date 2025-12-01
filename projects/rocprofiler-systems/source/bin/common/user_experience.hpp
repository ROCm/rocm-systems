// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace rocprofsys
{
namespace user_experience
{
inline void
print_cheatsheet()
{
    std::cout << R"(
╔════════════════════════════════════════════════════════════════════════╗
║              ROCm Systems Profiler Quick Reference                     ║
╠════════════════════════════════════════════════════════════════════════╣
║                                                                        ║
║ BASIC USAGE                                                            ║
║   rocprof-sys-sample --quick -- ./app                                  ║
║   rocprof-sys-instrument --quick -- ./app                              ║
║   rocprof-sys-run --quick -- ./app.inst                                ║
║                                                                        ║
║ WORKLOAD PRESETS                                                       ║
║   --quick           Fast profiling with sensible defaults              ║
║   --simple          Flat profile, minimal overhead                     ║
║   --detailed        Full trace + hardware counters                     ║
║   --trace-hpc       MPI/OpenMP/HPC applications                        ║
║   --trace-ai        PyTorch/TensorFlow/JAX                             ║
║                                                                        ║
║ PROFILING WORKFLOW                                                     ║
║   1. Sample    rocprof-sys-sample --quick -- ./app                     ║
║   2. Analyze   cat rocprof-sys-output/wall_clock.txt                   ║
║   3. Visualize Open rocprof-sys-output/perfetto-trace.proto            ║
║                in ui.perfetto.dev                                      ║
║                                                                        ║
║ COMMON OPTIONS                                                         ║
║   -f 100            Sample at 100Hz (rocprof-sys-sample)               ║
║   -o ./results      Custom output directory                            ║
║   --hip-trace       Enable GPU tracing                                 ║
║   -v, --verbose     Show detailed information                          ║
║                                                                        ║
║ INSTRUMENTATION                                                        ║
║   Binary Rewrite:                                                      ║
║     rocprof-sys-instrument -o app.inst -- ./app                        ║
║     rocprof-sys-run -- ./app.inst                                      ║
║                                                                        ║
║   Runtime:                                                             ║
║     rocprof-sys-instrument -- ./app                                    ║
║                                                                        ║
║ TIPS                                                                   ║
║   • Start with --quick for immediate insights                          ║
║   • Use --trace-hpc for compute-intensive codes                        ║
║   • Use --trace-ai for GPU-heavy ML workloads                          ║
║   • Check rocprof-sys-output/ for all results                          ║
║                                                                        ║
╚════════════════════════════════════════════════════════════════════════╝

📖 Full help: rocprof-sys-sample --help
🎓 Documentation: /opt/rocprofiler-systems/share/docs/
🌐 Online: https://rocm.docs.amd.com/projects/rocprofiler-systems/

)" << std::endl;
}

inline std::string
get_output_directory(const char* env_var = "ROCPROFSYS_OUTPUT_PATH")
{
    const char* output_path = getenv(env_var);
    if(output_path && strlen(output_path) > 0) return std::string(output_path);

    return "rocprof-sys-output";
}

inline bool
check_directory_writable(const std::string& dir)
{
    struct stat st;
    if(stat(dir.c_str(), &st) == 0)
    {
        return (access(dir.c_str(), W_OK) == 0);
    }

    std::string parent = dir;
    size_t      pos    = parent.find_last_of('/');
    if(pos != std::string::npos)
    {
        parent = parent.substr(0, pos);
        if(parent.empty()) parent = ".";
    }
    else
    {
        parent = ".";
    }

    return (access(parent.c_str(), W_OK) == 0);
}

inline void
print_pre_execution_info(const std::string& tool_name,
                         const std::string& preset_mode = "")
{
    auto output_dir = get_output_directory();

    if(!preset_mode.empty())
    {
        std::cout << "\n"
                  << "╔════════════════════════════════════════════════════════════╗\n"
                  << "║ ROCm Systems Profiler - " << tool_name
                  << std::string(28 - tool_name.size(), ' ') << "║\n"
                  << "╚════════════════════════════════════════════════════════════╝\n"
                  << "\n";

        std::cout << "Preset:        " << preset_mode << "\n";
    }

    std::cout << "Output:        " << output_dir << "\n";

    if(!check_directory_writable(output_dir))
    {
        std::cerr << "\n⚠️  WARNING: Output directory may not be writable!\n";
        std::cerr << "   Try: rocprof-sys-" << tool_name
                  << " -o /tmp/profile -- <command>\n\n";
    }

    std::cout << "\nResults will be available in:\n"
              << "  • Text profile:  " << output_dir << "/wall_clock.txt\n"
              << "  • Trace (visual): " << output_dir << "/perfetto-trace.proto\n"
              << "  • JSON data:      " << output_dir << "/wall_clock.json\n"
              << "\nTo visualize trace:\n"
              << "  Open " << output_dir
              << "/perfetto-trace.proto in https://ui.perfetto.dev\n"
              << "\n";
}

inline void
print_error_with_guidance(const std::string& error_msg, const std::string& tool_name)
{
    std::cerr << "\n❌ ERROR: " << error_msg << "\n\n";

    if(error_msg.find("output") != std::string::npos ||
       error_msg.find("write") != std::string::npos)
    {
        std::cerr << "Possible solutions:\n"
                  << "  1. Specify writable output: rocprof-sys-" << tool_name
                  << " -o /tmp/profile -- ./app\n"
                  << "  2. Check permissions: ls -ld ./\n"
                  << "  3. Set environment: export ROCPROFSYS_OUTPUT_PATH=/tmp/profile\n";
    }
    else if(error_msg.find("HIP") != std::string::npos ||
            error_msg.find("GPU") != std::string::npos ||
            error_msg.find("ROCm") != std::string::npos)
    {
        std::cerr << "GPU/ROCm troubleshooting:\n"
                  << "  1. Verify ROCm installation: hipconfig\n"
                  << "  2. Check devices: rocminfo\n"
                  << "  3. Ensure ROCm in PATH: which hipcc\n"
                  << "  4. Source environment: source /opt/rocm/setup.sh\n";
    }
    else if(error_msg.find("command") != std::string::npos ||
            error_msg.find("executable") != std::string::npos)
    {
        std::cerr << "Command troubleshooting:\n"
                  << "  1. Check executable exists: ls -l ./app\n"
                  << "  2. Verify it's executable: chmod +x ./app\n"
                  << "  3. Try absolute path: rocprof-sys-" << tool_name
                  << " -- $(pwd)/app\n";
    }
    else
    {
        std::cerr << "General troubleshooting:\n"
                  << "  1. Check help: rocprof-sys-" << tool_name << " --help\n"
                  << "  2. Enable verbose mode: rocprof-sys-" << tool_name
                  << " -v -- ./app\n"
                  << "  3. Try quick preset: rocprof-sys-" << tool_name
                  << " --quick -- ./app\n";
    }

    std::cerr
        << "\n📖 Documentation: /opt/rocprofiler-systems/share/docs/\n"
        << "🌐 Online help: https://rocm.docs.amd.com/projects/rocprofiler-systems/\n"
        << "\n";
}

inline int
validate_preset_modes(const std::vector<std::string>& active_presets)
{
    if(active_presets.size() > 1)
    {
        std::cerr << "\n❌ ERROR: Multiple preset modes specified: ";
        for(size_t i = 0; i < active_presets.size(); ++i)
        {
            std::cerr << active_presets[i];
            if(i < active_presets.size() - 1) std::cerr << ", ";
        }
        std::cerr << "\n\n";

        std::cerr << "Only ONE preset mode can be used at a time.\n\n";
        std::cerr << "Available presets:\n"
                  << "  --quick         Fast profiling with sensible defaults\n"
                  << "  --simple        Flat profile, minimal overhead\n"
                  << "  --detailed      Full trace + hardware counters\n"
                  << "  --trace-hpc     MPI/OpenMP/HPC applications\n"
                  << "  --trace-ai      PyTorch/TensorFlow/JAX\n\n";

        std::cerr
            << "Choose one preset or use manual options for custom configuration.\n";
        std::cerr << "See --help for all options.\n\n";

        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

inline void
run_interactive_wizard(const std::string& tool_name)
{
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       ROCm Systems Profiler Setup Wizard                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << "\n";
    std::cout << "This wizard will help you choose the right profiling options.\n\n";

    std::cout << "What type of application are you profiling?\n";
    std::cout << "  1. HIP/GPU application (ML, rendering, compute)\n";
    std::cout << "  2. HPC application (MPI, OpenMP, parallel compute)\n";
    std::cout << "  3. General CPU application\n";
    std::cout << "  4. Python application\n";
    std::cout << "\nYour choice [1-4]: ";
    std::cout.flush();

    int app_type = 0;
    std::cin >> app_type;

    if(app_type < 1 || app_type > 4)
    {
        std::cerr << "Invalid choice. Exiting wizard.\n";
        exit(EXIT_FAILURE);
    }

    std::cout << "\nDo you want detailed traces or quick profiling?\n";
    std::cout << "  1. Quick profile (faster, less overhead)\n";
    std::cout << "  2. Detailed trace (more data, slower)\n";
    std::cout << "\nYour choice [1-2]: ";
    std::cout.flush();

    int detail_level = 0;
    std::cin >> detail_level;

    if(detail_level < 1 || detail_level > 2)
    {
        std::cerr << "Invalid choice. Exiting wizard.\n";
        exit(EXIT_FAILURE);
    }

    std::string preset          = "--quick";
    std::string additional_opts = "";

    if(app_type == 1)
    {
        preset          = (detail_level == 1) ? "--trace-ai" : "--trace-ai";
        additional_opts = "--hip-trace";
    }
    else if(app_type == 2)
    {
        preset = (detail_level == 1) ? "--trace-hpc" : "--trace-hpc";
    }
    else if(app_type == 3)
    {
        preset = (detail_level == 1) ? "--quick" : "--detailed";
    }
    else if(app_type == 4)
    {
        preset = (detail_level == 1) ? "--trace-ai" : "--trace-ai";
    }

    std::cout << "\n✅ Configuration complete!\n\n";
    std::cout << "Recommended command:\n";
    std::cout << "  rocprof-sys-" << tool_name << " " << preset;
    if(!additional_opts.empty()) std::cout << " " << additional_opts;
    std::cout << " -- <your_command>\n\n";

    std::cout << "Examples:\n";
    if(app_type == 1)
    {
        std::cout << "  rocprof-sys-" << tool_name << " " << preset << " "
                  << additional_opts << " -- ./gpu_app\n";
    }
    else if(app_type == 2)
    {
        std::cout << "  mpirun -n 4 rocprof-sys-" << tool_name << " " << preset
                  << " -- ./mpi_app\n";
    }
    else if(app_type == 3)
    {
        std::cout << "  rocprof-sys-" << tool_name << " " << preset << " -- ./my_app\n";
    }
    else if(app_type == 4)
    {
        std::cout << "  rocprof-sys-" << tool_name << " " << preset
                  << " -- python script.py\n";
    }

    std::cout << "\nWould you like to:\n";
    std::cout << "  1. Run this command now\n";
    std::cout << "  2. Exit and run manually\n";
    std::cout << "\nYour choice [1-2]: ";
    std::cout.flush();

    int run_choice = 0;
    std::cin >> run_choice;

    if(run_choice == 1)
    {
        std::cout << "\nPlease run the command manually with your application.\n";
        std::cout << "The wizard cannot execute your command directly.\n\n";
    }

    std::cout << "💡 TIP: See all options with: rocprof-sys-" << tool_name << " --help\n";
    std::cout << "📖 Documentation: /opt/rocprofiler-systems/share/docs/\n\n";

    exit(EXIT_SUCCESS);
}

inline bool
check_rocm_available()
{
    return (system("which hipconfig > /dev/null 2>&1") == 0 ||
            access("/opt/rocm/bin/hipconfig", X_OK) == 0);
}

inline void
warn_if_hip_trace_without_rocm(bool hip_trace_requested, const std::string& tool_name)
{
    if(hip_trace_requested && !check_rocm_available())
    {
        std::cerr
            << "\n⚠️  WARNING: HIP tracing requested but ROCm may not be available\n\n";
        std::cerr << "Verify ROCm installation:\n";
        std::cerr << "  $ hipconfig\n";
        std::cerr << "  $ rocminfo\n\n";
        std::cerr << "If ROCm is installed, ensure it's in your PATH:\n";
        std::cerr << "  $ export PATH=/opt/rocm/bin:$PATH\n";
        std::cerr << "  $ export LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH\n\n";
        std::cerr << "Continuing without GPU tracing...\n\n";
    }
}

}  // namespace user_experience
}  // namespace rocprofsys
