/******************************************************************************
Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*******************************************************************************/

#include <dlfcn.h>
#include <link.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <unordered_set>
#include <fmt/core.h>
#include <fmt/color.h>

#include "src/utils/filesystem.hpp"

// filesystem for path resolutions
namespace fs = rocprofiler::common::filesystem;

namespace rocprofiler {
namespace src {
namespace tools {

// supported plugin list
std::vector<std::string> plugins{"ctf", "perfetto", "file", "att"};

// set rocm path and creates sym_link to /opt/rocm if not exists already
fs::path set_rocm_path() {
  // check symlink
  std::vector<std::string> rocm_paths;
  fs::path opt_path = "/opt";
  fs::path rocm_path = "/opt/rocm";

  // iterate and save all dirs under /opt that matches rocm
  for (const auto& entry : fs::directory_iterator(opt_path)) {
    if (entry.is_directory()) {
      std::string dirName = entry.path().filename().string();
      if (dirName.compare(0, 4, "rocm") == 0) {
        rocm_paths.push_back(entry.path());
      }
    }
  }

  // check if symlink already exists
  bool is_sym_link = false;
  for (const auto& rocm_dir : rocm_paths) {
    fs::path dir_path = opt_path / rocm_dir;
    if (fs::is_symlink(dir_path)) {
      is_sym_link = true;
    }
  }

  // create a symlink if not already exists
  if (!is_sym_link) {
    try {
      fs::create_symlink(rocm_paths[rocm_paths.size() - 1], "/opt/rocm");
      std::cout << "symbolic link created successfully." << std::endl;
    } catch (const fs::filesystem_error& e) {
      std::cerr << "error creating symbolic link: " << e.what() << std::endl;
    }
  }
  return rocm_path;
}

// usage message
void print_usage(fs::path current_path) {
  fmt::print("Usage: rocprofv2 [options] [target]\n");
  fmt::print("Options:\n");

  // help message
  fmt::print(fg(fmt::color::cyan), " -h   | --help\t\t");
  fmt::print("For showing this message\n");

  // list counters
  fmt::print(fg(fmt::color::cyan), " --list-counters\t");
  fmt::print("For showing all available counters for the current GPUs\n");

  // only show tests when running from build
  if (current_path.string().find("build") != std::string::npos) {
    fmt::print(fg(fmt::color::cyan), " -t   | --test\t\t");
    fmt::print("For Running the tests\n");
    fmt::print(fg(fmt::color::cyan), " -mt  | --mem-test \t");
    fmt::print("For Running the Memory Leak tests\n");
  }

  // HIP API traces
  fmt::print(fg(fmt::color::cyan), " --hip-api\t\t");
  fmt::print("For Collecting HIP API Traces\n");

  // HIP API and Activity traces
  fmt::print(fg(fmt::color::cyan), " --hip-activity | --hip-trace\n\t\t\t");
  fmt::print("For Collecting HIP API Activities Traces\n");

  // HSA API traces
  fmt::print(fg(fmt::color::cyan), " --hsa-api\t\t");
  fmt::print("For Collecting HSA API Traces\n");

  // HSA API and Activity traces
  fmt::print(fg(fmt::color::cyan), " --hsa-activity | --hsa-trace\n\t\t\t");
  fmt::print("For Collecting HSA API Activities Traces\n");

  // ROCtx traces
  fmt::print(fg(fmt::color::cyan), " --roctx-trace\t\t");
  fmt::print("For Collecting ROCTx Traces\n");

  // HSA API traces
  fmt::print(fg(fmt::color::cyan), " --kernel-trace\t\t");
  fmt::print("For Collecting Kernel dispatch Traces\n");

  // HSA API traces
  fmt::print(fg(fmt::color::cyan), " --sys-trace\t\t");
  fmt::print(
      "For Collecting HIP and HSA APIs and their Activities Traces along with ROCTX and Kernel "
      "Dispatch traces\n\t\t\t");

  fmt::print(fg(fmt::color::gray),
             "usage e.g: rocprofv2 --[hip-trace|hsa-trace|roctx-trace|kernel-trace|sys-trace] "
             "[target]\n");

  // plugins
  fmt::print(fg(fmt::color::cyan), " --plugin [PLUGIN_NAME]\n\t\t\t");
  fmt::print("For enabling a plugin (file/perfetto/att/ctf)\n\t\t\t");
  fmt::print(fg(fmt::color::gray),
             "usage(file/perfetto/ctf) e.g: rocprofv2 -i pmc.txt --plugin [file/perfetto/ctf] -d "
             "out-dir [target]\n\t\t\t");
  fmt::print(fg(fmt::color::gray),
             "usage(att): rocprofv2 [rocprofv2_params] --plugin att [ISA_file] [att_parameters] "
             "[target]\n\t\t\t");
  fmt::print(fg(fmt::color::gray),
             "use rocprofv2 --plugin att --help for ATT-specific parameters help\n");

  // input file
  fmt::print(fg(fmt::color::cyan), " -i   | --input\t\t");
  fmt::print(
      "For adding counters file path (every line in the text file represents a counter)\n\t\t\t");
  fmt::print(fg(fmt::color::gray), "usage: rocprofv2 -i pmc.txt -d [target]\n");

  // output file
  fmt::print(fg(fmt::color::cyan), " -o   | --output-file\t\t");
  fmt::print("For the output file name\n\t\t\t");
  fmt::print(fg(fmt::color::gray),
             "usage e.g:(with current dir): rocprofv2 --hip-trace -o <file_name> [target]\n\t\t\t");
  fmt::print(
      fg(fmt::color::gray),
      "usage e.g:(with custom dir):  rocprofv2 --hip-trace -d <out_dir> -o <file_name> [target]\n");

  // output directory
  fmt::print(fg(fmt::color::cyan), " -d   | --output-directory\n\t\t\t");
  fmt::print("For adding output path where the output files will be saved\n\t\t\t");
  fmt::print(fg(fmt::color::gray),
             "usage e.g:(with custom dir):  rocprofv2 --hip-trace -d <out_dir> [target]\n");

  // flush interval
  fmt::print(fg(fmt::color::cyan), " -fi  | --flush-interval\n\t\t\t");
  fmt::print(
      "For adding a flush interval in milliseconds, every flush interval the buffers will be "
      "flushed\n\t\t\t");
  fmt::print(fg(fmt::color::gray), "usage e.g : rocprofv2 --hip-trace - fi 1000  [target]\n");

  // trace period
  fmt::print(fg(fmt::color::cyan), " -tp  | --trace-period\n\t\t\t");
  fmt::print(
      "For specifing a trace period in milliseconds with -tp "
      "<DELAY>:<ACTIVE_TIME>:<LOOP_RESET_TIME>\n\t\t\t");
  fmt::print(fg(fmt::color::gray),
             "usage e.g:  rocprofv2 --hip-trace -tp 1000:2000:4000  [target]\n");

  // no serialization
  fmt::print(fg(fmt::color::cyan), " -ns  | --no-serialization\n\t\t\t");
  fmt::print(
      "For disabling serilization when running in counter-collection mode\n\t\t\t");
  fmt::print(fg(fmt::color::gray),
             "usage e.g:  rocprofv2 -i pmc.txt -ns\n");
}

// runs memory check on hip_vectoradd
void run_mem_check(fs::path bin_path) {
  fs::path app_path = bin_path;
  fs::path log_path = bin_path;
  fs::path asan_script_path = bin_path;

  asan_script_path =
      asan_script_path.string() + std::string("/tests-v2/memorytests/run_asan_tests.sh");
  app_path = app_path.string() + std::string("/tests-v2/featuretests/profiler/apps/hip_vectoradd");
  log_path = log_path.string() + std::string("memleaks.log");

  std::cout << "Log with all detected leaks is available at " << log_path.string() << std::endl;
  int status = system((asan_script_path.string() + " " + app_path.string()).c_str());
  if (status < 0) {
    std::cerr << "Invalid Command!" << std::endl;
  }
}

// passes counter
static std::atomic<uint32_t> pass_num{1};

// function to run the app for the amount of passes needed
void run_application(const char* app_path, char* const envp[], char* const args[]) {
  std::vector<char*> envp_vector;
    std::vector<char*> counter_passes;

    // each line of counter input file needs a single pass
    for(int i = 0; envp[i] != nullptr; ++i)
    {
        std::string envpString = envp[i];
        if(envpString.find("ROCPROFILER_COUNTERS") != std::string::npos)
        {
            counter_passes.push_back(const_cast<char*>(envp[i]));
        }
        else
        {
            envp_vector.push_back(const_cast<char*>(envp[i]));
        }
    }

    // no counter collection. hence need only 1 pass
    if(counter_passes.empty())
    {
        envp_vector[envp_vector.size()] = nullptr;

        int status = 0;
        if(fork() > 0)
        {
            status = execve(app_path, args, envp_vector.data());
            std::cout << "error: can't launch(" << app_path << "):" << std::endl;
            perror("errno");
            exit(EXIT_FAILURE);
        }
        envp_vector.pop_back();
        if(status == 0)
            std::cout << "pass(" << pass_num.fetch_add(1, std::memory_order_release)
                      << ") of the application: " << fs::path(app_path).filename()
                      << " is executed successfully!" << std::endl;
    }
    // counter collection. might need multipass, depends on input file pmc lines
    else
    {
        for(const auto& pass : counter_passes)
        {
            int status = 0;

            envp_vector.push_back(pass);
            envp_vector[envp_vector.size()] = nullptr;

            if(fork() > 0)
            {
                status = execve(app_path, args, envp_vector.data());
                std::cout << "error: can't launch(" << app_path << "):" << std::endl;
                perror("errno");
                exit(EXIT_FAILURE);
            }
            envp_vector.pop_back();
            if(status == 0)
                std::cout << "pass(" << pass_num.fetch_add(1, std::memory_order_release)
                          << ") of the application: " << fs::path(app_path).filename()
                          << " is executed successfully!" << std::endl;
        }
    }
    envp_vector.clear();
}

}  // namespace tools
}  // namespace src
}  // namespace rocprofiler

// creating a shorter alias
namespace rocprofv2 = rocprofiler::src::tools;

int main(int argc, char** argv) {
  // environment variables vector
  std::vector<std::string> pathenv;

  // reading arguments: one env var for the whole options!!
  int app_argc_start = 0;
  char* app_path = nullptr;

  // att params
  std::string att_argv;
  std::string att_pthon3_arg;
  fs::path att_py_path;
  std::string att_input_path;

  // getting current path
  char current_path[FILENAME_MAX];
  if (!getcwd(current_path, sizeof(current_path))) {
    throw errno;
  }
  current_path[sizeof(current_path) - 1] = '\0';

  // set rocm path (/opt/rocm)
  fs::path rocm_path = rocprofv2::set_rocm_path();

  // to show usage when no arguments are given
  if (argc <= 1) {
    rocprofv2::print_usage(current_path);
    return (EXIT_SUCCESS);
  }

  // providing LD_PRELOAD of rocprofiler library to run_application function
  std::string tool_lib_path = rocm_path.string() + "/lib/rocprofiler/librocprofiler_tool.so";
  std::string current_ld_preload;
  if (getenv("LD_PRELOAD")) current_ld_preload = getenv("LD_PRELOAD");
  if (!current_ld_preload.empty())
    tool_lib_path = "LD_PRELOAD=" + current_ld_preload + ":" + tool_lib_path;
  else
    tool_lib_path = "LD_PRELOAD=" + current_ld_preload + tool_lib_path;
  pathenv.emplace_back(tool_lib_path);

  // iterate through all options
  for (int i = 1; i < argc; i++) {
    // help message
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      rocprofv2::print_usage(current_path);
      return (EXIT_SUCCESS);
    }
    // list counters
    if (strcmp(argv[i], "--list-counters") == 0) {
      fs::path ctrl_path = rocm_path.string() + "/libexec/rocprofiler/ctrl";
      std::string iterate_counters = tool_lib_path + " " + ctrl_path.string();
      int status = system(iterate_counters.c_str());
      if (status < 0) {
        std::cerr << "Invalid Command!" << std::endl;
        return (EXIT_FAILURE);
      }
      return (EXIT_SUCCESS);
    }
    // rocprofiler tests
    else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
      fs::path test_path = current_path + std::string("/run_tests.sh");
      int status = system(test_path.string().c_str());
      if (status < 0) {
        std::cerr << "Invalid Command!" << std::endl;
        return (EXIT_FAILURE);
      }
      return (EXIT_SUCCESS);
    }
    // memory check tests
    else if (strcmp(argv[i], "-mt") == 0 || strcmp(argv[i], "--mem-test") == 0) {
      rocprofv2::run_mem_check(current_path);
      return (EXIT_SUCCESS);
    }
    // counters file
    else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
      i++;
      if (argv[i]) {
        std::string counters_path;
        if (!fs::path(argv[i]).is_absolute()) {
          counters_path = std::string(current_path) + "/" + argv[i];
        } else {
          counters_path = argv[i];
        }
        std::cout << "reading counters from: " << counters_path << std::endl;
        pathenv.emplace_back("COUNTERS_PATH=" + counters_path);

        // setting att input path for later
        att_input_path = counters_path;

        std::ifstream inputfile(counters_path);
        std::string pmc_line;
        while (std::getline(inputfile, pmc_line)) {
          // std::cout << "ROCPROFILER_COUNTERS=" + pmc_line << std::endl;
          pathenv.emplace_back("ROCPROFILER_COUNTERS=" + pmc_line);
        }

        // close the file.
        inputfile.close();


      } else {
        std::cerr << "Error: Missing Counters File path!" << std::endl;
        rocprofv2::print_usage(current_path);
        exit(EXIT_FAILURE);
      }

    }  // HIP API trace
    else if (strcmp(argv[i], "--hip-api") == 0) {
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_HIP_API_TRACE=1");
      }
      // HIP trace
    } else if (strcmp(argv[i], "--hip-trace") == 0 || strcmp(argv[i], "--hip-activity") == 0) {
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_HIP_API_TRACE=1");
        pathenv.emplace_back("ROCPROFILER_HIP_ACTIVITY_TRACE=1");
      }
      // HSA API trace
    } else if (strcmp(argv[i], "--hsa-api") == 0) {
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_HSA_API_TRACE=1");
      }
      // HSA trace
    } else if (strcmp(argv[i], "--hsa-trace") == 0 || strcmp(argv[i], "--hsa-activity") == 0) {
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_HSA_API_TRACE=1");
        pathenv.emplace_back("ROCPROFILER_HSA_ACTIVITY_TRACE=1");
      }
      // ROCTx trace
    } else if (strcmp(argv[i], "--roctx-trace") == 0) {
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_ROCTX_TRACE=1");
      }
      // kernel trace
    } else if (strcmp(argv[i], "--kernel-trace") == 0) {
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_KERNEL_TRACE=1");
      }
      // sys trace
    } else if (strcmp(argv[i], "--sys-trace") == 0) {
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_HIP_API_TRACE=1");
        pathenv.emplace_back("ROCPROFILER_HIP_ACTIVITY_TRACE=1");
        pathenv.emplace_back("ROCPROFILER_HSA_API_TRACE=1");
        pathenv.emplace_back("ROCPROFILER_HSA_ACTIVITY_TRACE=1");
        pathenv.emplace_back("ROCPROFILER_ROCTX_TRACE=1");
        pathenv.emplace_back("ROCPROFILER_KERNEL_TRACE=1");
      }
      // plugins
    } else if (strcmp(argv[i], "--plugin") == 0) {
      i++;
      auto it =
          std::find(rocprofv2::plugins.begin(), rocprofv2::plugins.end(), std::string(argv[i]));
      if (it == rocprofv2::plugins.end()) {
        rocprofv2::print_usage(current_path);
        exit(EXIT_FAILURE);
      }
      pathenv.emplace_back("ROCPROFILER_PLUGIN_LIB=lib" + std::string(argv[i]) + "_plugin.so");
      if (std::string(argv[i]) == "att") {
        att_py_path = rocm_path.string() + std::string("/libexec/rocprofiler/att/att.py");
        pathenv.emplace_back("ROCPROFV2_ATT_LIB_PATH=" + rocm_path.string() +
                             "/lib/hsa-amd-aqlprofile/librocprofv2_att.so");
        i++;
        att_argv = argv[3];
        att_pthon3_arg = "python3";
        i++;
        att_argv = "";
        for (int i = 4; i < argc; i++) {
          if (std::string(argv[i]) == "--trace-file") {
            att_argv = att_argv + " " + argv[i] + " " + argv[i + 1];
            i = i + 1;
          } else if (std::string(argv[i]) == "--mpi") {
            att_pthon3_arg = "mpirun -np " + std::string(argv[i]) + " python3";
            i = i + 1;
          } else if (std::string(argv[i]) == "--mode" || std::string(argv[i]) == "--ports" ||
                     std::string(argv[i]) == "--genasm" || std::string(argv[i]) == "--att_kernel" ||
                     std::string(argv[i]) == "--depth") {
            att_argv = att_argv + " " + argv[i] + " " + argv[i + 1] + " " + argv[i - 1];
            i = i + 1;
          } else {
            continue;
          }
        }  // for loop
      }
      // output directory
    } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--output-directory") == 0) {
      i++;
      if (argv[i]) {
        pathenv.emplace_back("OUTPUT_PATH_INTERNAL=" + std::string(argv[i]));
        pathenv.emplace_back("OUTPUT_PATH=" + std::string(argv[i]));
      } else {
        std::cerr << "Error: Missing output directory path!" << std::endl;
        rocprofv2::print_usage(current_path);
        exit(EXIT_FAILURE);
      }
    }  // output file name
    else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output-file-name") == 0) {
      i++;
      if (argv[i]) {
        pathenv.emplace_back("OUT_FILE_NAME=" + std::string(argv[i]));
      } else {
        std::cerr << "Error: Missing output file name!" << std::endl;
        rocprofv2::print_usage(current_path);
        exit(EXIT_FAILURE);
      }
      // flush interval
    } else if (strcmp(argv[i], "-fi") == 0 || strcmp(argv[i], "--flush-interval") == 0) {
      i++;
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_FLUSH_INTERVAL=" + std::string(argv[i]));
      } else {
        std::cerr << "Error: Missing flush interval value!" << std::endl;
        rocprofv2::print_usage(current_path);
        exit(EXIT_FAILURE);
      }
      // trace period
    } else if (strcmp(argv[i], "-tp") == 0 || strcmp(argv[i], "--trace-period") == 0) {
      i++;
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_TRACE_PERIOD=" + std::string(argv[i]));
      } else {
        std::cerr << "Error: Missing trace period value!" << std::endl;
        rocprofv2::print_usage(current_path);
        exit(EXIT_FAILURE);
      }
      // no serialization for counter-collection mode
    } else if (strcmp(argv[i], "-ns") == 0 || strcmp(argv[i], "--no-serialization") == 0) {
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_NO_SERIALIZATION=1");
      }
      // wrong argument given
    } else if (argv[i][0] == '-') {
      std::cerr << "Wrong option (" << argv[i] << "), Please use the following options:\n"
                << std::endl;
      rocprofv2::print_usage(current_path);
      exit(EXIT_FAILURE);
      // taking up the Application path with its arguments
    } else {
      app_path = argv[i];
      app_argc_start = i;
      break;
    }
  }

  // getting original application arguments
  char* app_args[argc - app_argc_start + 1];
  int j = 0;
  for (int i = app_argc_start; i < argc; i++) {
    app_args[j] = argv[i];
    j++;
  }
  app_args[j] = NULL;

  // providing all environment variables needed by the arguments
  char* envp_run[pathenv.size() + 1];
  for (uint32_t i = 0; i < pathenv.size(); i++)
    envp_run[i] = const_cast<char*>(pathenv.at(i).c_str());
  envp_run[pathenv.size()] = NULL;

  // getting application executable path to be provided to run_application function
  std::string app_path_str;
  if (!fs::path(app_path).is_absolute()) {
    app_path_str = std::string(current_path) + "/" + std::string(app_path);
  } else {
    app_path_str = app_path;
  }

  // providing metrics path
  std::string metrics_path_str =
      rocm_path.string() + "/libexec/rocprofiler/counters/derived_counters.xml";
  pathenv.emplace_back("ROCPROFILER_METRICS_PATH=" + metrics_path_str);

  // ATT: check if the att required environment variables are set.
  if (!att_py_path.empty()) {
    if (!att_argv.empty()) {
      std::string command = att_pthon3_arg + " " + att_py_path.string() + " " + att_argv;
      setenv("COUNTERS_PATH", att_input_path.c_str(), 1);
      int status = system(command.c_str());
      if (status < 0) {
        std::cerr << "Invalid Command!" << std::endl;
        return (EXIT_FAILURE);
      }
    }
  }
  // calling run_application function to execute the application with environment variables and
  // original arguments
  rocprofv2::run_application(app_path_str.c_str(), envp_run, app_args);

  return 1;
}
