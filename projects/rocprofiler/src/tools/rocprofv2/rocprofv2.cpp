/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 THE SOFTWARE. */

#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <experimental/filesystem>
#include <iostream>
#include <vector>

namespace fs = std::experimental::filesystem;

// Usage message
void printUsage() {
  std::cout << "ROCMTools Run Binary Usage:" << std::endl;
  std::cout << "\nTo run ./run.sh PARAMs, PARAMs can be the following:" << std::endl;
  std::cout << "-h   | --help               For showing this message" << std::endl;
  std::cout << "-t   | --test               For Running the tests" << std::endl;
  std::cout << "-mt  | --mem-test           For Running the Memory Leak tests" << std::endl;
  std::cout << "\nTo run ./run.sh PARAMs APP_EXEC, PARAMs can be the following:" << std::endl;
  std::cout << "-i   | --input              For adding counters file path "
               "(every line in the text file represents a counter)"
            << std::endl;
  std::cout << "-d   | --output-directory   For adding output path where the "
               "output files will be saved"
            << std::endl;
  std::cout << "-fi  | --flush-interval     For adding a flush interval in "
               "milliseconds, every \"flush interval\" the buffers will be flushed"
            << std::endl;
  std::cout << "-a   | --asan               For adding libasan.so.6 for memory leak "
               "check run requires building using -act | --asan-clean-build option"
            << std::endl;
}

void runMemCheck(fs::path bin_path) {
  fs::path counter_path = bin_path;
  fs::path app_path = bin_path;
  fs::path log_path = bin_path;
  fs::path suppr_path = bin_path;
  fs::path pyscript_path = bin_path;

  std::string pathenv = bin_path.replace_filename("librocprofiler_tool.so");
  pyscript_path = pyscript_path.replace_filename("tests/memorytests/test_mem.py");
  counter_path = counter_path.replace_filename("tests/memorytests/input.txt");
  suppr_path = suppr_path.replace_filename("tests/memorytests/suppr.txt");
  app_path = app_path.replace_filename(
      "tests/featuretests/profiler/gtests/apps/"
      "hip_vectoradd");
  log_path = log_path.replace_filename("memleaks.log");

  std::cout << "Running Memory Leaks Check...." << std::endl;
  std::string run_command = "LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libasan.so.6:" + pathenv +
      " ASAN_OPTIONS=detect_leaks=1 LSAN_OPTIONS=suppressions=" + suppr_path.string() +
      " COUNTERS_PATH=" + counter_path.string() + " " + app_path.string() +
      " > /tmp/rocprofv2-temp 2> " + log_path.string();
  int status = system(run_command.c_str());
  if (status < 0) std::cerr << "Invalid Command!" << std::endl;

  std::cout << "Log with all detected leaks is available at " << log_path.string() << std::endl;
  std::string python_command = "python3 " + pyscript_path.string() + " " + log_path.string();
  status = system(python_command.c_str());
  if (status < 0) std::cerr << "Invalid Command!" << std::endl;
}

// Passes Counter
static std::atomic<uint32_t> pass_num{1};

// Getting the number of passes needed using the counters file path
uint32_t getNumberOfPasses(std::string counter_path) {
  // TODO(aelwazir): Adding a way to get the number of passes;
  return 1;
}

// Function to run the app for the amount of passes needed
void runApp(const char* app_path, char* const envp[], char* const args[],
            uint32_t number_of_passes = 1) {
  while (pass_num.load(std::memory_order_relaxed) <= number_of_passes) {
    // Forking to keep the original
    int status = 0;
    if (fork() > 0) {
      status = execve(app_path, args, envp);
      std::cout << "Error: can't launch(" << app_path << "):" << std::endl;
      perror("errno");
      exit(EXIT_FAILURE);
    }
    if (status == 0)
      std::cout << "Pass(" << pass_num.fetch_add(1, std::memory_order_release)
                << ") of the application: " << fs::path(app_path).filename()
                << " is executed successfully!" << std::endl;
  }
}

uint32_t number_of_passes = 1;

int main(int argc, char** argv) {
  // Getting Current Path
  char current_path[FILENAME_MAX];
  if (!getcwd(current_path, sizeof(current_path))) {
    throw errno;
  }
  current_path[sizeof(current_path) - 1] = '\0';

  // Getting the rocprofv2 binary path to locate rocmtools library path
  fs::path bin_path;
  if (Dl_info dl_info; dladdr((void*)runApp, &dl_info) != 0) {
    bin_path = fs::path(dl_info.dli_fname);
  }

  // Environment cariables vector
  std::vector<std::string> pathenv;

  // Reading Arguments
  // One ENV Var for the whole options!!
  int app_argc_start = 0;
  char* app_path = nullptr;
  for (int i = 1; i < argc; i++) {
    // Help Message
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "-h") == 0) {
      printUsage();
      return 1;
      // Normal ROCMTools Tests
    } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--test") == 0) {
      fs::path test_path = bin_path;
      test_path = test_path.replace_filename("run_tests.sh");
      int status = system(test_path.string().c_str());
      if (status < 0) std::cerr << "Invalid Command!" << std::endl;
      return 1;
      // Memory Check Test
    } else if (strcmp(argv[i], "-mt") == 0 || strcmp(argv[i], "--mem-test") == 0) {
      runMemCheck(bin_path);
      return 1;
      // Counters File
    } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--input") == 0) {
      i++;
      if (argv[i]) {
        std::string counters_path = std::string(current_path) + "/" + argv[2];
        std::cout << "Reading counters from: " << counters_path << std::endl;
        number_of_passes = getNumberOfPasses(counters_path);
        pathenv.emplace_back("COUNTERS_PATH=" + counters_path);
      } else {
        std::cerr << "Error: Missing Counters File path!" << std::endl;
        printUsage();
        exit(EXIT_FAILURE);
      }
      // Output Directory
    } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--output-directory") == 0) {
      i++;
      if (argv[i]) {
        std::string output_path = std::string(current_path) + "/" + argv[2];
        fs::create_directory(output_path);
        std::cout << "Output directory path: " << output_path << std::endl;
        pathenv.emplace_back("OUTPUT_PATH=" + output_path);
      } else {
        std::cerr << "Error: Missing output directory path!" << std::endl;
        printUsage();
        exit(EXIT_FAILURE);
      }
      // ASAN run
    } else if (strcmp(argv[i], "-a") == 0 || strcmp(argv[i], "--asan") == 0) {
      std::string current_ld_preload{getenv("LD_PRELOAD")};
      std::string ld_preload = "/usr/lib/x86_64-linux-gnu/libasan.so.6";
      if (!current_ld_preload.empty()) ld_preload = current_ld_preload + ":" + ld_preload;

      std::cout << ld_preload << std::endl;
      pathenv.emplace_back("LD_PRELOAD=" + ld_preload);
      pathenv.emplace_back("ASAN_OPTIONS=detect_leaks=1");

      fs::path suppr_path = bin_path;
      std::string suppr =
          "suppressions=" + suppr_path.replace_filename("tests/memorytests/suppr.txt").string();
      std::cout << suppr << std::endl;
      pathenv.emplace_back("LSAN_OPTIONS=" + suppr);
      // Flush Interval
    } else if (strcmp(argv[i], "-fi") == 0 || strcmp(argv[i], "--flush-interval") == 0) {
      i++;
      if (argv[i]) {
        pathenv.emplace_back("ROCPROFILER_FLUSH_INTERVAL=" + std::string(argv[i]));
      } else {
        std::cerr << "Error: Missing flush interval value!" << std::endl;
        printUsage();
        exit(EXIT_FAILURE);
      }
      // Wrong argument given
    } else if (argv[i][0] == '-') {
      std::cerr << "Wrong option (" << argv[i] << "), Please use the following options:\n"
                << std::endl;
      printUsage();
      exit(EXIT_FAILURE);
      // Taking up the Application path with its arguments
    } else {
      app_path = argv[i];
      app_argc_start = i;
      break;
    }
  }

  // Getting Original Application Arguments
  char* app_args[argc - app_argc_start + 1];
  int j = 0;
  for (int i = app_argc_start; i < argc; i++) {
    app_args[j] = argv[i];
    j++;
  }
  app_args[j] = NULL;

  // Providing LD_PRELOAD of ROCMTools Library to runApp function
  std::string pathenv_str = "LD_PRELOAD=librocprofiler_tool.so";
  std::string current_ld_preload;
  if (getenv("LD_PRELOAD")) current_ld_preload = getenv("LD_PRELOAD");
  if (strstr(bin_path.c_str(), "build") != nullptr)
    pathenv_str = bin_path.replace_filename("librocprofiler_tool.so");
  else
    pathenv_str = bin_path.remove_filename().replace_filename("lib/librocprofiler_tool.so");
  if (!current_ld_preload.empty())
    pathenv_str = "LD_PRELOAD=" + current_ld_preload + ":" + pathenv_str;
  else
    pathenv_str = "LD_PRELOAD=" + current_ld_preload + pathenv_str;
  pathenv.emplace_back(pathenv_str);

  // Providing all Environment variables needed by the arguments
  char* envp_run[pathenv.size() + 1];
  for (uint32_t i = 0; i < pathenv.size(); i++)
    envp_run[i] = const_cast<char*>(pathenv.at(i).c_str());
  envp_run[pathenv.size()] = NULL;

  // Getting Application Executable path to be provided to runApp function
  std::string app_path_str;
  if (app_path) app_path_str = std::string(current_path) + "/" + std::string(app_path);

  // Calling runApp function to execute the application with Environment
  // variables and original arguments
  runApp(app_path_str.c_str(), envp_run, app_args, number_of_passes);
  return 1;
}