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
#include <cassert>
#include <stdexcept>
#include <iostream>
#include <string>
#include <thread>
#include <link.h>
#include <chrono>
#include <regex>
#include <unistd.h>
#include "src/utils/filesystem.hpp"
#include <type_traits>

#include "rocprofiler.h"
#include "rocprofiler_plugin.h"

#include "plugin.h"
#ifdef USE_GET_ROCM_PATH_API
#include <rocm-core/rocm_getpath.h>
#endif

namespace fs = rocprofiler::common::filesystem;

namespace {

// Global plugin instance
rocm_ctf::Plugin* the_plugin = nullptr;

}  // namespace

ROCPROFILER_EXPORT int rocprofiler_plugin_initialize(const uint32_t rocprofiler_major_version,
                                                     const uint32_t rocprofiler_minor_version,
                                                     void* data) {
  if (rocprofiler_major_version != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_minor_version < ROCPROFILER_VERSION_MINOR) {
    return -1;
  }

  if (the_plugin) {
    return -1;
  }

  auto output_dir = []() -> std::string {
    if (const char* output_dir_internal = getenv("OUTPUT_PATH"); output_dir_internal != nullptr) {
      return output_dir_internal;
    }
    return "./";
  }();

  auto output_file = []() -> std::string {
    auto _v = getenv("OUTPUT_FILE");
    return (_v) ? _v : "trace-{PID}";
  }();

  auto _replace = [&output_dir, &output_file](const char* _key, auto _value) {
    using value_type = std::remove_cv_t<std::remove_reference_t<std::decay_t<decltype(_value)>>>;
    auto _value_str = std::to_string(_value);

    const auto _re = std::regex{_key, std::regex_constants::icase};
    output_dir = std::regex_replace(output_dir, _re, _value_str);
    output_file = std::regex_replace(output_file, _re, _value_str);
  };

  _replace("\\{PID\\}", getpid());
  _replace("\\$ENV\\{PID\\}", getpid());
  _replace("\\{PPID\\}", getppid());
  _replace("\\$ENV\\{PPID\\}", getppid());

  // Create the plugin instance.
  #ifdef USE_GET_ROCM_PATH_API
  char *installPath = nullptr;
  unsigned int installPathLen = 0;
  PathErrors_t retVal = PathSuccess;
  auto metadata_path = std::string{CTF_PLUGIN_METADATA_FILE_PATH};
  // Get ROCM install path
  retVal = getROCmInstallPath( &installPath, &installPathLen );
  if(PathSuccess == retVal){
      metadata_path = fs::path(installPath) / fs::path{CTF_PLUGIN_METADATA_FILE_PATH};
  }else {
      std::cout << "Failed to get ROCm Install Path: " << retVal << std::endl;
  }
  // free allocated memory
  if(nullptr != installPath) {
      free(installPath);
  }
  #else
  auto* this_plugin_handle = dlopen("libctf_plugin.so", RTLD_LAZY | RTLD_NOLOAD);
  auto* librocprofiler_handle = dlopen("librocprofiler64.so", RTLD_LAZY | RTLD_NOLOAD);
  auto metadata_path = std::string{CTF_PLUGIN_METADATA_FILE_PATH};
  struct link_map* _link_map = nullptr;
  if (this_plugin_handle && dlinfo(this_plugin_handle, RTLD_DI_LINKMAP, &_link_map) == 0) {
    metadata_path = fs::path{_link_map->l_name}.parent_path() / fs::path{"../.."} /
        CTF_PLUGIN_METADATA_FILE_PATH;
  } else if (librocprofiler_handle &&
             dlinfo(librocprofiler_handle, RTLD_DI_LINKMAP, &_link_map) == 0) {
    metadata_path =
        fs::path{_link_map->l_name}.parent_path() / ".." / CTF_PLUGIN_METADATA_FILE_PATH;
  }

  if (!fs::exists(metadata_path)) {
    metadata_path = fs::path{CTF_PLUGIN_INSTALL_PREFIX} / CTF_PLUGIN_METADATA_FILE_PATH;
  }
  #endif // USE_GET_ROCM_PATH_API

  try {
    the_plugin = new rocm_ctf::Plugin{256 * 1024, fs::path{output_dir} / output_file,
                                      fs::absolute(metadata_path)};
  } catch (const std::exception& exc) {
    std::cerr << "rocprofiler_plugin_initialize(): " << exc.what() << std::endl;
    return -1;
  }

  return 0;
}

ROCPROFILER_EXPORT void rocprofiler_plugin_finalize() {
  delete the_plugin;
  the_plugin = nullptr;
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_buffer_records(
    const rocprofiler_record_header_t* const begin, const rocprofiler_record_header_t* const end,
    const rocprofiler_session_id_t session_id, const rocprofiler_buffer_id_t buffer_id) {
  assert(the_plugin);

  try {
    the_plugin->HandleBufferRecords(begin, end, session_id, buffer_id);
  } catch (const std::exception& exc) {
    std::cerr << "rocprofiler_plugin_write_buffer_records(): " << exc.what() << std::endl;
    return -1;
  }

  return 0;
}

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(const rocprofiler_record_tracer_t record) {
  assert(the_plugin);

  if (record.header.id.handle == 0) {
    return 0;
  }

  try {
    the_plugin->HandleTracerRecord(record, rocprofiler_session_id_t{0});
  } catch (const std::exception& exc) {
    std::cerr << "rocprofiler_plugin_write_record(): " << exc.what() << std::endl;
    return -1;
  }

  return 0;
}
