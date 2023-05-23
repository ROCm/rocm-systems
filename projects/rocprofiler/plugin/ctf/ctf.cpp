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

#include <cassert>
#include <stdexcept>
#include <iostream>
#include <experimental/filesystem>

#include "rocprofiler.h"
#include "rocprofiler_plugin.h"

#include "plugin.h"

namespace fs = std::experimental::filesystem;

namespace {

// Global plugin instance
rocm_ctf::Plugin* the_plugin = nullptr;

}  // namespace

ROCPROFILER_EXPORT int rocprofiler_plugin_initialize(const uint32_t rocprofiler_major_version,
                                                     const uint32_t rocprofiler_minor_version) {
  if (rocprofiler_major_version != ROCPROFILER_VERSION_MAJOR ||
      rocprofiler_minor_version < ROCPROFILER_VERSION_MINOR) {
    return -1;
  }

  if (the_plugin) {
    return -1;
  }

  const auto output_dir = getenv("OUTPUT_PATH");

  if (!output_dir) {
    std::cerr << "rocprofiler_plugin_initialize(): "
              << "`OUTPUT_PATH` environment variable isn't set" << std::endl;
    return -1;
  }

  // Create the plugin instance.
  try {
    the_plugin = new rocm_ctf::Plugin{256 * 1024, fs::path{output_dir} / "trace",
                                      CTF_PLUGIN_METADATA_FILE_PATH};
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

ROCPROFILER_EXPORT int rocprofiler_plugin_write_record(
    const rocprofiler_record_tracer_t record, rocprofiler_plugin_tracer_extra_data_t tracer_data) {
  assert(the_plugin);

  if (record.header.id.handle == 0) {
    return 0;
  }

  try {
    the_plugin->HandleTracerRecord(record, rocprofiler_session_id_t{0}, tracer_data);
  } catch (const std::exception& exc) {
    std::cerr << "rocprofiler_plugin_write_record(): " << exc.what() << std::endl;
    return -1;
  }

  return 0;
}
