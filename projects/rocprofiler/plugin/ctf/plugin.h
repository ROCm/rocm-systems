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

#ifndef PLUGIN_CTF_PLUGIN_H
#define PLUGIN_CTF_PLUGIN_H

#include <mutex>
#include <cstdlib>

#include "src/utils/filesystem.hpp"

#include "rocprofiler.h"
#include "rocprofiler_plugin.h"

#include "barectf.h"
#include "barectf_tracer.h"

namespace rocm_ctf {

// CTF plugin.
//
// Build a plugin instance, and then call HandleTracerRecord(),
// HandleProfilerRecord(), and HandleBufferRecords() to add event
// records.
//
// A plugin instance performs important tasks at destruction time.
class Plugin final {
 public:
  // Builds a plugin instance to write a CTF trace in the `trace_dir`
  // directory with packets of size `packet_size` bytes.
  //
  // `trace_dir` must not exist.
  //
  // This constructor immediately adjusts and copies the metadata stream
  // file `metadata_stream_path` to the trace directory (`trace_dir`).
  explicit Plugin(std::size_t packet_size, const rocprofiler::common::filesystem::path& trace_dir,
                  const rocprofiler::common::filesystem::path& metadata_stream_path);

  // Handles a tracer record.
  void HandleTracerRecord(const rocprofiler_record_tracer_t& record,
                          rocprofiler_session_id_t session_id);


  // Handles a profiler record.
  void HandleProfilerRecord(const rocprofiler_record_profiler_t& record,
                            rocprofiler_session_id_t session_id);

  // Handles tracer or profiler records from `begin` to `end`
  // (excluded).
  void HandleBufferRecords(const rocprofiler_record_header_t* begin,
                           const rocprofiler_record_header_t* end,
                           rocprofiler_session_id_t session_id, rocprofiler_buffer_id_t buffer_id);

 private:
  // rocTX barectf platform descriptor.
  struct RocTxPlatformDescr final {
    using Ctx = barectf_roctx_ctx;

    static void OpenPacket(Ctx& ctx) { barectf_roctx_open_packet(&ctx); }
    static void ClosePacket(Ctx& ctx) { barectf_roctx_close_packet(&ctx); }
  };

  // HSA API barectf platform descriptor.
  struct HsaApiPlatformDescr final {
    using Ctx = barectf_hsa_api_ctx;

    static void OpenPacket(Ctx& ctx) { barectf_hsa_api_open_packet(&ctx); }
    static void ClosePacket(Ctx& ctx) { barectf_hsa_api_close_packet(&ctx); }
  };

  // HIP API barectf platform descriptor.
  struct HipApiPlatformDescr final {
    using Ctx = barectf_hip_api_ctx;

    static void OpenPacket(Ctx& ctx) { barectf_hip_api_open_packet(&ctx); }
    static void ClosePacket(Ctx& ctx) { barectf_hip_api_close_packet(&ctx); }
  };

  // HSA handles barectf platform descriptor.
  struct HsaHandlesPlatformDescr final {
    using Ctx = barectf_hsa_handles_ctx;

    static void OpenPacket(Ctx& ctx) { barectf_hsa_handles_open_packet(&ctx); }
    static void ClosePacket(Ctx& ctx) { barectf_hsa_handles_close_packet(&ctx); }
  };

  // API operations barectf platform descriptor.
  struct ApiOpsPlatformDescr final {
    using Ctx = barectf_api_ops_ctx;

    static void OpenPacket(Ctx& ctx) { barectf_api_ops_open_packet(&ctx); }
    static void ClosePacket(Ctx& ctx) { barectf_api_ops_close_packet(&ctx); }
  };

  // Profiler barectf platform descriptor.
  struct ProfilerPlatformDescr final {
    using Ctx = barectf_profiler_ctx;

    static void OpenPacket(Ctx& ctx) { barectf_profiler_open_packet(&ctx); }
    static void ClosePacket(Ctx& ctx) { barectf_profiler_close_packet(&ctx); }
  };

  // barectf tracer for HSA handle mappings.
  using HsaHandlesTracer = BarectfTracer<HsaHandlesPlatformDescr>;

  // Writes the HSA handle type mappings to a dedicated data stream
  // file.
  void WriteHsaHandleTypes();

  // Loads the existing metadata stream file `metadata_stream_path`,
  // adjusts the `offset` property of its single clock class, and writes
  // the result to the `metadata` file within the `trace_dir` directory.
  void CopyAdjustedMetadataStreamFile(
      const rocprofiler::common::filesystem::path& metadata_stream_path,
      const rocprofiler::common::filesystem::path& trace_dir);

  // Dedicated tracers.
  BarectfTracer<RocTxPlatformDescr> roctx_tracer_;
  BarectfTracer<HsaApiPlatformDescr> hsa_api_tracer_;
  BarectfTracer<HipApiPlatformDescr> hip_api_tracer_;
  BarectfTracer<ApiOpsPlatformDescr> api_ops_tracer_;
  HsaHandlesTracer hsa_handles_tracer_;
  BarectfTracer<ProfilerPlatformDescr> profiler_tracer_;

  // Locks any operation performed on the data of this.
  std::mutex lock_;
};

}  // namespace rocm_ctf

#endif  // PLUGIN_CTF_PLUGIN_H
