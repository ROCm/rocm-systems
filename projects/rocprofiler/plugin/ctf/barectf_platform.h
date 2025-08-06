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

#ifndef PLUGIN_CTF_BARECTF_PLATFORM_H
#define PLUGIN_CTF_BARECTF_PLATFORM_H

#include <cstdlib>
#include <cstdint>
#include <fstream>
#include <vector>
#include <functional>
#include "src/utils/filesystem.hpp"

#include "barectf.h"

namespace rocm_ctf {

template <typename> class BarectfWriter;

// A barectf platform for any barectf writer.
//
// The user doesn't deal directly with such an object: it's closely
// coupled with a barectf writer.
//
// Each platform takes care of a single CTF data stream file.
//
// After building such a platform, get the raw barectf context with
// GetCtx() to call tracing functions. The platform must still exist
// when calling a tracing function.
//
// Such a platform opens the data stream file on construction and closes
// it on destruction.
//
// `DescrT` is the specific barectf platform descriptor. It must be a
// structure having:
//
// `Ctx`:
//     Specific barectf context type.
//
// `static void OpenPacket(Ctx&)`:
//     Packet opening function.
//
// `static void ClosePacket(Ctx&)`:
//     Packet closing function.
template <typename DescrT> class BarectfPlatform final {
  friend class BarectfWriter<DescrT>;

 private:
  // Builds a barectf platform.
  //
  // The platform writes CTF packets of size `packet_size` bytes to the
  // CTF data stream file `data_stream_file_path`.
  //
  // For each event record to write, the platform reads `clock_val` to
  // know the current timestamp.
  explicit BarectfPlatform(const std::size_t packet_size,
                           const rocprofiler::common::filesystem::path& data_stream_file_path,
                           const std::uint64_t& clock_val)
      : clock_val_{&clock_val}, buffer_(packet_size) {
    // Initialize barectf callbacks.
    barectf_platform_callbacks callbacks;

    callbacks.default_clock_get_value = GetClockCb;
    callbacks.is_backend_full = IsBackendFullCb;
    callbacks.open_packet = OpenPacketCb;
    callbacks.close_packet = ClosePacketCb;

    // Configure exceptions so that stream operations throw instead of
    // just setting flags on error.
    output_.exceptions(std::ofstream::failbit | std::ofstream::badbit);

    // Open CTF data stream output file in binary mode.
    output_.open(data_stream_file_path, std::ios_base::out | std::ios_base::binary);

    // Initialize the raw barectf context.
    barectf_init(&ctx_, buffer_.data(), buffer_.size(), callbacks, this);

    // Open the initial packet.
    OpenPacketCb();
  }

 public:
  // Disabled copy operations to make this class simpler.
  BarectfPlatform(const BarectfPlatform&) = delete;
  BarectfPlatform& operator=(const BarectfPlatform&) = delete;

  // Closes/writes any last CTF packet and closes the data stream file.
  ~BarectfPlatform() {
    if (barectf_packet_is_open(&ctx_) && !barectf_packet_is_empty(&ctx_)) {
      // Close and write last CTF packet (not empty).
      ClosePacketCb();
    }

    // Close data stream output file.
    output_.close();
  }

  // Returns the raw barectf context of this platform.
  const typename DescrT::Ctx& GetCtx() const noexcept { return ctx_; }
  typename DescrT::Ctx& GetCtx() noexcept { return ctx_; }

 private:
  static BarectfPlatform& AsPlatform(void* const data) noexcept {
    return *static_cast<BarectfPlatform*>(data);
  }

  // Four callbacks for barectf.
  //
  // Those four functions receive an instance of this class as `data`.

  static std::uint64_t GetClockCb(void* const data) noexcept {
    // Forward to instance method.
    return AsPlatform(data).GetClockCb();
  }

  static int IsBackendFullCb(void* const data) noexcept {
    // Forward to instance method.
    return AsPlatform(data).IsBackendFullCb();
  }

  static void OpenPacketCb(void* const data) {
    // Forward to instance method.
    AsPlatform(data).OpenPacketCb();
  }

  static void ClosePacketCb(void* const data) {
    // Forward to instance method.
    AsPlatform(data).ClosePacketCb();
  }

  // Instance version of the "get clock value" callback.
  std::uint64_t GetClockCb() noexcept { return *clock_val_; }

  // Instance version of the "is the back end full?" callback.
  int IsBackendFullCb() noexcept {
    // Never full.
    return 0;
  }

  // Instance version of the "open packet" callback.
  void OpenPacketCb() {
    // Forward to user (descriptor) function.
    DescrT::OpenPacket(ctx_);
  }

  // Instance version of the "close packet" callback.
  void ClosePacketCb() {
    // Forward to user (descriptor) function to finalize the packet.
    DescrT::ClosePacket(ctx_);

    // Write to the data stream file.
    WriteCurrentPacket();
  }

  // Writes the current CTF packet (`buffer_`) to the data stream file.
  void WriteCurrentPacket() {
    output_.write(reinterpret_cast<const char*>(buffer_.data()), buffer_.size());
  }

  // Clock value pointer.
  const std::uint64_t* clock_val_;

  // CTF data stream output file stream.
  std::ofstream output_;

  // Raw barectf context.
  typename DescrT::Ctx ctx_;

  // CTF packet buffer.
  std::vector<std::uint8_t> buffer_;
};

}  // namespace rocm_ctf

#endif  // PLUGIN_CTF_BARECTF_PLATFORM_H
