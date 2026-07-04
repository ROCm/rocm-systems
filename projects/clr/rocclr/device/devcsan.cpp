/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/devcsan.hpp"

#include "device/device.hpp"

#include "amd_comgr/amd_comgr.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <tuple>
#include <utility>

namespace amd {

namespace {

// Deduplicates races on the conflicting PC pair and kind.
struct SanitizerTables {
  bool isNewRace(uint32_t device, uint64_t pc, uint64_t peer_pc, unsigned kind) {
    if (pc > peer_pc) std::swap(pc, peer_pc);
    std::lock_guard<std::mutex> guard(mtx_);
    return races_.insert({device, pc ^ (static_cast<uint64_t>(kind) << 56), peer_pc}).second;
  }

 private:
  std::mutex mtx_;
  std::set<std::tuple<uint32_t, uint64_t, uint64_t>> races_;
};

SanitizerTables& sanitizerTables() {
  static SanitizerTables tables;
  return tables;
}

// The symbolizer used to extract information about the program counter from the
// loader code objects.
struct Symbolizer {
  amd_comgr_data_t data{};
  amd_comgr_symbolizer_info_t info{};
  bool haveData = false, ready = false;
  int64_t bias = 0;

  static void sink(const char* text, void* user) {
    if (text) static_cast<std::string*>(user)->append(text);
  }

  bool init(const amd::Device::SanitizerCodeObject& co) {
    bias = co.bias;
    if (amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data) != AMD_COMGR_STATUS_SUCCESS)
      return false;
    haveData = true;
    amd_comgr_status_t status =
        co.memory ? amd_comgr_set_data(data, co.size, static_cast<const char*>(co.memory))
                  : amd_comgr_set_data_from_file_slice(data, co.fd, co.offset, co.size);
    if (status != AMD_COMGR_STATUS_SUCCESS) return false;
    if (amd_comgr_create_symbolizer_info(data, &sink, &info) != AMD_COMGR_STATUS_SUCCESS)
      return false;
    ready = true;
    return true;
  }

  ~Symbolizer() {
    if (ready) amd_comgr_destroy_symbolizer_info(info);
    if (haveData) amd_comgr_release_data(data);
  }

  std::string symbolize(uint64_t imageVa, bool isCode) {
    std::string out;
    if (ready) amd_comgr_symbolize(info, imageVa, isCode, &out);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) out.pop_back();
    if (out == "??" || out == "<invalid>") out.clear();
    return out;
  }
};

// Print the source frames for a device PC; comgr emits one line per inlined
// frame.
void printBacktrace(Symbolizer& sym, uint64_t devicePc) {
  uint64_t imagePc = devicePc + sym.bias;
  // The device records return addresses; step back into the call site.
  std::string text = sym.symbolize(imagePc ? imagePc - 1 : imagePc, /*isCode=*/true);
  if (text.empty()) {
    fprintf(stderr, "==CSAN==     #0 ?? (0x%" PRIx64 ")\n", imagePc);
    return;
  }
  unsigned frame = 0;
  for (size_t pos = 0; pos < text.size();) {
    size_t nl = text.find('\n', pos);
    if (nl == std::string::npos) nl = text.size();
    fprintf(stderr, "==CSAN==     #%u %.*s (0x%" PRIx64 ")\n", frame++, static_cast<int>(nl - pos),
            text.c_str() + pos, imagePc);
    pos = nl + 1;
  }
}

}  // namespace

void reportGpuCSanRace(const amd::Device& dev, const __tsan_gpu_race& race) {
  if (!sanitizerTables().isNewRace(dev.index(), race.pc, race.peer_pc, race.kind)) return;

  // Resolve the owning code object from the loader's live view every time; the
  // reporting kernel (hence its image) is still loaded while this RPC blocks.
  amd::Device::SanitizerCodeObject co;
  Symbolizer sym;
  if (dev.ResolveSanitizerCodeObject(race.pc, &co)) sym.init(co);

  const char* op = (race.access_type & TSAN_GPU_ACCESS_COMPOUND) ? "Read-modify-write"
                   : (race.access_type & TSAN_GPU_ACCESS_WRITE)  ? "Write"
                                                                 : "Read";
  const char* atomic = (race.access_type & TSAN_GPU_ACCESS_ATOMIC) ? "atomic " : "";
  const char* kind = race.kind == TSAN_GPU_UNKNOWN_ORIGIN ? "data race (unknown origin)"
                     : race.kind == TSAN_GPU_INTRA_WAVE   ? "data race (intra-wave)"
                                                          : "data race";

  fprintf(stderr, "==CSAN== WARNING: ConcurrencySanitizer: %s\n", kind);
  fprintf(stderr,
          "==CSAN==   %s%s of size %u at 0x%" PRIx64
          " in block (%u,%u,%u) thread (%u,%u,%u) lane %u\n",
          atomic, op, race.size, race.addr, race.block[0], race.block[1], race.block[2],
          race.thread[0], race.thread[1], race.thread[2], race.lane);

  printBacktrace(sym, race.pc);
  if (race.kind == TSAN_GPU_INTRA_WAVE) {
    // The conflicting lane executed the same instruction, so it shares the PC.
    fprintf(stderr,
            "==CSAN==   Previous access in block (%u,%u,%u) thread (%u,%u,%u) "
            "lane %u\n",
            race.block[0], race.block[1], race.block[2], race.peer_thread[0], race.peer_thread[1],
            race.peer_thread[2], race.peer_lane);
    printBacktrace(sym, race.pc);
  } else if (race.peer_pc) {
    fprintf(stderr, "==CSAN==   Previous access:\n");
    printBacktrace(sym, race.peer_pc);
  }
  if (sym.ready) {
    // comgr renders data as "name\nstart\nsize"; keep the name if it resolved.
    std::string var = sym.symbolize(race.addr + sym.bias, /*isCode=*/false);
    var = var.substr(0, var.find('\n'));
    if (!var.empty() && var != "??")
      fprintf(stderr, "==CSAN==   Address 0x%" PRIx64 " is global variable '%s'\n", race.addr,
              var.c_str());
  }
}

}  // namespace amd
