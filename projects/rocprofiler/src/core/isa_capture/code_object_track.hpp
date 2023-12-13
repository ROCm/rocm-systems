/* Copyright (c) 2023 Advanced Micro Devices, Inc.

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

#pragma once

#include "src/utils/helper.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <mutex>
#include <string>
#include <fstream>

/**
 * A class to keep track of currently loaded code objects.
 * Only the public static methods are thread-safe and expected to be used.
 */
class codeobj_capture_instance {
 public:
  codeobj_capture_instance(
    uint64_t _addr,
    uint64_t _load_size,
    const std::string& _uri,
    uint64_t mem_addr,
    uint64_t mem_size,
    uint64_t start_time,
    uint32_t id
  )
    : addr(_addr), load_size(_load_size), start_time(start_time),
    load_id(id), URI(_uri), mem_addr(mem_addr), mem_size(mem_size) {};

  void setmode(rocprofiler_codeobj_capture_mode_t mode);

  rocprofiler_intercepted_codeobj_t get() const {
    const char* buf_ptr = buffer.size() ? buffer.data() : nullptr;
    return {URI.c_str(), addr, load_size, buf_ptr, buffer.size(), start_time, end_time, load_id};
  };

  const uint64_t addr;
  const uint64_t load_size;
  const uint64_t start_time;
  const uint32_t load_id;

  static void Load(
    uint64_t addr,
    uint64_t load_size,
    const std::string& URI,
    uint64_t mem_addr,
    uint64_t mem_size
  );
  static void Unload(uint64_t addr);
  static uint32_t GetEventCount() { return eventcount.load(std::memory_order_relaxed); }

 private:
  //! 32 bits ID because this is the natural channel width for ATT Markers.
  //! There is no world in which 4 billions markers can be sent anyway.
  static std::atomic<uint32_t> eventcount;
  void reset(rocprofiler_codeobj_capture_mode_t mode);

  std::pair<size_t, size_t> parse_uri();
  void DecodePath();
  void copyCodeobjFromFile(uint64_t offset, uint64_t size, const std::string& decoded_path);
  void copyCodeobjFromMemory(uint64_t mem_addr, uint64_t mem_size);

  std::string URI;
  std::string decoded_path;
  std::string protocol;
  std::vector<char> buffer;

  uint64_t mem_addr;
  uint64_t mem_size;
  uint64_t end_time = 0;
  int capture_mode = -1;
};

typedef std::shared_ptr<codeobj_capture_instance> CodeobjPtr;

template <> struct std::hash<CodeobjPtr> {
  // addr is typically 2^12-byte aligned. Taking last 44 bits of time == cycle time of many hours.
  uint64_t operator()(const CodeobjPtr& p) const {
    return (p->addr >> 12) ^ (p->start_time << 20);
  };
};

template <> struct std::equal_to<CodeobjPtr> {
  bool operator()(const CodeobjPtr& a, const CodeobjPtr& b) const {
    return (a->addr == b->addr) & (a->start_time == b->start_time);
  };
};

/**
 * A class to keep track of the history of loaded code objets.
 * Only the public static methods are thread-safe and expected to be used.
 */
class codeobj_record {
 public:
  codeobj_record(rocprofiler_codeobj_capture_mode_t mode);
  ~codeobj_record() {
    if (listeners.find(this) != listeners.end()) stop_capture();
  };

  void addcapture(CodeobjPtr& capture);

 public:
  static void make_capture(rocprofiler_record_id_t id, rocprofiler_codeobj_capture_mode_t mode,
                           uint64_t userdata);
  static void free_capture(rocprofiler_record_id_t id);
  static void start_capture(rocprofiler_record_id_t id);
  static void stop_capture(rocprofiler_record_id_t id);
  static rocprofiler_codeobj_symbols_t get_capture(rocprofiler_record_id_t id);

  static std::unordered_set<codeobj_record*> listeners;
  static std::mutex mutex;
  static std::unordered_map<uint64_t, CodeobjPtr> codeobjs;

 private:
  rocprofiler_codeobj_symbols_t get(uint64_t userdata) {
    persist.clear();
    for (auto& capt : captures) persist.push_back(capt->get());
    return rocprofiler_codeobj_symbols_t{persist.data(), persist.size(), userdata};
  };

  void start_capture();
  void stop_capture();

  rocprofiler_codeobj_capture_mode_t capture_mode;
  std::vector<rocprofiler_intercepted_codeobj_t> persist;
  std::unordered_set<CodeobjPtr> captures;

  // Record_id -> codeobj
  using RecordInstance = std::pair<uint64_t, std::unique_ptr<codeobj_record>>;
  static std::unordered_map<uint64_t, RecordInstance> record_id_map;
};
