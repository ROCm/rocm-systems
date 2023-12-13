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

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/mman.h>
#include <hsa/hsa.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa_ven_amd_loader.h>
#include <iostream>
#include "src/utils/helper.h"
#include "src/api/rocprofiler_singleton.h"
#include "src/core/isa_capture/code_object_track.hpp"
#include <amd_comgr/amd_comgr.h>

std::mutex codeobj_record::mutex;

std::unordered_map<uint64_t, CodeobjPtr> codeobj_record::codeobjs{};
std::unordered_map<uint64_t, codeobj_record::RecordInstance> codeobj_record::record_id_map{};
std::unordered_set<codeobj_record*> codeobj_record::listeners;
std::atomic<uint32_t> codeobj_capture_instance::eventcount{0};

// Codeobj Record
codeobj_record::codeobj_record(rocprofiler_codeobj_capture_mode_t mode) : capture_mode(mode){};

void codeobj_record::start_capture() {
  listeners.insert(this);
  for (auto& [addr, capture] : codeobjs) this->addcapture(capture);
}

void codeobj_record::addcapture(CodeobjPtr& capture) {
  if (captures.find(capture) != captures.end()) return;
  capture->setmode(capture_mode);
  captures.insert(capture);
}

void codeobj_record::stop_capture() {
  listeners.erase(this);
}

// Codeobj Capture
void codeobj_capture_instance::Load(
  uint64_t addr,
  uint64_t load_size,
  const std::string& URI,
  uint64_t mem_addr,
  uint64_t mem_size
) {
  uint32_t id = eventcount.fetch_add(1, std::memory_order_relaxed)+1;
  auto time = rocprofiler::ROCProfiler_Singleton::GetInstance().timestamp_ns().value;

  std::lock_guard<std::mutex> lock(codeobj_record::mutex);

  auto inst = std::make_shared<codeobj_capture_instance>(addr, load_size, URI, mem_addr, mem_size, time, id);
  codeobj_record::codeobjs[addr] = inst;
  for (auto* listen : codeobj_record::listeners) listen->addcapture(inst);
}

void codeobj_capture_instance::Unload(uint64_t addr) {
  std::lock_guard<std::mutex> lock(codeobj_record::mutex);

  if (codeobj_record::codeobjs.find(addr) == codeobj_record::codeobjs.end()) return;

  eventcount.fetch_add(1, std::memory_order_relaxed);
  auto time = rocprofiler::ROCProfiler_Singleton::GetInstance().timestamp_ns().value;
  codeobj_record::codeobjs.at(addr)->end_time = time;
  codeobj_record::codeobjs.erase(addr);
}

void codeobj_capture_instance::copyCodeobjFromFile(uint64_t offset, uint64_t size,
                                                   const std::string& decoded_path) {
  std::ifstream file(decoded_path, std::ios::in | std::ios::binary);
  if (!file) {
    printf("could not open `%s'\n", decoded_path.c_str());
    return;
  }

  if (!size) {
    file.ignore(std::numeric_limits<std::streamsize>::max());
    size_t bytes = file.gcount();
    file.clear();

    if (bytes < offset) {
      printf("invalid uri `%s' (file size < offset)\n", decoded_path.c_str());
      return;
    }
    size = bytes - offset;
  }

  file.seekg(offset, std::ios_base::beg);
  buffer.resize(size);
  file.read(&buffer[0], size);
}

void codeobj_capture_instance::copyCodeobjFromMemory(uint64_t mem_addr, uint64_t mem_size) {
  buffer.resize(mem_size);
  std::memcpy(buffer.data(), (uint64_t*)mem_addr, mem_size);
}

std::pair<size_t, size_t> codeobj_capture_instance::parse_uri() {
  const std::string protocol_delim{"://"};

  size_t protocol_end = URI.find(protocol_delim);
  protocol = URI.substr(0, protocol_end);
  protocol_end += protocol_delim.length();

  std::transform(protocol.begin(), protocol.end(), protocol.begin(),
                 [](unsigned char c) { return std::tolower(c); });

  std::string path;
  size_t path_end = URI.find_first_of("#?", protocol_end);
  if (path_end != std::string::npos) {
    path = URI.substr(protocol_end, path_end++ - protocol_end);
  } else {
    path = URI.substr(protocol_end);
  }

  /* %-decode the string.  */
  decoded_path = std::string{};
  decoded_path.reserve(path.length());
  for (size_t i = 0; i < path.length(); ++i) {
    if (path[i] == '%' && std::isxdigit(path[i + 1]) && std::isxdigit(path[i + 2])) {
      decoded_path += std::stoi(path.substr(i + 1, 2), 0, 16);
      i += 2;
    } else {
      decoded_path += path[i];
    }
  }

  /* Tokenize the query/fragment.  */
  std::vector<std::string> tokens;
  size_t pos, last = path_end;
  while ((pos = URI.find('&', last)) != std::string::npos) {
    tokens.emplace_back(URI.substr(last, pos - last));
    last = pos + 1;
  }
  if (last != std::string::npos) tokens.emplace_back(URI.substr(last));

  /* Create a tag-value map from the tokenized query/fragment.  */
  std::unordered_map<std::string, std::string> params;
  std::for_each(tokens.begin(), tokens.end(), [&](std::string& token) {
    size_t delim = token.find('=');
    if (delim != std::string::npos) {
      params.emplace(token.substr(0, delim), token.substr(delim + 1));
    }
  });

  size_t offset{0}, size{0};

  if (auto offset_it = params.find("offset"); offset_it != params.end())
    offset = std::stoul(offset_it->second, nullptr, 0);

  if (auto size_it = params.find("size"); size_it != params.end()) {
    if (!(size = std::stoul(size_it->second, nullptr, 0))) throw std::exception();
  }

  return {offset, size};
}

void codeobj_capture_instance::setmode(rocprofiler_codeobj_capture_mode_t mode) {
  // Only reset when needed & check if codeobj was not unloaded
  if (static_cast<int>(mode) > capture_mode) reset(mode);
}

void codeobj_capture_instance::reset(rocprofiler_codeobj_capture_mode_t mode) {
  capture_mode = static_cast<int>(mode);
  if (!buffer.empty()) return;

  size_t offset, size;
  try {
    std::tie(offset, size) = parse_uri();
  } catch (...) {
    rocprofiler::warning("Error parsing URI %s", URI.c_str());
    return;
  }
  if (protocol == "file") {
    if (mode == ROCPROFILER_CAPTURE_COPY_FILE_AND_MEMORY)
      copyCodeobjFromFile(offset, size, decoded_path);
  } else if (protocol == "memory") {
    if (mode != ROCPROFILER_CAPTURE_SYMBOLS_ONLY && end_time == 0)
      copyCodeobjFromMemory(mem_addr, mem_size);
  } else {
    printf("\"%s\" protocol not supported\n", protocol.c_str());
  }
}

// Public static funcs
void codeobj_record::make_capture(rocprofiler_record_id_t id,
                                  rocprofiler_codeobj_capture_mode_t mode, uint64_t userdata) {
  std::lock_guard<std::mutex> lock(mutex);
  record_id_map[id.handle] = {userdata, std::unique_ptr<codeobj_record>{new codeobj_record(mode)}};
}

void codeobj_record::free_capture(rocprofiler_record_id_t id) {
  std::lock_guard<std::mutex> lock(mutex);
  record_id_map.erase(id.handle);
}

void codeobj_record::start_capture(rocprofiler_record_id_t id) {
  std::lock_guard<std::mutex> lock(mutex);
  record_id_map.at(id.handle).second->start_capture();
}

void codeobj_record::stop_capture(rocprofiler_record_id_t id) {
  std::lock_guard<std::mutex> lock(mutex);
  record_id_map.at(id.handle).second->stop_capture();
}

rocprofiler_codeobj_symbols_t codeobj_record::get_capture(rocprofiler_record_id_t id) {
  std::lock_guard<std::mutex> lock(mutex);
  auto& pair = record_id_map.at(id.handle);
  return pair.second->get(pair.first);
}
