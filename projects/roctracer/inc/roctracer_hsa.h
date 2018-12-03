/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef INC_ROCTRACER_HSA_H_
#define INC_ROCTRACER_HSA_H_
#include <iostream>
#include <mutex>

#include <hsa.h>
#include <hsa_api_trace.h>
#include <hsa_ext_amd.h>

#include "ext/prof_protocol.h"
#include "roctracer.h"

namespace roctracer {
namespace hsa_support {
template <int N>
class CbTable {
  public:
  typedef std::mutex mutex_t;

  CbTable() {
    std::lock_guard<mutex_t> lck(mutex_);
    for (int i = 0; i < N; i++) {
      callback_[i] = NULL;
      arg_[i] = NULL;
    }
  }

  void set(uint32_t id, activity_rtapi_callback_t callback, void* arg) {
    std::lock_guard<mutex_t> lck(mutex_);
    callback_[id] = callback;
    arg_[id] = arg;
  }

  void get(uint32_t id, activity_rtapi_callback_t* callback, void** arg) {
    std::lock_guard<mutex_t> lck(mutex_);
    *callback = callback_[id];
    *arg = arg_[id];
  }

  private:
  activity_rtapi_callback_t callback_[N];
  void* arg_[N];
  mutex_t mutex_;
};

extern CoreApiTable CoreApiTable_saved;
extern AmdExtTable AmdExtTable_saved;
extern ImageExtTable ImageExtTable_saved;
};
};

inline std::ostream& operator<< (std::ostream& out, const hsa_callback_data_t& v) { out << "<callback_data>"; return out; }
inline std::ostream& operator<< (std::ostream &out, const hsa_signal_t& v) { out << "<signal " << std::hex << "0x" << v.handle << ">" << std::dec; return out; }
inline std::ostream& operator<< (std::ostream &out, const hsa_signal_group_t& v) { out << "<signal_group>"; return out; }
inline std::ostream& operator<< (std::ostream &out, const hsa_wavefront_t& v) { out << "<wavefront " << std::hex << "0x" << v.handle << ">" << std::dec; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_cache_t& v) { out << "<cache>"; return out; }
inline std::ostream& operator<< (std::ostream &out, const hsa_region_t& v) { out << "<region " << std::hex << "0x" << v.handle << ">" << std::dec; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_amd_memory_pool_t& v) { out << "<amd_memory_pool>"; return out; }
inline std::ostream& operator<< (std::ostream &out, const hsa_agent_t& v) { out << "<agent " << std::hex << "0x" << v.handle << ">" << std::dec; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_isa_t& v) { out << "<isa>"; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_code_symbol_t& v) { out << "<code_symbol>"; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_code_object_t& v) { out << "<code_object>"; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_code_object_reader_t& v) { out << "<code_object_reader>"; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_executable_symbol_t& v) { out << "<executable_symbol>"; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_executable_t& v) { out << "<executable>"; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_ext_image_t& v) { out << "<ext_image>"; return out; }
inline std::ostream& operator<< (std::ostream& out, const hsa_ext_sampler_t& v) { out << "<ext_sampler>"; return out; }

namespace roctracer {
namespace hsa_support {
template <typename T>
struct output_streamer {
  inline static std::ostream& put(std::ostream& out, const T& v) { out << v; return out; }
};

template<>
struct output_streamer<bool> {
  inline static std::ostream& put(std::ostream& out, bool v) { out << std::hex << "<bool " << "0x" << v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint8_t> {
  inline static std::ostream& put(std::ostream& out, uint8_t v) { out << std::hex << "<uint8_t " << "0x" << v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint16_t> {
  inline static std::ostream& put(std::ostream& out, uint16_t v) { out << std::hex << "<uint16_t " << "0x" << v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint32_t> {
  inline static std::ostream& put(std::ostream& out, uint32_t v) { out << std::hex << "<uint32_t " << "0x" << v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint64_t> {
  inline static std::ostream& put(std::ostream& out, uint64_t v) { out << std::hex << "<uint64_t " << "0x" << v << std::dec << ">"; return out; }
};

template<>
struct output_streamer<bool*> {
  inline static std::ostream& put(std::ostream& out, bool* v) { out << std::hex << "<bool " << "0x" << *v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint8_t*> {
  inline static std::ostream& put(std::ostream& out, uint8_t* v) { out << std::hex << "<uint8_t " << "0x" << *v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint16_t*> {
  inline static std::ostream& put(std::ostream& out, uint16_t* v) { out << std::hex << "<uint16_t " << "0x" << *v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint32_t*> {
  inline static std::ostream& put(std::ostream& out, uint32_t* v) { out << std::hex << "<uint32_t " << "0x" << *v << std::dec << ">"; return out; }
};
template<>
struct output_streamer<uint64_t*> {
  inline static std::ostream& put(std::ostream& out, uint64_t* v) { out << std::hex << "<uint64_t " << "0x" << *v << std::dec << ">"; return out; }
};

template<>
struct output_streamer<hsa_queue_t*> {
  inline static std::ostream& put(std::ostream& out, hsa_queue_t* v) { out << "<queue " << v << ">"; return out; }
};
template<>
struct output_streamer<hsa_queue_t**> {
  inline static std::ostream& put(std::ostream& out, hsa_queue_t** v) { out << "<queue " << *v << ">"; return out; }
};
};};

#include "inc/hsa_prof_str.h"
#endif // INC_ROCTRACER_HSA_H_
