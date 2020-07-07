/******************************************************************************
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
*******************************************************************************/

#ifndef SRC_CORE_PROFILE_H_
#define SRC_CORE_PROFILE_H_

#include "inc/rocprofiler.h"

#include <hsa.h>
#include <vector>

#include "core/types.h"
#include "util/exception.h"
#include "util/hsa_rsrc_factory.h"

namespace rocprofiler {
struct profile_info_t {
  const event_t* event;
  const parameter_t* parameters;
  uint32_t parameter_count;
  rocprofiler_feature_t* rinfo;
};
typedef std::vector<rocprofiler_feature_t*> info_vector_t;
typedef std::vector<packet_t> pkt_vector_t;
struct profile_tuple_t {
  const profile_t* profile;
  info_vector_t* info_vector;
  hsa_signal_t completion_signal;
};
typedef std::vector<profile_tuple_t> profile_vector_t;

template <class Item> class ConfigBase {};

template <> class ConfigBase<event_t> {
 public:
  ConfigBase(profile_t* profile) : profile_(profile) {}

 protected:
  void* Array() { return const_cast<event_t*>(profile_->events); }
  unsigned Count() const { return profile_->event_count; }
  void Set(event_t* events, const unsigned& count) {
    profile_->events = events;
    profile_->event_count = count;
  }
  profile_t* profile_;
};

template <> class ConfigBase<parameter_t> {
 public:
  ConfigBase(profile_t* profile) : profile_(profile) {}

 protected:
  void* Array() { return const_cast<parameter_t*>(profile_->parameters); }
  unsigned Count() const { return profile_->parameter_count; }
  void Set(parameter_t* parameters, const unsigned& count) {
    profile_->parameters = parameters;
    profile_->parameter_count = count;
  }
  profile_t* profile_;
};

template <class Item> class Config : protected ConfigBase<Item> {
  typedef ConfigBase<Item> Parent;

 public:
  Config(profile_t* profile) : Parent(profile) {}
  void Insert(const Item& item) {
    auto count = Parent::Count();
    count += 1;
    Item* array =
        reinterpret_cast<Item*>(realloc(const_cast<void*>(Parent::Array()), count * sizeof(Item)));
    array[count - 1] = item;
    Parent::Set(array, count);
  }
};

class Profile {
 public:
  static const uint32_t LEGACY_SLOT_SIZE_PKT =
      HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(packet_t);

  Profile(const util::AgentInfo* agent_info) : agent_info_(agent_info) {
    profile_ = {};
    profile_.agent = agent_info->dev_id;
    completion_signal_ = {};
    is_legacy_ = (strncmp(agent_info->name, "gfx8", 4) == 0);
  }

  virtual ~Profile() {
    info_vector_.clear();
    if (profile_.command_buffer.ptr) util::HsaRsrcFactory::FreeMemory(profile_.command_buffer.ptr);
    if (profile_.output_buffer.ptr) util::HsaRsrcFactory::FreeMemory(profile_.output_buffer.ptr);
    if (profile_.events) free(const_cast<event_t*>(profile_.events));
    if (profile_.parameters) free(const_cast<parameter_t*>(profile_.parameters));
    if (completion_signal_.handle) {
      hsa_status_t status = hsa_signal_destroy(completion_signal_);
      if (status != HSA_STATUS_SUCCESS) EXC_ABORT(status, "signal_destroy " << std::hex << status);
    }
  }

  virtual void Insert(const profile_info_t& info) { info_vector_.push_back(info.rinfo); }

  void SetConcurrent(profile_t* profile) {
    // Check whether conconcurrent has been set
    for (const parameter_t* p = profile->parameters;
            p < (profile->parameters + profile->parameter_count); ++p) {
      // If yes, stop here
      if (p->parameter_name == HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT) {
        return;
      }
    }

    // Otherwise, try to set
    parameter_t* parameters = new parameter_t[profile->parameter_count+1];
    for (unsigned i = 0; i < profile->parameter_count; ++i) {
      parameters[i].parameter_name = profile->parameters[i].parameter_name;
      parameters[i].value = profile->parameters[i].value;
    }
    if (profile->parameters) free(const_cast<parameter_t*>(profile->parameters));
    parameters[profile->parameter_count].parameter_name =
        HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_K_CONCURRENT;
    parameters[profile->parameter_count].value = 1;
    profile->parameters = parameters;
    profile->parameter_count += 1;
  }

  hsa_status_t Finalize(pkt_vector_t& start_vector, pkt_vector_t& stop_vector,
          pkt_vector_t& read_vector, bool is_concurrent = false) {
    if (is_concurrent) SetConcurrent(&profile_);

    hsa_status_t status = HSA_STATUS_SUCCESS;

    if (!info_vector_.empty()) {
      util::HsaRsrcFactory* rsrc = &util::HsaRsrcFactory::Instance();
      const pfn_t* api = rsrc->AqlProfileApi();
      packet_t start{};
      packet_t stop{};
      packet_t read{};      // read at kernel start
      packet_t read2{};     // read at kernel end

      // Check the profile buffer sizes
      status = api->hsa_ven_amd_aqlprofile_start(&profile_, NULL);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "aqlprofile_start(NULL)");
      // Double output buffer size if concurrent
      if (is_concurrent) profile_.output_buffer.size *= 2;
      status = Allocate(rsrc);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "Allocate()");

      // Generate start/stop/read profiling packets
      status = api->hsa_ven_amd_aqlprofile_start(&profile_, &start);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "aqlprofile_start");
      status = api->hsa_ven_amd_aqlprofile_stop(&profile_, &stop);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "aqlprofile_stop");
      hsa_status_t rd_status = HSA_STATUS_ERROR;
#ifdef AQLPROF_NEW_API
      if (profile_.type == HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC) {
        rd_status = api->hsa_ven_amd_aqlprofile_read(&profile_, &read);
        if (is_concurrent){         // concurrent: one more read
          if (rd_status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "aqlprofile_read");
          rd_status = api->hsa_ven_amd_aqlprofile_read(&profile_, &read2);
        }
      }
#if 0 // Read API returns error if disabled
      if (rd_status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "aqlprofile_read");
#endif
#endif

      // Set completion signal of start
      hsa_signal_t dummy_signal{};
      dummy_signal.handle = 0;
      start.completion_signal = dummy_signal;

      // Set completion signal of read/stop
      hsa_signal_t post_signal;
      status = hsa_signal_create(1, 0, NULL, &post_signal);
      if (status != HSA_STATUS_SUCCESS) EXC_RAISING(status, "signal_create " << std::hex << status);
      stop.completion_signal = post_signal;
      read.completion_signal = post_signal;
      read2.completion_signal = post_signal;
      completion_signal_ = post_signal;

      // Fill packet vectors
      if (is_legacy_) {
        const uint32_t start_index = start_vector.size();
        const uint32_t stop_index = stop_vector.size();

        start_vector.insert(start_vector.end(), LEGACY_SLOT_SIZE_PKT, packet_t{});
        stop_vector.insert(stop_vector.end(), LEGACY_SLOT_SIZE_PKT, packet_t{});

        status = api->hsa_ven_amd_aqlprofile_legacy_get_pm4(
            &start, reinterpret_cast<void*>(&start_vector[start_index]));
        if (status != HSA_STATUS_SUCCESS)
          AQL_EXC_RAISING(status, "hsa_ven_amd_aqlprofile_legacy_get_pm4");

        status = api->hsa_ven_amd_aqlprofile_legacy_get_pm4(
            &stop, reinterpret_cast<void*>(&stop_vector[stop_index]));
        if (status != HSA_STATUS_SUCCESS)
          AQL_EXC_RAISING(status, "hsa_ven_amd_aqlprofile_legacy_get_pm4");

        if (rd_status == HSA_STATUS_SUCCESS) {
          pkt_vector_t reads = {read};
          if (is_concurrent) reads.push_back(read2);
          for (auto rd : reads) {
            const uint32_t read_index = read_vector.size();
            read_vector.insert(read_vector.end(), LEGACY_SLOT_SIZE_PKT, packet_t{});
            status = api->hsa_ven_amd_aqlprofile_legacy_get_pm4(
                &rd, reinterpret_cast<void*>(&read_vector[read_index]));
            if (status != HSA_STATUS_SUCCESS)
              AQL_EXC_RAISING(status, "hsa_ven_amd_aqlprofile_legacy_get_pm4");
          }
        }
      } else {
        start_vector.push_back(start);
        stop_vector.push_back(stop);
        if (rd_status == HSA_STATUS_SUCCESS) {
          read_vector.push_back(read);
          if (is_concurrent)
            read_vector.push_back(read2);
        }
      }
    }

    return status;
  }

  void GetProfiles(profile_vector_t& vec) {
    if (!info_vector_.empty()) {
      vec.push_back(profile_tuple_t{&profile_, &info_vector_, completion_signal_});
    }
  }

  bool Empty() const { return info_vector_.empty(); }

 protected:
  virtual hsa_status_t Allocate(util::HsaRsrcFactory* rsrc) = 0;

  const util::AgentInfo* const agent_info_;
  bool is_legacy_;
  profile_t profile_;
  info_vector_t info_vector_;
  hsa_signal_t completion_signal_;
};

class PmcProfile : public Profile {
 public:
  PmcProfile(const util::AgentInfo* agent_info) : Profile(agent_info) {
    profile_.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC;
  }

  void Insert(const profile_info_t& info) {
    Profile::Insert(info);
    Config<event_t>(&profile_).Insert(*(info.event));
  }

  hsa_status_t Allocate(util::HsaRsrcFactory* rsrc) {
    profile_.command_buffer.ptr =
      rsrc->AllocateSysMemory(agent_info_, profile_.command_buffer.size);
    profile_.output_buffer.ptr = rsrc->AllocateSysMemory(agent_info_, profile_.output_buffer.size);
    return (profile_.command_buffer.ptr && profile_.output_buffer.ptr) ? HSA_STATUS_SUCCESS
                                                                       : HSA_STATUS_ERROR;
  }
};

class TraceProfile : public Profile {
 public:
  static inline void SetSize(const uint32_t& size) { output_buffer_size_ = size; }
  static inline uint32_t GetSize() { return output_buffer_size_; }
  static inline void SetLocal(const bool& b) { output_buffer_local_ = b; }
  static inline bool IsLocal() { return output_buffer_local_; }

  TraceProfile(const util::AgentInfo* agent_info) : Profile(agent_info) {
    profile_.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE;
  }

  void Insert(const profile_info_t& info) {
    if (info.parameters != NULL) {
      Profile::Insert(info);
      for (unsigned j = 0; j < info.parameter_count; ++j) {
        Config<parameter_t>(&profile_).Insert(info.parameters[j]);
      }
    } else if (info.event != NULL) {
      Config<event_t>(&profile_).Insert(*(info.event));
    } else {
      EXC_ABORT(HSA_STATUS_ERROR, "invalid trace info inserted");
    }
  }

  hsa_status_t Allocate(util::HsaRsrcFactory* rsrc) {
    profile_.command_buffer.ptr =
      rsrc->AllocateSysMemory(agent_info_, profile_.command_buffer.size);
    profile_.output_buffer.size = output_buffer_size_;
    profile_.output_buffer.ptr = (output_buffer_local_) ?
      rsrc->AllocateLocalMemory(agent_info_, profile_.output_buffer.size) :
      rsrc->AllocateSysMemory(agent_info_, profile_.output_buffer.size);
    return (profile_.command_buffer.ptr && profile_.output_buffer.ptr) ? HSA_STATUS_SUCCESS
                                                                       : HSA_STATUS_ERROR;
  }

 private:
  static uint32_t output_buffer_size_;
  static bool output_buffer_local_;
};

}  // namespace rocprofiler

#endif  // SRC_CORE_PROFILE_H_
