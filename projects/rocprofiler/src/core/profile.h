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

template<class Item> class ConfigBase {};

template<> class ConfigBase<event_t> {
  public:
  ConfigBase(profile_t *profile) : profile_(profile) {}

  protected:
  void* Array() { return const_cast<event_t*>(profile_->events); }
  unsigned Count() const { return profile_->event_count; }
  void Set(event_t* events, const unsigned& count) {
    profile_->events = events;
    profile_->event_count = count;
  }
  profile_t* profile_;
};

template<> class ConfigBase<parameter_t> {
  public:
  ConfigBase(profile_t *profile) : profile_(profile) {}

  protected:
  void* Array() { return const_cast<parameter_t*>(profile_->parameters); }
  unsigned Count() const { return profile_->parameter_count; }
  void Set(parameter_t* parameters, const unsigned& count) {
    profile_->parameters = parameters;
    profile_->parameter_count = count;
  }
  profile_t* profile_;
};

template<class Item> 
class Config : protected ConfigBase<Item> {
  typedef ConfigBase<Item> Parent;
  public:
  Config(profile_t *profile) : Parent(profile) {}
  void Insert(const Item& item) {
    auto count = Parent::Count();
    count += 1;
    Item* array = reinterpret_cast<Item*>(realloc(const_cast<void*>(Parent::Array()), count * sizeof(Item)));
    array[count - 1] = item;
    Parent::Set(array, count);
  }
};

class Profile {
  public:
  static const uint32_t LEGACY_SLOT_SIZE_PKT = HSA_VEN_AMD_AQLPROFILE_LEGACY_PM4_PACKET_SIZE / sizeof(packet_t);

  Profile(const util::AgentInfo* agent_info) : agent_info_(agent_info) {
    profile_ = {};
    profile_.agent = agent_info->dev_id;
    is_legacy_ = (strncmp(agent_info->name, "gfx8", 4) == 0);
  }
  virtual ~Profile() {
    hsa_memory_free(profile_.command_buffer.ptr);
    hsa_memory_free(profile_.output_buffer.ptr);
    free(const_cast<event_t*>(profile_.events));
    free(const_cast<parameter_t*>(profile_.parameters));
  }

  virtual void Insert(const profile_info_t& info) {
    info_vector_.push_back(info.rinfo);
  }

  hsa_status_t Finalize(pkt_vector_t& start_vector, pkt_vector_t& stop_vector) {
    hsa_status_t status = HSA_STATUS_SUCCESS;

    if (!info_vector_.empty()) {
      util::HsaRsrcFactory* rsrc = &util::HsaRsrcFactory::Instance();
      const pfn_t* api = rsrc->AqlProfileApi();
      packet_t start{};
      packet_t stop{};

      // Check the profile buffer sizes
      status = api->hsa_ven_amd_aqlprofile_start(&profile_, NULL);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "aqlprofile_start(NULL)");
      Allocate(rsrc);
      // Generate start/stop profiling packets
      status = api->hsa_ven_amd_aqlprofile_start(&profile_, &start);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "aqlprofile_start");
      status = api->hsa_ven_amd_aqlprofile_stop(&profile_, &stop);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "aqlprofile_stop");
      // Set completion signals
      hsa_signal_t dummy_signal{};
      dummy_signal.handle = 0;
      start.completion_signal = dummy_signal;
      hsa_signal_t post_signal;
      status = hsa_signal_create(1, 0, NULL, &post_signal);
      if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "hsa_signal_create");
      stop.completion_signal = post_signal;
      completion_signal_ = post_signal;

      if (is_legacy_) {
        const uint32_t start_index = start_vector.size();
        const uint32_t stop_index = stop_vector.size();

        start_vector.insert(start_vector.end(), LEGACY_SLOT_SIZE_PKT, packet_t{});
        stop_vector.insert(stop_vector.end(), LEGACY_SLOT_SIZE_PKT, packet_t{});
        status = api->hsa_ven_amd_aqlprofile_legacy_get_pm4(&start, reinterpret_cast<void*>(&start_vector[start_index]));
        if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "hsa_ven_amd_aqlprofile_legacy_get_pm4");
        status = api->hsa_ven_amd_aqlprofile_legacy_get_pm4(&stop, reinterpret_cast<void*>(&stop_vector[stop_index]));
        if (status != HSA_STATUS_SUCCESS) AQL_EXC_RAISING(status, "hsa_ven_amd_aqlprofile_legacy_get_pm4");
      } else {
        start_vector.push_back(start);
        stop_vector.push_back(stop);
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
    profile_.command_buffer.ptr = rsrc->AllocateSysMemory(agent_info_, profile_.command_buffer.size);
    profile_.output_buffer.ptr = rsrc->AllocateSysMemory(agent_info_, profile_.output_buffer.size);
    return (profile_.command_buffer.ptr && profile_.output_buffer.ptr) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
  }
};

class SqttProfile : public Profile {
  public:
  static const uint32_t output_buffer_size = 0x2000000;  // 32M

  SqttProfile(const util::AgentInfo* agent_info) : Profile(agent_info) {
    profile_.type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_SQTT;
  }

  void Insert(const profile_info_t& info) {
    Profile::Insert(info);
    for (unsigned j = 0; j < info.parameter_count; ++j) {
      Config<parameter_t>(&profile_).Insert(info.parameters[j]);
    }

    info.rinfo->data.result_bytes.size = output_buffer_size;
    if (info.rinfo->data.result_bytes.copy) {
      const uint32_t output_buffer_size64 = output_buffer_size / sizeof(uint64_t);
      info.rinfo->data.result_bytes.ptr = calloc(output_buffer_size64, sizeof(uint64_t));
      memset(info.rinfo->data.result_bytes.ptr, 0, output_buffer_size);
    }
  }

  hsa_status_t Allocate(util::HsaRsrcFactory* rsrc) {
    profile_.output_buffer.size = output_buffer_size;
    profile_.command_buffer.ptr = rsrc->AllocateSysMemory(agent_info_, profile_.command_buffer.size);
    profile_.output_buffer.ptr = rsrc->AllocateLocalMemory(agent_info_, profile_.output_buffer.size);
    return (profile_.command_buffer.ptr && profile_.output_buffer.ptr) ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR;
  }
};

}  // namespace rocprofiler

#endif  // SRC_CORE_PROFILE_H_
