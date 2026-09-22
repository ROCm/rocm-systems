#ifndef SRC_CORE_SESSION_SPM_H_
#define SRC_CORE_SESSION_SPM_H_

#include <map>
#include <vector>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include "hsa/hsa_ext_amd.h"
#include "src/core/hsa/packets/packets_generator.h"
#include "src/utils/exception.h"
#include "rocprofiler.h"


namespace rocprofiler {

namespace spm {


class SpmCounters {
 private:
  rocprofiler_buffer_id_t buffer_id_;
  rocprofiler_filter_id_t filter_id_;
  rocprofiler_spm_parameter_t* spmparameter_;
  rocprofiler_session_id_t session_id_;
  std::map<uint64_t, hsa_queue_t*> agent_queue_map_;
  typedef std::vector<std::pair<profiling_context_t*, hsa_ven_amd_aqlprofile_profile_t*>>
      profile_vector_t;

  profile_vector_t* profiles_;
  hsa_queue_t* queue_;
  hsa_agent_t defaultGpuNode_;
  hsa_agent_t defaultCpuNode_;
  hsa_agent_t preferredGpuNode_;
  hsa_signal_t start_signal_;
  hsa_signal_t stop_signal_;

 public:
  SpmCounters(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
              rocprofiler_spm_parameter_t* spmparameter, rocprofiler_session_id_t session_id);

  rocprofiler_status_t startSpm();
  rocprofiler_status_t stopSpm();


};  // class SpmCounters

}  // namespace spm

bool find_hsa_agent_cpu(uint64_t index, hsa_agent_t* agent);
bool find_hsa_agent_gpu(uint64_t index, hsa_agent_t* agent);

}  // namespace rocprofiler

#endif  // SRC_CORE_SESSION_SPM_H_