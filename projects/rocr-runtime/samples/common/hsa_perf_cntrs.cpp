#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <cassert>

#include <iostream>
#include <vector>
#include <string>

#include <stdlib.h>

#include "hsa.h"
#include "tools/inc/hsa_ext_profiler.h"
#include "tools/inc/amd_hsa_tools_interfaces.h"

#include "hsa_perf_cntrs.hpp"

using namespace std;

void PreDispatchCallback(const hsa_dispatch_callback_t* dispParam, void* usrArg) {
  assert((dispParam->pre_dispatch) && "Pre Dispatch Callback Param is Malformed");

  hsa_ext_tools_pmu_t* perfMgr = reinterpret_cast<hsa_ext_tools_pmu_t*>(usrArg);
  hsa_status_t status = hsa_ext_tools_pmu_begin(*perfMgr, dispParam->queue,
                                                dispParam->aql_translation_handle, true);
  assert((status == HSA_STATUS_SUCCESS) && "Error in beginning Perf Cntr Session");
}

void PostDispatchCallback(const hsa_dispatch_callback_t* dispParam, void* usrArg) {
  assert((!dispParam->pre_dispatch) && "Post Dispatch Callback Param is Malformed");

  hsa_ext_tools_pmu_t* perfMgr = reinterpret_cast<hsa_ext_tools_pmu_t*>(usrArg);
  hsa_status_t status = hsa_ext_tools_pmu_end(*perfMgr, dispParam->queue,
                                              dispParam->aql_translation_handle);
  assert((status == HSA_STATUS_SUCCESS) && "Error in endning Perf Cntr Session");
}

// Constructor of the class
RocrPerfCntrApp::RocrPerfCntrApp( ) : perfMgr_(NULL) {

}

// Destructor of the class. Ideally it should delete the
// PMU and its counters
RocrPerfCntrApp::~RocrPerfCntrApp( ) {

}

// Return the number of perf counters
uint32_t RocrPerfCntrApp::GetNumPerfCntrs( ) {
  return uint32_t(cntrList_.size());
}

// Return the handle of perf counter at specified index
CntrInfo* RocrPerfCntrApp::GetPerfCntr(uint32_t idx) {
  return cntrList_[idx];
}

// Print the various fields of Perf Cntrs being programmed
bool RocrPerfCntrApp::PrintCntrs( ) {

  CntrInfo *info;
  int size = uint32_t(cntrList_.size());
  for (int idx = 0; idx < size; idx++) {
    info = cntrList_[idx];
    std::cout << std::endl;
    std::cout << "Rocr Perf Cntr Id: " << info->cntrId << std::endl;
    std::cout << "Rocr Perf Cntr Name: " << info->cntrName << std::endl;
    std::cout << "Rocr Perf Cntr Blk Id: " << info->blkId << std::endl;
    std::cout << "Rocr Perf Cntr Value: " << info->cntrResult << std::endl;
    std::cout << "Rocr Perf Cntr Validation: " << info->cnfType << std::endl;
    std::cout << std::endl;
  }
  return true;
}

// Initialize the list of perf counters
// block id of kHsaAiCounterBlockSQ = 14 == 0x0E
hsa_status_t RocrPerfCntrApp::Init(hsa_agent_t agent) {

  // Initialize the list of Perf Cntrs
  // Add SQ counter for number of waves
  CntrInfo* info = NULL;
  cntrList_.reserve(23);
  
  char *cntrChoice = getenv("IOMMU");
  if (cntrChoice == NULL) {
    // Event for number of Waves
    info = new CntrInfo(0x4, "SQ_SQ_PERF_SEL_WAVES", NULL,
                                  0x0E, NULL, 0x00, 0xFFFFFFFF, CntrValCnf_Exact);
    cntrList_.push_back(info);
    
    // Event for number of Threads
    info = new CntrInfo(0xE, "SQ_SQ_PERF_SEL_ITEMS", NULL,
                                  0x0E, NULL, 0x00, 0xFFFFFFFF, CntrValCnf_Exact);
    cntrList_.push_back(info);
  
  } else {

    // Program to collect event number 4
    info = new CntrInfo(0x4, "Iommu_Cntr_4", NULL,
                        0x63, NULL, 0x00, 0xFFFFFFFF, CntrValCnf_None);
    cntrList_.push_back(info);
  
    // Program to collect event number 6
    info = new CntrInfo(0x6, "Iommu_Cntr_6", NULL,
                        0x63, NULL, 0x00, 0xFFFFFFFF, CntrValCnf_None);
    cntrList_.push_back(info);
  }
  

  // Create an instance of Perf Mgr
  hsa_status_t status;
  status = hsa_ext_tools_create_pmu(agent, &perfMgr_);
  assert((status == HSA_STATUS_SUCCESS) && "Error in creating Perf Cntr Mgr");

  // Process each counter from the list as necessary
  // each counter descriptor with its perf block handle
  // and create an instance of counter in that block
  uint32_t size = GetNumPerfCntrs();
  for (uint32_t idx = 0; idx < size; idx++) {
    info = GetPerfCntr(idx);
    
    // Obtain the handle of perf block
    if (info->blkHndl == NULL) {
      status = hsa_ext_tools_get_counter_block_by_id(perfMgr_, info->blkId, &info->blkHndl);
      assert((status == HSA_STATUS_SUCCESS) && "Error in getting Perf Cntr Blk Hndl");
    }

    // Create an instance of counter in the perf block
    status = hsa_ext_tools_create_counter(info->blkHndl, &info->cntrHndl);
    assert((status == HSA_STATUS_SUCCESS) && "Error in creating Perf Cntr in Perf Blk");

    // Update the Event Index property of counter
    uint32_t cntrProp = HSA_EXT_TOOLS_COUNTER_PARAMETER_EVENT_INDEX;
    status = hsa_ext_tools_set_counter_parameter(info->cntrHndl, cntrProp,
                                                 sizeof(uint32_t), (void*)&info->cntrId);
    assert((status == HSA_STATUS_SUCCESS) && "Error in updating Perf Cntr Property Event Index");

    // Enable the updated perf counter
    status = hsa_ext_tools_set_counter_enabled(info->cntrHndl, true);
    assert((status == HSA_STATUS_SUCCESS) && "Error in enabing Perf Cntr");
  }

  return status;
}

// Register Pre and Post dispatch callbacks
void RocrPerfCntrApp::RegisterCallbacks(hsa_queue_t *queue){
  
  hsa_status_t status;
  status = hsa_ext_tools_set_callback_functions(queue, PreDispatchCallback, PostDispatchCallback);
  assert((status == HSA_STATUS_SUCCESS) && "Error in registering Pre & Post Dispatch Callbacks");
  status = hsa_ext_tools_set_callback_arguments(queue, &perfMgr_, &perfMgr_);
  assert((status == HSA_STATUS_SUCCESS) && "Error in registering Pre & Post Dispatch Callback Params");
  return;
}

// Wait for perf counter collection to complete
hsa_status_t RocrPerfCntrApp::Wait() {

  hsa_status_t status;
  status = hsa_ext_tools_pmu_wait_for_completion(perfMgr_, 5000);
  assert((status == HSA_STATUS_SUCCESS) && "Error in Waiting for Perf Cntr Completion");
  return status;
}

// Validate perf counter values
hsa_status_t RocrPerfCntrApp::Validate() {

  // Retrieve the results of the different Perf Cntrs
  // and validate them as configured
  CntrInfo* info = NULL;
  hsa_status_t status = HSA_STATUS_SUCCESS;
  uint32_t size = GetNumPerfCntrs();
  for (uint32_t idx = 0; idx < size; idx++) {
    info = GetPerfCntr(idx);
    status = hsa_ext_tools_get_counter_result(info->cntrHndl, &info->cntrResult);
    std::cout << "Value of Perf Cntr is: " << info->cntrResult << std::endl;
  }

  return status;
}
