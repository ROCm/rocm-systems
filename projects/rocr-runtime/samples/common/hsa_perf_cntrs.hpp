#ifndef ROCR_PERF_CNTR_APP_H_
#define ROCR_PERF_CNTR_APP_H_

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <string>

#include "hsa.h"
#include "tools/inc/hsa_ext_profiler.h"

typedef enum CntrValCnfType {
  
  ///< no counter value validation should be performed
  CntrValCnf_None,

  ///< counter value should be an exact match to expectedResult
  CntrValCnf_Exact,

  ///< counter value should be greater than expectedResult
  CntrValCnf_GreaterThan,
  
  ///< counter value should be less than expectedResult
  CntrValCnf_LessThan

} CntrValCnfType;

/// Struct used to encapsulate Counter Info
typedef struct CntrInfo {
  
  ///< Id of counter in hardware block
  uint32_t cntrId;
  
  ///< Name of counter
  char cntrName[72];
  
  ///< Handle of perf counter
  hsa_ext_tools_counter_t cntrHndl;
  
  ///< Id of hardware block containing the counter
  uint32_t blkId;
  
  ///< Handle of counter block
  hsa_ext_tools_counter_block_t blkHndl;
  
  ///< Expected value of perf counte
  uint64_t  expectedResult;

  ///< Value of perf counter expected
  uint64_t cntrResult;
  
  ///< Type of validation upon completion of dispatch
  CntrValCnfType cnfType;

  CntrInfo(uint32_t cntrId, char* cntrName, void* cntrHndl,
           uint32_t blkId, void* blkHndl,
           uint64_t expResult, uint64_t result, CntrValCnfType cnfType) {
    this->cntrId = cntrId;
    this->cntrHndl = cntrHndl;
    this->blkId = blkId;
    this->blkHndl = blkHndl;
    this->expectedResult = expResult;
    this->cntrResult = result;
    this->cnfType = cnfType;
    memcpy(this->cntrName, cntrName, strlen(cntrName));
  }
  
} CntrInfo;

class RocrPerfCntrApp {

 public:

  // Constructor of the class. Will initialize the list of perf counters
  // that will be used to program the device
  RocrPerfCntrApp( );

  // Destructor of the class
  ~RocrPerfCntrApp( );

  // Return the number of perf counters
  uint32_t GetNumPerfCntrs();

  // Return the handle of perf counter at specified index
  CntrInfo* GetPerfCntr(uint32_t idx);

  // Print the list of perf counters
  bool PrintCntrs();

  // Initialize the list of perf counters
  hsa_status_t Init(hsa_agent_t agent);

  // Register Pre and Post dispatch callbacks
  void RegisterCallbacks(hsa_queue_t *queue);

  // Wait for perf counter collection to complete
  hsa_status_t Wait();

  // Validate perf counter values
  hsa_status_t Validate();
 
 private:
 
  // Number of queues to create
  std::vector<CntrInfo *> cntrList_;

  // Handle of Perf Cntr Manager
  hsa_ext_tools_pmu_t perfMgr_;
};

#endif  //  ROCR_PERF_CNTR_APP_H_
