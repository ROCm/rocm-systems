/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_METADATA_PREFETCH_H_
#define ROCRTST_SUITES_FUNCTIONAL_METADATA_PREFETCH_H_
#include <pthread.h>
#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

class MetadataPrefetch : public TestBase {
    public:
    MetadataPrefetch();

     // @Brief: Destructor for test case of TestExample
     virtual ~MetadataPrefetch();

     // @Brief: Setup the environment for measurement
     virtual void SetUp();

     // @Brief: Core measurement execution
     virtual void Run();

     // @Brief: Clean up and retrieve the resource
     virtual void Close();

     // @Brief: Display  results
     virtual void DisplayResults() const;

     // @Brief: Display information about what this test does
     virtual void DisplayTestInfo(void);

    private:
     uint32_t RealIterationNum(void);

     double time_mean_;
     void *src_buffer_;
     void *dst_buffer_;
   };


#endif  // ROCRTST_SUITES_FUNCTIONAL_METADATA_PREFETCH_H_
