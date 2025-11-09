/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/

#ifndef ROCRTST_SUITES_IMAGES_MIPMAP_2DARRAY_H_
#define ROCRTST_SUITES_IMAGES_MIPMAP_2DARRAY_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_image.h"
#include "hsa/hsa_amd_mipmap.h"
#include "suites/test_common/test_base.h"

class Mipmap2DArrayTest : public TestBase {
public:
   Mipmap2DArrayTest();

   // @Brief: Destructor for test case of Mipmap2DArrayTest
   virtual ~Mipmap2DArrayTest();

   // @Brief: Setup the environment for measurement
   virtual void SetUp();

   // @Brief: Core measurement execution
   virtual void Run();

   // @Brief: Clean up and retrieve the resource
   virtual void Close();

   // @Brief: Display results
   virtual void DisplayResults() const;

   // @Brief: Display information about what this test does
   virtual void DisplayTestInfo(void);

   // Geometry-specific test methods
   void MipmapCreateDestroy2DArrayTest(void);
   void MipmapGetLevel2DArrayTest(void);

   void MipmapDataIntegrity2DTest(void);
   void MipmapSampling2DTest(void);
   void MipmapErrorHandling2DTest(void);

private:
   hsa_ext_image_t test_image_;
   hsa_ext_image_descriptor_t mipmap_desc_;
   hsa_ext_image_format_t image_format_;
   uint32_t num_mipmap_levels_;
   bool image_ext_supported;
};

#endif  // ROCRTST_SUITES_IMAGES_MIPMAP_ARRAY_H_
