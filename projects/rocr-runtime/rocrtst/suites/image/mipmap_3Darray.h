/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 *
 */

#ifndef ROCRTST_SUITES_IMAGES_MIPMAP_3DARRAY_H_
#define ROCRTST_SUITES_IMAGES_MIPMAP_3DARRAY_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_image.h"
#include "hsa/hsa_amd_mipmap.h"
#include "suites/test_common/test_base.h"

class Mipmap3DArrayTest : public TestBase {
public:
   Mipmap3DArrayTest();

   // @Brief: Destructor for test case of Mipmap3DArrayTest
   virtual ~Mipmap3DArrayTest();

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
   void MipmapCreateDestroy3DArrayTest(void);
   void MipmapGetLevel3DArrayTest(void);

private:
   hsa_ext_image_t test_image_;
   hsa_ext_image_descriptor_t mipmap_desc_;
   hsa_ext_image_format_t image_format_;
   uint32_t num_mipmap_levels_;
   bool image_ext_supported;
};

#endif  // ROCRTST_SUITES_IMAGES_MIPMAP_ARRAY_H_
