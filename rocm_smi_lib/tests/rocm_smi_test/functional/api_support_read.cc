/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2019, Advanced Micro Devices, Inc.
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
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <stdint.h>
#include <stddef.h>

#include <iostream>
#include <string>

#include "gtest/gtest.h"
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi_test/functional/api_support_read.h"
#include "rocm_smi_test/test_common.h"
#include "rocm_smi_test/test_utils.h"

TestAPISupportRead::TestAPISupportRead() : TestBase() {
  set_title("RSMI API Support Read Test");
  set_description("This test verifies that the supported APIs are corretly "
                                                               "identified.");
}

TestAPISupportRead::~TestAPISupportRead(void) {
}

void TestAPISupportRead::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestAPISupportRead::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TestAPISupportRead::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestAPISupportRead::Close() {
  // This will close handles opened within rsmitst utility calls and call
  // rsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestAPISupportRead::Run(void) {
  rsmi_status_t err;

  TestBase::Run();
  if (setup_failed_) {
    IF_VERB(STANDARD) {
      std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    }
    return;
  }

  rsmi_func_id_iter_handle_t iter_handle, var_iter, sub_var_iter;
  rsmi_func_id_value_t value;

  for (uint32_t x = 0; x < num_iterations(); ++x) {
    for (uint32_t i = 0; i < num_monitor_devs(); ++i) {
      IF_VERB(STANDARD) {
        PrintDeviceHeader(i);
        std::cout << "Supported RSMI Functions:" << std::endl;
        std::cout << "\tVariants (Monitors)" << std::endl;
      }
      err = rsmi_dev_supported_func_iterator_open(i, &iter_handle);
      CHK_ERR_ASRT(err)

      while (1) {
        err = rsmi_func_iter_value_get(iter_handle, &value);
        CHK_ERR_ASRT(err)
        IF_VERB(STANDARD) {
          std::cout << "Function Name: " << value.name << std::endl;
        }
        err = rsmi_dev_supported_variant_iterator_open(iter_handle, &var_iter);
        if (err != RSMI_STATUS_NO_DATA) {
          CHK_ERR_ASRT(err)
          IF_VERB(STANDARD) {
            std::cout << "\tVariants/Monitors: ";
          }
          while (1) {
            err = rsmi_func_iter_value_get(var_iter, &value);
            CHK_ERR_ASRT(err)
            IF_VERB(STANDARD) {
              if (value.id == RSMI_DEFAULT_VARIANT) {
                std::cout << "Default Variant ";
              } else {
                std::cout << value.id;
              }
              std::cout << " (";
            }
            err =
              rsmi_dev_supported_variant_iterator_open(var_iter, &sub_var_iter);
            if (err != RSMI_STATUS_NO_DATA) {
              CHK_ERR_ASRT(err)

              while (1) {
                err = rsmi_func_iter_value_get(sub_var_iter, &value);
                CHK_ERR_ASRT(err)
                IF_VERB(STANDARD) {
                  std::cout << value.id << ", ";
                }
                err = rsmi_func_iter_next(sub_var_iter);

                if (err == RSMI_STATUS_NO_DATA) {
                  break;
                }
                CHK_ERR_ASRT(err)
              }
              err = rsmi_dev_supported_func_iterator_close(&sub_var_iter);
              CHK_ERR_ASRT(err)
            }

            IF_VERB(STANDARD) {
              std::cout << "), ";
            }
            err = rsmi_func_iter_next(var_iter);

            if (err == RSMI_STATUS_NO_DATA) {
              break;
            }
            CHK_ERR_ASRT(err)
          }
          IF_VERB(STANDARD) {
            std::cout << std::endl;
          }
          err = rsmi_dev_supported_func_iterator_close(&var_iter);
          CHK_ERR_ASRT(err)
        }

        err = rsmi_func_iter_next(iter_handle);

        if (err == RSMI_STATUS_NO_DATA) {
          break;
        }
        CHK_ERR_ASRT(err)

    //  err = rsmi_dev_supported_variant_iterator_open(iter_handle, &var_iter);
    //
      }
      err = rsmi_dev_supported_func_iterator_close(&iter_handle);
      CHK_ERR_ASRT(err)
    }
  }
}
