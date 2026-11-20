/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_FUNCTIONAL_GL2_CACHE_TEST_H_
#define ROCRTST_SUITES_FUNCTIONAL_GL2_CACHE_TEST_H_

#include <functional>
#include <vector>
#include "suites/test_common/test_case_template.h"
#include "suites/test_common/test_base.h"

class GL2CacheTest : public TestBase {
 public:
    GL2CacheTest();
    virtual ~GL2CacheTest();

    virtual void SetUp();
    virtual void Run();
    virtual void Close();
    virtual void DisplayResults() const;
    virtual void DisplayTestInfo(void);

    void SetSubtest(std::function<void()> fn) { subtest_ = fn; }

    void QueryMaxPersistingCacheSize();
    void QueryRequestPersistingCacheSize();
    void SetPersistingCacheSize();
    void SetAndReadBackPersistingCacheSize();
    void NegativeInvalidSize();
    void NegativeCPUAgent();
    void QueryStatusConsistency();

 private:
    std::vector<std::string> resultList_;
    std::function<void()> subtest_;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_GL2_CACHE_TEST_H_
