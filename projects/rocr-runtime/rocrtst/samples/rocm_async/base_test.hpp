
#ifndef ROCM_ASYNC_BW_BASE_TEST_H_
#define ROCM_ASYNC_BW_BASE_TEST_H_

#include "hsa/hsa.h"
#include <iostream>
#include <string>
#include <vector>

using namespace std;

// @Brief: An interface for tests to do some basic things,

class BaseTest {

 public:

  BaseTest(size_t num = 10);

  virtual ~BaseTest();

  // @Brief: Allows setup proceedures to be completed
  // before running the benchmark test case
  virtual void SetUp() = 0;

  // @Brief: Launches the proceedures of test scenario
  virtual void Run() = 0;

  // @Brief: Allows clean up proceedures to be invoked
  virtual void Close() = 0;

  // @Brief: Display the results
  virtual void Display() const = 0;

  // @Brief: Set number of iterations to run
  void set_num_iteration(size_t num) {
    num_iteration_ = num;
    return;
  }

  // @Brief: Pre-declare some variables for deriviation, the
  // derived class may declare more if needed
 protected:

  // @Brief: Real iteration number
  uint64_t num_iteration_;

  // @Brief: Status code
  hsa_status_t err_;
};

#endif  //  ROCM_ASYNC_BW_BASE_TEST_H_
