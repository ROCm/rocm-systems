/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

/////////////////////////////////////////////////////////////////////////////

#include <CL/cl.h>

#ifdef _WIN32
#include <windows.h>

#include "Window.h"
typedef HMODULE ModuleHandle;
#endif

/////////////////////////////////////////////////////////////////////////////

#ifdef __linux__
#include <dlfcn.h>
typedef void* ModuleHandle;
#endif

/////////////////////////////////////////////////////////////////////////////

#include "BaseTestImp.h"
#include "Module.h"
#include "OCLLog.h"
#include "OCLTest.h"
#include "OCLTestImp.h"
#include "OCLTestList.h"
#include "OCLWrapper.h"
#include "Timer.h"
#include "Worker.h"
#include "getopt.h"
#include "oclsysinfo.h"
#include "pfm.h"

//! Including OCLutilities Thread utility
#include "OCL/Thread.h"

//! Lock that needs to be obtained to access the global
//! module variable
static OCLutil::Lock moduleLock;

#include <assert.h>
#include <stdio.h>

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

/////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32
static LONG WINAPI xFilter(LPEXCEPTION_POINTERS xEP);
void serviceStubCall();
#endif

#define MAX_DEVICES 16
#undef CHECK_RESULT
#define CHECK_RESULT(test, msg)                                                                    \
  if ((test)) {                                                                                    \
    printf("\n%s\n", msg);                                                                         \
    exit(1);                                                                                       \
  }

//! Declaration of a function that find devices of a specific type for the
//! chosen platform
int findAdapters(unsigned int platformIdx, bool useCPU, cl_platform_id*);

//! class App that is used to run the tests on the system
class App {
 public:
  static bool m_reRunFailed;
  static const char* m_svcMsg;
  //! Constructor for App
  App(unsigned int platform)
      : m_list(false),
        m_console(true),
        m_useCPU(false),
        m_dump(false),
        m_perflab(false),
        m_noSysInfoPrint(false),
        m_numItr(1),
        mp_testOrder(NULL),
        m_rndOrder(false),
        m_spawned(0),
        m_threads(1),
        m_runthread(0),
        m_width(512),
        m_height(512),
        m_window(0),
        m_platform(platform) {
    // initialize OCLWrapper reference

    // m_workers = Set of worker objects that are used to run a subtest from a
    // module
    for (unsigned int i = 0; i < 256; i++) m_workers[i] = 0;

    // Setting the number of devices
    /*
     * Force caltst to use 1 thread at a time in Windows
     * only contextual calls are thread safe currently
     */
    m_numDevices = findAdapters(m_platform, m_useCPU, &mpform_id);
    m_wrapper = new OCLWrapper(mpform_id);
    // m_numDevices = 1;

    // Report structure used to store the results of the tests
#if 0
            testReport = (Report **)malloc(sizeof(Report *) * m_numDevices);
            for(unsigned int i = 0; i < m_numDevices; i++)
            {
                testReport[i] = new Report;
            }
#else
    testReport = (Report**)malloc(sizeof(Report*));
    testReport[0] = new Report;
#endif
  }

  //! Destructor for App
  ~App() {
    // Deleting the Worker objects
    for (unsigned int i = 0; i < 256; i++) {
      if (m_workers[i]) {
        delete m_workers[i];
        m_workers[i] = 0;
      }
    }

    // Deleting the report structures
    // for(unsigned int i = 0; i < m_numDevices; i++)
    for (unsigned int i = 0; i < 1; i++) {
      delete testReport[i];
    }
    free(testReport);
    m_wrapper->clUnloadPlatformAMD(mpform_id);

    delete m_wrapper;
  }

  //! Function used to create a worker object corresponding to a subtest in a
  //! module
  void SetWorker(unsigned int index, OCLWrapper* wrapper, Module* module, TestMethod run,
                 unsigned int id, unsigned int subtest, unsigned int test, bool dump, bool view,
                 bool useCPU, void* window, unsigned int x, unsigned int y, bool perflab,
                 unsigned int deviceId, unsigned int platform) {
    if (index >= 256) return;

    if (m_workers[index]) delete m_workers[index];

    m_workers[index] = new Worker(wrapper, module, run, id, subtest, test, dump, view, useCPU,
                                  window, x, y, perflab, deviceId, platform);

    assert(m_workers[index] != 0);
    // oclTestLog(OCLTEST_LOG_ALWAYS, "Worker Device Id = %d\n",
    // m_workers[index]->getDeviceId());
  }

  //! Function to return the 'index'th m_workers
  Worker* GetWorker(unsigned int index) {
    if (index >= 256) return 0;

    return m_workers[index];
  }

  //! Create a thread to run the subtest
  void AddThread(unsigned int workerindex, unsigned int usage) {
    Worker* worker = GetWorker(workerindex);
    if (worker == 0) {
      return;
    }

    // usage = Whether to use threads or not
    if (usage != 0) {
      // Creating a thread
      // getTestMethod = runSubTest here
      // which takes a Worker object as an argument
      m_pool[workerindex].create(worker->getTestMethod(), (void*)(worker));
      m_spawned++;
    } else {
      // Same as above without using threads
      TestMethod run = worker->getTestMethod();
      if (run) {
        run(worker);
        UpdateTestReport(workerindex, worker->getResult());
      }
    }
    return;
  }

  //! Function which waits for all threads to execute and also updates the
  //! report
  void WaitAllThreads() {
    for (unsigned int w = 0; w < m_spawned; w++) {
      m_pool[w].join();
      UpdateTestReport(w, m_workers[w]->getResult());
    }
    m_spawned = 0;
  }

  //! Function to add a worker thread so as to run a subtest of a module
  //! @param run = runSubtest function
  //! @param index = index of the module in m_modules
  //! @param subtest = the subtest number to run
  //! @param usage = whether to use threads or not
  //! @param test = The test in the module to be executed
  void AddWorkerThread(unsigned int index, unsigned int subtest, unsigned int test,
                       unsigned int usage, TestMethod run) {
    if (m_spawned > m_threads) {
      WaitAllThreads();
    }

    // Creating a worker thread for each device
#if 0
            for(unsigned int i = 0; i < m_numDevices; i++)
            {
                SetWorker(i,
                          m_wrapper,
                          &m_modules[index],
                          run,
                          m_spawned,
                          subtest,
                          test,
                          m_dump,
                          !m_console,
                          m_useCPU,
                          m_window,
                          m_width,
                          m_height,
                          m_perflab,
                          i,
                          m_platform);
            }
#else
    for (unsigned int i = 0; i < 1; i++) {
      SetWorker(i, m_wrapper, &m_modules[index], run, m_spawned, subtest, test, m_dump, !m_console,
                m_useCPU, m_window, m_width, m_height, m_perflab, m_deviceId, m_platform);
    }
#endif

    // Creating and executing a thread for each device
    // for(unsigned int i = 0; i < m_numDevices; i++)
    for (unsigned int i = 0; i < 1; i++) {
      AddThread(i, usage);
    }
  }

  void printOCLinfo(void);

  //! Function to process the commandline arguments
  void CommandLine(unsigned int argc, char** argv);

  //! Function to scan for the different tests in the module
  void ScanForTests();

  //! Function to run all the specified tests
  void RunAllTests();

  //! Free memory
  void CleanUp();

  //! Function to set the order in which test are executed.
  void SetTestRunOrder(int);

  //! Function to print the test order
  void PrintTestOrder(int);

  //! Function to get the number of iterations.
  int GetNumItr(void) { return m_numItr; }

 private:
  typedef std::vector<unsigned int> TestIndexList;
  typedef std::vector<std::string> StringList;

  void AddToList(StringList& strlist, const char* str);
  void LoadList(StringList& strlist, const char* filename);

  bool TestInList(StringList& strlist, const char* testname);

  //! Array storing the report for each device
  Report** testReport;

  //! Function to update the result of each device
  void UpdateTestReport(int index, TestResult* result) {
    if (result != NULL) {
      if (result->passed) {
        if (testReport[index]->max->value < result->value) {
          testReport[index]->max->value = result->value;
          testReport[index]->max->resultString = result->resultString;
        }
        if (testReport[index]->min->value > result->value) {
          testReport[index]->min->value = result->value;
          testReport[index]->min->resultString = result->resultString;
        }
      } else {
        testReport[index]->numFailedTests++;
        testReport[index]->success = false;
      }
    } else {
      testReport[index]->numFailedTests++;
      testReport[index]->success = false;
    }
  }

  //! Functions used to find the range of the tests to be run
  void GetTestIndexList(TestIndexList& testIndices, StringList& testList,
                        const char* szModuleTestname, int maxIndex);
  void PruneTestIndexList(TestIndexList& testIndices, TestIndexList& avoidIndices,
                          TestIndexList& erasedIndices);

  StringList m_paths;
  StringList m_tests;
  StringList m_avoid;
  std::vector<Module> m_modules;
  bool m_list;
  bool m_console;
  bool m_useCPU;
  bool m_dump;
  bool m_perflab;
  bool m_noSysInfoPrint;
  int m_numItr;
  int* mp_testOrder;
  bool m_rndOrder;

  //! m_pool = Various threads created to execute tests on multiple devices
  OCLutil::Thread m_pool[256];

  Worker* m_workers[256];

  //! Number of threads spawned
  unsigned int m_spawned;

  //! Upper limit on the number of threads that can be spawned
  unsigned int m_threads;
  unsigned int m_runthread;
  unsigned int m_width;
  unsigned int m_height;
  void* m_window;

  //! which index/platform id from the platforms vector returned by
  //! cl::Platform::get we should run on
  unsigned int m_platform;
  cl_platform_id mpform_id;

  //! Number of devices on the system
  unsigned int m_numDevices;
  //
  //! Device ID to use on the system
  unsigned int m_deviceId;

  // OCLWrapper reference
  OCLWrapper* m_wrapper;
};

void App::printOCLinfo(void) {
  std::string calinfo;
  if (!m_noSysInfoPrint) {
    oclSysInfo(calinfo, m_useCPU, m_deviceId, m_platform);
    oclTestLog(OCLTEST_LOG_ALWAYS, calinfo.c_str());
  }
}

/*-----------------------------------------------------
Function to randomize the order in which tests are executed
-------------------------------------------------------*/
#ifdef _WIN32
#include <time.h>
#endif
// void App::SetTestRunOrder(int test_count)
void App::SetTestRunOrder(int mod_index) {
  assert(mp_testOrder != NULL);
  unsigned int test_count = m_modules[mod_index].get_count();

  StringList uniqueTests;
  for (unsigned int i = 0; i < m_tests.size(); ++i) {
    // see if the tests are being run using indices
    size_t nFirstBracket = m_tests[i].find("[");
    // set the test name
    std::string szTestName = m_tests[i];

    // order of execution is set based on base name so get the base name
    if (nFirstBracket != std::string::npos) szTestName = szTestName.substr(0, nFirstBracket);

    bool bTestExists = false;
    for (unsigned int j = 0; j < uniqueTests.size(); ++j) {
      if (strcmp(szTestName.c_str(), uniqueTests[j].c_str()) == 0) {
        bTestExists = true;
        break;
      }
    }

    if (!bTestExists) {
      AddToList(uniqueTests, szTestName.c_str());
    }
  }

  for (unsigned int i = 0; i < test_count && i < uniqueTests.size(); i++) {
    for (unsigned int j = 0; j < test_count; j++) {
      unsigned int index = i;
      // add all the prev test indices
      for (int k = 0; k < mod_index; k++) index += m_modules[k].get_count();

      std::string szTestName = uniqueTests[index];

      if (strcmp(szTestName.c_str(), m_modules[mod_index].get_name(j)) == 0) {
        mp_testOrder[i] = j;
        break;
      }
    }
  }

  if (m_rndOrder) {
    srand((unsigned int)time(NULL));
    for (unsigned int i = 0; i < test_count; i++) {
      // find two random indices
      int index1 = (int)((float)test_count * (rand() / (RAND_MAX + 1.0)));
      int index2 = (int)((float)test_count * (rand() / (RAND_MAX + 1.0)));
      // swap the data
      int tmp = mp_testOrder[index1];
      mp_testOrder[index1] = mp_testOrder[index2];
      mp_testOrder[index2] = tmp;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////

// Process device string. Returns true if there is a primary ATI Radeon device
// adapter, false otherwise
static bool procDevString(const char* devString) {
  // Search for the string "Radeon" inside the device string
  if (strstr(devString, "Radeon") || strstr(devString, "R600") || strstr(devString, "RV630") ||
      strstr(devString, "RV670") ||
      (strstr(devString, "Stream") && strstr(devString, "Processor"))) {
    // Ignore if the device is a secondary device, i.e., not an adapter
    if (strstr(devString, "Secondary")) {
      return false;
    }
  } else {
    return false;
  }

  return true;
}

//!
//! Function to find the total number of adapters on the system
//!
int findAdapters(unsigned int platformIdx, bool useCPU, cl_platform_id* mpform) {
  unsigned int numOfAdapters = 0;
  cl_int error = 0;
  cl_uint numPlatforms = 0;

  error = clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT((error != CL_SUCCESS), "clGetPlatformIDs failed");

  CHECK_RESULT((platformIdx >= numPlatforms), "Invalid platform");

  cl_platform_id* platforms = new cl_platform_id[numPlatforms];
  error = clGetPlatformIDs(numPlatforms, platforms, NULL);
  CHECK_RESULT(error != CL_SUCCESS, "clGetPlatformIDs failed");

  cl_platform_id platform = 0;

  platform = platforms[platformIdx];

  delete[] platforms;

  cl_device_type devType = CL_DEVICE_TYPE_GPU;
  if (useCPU) devType = CL_DEVICE_TYPE_CPU;
  error = clGetDeviceIDs(platform, devType, 0, 0, &numOfAdapters);
  CHECK_RESULT((error != CL_SUCCESS), "clGetDeviceIDs failed");
  if (mpform) {
    (*mpform) = platform;
  }

  return (int)numOfAdapters;
}

int calibrate(OCLTest* test) {
  int n = 1;

#if 0
    while (1)
    {
        double timer = run(test, n);
        if (timer > 2.)
        {
            break;
        }
        n *= 2;
    }
#endif

  return n;
}

void* dummyThread(void* argv) {
  unsigned int counter = 0;
  while (counter < 1000000) counter++;

  return argv;
}

//! Function used to run the test specified
//! It would look something like OCLPerfInputspeed[0]
double run(OCLTest* test, int passes) {
  CPerfCounter counter;

  counter.Reset();
  counter.Start();
  int i;
  for (i = 0; i < passes; i++) {
    test->run();
  }
  counter.Stop();
  double timer = counter.GetElapsedTime();
  counter.Reset();

  return timer;
}

//! Function to display the result after a test is finished
//! It also stores the result in a TestResult object
void report(Worker* w, const char* testname, int testnum, unsigned int crc, const char* errorMsg,
            float timer, TestResult* tr, const char* testDesc) {
  unsigned int thread = w->getId();
  bool perflab = w->getPerflab();
  unsigned int deviceId = w->getDeviceId();

  char tmpUnits[256];
  if (perflab) {
    oclTestLog(OCLTEST_LOG_ALWAYS, "%10.3f\n", timer);
  } else {
    const char* passedOrFailed[] = {"FAILED", "PASSED"};

    // char teststring[256];
    // sprintf(teststring, "%s[%d]", testname, testnum);
    // sprintf(tmpUnits, "Device[%d]:\t%-32s:\t%s\n", deviceId, teststring,
    // ((tr->passed) ? passedOrFailed[1] : passedOrFailed[0]));
    // If crc is not 0 or errorMsg is not empty, print the full stats
    if ((crc != 0) || (errorMsg && (errorMsg[0] != '\0'))) {
      sprintf(tmpUnits, "%s %s: %s[%d] T[%1d] [%3d], %10.3f %-20s (chksum 0x%08x)\n", testDesc,
              ((tr->passed) ? passedOrFailed[1] : passedOrFailed[0]),
              w->isCPUEnabled() ? "CPU" : "GPU", deviceId, thread, testnum, timer, errorMsg, crc);
    } else {
      sprintf(tmpUnits, "%s %s: %s[%d] T[%1d] [%3d], %10.3f\n", testDesc,
              ((tr->passed) ? passedOrFailed[1] : passedOrFailed[0]),
              w->isCPUEnabled() ? "CPU" : "GPU", deviceId, thread, testnum, timer);
    }

    oclTestLog(OCLTEST_LOG_ALWAYS, tmpUnits);

    tr->value = timer;
    tr->resultString.assign(tmpUnits);

    if (App::m_svcMsg && !tr->passed) {
      char escaped[2 * sizeof(tmpUnits)];

      char* ptr = escaped;
      for (int i = 0; tmpUnits[i] != '\0'; ++i) {
        switch (tmpUnits[i]) {
          case '\n':
            *ptr++ = '|';
            *ptr++ = 'n';
            break;
          case '\r':
            *ptr++ = '|';
            *ptr++ = 'r';
            break;
          case '\'':
          case '|':
          case ']':
          case '[':
            *ptr++ = '|';
          default:
            *ptr++ = tmpUnits[i];
        }
      }
      *ptr = '\0';

      oclTestLog(OCLTEST_LOG_ALWAYS,
                 "##%s[testFailed name='%s.%s.%d' message='FAILED' "
                 "details='%s']\n",
                 App::m_svcMsg, w->getModule()->get_libname(), testname, testnum, escaped);
    }
  }
}

//! Thread Entry point
void* runSubtest(void* worker) {
  char units[256];
  double conversion;
  unsigned int crc = 0;
  bool second_run = false;

  // Getting the worker object that is running in this thread
  Worker* w = (Worker*)worker;

  if (w == 0) return NULL;

  unsigned int test = w->getTestIndex();
  unsigned int subtest = w->getSubTest();
  unsigned int deviceId = w->getDeviceId();
  unsigned int platformIndex = w->getPlatformID();
  TestResult* result = w->getResult();

RERUN_TEST:
  // Acquiring lock on the 'module' object common to all threads
  moduleLock.lock();
  Module* m = w->getModule();
  if (m == 0 || m->create_test == 0) return NULL;
  // If we can, used the cached version,
  // otherwise create the test.
  OCLTest* pt = (m->cached_test ? m->cached_test[subtest] : NULL);
  if (!pt) {
    pt = m->create_test(subtest);
    if (pt->cache_test() && m->cached_test) {
      m->cached_test[subtest] = pt;
    }
  }
  pt->clearError();
  OCLTestImp* tmp = pt->toOCLTestImp();
  if (tmp) {
    tmp->setOCLWrapper(w->getOCLWrapper());
  }
  std::string subtestName = m->get_name(subtest);
  moduleLock.unlock();

  if (pt == 0) return NULL;

  pt->resetDescString();
  if (App::m_svcMsg) {
    oclTestLog(OCLTEST_LOG_ALWAYS,
               "##%s[testStarted name='%s.%s.%d' "
               "captureStandardOutput='true']\n",
               App::m_svcMsg, m->get_libname(), subtestName.c_str(), test);
  }
  // setting the type to CPU.
  if (w->isCPUEnabled()) {
    pt->useCPU();
  }
  // Setting the device according to the worker thread
  pt->setDeviceId(w->getDeviceId());
  pt->setPlatformIndex(w->getPlatformID());
  // Opening the 'test'th subtest of 'pt'
  pt->open(test, units, conversion, deviceId);
  pt->clearPerfInfo();

  char buffer[256];
  sprintf(buffer, "%s[%3d]", subtestName.c_str(), test);
  oclTestLog(OCLTEST_LOG_ALWAYS, "%-32s", buffer);

  if (pt->hasErrorOccured()) {
    result->passed = false;
    report(w, subtestName.c_str(), test, crc, pt->getErrorMsg(), pt->getPerfInfo(), result,
           pt->testDescString.c_str());
  } else {
    unsigned int n = calibrate(pt);
    double timer = run(pt, n);
    crc = pt->close();

    if (pt->hasErrorOccured()) {
      // run second time if the test fails the first time.
      if (!second_run && App::m_reRunFailed && !App::m_svcMsg) {
        second_run = true;

        // Destroying a test object
        moduleLock.lock();
        if (!pt->cache_test()) {
          m->destroy_test(pt);
        }
        moduleLock.unlock();

        pt->clearError();
        goto RERUN_TEST;
      }
    }
    result->passed = !pt->hasErrorOccured();
    /// print conditional pass if it is passes the second time.
    if (second_run && result->passed) {
      report(w, subtestName.c_str(), test, crc, "Conditional PASS", pt->getPerfInfo(), result,
             pt->testDescString.c_str());
    } else {
      report(w, subtestName.c_str(), test, crc, pt->getErrorMsg(), pt->getPerfInfo(), result,
             pt->testDescString.c_str());
    }
  }
  if (App::m_svcMsg) {
    oclTestLog(OCLTEST_LOG_ALWAYS, "##%s[testFinished name='%s.%s.%d']\n", App::m_svcMsg,
               m->get_libname(), subtestName.c_str(), test);
  }

  // Make sure we clear the error after we report that there was an error.
  pt->clearError();

  // Destroying a test object
  moduleLock.lock();
  if (!pt->cache_test()) {
    m->destroy_test(pt);
  }
  moduleLock.unlock();
  return NULL;
}

void App::PrintTestOrder(int mod_index) {
  oclTestLog(OCLTEST_LOG_ALWAYS, "Module: %s (%d tests)\n", m_modules[mod_index].name.c_str(),
             m_modules[mod_index].get_count());

  for (unsigned int j = 0; j < m_modules[mod_index].get_count(); j++) {
    oclTestLog(OCLTEST_LOG_ALWAYS, "%s\n", m_modules[mod_index].get_name(mp_testOrder[j]));
  }
}

//! Function that runs all the tests specified in the command-line
void App::RunAllTests() {
#ifdef _WIN32

  if (!m_console) m_window = new Window("Test", 100, 100, m_width, m_height, 0);
#endif

  //
  //  Add all tests to run list if none specified
  //
  if (m_tests.size() < 1) {
    for (unsigned int i = 0; i < m_modules.size(); i++) {
      for (unsigned int j = 0; j < m_modules[i].get_count(); j++) {
        AddToList(m_tests, m_modules[i].get_name(j));
      }
    }
  }

  unsigned int num_passes = 0;
  unsigned int num_failures = 0;


  //
  //  Run each test
  //
  for (unsigned int i = 0; i < m_modules.size(); i++) {
    oclTestLog(OCLTEST_LOG_ALWAYS, "\n-------------------------------------------------\n");
    oclTestLog(OCLTEST_LOG_ALWAYS, "The OpenCL Testing Module %s Version = %d \n",
               m_modules[i].get_libname(), m_modules[i].get_version());
    oclTestLog(OCLTEST_LOG_ALWAYS, "-------------------------------------------------\n");

    if (App::m_svcMsg) {
      oclTestLog(OCLTEST_LOG_ALWAYS, "##%s[testSuiteStarted name='ocltst %s']\n", App::m_svcMsg,
                 m_modules[i].get_libname());
    }

    // array to keep track of order of test execution.
    int test_count = m_modules[i].get_count();
    mp_testOrder = new int[test_count];
    memset((void*)mp_testOrder, 0, sizeof(*mp_testOrder) * test_count);
    SetTestRunOrder(i);

    //
    //  List all tests first if the option was turned on
    //
    if (m_list) {
      PrintTestOrder(i);
      delete[] mp_testOrder;
      continue;
      // return;
    }

    for (unsigned int itr_var = 0; itr_var < m_modules[i].get_count(); itr_var++) {
      // done for random order generation
      unsigned int subtest = mp_testOrder[itr_var];

      const char* name = m_modules[i].get_name(subtest);
      if (itr_var < m_tests.size() && TestInList(m_tests, name)) {
        OCLTest* pt = NULL;
        if (m_modules[i].cached_test) {
          pt = m_modules[i].cached_test[subtest];
        }
        // Try to use the cached version first!
        if (!pt) {
          pt = m_modules[i].create_test(subtest);
          if (pt->cache_test() && m_modules[i].cached_test) {
            m_modules[i].cached_test[subtest] = pt;
          }
        }

        int numSubTests = pt->getNumSubTests();
        assert(numSubTests > 0);

        TestIndexList testIndices;
        GetTestIndexList(testIndices, m_tests, name, numSubTests - 1);

        TestIndexList avoidIndices;
        GetTestIndexList(avoidIndices, m_avoid, name, numSubTests - 1);

        TestIndexList erasedIndices;
        PruneTestIndexList(testIndices, avoidIndices, erasedIndices);

        int numTestsRun = 0;
        for (unsigned int j = 0; j < testIndices.size(); j++) {
          unsigned int test = testIndices[j];

          WaitAllThreads();
          AddWorkerThread(i, subtest, test, pt->getThreadUsage(), runSubtest);

          for (unsigned int thread = 1; (thread < m_threads) && (thread < m_modules.size());
               thread++) {
            AddWorkerThread(thread, subtest, test, pt->getThreadUsage(), dummyThread);
          }

          numTestsRun++;
        }

        WaitAllThreads();
        // Printing the test report
        // First checking whether the number of subtests is greater than 1.
        // No point printing report for a one subtest test

        if (numTestsRun > 0) {
          if (testReport[0]->success) {
            num_passes++;
          } else {
            num_failures++;
          }
        }
        if (App::m_svcMsg) {
          for (unsigned int j = 0; j < erasedIndices.size(); j++) {
            oclTestLog(OCLTEST_LOG_ALWAYS, "##%s[testIgnored name='%s.%s.%d']\n", App::m_svcMsg,
                       m_modules[i].get_libname(), name, erasedIndices[j]);
          }
        }

        // Resetting the values of the test reports
        // for(unsigned int j = 0; j < m_numDevices; j++)
        for (unsigned int j = 0; j < 1; j++) {
          testReport[j]->reset();
        }
        m_modules[i].destroy_test(pt);
        if (m_modules[i].cached_test) {
          m_modules[i].cached_test[subtest] = NULL;
        }
      }
    }

    if (App::m_svcMsg) {
      oclTestLog(OCLTEST_LOG_ALWAYS, "##%s[testSuiteFinished name='ocltst %s']\n", App::m_svcMsg,
                 m_modules[i].get_libname());
    }

    // print the order in which the test are executed if they are
    // randomized.
    if (m_rndOrder) {
      PrintTestOrder(i);
    }
    // deleting the test order
    delete[] mp_testOrder;
  }

#ifdef _WIN32
  if (!m_console && m_window) {
    ((Window*)m_window)->ConsumeEvents();
  }
#endif
  float total_tests = (float)(num_passes + num_failures);

  float percent_passed = 0.0f;
  float percent_failed = 0.0f;
  float percent_total = 0.0f;
  if (total_tests > 0) {
    percent_passed = 100.0f * ((float)num_passes / total_tests);
    percent_failed = 100.0f * ((float)num_failures / total_tests);
    percent_total = 100.0f * ((float)total_tests / total_tests);
  }

  oclTestLog(OCLTEST_LOG_ALWAYS, "\n\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "----------------------------------------\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "Total Passed Tests:  %8d (%6.2f%s)\n", num_passes, percent_passed,
             "%");
  oclTestLog(OCLTEST_LOG_ALWAYS, "Total Failed Tests:  %8d (%6.2f%s)\n", num_failures,
             percent_failed, "%");
  oclTestLog(OCLTEST_LOG_ALWAYS, "----------------------------------------\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "Total Run Tests:     %8d (%6.2f%s)\n", (int)total_tests,
             percent_total, "%");
  oclTestLog(OCLTEST_LOG_ALWAYS, "\n\n");
}

/////////////////////////////////////////////////////////////////////////////

void App::AddToList(StringList& strlist, const char* str) {
  std::string s(str);

  strlist.push_back(s);
}

void App::LoadList(StringList& strlist, const char* filename) {
  char buffer[1024];

  FILE* fp = fopen(filename, "r");

  if (fp == NULL) return;

  while (fgets(buffer, 1000, fp) != NULL) {
    size_t length = strlen(buffer);
    if (length > 0) {
      if (buffer[length - 1] != '\n') {
        length++;
      }
      buffer[length - 1] = 0;
      AddToList(strlist, buffer);
    }
  }

  fclose(fp);
}

static void Help(const char* name) {
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "%s (-w | -V | -m | -M | -l | -t | -T | -p | -d | -x | -y | -g| "
             "-o | -n )\n",
             name);
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -w            : enable window mode\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -V            : enable TeamCity service messages\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -J            : enable Jenkins service messages\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -d            : dump test output to portable float map (pfm)\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -m <module>   : specify a DLL module with tests\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -M <filename> : specify a text file with one DLL module per line\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -l            : list test names in DLL modules and exit\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -s <count>    : number of threads to spawn\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -t <testname> : run test\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -T <filename> : specify a text file with one test per line\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -a <testname> : specify a test to avoid\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -A <filename> : specify a text file of tests to avoid with "
             "one test per line\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -p <platform> : specify a platform to run on, 'amd','nvidia' "
             ",'intel' or 'ms'\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -h            : this help text\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -x            : x dimension for debug output image (and window)\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -y            : y dimension for debug output image (and window)\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -P            : Perflab mode (just print the result without "
             "any supplementary information)\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -n #number    : run the tests specified with -m, -M, -t or -T "
             "options multiple times\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -r            : Option to Randomize the order in which the "
             "tests are executed.\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "   -R            : Option to ReRun failed tests for conditional "
             "pass.\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -i            : Don't print system information\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -g <GPUid>    : GPUid to run the tests on\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -o <filename> : dump the output to a specified file\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "   -c            : Run the test on the CPU device.\n");
  oclTestLog(OCLTEST_LOG_ALWAYS, "                 : \n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "                 : To run only one subtest of a test, append the "
             "subtest to\n");
  oclTestLog(OCLTEST_LOG_ALWAYS,
             "                 : the end of the test name in brackets. i.e. test[1]");
  oclTestLog(OCLTEST_LOG_ALWAYS, "\n");

  exit(0);
}

unsigned int getPlatformID(const char* str) {
  std::string strOfCLVendor(str);
  std::string strOfCLPlatformName;
  unsigned int platform = 0;

  // currently, the only input values amd,nvidia and intel are supported
  if (strOfCLVendor == "amd") {
    strOfCLPlatformName = "Advanced Micro Devices, Inc.";
  } else if (strOfCLVendor == "intel") {
    strOfCLPlatformName = "Intel(R) Corporation";
  } else if (strOfCLVendor == "nvidia") {
    strOfCLPlatformName = "NVIDIA Corporation";
  } else if (strOfCLVendor == "ms") {
    strOfCLPlatformName = "Microsoft";
  } else {
    // fall-back on platform index 0
    return platform;
  }

  cl_int status;
  cl_uint numPlatforms = 0;

  status = clGetPlatformIDs(0, NULL, &numPlatforms);
  if (status != CL_SUCCESS) {
    return platform;
  }

  cl_platform_id* platforms = new cl_platform_id[numPlatforms];
  status = clGetPlatformIDs(numPlatforms, platforms, NULL);

  if (status == CL_SUCCESS) {
    unsigned int i;
    for (i = 0; i < numPlatforms; ++i) {
      char buff[200];
      status = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buff), buff, NULL);
      if (status != CL_SUCCESS) {
        break;
      }
      if (strcmp(buff, strOfCLPlatformName.c_str()) == 0) {
        platform = i;
        break;
      }
    }
  }

  delete[] platforms;
  return platform;
}

static const char* supported_options = "dg:lm:M:o:Ps:t:T:a:A:p:v:wxy:in:rcRVJ";

unsigned int parseCommandLineForPlatform(unsigned int argc, char** argv) {
  int c;
  unsigned int platform = 0;

  while ((c = getopt(argc, argv, supported_options)) != -1) {
    switch (c) {
      case 'p':
        platform = getPlatformID(optarg);
        break;
      default:
        break;
    }
  }
  return platform;
}

void App::CommandLine(unsigned int argc, char** argv) {
  unsigned int i = 1;
  int c;
  bool hasOption = false;
  unsigned int tmpNumDevices = 0;
  unsigned int tmpDeviceId = 0;
  m_deviceId = 0;
  int tmp;

  while ((c = getopt(argc, argv, supported_options)) != -1) {
    switch (c) {
      case 'c':
        m_useCPU = true;
        break;

      case 'p':
        break;

      case 'w':
        m_console = false;
        hasOption = true;
        break;

      case 'V':
        m_svcMsg = "teamcity";
        break;

      case 'J':
        m_svcMsg = "jenkins";
        break;

      case 'd':
        m_dump = true;
        hasOption = true;
        break;

      case 'm':
        AddToList(m_paths, optarg);
        hasOption = true;
        break;

      case 'M':
        LoadList(m_paths, optarg);
        hasOption = true;
        break;

      case 'a':
        AddToList(m_avoid, optarg);
        hasOption = true;
        break;

      case 'A':
        LoadList(m_avoid, optarg);
        hasOption = true;
        break;

      case 'l':
        m_list = true;
        hasOption = true;
        break;

      // command line switch to loop execution of any specified test or tests n
      // number of times
      case 'n':
        m_numItr = atoi(optarg);
        break;

      // command line switch to randomize the order of test execution in OCLTest
      case 'r':
        m_rndOrder = true;
        break;

      // command line switch to rerun the failed tests to see if they pass on
      // second run
      case 'R': {
        m_reRunFailed = true;
        break;
      }
      case 't':
        AddToList(m_tests, optarg);
        hasOption = true;
        break;

      case 'T':
        LoadList(m_tests, optarg);
        hasOption = true;
        break;

      case 's':
        m_threads = atoi(optarg);
        hasOption = true;
        break;

      case 'h':
        Help(argv[0]);
        break;

      case 'x':
        m_width = atoi(optarg);
        hasOption = true;
        break;

      case 'y':
        m_height = atoi(optarg);
        hasOption = true;
        break;

      case 'P':
        m_perflab = true;
        hasOption = true;
        break;
      case 'g':
#if 0
            tmpNumDevices = (unsigned int)atoi(optarg);
            if(m_numDevices < tmpNumDevices)
            {
                oclTestLog(OCLTEST_LOG_ALWAYS, "Number of Devices(%d) less than specified by the user(%d).  Using %d devices.\n", m_numDevices, tmpNumDevices, m_numDevices);
            }
            else
            {
                m_numDevices = tmpNumDevices;
            }
#else
        tmpDeviceId = (unsigned int)atoi(optarg);
#endif
        break;
      case 'v':
        tmp = atoi(optarg);
        if (tmp >= 0 && tmp < 100) {
          oclTestSetLogLevel(atoi(optarg));
        } else {
          oclTestLog(OCLTEST_LOG_ALWAYS, "Invalid verbose level\n");
        }
        break;
      case 'o': {
        hasOption = true;
        oclTestEnableLogToFile(optarg);
      } break;
      case 'i':
        m_noSysInfoPrint = true;
        break;
      default:
        Help(argv[0]);
        break;
    }
  }

  // Reset devices in case user overrode defaults
  m_numDevices = findAdapters(m_platform, m_useCPU, &mpform_id);
  if (m_numDevices < (tmpDeviceId + 1)) {
    m_deviceId = 0;
    oclTestLog(OCLTEST_LOG_ALWAYS,
               "User specified deviceId(%d) exceedes the number of "
               "Devices(%d).  Using device %d.\n",
               tmpDeviceId, m_numDevices, m_deviceId);
  } else {
    m_deviceId = tmpDeviceId;
  }

  if (!hasOption) {
    Help(argv[0]);
  }
}

bool App::TestInList(StringList& strlist, const char* szModuleTestname) {
  if (szModuleTestname == NULL) {
    return false;
  }
  for (unsigned int i = 0; i < strlist.size(); i++) {
    // check to see if an index is specified for this test name
    int nIndex = -1;
    std::string szTestName = strlist[i];
    if (szTestName.find("[") != std::string::npos) {
      size_t nFirstBracket = szTestName.find("[");
      size_t nLastBracket = szTestName.find("]");
      if ((nFirstBracket != std::string::npos) && (nLastBracket != std::string::npos) &&
          (nLastBracket > nFirstBracket)) {
        szTestName = szTestName.substr(0, nFirstBracket);
      }
    }
    if (strcmp(szModuleTestname, szTestName.c_str()) == 0) {
      return true;
    }
  }

  return false;
}

void App::GetTestIndexList(TestIndexList& testIndices, StringList& testList,
                           const char* szModuleTestname, int maxIndex) {
  for (unsigned int i = 0; i < testList.size(); i++) {
    IndicesRange nIndex = {0, maxIndex};

    // If the test name string ends with [...] parse the text
    // between the brackets to determine the index range.
    std::string szTestName = testList[i];
    if (szTestName.find("[") != std::string::npos) {
      size_t nFirstBracket = szTestName.find("[");
      size_t nLastBracket = szTestName.find("]");
      if ((nFirstBracket != std::string::npos) && (nLastBracket != std::string::npos) &&
          (nLastBracket > nFirstBracket)) {
        // Getting the string between the brackets '[' and ']'
        // The values can be one of the following:-
        // [a-b] - Run tests from a to b
        // [a-] - Run tests from subtest a to subtest total_tests
        // [-b] - Run tests from subtest 0 to subtest b
        // a and b are indices of the tests to run

        std::string nIndexString =
            szTestName.substr(nFirstBracket + 1, nLastBracket - nFirstBracket - 1);
        size_t nIntermediateHyphen = szTestName.find("-");
        if ((nIntermediateHyphen != std::string::npos) && (nIntermediateHyphen < nLastBracket) &&
            (nIntermediateHyphen > nFirstBracket)) {
          // Getting the start index
          if ((nIntermediateHyphen - 1) == nFirstBracket) {
            nIndex.startIndex = 0;
          } else {
            nIndex.startIndex =
                atoi(szTestName.substr(nFirstBracket + 1, nIntermediateHyphen - nFirstBracket - 1)
                         .c_str());
          }

          // Getting the end index
          if ((nIntermediateHyphen + 1) == nLastBracket) {
            nIndex.endIndex = maxIndex;
          } else {
            nIndex.endIndex = atoi(
                szTestName.substr(nIntermediateHyphen + 1, nLastBracket - nIntermediateHyphen - 1)
                    .c_str());
          }
        } else {
          nIndex.startIndex =
              atoi(szTestName.substr(nFirstBracket + 1, nLastBracket - nFirstBracket - 1).c_str());
          nIndex.endIndex = nIndex.startIndex;
        }
      }

      szTestName = szTestName.substr(0, nFirstBracket);
    }

    if (strcmp(szModuleTestname, szTestName.c_str()) == 0) {
      // If the values are out of order, swap them.
      if (nIndex.startIndex > nIndex.endIndex) {
        int tmp = nIndex.startIndex;
        nIndex.startIndex = nIndex.endIndex;
        nIndex.endIndex = tmp;
      }

      // Add the indices in the specified range to the list.
      for (int i = nIndex.startIndex; i <= nIndex.endIndex; ++i) {
        if (i <= maxIndex) {
          testIndices.push_back(i);
        } else {
          oclTestLog(OCLTEST_LOG_ALWAYS, "Error: Invalid test index for subtest: %s!\n",
                     szModuleTestname);
        }
      }

      // Now sort and prune duplicates.
      std::sort(testIndices.begin(), testIndices.end());
      std::unique(testIndices.begin(), testIndices.end());
    }
  }
}

void App::PruneTestIndexList(TestIndexList& testIndices, TestIndexList& avoidIndices,
                             TestIndexList& erasedIndices) {
  for (TestIndexList::iterator it = testIndices.begin(); it != testIndices.end();) {
    unsigned int index = *it;
    TestIndexList::iterator result = std::find(avoidIndices.begin(), avoidIndices.end(), index);
    if (result != avoidIndices.end()) {
      it = testIndices.erase(it);
      erasedIndices.push_back(index);
    } else {
      ++it;
    }
  }
}

void App::ScanForTests() {
  for (unsigned int i = 0; i < m_paths.size(); i++) {
    Module mod;

#ifdef _WIN32
    std::string::iterator myIter;
    myIter = m_paths[i].end();
    myIter--;
    if (*myIter == 0x0a) m_paths[i].erase(myIter);

    mod.hmodule = LoadLibrary(m_paths[i].c_str());
#endif
#ifdef __linux__
    mod.hmodule = dlopen(m_paths[i].c_str(), RTLD_NOW);
#endif

    if (mod.hmodule == NULL) {
      fprintf(stderr, "Could not load module: %s\n", m_paths[i].c_str());
#ifdef __linux__
      fprintf(stderr, "Error : %s\n", dlerror());
#else
#endif
    } else {
      mod.name = m_paths[i];

#ifdef _WIN32
      mod.get_count = (TestCountFuncPtr)GetProcAddress(mod.hmodule, "OCLTestList_TestCount");
      mod.get_name = (TestNameFuncPtr)GetProcAddress(mod.hmodule, "OCLTestList_TestName");
      mod.create_test = (CreateTestFuncPtr)GetProcAddress(mod.hmodule, "OCLTestList_CreateTest");
      mod.destroy_test = (DestroyTestFuncPtr)GetProcAddress(mod.hmodule, "OCLTestList_DestroyTest");
      mod.get_version =
          (TestVersionFuncPtr)GetProcAddress(mod.hmodule, "OCLTestList_TestLibVersion");
      mod.get_libname = (TestLibNameFuncPtr)GetProcAddress(mod.hmodule, "OCLTestList_TestLibName");
#endif
#ifdef __linux__
      mod.get_count = (TestCountFuncPtr)dlsym(mod.hmodule, "OCLTestList_TestCount");
      mod.get_name = (TestNameFuncPtr)dlsym(mod.hmodule, "OCLTestList_TestName");
      mod.create_test = (CreateTestFuncPtr)dlsym(mod.hmodule, "OCLTestList_CreateTest");
      mod.destroy_test = (DestroyTestFuncPtr)dlsym(mod.hmodule, "OCLTestList_DestroyTest");
      mod.get_version = (TestVersionFuncPtr)dlsym(mod.hmodule, "OCLTestList_TestLibVersion");
      mod.get_libname = (TestLibNameFuncPtr)dlsym(mod.hmodule, "OCLTestList_TestLibName");
#endif
      mod.cached_test = new OCLTest*[mod.get_count()];
      for (int x = 0, y = mod.get_count(); x < y; ++x) {
        mod.cached_test[x] = NULL;
      }
      m_modules.push_back(mod);
    }
  }
}

void App::CleanUp() {
  for (unsigned int i = 0; i < m_modules.size(); i++) {
    if (m_modules[i].cached_test) {
      delete[] m_modules[i].cached_test;
    }
#ifdef _WIN32
    FreeLibrary(m_modules[i].hmodule);
#endif
#ifdef __linux__
    dlclose(m_modules[i].hmodule);
#endif
  }

#ifdef _WIN32
  if (m_window) delete m_window;
  m_window = 0;
#endif
}

extern int optind;
/////////////////////////////////////////////////////////////////////////////
bool App::m_reRunFailed = false;
const char* App::m_svcMsg = nullptr;

int main(int argc, char** argv) {
#if EMU_ENV
  printf("Built for Emulation Environment\n");
#endif  // EMU_ENV
  unsigned int platform = 0;
  platform = parseCommandLineForPlatform(argc, argv);
  // reset optind as we really didn't parse the full command line
  optind = 1;
  App app(platform);
#ifdef _WIN32
  // this function is registers windows service routine when ocltst is launched
  // by the OS on service initialization. On other scenarios, this function does
  // nothing.
  serviceStubCall();
  // SetErrorMode(SEM_NOGPFAULTERRORBOX);
  // const LPTOP_LEVEL_EXCEPTION_FILTER oldFilter =
  // SetUnhandledExceptionFilter(xFilter);
#endif  // _WIN32
#ifdef AUTO_REGRESS
  try {
#endif /* AUTO_REGRESS */
    app.CommandLine(argc, argv);
    app.printOCLinfo();
    app.ScanForTests();
    for (int i = 0; i < app.GetNumItr(); i++) {
      app.RunAllTests();
    }
    app.CleanUp();
#ifdef AUTO_REGRESS
  } catch (...) {
    oclTestLog(OCLTEST_LOG_ALWAYS, "Exiting due to unhandled exception!\n");
    return (-1);
  }
#endif /* AUTO_REGRESS */

  return 0;
}

#ifdef _WIN32

#include <dbghelp.h>

typedef unsigned int uint32;
typedef size_t uintp;

struct StackEntry {
  uintp addr;
  uint32 line;
  uint32 disp;
  char symbol[128];
  char file[128];
};

static const unsigned int MAX_DEPTH_PER_NODE = 24;
struct Info {
  bool operator==(const Info& b) const { return key == b.key; }

  uintp key;  // pointer, handle, whatever
  StackEntry stack[MAX_DEPTH_PER_NODE];
};

static void dumpTraceBack(CONTEXT& context) {
  Info info;

  oclTestLog(OCLTEST_LOG_ALWAYS, "Exception: exiting!\n");
  HANDLE process = GetCurrentProcess();

  STACKFRAME64 stackframe;
  memset(&stackframe, 0, sizeof(STACKFRAME64));

#if defined(_WIN64)
  stackframe.AddrPC.Offset = context.Rip;
  stackframe.AddrPC.Mode = AddrModeFlat;
  stackframe.AddrStack.Offset = context.Rsp;
  stackframe.AddrStack.Mode = AddrModeFlat;
  stackframe.AddrFrame.Offset = context.Rbp;
  stackframe.AddrFrame.Mode = AddrModeFlat;
#else
  stackframe.AddrPC.Offset = context.Eip;
  stackframe.AddrPC.Mode = AddrModeFlat;
  stackframe.AddrStack.Offset = context.Esp;
  stackframe.AddrStack.Mode = AddrModeFlat;
  stackframe.AddrFrame.Offset = context.Ebp;
  stackframe.AddrFrame.Mode = AddrModeFlat;
#endif
  unsigned int depth = 0;

  if (SymInitialize(process, NULL, true)) {
    while ((depth < MAX_DEPTH_PER_NODE) &&
           StackWalk64(IMAGE_FILE_MACHINE_I386, process, GetCurrentThread(), &stackframe, &context,
                       NULL, SymFunctionTableAccess64, SymGetModuleBase64, NULL)) {
      if (stackframe.AddrPC.Offset != 0) {
        //
        //  we don't want to evaluate the names/lines yet
        //  so just record the address
        //
        info.stack[depth].addr = (uintp)stackframe.AddrPC.Offset;

        DWORD64 disp64;
        DWORD disp;
        IMAGEHLP_SYMBOL64* symInfo;
        IMAGEHLP_LINE64 lineInfo;
        uintp addr = (uintp)stackframe.AddrPC.Offset;
        char buffer[128];

        symInfo = (IMAGEHLP_SYMBOL64*)&buffer[0];
        symInfo->SizeOfStruct = sizeof(symInfo);
        symInfo->MaxNameLength = (sizeof(buffer) - sizeof(IMAGEHLP_SYMBOL64));

        lineInfo.SizeOfStruct = sizeof(lineInfo);

        if (SymGetSymFromAddr64(process, addr, &disp64, symInfo)) {
          sprintf(info.stack[depth].symbol, "%s", symInfo->Name);
          info.stack[depth].disp = (uint32)disp64;
        } else {
          sprintf(info.stack[depth].symbol, "");
        }

        if (SymGetLineFromAddr64(process, addr, &disp, &lineInfo)) {
          sprintf(info.stack[depth].file, "%s", lineInfo.FileName);
          info.stack[depth].line = lineInfo.LineNumber;
        } else {
          info.stack[depth].file[0] = '\0';
        }
        depth++;
      }
    }
  }

  SymCleanup(process);

  int j = 0;
  while (j < MAX_DEPTH_PER_NODE && info.stack[j].addr != 0) {
    oclTestLog(OCLTEST_LOG_ALWAYS, "        %s()+%d (0x%.8x)  %s:%d\n", info.stack[j].symbol,
               info.stack[j].disp, info.stack[j].addr, info.stack[j].file, info.stack[j].line);

    j++;
  }
}

static LONG WINAPI xFilter(LPEXCEPTION_POINTERS xEP) {
  CONTEXT context;
  CONTEXT* xCtx = &context;
  memset(xCtx, 0, sizeof(CONTEXT));
  context.ContextFlags = CONTEXT_FULL;
  memcpy(xCtx, xEP->ContextRecord, sizeof(CONTEXT));

  dumpTraceBack(context);

  return (EXCEPTION_EXECUTE_HANDLER);
}
#undef CHECK_RESULT
#endif  // WIN_OS

/////////////////////////////////////////////////////////////////////////////
