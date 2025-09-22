
#ifndef ROCM_ASYNC_BW_MYTIME_H_
#define ROCM_ASYNC_BW_MYTIME_H_

// Will use AMD timer and general Linux timer based on users'
// need --> compilation flag. Support for windows platform is
// not currently available

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <x86intrin.h>
#include <string.h>

#include <iostream>
#include <vector>
#include <string>

using namespace std;

#include <sys/time.h>

#define HSA_FAILURE 1
#define HSA_SUCCESS 0

class PerfTimer {

 private:

  struct Timer {
    string name;       /* < name name of time object*/
    long long _freq;   /* < _freq frequency*/
    long long _clocks; /* < _clocks number of ticks at end*/
    long long _start;  /* < _start start point ticks*/
  };

  std::vector<Timer*> _timers; /*< _timers vector to Timer objects */
  double freq_in_100mhz;

 public:

  PerfTimer();
  ~PerfTimer();

 private:

  // AMD timing method
  uint64_t CoarseTimestampUs();
  uint64_t MeasureTSCFreqHz();

  // General Linux timing method

 public:
  
  int CreateTimer();
  int StartTimer(int index);
  int StopTimer(int index);
  void ResetTimer(int index);

 public:
 
  // retrieve time
  double ReadTimer(int index);
  
  // write into a file
  double WriteTimer(int index);

 public:
  void Error(string str);
};

#endif    //  ROCM_ASYNC_BW_MYTIME_H_
