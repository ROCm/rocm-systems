#ifndef __MYTIME__
#define __MYTIME__

// Will use AMD timer and general Linux timer based on users' need --> compilation flag
// need to consider platform is Windows or Linux

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <string>
using namespace std;

#if defined(_MSC_VER)
  #include <time.h>
  #include <windows.h>
  #include <intrin.h>
#else
  #if defined(__GNUC__)
    #include <sys/time.h>
    #include <x86intrin.h>
  #endif // __GNUC__
#endif //_MSC_VER

#define HSA_FAILURE  1
#define HSA_SUCCESS 0

class PerfTimer {
	private:
		struct Timer
		{
			string name;          /* < name name of time object*/
			long long _freq;      /* < _freq frequency*/
			double _clocks;       /* < _clocks number of ticks at end*/
			double _start;        /* < _start start point ticks*/
		};

		std::vector<Timer*> _timers;  /*< _timers vector to Timer objects */
		double freq_in_100mhz;

	public:
		PerfTimer();
		~PerfTimer();

	private:
		//AMD timing method
		uint64_t CoarseTimestampUs();
		uint64_t MeasureTSCFreqHz();

		//General Linux timing method

	public:
		int CreateTimer();
		int StartTimer(int index);
		int StopTimer(int index);

	public:
		// retrieve time
		double ReadTimer(int index);
		// write into a file
		double WriteTimer(int index);

	public:
		void Error(string str);
};

#endif

