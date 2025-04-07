#include "hsatimer.hpp"

#define NANOSECONDS_PER_SECOND 1000000000

PerfTimer::PerfTimer() {
  freq_in_100mhz = MeasureTSCFreqHz();
}

PerfTimer::~PerfTimer() {
  while (!_timers.empty()) {
    Timer *temp = _timers.back();
    _timers.pop_back();
    delete temp;
  }
}

// Create a new timer instance and return its index
int PerfTimer::CreateTimer() {

  Timer *newTimer = new Timer;
  newTimer->_start = 0.0;
  newTimer->_clocks = 0.0;

  #ifdef _WIN32
  QueryPerformanceFrequency((LARGE_INTEGER *)&newTimer->_freq);
  #endif

  #ifdef  __linux__
  newTimer->_freq = NANOSECONDS_PER_SECOND;
  #endif

  // Save the timer object in timer list
  _timers.push_back(newTimer);
  return (int)(_timers.size() - 1);
}

int PerfTimer::StartTimer(int index) {

  if (index >= (int)_timers.size()) {
    Error("Cannot reset timer. Invalid handle.");
    return HSA_FAILURE;
  }

  #ifdef _WIN32
    // General Windows timing method
    #ifndef _AMD
      long long tmpStart;
      QueryPerformanceCounter((LARGE_INTEGER *)&(tmpStart));
  _   timers[index]->_start = (double)tmpStart;
    // AMD Windows timing method
    #else
    #endif
  #endif

  #ifdef  __linux__
    // General Linux timing method
    #ifndef _AMD
      struct timespec s;
      clock_gettime(CLOCK_MONOTONIC, &s);
      _timers[index]->_start =
      (long long)s.tv_sec * NANOSECONDS_PER_SECOND + (long long)s.tv_nsec;
    // AMD Linux timing method
    #else
      unsigned int unused;
    _timers[index]->_start = __rdtscp(&unused);
    #endif
  #endif

  return HSA_SUCCESS;
}

int PerfTimer::StopTimer(int index) {

  long long n = 0;
  if (index >= (int)_timers.size()) {
    Error("Cannot reset timer. Invalid handle.");
    return HSA_FAILURE;
  }
  
  #ifdef _WIN32
    #ifndef _AMD
      long long n1;
      QueryPerformanceCounter((LARGE_INTEGER *)&(n1));
      n = n1;
    // AMD Window Timing
    #else
    #endif
  #endif

  #ifdef  __linux__
    // General Linux timing method
    #ifndef _AMD
      struct timespec s;
      clock_gettime(CLOCK_MONOTONIC, &s);
      n = (long long)s.tv_sec * NANOSECONDS_PER_SECOND + (long long)s.tv_nsec;
    // AMD Linux timing
    #else
      unsigned int unused;
      n = __rdtscp(&unused);
    #endif
  #endif

  n -= _timers[index]->_start;
  _timers[index]->_start = 0;

  #ifndef _AMD
    _timers[index]->_clocks += n;
  #endif

  #ifdef  __linux__
    //_timers[index]->_clocks += 10 * n /freq_in_100mhz;      // unit is ns
    _timers[index]->_clocks += 1.0E-6 * 10 * n / freq_in_100mhz;  // convert to ms
    // cout << "_AMD is enabled!!!" << endl;
  #endif

  return HSA_SUCCESS;
}

void PerfTimer::Error(string str) { cout << str << endl; }

double PerfTimer::ReadTimer(int index) {

  if (index >= (int)_timers.size()) {
    Error("Cannot read timer. Invalid handle.");
    return HSA_FAILURE;
  }

  double reading = double(_timers[index]->_clocks);

  reading = double(reading / _timers[index]->_freq);

  return reading;
}

void PerfTimer::ResetTimer(int index) {
  
  // Check if index value is over the timer's size
  if (index >= (int)_timers.size()) {
    Error("Invalid index value\n");
    exit(1);
  }

  _timers[index]->_clocks = 0.0;
  _timers[index]->_start = 0.0;
}

uint64_t PerfTimer::CoarseTimestampUs() {
  
  #ifdef _WIN32
    uint64_t freqHz, ticks;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freqHz);
    QueryPerformanceCounter((LARGE_INTEGER *)&ticks);

    // Scale numerator and divisor until (ticks * 1000000) fits in uint64_t.
    while (ticks > (1ULL << 44)) {
      ticks /= 16;
      freqHz /= 16;
    }

    return (ticks * 1000000) / freqHz;
  #endif

  #ifdef  __linux__
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
  #endif
}

uint64_t PerfTimer::MeasureTSCFreqHz() {
  
  // Make a coarse interval measurement of TSC ticks for 1 gigacycles.
  unsigned int unused;
  uint64_t tscTicksEnd;

  uint64_t coarseBeginUs = CoarseTimestampUs();
  uint64_t tscTicksBegin = __rdtscp(&unused);
  do {
    tscTicksEnd = __rdtscp(&unused);
  } while (tscTicksEnd - tscTicksBegin < 1000000000);

  uint64_t coarseEndUs = CoarseTimestampUs();

  // Compute the TSC frequency and round to nearest 100MHz.
  uint64_t coarseIntervalNs = (coarseEndUs - coarseBeginUs) * 1000;
  uint64_t tscIntervalTicks = tscTicksEnd - tscTicksBegin;
  return (tscIntervalTicks * 10 + (coarseIntervalNs / 2)) / coarseIntervalNs;
}
