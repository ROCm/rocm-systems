#ifndef _CORE_TIMER_H_
#define _CORE_TIMER_H_

template <int Size> class CoreTimer {
  CoreTimer() {
    index_ = 0;
    freq_in_100mhz_ = MeasureTSCFreqHz();
  }
  ~CoreTimer() {
    if (index_ >= Size) {
      printf("ERROR: memory corruption: out of timer data");
      abort();
    }
  }

  // retrieve time
  double Get() {
    double n = 0;
    // AMD Linux timing
    unsigned int unused;
    n = __rdtscp(&unused);
    data_[index_] = 10 * n / freq_in_100mhz_;  // unit is ns
    index_ += 1;
  }

  double Print()

      private :
      // timer data
      double data_[Size];
  // data index
  uint32_t index_;
  // frequency
  double freq_in_100mhz_;

  // timing methods
  uint64_t CoreTimer::CoarseTimestampUs() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return uint64_t(ts.tv_sec) * 1000000 + ts.tv_nsec / 1000;
  }

  uint64_t CoreTimer::MeasureTSCFreqHz() {
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
};

#endif  // _CORE_TIMER_H_
