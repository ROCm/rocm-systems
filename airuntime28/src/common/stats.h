// AIRUNTIME-28 benchmark support: the statistics that make the claims credible.
//
// One definition, because this was previously duplicated across seven programs
// and had already drifted: the bootstrap ran 2000 resamples in one experiment and
// 3000 in the others, so confidence intervals in different tables of the same
// report were computed with different estimators.
#ifndef AIRUNTIME28_STATS_H_
#define AIRUNTIME28_STATS_H_

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "config.h"

namespace bench {

inline double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

inline double percentile(std::vector<double> v, double p) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const double idx = p * static_cast<double>(v.size() - 1);
  const size_t lo = static_cast<size_t>(std::floor(idx));
  const size_t hi = static_cast<size_t>(std::ceil(idx));
  return v[lo] + (v[hi] - v[lo]) * (idx - static_cast<double>(lo));
}

// A percentage difference with a 95% interval.
struct Delta {
  double median = 0.0;
  double lo = 0.0;
  double hi = 0.0;

  double width() const { return hi - lo; }
  bool excludesZero() const { return (lo > 0.0) || (hi < 0.0); }
};

// Paired per-iteration ratio of x against base, in percent, summarised by a
// bootstrap on the median.
//
// Pairing is the point: because every arm is sampled once per iteration, sample i
// of x and sample i of base were taken seconds apart under the same conditions, so
// differencing them removes drift instead of hoping it averaged out.
inline Delta pairedDelta(const std::vector<double>& x, const std::vector<double>& base) {
  Delta r;
  const size_t n = std::min(x.size(), base.size());
  if (n == 0) return r;

  std::vector<double> d;
  d.reserve(n);
  for (size_t i = 0; i < n; ++i) d.push_back((x[i] / base[i] - 1.0) * 100.0);
  r.median = median(d);

  std::mt19937 rng(kBootstrapSeed);
  std::vector<double> medians;
  medians.reserve(kBootstrapResamples);
  std::vector<double> resample(d.size());
  for (int b = 0; b < kBootstrapResamples; ++b) {
    for (size_t i = 0; i < d.size(); ++i) resample[i] = d[rng() % d.size()];
    medians.push_back(median(resample));
  }
  r.lo = percentile(medians, 0.025);
  r.hi = percentile(medians, 0.975);
  return r;
}

// A result counts only if its interval excludes zero AND its magnitude exceeds the
// widest same-arm-twice gap. The second test is what distinguishes a real 1% from
// a confidently-measured artifact: a tight interval says a measurement is
// repeatable, not that it is measuring the intended thing.
inline bool separable(const Delta& effect, const std::vector<Delta>& noiseFloors) {
  if (!effect.excludesZero()) return false;
  double widest = 0.0;
  for (const Delta& f : noiseFloors) widest = std::max(widest, f.width());
  return std::abs(effect.median) > widest;
}

// "+1.23% [+1.10,+1.40]", or the same with a trailing " (noise)" when the effect
// did not clear both bars in separable() above.
//
// The marker spells the conclusion rather than abbreviating it. It used to read
// "(ns)", which readers consistently could not decode without hunting for a
// glossary - and a table cell that needs a glossary is a table cell that gets
// misread.
inline void formatDelta(char* out, size_t n, const Delta& d, bool isSignificant) {
  std::snprintf(out, n, "%+6.2f%% [%+.2f,%+.2f]%s", d.median, d.lo, d.hi,
                isSignificant ? "" : " (noise)");
}

}  // namespace bench

#endif  // AIRUNTIME28_STATS_H_
