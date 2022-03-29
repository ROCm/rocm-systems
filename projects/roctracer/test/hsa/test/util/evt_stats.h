#ifndef EVT_STATS_H_
#define EVT_STATS_H_

#include <stdint.h>

#include <map>
#include <set>
#include <sstream>
#include <utility>

template <class evt_id_t, class evt_weight_t>
class EvtStatsT {
  public:
  typedef std::mutex mutex_t;
  typedef uint64_t evt_count_t;
  typedef double evt_avr_t;
  struct evt_record_t {
    uint64_t count;
    evt_avr_t avr;
    evt_record_t() : count(0), avr(0) {}
  };
  typedef typename std::map<evt_id_t, evt_record_t> map_t;
  typedef typename std::map<evt_id_t, const char*> labels_t;

  // Comparison function
  struct cmpfun {
    template <typename T> bool operator()(const T& a, const T& b) const {
      return (a.second.avr != b.second.avr) ? a.second.avr < b.second.avr : a.first < b.first;
    }
  };

  inline void add_event(evt_id_t id, evt_weight_t weight) {
    std::lock_guard<mutex_t> lck(mutex_);
    //printf("EvtStats %p ::add_event %u %lu\n", this, id, weight); fflush(stdout);

    evt_record_t& rec = map_[id];
    const evt_count_t prev_count = rec.count;
    const evt_count_t new_count = prev_count + 1;
    const evt_avr_t prev_avr = rec.avr;
    const evt_avr_t new_avr = ((prev_avr * prev_count) + weight) / new_count;

    rec.count = new_count;
    rec.avr = new_avr;
  }

  void dump() {
    std::lock_guard<mutex_t> lck(mutex_);
    fprintf(stdout, "Dumping %s\n", path_); fflush(stdout);

    typedef typename std::set<std::pair<evt_id_t, evt_record_t>, cmpfun> set_t;
    set_t s_(map_.begin(), map_.end());

    uint64_t index = 0;
    for (auto& e : s_) {
      const evt_id_t id = e.first;
      const char* label = get_label(id);
      std::ostringstream oss;
      oss << index << ",\"" << label << "\"," << e.second.count << "," << (uint64_t)(e.second.avr) << "," << (uint64_t)(e.second.count * e.second.avr);
      fprintf(fdes_, "%s\n", oss.str().c_str());
      index += 1;
    }

    fclose(fdes_);
  }

  const char* get_label(const uint32_t& id) {
    auto ret = labels_.insert({id, NULL});
    const char* label = ret.first->second;
    return label;
  }
  const char* get_label(const char* id) {
    return id;
  }
  const char* get_label(const std::string& id) {
    return id.c_str();
  }

  void set_label(evt_id_t id, const char* label) {
    //printf("EvtStats %p ::set_label %u %s\n", this, id, label); fflush(stdout);
    labels_[id] = label;
  }

  EvtStatsT(FILE* f, const char* path) : fdes_(f), path_(path) {
    //printf("EvtStats %p ::EvtStatsT()\n", this); fflush(stdout);
    fprintf(fdes_, "Index,Name,Count,Avr,Total\n");
  }

  private:
  mutex_t mutex_;
  map_t map_;
  labels_t labels_;
  FILE* fdes_;
  const char* path_;
};

typedef EvtStatsT<uint32_t, uint64_t> EvtStats;

#endif // EVT_STATS_H_
