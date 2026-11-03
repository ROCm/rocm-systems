// Local POC collector: drains the CP dispatch log of one target process, out of process.
//   dlogc <target_pid> <out_dir>
// Writes <out_dir>/dlog.csv (raw START/END records) and <out_dir>/clocks.csv (GPU/CPU clock
// pairs, CPU side bracketed with CLOCK_MONOTONIC = LTTng's trace clock). Exits after the target
// exits and a final drain.
#include "kfd_ioctl.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dirent.h>
#include <fcntl.h>
#include <poll.h>
#include <string>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace {
struct fw_record {
  uint32_t ts_lo;
  uint32_t ts_hi;
  uint32_t record_type;   // 1 START, 2 END (EOP)
  uint32_t dispatch_id;   // low 32 bits of the AQL write index
  uint32_t doorbell_off;  // absolute doorbell dword index
};
static_assert(sizeof(fw_record) == KFD_DISPATCH_LOG_FW_RECORD_BYTES, "record layout");

struct stream {
  uint32_t gpu_id = 0;
  int fd = -1;
  const uint8_t* base = nullptr;
  kfd_dlog_stream_info info = {};
  std::vector<uint32_t> cursor;
  uint64_t records = 0;
  uint64_t notifies = 0;
};

volatile sig_atomic_t g_stop = 0;

uint64_t mono_ns() {
  timespec ts{};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

std::vector<uint32_t> gpu_ids() {
  std::vector<uint32_t> ids;
  const char* root = "/sys/class/kfd/kfd/topology/nodes";
  DIR* d = opendir(root);
  if (d == nullptr) return ids;
  while (dirent* e = readdir(d)) {
    if (e->d_name[0] == '.') continue;
    std::string p = std::string(root) + "/" + e->d_name + "/gpu_id";
    FILE* f = std::fopen(p.c_str(), "r");
    if (f == nullptr) continue;
    unsigned v = 0;
    if (std::fscanf(f, "%u", &v) == 1 && v != 0) ids.push_back(v);
    std::fclose(f);
  }
  closedir(d);
  return ids;
}

void sample_clock(int kfd, uint32_t gpu_id, FILE* out) {
  kfd_ioctl_get_clock_counters_args a{};
  a.gpu_id = gpu_id;
  const uint64_t t0 = mono_ns();
  const int rc = ioctl(kfd, AMDKFD_IOC_GET_CLOCK_COUNTERS, &a);
  const uint64_t t1 = mono_ns();
  if (rc != 0) return;
  std::fprintf(out, "%u,%llu,%llu,%llu,%llu,%llu,%llu\n", gpu_id, (unsigned long long)t0,
               (unsigned long long)t1, (unsigned long long)a.gpu_clock_counter,
               (unsigned long long)a.cpu_clock_counter, (unsigned long long)a.system_clock_counter,
               (unsigned long long)a.system_clock_freq);
}

void drain(stream& s, FILE* out) {
  const uint32_t n = s.info.region_record_count;
  const auto* wptr = reinterpret_cast<const uint64_t*>(s.base + s.info.wptr_offset);
  const auto* recs = reinterpret_cast<const fw_record*>(s.base + s.info.records_offset);
  for (uint32_t region = 0; region < s.info.num_regions; ++region) {
    const uint32_t w =
        __atomic_load_n(reinterpret_cast<const uint32_t*>(&wptr[region]), __ATOMIC_ACQUIRE);
    const uint32_t r = s.cursor[region];
    const uint32_t unread = (w - r) & (n - 1);
    for (uint32_t k = 0; k < unread; ++k) {
      const fw_record rec = recs[static_cast<uint64_t>(region) * n + ((r + k) & (n - 1))];
      if (rec.record_type == 0 || rec.doorbell_off == 0) continue;
      const uint64_t ts = static_cast<uint64_t>(rec.ts_lo) | (static_cast<uint64_t>(rec.ts_hi) << 32);
      std::fprintf(out, "%u,%u,%u,%u,%u,%llu\n", s.gpu_id, region, rec.record_type,
                   rec.dispatch_id, rec.doorbell_off, (unsigned long long)ts);
      ++s.records;
    }
    s.cursor[region] = w;
  }
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <target_pid> <out_dir>\n", argv[0]);
    return 2;
  }
  const uint32_t pid = static_cast<uint32_t>(std::atoi(argv[1]));
  const std::string dir = argv[2];
  std::signal(SIGINT, [](int) { g_stop = 1; });
  std::signal(SIGTERM, [](int) { g_stop = 1; });

  const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
  if (kfd < 0) {
    std::perror("open /dev/kfd");
    return 1;
  }

  std::vector<stream> streams;
  for (uint32_t gid : gpu_ids()) {
    kfd_ioctl_profiler_args a{};
    a.op = KFD_IOC_PROFILER_DLOG;
    a.dlog.dlog_op = KFD_IOC_PROFILER_DLOG_OPEN_STREAM;
    a.dlog.gpu_id = gid;
    a.dlog.target_pid = pid;
    a.dlog.flags = KFD_DLOG_OPEN_F_RAW_MMAP;
    a.dlog.buffer_size = 4u * 20u * 65536u;  // 4 regions x 64Ki records
    if (ioctl(kfd, AMDKFD_IOC_PROFILER, &a) != 0) {
      std::fprintf(stderr, "gpu %u: open stream: %s\n", gid, std::strerror(errno));
      continue;
    }
    stream s;
    s.gpu_id = gid;
    s.fd = a.dlog.stream_fd;
    kfd_dlog_stream_args q{};
    q.op = KFD_DLOG_STREAM_OP_INFO;
    if (ioctl(s.fd, KFD_DLOG_STREAM_IOC, &q) != 0) {
      std::fprintf(stderr, "gpu %u: info: %s\n", gid, std::strerror(errno));
      close(s.fd);
      continue;
    }
    s.info = q.info;
    void* m = mmap(nullptr, s.info.mmap_size, PROT_READ, MAP_SHARED, s.fd, 0);
    if (m == MAP_FAILED) {
      std::fprintf(stderr, "gpu %u: mmap: %s\n", gid, std::strerror(errno));
      close(s.fd);
      continue;
    }
    s.base = static_cast<const uint8_t*>(m);
    // Start each region at its current write pointer: older records belong to nothing we traced.
    s.cursor.resize(s.info.num_regions);
    const auto* wptr = reinterpret_cast<const uint64_t*>(s.base + s.info.wptr_offset);
    for (uint32_t i = 0; i < s.info.num_regions; ++i)
      s.cursor[i] = __atomic_load_n(reinterpret_cast<const uint32_t*>(&wptr[i]), __ATOMIC_ACQUIRE);
    std::fprintf(stderr,
                 "gpu %u: stream abi %u, %u regions x %u records, pasid %u, target pid %u\n", gid,
                 s.info.abi_version, s.info.num_regions, s.info.region_record_count,
                 s.info.pasid, s.info.target_pid);
    streams.push_back(std::move(s));
  }
  if (streams.empty()) {
    std::fprintf(stderr, "no dispatch-log stream opened\n");
    return 1;
  }

  FILE* out = std::fopen((dir + "/dlog.csv").c_str(), "w");
  FILE* clk = std::fopen((dir + "/clocks.csv").c_str(), "w");
  std::fprintf(out, "gpu_id,region,record_type,dispatch_id,doorbell_off,gpu_ts\n");
  std::fprintf(clk, "gpu_id,mono_before,mono_after,gpu_clock,cpu_clock,system_clock,system_freq\n");
  for (auto& s : streams) sample_clock(kfd, s.gpu_id, clk);
  std::printf("ready\n");
  std::fflush(stdout);

  std::vector<pollfd> pfds;
  for (auto& s : streams) pfds.push_back({s.fd, POLLIN, 0});
  uint64_t last_clock = mono_ns();
  bool exited = false;
  while (!g_stop && !exited) {
    poll(pfds.data(), pfds.size(), 20);  // wake on notify, or drain on timeout
    for (size_t i = 0; i < streams.size(); ++i) {
      uint64_t count = 0;
      if (read(streams[i].fd, &count, sizeof(count)) == sizeof(count)) streams[i].notifies += count;
      drain(streams[i], out);
    }
    if (mono_ns() - last_clock > 100000000ull) {  // clock pair every 100 ms
      for (auto& s : streams) sample_clock(kfd, s.gpu_id, clk);
      last_clock = mono_ns();
    }
    for (auto& s : streams) {
      kfd_dlog_stream_args q{};
      q.op = KFD_DLOG_STREAM_OP_STATUS;
      if (ioctl(s.fd, KFD_DLOG_STREAM_IOC, &q) == 0 &&
          (q.status.status & KFD_DLOG_STATUS_TARGET_EXITED))
        exited = true;
    }
    if (kill(static_cast<pid_t>(pid), 0) != 0 && errno == ESRCH) exited = true;
  }
  for (auto& s : streams) {
    drain(s, out);  // final drain after the target exited
    sample_clock(kfd, s.gpu_id, clk);
    std::fprintf(stderr, "gpu %u: %llu records, %llu notifies\n", s.gpu_id,
                 (unsigned long long)s.records, (unsigned long long)s.notifies);
  }
  std::fclose(out);
  std::fclose(clk);
  return 0;
}
