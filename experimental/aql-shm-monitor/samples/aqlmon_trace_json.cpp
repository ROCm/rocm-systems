// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aqlmon/aqlmon.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct DispatchKey {
  uint32_t pid = 0;
  uint64_t queue_id = 0;
  uint64_t dispatch_id = 0;

  bool operator==(const DispatchKey& rhs) const {
    return pid == rhs.pid && queue_id == rhs.queue_id && dispatch_id == rhs.dispatch_id;
  }
};

struct QueueLaneKey {
  uint32_t pid = 0;
  uint64_t queue_id = 0;

  bool operator==(const QueueLaneKey& rhs) const { return pid == rhs.pid && queue_id == rhs.queue_id; }
};

struct DispatchKeyHash {
  std::size_t operator()(const DispatchKey& value) const {
    std::size_t seed = static_cast<std::size_t>(value.pid);
    seed ^= std::hash<uint64_t>{}(value.queue_id) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    seed ^= std::hash<uint64_t>{}(value.dispatch_id) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct QueueLaneKeyHash {
  std::size_t operator()(const QueueLaneKey& value) const {
    std::size_t seed = static_cast<std::size_t>(value.pid);
    seed ^= std::hash<uint64_t>{}(value.queue_id) + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
    return seed;
  }
};

struct CodeObjectRange {
  uint64_t handle = 0;
  uint64_t executable_handle = 0;
  uint64_t agent_handle = 0;
  uint64_t load_base = 0;
  uint64_t load_size = 0;
  std::string uri = {};
};

struct PacketInfo {
  uint32_t pid = 0;
  uint32_t host_tid = 0;
  uint64_t queue_id = 0;
  uint64_t queue_ptr = 0;
  uint64_t queue_base = 0;
  uint64_t queue_size = 0;
  uint64_t dispatch_id = 0;
  uint64_t kernel_object = 0;
  uint64_t completion_signal = 0;
  uint64_t packet_header = 0;
  uint64_t code_object_handle = 0;
  uint64_t executable_handle = 0;
  uint64_t agent_handle = 0;
  std::string kernel_name = {};
  std::string code_object_uri = {};
};

struct TraceEvent {
  PacketInfo packet = {};
  uint64_t start_ns = 0;
  uint64_t end_ns = 0;
  uint64_t signal_value = 0;
  bool injected_signal = false;
  bool has_timestamps = false;
};

constexpr uint16_t kKernelDispatchPacketType = 2;  // HSA_PACKET_TYPE_KERNEL_DISPATCH

std::string json_escape(const std::string& input) {
  std::string output = {};
  output.reserve(input.size() + 8);
  for(unsigned char ch : input) {
    switch(ch) {
      case '\"': output += "\\\""; break;
      case '\\': output += "\\\\"; break;
      case '\b': output += "\\b"; break;
      case '\f': output += "\\f"; break;
      case '\n': output += "\\n"; break;
      case '\r': output += "\\r"; break;
      case '\t': output += "\\t"; break;
      default:
        if(ch < 0x20) {
          char buffer[7];
          std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned>(ch));
          output += buffer;
        } else {
          output += static_cast<char>(ch);
        }
        break;
    }
  }
  return output;
}

std::string ns_to_us(uint64_t ns) {
  const uint64_t whole_us = ns / UINT64_C(1000);
  const uint64_t frac_ns = ns % UINT64_C(1000);
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%" PRIu64 ".%03" PRIu64, whole_us, frac_ns);
  return buffer;
}

std::string hex_u64(uint64_t value) {
  char buffer[32];
  std::snprintf(buffer, sizeof(buffer), "0x%llx", static_cast<unsigned long long>(value));
  return buffer;
}

std::string dispatch_name(const PacketInfo& packet) {
  if(!packet.kernel_name.empty()) return packet.kernel_name;
  if(!packet.code_object_uri.empty()) {
    std::ostringstream os;
    os << packet.code_object_uri;
    if(packet.code_object_handle != 0) os << " @" << hex_u64(packet.kernel_object);
    return os.str();
  }

  std::ostringstream os;
  os << "kernel " << hex_u64(packet.kernel_object);
  return os.str();
}

const CodeObjectRange* find_code_object(const std::unordered_map<uint64_t, CodeObjectRange>& live_code_objects,
                                        uint64_t kernel_object, uint64_t agent_handle) {
  for(const auto& itr : live_code_objects) {
    const auto& code_object = itr.second;
    if(code_object.load_size == 0) continue;
    if(kernel_object < code_object.load_base ||
       kernel_object >= (code_object.load_base + code_object.load_size)) {
      continue;
    }
    if(agent_handle != 0 && code_object.agent_handle != 0 && agent_handle != code_object.agent_handle) {
      continue;
    }
    return &code_object;
  }
  return nullptr;
}

int queue_lane(std::unordered_map<QueueLaneKey, int, QueueLaneKeyHash>& lanes,
               const QueueLaneKey& key) {
  const auto itr = lanes.find(key);
  if(itr != lanes.end()) return itr->second;

  const int lane = static_cast<int>(lanes.size()) + 1;
  lanes.emplace(key, lane);
  return lane;
}

void write_metadata_events(std::ofstream& output,
                           const std::unordered_map<QueueLaneKey, int, QueueLaneKeyHash>& lanes,
                           bool& first_event) {
  std::unordered_map<uint32_t, bool> seen_pids = {};
  for(const auto& itr : lanes) {
    const auto& key = itr.first;
    const int lane = itr.second;

    if(!seen_pids[key.pid]) {
      if(!first_event) output << ",\n";
      output << "    {\"ph\":\"M\",\"name\":\"process_name\",\"pid\":" << key.pid
             << ",\"args\":{\"name\":\"pid " << key.pid << "\"}}";
      first_event = false;
      seen_pids[key.pid] = true;
    }

    if(!first_event) output << ",\n";
    output << "    {\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":" << key.pid
           << ",\"tid\":" << lane << ",\"args\":{\"name\":\"queue "
           << json_escape(std::to_string(key.queue_id)) << "\"}}";
    first_event = false;
  }
}

void write_trace_event(std::ofstream& output, const TraceEvent& event, int lane, bool& first_event) {
  if(!first_event) output << ",\n";

  const auto duration_ns = (event.end_ns >= event.start_ns) ? (event.end_ns - event.start_ns) : 0;
  output << "    {"
         << "\"name\":\"" << json_escape(dispatch_name(event.packet)) << "\","
         << "\"cat\":\"kernel_dispatch\","
         << "\"ph\":\"X\","
         << "\"pid\":" << event.packet.pid << ","
         << "\"tid\":" << lane << ","
         << "\"ts\":" << ns_to_us(event.start_ns) << ","
         << "\"dur\":" << ns_to_us(duration_ns) << ","
         << "\"args\":{"
         << "\"dispatch_id\":" << event.packet.dispatch_id << ","
         << "\"queue_id\":" << event.packet.queue_id << ","
         << "\"queue_ptr\":\"" << hex_u64(event.packet.queue_ptr) << "\","
         << "\"host_tid\":" << event.packet.host_tid << ","
         << "\"kernel_object\":\"" << hex_u64(event.packet.kernel_object) << "\","
         << "\"completion_signal\":\"" << hex_u64(event.packet.completion_signal) << "\","
         << "\"signal_value\":" << event.signal_value << ","
         << "\"packet_header\":\"" << hex_u64(event.packet.packet_header) << "\","
         << "\"injected_signal\":" << (event.injected_signal ? "true" : "false");

  if(event.packet.code_object_handle != 0) {
    output << ",\"code_object_handle\":\"" << hex_u64(event.packet.code_object_handle) << "\"";
  }
  if(event.packet.executable_handle != 0) {
    output << ",\"executable_handle\":\"" << hex_u64(event.packet.executable_handle) << "\"";
  }
  if(!event.packet.code_object_uri.empty()) {
    output << ",\"code_object_uri\":\"" << json_escape(event.packet.code_object_uri) << "\"";
  }
  output << "}}";

  first_event = false;
}

}  // namespace

int main(int argc, char** argv) {
  if(argc != 3) {
    std::fprintf(stderr, "usage: %s <shm-name> <output-trace.json>\n", argv[0]);
    return 1;
  }

  const char* shm_name = argv[1];
  const char* output_path = argv[2];

  int fd = shm_open(shm_name, O_RDONLY, 0);
  if(fd < 0) {
    std::perror("shm_open");
    return 1;
  }

  struct stat st {};
  if(fstat(fd, &st) != 0) {
    std::perror("fstat");
    close(fd);
    return 1;
  }

  void* mapping = mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
  if(mapping == MAP_FAILED) {
    std::perror("mmap");
    close(fd);
    return 1;
  }

  auto* header = static_cast<const aqlmon_shm_header_t*>(mapping);
  if(header->magic != AQLMON_MAGIC) {
    std::fprintf(stderr, "unexpected shm magic: 0x%llx\n",
                 static_cast<unsigned long long>(header->magic));
    munmap(mapping, static_cast<size_t>(st.st_size));
    close(fd);
    return 1;
  }

  auto* records = reinterpret_cast<const aqlmon_record_t*>(
      static_cast<const unsigned char*>(mapping) + sizeof(aqlmon_shm_header_t));

  const uint64_t write_seq = header->write_seq;
  const uint64_t begin = (write_seq > header->capacity) ? (write_seq - header->capacity) : 0;

  std::unordered_map<uint64_t, CodeObjectRange> live_code_objects = {};
  std::unordered_map<DispatchKey, PacketInfo, DispatchKeyHash> packets = {};
  std::vector<TraceEvent> events = {};

  for(uint64_t seq = begin; seq < write_seq; ++seq) {
    const auto& rec = records[seq % header->capacity];
    if(rec.seq != (seq + 1)) continue;

    switch(rec.kind) {
      case AQLMON_RECORD_CODE_OBJECT_LIVE: {
        CodeObjectRange range{};
        range.handle = rec.code_object_handle;
        range.executable_handle = rec.executable_handle;
        range.agent_handle = rec.agent_handle;
        range.load_base = rec.code_object_load_base;
        range.load_size = rec.code_object_load_size;
        range.uri = rec.code_object_uri;
        live_code_objects[range.handle] = std::move(range);
        break;
      }
      case AQLMON_RECORD_CODE_OBJECT_DEAD:
        live_code_objects.erase(rec.code_object_handle);
        break;
      case AQLMON_RECORD_PACKET:
        if(rec.packet_type == kKernelDispatchPacketType) {
          PacketInfo packet{};
          packet.pid = rec.pid;
          packet.host_tid = rec.tid;
          packet.queue_id = rec.queue_id;
          packet.queue_ptr = rec.queue_ptr;
          packet.queue_base = rec.queue_base;
          packet.queue_size = rec.queue_size;
          packet.dispatch_id = rec.dispatch_id;
          packet.kernel_object = rec.kernel_object;
          packet.completion_signal = rec.completion_signal;
          packet.packet_header = rec.packet_header;
          packet.code_object_handle = rec.code_object_handle;
          packet.executable_handle = rec.executable_handle;
          packet.agent_handle = rec.agent_handle;
          packet.kernel_name = rec.kernel_name;
          packet.code_object_uri = rec.code_object_uri;

          if(packet.code_object_uri.empty()) {
            if(const auto* code_object =
                   find_code_object(live_code_objects, packet.kernel_object, packet.agent_handle)) {
              packet.code_object_handle = code_object->handle;
              packet.executable_handle = code_object->executable_handle;
              packet.code_object_uri = code_object->uri;
            }
          }

          packets[{packet.pid, packet.queue_id, packet.dispatch_id}] = std::move(packet);
        }
        break;
      case AQLMON_RECORD_DISPATCH_COMPLETE: {
        TraceEvent event{};
        const DispatchKey key{rec.pid, rec.queue_id, rec.dispatch_id};
        const auto packet_itr = packets.find(key);
        if(packet_itr != packets.end()) {
          event.packet = packet_itr->second;
        } else {
          event.packet.pid = rec.pid;
          event.packet.host_tid = rec.tid;
          event.packet.queue_id = rec.queue_id;
          event.packet.queue_ptr = rec.queue_ptr;
          event.packet.queue_base = rec.queue_base;
          event.packet.queue_size = rec.queue_size;
          event.packet.dispatch_id = rec.dispatch_id;
          event.packet.kernel_object = rec.kernel_object;
          event.packet.completion_signal = rec.completion_signal;
          event.packet.agent_handle = rec.agent_handle;
        }
        event.start_ns = rec.dispatch_start_ns;
        event.end_ns = rec.dispatch_end_ns;
        event.signal_value = rec.signal_value;
        event.injected_signal = (rec.flags & AQLMON_FLAG_INJECTED_SIGNAL) != 0;
        event.has_timestamps = (rec.flags & AQLMON_FLAG_SIGNAL_TIMESTAMPS_VALID) != 0;
        if(event.has_timestamps && event.end_ns >= event.start_ns) {
          events.emplace_back(std::move(event));
        }
        break;
      }
      default: break;
    }
  }

  std::sort(events.begin(), events.end(), [](const TraceEvent& lhs, const TraceEvent& rhs) {
    return std::tie(lhs.start_ns, lhs.packet.pid, lhs.packet.queue_id, lhs.packet.dispatch_id) <
           std::tie(rhs.start_ns, rhs.packet.pid, rhs.packet.queue_id, rhs.packet.dispatch_id);
  });

  std::unordered_map<QueueLaneKey, int, QueueLaneKeyHash> lanes = {};
  for(const auto& event : events) {
    queue_lane(lanes, {event.packet.pid, event.packet.queue_id});
  }

  std::ofstream output{output_path, std::ios::out | std::ios::trunc};
  if(!output) {
    std::fprintf(stderr, "failed to open %s for writing\n", output_path);
    munmap(mapping, static_cast<size_t>(st.st_size));
    close(fd);
    return 1;
  }

  output << "{\n"
         << "  \"displayTimeUnit\": \"ns\",\n"
         << "  \"traceEvents\": [\n";

  bool first_event = true;
  write_metadata_events(output, lanes, first_event);

  for(const auto& event : events) {
    const int lane = queue_lane(lanes, {event.packet.pid, event.packet.queue_id});
    write_trace_event(output, event, lane, first_event);
  }

  output << "\n  ]\n}\n";
  output.close();

  std::printf("wrote %zu correlated kernel dispatch events to %s\n", events.size(), output_path);

  munmap(mapping, static_cast<size_t>(st.st_size));
  close(fd);
  return 0;
}
