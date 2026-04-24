// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aqlmon/aqlmon.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>

namespace {

const char* record_kind_name(uint16_t kind) {
  switch(kind) {
    case AQLMON_RECORD_PACKET: return "packet";
    case AQLMON_RECORD_DOORBELL: return "doorbell";
    case AQLMON_RECORD_DROP: return "drop";
    case AQLMON_RECORD_CODE_OBJECT_LIVE: return "code_object_live";
    case AQLMON_RECORD_CODE_OBJECT_DEAD: return "code_object_dead";
    case AQLMON_RECORD_DISPATCH_COMPLETE: return "dispatch_complete";
    default: return "unknown";
  }
}

void print_packet_record(const aqlmon_record_t& rec) {
  std::printf(
      "[%llu] %-16s type=%u pid=%u tid=%u qid=%llu dispatch_id=%llu idx=%llu wptr=%llu rptr=%llu "
      "kernel_object=0x%llx completion=0x%llx kernel=%s\n",
      static_cast<unsigned long long>(rec.seq), record_kind_name(rec.kind), rec.packet_type,
      rec.pid, rec.tid, static_cast<unsigned long long>(rec.queue_id),
      static_cast<unsigned long long>(rec.dispatch_id),
      static_cast<unsigned long long>(rec.write_index),
      static_cast<unsigned long long>(rec.observed_wptr),
      static_cast<unsigned long long>(rec.observed_rptr),
      static_cast<unsigned long long>(rec.kernel_object),
      static_cast<unsigned long long>(rec.completion_signal),
      (rec.kernel_name[0] != '\0') ? rec.kernel_name : "-");
}

void print_completion_record(const aqlmon_record_t& rec) {
  std::printf(
      "[%llu] %-16s pid=%u tid=%u qid=%llu dispatch_id=%llu signal=0x%llx "
      "start_ns=%llu end_ns=%llu kernel_object=0x%llx\n",
      static_cast<unsigned long long>(rec.seq), record_kind_name(rec.kind), rec.pid, rec.tid,
      static_cast<unsigned long long>(rec.queue_id),
      static_cast<unsigned long long>(rec.dispatch_id),
      static_cast<unsigned long long>(rec.completion_signal),
      static_cast<unsigned long long>(rec.dispatch_start_ns),
      static_cast<unsigned long long>(rec.dispatch_end_ns),
      static_cast<unsigned long long>(rec.kernel_object));
}

void print_code_object_record(const aqlmon_record_t& rec) {
  std::printf(
      "[%llu] %-16s exec=0x%llx code_object=0x%llx agent=0x%llx "
      "load=[0x%llx + 0x%llx] uri=%s\n",
      static_cast<unsigned long long>(rec.seq), record_kind_name(rec.kind),
      static_cast<unsigned long long>(rec.executable_handle),
      static_cast<unsigned long long>(rec.code_object_handle),
      static_cast<unsigned long long>(rec.agent_handle),
      static_cast<unsigned long long>(rec.code_object_load_base),
      static_cast<unsigned long long>(rec.code_object_load_size), rec.code_object_uri);
}

}  // namespace

int main(int argc, char** argv) {
  if(argc != 2) {
    std::fprintf(stderr, "usage: %s <shm-name>\n", argv[0]);
    return 1;
  }

  const char* shm_name = argv[1];
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

  std::printf(
      "shm=%s version=%u capacity=%u writes=%llu dropped_records=%llu dropped_packets=%llu\n",
      header->shm_name, header->version, header->capacity,
      static_cast<unsigned long long>(header->write_seq),
      static_cast<unsigned long long>(header->dropped_records),
      static_cast<unsigned long long>(header->dropped_packets));

  const uint64_t write_seq = header->write_seq;
  const uint64_t begin = (write_seq > 24) ? (write_seq - 24) : 0;
  for(uint64_t seq = begin; seq < write_seq; ++seq) {
    const auto& rec = records[seq % header->capacity];
    if(rec.seq != (seq + 1)) continue;
    switch(rec.kind) {
      case AQLMON_RECORD_CODE_OBJECT_LIVE:
      case AQLMON_RECORD_CODE_OBJECT_DEAD:
        print_code_object_record(rec);
        break;
      case AQLMON_RECORD_DISPATCH_COMPLETE:
        print_completion_record(rec);
        break;
      default:
        print_packet_record(rec);
        break;
    }
  }

  munmap(mapping, static_cast<size_t>(st.st_size));
  close(fd);
  return 0;
}
