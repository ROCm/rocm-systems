// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/diagnostics/hazard_record.h"

#include "util/json.h"
#include "util/log.h"

#include <cerrno>
#include <cstring>
#include <sstream>

#include <fcntl.h>
#include <unistd.h>

namespace rocjitsu::diagnostics {
namespace {

void append_site_lines(std::ostringstream &out, const HazardSite &site, bool marked,
                       std::string_view note) {
  if (site.pc == 0 && site.disasm.empty())
    return;
  out << (marked ? "  ==>  " : "       ");
  if (site.pc != 0)
    out << "0x" << std::hex << site.pc << std::dec << "  ";
  out << site.disasm;
  if (!note.empty())
    out << "  ; <-- " << note;
  out << '\n';
}

} // namespace

std::string format_text(const HazardRecord &record, std::string_view tag) {
  std::ostringstream out;
  out << tag;
  if (tag != "RACE")
    out << " checker=" << record.checker;
  out << " kernel=" << record.kernel << " symbol=" << record.symbol
      << " dispatch=" << record.dispatch << " type=" << record.type_name
      << " access=" << (record.write ? "write" : "read") << " reg=" << record.reg
      << " wave=" << record.wave << " lane=" << record.lane << " wg=" << record.wg[0] << ','
      << record.wg[1] << ',' << record.wg[2];
  if (!record.cu.empty())
    out << " cu=" << record.cu;
  if (tag == "RACE")
    out << " conflict=unknown checker=" << record.checker;
  if (record.producer.pc != 0)
    out << " producer_pc=0x" << std::hex << record.producer.pc << std::dec;
  if (record.consumer.pc != 0)
    out << " consumer_pc=0x" << std::hex << record.consumer.pc << std::dec;
  if (record.counter)
    out << " counter=" << *record.counter;
  if (record.required_wait)
    out << " required_wait=" << *record.required_wait;
  out << '\n' << record.summary << '\n';
  if (!record.trace_block.empty()) {
    out << record.trace_block;
    if (record.trace_block.back() != '\n')
      out << '\n';
  } else if (!record.trace.empty()) {
    for (const auto &site : record.trace)
      append_site_lines(out, site, site.marked, site.note);
  } else {
    append_site_lines(out, record.producer, true, "producer");
    append_site_lines(out, record.consumer, true, "consumer");
  }
  if (!record.trace_note.empty())
    out << "       " << record.trace_note << '\n';
  out << (tag == "RACE" ? "END_RACE\n" : "END_HAZARD\n");
  return out.str();
}

std::string format_jsonl(const HazardRecord &record) {
  using util::json_escape;
  std::ostringstream out;
  out << "{\"record\":\"site\",\"checker\":\"" << json_escape(record.checker)
      << "\",\"gpu\":" << record.gpu << ",\"kernel\":\"" << json_escape(record.kernel)
      << "\",\"symbol\":\"" << json_escape(record.symbol) << "\",\"dispatch\":" << record.dispatch
      << ",\"process\":" << record.process << ",\"wg\":[" << record.wg[0] << ',' << record.wg[1]
      << ',' << record.wg[2] << "],\"wave\":" << record.wave << ",\"lane\":" << record.lane
      << ",\"cu\":\"" << json_escape(record.cu) << "\",\"type\":\"" << json_escape(record.type_name)
      << "\",\"access\":\"" << (record.write ? "write" : "read") << "\",\"reg\":" << record.reg
      << ",\"producer_pc\":" << record.producer.pc << ",\"consumer_pc\":" << record.consumer.pc
      << ",\"producer_disasm\":\"" << json_escape(record.producer.disasm)
      << "\",\"consumer_disasm\":\"" << json_escape(record.consumer.disasm) << '"';
  if (record.counter)
    out << ",\"counter\":\"" << json_escape(*record.counter) << '"';
  if (record.required_wait)
    out << ",\"required_wait\":" << *record.required_wait;
  out << ",\"trace_executed\":" << (record.trace_executed ? "true" : "false") << '}';
  return out.str();
}

ReportWriter::~ReportWriter() {
  for (const auto &[path, file] : files_)
    if (file >= 0)
      ::close(file);
}

void ReportWriter::append(const std::string &path, std::string_view json_line) {
  auto [file, inserted] = files_.try_emplace(path, -1);
  if (inserted) {
    file->second = ::open(path.c_str(), O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (file->second < 0)
      util::Logger::warn("hazard report: cannot open ", path, ": ", std::strerror(errno));
  }
  if (file->second < 0)
    return;
  std::string line(json_line);
  line += '\n';
  for (size_t written = 0; written < line.size();) {
    const ssize_t count = ::write(file->second, line.data() + written, line.size() - written);
    if (count < 0 && errno == EINTR)
      continue;
    if (count <= 0) {
      util::Logger::warn("hazard report: cannot write ", path, ": ", std::strerror(errno));
      ::close(file->second);
      file->second = -1;
      return;
    }
    written += static_cast<size_t>(count);
  }
}

} // namespace rocjitsu::diagnostics
