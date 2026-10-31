// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/diagnostics/hazard_sites.h"
#include "rocjitsu/vm/plugins/plugin_config_resolver.h"

#include "flatbuffers/flexbuffers.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <unistd.h>

namespace {

using rocjitsu::RegClass;
using rocjitsu::diagnostics::format_jsonl;
using rocjitsu::diagnostics::format_text;
using rocjitsu::diagnostics::HazardDisposition;
using rocjitsu::diagnostics::HazardRecord;
using rocjitsu::diagnostics::HazardSiteKey;
using rocjitsu::diagnostics::HazardSites;
using rocjitsu::diagnostics::ReportWriter;
using rocjitsu::diagnostics::Settings;

HazardRecord sample_record() {
  HazardRecord record;
  record.checker = "memory-wait";
  record.type_name = "VGPR";
  record.kernel = "multi_wg_race_kernel";
  record.symbol = "_Z20multi_wg_race_kernel";
  record.dispatch = 3;
  record.lane = 0;
  record.cu = "soc.xcd0.se0.cu0";
  record.reg = 2;
  record.producer.pc = 0xa40;
  record.producer.disasm = "global_load_dword v2, v[2:3]";
  record.consumer.pc = 0xa48;
  record.consumer.disasm = "global_store_dword v[0:1], v2";
  record.counter = "vmcnt";
  record.required_wait = 0;
  record.summary = "Hazard on VGPR v2 [workgroup (0, 0, 0), wave 0, lane 0]";
  return record;
}

HazardSiteKey key_for(uint32_t dispatch, uint64_t consumer_pc = 0xa48) {
  return HazardSiteKey{dispatch, 0xa40, consumer_pc, 2, static_cast<uint8_t>(RegClass::VGPR), 0};
}

Settings with_cap(uint32_t max_sites) {
  Settings settings;
  settings.max_sites_per_dispatch = max_sites;
  return settings;
}

std::string finish_and_capture(HazardSites &sites, uint32_t dispatch) {
  testing::internal::CaptureStderr();
  sites.finish_dispatch(dispatch);
  return testing::internal::GetCapturedStderr();
}

unsigned count_substr(std::string_view text, std::string_view needle) {
  unsigned count = 0;
  for (size_t pos = 0; (pos = text.find(needle, pos)) != std::string_view::npos;
       pos += needle.size())
    ++count;
  return count;
}

/// A fresh path on every call: the writer keeps each report file open for the process.
std::string temp_report_path(std::string_view name) {
  static std::atomic<unsigned> serial{0};
  return (std::filesystem::temp_directory_path() /
          std::format("rj_hazard_{}_{}_{}.jsonl", ::getpid(), serial++, name))
      .string();
}

std::vector<std::string> read_lines(const std::string &path) {
  std::ifstream in(path);
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);)
    lines.push_back(line);
  return lines;
}

} // namespace

TEST(HazardReportFormat, TextAndJsonlShareTheRegisterSite) {
  const HazardRecord record = sample_record();
  const std::string text = format_text(record, "HAZARD");
  EXPECT_EQ(text.find("HAZARD checker=memory-wait kernel=multi_wg_race_kernel"), 0u);
  EXPECT_NE(text.find(" producer_pc=0xa40 consumer_pc=0xa48 counter=vmcnt"), std::string::npos);
  EXPECT_NE(text.find("global_load_dword v2"), std::string::npos);
  EXPECT_NE(text.find("END_HAZARD\n"), std::string::npos);

  HazardRecord race = record;
  race.checker = "race";
  race.cu.clear();
  race.counter.reset();
  race.required_wait.reset();
  race.summary = "Race on VGPR v2 [workgroup (0, 0, 0), wave 0, lane 0]";
  const std::string race_text = format_text(race, "RACE");
  EXPECT_EQ(race_text.find("RACE kernel=multi_wg_race_kernel symbol=_Z20multi_wg_race_kernel "
                           "dispatch=3 type=VGPR access=read reg=2 wave=0 lane=0 wg=0,0,0 "
                           "conflict=unknown checker=race producer_pc=0xa40 consumer_pc=0xa48\n"
                           "Race on VGPR v2"),
            0u);
  EXPECT_NE(race_text.find("END_RACE\n"), std::string::npos);

  const std::string json = format_jsonl(record);
  EXPECT_EQ(json.find(R"({"record":"site","checker":"memory-wait","gpu":0,)"), 0u) << json;
  EXPECT_NE(json.find("\"producer_pc\":2624"), std::string::npos);
  EXPECT_NE(json.find("\"consumer_pc\":2632"), std::string::npos);
}

TEST(HazardReportFormat, JsonlRecordsParseAsJson) {
  HazardRecord record = sample_record();
  record.kernel = "kernel \"quoted\" \\ name";
  std::string control;
  for (char ch = 1; ch < 0x20; ++ch)
    control += ch;
  record.consumer.disasm = control;
  const std::string json = format_jsonl(record);
  flexbuffers::Builder builder;
  ASSERT_TRUE(rocjitsu::plugin_detail::flexbuffer_from_json(json, builder)) << json;
  const auto object = flexbuffers::GetRoot(builder.GetBuffer()).AsMap();
  EXPECT_EQ(object["record"].AsString().str(), "site");
  EXPECT_EQ(object["kernel"].AsString().str(), record.kernel);
  EXPECT_EQ(object["consumer_disasm"].AsString().str(), control);
  EXPECT_EQ(object["producer_pc"].AsUInt64(), 0xa40u);
  EXPECT_EQ(object["wg"].AsVector().size(), 3u);
  EXPECT_FALSE(object["trace_executed"].AsBool());
}

TEST(HazardSitesTest, RepeatsAreCountedAndOnlyTheFirstOccurrencePrints) {
  HazardSites sites(with_cap(8));
  sites.register_dispatch({.dispatch_id = 77, .kernel = "storm", .symbol = "_Z5stormv"});
  const int unit = 0;
  unsigned fresh = 0;
  for (int repeat = 0; repeat < 20; ++repeat)
    fresh += sites.classify(&unit, key_for(77)) == HazardDisposition::New;
  EXPECT_EQ(fresh, 1u);

  const std::string summary = finish_and_capture(sites, 77);
  EXPECT_NE(summary.find("HAZARD_SUMMARY dispatch=77 kernel=storm sites=1 occurrences=20\n"),
            std::string::npos)
      << summary;
  EXPECT_TRUE(sites.find_dispatch(77).kernel.empty());
  // The dispatch's sites are gone, so finishing it again prints nothing.
  EXPECT_TRUE(finish_and_capture(sites, 77).empty());
}

TEST(HazardSitesTest, ComputeUnitsOfOneSocShareASite) {
  HazardSites sites(with_cap(8));
  const int first = 0;
  const int second = 0;
  EXPECT_EQ(sites.classify(&first, key_for(5)), HazardDisposition::New);
  for (int repeat = 0; repeat < 3; ++repeat) {
    EXPECT_EQ(sites.classify(&first, key_for(5)), HazardDisposition::Repeat);
    EXPECT_EQ(sites.classify(&second, key_for(5)), HazardDisposition::Repeat);
  }
  const std::string summary = finish_and_capture(sites, 5);
  EXPECT_NE(summary.find("sites=1 occurrences=7\n"), std::string::npos) << summary;
}

TEST(HazardSitesTest, ConcurrentUnitsCountEveryOccurrence) {
  HazardSites sites(with_cap(64));
  constexpr uint32_t kUnits = 8;
  constexpr unsigned kRepeats = 1000;
  constexpr uint64_t kKeys = 6;
  std::array<int, kUnits> units{};
  std::atomic<unsigned> fresh{0};
  std::vector<std::thread> threads;
  for (uint32_t unit = 0; unit < kUnits; ++unit)
    threads.emplace_back([&, unit] {
      for (unsigned repeat = 0; repeat < kRepeats; ++repeat)
        for (uint64_t key = 0; key < kKeys; ++key)
          if (sites.classify(&units[unit], key_for(30, 0xc00 + key * 4)) == HazardDisposition::New)
            ++fresh;
    });
  for (std::thread &thread : threads)
    thread.join();
  EXPECT_EQ(fresh.load(), kKeys);
  const std::string summary = finish_and_capture(sites, 30);
  EXPECT_NE(summary.find("sites=6 occurrences=48000\n"), std::string::npos) << summary;
}

TEST(HazardSitesTest, SocsWithTheSameDispatchIdKeepSeparateSites) {
  HazardSites first(with_cap(8));
  HazardSites second(with_cap(8));
  const int unit = 0;
  EXPECT_EQ(first.classify(&unit, key_for(1)), HazardDisposition::New);
  EXPECT_EQ(second.classify(&unit, key_for(1)), HazardDisposition::New);
}

TEST(HazardSitesTest, SitesPastTheCapAreSuppressedButCounted) {
  HazardSites sites(with_cap(2));
  const int unit = 0;
  testing::internal::CaptureStderr();
  EXPECT_EQ(sites.classify(&unit, key_for(9, 0xa48)), HazardDisposition::New);
  EXPECT_EQ(sites.classify(&unit, key_for(9, 0xa4c)), HazardDisposition::New);
  for (int repeat = 0; repeat < 3; ++repeat)
    EXPECT_EQ(sites.classify(&unit, key_for(9, 0xa50)), HazardDisposition::Suppressed);
  sites.finish_dispatch(9);
  const std::string output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(count_substr(output, "HAZARD_SUPPRESSED dispatch=9 max_sites_per_dispatch=2"), 1u)
      << output;
  EXPECT_NE(output.find("sites=2 occurrences=2 suppressed=3"), std::string::npos) << output;
}

TEST(HazardSitesTest, EvictedRepeatsStillReachTheSummary) {
  // More live sites than a unit's 64 cache slots, so repeats keep evicting each other.
  HazardSites sites(with_cap(128));
  const int unit = 0;
  for (int round = 0; round < 5; ++round)
    for (uint64_t site = 0; site < 80; ++site)
      sites.classify(&unit, key_for(11, 0xb00 + site * 4));
  const std::string summary = finish_and_capture(sites, 11);
  EXPECT_NE(summary.find("sites=80 occurrences=400\n"), std::string::npos) << summary;
}

TEST(HazardSitesTest, UnitsWithoutACommandProcessorHaveTheirOwnSites) {
  const int first = 0;
  const int second = 0;
  HazardSites &sites = rocjitsu::diagnostics::unattached_sites(&first);
  EXPECT_EQ(&sites, &rocjitsu::diagnostics::unattached_sites(&first));
  EXPECT_NE(&sites, &rocjitsu::diagnostics::unattached_sites(&second));
  rocjitsu::diagnostics::release_unattached_sites(&first);
  rocjitsu::diagnostics::release_unattached_sites(&second);
}

TEST(HazardSitesTest, ReportFileGetsSiteSuppressionAndSummaryRecords) {
  Settings settings = with_cap(1);
  settings.report_path = temp_report_path("records");
  HazardSites sites(settings, 3);
  sites.register_dispatch({.dispatch_id = 21, .kernel = "storm", .symbol = "_Z5stormv"});
  const int unit = 0;
  testing::internal::CaptureStderr();
  ASSERT_EQ(sites.classify(&unit, key_for(21, 0xa48)), HazardDisposition::New);
  HazardRecord record = sample_record();
  record.gpu = sites.gpu();
  record.dispatch = 21;
  sites.publish(record);
  EXPECT_EQ(sites.classify(&unit, key_for(21, 0xa4c)), HazardDisposition::Suppressed);
  sites.finish_dispatch(21);
  const std::string output = testing::internal::GetCapturedStderr();
  const auto lines = read_lines(settings.report_path);
  std::filesystem::remove(settings.report_path);
  ASSERT_EQ(lines.size(), 3u) << output;
  EXPECT_EQ(lines[0].find(R"({"record":"site","checker":"memory-wait","gpu":3,)"), 0u) << lines[0];
  EXPECT_EQ(lines[1],
            R"({"record":"suppressed","gpu":3,"dispatch":21,"max_sites_per_dispatch":1})");
  EXPECT_EQ(lines[2], R"({"record":"summary","gpu":3,"dispatch":21,"kernel":"storm","sites":1,)"
                      R"("occurrences":1,"suppressed":1})");
  EXPECT_EQ(output.find("END_HAZARD\n\n"), std::string::npos) << output;
}

TEST(HazardSitesTest, ConcurrentReportsKeepWholeLines) {
  const std::string path = temp_report_path("concurrent");
  // Longer than a stream buffer, so a record written in pieces could interleave.
  const std::string first_line = "{\"a\":\"" + std::string(20000, 'a') + "\"}";
  const std::string second_line = "{\"b\":\"" + std::string(20000, 'b') + "\"}";
  ReportWriter writer;
  const auto append = [&](const std::string &line) {
    for (int repeat = 0; repeat < 50; ++repeat)
      writer.publish([] {}, path, line);
  };
  std::thread first(append, std::cref(first_line));
  std::thread second(append, std::cref(second_line));
  first.join();
  second.join();
  const auto lines = read_lines(path);
  std::filesystem::remove(path);
  ASSERT_EQ(lines.size(), 100u);
  for (const std::string &line : lines)
    EXPECT_TRUE(line == first_line || line == second_line);
}

TEST(HazardSitesTest, UnwritableReportPathWarnsOnce) {
  const std::string path = temp_report_path("missing") + "/report.jsonl";
  ReportWriter writer;
  testing::internal::CaptureStderr();
  for (int repeat = 0; repeat < 3; ++repeat)
    writer.publish([] {}, path, "{}");
  const std::string output = testing::internal::GetCapturedStderr();
  EXPECT_EQ(count_substr(output, "hazard report: cannot open " + path), 1u) << output;
}
