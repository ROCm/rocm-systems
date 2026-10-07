/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

// Host-only check of ParseLogName and MatchHostsByRankCount; build and run with "make test".

#include <cstdio>
#include <map>
#include <string>

#include "replay_log_name.hpp"

namespace {

int failures = 0;

void ExpectParse(const std::string& name, const std::string& base, const std::string& ext, int wantPid,
                 const std::string& wantHost) {
  int pid = -1;
  std::string host;
  if (!ParseLogName(name, base, ext, &pid, &host) || pid != wantPid || host != wantHost) {
    printf("FAIL: %s -> pid %d host '%s', want pid %d host '%s'\n", name.c_str(), pid, host.c_str(), wantPid,
           wantHost.c_str());
    failures++;
  }
}

void ExpectReject(const std::string& name, const std::string& base, const std::string& ext) {
  int pid = -1;
  std::string host;
  if (ParseLogName(name, base, ext, &pid, &host)) {
    printf("FAIL: %s accepted as pid %d host '%s'\n", name.c_str(), pid, host.c_str());
    failures++;
  }
}

void ExpectMatch(const char* what, const std::map<std::string, int>& replayRanks,
                 const std::map<std::string, int>& logCounts, const std::map<std::string, std::string>& want) {
  std::map<std::string, std::string> got;
  if (!MatchHostsByRankCount(replayRanks, logCounts, &got) || got != want) {
    printf("FAIL: %s: host pairing differs\n", what);
    failures++;
  }
}

void ExpectNoMatch(const char* what, const std::map<std::string, int>& replayRanks,
                   const std::map<std::string, int>& logCounts) {
  std::map<std::string, std::string> got;
  if (MatchHostsByRankCount(replayRanks, logCounts, &got)) {
    printf("FAIL: %s: accepted\n", what);
    failures++;
  }
}

}  // namespace

int main() {
  ExpectParse("rep.1275.cv350-zts-gtu-h24-08.prov.gtu.zts.cpe.ice.amd.com", "rep", "", 1275,
              "cv350-zts-gtu-h24-08.prov.gtu.zts.cpe.ice.amd.com");
  ExpectParse("rep.1275.cv350-zts-gtu-h24-08.prov.gtu.zts.cpe.ice.amd.com.bin", "rep", ".bin", 1275,
              "cv350-zts-gtu-h24-08.prov.gtu.zts.cpe.ice.amd.com");
  ExpectParse("rep.42.useocpm2m-097-079", "rep", "", 42, "useocpm2m-097-079");
  ExpectParse("replayer_log.1270.quanta-cx77-11.bin", "replayer_log", ".bin", 1270, "quanta-cx77-11");
  ExpectReject("rep.1275.host.json", "rep", "");
  ExpectReject("rep.1275.host", "rep", ".bin");
  ExpectReject("rep.1275.hostname", "rep", ".bin");
  ExpectReject("rep.host.amd.com", "rep", "");
  ExpectReject("rep.1275.", "rep", "");
  ExpectReject("rep.1275", "rep", "");
  ExpectReject("rep2.1275.host", "rep", "");
  ExpectReject("rep.99999999999.host", "rep", "");
  ExpectReject("other.1275.host", "rep", "");
  ExpectReject("abc.1275.host", "rep", "");
  ExpectReject("rep..host", "rep", "");
  // Recorded 1 + 3 ranks: the host replaying 3 ranks must get the host that left 3 logs, whatever the name order.
  ExpectMatch("uneven", {{"a", 3}, {"b", 1}}, {{"x", 1}, {"y", 3}}, {{"a", "y"}, {"b", "x"}});
  ExpectMatch("names against count order", {{"a", 1}, {"b", 3}}, {{"x", 3}, {"y", 1}}, {{"a", "y"}, {"b", "x"}});
  ExpectMatch("even", {{"a", 8}, {"b", 8}}, {{"x", 8}, {"y", 8}}, {{"a", "x"}, {"b", "y"}});
  ExpectNoMatch("same totals, other split", {{"a", 2}, {"b", 2}}, {{"x", 1}, {"y", 3}});
  ExpectNoMatch("fewer replay hosts", {{"a", 4}}, {{"x", 1}, {"y", 3}});
  ExpectNoMatch("fewer replay hosts, first count equal", {{"a", 1}}, {{"x", 1}, {"y", 3}});
  ExpectNoMatch("more ranks than logs", {{"a", 4}, {"b", 4}}, {{"x", 4}, {"y", 3}});
  if (failures != 0) {
    printf("%d log name check(s) failed\n", failures);
    return 1;
  }
  printf("Log name checks passed\n");
  return 0;
}
