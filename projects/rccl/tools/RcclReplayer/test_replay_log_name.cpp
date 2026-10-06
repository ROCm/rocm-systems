/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

// Host-only check of ParseLogName; build and run with "make test".

#include <cstdio>
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
  ExpectReject("rep.host.amd.com", "rep", "");
  ExpectReject("rep.1275.", "rep", "");
  ExpectReject("rep.1275", "rep", "");
  ExpectReject("rep2.1275.host", "rep", "");
  ExpectReject("rep.99999999999.host", "rep", "");
  ExpectReject("other.1275.host", "rep", "");
  if (failures != 0) {
    printf("%d ParseLogName check(s) failed\n", failures);
    return 1;
  }
  printf("ParseLogName checks passed\n");
  return 0;
}
