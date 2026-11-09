/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Recovery
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>

namespace fs = std::filesystem;


namespace {

hrr_event_header make_min_record(uint64_t seq) {
  hrr_event_header h{};
  std::memset(&h, 0, sizeof(h));
  h.event_type     = static_cast<uint16_t>(HRR_API_HIPDEVICESYNCHRONIZE);
  h.sequence_id    = seq;
  h.timestamp_ns   = 1000 + seq;
  h.thread_id      = 42;
  h.payload_length = static_cast<uint16_t>(sizeof(hrr_event_header));
  return h;
}

struct TmpArchive {
  fs::path root;
  std::ofstream f;

  explicit TmpArchive(const std::string& name) {
    root = fs::temp_directory_path() / ("hrr_rec_" + name);
    fs::remove_all(root);
    fs::create_directories(root / "blobs");
    fs::create_directories(root / "code_objects");
    f.open(root / "events.bin", std::ios::binary);
    hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
    f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
  }

  ~TmpArchive() { if (f.is_open()) f.close(); fs::remove_all(root); }

  std::string path() const { return root.string(); }

  void write_records(int n) {
    for (int i = 0; i < n; ++i) {
      hrr_event_header h = make_min_record(static_cast<uint64_t>(i));
      f.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }
  }

  void write_trailer(int total) {
    hrr_eof_record eof{};
    std::memset(&eof, 0, sizeof(eof));
    eof.hdr.event_type     = HRR_EOF_MARKER;
    eof.hdr.sequence_id    = static_cast<uint64_t>(total);
    eof.hdr.payload_length = static_cast<uint16_t>(sizeof(eof));
    eof.total_events       = static_cast<uint64_t>(total);
    eof.eof_magic          = HRR_EOF_MAGIC;
    f.write(reinterpret_cast<const char*>(&eof), sizeof(eof));
  }

  void write_bytes(const void* p, size_t n) {
    f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
  }

  void finish() { f.flush(); f.close(); }
};

struct TmpRootArchive {
  fs::path root;

  explicit TmpRootArchive(const std::string& name) {
    root = fs::temp_directory_path() / ("hrr_rec_" + name);
    fs::remove_all(root);
    fs::create_directories(root);
  }

  ~TmpRootArchive() { fs::remove_all(root); }

  // trailer_count < 0 makes the trailer count the records written.
  fs::path add_process(uint64_t pid, uint64_t parent_pid, int records,
                       bool trailer = true, bool manifest = true,
                       int trailer_count = -1) {
    fs::path proc = root / ("pid-" + std::to_string(pid));
    fs::create_directories(proc / "blobs");
    fs::create_directories(proc / "code_objects");

    std::ofstream events(proc / "events.bin", std::ios::binary);
    hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
    events.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
    for (int i = 0; i < records; ++i) {
      hrr_event_header h = make_min_record(static_cast<uint64_t>(i));
      events.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }
    if (trailer) {
      const int counted = trailer_count < 0 ? records : trailer_count;
      hrr_eof_record eof = hrr_make_eof_record(static_cast<uint64_t>(records),
                                               static_cast<uint64_t>(counted));
      events.write(reinterpret_cast<const char*>(&eof), sizeof(eof));
    }
    events.close();

    if (manifest) {
      std::ofstream mf(proc / "manifest.json");
      mf << "{\n"
         << "  \"pid\": " << pid << ",\n"
         << "  \"parent_pid\": " << parent_pid << ",\n"
         << "  \"complete\": " << (trailer ? "true" : "false") << ",\n"
         << "  \"event_count\": " << records << ",\n"
         << "  \"blob_count\": 0\n"
         << "}\n";
    }
    return proc;
  }
};

static std::pair<int, std::string> run_hrr_playback(const fs::path& archive,
                                                    const std::string& args) {
  hrr::test::SpawnProc proc(HRR_PLAYBACK_EXE, /*capture_stdout=*/true);
  set_proc_search_path(proc);
#ifdef _WIN32
  std::string path_arg = "\"" + archive.string() + "\"";
#else
  std::string path_arg = archive.string();
#endif
  int ret = proc.run(path_arg + (args.empty() ? "" : " " + args));
  return {ret, proc.getOutput()};
}

static std::string root_info_row(const std::string& out, const std::string& dir) {
  const size_t at = out.find(dir);
  if (at == std::string::npos) return std::string();
  const size_t bol = out.rfind('\n', at);
  return out.substr(bol + 1, at - bol - 1);
}

}  // namespace


HRR_TEST_CASE(Unit_HRR_Recovery_CompleteArchive) {
  TmpArchive arc("complete");
  arc.write_records(5);
  arc.write_trailer(5);
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 5);
  CHECK(a.complete);
  CHECK_FALSE(a.truncated);
  CHECK(hrr::has_clean_trailer(arc.path()));
}

HRR_TEST_CASE(Unit_HRR_Recovery_TornHeader) {
  TmpArchive arc("torn_header");
  arc.write_records(5);
  hrr_event_header partial = make_min_record(5);
  arc.write_bytes(&partial, sizeof(partial) / 2);  // only half the header
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 5);
  CHECK_FALSE(a.complete);
  CHECK(a.truncated);
}

HRR_TEST_CASE(Unit_HRR_Recovery_TornPayload) {
  TmpArchive arc("torn_payload");
  arc.write_records(3);
  hrr_event_header h = make_min_record(3);
  h.payload_length = 200;            // claims 168 bytes of payload that follow
  arc.write_bytes(&h, sizeof(h));    // ...but write none
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 3);
  CHECK_FALSE(a.complete);
  CHECK(a.truncated);
}

HRR_TEST_CASE(Unit_HRR_Recovery_ImplausiblePayloadLength) {
  TmpArchive arc("huge_payload");
  arc.write_records(2);
  hrr_event_header h = make_min_record(2);
  h.payload_length = 0xFFFFFFF0u;
  arc.write_bytes(&h, sizeof(h));
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 2);
  CHECK_FALSE(a.complete);
  CHECK(a.truncated);
}

HRR_TEST_CASE(Unit_HRR_Recovery_NoTrailer) {
  TmpArchive arc("no_trailer");
  arc.write_records(4);
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 4);
  CHECK_FALSE(a.complete);
  CHECK_FALSE(a.truncated);
}

HRR_TEST_CASE(Unit_HRR_Recovery_RecordAfterTrailer) {
  TmpArchive arc("after_trailer");
  arc.write_records(5);
  arc.write_trailer(5);
  for (uint64_t seq : {100u, 101u}) {
    hrr_event_header h = make_min_record(seq);
    arc.write_bytes(&h, sizeof(h));
  }
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 5);
  CHECK(a.complete);
  CHECK_FALSE(a.truncated);
  CHECK(a.skipped_after_trailer == 2);
  for (const auto& ev : a.events) CHECK(ev.header().sequence_id < 100);
  CHECK(hrr::has_clean_trailer(arc.path()));

  auto [ret, out] = run_hrr_playback(arc.root, "--info");
  INFO("hrr-playback --info stdout:\n" << out);
  REQUIRE(ret == 0);
  CHECK(out.find("Ignored:      2 records after the shutdown trailer") != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_Recovery_TrailerCountMismatch) {
  TmpArchive arc("count_mismatch");
  arc.write_records(5);
  arc.write_trailer(4);
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 5);
  CHECK_FALSE(a.complete);
  CHECK_FALSE(a.truncated);
  CHECK(a.skipped_after_trailer == 0);
  CHECK(a.trailer);
  CHECK(a.trailer_events == 4);
  CHECK_FALSE(hrr::has_clean_trailer(arc.path()));

  auto [ret, out] = run_hrr_playback(arc.root, "--info");
  INFO("hrr-playback --info stdout:\n" << out);
  REQUIRE(ret == 0);
  CHECK(out.find("trailer counts 4 events; 5 precede it") != std::string::npos);
  CHECK(out.find("no shutdown trailer") == std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_Recovery_TrailerOvercounts) {
  TmpArchive arc("count_over");
  arc.write_records(5);
  arc.write_trailer(6);
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 5);
  CHECK_FALSE(a.complete);
  CHECK_FALSE(hrr::has_clean_trailer(arc.path()));
}

HRR_TEST_CASE(Unit_HRR_Recovery_TornMiddleRecord) {
  TmpArchive arc("torn_middle");
  arc.write_records(2);
  hrr_event_header h = make_min_record(2);
  h.payload_length = 16;
  arc.write_bytes(&h, sizeof(h));
  for (uint64_t seq : {3u, 4u}) {
    hrr_event_header r = make_min_record(seq);
    arc.write_bytes(&r, sizeof(r));
  }
  arc.write_trailer(5);
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 2);
  CHECK(a.truncated);
  CHECK_FALSE(a.complete);
  CHECK_FALSE(hrr::has_clean_trailer(arc.path()));
}

HRR_TEST_CASE(Unit_HRR_Recovery_OversizeRecordBeforeTrailer) {
  TmpArchive arc("oversize");
  arc.write_records(2);
  hrr_event_header h = make_min_record(2);
  h.payload_length = hrr::kMaxRecordBytes + 1;
  arc.write_bytes(&h, sizeof(h));
  // Leave the body as a hole rather than writing 64 MiB.
  arc.f.seekp(static_cast<std::streamoff>(h.payload_length - sizeof(h)), std::ios::cur);
  arc.write_trailer(3);
  arc.finish();

  hrr::Archive a;
  REQUIRE(hrr::load_archive(arc.path(), a));
  CHECK(a.events.size() == 2);
  CHECK(a.truncated);
  CHECK_FALSE(a.complete);
  CHECK_FALSE(hrr::has_clean_trailer(arc.path()));
}

HRR_TEST_CASE(Unit_HRR_Recovery_RepairWritesProcessManifest) {
  TmpRootArchive root("repair_manifest");
  fs::path proc = root.add_process(/*pid=*/4242, /*parent_pid=*/0,
                                   /*records=*/3, /*trailer=*/false,
                                   /*manifest=*/false);
  {
    std::ofstream mf(proc / "manifest.json");
    mf << "{\n"
       << "  \"version\": 1,\n"
       << "  \"capture_mode\": \"in-tree\",\n"
       << "  \"complete\": false,\n"
       << "  \"event_count\": 3,\n"
       << "  \"blob_count\": 0\n"
       << "}\n";
  }

  auto [ret, out] = run_hrr_playback(proc, "--repair");
  INFO("hrr-playback --repair stdout:\n" << out);
  INFO("hrr-playback --repair exit code: " << ret);
  REQUIRE(ret == 0);

  hrr::Archive repaired;
  REQUIRE(hrr::load_archive(proc.string(), repaired));
  CHECK(repaired.complete);
  CHECK_FALSE(repaired.truncated);
  CHECK(repaired.events.size() == 3);

  std::ifstream mf(proc / "manifest.json");
  std::string manifest((std::istreambuf_iterator<char>(mf)),
                       std::istreambuf_iterator<char>());
  INFO("manifest.json:\n" << manifest);
  CHECK(manifest.find("\"pid\": 4242") != std::string::npos);
  CHECK(manifest.find("\"parent_pid\": 0") != std::string::npos);
  CHECK(manifest.find("\"complete\": true") != std::string::npos);
  CHECK(manifest.find("\"capture_mode\"") == std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_Recovery_RepairRootRewritesMiscountedTrailer) {
  TmpRootArchive root("repair_root_miscount");
  root.add_process(/*pid=*/111, /*parent_pid=*/0, /*records=*/2);
  fs::path bad = root.add_process(/*pid=*/222, /*parent_pid=*/111,
                                  /*records=*/4, /*trailer=*/true,
                                  /*manifest=*/true, /*trailer_count=*/3);

  auto [ret, out] = run_hrr_playback(root.root, "--repair");
  INFO("hrr-playback --repair stdout:\n" << out);
  REQUIRE(ret == 0);
  CHECK(out.find("1 repaired, 1 already complete, 0 failed") != std::string::npos);

  hrr::Archive repaired;
  REQUIRE(hrr::load_archive(bad.string(), repaired));
  CHECK(repaired.complete);
  CHECK(repaired.events.size() == 4);
  CHECK(repaired.trailer_events == 4);
}

HRR_TEST_CASE(Unit_HRR_Recovery_RepairRootHonoursMaxHostBytes) {
  TmpRootArchive root("repair_root_cap");
  root.add_process(/*pid=*/111, /*parent_pid=*/0, /*records=*/5, /*trailer=*/false);
  root.add_process(/*pid=*/222, /*parent_pid=*/111, /*records=*/5, /*trailer=*/false);

  auto [ret, out] = hrr_playback_merged(root.root, "--repair --max-host-bytes 1024");
  INFO("hrr-playback --repair output:\n" << out);
  CHECK(ret != 0);
  CHECK(out.find("--max-host-bytes") != std::string::npos);
  CHECK(out.find("0 repaired, 0 already complete, 2 failed") != std::string::npos);

  // One load: 5 records, 64 event slots, and the sort's scratch of 3 slots.
  const uint64_t one_pid = 5 * sizeof(hrr_event_header) + 67 * sizeof(hrr::Event);
  auto [fit_ret, fit_out] = hrr_playback_merged(
      root.root, "--repair --max-host-bytes " + std::to_string(one_pid));
  INFO("hrr-playback --repair output:\n" << fit_out);
  CHECK(fit_ret == 0);
  CHECK(fit_out.find("2 repaired, 0 already complete, 0 failed") != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_Recovery_RootInfoChecksTrailerCount) {
  TmpRootArchive root("root_info_count");
  root.add_process(/*pid=*/111, /*parent_pid=*/0, /*records=*/2);
  root.add_process(/*pid=*/222, /*parent_pid=*/111, /*records=*/4,
                   /*trailer=*/true, /*manifest=*/true, /*trailer_count=*/3);

  auto [ret, out] = run_hrr_playback(root.root, "--info");
  INFO("hrr-playback --info stdout:\n" << out);
  REQUIRE(ret == 0);
  CHECK(root_info_row(out, "pid-111").find(" yes ") != std::string::npos);
  CHECK(root_info_row(out, "pid-222").find(" NO ") != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_Recovery_MultiProcessRootInfo) {
  TmpRootArchive root("multi_process_root");
  root.add_process(/*pid=*/111, /*parent_pid=*/0, /*records=*/2);
  root.add_process(/*pid=*/222, /*parent_pid=*/111, /*records=*/4);

  hrr::Archive archive;
  CHECK_FALSE(hrr::load_archive(root.root.string(), archive));

  auto [ret, out] = run_hrr_playback(root.root, "--info");
  INFO("hrr-playback --info stdout:\n" << out);
  INFO("hrr-playback --info exit code: " << ret);
  REQUIRE(ret == 0);
  CHECK(out.find("Processes:    2") != std::string::npos);
  CHECK(out.find("111") != std::string::npos);
  CHECK(out.find("222") != std::string::npos);
}
