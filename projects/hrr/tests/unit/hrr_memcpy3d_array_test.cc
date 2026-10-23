/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR hipMemcpy3D Array Handles
 * @{
 * @ingroup HRRTest
 * Handler-level tests that the srcArray and dstArray of a replayed
 * hipMemcpy3D, hipMemcpy3DAsync and their _spt forms are array handles replay
 * created, never the address the archive recorded: CLR dereferences those
 * fields, so a recorded address would make the archive choose what the runtime
 * reads. The refusals return before any HIP call, so no GPU is needed.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

hipError_t dispatch(PlaybackContext& ctx, hrr_api_id_t id, const uint8_t* record) {
  REQUIRE(hrr_playback_dispatch[id] != nullptr);
  return hrr_playback_dispatch[id](ctx, record);
}

// What a dispatch did, made in a child process. A handler that wrongly lets an
// unknown array handle reach the HIP runtime crashes inside it; in a child that
// is a failed CHECK on `survived`, not a dead test binary.
struct Outcome {
  bool survived = false;
  int signal = 0;
  hipError_t r = hipSuccess;
  std::string err;
};

Outcome dispatch_isolated(PlaybackContext& ctx, hrr_api_id_t id, const uint8_t* record) {
  const hrr_test::Isolated iso = hrr_test::run_isolated([&] {
    return "\nr=" + std::to_string(static_cast<int>(dispatch(ctx, id, record)));
  });
  Outcome o;
  const size_t at = iso.out.rfind("\nr=");
  o.survived = iso.exited && at != std::string::npos;
  o.signal = iso.signal;
  o.err = iso.out.substr(0, at);
  if (at != std::string::npos) o.r = static_cast<hipError_t>(std::atoi(iso.out.c_str() + at + 3));
  return o;
}

#define CHECK_REFUSED(o)                                                     \
  do {                                                                       \
    INFO("child died of signal " << (o).signal);                             \
    CHECK((o).survived);                                                     \
    CHECK((o).r == hipErrorInvalidValue);                                    \
  } while (0)

struct Spelling {
  hrr_api_id_t id;
  const char* api;
  size_t parms_offset;
};

// The four spellings differ only in their record type; the parms sit at the
// same place in the plain and _spt records of each family.
const Spelling kSpellings[] = {
    {HRR_API_HIPMEMCPY3D, "hipMemcpy3D", offsetof(hrr_args_hipMemcpy3D, parms_bytes)},
    {HRR_API_HIPMEMCPY3DASYNC, "hipMemcpy3DAsync", offsetof(hrr_args_hipMemcpy3DAsync, parms_bytes)},
    {HRR_API_HIPMEMCPY3D_SPT, "hipMemcpy3D_spt", offsetof(hrr_args_hipMemcpy3D_spt, parms_bytes)},
    {HRR_API_HIPMEMCPY3DASYNC_SPT, "hipMemcpy3DAsync_spt",
     offsetof(hrr_args_hipMemcpy3DAsync_spt, parms_bytes)},
};

size_t record_size(hrr_api_id_t id) { return hrr_api_min_payload_size[id]; }

// A device-to-device copy: the pointers are recorded addresses in no map, so
// without the array check the copy would reach HIP.
std::vector<uint8_t> make_record(const Spelling& s, uint64_t src_array, uint64_t dst_array) {
  std::vector<uint8_t> rec(record_size(s.id), 0);
  hipMemcpy3DParms parms{};
  parms.srcArray = reinterpret_cast<hipArray_t>(src_array);
  parms.dstArray = reinterpret_cast<hipArray_t>(dst_array);
  parms.extent = make_hipExtent(16, 2, 1);
  parms.kind = hipMemcpyDeviceToDevice;
  std::memcpy(rec.data() + s.parms_offset, &parms, sizeof(parms));
  return rec;
}

constexpr uint64_t kRecordedSrc = 0x7f1100000000ull;
constexpr uint64_t kRecordedDst = 0x7f2200000000ull;

}  // namespace

/**
 * Test Description
 * ----------------
 *   - A copy whose srcArray or dstArray was never created at replay is refused
 *     with hipErrorInvalidValue, naming the end and the recorded address, in
 *     all four spellings. Nothing reaches HIP.
 *   - A handle that was created for the other end does not excuse the missing
 *     one.
 */
TEST_CASE("Unit_HRR_Memcpy3DArray_UnknownHandleIsRefused", "[hrr]") {
  PlaybackContext ctx;
  // Only the destination array exists at replay.
  ctx.record_array(kRecordedDst, reinterpret_cast<hipArray_t>(0x1000));

  for (const Spelling& s : kSpellings) {
    INFO(s.api);
    {  // srcArray never created
      auto rec = make_record(s, kRecordedSrc, kRecordedDst);
      const Outcome o = dispatch_isolated(ctx, s.id, rec.data());
      CHECK_REFUSED(o);
      CHECK(o.err.find(s.api) != std::string::npos);
      CHECK(o.err.find("source array 0x7f1100000000") != std::string::npos);
    }
    {  // dstArray never created
      auto rec = make_record(s, kRecordedDst, kRecordedSrc);
      const Outcome o = dispatch_isolated(ctx, s.id, rec.data());
      CHECK_REFUSED(o);
      CHECK(o.err.find("destination array 0x7f1100000000") != std::string::npos);
    }
  }
}

/**
 * Test Description
 * ----------------
 *   - A host-to-device and a device-to-host copy with an unknown array handle
 *     are refused too: the check does not depend on the copy kind.
 */
TEST_CASE("Unit_HRR_Memcpy3DArray_UnknownHandleIsRefusedForEveryKind", "[hrr]") {
  PlaybackContext ctx;
  for (hipMemcpyKind kind : {hipMemcpyHostToDevice, hipMemcpyDeviceToHost}) {
    for (const Spelling& s : kSpellings) {
      auto rec = make_record(s, kRecordedSrc, 0);
      hipMemcpy3DParms parms{};
      std::memcpy(&parms, rec.data() + s.parms_offset, sizeof(parms));
      parms.kind = kind;
      std::memcpy(rec.data() + s.parms_offset, &parms, sizeof(parms));
      const Outcome o = dispatch_isolated(ctx, s.id, rec.data());
      INFO(s.api << " kind " << kind);
      CHECK_REFUSED(o);
    }
  }
}

/**
 * Test Description
 * ----------------
 *   - A host-to-array or array-to-host copy whose array replay did create is not
 *     issued: the host rect is sized in bytes and the copy in array elements, so
 *     issuing it would read past the blob or write past the scratch buffer. The
 *     handler returns hipSuccess and says so on stderr, in all four spellings,
 *     and nothing reaches HIP (the live handle here is not a real array).
 */
TEST_CASE("Unit_HRR_Memcpy3DArray_HostCopiesAreNotIssued", "[hrr]") {
  PlaybackContext ctx;
  ctx.record_array(kRecordedSrc, reinterpret_cast<hipArray_t>(0x1000));

  for (hipMemcpyKind kind : {hipMemcpyHostToDevice, hipMemcpyDeviceToHost}) {
    for (const Spelling& s : kSpellings) {
      auto rec = kind == hipMemcpyHostToDevice ? make_record(s, 0, kRecordedSrc)
                                               : make_record(s, kRecordedSrc, 0);
      hipMemcpy3DParms parms{};
      std::memcpy(&parms, rec.data() + s.parms_offset, sizeof(parms));
      parms.kind = kind;
      std::memcpy(rec.data() + s.parms_offset, &parms, sizeof(parms));
      const Outcome o = dispatch_isolated(ctx, s.id, rec.data());
      INFO(s.api << " kind " << kind << ", child died of signal " << o.signal);
      CHECK(o.survived);
      CHECK(o.r == hipSuccess);
      CHECK(o.err.find("between host memory and an array is not replayed") != std::string::npos);
    }
  }
}

/**
 * Test Description
 * ----------------
 *   - The translation step replaces a recorded srcArray and dstArray with the
 *     live handles, leaves a null field null, and does not touch the pointers.
 *   - It reports false, and names the end, when either handle is unknown.
 */
TEST_CASE("Unit_HRR_Memcpy3DArray_TranslatesRecordedHandles", "[hrr]") {
  PlaybackContext ctx;
  auto* const live_src = reinterpret_cast<hipArray_t>(0x1000);
  auto* const live_dst = reinterpret_cast<hipArray_t>(0x2000);
  ctx.record_array(kRecordedSrc, live_src);
  ctx.record_array(kRecordedDst, live_dst);

  hipMemcpy3DParms parms{};
  parms.srcArray = reinterpret_cast<hipArray_t>(kRecordedSrc);
  parms.dstArray = reinterpret_cast<hipArray_t>(kRecordedDst);
  parms.srcPtr.ptr = reinterpret_cast<void*>(0xabc0);
  REQUIRE(hrr_translate_memcpy3d_arrays(ctx, "hipMemcpy3D", parms));
  CHECK(parms.srcArray == live_src);
  CHECK(parms.dstArray == live_dst);
  CHECK(parms.srcPtr.ptr == reinterpret_cast<void*>(0xabc0));

  hipMemcpy3DParms none{};
  REQUIRE(hrr_translate_memcpy3d_arrays(ctx, "hipMemcpy3D", none));
  CHECK(none.srcArray == nullptr);
  CHECK(none.dstArray == nullptr);

  hipMemcpy3DParms one{};
  one.dstArray = reinterpret_cast<hipArray_t>(kRecordedDst);
  REQUIRE(hrr_translate_memcpy3d_arrays(ctx, "hipMemcpy3D", one));
  CHECK(one.srcArray == nullptr);
  CHECK(one.dstArray == live_dst);

  hipMemcpy3DParms bad{};
  bad.srcArray = reinterpret_cast<hipArray_t>(kRecordedSrc);
  bad.dstArray = reinterpret_cast<hipArray_t>(0x7f3300000000ull);
  bool ok = true;
  const std::string err = hrr_test::capture_stderr(
      [&] { ok = hrr_translate_memcpy3d_arrays(ctx, "hipMemcpy3D", bad); });
  CHECK_FALSE(ok);
  CHECK(err.find("destination array 0x7f3300000000") != std::string::npos);
}

/**
 * Test Description
 * ----------------
 *   - The driver-API graph copies (hipDrvGraphAddMemcpyNode,
 *     hipDrvGraphMemcpyNodeSetParams, hipDrvGraphExecMemcpyNodeSetParams) do
 *     not pass a null array to HIP when the recorded array was never created:
 *     the copy is skipped, the stderr line names the API, the end and the
 *     recorded address, and the graph the node belonged to is marked
 *     incomplete where the API adds a node.
 */
TEST_CASE("Unit_HRR_DrvMemcpy3DArray_UnknownHandleIsRefused", "[hrr]") {
  constexpr uint64_t kGraph = 0x7f4400000000ull;
  constexpr uint64_t kNode = 0x7f5500000000ull;
  constexpr uint64_t kExec = 0x7f6600000000ull;

  PlaybackContext ctx;
  ctx.record_graph(kGraph, reinterpret_cast<hipGraph_t>(0x3000));
  ctx.record_graph_node(kNode, reinterpret_cast<hipGraphNode_t>(0x3100));
  ctx.record_graph_exec(kExec, reinterpret_cast<hipGraphExec_t>(0x3200));

  for (bool src_side : {true, false}) {
    HIP_MEMCPY3D copy{};
    (src_side ? copy.srcMemoryType : copy.dstMemoryType) = hipMemoryTypeArray;
    (src_side ? copy.dstMemoryType : copy.srcMemoryType) = hipMemoryTypeDevice;
    (src_side ? copy.srcArray : copy.dstArray) = reinterpret_cast<hipArray_t>(kRecordedSrc);
    const char* end = src_side ? "source array 0x7f1100000000" : "destination array 0x7f1100000000";
    INFO(end);

    {
      hrr_test::ExactRecord<hrr_args_hipDrvGraphAddMemcpyNode> r;
      r.get()->hGraph = kGraph;
      r.get()->copyParams_present = 1;
      std::memcpy(r.get()->copyParams_bytes, &copy, sizeof(copy));
      const Outcome o = dispatch_isolated(ctx, HRR_API_HIPDRVGRAPHADDMEMCPYNODE, r.bytes());
      INFO("child died of signal " << o.signal);
      CHECK(o.survived);
      CHECK(o.err.find("hipDrvGraphAddMemcpyNode") != std::string::npos);
      CHECK(o.err.find(end) != std::string::npos);
    }
    {
      hrr_test::ExactRecord<hrr_args_hipDrvGraphMemcpyNodeSetParams> r;
      r.get()->hNode = kNode;
      r.get()->nodeParams_present = 1;
      std::memcpy(r.get()->nodeParams_bytes, &copy, sizeof(copy));
      const Outcome o = dispatch_isolated(ctx, HRR_API_HIPDRVGRAPHMEMCPYNODESETPARAMS, r.bytes());
      INFO("child died of signal " << o.signal);
      CHECK(o.survived);
      CHECK(o.err.find("hipDrvGraphMemcpyNodeSetParams") != std::string::npos);
      CHECK(o.err.find(end) != std::string::npos);
    }
    {
      hrr_test::ExactRecord<hrr_args_hipDrvGraphExecMemcpyNodeSetParams> r;
      r.get()->hGraphExec = kExec;
      r.get()->hNode = kNode;
      r.get()->copyParams_present = 1;
      std::memcpy(r.get()->copyParams_bytes, &copy, sizeof(copy));
      const Outcome o =
          dispatch_isolated(ctx, HRR_API_HIPDRVGRAPHEXECMEMCPYNODESETPARAMS, r.bytes());
      INFO("child died of signal " << o.signal);
      CHECK(o.survived);
      CHECK(o.err.find("hipDrvGraphExecMemcpyNodeSetParams") != std::string::npos);
      CHECK(o.err.find(end) != std::string::npos);
    }
  }
}

/**
 * Test Description
 * ----------------
 *   - hipMemcpy3DBatchAsync refuses a batch in which an array operand, source
 *     or destination, names a handle replay never created, with
 *     hipErrorInvalidValue and a stderr line naming the end and address.
 */
TEST_CASE("Unit_HRR_Memcpy3DBatchArray_UnknownHandleIsRefused", "[hrr]") {
  PlaybackContext ctx;
  ctx.record_array(kRecordedDst, reinterpret_cast<hipArray_t>(0x1000));

  for (bool src_side : {true, false}) {
    hrr_test::ExactRecord<hrr_args_hipMemcpy3DBatchAsync> r;
    hipMemcpy3DBatchOp op{};
    op.src.type = hipMemcpyOperandTypeArray;
    op.dst.type = hipMemcpyOperandTypeArray;
    op.src.op.array.array = reinterpret_cast<hipArray_t>(src_side ? kRecordedSrc : kRecordedDst);
    op.dst.op.array.array = reinterpret_cast<hipArray_t>(src_side ? kRecordedDst : kRecordedSrc);
    op.extent = make_hipExtent(16, 2, 1);
    r.get()->opList_present = 1;
    r.get()->opList_n = 1;
    r.get()->numOps = 1;
    std::memcpy(r.get()->opList_bytes, &op, sizeof(op));
    const Outcome o = dispatch_isolated(ctx, HRR_API_HIPMEMCPY3DBATCHASYNC, r.bytes());
    CHECK_REFUSED(o);
    CHECK(o.err.find("hipMemcpy3DBatchAsync") != std::string::npos);
    CHECK(o.err.find(src_side ? "source array 0x7f1100000000" : "destination array 0x7f1100000000") !=
          std::string::npos);
  }
}

/** @} */
