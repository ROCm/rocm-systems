/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Graph Capture State
 * @{
 * @ingroup HRRTest
 * Handler-level tests that replay's graph-capture flag follows the recorded
 * application. The hipStreamEndCapture handler is called through the dispatch
 * table with a record whose stream replay never created, so it throws
 * hrr_unmapped_stream before any HIP call and needs no GPU.
 */

#include <catch2/catch_test_macros.hpp>

#include "hip_playback.h"
#include "hrr/hrr_api_args.h"
#include "hrr_handler_test_util.h"

#include <cstdint>

/**
 * Test Description
 * ----------------
 *   - A recorded hipStreamEndCapture ends the capture whatever it returned:
 *     capture records it as failed only for an invalidated or unjoined capture,
 *     and both still end it. The flag that keeps replay from synchronizing
 *     inside a capture is cleared for a success and for both failures, before
 *     the stream is resolved: a stream not in the map ends the replay too.
 */
TEST_CASE("Unit_HRR_CaptureState_EndCaptureClearsFlag", "[hrr]") {
  for (hipError_t recorded : {hipSuccess, hipErrorStreamCaptureInvalidated,
                              hipErrorStreamCaptureUnjoined}) {
    PlaybackContext ctx;
    ctx.in_graph_capture = true;
    hrr_test::ExactRecord<hrr_args_hipStreamEndCapture> rec;
    rec.get()->ret = static_cast<int32_t>(recorded);
    rec.get()->stream = 0x5000;

    REQUIRE(hrr_playback_dispatch[HRR_API_HIPSTREAMENDCAPTURE] != nullptr);
    INFO("recorded ret " << recorded);
    CHECK_THROWS_AS(hrr_playback_dispatch[HRR_API_HIPSTREAMENDCAPTURE](ctx, rec.bytes()),
                    hrr_unmapped_stream);
    CHECK_FALSE(ctx.in_graph_capture);
  }
}

/** @} */
