/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

/*
 * hrr_host_rect.h - The host footprint of a pitched copy.
 *
 * Capture snapshots the host side of a pitched copy (hipMemcpy2D, hipMemcpy3D,
 * the hipDrvMemcpy* family) and replay sizes and reads that snapshot back. A
 * capture round-trips only while both measure the host buffer the same way, so
 * the arithmetic lives here, once, and both sides include it.
 *
 * The runtime addresses the host side as a pitched rect, as
 * amd::BufferRect::create() does:
 *   row   = pitch ? pitch : width
 *   slice = pitch*pitch_height ? pitch*pitch_height : row*height
 *   first = z*slice + y*row + x
 * The copy touches `height` rows of `width` bytes in each of `depth` slices and
 * nothing in between. The span from the host base pointer through the last
 * copied byte is
 *   extent = first + (depth-1)*slice + (height-1)*row + width
 * For a copy the runtime accepted, extent >= width*height*depth.
 *
 * Every step is checked. The geometry comes from the application at capture and
 * from the archive at replay, and a wrapped extent would place rows outside the
 * buffer the copy was given, or size a buffer below what the runtime then
 * writes into it.
 */

#include <stddef.h>
#include <stdint.h>
#ifndef __cplusplus
#include <stdbool.h>
#endif

typedef struct HrrHostRect {
    size_t first;   /* offset of the first copied byte from the host base   */
    size_t row;     /* bytes from the start of one row to the next          */
    size_t slice;   /* bytes from the start of one slice to the next        */
    size_t width, height, depth;
    size_t extent;  /* bytes from the host base through the last copied byte;
                     * 0 when nothing is copied                              */
    bool ok;        /* false when the extent does not fit in size_t          */
} HrrHostRect;

/* a*b + c, refusing to wrap. */
static inline bool hrr_size_mad(size_t a, size_t b, size_t c, size_t* out) {
    if (a != 0 && b > (SIZE_MAX - c) / a) return false;
    *out = a * b + c;
    return true;
}

static inline HrrHostRect hrr_host_rect(size_t pitch, size_t pitch_height,
                                        size_t x, size_t y, size_t z,
                                        size_t width, size_t height, size_t depth) {
    HrrHostRect r = {0, 0, 0, width, height, depth, 0, false};
    size_t dense = 0, yx = 0, last = 0;
    if (width == 0 || height == 0 || depth == 0) {
        r.ok = true;  /* an empty copy touches nothing */
        return r;
    }
    r.row = (pitch != 0) ? pitch : width;
    if (r.row < width) r.row = width;  /* defensive: degenerate pitch */
    if (!hrr_size_mad(r.row, height, 0, &dense) ||
        !hrr_size_mad(pitch, pitch_height, 0, &r.slice))
        return r;
    if (r.slice < dense) r.slice = dense;  /* 0 => runtime default */
    if (!hrr_size_mad(y, r.row, x, &yx) ||
        !hrr_size_mad(z, r.slice, yx, &r.first) ||
        !hrr_size_mad(height - 1, r.row, width, &last) ||
        !hrr_size_mad(depth - 1, r.slice, last, &last) ||
        !hrr_size_mad(1, last, r.first, &r.extent))
        return r;
    r.ok = true;
    return r;
}

/* The extent of a rect, or SIZE_MAX, which no buffer covers, when it overflows. */
static inline size_t hrr_host_rect_extent(size_t pitch, size_t pitch_height,
                                          size_t x, size_t y, size_t z,
                                          size_t width, size_t height, size_t depth) {
    const HrrHostRect r = hrr_host_rect(pitch, pitch_height, x, y, z, width, height, depth);
    return r.ok ? r.extent : SIZE_MAX;
}

/* True when the copied bytes of an ok rect are one contiguous run from `first`.
 * Neither product wraps: both are at most the extent. */
static inline bool hrr_host_rect_dense(const HrrHostRect* rect) {
    return (rect->height == 1 || rect->row == rect->width) &&
           (rect->depth == 1 || rect->slice == rect->width * rect->height);
}
