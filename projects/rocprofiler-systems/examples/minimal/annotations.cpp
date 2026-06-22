// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include <rocprofiler-systems/categories.h>
#include <rocprofiler-systems/types.h>
#include <rocprofiler-systems/user.h>

#include <cstddef>
#include <cstdint>

int
main()
{
    int64_t size   = 4096;
    int64_t status = 0;

    rocprofsys_annotation_t push_annotations[] = {
        { "size", ROCPROFSYS_INT64, &size },
    };
    rocprofsys_annotation_t pop_annotations[] = {
        { "status", ROCPROFSYS_INT64, &status },
    };

    rocprofsys_user_push_annotated_region("annotated_region", push_annotations,
                                          sizeof(push_annotations) /
                                              sizeof(rocprofsys_annotation_t));

    rocprofsys_user_pop_annotated_region("annotated_region", pop_annotations,
                                         sizeof(pop_annotations) /
                                             sizeof(rocprofsys_annotation_t));

    return 0;
}
