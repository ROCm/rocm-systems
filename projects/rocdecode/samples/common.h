/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include "roc_video_dec.h"
#include "md5.h"

typedef enum ReconfigFlushMode_enum {
    RECONFIG_FLUSH_MODE_NONE = 0x0,                      /**<  Just flush to get the frame count */
    RECONFIG_FLUSH_MODE_DUMP_TO_FILE = 0x1,              /**<  The remaining frames will be dumped to file in this mode */
    RECONFIG_FLUSH_MODE_CALCULATE_MD5 = (0x1 << 1),      /**<  Calculate the MD5 of the flushed frames */
} ReconfigFlushMode;

// This struct is used by sample apps to dump last frames to file
typedef struct ReconfigDumpFileStruct_t {
    bool b_dump_frames_to_file;
    std::string output_file_name;
    void *md5_generator_handle;
} ReconfigDumpFileStruct;


// callback function to flush last frames and save it to file when reconfigure happens
int ReconfigureFlushCallback(void *p_viddec_obj, uint32_t flush_mode, void *p_user_struct) {
    int n_frames_flushed = 0;
    if ((p_viddec_obj == nullptr) ||  (p_user_struct == nullptr)) return n_frames_flushed;

    RocVideoDecoder *viddec = static_cast<RocVideoDecoder *> (p_viddec_obj);
    OutputSurfaceInfo *surf_info;
    if (!viddec->GetOutputSurfaceInfo(&surf_info)) {
        std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
        return n_frames_flushed;
    }

    uint8_t *pframe = nullptr;
    int64_t pts;
    while ((pframe = viddec->GetFrame(&pts))) {
        if (flush_mode != RECONFIG_FLUSH_MODE_NONE) {
            ReconfigDumpFileStruct *p_dump_file_struct = static_cast<ReconfigDumpFileStruct *>(p_user_struct);
            if (flush_mode & ReconfigFlushMode::RECONFIG_FLUSH_MODE_DUMP_TO_FILE) {
                if (p_dump_file_struct->b_dump_frames_to_file) {
                    viddec->SaveFrameToFile(p_dump_file_struct->output_file_name, pframe, surf_info);
                }
            }
            if (flush_mode & ReconfigFlushMode::RECONFIG_FLUSH_MODE_CALCULATE_MD5) {
                MD5Generator *md5_generator = static_cast<MD5Generator*>(p_dump_file_struct->md5_generator_handle);
                md5_generator->UpdateMd5ForFrame(pframe, surf_info);
            }
        }
        // release and flush frame
        viddec->ReleaseFrame(pts, true);
        n_frames_flushed++;
    }

    return n_frames_flushed;
}

int GetEnvVar(const char *name, int &dev_count) {
    char *v = std::getenv(name);
    if (v) {
        char* p_tkn = std::strtok(v, ",");
        while (p_tkn != nullptr) {
            dev_count++;
            p_tkn = strtok(nullptr, ",");
        }
    }
    return dev_count;
}