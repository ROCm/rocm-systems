/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include "amd_smi/amdsmi.h"

#pragma pack(1)

#define CPER_MAX_OAM_COUNT  (8)

typedef enum cper_error_severity {
    CPER_SEV_FATAL_UNCORRECTED = 0,
    CPER_SEV_FATAL             = 1,
    CPER_SEV_FATAL_CORRECTED   = 2,

    CPER_SEV_UNUSED            = 10,
};

typedef enum cper_aca_reg {
    CPER_ACA_REG_CTL_LO    = 0,
    CPER_ACA_REG_CTL_HI    = 1,
    CPER_ACA_REG_STATUS_LO = 2,
    CPER_ACA_REG_STATUS_HI = 3,
    CPER_ACA_REG_ADDR_LO   = 4,
    CPER_ACA_REG_ADDR_HI   = 5,
    CPER_ACA_REG_MISC0_LO  = 6,
    CPER_ACA_REG_MISC0_HI  = 7,
    CPER_ACA_REG_CONFIG_LO = 8,
    CPER_ACA_REG_CONFIG_HI = 9,
    CPER_ACA_REG_IPID_LO   = 10,
    CPER_ACA_REG_IPID_HI   = 11,
    CPER_ACA_REG_SYND_LO   = 12,
    CPER_ACA_REG_SYND_HI   = 13,

    CPER_ACA_REG_COUNT     = 32,
};

struct cper_sec_desc {
    uint32_t sec_offset;      /* Offset from the start of CPER entry */
    uint32_t sec_length;
    uint8_t  revision_minor;  /* CPER_SEC_MINOR_REV_1 */
    uint8_t  revision_major;  /* CPER_SEC_MAJOR_REV_22 */
    union {
        struct {
            uint8_t fru_id   : 1;
            uint8_t fru_text : 1;
            uint8_t reserved : 6;
        } valid_bits;
        uint8_t valid_mask;
    };
    uint8_t reserved;
    union {
        struct {
            uint32_t primary              : 1;
            uint32_t reserved1            : 2;
            uint32_t exceed_err_threshold : 1;
            uint32_t latent_err           : 1;  /* "Deferred" error Creation*/
            uint32_t reserved2            : 27;
        } flags_bits;
        uint32_t flags_mask;
    };
    amdsmi_cper_guid_t sec_type;      /* AMD non-Standard, AMD Crashdump */
    char               fru_id[16];    /* FRU Serial ID */
    amdsmi_cper_sev_t  severity;
    char               fru_text[20];  /* "OAM%d" */
};

struct cper_sec_nonstd_err_info {
    amdsmi_cper_guid_t error_type;
    union {
        struct {
            uint64_t ms_chk         : 1;
            uint64_t target_addr_id : 1;
            uint64_t req_id         : 1;
            uint64_t resp_id        : 1;
            uint64_t instr_ptr      : 1;
            uint64_t reserved       : 59;
        } valid_bits;
        uint64_t valid_mask;
    };
    union {
        struct {
            uint64_t err_type_valid       : 1;
            uint64_t pcc_valid            : 1;
            uint64_t uncorr_valid         : 1;
            uint64_t precise_ip_valid     : 1;
            uint64_t restartable_ip_valid : 1;
            uint64_t overflow_valid       : 1;
            uint64_t reserved1            : 10;

            uint64_t err_type       : 2;
            uint64_t pcc            : 1;
            uint64_t uncorr         : 1;
            uint64_t precised_ip    : 1;
            uint64_t restartable_ip : 1;
            uint64_t overflow       : 1;
            uint64_t reserved2      : 41;
        } ms_chk_bits;
        uint64_t ms_chk_mask;
    };

    uint64_t target_addr_id;
    uint64_t req_id;
    uint64_t resp_id;
    uint64_t instr_ptr;
};

struct cper_sec_nonstd_err_ctx {
    uint16_t reg_ctx_type;
    uint16_t reg_arr_size;
    uint32_t msr_addr;
    uint64_t mm_reg_addr;
    uint32_t reg_dump[CPER_ACA_REG_COUNT];  /* This buffer can grow */
};

struct cper_sec_nonstd_err_hdr {
    union {
        struct {
            uint64_t apic_id         : 1;
            uint64_t fw_id           : 1;
            uint64_t err_info_cnt    : 6;  /* should match context_cnt */
            uint64_t err_context_cnt : 6;  /* should match info_cnt */
        } valid_bits;
        uint64_t valid_mask;
    };

    uint64_t apic_id;
    char     fw_id[48];
};

struct cper_sec_nonstd_err_body {
    struct cper_sec_nonstd_err_info err_info;
    struct cper_sec_nonstd_err_ctx  err_ctx;
};

struct cper_sec_nonstd_err {
    struct cper_sec_nonstd_err_hdr hdr;
    struct cper_sec_nonstd_err_body body[];  /* Variable Size, today only 1 entry */
};

struct cper_sec_crashdump_data {
    uint16_t reg_ctx_type;
    uint16_t reg_arr_size;
    uint32_t reserved1;
    uint64_t reserved2;

    union {
        struct {
            uint32_t status_lo;
            uint32_t status_hi;
            uint32_t addr_lo;
            uint32_t addr_hi;
            uint32_t ipid_lo;
            uint32_t ipid_hi;
            uint32_t synd_lo;
            uint32_t synd_hi;
        } fatal_err;

        struct {
            uint64_t msg[CPER_MAX_OAM_COUNT];
        } boot_err;
    } dump;

};

struct cper_sec_crashdump {
    uint64_t reserved1;
    uint64_t reserved2;
    char     fw_id[48];
    uint64_t reserved3[8];

    struct cper_sec_crashdump_data data;
};

struct cper_sec {
    union {
        struct {
            uint8_t fru_id   : 1;
            uint8_t fru_text : 1;
            uint8_t reserved : 6;
        } valid_bits;
        uint8_t valid_mask;
    };

    union {
        struct cper_sec_crashdump  crashdump;
        struct cper_sec_nonstd_err runtime_err;
    };
};

/* General CPER record structure */
struct cper_1_0 {
    struct cper_hdr      *hdr;
    struct cper_sec_desc *sec_desc; /* Variable Size */
    struct cper_sec      *sec;      /* Variable Size */
};

#pragma pack()

amdsmi_status_t amdsmi_get_gpu_cper_entries_by_path(const char *amdgpu_ring_cper_file, uint32_t severity_mask,
                                                    char *cper_data, uint64_t *buf_size, amdsmi_cper_hdr_t **cper_hdrs,
                                                    uint64_t *entry_count, uint64_t *cursor);
std::vector<int> cper_decode(const amdsmi_cper_hdr_t *cper);
