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

#include "rocm_smi/rocm_smi_binary_parser.h"
#include "rocm_smi/rocm_smi.h"

#include <dirent.h>
#include <cinttypes>
#include <pthread.h>

#include <cassert>
#include <cstdint>
#include <cstring>

namespace amd::smi {
static uint64_t get_value(uint8_t **ptr, struct metric_field *field) {
    uint64_t v;
    switch (field->field_type) {
        case FIELD_TYPE_U8:
            v = *(uint8_t*)(*ptr);
            ++(*ptr);
            break;
        case FIELD_TYPE_U16:
            v = *(uint16_t*)(*ptr);
            (*ptr) += 2;
            break;
        case FIELD_TYPE_U32:
            v = *(uint32_t*)(*ptr);
            (*ptr) += 4;
            break;
        case FIELD_TYPE_U64:
            v = *(uint64_t*)(*ptr);
            (*ptr) += 8;
            break;
    }
    return v;
}

static int parse_pmmetric_table(uint8_t *buf, struct metric_field *table,
              int32_t buflen, rsmi_name_value_t **kv, uint32_t *kvnum) {
    uint64_t v1;
    int x, y;
    uint8_t *origbuf = buf;
    uint32_t kvsize = 64;

    *kv = reinterpret_cast<rsmi_name_value_t*>(calloc(kvsize, sizeof **kv));
    *kvnum = 0;

    for (x = 0; table[x].field_name; x++) {
        for (y = 0; y < table[x].field_arr_size; y++) {
            v1 = get_value(&buf, &table[x]);
            if ((intptr_t)(buf - origbuf) > buflen) {
                fprintf(stderr,
                    "[ERROR]: Invalid buffer as buffer length exceeded\n");
                return -1;
            }

            if (*kvnum == kvsize) {
                kvsize += 64;
                *kv = reinterpret_cast<rsmi_name_value_t*>
                              (realloc(*kv,kvsize * (sizeof **kv)));
            }
            if (table[x].field_arr_size == 1) {
                sprintf((*kv)[*kvnum].name, "%s", table[x].field_name);
            } else {
                sprintf((*kv)[*kvnum].name, "%s[%d]", table[x].field_name, y);
            }
            (*kv)[*kvnum].value = v1;

            ++(*kvnum);
        }
    }
    return 0;
}

/** present the PM metrics data
 *
 * @dri_instance: which card to pick under /sys/class/drm/card${instance}/device/
 * @kv: pointer to pointer of rsmi_name_value pairs
 * @kvnum: pointer to number of used rsmi_name_value pairs
 */
int present_pmmetrics(const char* fname,
                    rsmi_name_value_t **kv, uint32_t *kvnum)
{
    uint8_t *buf1;
    FILE *infile;
    uint32_t pmmetrics_version;
    metric_field *table;
    int r;
    int32_t len;

    infile = fopen(fname, "rb");
    if (!infile) {
        fprintf(stderr, "[ERROR]: pm_metrics file not found \n");
        return -1;
    }

    buf1 = reinterpret_cast<uint8_t *>(calloc(1, 65536));
    if (!buf1) {
        return -1;
    }

    table = NULL;
    len = static_cast<int32_t>(fread(buf1, 1, 65536, infile));
    fseek(infile, 0, SEEK_SET);
    memcpy(&pmmetrics_version, &buf1[12], 4);

    switch (pmmetrics_version) {
        case 4:   // ??? why 4?
            table = &smu_13_0_6_v8[0];
            break;
        default:
            fprintf(stderr, "Metrics version %d not supported\n"
                , pmmetrics_version);
            return -1;
    }
    r = parse_pmmetric_table(buf1, table, len, kv, kvnum);
    fclose(infile);
    free(buf1);
    return r;
}

static int parse_reg_state_table(uint8_t *buf, int32_t buflen,
            struct metric_field *table,
            rsmi_name_value_t **kv, uint32_t *kvnum) {
    uint64_t skip_smn, x, y, cur_instance, cur_smn,
             num_instance, num_smn, instance_start, smn_start;
    uint64_t v;
    uint8_t *obuf, *origbuf;
    uint32_t kvsize = 64;

    *kv = reinterpret_cast<rsmi_name_value_t*>(calloc(kvsize, sizeof **kv));
    *kvnum = 0;

    skip_smn = cur_instance = num_instance = num_smn = 0;
    instance_start = smn_start = 0x1000;
    x = 0;
    origbuf = buf;
top:
    while (table[x].field_name != NULL) {
        for (y = 0; y < static_cast<uint64_t>(table[x].field_arr_size); y++) {
            obuf = buf;
            v = get_value(&buf, &table[x]);
            if ((intptr_t)(buf - origbuf) > buflen) {
                fprintf(stderr,
                    "[ERROR] Invalid buffer as read length was exceeded\n");
                return -1;
            }
            switch (table[x].field_flag) {
                case FIELD_FLAG_INSTANCE_START:
                    instance_start = x;
                    num_smn = cur_smn = 0;
                    break;
                case FIELD_FLAG_SMN_START:
                    // if we hit an SMN start but there are no registers then skip back to the start
                    // of the instance block
                    if (skip_smn) {
                        // out of instances we're done so bail!
                        if (!num_instance)
                            return 0;
                        x = instance_start;
                        --num_instance;
                        ++cur_instance;
                        // rewind the buffer since we didn't actually consume
                        // this word
                        buf = obuf;
                        goto top;
                    } else {
                        smn_start = x;
                    }
                    break;
                case FIELD_FLAG_NUM_INSTANCE:
                    num_instance = static_cast<int64_t>(v);
                    break;
                case FIELD_FLAG_NUM_SMN:
                    num_smn = static_cast<int64_t>(v);
                    if (v)
                        skip_smn = 0;
                    else
                        skip_smn = 1;
                    break;
            }
            if (*kvnum == kvsize) {
                kvsize += 64;
                *kv = reinterpret_cast<rsmi_name_value_t*>(
                              realloc(*kv, kvsize * (sizeof **kv)));
            }
            sprintf((*kv)[*kvnum].name, "%s", table[x].field_name);
            if (table[x].field_arr_size > 1) {
                sprintf((*kv)[*kvnum].name + strlen((*kv)[*kvnum].name), "[%" PRId64 "]", y);
            }
            if (x >= instance_start)
                sprintf((*kv)[*kvnum].name + strlen((*kv)[*kvnum].name), ".instance[%" PRId64 "]", cur_instance);
            if (x >= smn_start)
                sprintf((*kv)[*kvnum].name + strlen((*kv)[*kvnum].name), ".smn[%" PRId64 "]", cur_smn);
            (*kv)[*kvnum].value = v;
            ++(*kvnum);
        }

        // done move to next or loop
        ++x;
        if (table[x].field_name == NULL && --num_smn) {
            x = smn_start;
            ++cur_smn;
        } else if (table[x].field_name == NULL && --num_instance) {
            x = instance_start;
            ++cur_instance;
        }
    }
    return 0;
}

/** present_reg_state: present register state data
 *
 * @dri_instance: The DRI instance to select under /sys/class/drm/card${instance}/device
 * @table: Name of the table to read {"xgmi', "wafl", "pcie", "usr", "usr_1"}
 * @kv: pointer to pointer of rsmi_name_value pairs
 * @kvnum: pointer to number of used rsmi_name_value pairs
 */
int present_reg_state(const char* fname,
      rsmi_reg_type_t reg_type, rsmi_name_value_t **kv, uint32_t *kvnum) {
    uint8_t buf[4096];
    FILE *infile;
    struct metric_field *tab;
    int32_t len;

    infile = fopen(fname, "rb");
    if (!infile) {
        fprintf(stderr, "[ERROR]: reg_state file not found\n");
        return -1;
    }

    tab = NULL;
    if (reg_type == RSMI_REG_XGMI) {
        fseek(infile, AMDGPU_SYS_REG_STATE_XGMI, SEEK_SET);
        tab = &xgmi_regs[0];
    }
    if (reg_type == RSMI_REG_WAFL) {
        fseek(infile, AMDGPU_SYS_REG_STATE_WAFL, SEEK_SET);
        tab = &wafl_regs[0];
    }
    if (reg_type == RSMI_REG_PCIE) {
        fseek(infile, AMDGPU_SYS_REG_STATE_PCIE, SEEK_SET);
        tab = &pcie_regs[0];
    }
    if (reg_type == RSMI_REG_USR) {
        fseek(infile, AMDGPU_SYS_REG_STATE_USR, SEEK_SET);
         tab = &usr_regs[0];
    }
    if (reg_type == RSMI_REG_USR1) {
        fseek(infile, AMDGPU_SYS_REG_STATE_USR_1, SEEK_SET);
        tab = &usr_regs[0];
    }
    if (!tab) {
        fprintf(stderr, "[ERROR]: Invalid register space named <%d>\n", reg_type);
        fclose(infile);
        return -2;
    }

    len = static_cast<int32_t>(fread(buf, 1, sizeof buf, infile));
    fclose(infile);
    return parse_reg_state_table(buf, len, tab, kv, kvnum);
}


}    //  namespace amd::smi
