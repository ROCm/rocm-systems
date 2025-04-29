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

#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include <memory>
#include <cstring>
#include <sstream>
#include <vector>

#include "aca-decode/aca_decode.h"
#include "amd_smi/impl/amd_smi_cper.h"
#include "rocm_smi/rocm_smi_logger.h"

namespace {
static std::vector<const amdsmi_cper_hdr_t *>
amdsmi_get_gpu_cper_headers(const char *buffer, size_t buffer_sz) {

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
        << "[CPER] buffer_sz: " << buffer_sz;
    LOG_DEBUG(ss);

    std::vector<const amdsmi_cper_hdr_t *> headers;
    if(!buffer) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
            << "[CPER] buffer is null";
        LOG_ERROR(ss);
        return headers;
    }
    static constexpr char cper_signature[] = "CPER";
    static constexpr size_t cper_signature_size = sizeof(cper_signature) - 1;
    for(size_t data_idx = 0;
        buffer_sz >= cper_signature_size &&
        data_idx < buffer_sz - cper_signature_size;
        ++data_idx) {

        const amdsmi_cper_hdr_t *hdr = reinterpret_cast<const amdsmi_cper_hdr_t *>(
            &buffer[data_idx]);
        if(hdr->signature[0] != 'C' || hdr->signature[1] != 'P' ||
            hdr->signature[2] != 'E' || hdr->signature[3] != 'R' ) {
            continue;
        }
        if(hdr->signature_end != 0xFFFFFFFF) {
            continue;
        }
        if(hdr->record_length > buffer_sz) {
            continue;
        }
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
            << "[CPER] add header at data_idx: " << data_idx
            << ", sig: " << hdr->signature[0] << hdr->signature[1] << hdr->signature[2] << hdr->signature[3];
        LOG_DEBUG(ss);
        headers.emplace_back(hdr);
    }
    return headers;
}

struct CperFileCtx {
    amdsmi_status_t status = AMDSMI_STATUS_FILE_ERROR;
    std::unique_ptr<char[]> buffer;
    long file_size = 0;
};

static auto amdsmi_read_cper_file(const std::string &filepath) -> CperFileCtx {

    std::ostringstream ss;

    CperFileCtx ctx;
    ctx.status = AMDSMI_STATUS_FILE_ERROR;
    ctx.file_size = 0;

    struct stat file_stats;
    if (stat(filepath.c_str(), &file_stats) == 0) {
        if (!S_ISREG(file_stats.st_mode)) {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] file is not a regular file: "
                << filepath << ", errno: " << errno << "): " << strerror(errno);
            return ctx;
        }
    } else {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] file does not exist: "
            << filepath << ", errno: " << errno << "): " << strerror(errno);
        ctx.status = AMDSMI_STATUS_NOT_SUPPORTED;
        return ctx;
    }

    ctx.file_size = file_stats.st_size;
    ctx.buffer = std::make_unique<char[]>(ctx.file_size);
    int file = open(filepath.c_str(), O_RDONLY);
    if (file == -1) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] failed to open file: "
            << filepath << ", errno:()" << errno << "): " << strerror(errno);
        LOG_ERROR(ss);
        return ctx;
    }
    long bytes_read = read(file, ctx.buffer.get(), ctx.file_size);
    if (bytes_read <= 0) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
            << "[CPER] failed to read complete file, read only  "
            << bytes_read << " of " << ctx.file_size << " bytes";
        LOG_ERROR(ss);
        return ctx;
    }
    close(file);

    ctx.status = AMDSMI_STATUS_SUCCESS;
    ctx.file_size = bytes_read;
    return ctx;
}

#define GUID_INIT(a, b, c, d0, d1, d2, d3, d4, d5, d6, d7)          \
{ (a) & 0xff, ((a) >> 8) & 0xff, ((a) >> 16) & 0xff, ((a) >> 24) & 0xff, \
   (b) & 0xff, ((b) >> 8) & 0xff,                   \
   (c) & 0xff, ((c) >> 8) & 0xff,                   \
   (d0), (d1), (d2), (d3), (d4), (d5), (d6), (d7) };

/* Machine Check Exception */
#define CPER_NOTIFY_MCE                         \
    GUID_INIT(0xE8F56FFE, 0x919C, 0x4cc5, 0xBA, 0x88, 0x65, 0xAB,   \
          0xE1, 0x49, 0x13, 0xBB)
#define CPER_NOTIFY_CMC                         \
    GUID_INIT(0x2DCE8BB1, 0xBDD7, 0x450e, 0xB9, 0xAD, 0x9C, 0xF4,   \
          0xEB, 0xD4, 0xF8, 0x90)
#define BOOT_TYPE                               \
    GUID_INIT(0x3D61A466, 0xAB40, 0x409a, 0xA6, 0x98,0xF3, 0x62,    \
          0xD4, 0x64, 0xB3, 0x8F)
#define AMD_OOB_CRASHDUMP                       \
    GUID_INIT(0x32AC0C78, 0x2623, 0x48F6, 0xB0, 0xD0, 0x73, 0x65,   \
          0x72, 0x5F, 0xD6, 0xAE)
#define AMD_GPU_NONSTANDARD_ERROR                       \
    GUID_INIT(0x32AC0C78, 0x2623, 0x48F6, 0x81, 0xA2, 0xAC, 0x69,   \
          0x17, 0x80, 0x55, 0x1D)          
#define PROC_ERR_SECTION_TYPE                   \
    GUID_INIT(0xDC3EA0B0, 0xA144, 0x4797, 0xB9, 0x5B, 0x53, 0xFA,   \
          0x24, 0x2B, 0x6E, 0x1D)

static amdsmi_cper_guid_t mce = CPER_NOTIFY_MCE;
static amdsmi_cper_guid_t cmc = CPER_NOTIFY_CMC;
static amdsmi_cper_guid_t bt = BOOT_TYPE;
static amdsmi_cper_guid_t cr = AMD_OOB_CRASHDUMP;
static amdsmi_cper_guid_t nonstd = AMD_GPU_NONSTANDARD_ERROR;
static amdsmi_cper_guid_t proc_err = PROC_ERR_SECTION_TYPE;

static int cper_is_cr(const amdsmi_cper_guid_t *guid)
{
    return !memcmp(&cr, guid, sizeof(amdsmi_cper_guid_t));
}

static int cper_is_nonstd(const amdsmi_cper_guid_t *guid)
{
    return !memcmp(&nonstd, guid, sizeof(amdsmi_cper_guid_t));
}

static int cper_is_proc_err(const amdsmi_cper_guid_t *guid)
{
    return !memcmp(&proc_err, guid, sizeof(amdsmi_cper_guid_t));
}

static int cper_is_bt(const amdsmi_cper_guid_t *guid)
{
    return !memcmp(&bt, guid, sizeof(amdsmi_cper_guid_t));
}

static int cper_num_sec(const amdsmi_cper_hdr_t *hdr)
{
    return hdr->sec_cnt;
}

static const amdsmi_cper_guid_t *get_sec_desc_type(const struct cper_sec_desc *desc)
{
    return &desc->sec_type;
}

static const amdsmi_cper_guid_t *get_cper_type(const amdsmi_cper_hdr_t *hdr)
{
    return &hdr->notify_type;
}

static void* cper_get_sec_desc_offset(const amdsmi_cper_hdr_t *hdr, int idx)
{
    char *offset;

    if (idx >= hdr->sec_cnt)
        return 0;

    offset = (char *)hdr + sizeof(amdsmi_cper_hdr_t);
    offset += sizeof(struct cper_sec_desc) * idx;

    return offset;
}

static void* cper_get_sec_offset(const amdsmi_cper_hdr_t *hdr, int idx)
{
    struct cper_sec_desc *tmp_desc;
    char *offset;

    if (idx >= hdr->sec_cnt)
        return 0;

    tmp_desc = reinterpret_cast<struct cper_sec_desc *>(
        (char *)hdr + sizeof(amdsmi_cper_hdr_t) + sizeof(struct cper_sec_desc) * idx
    );

    return (char *)hdr + tmp_desc->sec_offset;
}

static int cper_dump_sec_desc(const struct cper_sec_desc *desc)
{
    std::ostringstream ss;

    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS]\n~~~~SECTION DESCRIPTION~~~\n";

    ss << "[SEC DESC] REV Major = 0x" << std::hex << static_cast<int>(desc->revision_major) << "\n";
    ss << "[SEC DESC] REV Minor = 0x" << std::hex << static_cast<int>(desc->revision_minor) << "\n";
    ss << "[SEC DESC] Length    = 0x" << std::hex << desc->sec_length << "\n";
    ss << "[SEC DESC] Offset    = 0x" << std::hex << desc->sec_offset << "\n";

    ss << "[SEC DESC] fru_id    = " << desc->fru_id << "\n";
    ss << "[SEC DESC] fru_text  = " << desc->fru_text << "\n";
    
    ss << std::dec << "\n";
    
    if (cper_is_cr(&desc->sec_type))
        ss << "[SEC DESC] AMD CrashDump Section\n";
    else if (cper_is_nonstd(&desc->sec_type))
        ss << "[SEC DESC] AMD NonStandard Section\n";
    else if (cper_is_proc_err(&desc->sec_type))
        ss << "[SEC DESC] AMD Proc Error Section\n";
    else
        ss << "UNKNOWN ERROR TYPE!!\n";

    ss << "~~~~SECTION DESCRIPTION~~~\n\n";

    LOG_DEBUG(ss);
    return 0;
}

static int aca_decode_fatal(const cper_sec_crashdump_data &data) 
{
    std::ostringstream ss;

    const uint64_t *register_array = reinterpret_cast<const uint64_t *>(&data.dump.fatal_err);
    aca_raw_data_t raw_data;
    raw_data.aca_status = register_array[0];
    raw_data.aca_ipid = register_array[2];
    raw_data.aca_synd = register_array[3];
    
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] aca_status: 0x" << std::hex << raw_data.aca_status << "\n";
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] aca_ipid: 0x" << std::hex << raw_data.aca_ipid << "\n";
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] aca_synd: 0x" << std::hex << raw_data.aca_synd << "\n";

    raw_data.flags = 0;
    raw_data.hw_revision = 1;

    aca_error_info_t error_info = aca_decode(&raw_data);
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] fatal error_info.afid: " << std::dec << error_info.afid << "\n";
    LOG_DEBUG(ss);

    return error_info.afid;
}

static int aca_decode_corrected_error(const uint32_t *reg_dump, size_t num_bytes)  {

    std::ostringstream ss;
    if(num_bytes != CPER_ACA_REG_COUNT * sizeof(uint32_t)) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] Size of register array must be " << std::dec << (CPER_ACA_REG_COUNT * sizeof(uint32_t)) << " bytes\n";
        LOG_ERROR(ss);
        return AMDSMI_STATUS_INVAL;
    }
    const uint64_t *register_array = reinterpret_cast<const uint64_t *>(reg_dump);
    aca_raw_data_t raw_data;
    raw_data.aca_status = register_array[2];
    raw_data.aca_ipid = register_array[5];
    raw_data.aca_synd = register_array[6];
    
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] aca_status: 0x" << std::hex << raw_data.aca_status << "\n";
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] aca_ipid: 0x" << std::hex << raw_data.aca_ipid << "\n";
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] aca_synd: 0x" << std::hex << raw_data.aca_synd << "\n";

    raw_data.flags = 0;
    raw_data.hw_revision = 1;

    aca_error_info_t error_info = aca_decode(&raw_data);
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] non-fatal error_info.afid: " << std::dec << error_info.afid << "\n";
    LOG_DEBUG(ss);

    return error_info.afid;
}

static int cper_dump_nonstd_err(const struct cper_sec_nonstd_err *nonstd_err)
{
    std::ostringstream ss;

    struct cper_sec_nonstd_err_body *body;
    char *offset;

    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS]\n~~~~NON STANDARD SECTION~~~\n";

    ss << "[NonSTD SEC] Err Info Count    = 0x" << std::hex << nonstd_err->hdr.valid_bits.err_info_cnt << "\n";
    ss << "[NonSTD SEC] Err Context Count = 0x" << std::hex << nonstd_err->hdr.valid_bits.err_context_cnt << "\n";

    if (nonstd_err->hdr.valid_bits.err_context_cnt != nonstd_err->hdr.valid_bits.err_context_cnt) {
        ss << "~~~~Malformed Non Standard Section!~~~~\n\n";
        goto exit;
    }

    body = reinterpret_cast<struct cper_sec_nonstd_err_body *>(
        (char *)nonstd_err + sizeof(struct cper_sec_nonstd_err_hdr)
    );

    ss << "[NonSTD SEC] Reg Ctx Type   = 0x" << std::hex << body->err_ctx.reg_ctx_type << "\n";
    ss << "[NonSTD SEC] Reg Array Size = 0x" << std::hex << body->err_ctx.reg_arr_size << "\n";

    for (int i = 0; i < CPER_ACA_REG_COUNT; i++) {
        ss << "[NonSTD SEC] reg_dump[" << std::dec << i << "] = 0x" << std::hex << body->err_ctx.reg_dump[i] << "\n";
    }
    
exit:    
    ss << std::dec << "~~~~NON STANDARD SECTION~~~\n\n";

    LOG_DEBUG(ss);

    return aca_decode_corrected_error(body->err_ctx.reg_dump, sizeof(body->err_ctx.reg_dump));
}

static int cper_dump_cr_fatal(const struct cper_sec_crashdump *crashdump)
{
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS]\n~~~~CRASH DUMP - FATAL~~~\n";

    ss << "[Crash Dump - Fatal] status_lo = 0x" << std::hex << crashdump->data.dump.fatal_err.status_lo << "\n";
    ss << "[Crash Dump - Fatal] status_hi = 0x" << std::hex << crashdump->data.dump.fatal_err.status_hi << "\n";
    ss << "[Crash Dump - Fatal] addr_lo   = 0x" << std::hex << crashdump->data.dump.fatal_err.addr_lo << "\n";
    ss << "[Crash Dump - Fatal] addr_hi   = 0x" << std::hex << crashdump->data.dump.fatal_err.addr_hi << "\n";
    ss << "[Crash Dump - Fatal] ipid_lo   = 0x" << std::hex << crashdump->data.dump.fatal_err.ipid_lo << "\n";
    ss << "[Crash Dump - Fatal] ipid_hi   = 0x" << std::hex << crashdump->data.dump.fatal_err.ipid_hi << "\n";
    ss << "[Crash Dump - Fatal] synd_lo   = 0x" << std::hex << crashdump->data.dump.fatal_err.synd_lo << "\n";
    ss << "[Crash Dump - Fatal] synd_hi   = 0x" << std::hex << crashdump->data.dump.fatal_err.synd_hi << "\n";

    ss << std::dec << "~~~~CRASH DUMP - FATAL~~~\n\n";

    LOG_DEBUG(ss);

    return aca_decode_fatal(crashdump->data);
}

static int cper_dump_cr_boot(const struct cper_sec_crashdump *crashdump)
{
    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS]\n~~~~CRASH DUMP - BOOT TIME~~~\n";

    for (int i = 0; i < CPER_MAX_OAM_COUNT; i++) {
        ss << "[Crash Dump - Boot] bootmsg[" << std::dec << i << "] = 0x" << std::hex << crashdump->data.dump.boot_err.msg[i] << "\n";
    }

    ss << "~~~~CRASH DUMP - BOOT TIME~~~\n\n";
    LOG_DEBUG(ss);

    return aca_decode_fatal(crashdump->data);
}

} //namespace 

amdsmi_status_t amdsmi_get_gpu_cper_entries_by_path(
    const char *amdgpu_ring_cper_file,
    uint32_t severity_mask,
    char *cper_data,
    uint64_t *buf_size,
    amdsmi_cper_hdr_t **cper_hdrs,
    uint64_t *entry_count,
    uint64_t *cursor) {

    std::ostringstream ss;
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] begin\n"
        << ", amdgpu_ring_cper_file: " << amdgpu_ring_cper_file
        << ", severity_mask: " << severity_mask;
    LOG_DEBUG(ss);

    if(!cper_data) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] cper_data should be a valid memory address\n";
        LOG_ERROR(ss);
        if(entry_count) {*entry_count = 0;}
        if(buf_size) { *buf_size = 0; }
        return AMDSMI_STATUS_OUT_OF_RESOURCES;
    }
    else if(!buf_size) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] buf_size should be a valid memory address";
        LOG_ERROR(ss);
        if(entry_count) {*entry_count = 0;}
        return AMDSMI_STATUS_OUT_OF_RESOURCES;
    }
    else if(!entry_count) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] entry_count should be a valid memory address";
        LOG_ERROR(ss);
        *buf_size = 0;
        return AMDSMI_STATUS_OUT_OF_RESOURCES;
    }
    else if(!*buf_size) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] buf_size should be greater than zero";
        LOG_ERROR(ss);
        *entry_count = 0;
        return AMDSMI_STATUS_OUT_OF_RESOURCES;
    }
    else if(!*entry_count) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] entry_count should be greater than 0";
        LOG_ERROR(ss);
        *buf_size = 0;
        return AMDSMI_STATUS_OUT_OF_RESOURCES;
    }
    else if(!cper_hdrs) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] cper_hdrs should be a valid memory address";
        LOG_ERROR(ss);
        *entry_count = 0;
        *buf_size = 0;
        return AMDSMI_STATUS_OUT_OF_RESOURCES;
    }
    else if(!cursor) {
        ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] cursor should be a valid memory address";
        LOG_ERROR(ss);
        *entry_count = 0;
        *buf_size = 0;
        return AMDSMI_STATUS_OUT_OF_RESOURCES;
    }

    auto ctx = amdsmi_read_cper_file(amdgpu_ring_cper_file);
    if(ctx.status != AMDSMI_STATUS_SUCCESS) {
        *entry_count = 0;
        *buf_size = 0;
        return ctx.status;
    }

    auto headers = amdsmi_get_gpu_cper_headers(ctx.buffer.get(), ctx.file_size);
    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] num headers: " << headers.size();
    LOG_DEBUG(ss);

    uint64_t data_idx = 0;
    uint64_t header_idx = 0;
    size_t num_headers_copied = 0;
    for(const amdsmi_cper_hdr_t *header: headers) {
        if(((1 << header->error_severity) & severity_mask) !=
            static_cast<uint32_t>(1 << header->error_severity)) {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] cper header rejected with severity: 0x"
                << std::hex << (1 << header->error_severity) << ", given severity_mask: 0x"
                << std::hex << severity_mask << ", record_length:"
                << std::dec << header->record_length;
            LOG_DEBUG(ss);
            continue;
        }
        else {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] cper header accepted with severity: 0x"
                << std::hex << (1 << header->error_severity) << ", given severity_mask: 0x"
                << std::hex << severity_mask << ", record_length:"
                << std::dec << header->record_length;
            LOG_DEBUG(ss);
        }
        if((*buf_size - data_idx) < header->record_length ) {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] buffer filled up without copying all cper entries, buf_size: " << std::dec << *buf_size;
            LOG_ERROR(ss);
            *entry_count = num_headers_copied;
            *buf_size = data_idx;
            return (data_idx == 0) ?
                AMDSMI_STATUS_OUT_OF_RESOURCES :
                AMDSMI_STATUS_MORE_DATA;
        }
        if(num_headers_copied == *entry_count) {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[CPER] cper_hdrs filled up before finished with copying all header pointers, entry_count: " << std::dec << *entry_count;
            LOG_ERROR(ss);
            *entry_count = num_headers_copied;
            *buf_size = data_idx;
            return (data_idx == 0) ?
                AMDSMI_STATUS_OUT_OF_RESOURCES :
                AMDSMI_STATUS_MORE_DATA;
        }
        if(*cursor != header_idx) {
            ++header_idx;
            continue;
        }
        cper_hdrs[num_headers_copied] = reinterpret_cast<amdsmi_cper_hdr_t*>(&cper_data[data_idx]);
        ++num_headers_copied;
        *cursor = ++header_idx;
        std::memcpy(
            &cper_data[data_idx],
            reinterpret_cast<const char*>(header),
            header->record_length);
        data_idx += header->record_length;

        // cper_decode(cper_hdrs[num_headers_copied - 1]);
   }
   *entry_count = num_headers_copied;
   *buf_size = data_idx;

    ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__
        << "[CPER] *entry_count: " << entry_count
        << ", *cursor: " << cursor
        << ", *buf_size: " << buf_size;

    LOG_DEBUG(ss);
    return AMDSMI_STATUS_SUCCESS;
}

std::vector<int> cper_decode(const amdsmi_cper_hdr_t *cper) {

    std::vector<int> afids;
    std::ostringstream ss;

    for (int i = 0; i < cper_num_sec(cper); i ++) {
        void *sec_desc_offset = cper_get_sec_desc_offset(cper, i);
        void *sec_offset = cper_get_sec_offset(cper, i);
        const amdsmi_cper_guid_t *sec_guid = get_sec_desc_type(static_cast<struct cper_sec_desc *>(sec_desc_offset));
        const amdsmi_cper_guid_t *cper_guid = get_cper_type(cper);

        cper_dump_sec_desc(static_cast<struct cper_sec_desc *>(sec_desc_offset));

        if (cper_is_cr(sec_guid)) {
            if (cper_is_bt(cper_guid)) {
                ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] decoding boot crash dump\n";
                LOG_DEBUG(ss);
                afids.emplace_back(cper_dump_cr_boot(static_cast<struct cper_sec_crashdump *>(sec_offset)));
            }
            else {
                ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] decoding crash dump\n";
                LOG_DEBUG(ss);
                afids.emplace_back(cper_dump_cr_fatal(static_cast<struct cper_sec_crashdump *>(sec_offset)));
            }
        }
        else if (cper_is_nonstd(sec_guid)) {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] decoding non-standard error\n";
            LOG_DEBUG(ss);
            afids.emplace_back(cper_dump_nonstd_err(static_cast<struct cper_sec_nonstd_err *>(sec_offset)));
        } 
        else if (cper_is_proc_err(sec_guid)) {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] decoding proc error section type\n";
            LOG_DEBUG(ss);
            afids.emplace_back(cper_dump_nonstd_err(static_cast<struct cper_sec_nonstd_err *>(sec_offset)));
        } 
        else {
            ss << __PRETTY_FUNCTION__ << "\n:" << __LINE__ << "[AFIDS] Unknown error type!!\n";
            for(int i = 0; i < sizeof(sec_guid->b); ++i) {
                ss << std::hex << static_cast<int>(sec_guid->b[i]) << ":";
            }
            ss << "\n";
            LOG_ERROR(ss);
        }
    }

    return afids;
}

