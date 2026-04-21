// tools/htl_dump.cpp — decode a .htl binary file to CSV on stdout.
// Usage: htl_dump <file>
#include "htl_record.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <file.htl>\n", argv[0]);
        return 2;
    }
    int fd = ::open(argv[1], O_RDONLY);
    if (fd < 0) { std::perror("open"); return 1; }
    struct stat st{};
    if (::fstat(fd, &st) < 0) { std::perror("fstat"); return 1; }
    void* map = ::mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { std::perror("mmap"); return 1; }
    const auto* base = static_cast<const uint8_t*>(map);

    if (st.st_size < (off_t)sizeof(htl::file_header_t)) {
        std::fprintf(stderr, "file too small\n"); return 1;
    }
    htl::file_header_t hdr{};
    std::memcpy(&hdr, base, sizeof(hdr));
    if (std::memcmp(hdr.magic, htl::kFileMagic, 4) != 0) {
        std::fprintf(stderr, "bad magic\n"); return 1;
    }

    // String section (if present).
    std::vector<std::string> strings;
    if (hdr.string_section_offset && hdr.string_section_size) {
        const uint8_t* p   = base + hdr.string_section_offset;
        const uint8_t* end = p + hdr.string_section_size;
        while (p + sizeof(uint32_t) <= end) {
            uint32_t len;
            std::memcpy(&len, p, sizeof(len));
            p += sizeof(len);
            if (p + len > end) break;
            strings.emplace_back(reinterpret_cast<const char*>(p), len);
            p += len;
        }
    }

    auto string_at = [&](uint64_t off) -> std::string {
        // off is the *byte offset within the string section*; we walk it.
        uint64_t cursor = 0;
        for (const auto& s : strings) {
            if (cursor == off) return s;
            cursor += sizeof(uint32_t) + s.size();
        }
        return {};
    };

    // Records: between header end and string section start (or footer).
    off_t records_end = hdr.string_section_offset ? (off_t)hdr.string_section_offset
                                                  : st.st_size;
    off_t records_begin = hdr.header_size;
    if ((records_end - records_begin) % sizeof(htl::record_t) != 0) {
        std::fprintf(stderr, "warning: record region not multiple of record_t\n");
    }

    std::printf("domain,op,corr,begin_ns,end_ns,pid,tid,dev,queue,bytes,kernel\n");
    const auto* rp = reinterpret_cast<const htl::record_t*>(base + records_begin);
    size_t n = (records_end - records_begin) / sizeof(htl::record_t);
    for (size_t i = 0; i < n; ++i) {
        const auto& r = rp[i];
        // offset 0 is valid if strings exist; only treat 0 as "no string" if string section is empty
        std::string k;
        if (r.kernel_name_off == 0 && strings.empty()) {
            k = "";
        } else {
            k = string_at(r.kernel_name_off);
        }
        std::printf("%u,%u,%llu,%llu,%llu,%u,%u,%d,%u,%llu,%s\n",
            r.domain, r.op,
            (unsigned long long)r.correlation_id,
            (unsigned long long)r.begin_ns, (unsigned long long)r.end_ns,
            r.process_id, r.thread_id, r.device_id, r.queue_id,
            (unsigned long long)r.bytes, k.c_str());
    }

    ::munmap(map, st.st_size);
    ::close(fd);
    return 0;
}
