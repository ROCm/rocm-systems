#include "htl_record.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>

int main() {
    htl::file_header_t hdr{};
    std::memcpy(hdr.magic, htl::kFileMagic, 4);
    hdr.version = htl::kFileVersion;
    hdr.record_size = sizeof(htl::record_t);
    hdr.header_size = htl::kHeaderSize;

    assert(sizeof(hdr) == 64);
    assert(sizeof(htl::record_t) == 64);
    assert(std::memcmp(hdr.magic, "HTL0", 4) == 0);
    std::printf("test_record: ok\n");
    return 0;
}
