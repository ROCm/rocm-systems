#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wgnu-anonymous-struct"
#pragma clang diagnostic ignored "-Wnested-anon-types"
#endif
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#if defined(__clang__)
#pragma clang diagnostic pop
#endif

#include <cstddef>
#include <cstdint>
#include <unordered_map>

namespace {

enum CreateKind {
  kCreateKindNone = 0,
  kCreateKindMemory = 1,
  kCreateKindFile = 2,
  kCreateKindFileWithOffset = 3,
};

struct ReaderRecord {
  const uint8_t *data = nullptr;
  size_t size = 0;
  CreateKind kind = kCreateKindNone;
};

uint64_t g_next_handle = 1;
CreateKind g_last_create_kind = kCreateKindNone;
int g_destroy_calls = 0;
std::unordered_map<uint64_t, ReaderRecord> g_readers;

hsa_status_t create_reader(CreateKind kind, const void *data, size_t size,
                           hsa_code_object_reader_t *reader) {
  if (reader == nullptr)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  reader->handle = g_next_handle++;
  g_last_create_kind = kind;
  g_readers[reader->handle] = ReaderRecord{static_cast<const uint8_t *>(data), size, kind};
  return HSA_STATUS_SUCCESS;
}

} // namespace

extern "C" {

hsa_status_t
hsa_code_object_reader_create_from_memory(const void *code_object, size_t size,
                                          hsa_code_object_reader_t *code_object_reader) {
  if (code_object == nullptr || size == 0)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return create_reader(kCreateKindMemory, code_object, size, code_object_reader);
}

hsa_status_t hsa_code_object_reader_create_from_file(hsa_file_t,
                                                     hsa_code_object_reader_t *code_object_reader) {
  return create_reader(kCreateKindFile, nullptr, 0, code_object_reader);
}

hsa_status_t hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size(
    hsa_file_t, size_t, size_t, hsa_code_object_reader_t *code_object_reader) {
  return create_reader(kCreateKindFileWithOffset, nullptr, 0, code_object_reader);
}

hsa_status_t hsa_code_object_reader_destroy(hsa_code_object_reader_t code_object_reader) {
  ++g_destroy_calls;
  g_readers.erase(code_object_reader.handle);
  return HSA_STATUS_SUCCESS;
}

void rocfuzz_fake_hsa_reset() {
  g_next_handle = 1;
  g_last_create_kind = kCreateKindNone;
  g_destroy_calls = 0;
  g_readers.clear();
}

int rocfuzz_fake_hsa_last_create_kind() { return static_cast<int>(g_last_create_kind); }

int rocfuzz_fake_hsa_destroy_calls() { return g_destroy_calls; }

const uint8_t *rocfuzz_fake_hsa_reader_data(uint64_t handle) {
  const auto it = g_readers.find(handle);
  return it == g_readers.end() ? nullptr : it->second.data;
}

size_t rocfuzz_fake_hsa_reader_size(uint64_t handle) {
  const auto it = g_readers.find(handle);
  return it == g_readers.end() ? 0 : it->second.size;
}

} // extern "C"
