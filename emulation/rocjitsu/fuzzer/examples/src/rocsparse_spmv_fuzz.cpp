#include "rocfuzz_example_input.h"

#include <rocsparse/rocsparse.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <vector>

__AFL_FUZZ_INIT();

#ifndef ROCFUZZ_EXAMPLE_AFL_LOOP_COUNT
#define ROCFUZZ_EXAMPLE_AFL_LOOP_COUNT 1000
#endif

namespace {

bool ok(rocsparse_status status) { return status == rocsparse_status_success; }

bool report_failure(const char *what, rocsparse_status status) {
  if (ok(status))
    return false;
  std::fprintf(stderr, "%s failed: %d\n", what, static_cast<int>(status));
  return true;
}

struct RocsparseHandle {
  rocsparse_handle handle = nullptr;

  ~RocsparseHandle() {
    if (handle != nullptr)
      (void)rocsparse_destroy_handle(handle);
  }
};

struct MatrixDescriptor {
  rocsparse_mat_descr descr = nullptr;

  ~MatrixDescriptor() {
    if (descr != nullptr)
      (void)rocsparse_destroy_mat_descr(descr);
  }
};

int interesting_sparse_dim(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<int, 16> kDims = {4,  5,  7,  8,  15, 16, 17, 31,
                                               32, 33, 48, 63, 64, 96, 127, 128};
  return stream.pick(kDims);
}

rocsparse_operation interesting_operation(rocfuzz::examples::ByteStream &stream) {
  static constexpr std::array<rocsparse_operation, 4> kOps = {
      rocsparse_operation_none, rocsparse_operation_none, rocsparse_operation_none,
      rocsparse_operation_transpose};
  return stream.pick(kOps);
}

std::vector<rocsparse_int> make_row_lengths(int rows, int cols,
                                            rocfuzz::examples::ByteStream &stream) {
  std::vector<rocsparse_int> lengths(rows);
  const int max_per_row =
      std::min(cols, stream.pick(std::array<int, 8>{1, 2, 3, 4, 8, 12, 16, 32}));

  rocsparse_int total = 0;
  for (int row = 0; row < rows; ++row) {
    const uint8_t pattern = stream.next() % 6;
    int count = 0;
    switch (pattern) {
    case 0:
      count = 0;
      break;
    case 1:
      count = 1;
      break;
    case 2:
      count = (row % max_per_row) + 1;
      break;
    case 3:
      count = stream.next() % (max_per_row + 1);
      break;
    case 4:
      count = (row & 1) ? max_per_row : 0;
      break;
    default:
      count = max_per_row;
      break;
    }
    lengths[row] = static_cast<rocsparse_int>(std::min(count, cols));
    total += lengths[row];
  }

  if (total == 0)
    lengths[stream.next() % rows] = 1;
  return lengths;
}

int run_case(rocsparse_handle handle, const std::vector<uint8_t> &input,
             rocfuzz::examples::PersistentHook begin,
             rocfuzz::examples::PersistentHook end) {
  using namespace rocfuzz::examples;

  ByteStream stream(input);
  const int rows = interesting_sparse_dim(stream);
  const int cols = interesting_sparse_dim(stream);
  const rocsparse_operation trans = interesting_operation(stream);
  const int x_count = trans == rocsparse_operation_none ? cols : rows;
  const int y_count = trans == rocsparse_operation_none ? rows : cols;

  MatrixDescriptor descr;
  if (!ok(rocsparse_create_mat_descr(&descr.descr)))
    return 0;
  if (!ok(rocsparse_set_mat_type(descr.descr, rocsparse_matrix_type_general)))
    return 0;
  if (!ok(rocsparse_set_mat_index_base(descr.descr, rocsparse_index_base_zero)))
    return 0;

  std::vector<rocsparse_int> row_ptr(rows + 1);
  std::vector<rocsparse_int> col_ind;
  const std::vector<rocsparse_int> row_lengths = make_row_lengths(rows, cols, stream);
  for (int row = 0; row < rows; ++row) {
    row_ptr[row] = static_cast<rocsparse_int>(col_ind.size());
    const int stride = std::max(1, static_cast<int>(stream.next() % 17));
    const int base = stream.next() % cols;
    for (rocsparse_int i = 0; i < row_lengths[row]; ++i)
      col_ind.push_back(static_cast<rocsparse_int>((base + i * stride + stream.next()) % cols));
  }
  row_ptr[rows] = static_cast<rocsparse_int>(col_ind.size());
  const rocsparse_int nnz = row_ptr[rows];

  std::vector<float> values(nnz);
  std::vector<float> x(x_count);
  std::vector<float> y(y_count);
  fill_floats(values, stream);
  fill_floats(x, stream);
  fill_floats(y, stream);

  DeviceBuffer<float> device_values;
  DeviceBuffer<rocsparse_int> device_row_ptr;
  DeviceBuffer<rocsparse_int> device_col_ind;
  DeviceBuffer<float> device_x;
  DeviceBuffer<float> device_y;
  if (!device_values.allocate(values.size()) || !device_row_ptr.allocate(row_ptr.size()) ||
      !device_col_ind.allocate(col_ind.size()) || !device_x.allocate(x.size()) ||
      !device_y.allocate(y.size()))
    return 0;

  if (!device_values.copy_from_host(values) || !device_row_ptr.copy_from_host(row_ptr) ||
      !device_col_ind.copy_from_host(col_ind) || !device_x.copy_from_host(x) ||
      !device_y.copy_from_host(y))
    return 0;

  float alpha = stream.next_float();
  if (alpha == 0.0f)
    alpha = 1.0f;
  const float beta = stream.next_float();

  if (call_persistent_hook(begin, "rocjitsu_afl_persistent_begin") != 0)
    return 3;

  const rocsparse_status status =
      rocsparse_scsrmv(handle, trans, rows, cols, nnz, &alpha, descr.descr,
                       device_values.get(), device_row_ptr.get(), device_col_ind.get(), nullptr,
                       device_x.get(), &beta, device_y.get());
  if (report_failure("rocsparse_scsrmv", status))
    return 0;

  if (end != nullptr) {
    if (call_persistent_hook(end, "rocjitsu_afl_persistent_end") != 0)
      return 3;
  } else {
    crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());
  }

  std::vector<float> out(y.size());
  crash_on_hip_error("hipMemcpy D2H", device_y.copy_to_host(out) ? hipSuccess : hipErrorUnknown);

  volatile float sink = 0.0f;
  for (size_t i = 0; i < std::min<size_t>(out.size(), 16); ++i)
    sink += out[i];
  (void)sink;

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  using namespace rocfuzz::examples;

  if (!have_hip_device())
    return 0;

  RocsparseHandle handle;
  if (!ok(rocsparse_create_handle(&handle.handle)))
    return 0;
  if (!ok(rocsparse_set_pointer_mode(handle.handle, rocsparse_pointer_mode_host)))
    return 0;

#if defined(ROCFUZZ_EXAMPLE_PERSISTENT) && ROCFUZZ_EXAMPLE_PERSISTENT
  PersistentHook begin = load_persistent_hook("rocjitsu_afl_persistent_begin");
  PersistentHook end = load_persistent_hook("rocjitsu_afl_persistent_end");
  if (persistent_hooks_required() && (begin == nullptr || end == nullptr)) {
    std::fprintf(stderr, "rocjitsu AFL persistent hooks are not available\n");
    return 4;
  }

  __AFL_INIT();

#ifdef __AFL_FUZZ_TESTCASE_BUF
  unsigned char *afl_buf = __AFL_FUZZ_TESTCASE_BUF;
  while (__AFL_LOOP(ROCFUZZ_EXAMPLE_AFL_LOOP_COUNT)) {
    const size_t len = __AFL_FUZZ_TESTCASE_LEN;
    std::vector<uint8_t> input(afl_buf, afl_buf + len);
    const int rc = run_case(handle.handle, input, begin, end);
    if (rc != 0)
      return rc;
  }
#else
  while (__AFL_LOOP(1)) {
    const int rc = run_case(handle.handle, read_input(argc, argv), begin, end);
    if (rc != 0)
      return rc;
  }
#endif
#else
#ifdef __AFL_FUZZ_TESTCASE_BUF
  std::vector<uint8_t> input;
  if (argc > 1) {
    input = read_input(argc, argv);
  } else {
    unsigned char *afl_buf = __AFL_FUZZ_TESTCASE_BUF;
    const size_t len = __AFL_FUZZ_TESTCASE_LEN;
    input.assign(afl_buf, afl_buf + len);
  }
#else
  std::vector<uint8_t> input = read_input(argc, argv);
#endif
  const int rc = run_case(handle.handle, input, nullptr, nullptr);
  if (rc != 0)
    return rc;
#endif

  return 0;
}
