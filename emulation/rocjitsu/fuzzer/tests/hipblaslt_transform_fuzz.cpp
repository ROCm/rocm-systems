#include "rj_fuzz_input.h"

#include <hipblaslt/hipblaslt.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

__AFL_FUZZ_INIT();

namespace {

bool ok(hipblasStatus_t status) { return status == HIPBLAS_STATUS_SUCCESS; }

struct HipblasLtHandle {
  hipblasLtHandle_t handle = nullptr;

  ~HipblasLtHandle() {
    if (handle != nullptr)
      (void)hipblasLtDestroy(handle);
  }
};

struct TransformDesc {
  hipblasLtMatrixTransformDesc_t desc = nullptr;

  ~TransformDesc() {
    if (desc != nullptr)
      (void)hipblasLtMatrixTransformDescDestroy(desc);
  }
};

struct MatrixLayout {
  hipblasLtMatrixLayout_t layout = nullptr;

  ~MatrixLayout() {
    if (layout != nullptr)
      (void)hipblasLtMatrixLayoutDestroy(layout);
  }
};

size_t matrix_elements(int leading_dim, int columns) {
  return static_cast<size_t>(leading_dim) * static_cast<size_t>(columns);
}

bool create_layout(MatrixLayout &layout, uint64_t rows, uint64_t cols, int64_t ld) {
  return ok(hipblasLtMatrixLayoutCreate(&layout.layout, HIP_R_32F, rows, cols, ld));
}

bool configure_transform(TransformDesc &desc, hipblasOperation_t op_a, hipblasOperation_t op_b) {
  if (!ok(hipblasLtMatrixTransformDescCreate(&desc.desc, HIP_R_32F)))
    return false;

  const hipblasLtPointerMode_t pointer_mode = HIPBLASLT_POINTER_MODE_HOST;
  if (!ok(hipblasLtMatrixTransformDescSetAttribute(desc.desc,
                                                   HIPBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE,
                                                   &pointer_mode, sizeof(pointer_mode))))
    return false;

  if (!ok(hipblasLtMatrixTransformDescSetAttribute(
          desc.desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSA, &op_a, sizeof(op_a))))
    return false;

  return ok(hipblasLtMatrixTransformDescSetAttribute(
      desc.desc, HIPBLASLT_MATRIX_TRANSFORM_DESC_TRANSB, &op_b, sizeof(op_b)));
}

int run_case(hipblasLtHandle_t handle, const std::vector<uint8_t> &input) {
  using namespace rj_fuzz;

  ByteStream stream(input);

  const int rows = interesting_dim(stream);
  const int cols = interesting_dim(stream);
  const bool transpose_a = (stream.next() & 1) != 0;
  const bool transpose_b = (stream.next() & 1) != 0;

  const int a_rows = transpose_a ? cols : rows;
  const int a_cols = transpose_a ? rows : cols;
  const int b_rows = transpose_b ? cols : rows;
  const int b_cols = transpose_b ? rows : cols;
  const int lda = a_rows + static_cast<int>(stream.next() & 3);
  const int ldb = b_rows + static_cast<int>(stream.next() & 3);
  const int ldc = rows + static_cast<int>(stream.next() & 3);

  std::vector<float> host_a(matrix_elements(lda, a_cols));
  std::vector<float> host_b(matrix_elements(ldb, b_cols));
  std::vector<float> host_c(matrix_elements(ldc, cols));
  fill_floats(host_a, stream);
  fill_floats(host_b, stream);
  fill_floats(host_c, stream);

  DeviceBuffer<float> device_a;
  DeviceBuffer<float> device_b;
  DeviceBuffer<float> device_c;
  if (!device_a.allocate(host_a.size()) || !device_b.allocate(host_b.size()) ||
      !device_c.allocate(host_c.size()))
    return 0;

  if (!device_a.copy_from_host(host_a) || !device_b.copy_from_host(host_b) ||
      !device_c.copy_from_host(host_c))
    return 0;

  MatrixLayout layout_a;
  MatrixLayout layout_b;
  MatrixLayout layout_c;
  if (!create_layout(layout_a, a_rows, a_cols, lda) ||
      !create_layout(layout_b, b_rows, b_cols, ldb) || !create_layout(layout_c, rows, cols, ldc))
    return 0;

  const hipblasOperation_t op_a = transpose_a ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  const hipblasOperation_t op_b = transpose_b ? HIPBLAS_OP_T : HIPBLAS_OP_N;
  TransformDesc transform;
  if (!configure_transform(transform, op_a, op_b))
    return 0;

  const float alpha = stream.next_float();
  const float beta = stream.next_float();

  const hipblasStatus_t status = hipblasLtMatrixTransform(
      handle, transform.desc, &alpha, device_a.get(), layout_a.layout, &beta, device_b.get(),
      layout_b.layout, device_c.get(), layout_c.layout, nullptr);
  if (!ok(status))
    return 0;

  rj_fuzz::crash_on_hip_error("hipDeviceSynchronize", hipDeviceSynchronize());

  std::vector<float> out(host_c.size());
  rj_fuzz::crash_on_hip_error("hipMemcpy D2H",
                              device_c.copy_to_host(out) ? hipSuccess : hipErrorUnknown);

  volatile float sink = 0.0f;
  for (size_t i = 0; i < std::min<size_t>(out.size(), 16); ++i)
    sink += out[i];
  (void)sink;

  return 0;
}

} // namespace

int main(int argc, char **argv) {
  if (!rj_fuzz::have_hip_device())
    return 0;

  HipblasLtHandle handle;
  if (!ok(hipblasLtCreate(&handle.handle)))
    return 0;

#ifdef RJ_AFL_PERSISTENT_MODE
  __AFL_INIT();

#ifdef __AFL_FUZZ_TESTCASE_BUF
  unsigned char *afl_buf = __AFL_FUZZ_TESTCASE_BUF;
  while (__AFL_LOOP(1000)) {
    const size_t len = __AFL_FUZZ_TESTCASE_LEN;
    std::vector<uint8_t> input(afl_buf, afl_buf + len);
    const int rc = run_case(handle.handle, input);
    if (rc != 0)
      return rc;
  }
#else
  while (__AFL_LOOP(1)) {
    const int rc = run_case(handle.handle, rj_fuzz::read_input(argc, argv));
    if (rc != 0)
      return rc;
  }
#endif
#else
  const int rc = run_case(handle.handle, rj_fuzz::read_input(argc, argv));
  if (rc != 0)
    return rc;
#endif

  return 0;
}
