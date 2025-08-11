/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANNTY OF ANY KIND, EXPRESS OR
IMPLIED, INNCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANNY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER INN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR INN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <hip_test_common.hh>
#include <hip_test_checkers.hh>
#include <hip_test_kernels.hh>
#include <array>
/*typedef std::tuple<int, int, int, int> tupletype;
static constexpr std::initializer_list<tupletype> tableItems {
               std::make_tuple(20,   20, 20, 20),
               std::make_tuple(10,   10,  4,  4),
               std::make_tuple(100, 100, 20, 40),
               std::make_tuple(256, 256, 39, 19),
               std::make_tuple(100, 100, 20,  0),
               std::make_tuple(100, 100,  0, 20),
               std::make_tuple(100, 100,  0,  0),
               };*/
TEST_CASE("Unit_hipMemsetD2D16_BasicFunctional") {
  constexpr uint16_t memsetval = static_cast<uint16_t>(0xDEADBEEF);
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t pitch_A;
  size_t width = numW * sizeof(uint16_t);
  size_t sizeElements = width * numH;
  size_t elements = numW * numH;
  uint16_t *A_d, *A_h;
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, numH, 2*sizeof(uint16_t)));
  A_h = reinterpret_cast<uint16_t*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t i = 0; i < elements; i++) {
    A_h[i] = 1;
  }
  HIP_CHECK(hipMemsetD2D16(A_d, pitch_A, memsetval, width, numH));
  HIP_CHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, width, numH,
                       hipMemcpyDeviceToHost));

  for (size_t i = 0; i < elements; i++) {
    if (A_h[i] != memsetval) {
      INFO("Memset2D mismatch at index:" << i << " computed:"
                                     << A_h[i] << " memsetval:" << memsetval);
      REQUIRE(false);
    }
  }
  HIP_CHECK(hipFree(A_d));
  free(A_h);
}
/*
TEST_CASE("Unit_hipMemsetD2D16_UniqueWidthHeight") {
  int width2D=10, height2D=10;
  int memsetWidth=4, memsetHeight=4;
  uint16_t *A_d, *A_h;
  size_t pitch_A;
  constexpr uint16_t memsetval = static_cast<uint16_t>(0x26);

  std::tie(width2D, height2D, memsetWidth, memsetHeight) =
                 GENERATE(table<int, int, int, int>(tableItems));

  size_t width = width2D * sizeof(uint16_t);
  size_t sizeElements = width * height2D;

  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &pitch_A, width, height2D, 2*sizeof(uint16_t)));

  A_h = reinterpret_cast<uint16_t*>(malloc(sizeElements));
  REQUIRE(A_h != nullptr);

  for (size_t index = 0; index < sizeElements; index++) {
    A_h[index] = 1;
  }

  INFO("2D Dimension: Width:" << width2D << " Height:" << height2D <<
           " MemsetWidth:" << memsetWidth << " MemsetHeight:" << memsetHeight);

  HIP_CHECK(hipMemsetD2D16(A_d, pitch_A, memsetval, memsetWidth, memsetHeight));
  HIP_CHECK(hipMemcpy2D(A_h, width, A_d, pitch_A, width, height2D,
                       hipMemcpyDeviceToHost));
  printf("width2D: %d\n", width2D);
  printf("height2D: %d\n", height2D);
	  printf("MemsetWidth: %d\n", memsetWidth);
	  printf("memsetHeight: %d\n", memsetHeight);
	  printf("\n");
  for (int row = 0; row < memsetHeight height2D; row++) {
    for (int column = 0; column < memsetWidth, width2D; column++) {
      std::cout<<"A_h["<<row<<"][" << column << "] : " <<  memsetval<<std::endl;
      //if (A_h[(row * width) + column] != memsetval) {
	      if (A_h[row][column] != memsetval) {
        INFO("A_h[" << row << "][" << column << "]" <<
                                         " didnot match " << memsetval);
       // REQUIRE(false);
      }
    }
  }

  HIP_CHECK(hipFree(A_d));
  free(A_h);
}
*/
TEST_CASE("Unit_hipMemsetD2D16_UnEvenRowsCols") {
  uint16_t *A_h, *B_h, *A_d;
  int rows, cols;
  rows = GENERATE(3, 4, 100);
  cols = GENERATE(3, 4, 100);
  size_t devPitch;

  A_h = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * rows * cols));
  B_h = reinterpret_cast<uint16_t*>(malloc(sizeof(uint16_t) * rows * cols));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      A_h[i * cols + j] = 1;
    }
  }
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &devPitch, sizeof(uint16_t) * cols, rows, 2*sizeof(uint16_t)));
  HIP_CHECK(hipMemcpy2D(A_d, devPitch, A_h, sizeof(uint16_t) * cols,
                        sizeof(uint16_t) * cols, rows, hipMemcpyHostToDevice));

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemsetD2D16(A_d, devPitch, 5, sizeof(uint16_t) * cols, rows));
  HIP_CHECK(hipDeviceSynchronize());

  HIP_CHECK(hipMemcpy2D(B_h, sizeof(uint16_t) * cols, A_d, devPitch,
                        sizeof(uint16_t) * cols, rows, hipMemcpyDeviceToHost));

  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      REQUIRE(B_h[i * cols + j] == 5);
    }
  }
  HIP_CHECK(hipFree(A_d));
  free(A_h);
  free(B_h);
}

TEST_CASE("Unit_hipMemsetD2D16_KernelOperation") {
  constexpr size_t size = 4096;
  constexpr uint16_t memsetval = static_cast<uint16_t>(0x26);
  constexpr unsigned blocksPerCU = 6;
  constexpr unsigned threadsPerBlock = 256;
  uint16_t *C_h, *A_d, *B_d, *C_d;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t devPitch;
  size_t width = numW * sizeof(uint16_t);
  size_t sizeElements = width * numH;

  C_h = reinterpret_cast<uint16_t*>(malloc(sizeElements));
  for (int i = 0; i < numH; i++) {
    for (int j = 0; j < numW; j++) {
      C_h[i * numH + j] = 0;
    }
  }
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &devPitch, width, numH, 2*sizeof(uint16_t)));
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&B_d), &devPitch, width, numH, 2*sizeof(uint16_t)));
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&C_d), &devPitch, width, numH, 2*sizeof(uint16_t)));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemsetD2D16(A_d, devPitch, memsetval, numW, numH));
  HIP_CHECK(hipMemsetD2D16(B_d, devPitch, memsetval, numW, numH));

  unsigned blocks = HipTest::setNumBlocks(blocksPerCU, threadsPerBlock, size);

  hipLaunchKernelGGL(HipTest::vectorADD, dim3(blocks), dim3(threadsPerBlock), 0, stream, A_d,
                     B_d, C_d, size);

  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy2D(C_h, width, C_d, devPitch, width, numH,
                       hipMemcpyDeviceToHost));
  for (int i = 0; i < numH; i++) {
    for (int j = 0; j < numW; j++) {
	  C_h[i * numH + j] =  memsetval + memsetval;
	}
  }
  HIP_CHECK(hipFree(A_d));
  HIP_CHECK(hipFree(B_d));
  HIP_CHECK(hipFree(C_d));
  free(C_h);
  HIP_CHECK(hipStreamDestroy(stream));
}

TEST_CASE("Unit_hipMemsetD2D16_NegTsts") {
  uint16_t *A_d;
  constexpr size_t numH = 256;
  constexpr size_t numW = 256;
  size_t width = numW * sizeof(uint16_t);
  size_t devPitch;
  constexpr uint16_t memsetval = static_cast<uint16_t>(0x26);
  HIP_CHECK(hipMemAllocPitch(reinterpret_cast<void**>(&A_d), &devPitch, width, numH, 2*sizeof(uint16_t)));
  SECTION("nullptr destination") {
    HIP_CHECK_ERROR(hipMemsetD2D16(nullptr, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("OutOfBound destination") {
    void* outOfBoundsDst{reinterpret_cast<uint16_t*>(A_d) + devPitch * numH + 1};
    HIP_CHECK_ERROR(hipMemsetD2D16(outOfBoundsDst, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Dst pointer points to Source Memory") {
    uint16_t *B_d;
    std::unique_ptr<uint16_t[]> hostPtr;
    hostPtr.reset(new uint16_t[numH * width]);
    B_d = hostPtr.get();
    HIP_CHECK_ERROR(hipMemsetD2D16(B_d, devPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Invalid Pitch") {
    size_t inValidPitch = 1;
    HIP_CHECK_ERROR(hipMemsetD2D16(A_d, inValidPitch, memsetval, numW, numH), hipErrorInvalidValue);
  }
  SECTION("Negative Values of Hight, Width") { 
    HIP_CHECK_ERROR(hipMemsetD2D16(A_d, devPitch, memsetval, numW, -10), hipErrorInvalidValue); 
    HIP_CHECK_ERROR(hipMemsetD2D16(A_d, devPitch, memsetval, -10, numH), hipErrorInvalidValue); 
  } //need to check on CUDA
  /*SECTION("OutOfbounds Hight, Width") {
    HIP_CHECK_ERROR(hipMemsetD2D16(A_d, devPitch, memsetval, numW, numH+256), hipErrorInvalidValue);
    HIP_CHECK_ERROR(hipMemsetD2D16(A_d, devPitch, memsetval, numW+256, numH), hipErrorInvalidValue);
  }*/
  HIP_CHECK(hipFree(A_d));
}
