// #define TRIG_UNARY_NEGATIVE_KERNELS(func_name)
// class Dummy {
//  public:
//   __device__ Dummy() {}
//   __device__ ~Dummy() {}
// };
// __global__ void func_name##f_kernel_v1(float* x) { float result = func_name##f(x); }
// __global__ void func_name##f_kernel_v2(Dummy x) { float result = func_name##f(x); }
// __global__ void func_name##_kernel_v1(double* x) { double result = func_name(x); }
// __global__ void func_name##_kernel_v2(Dummy x) { double result = func_name(x); }

static constexpr auto kSin{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void sinf_kernel_v1(float* x) { float result = sinf(x); }             
  __global__ void sinf_kernel_v2(Dummy x) { float result = sinf(x); }              
  __global__ void sin_kernel_v1(double* x) { double result = sin(x); }               
  __global__ void sin_kernel_v2(Dummy x) { double result = sin(x); }
  )"};

static constexpr auto kCos{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void cosf_kernel_v1(float* x) { float result = cosf(x); }             
  __global__ void cosf_kernel_v2(Dummy x) { float result = cosf(x); }              
  __global__ void cos_kernel_v1(double* x) { double result = cos(x); }               
  __global__ void cos_kernel_v2(Dummy x) { double result = cos(x); }
  )"};

static constexpr auto kTan{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void tanf_kernel_v1(float* x) { float result = tanf(x); }             
  __global__ void tanf_kernel_v2(Dummy x) { float result = tanf(x); }              
  __global__ void tan_kernel_v1(double* x) { double result = tan(x); }               
  __global__ void tan_kernel_v2(Dummy x) { double result = tan(x); }
  )"};

static constexpr auto kAsin{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void asinf_kernel_v1(float* x) { float result = asinf(x); }             
  __global__ void asinf_kernel_v2(Dummy x) { float result = asinf(x); }              
  __global__ void asin_kernel_v1(double* x) { double result = asin(x); }               
  __global__ void asin_kernel_v2(Dummy x) { double result = asin(x); }
  )"};

static constexpr auto kAcos{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void acosf_kernel_v1(float* x) { float result = acosf(x); }             
  __global__ void acosf_kernel_v2(Dummy x) { float result = acosf(x); }              
  __global__ void acos_kernel_v1(double* x) { double result = acos(x); }               
  __global__ void acos_kernel_v2(Dummy x) { double result = acos(x); }
  )"};

static constexpr auto kAtan{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void atanf_kernel_v1(float* x) { float result = atanf(x); }             
  __global__ void atanf_kernel_v2(Dummy x) { float result = atanf(x); }              
  __global__ void atan_kernel_v1(double* x) { double result = atan(x); }               
  __global__ void atan_kernel_v2(Dummy x) { double result = atan(x); }
  )"};

static constexpr auto kSinh{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void sinhf_kernel_v1(float* x) { float result = sinhf(x); }             
  __global__ void sinhf_kernel_v2(Dummy x) { float result = sinhf(x); }              
  __global__ void sinh_kernel_v1(double* x) { double result = sinh(x); }               
  __global__ void sinh_kernel_v2(Dummy x) { double result = sinh(x); }
  )"};

static constexpr auto kCosh{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void coshf_kernel_v1(float* x) { float result = coshf(x); }             
  __global__ void coshf_kernel_v2(Dummy x) { float result = coshf(x); }              
  __global__ void cosh_kernel_v1(double* x) { double result = cosh(x); }               
  __global__ void cosh_kernel_v2(Dummy x) { double result = cosh(x); }
  )"};

static constexpr auto kTanh{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void tanhf_kernel_v1(float* x) { float result = tanhf(x); }             
  __global__ void tanhf_kernel_v2(Dummy x) { float result = tanhf(x); }              
  __global__ void tanh_kernel_v1(double* x) { double result = tanh(x); }               
  __global__ void tanh_kernel_v2(Dummy x) { double result = tanh(x); }
  )"};

static constexpr auto kAsinh{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void asinhf_kernel_v1(float* x) { float result = asinhf(x); }             
  __global__ void asinhf_kernel_v2(Dummy x) { float result = asinhf(x); }              
  __global__ void asinh_kernel_v1(double* x) { double result = asinh(x); }               
  __global__ void asinh_kernel_v2(Dummy x) { double result = asinh(x); }
  )"};

static constexpr auto kAcosh{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void acoshf_kernel_v1(float* x) { float result = acoshf(x); }             
  __global__ void acoshf_kernel_v2(Dummy x) { float result = acoshf(x); }              
  __global__ void acosh_kernel_v1(double* x) { double result = acosh(x); }               
  __global__ void acosh_kernel_v2(Dummy x) { double result = acosh(x); }
  )"};

static constexpr auto kAtanh{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void atanhf_kernel_v1(float* x) { float result = atanhf(x); }             
  __global__ void atanhf_kernel_v2(Dummy x) { float result = atanhf(x); }              
  __global__ void atanh_kernel_v1(double* x) { double result = atanh(x); }               
  __global__ void atanh_kernel_v2(Dummy x) { double result = atanh(x); }
  )"};

static constexpr auto kSinpi{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void sinpif_kernel_v1(float* x) { float result = sinpif(x); }             
  __global__ void sinpif_kernel_v2(Dummy x) { float result = sinpif(x); }              
  __global__ void sinpi_kernel_v1(double* x) { double result = sinpi(x); }               
  __global__ void sinpi_kernel_v2(Dummy x) { double result = sinpi(x); }
  )"};

static constexpr auto kCospi{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void cospif_kernel_v1(float* x) { float result = cospif(x); }             
  __global__ void cospif_kernel_v2(Dummy x) { float result = cospif(x); }              
  __global__ void cospi_kernel_v1(double* x) { double result = cospi(x); }               
  __global__ void cospi_kernel_v2(Dummy x) { double result = cospi(x); }
  )"};

static constexpr auto kAtan2{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void atan2f_kernel_v1(float* x, float y) { float result = atan2f(x, y); }
  __global__ void atan2f_kernel_v2(float x, float* y) { float result = atan2f(x, y); }
  __global__ void atan2f_kernel_v3(Dummy x, float y) { float result = atan2f(x, y); }
  __global__ void atan2f_kernel_v4(float x, Dummy y) { float result = atan2f(x, y); }
  __global__ void atan2_kernel_v1(double* x, double y) { double result = atan2(x, y); }
  __global__ void atan2_kernel_v2(double x, double* y) { double result = atan2(x, y); }
  __global__ void atan2_kernel_v3(Dummy x, double y) { double result = atan2(x, y); }
  __global__ void atan2_kernel_v4(double x, Dummy y) { double result = atan2(x, y); }
  )"};

static constexpr auto kSincos{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void sincosf_kernel_v1(float* x, float* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v2(Dummy x, float* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v3(float x, char* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v4(float x, short* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v5(float x, int* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v6(float x, long* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v7(float x, long long* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v8(float x, double* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v9(float x, Dummy* sptr, float* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v10(float x, const float* sptr, float* cptr) {
    sincosf(x, sptr, cptr);
  }
  __global__ void sincosf_kernel_v11(float x, float* sptr, char* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v12(float x, float* sptr, short* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v13(float x, float* sptr, int* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v14(float x, float* sptr, long* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v15(float x, float* sptr, long long* cptr) {
    sincosf(x, sptr, cptr);
  }
  __global__ void sincosf_kernel_v16(float x, float* sptr, double* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v17(float x, float* sptr, Dummy* cptr) { sincosf(x, sptr, cptr); }
  __global__ void sincosf_kernel_v18(float x, float* sptr, const float* cptr) {
    sincosf(x, sptr, cptr);
  }
  __global__ void sincos_kernel_v1(double* x, double* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v2(Dummy x, double* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v3(double x, char* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v4(double x, short* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v5(double x, int* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v6(double x, long* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v7(double x, long long* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v8(double x, float* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v9(double x, Dummy* sptr, double* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v10(double x, const double* sptr, double* cptr) {
    sincos(x, sptr, cptr);
  }
  __global__ void sincos_kernel_v11(double x, double* sptr, char* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v12(double x, double* sptr, short* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v13(double x, double* sptr, int* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v14(double x, double* sptr, long* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v15(double x, double* sptr, long long* cptr) {
    sincos(x, sptr, cptr);
  }
  __global__ void sincos_kernel_v16(double x, double* sptr, float* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v17(double x, double* sptr, Dummy* cptr) { sincos(x, sptr, cptr); }
  __global__ void sincos_kernel_v18(double x, double* sptr, const double* cptr) {
    sincos(x, sptr, cptr);
  }
    )"};

static constexpr auto kSincospi{R"(
  class Dummy {                                                                                    
   public:                                                                                         
    __device__ Dummy() {}                                                                          
    __device__ ~Dummy() {}                                                                         
  };                                                                                               
  __global__ void sincospif_kernel_v1(float* x, float* sptr, float* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospif_kernel_v2(Dummy x, float* sptr, float* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v3(float x, char* sptr, float* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v4(float x, short* sptr, float* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v5(float x, int* sptr, float* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v6(float x, long* sptr, float* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v7(float x, long long* sptr, float* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospif_kernel_v8(float x, double* sptr, float* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospif_kernel_v9(float x, Dummy* sptr, float* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v10(float x, const float* sptr, float* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospif_kernel_v11(float x, float* sptr, char* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v12(float x, float* sptr, short* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospif_kernel_v13(float x, float* sptr, int* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v14(float x, float* sptr, long* cptr) { sincospif(x, sptr, cptr); }
  __global__ void sincospif_kernel_v15(float x, float* sptr, long long* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospif_kernel_v16(float x, float* sptr, double* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospif_kernel_v17(float x, float* sptr, Dummy* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospif_kernel_v18(float x, float* sptr, const float* cptr) {
    sincospif(x, sptr, cptr);
  }
  __global__ void sincospi_kernel_v1(float* x, float* sptr, float* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v2(Dummy x, float* sptr, float* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v3(float x, char* sptr, float* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v4(float x, short* sptr, float* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v5(float x, int* sptr, float* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v6(float x, long* sptr, float* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v7(float x, long long* sptr, float* cptr) {
    sincospi(x, sptr, cptr);
  }
  __global__ void sincospi_kernel_v8(float x, double* sptr, float* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v9(float x, Dummy* sptr, float* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v10(float x, const float* sptr, float* cptr) {
    sincospi(x, sptr, cptr);
  }
  __global__ void sincospi_kernel_v11(float x, float* sptr, char* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v12(float x, float* sptr, short* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v13(float x, float* sptr, int* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v14(float x, float* sptr, long* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v15(float x, float* sptr, long long* cptr) {
    sincospi(x, sptr, cptr);
  }
  __global__ void sincospi_kernel_v16(float x, float* sptr, double* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v17(float x, float* sptr, Dummy* cptr) { sincospi(x, sptr, cptr); }
  __global__ void sincospi_kernel_v18(float x, float* sptr, const float* cptr) {
    sincospi(x, sptr, cptr);
  }
  )"};