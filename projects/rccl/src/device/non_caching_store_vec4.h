#ifndef NON_CACHING_STORE_VEC4_H_
#define NON_CACHING_STORE_VEC4_H_

template<typename T>
inline
__attribute__((always_inline))
__host__ __device__ T __non_caching_store_vec4(const T val, const T* p)
{
    #if !defined(__GFX11__) && !defined(GFX12)
        #define ST "global_store_dwordx4"
        #if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
            #define BITS "sc0 sc1 nt"
        #elif defined(__GFX9__) || defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__)
            #define BITS "glc slc"
        #else
            #define BITS "glc slc dlc"
        #endif
    #else
        #define ST "global_store_b128"
        #define BITS "glc slc dlc"
    #endif

    #define STORE ST " %0 %1 %2 " BITS

    asm volatile(STORE :: "v"(0), "v"(val) , "s"(p));
   
    #undef STORE
    #undef BITS
    #undef ST
}

#endif

