#ifndef NON_CACHING_STORE_H_
#define NON_CACHING_STORE_H_

template<typename T>
inline
__attribute__((always_inline))
__host__ __device__ T __non_caching_store(const T val, const T* p)
{
    #if !defined(__GFX11__) && !defined(GFX12)
        #define ST "global_store_byte"
        #define ST2 "global_store_short"
        #define ST3 "global_store_dword"
        #define ST4 "global_store_dwordx2"
        #if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
            #define BITS "sc0 sc1 nt"
        #elif defined(__GFX9__) || defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__)
            #define BITS "glc slc"
        #else
            #define BITS "glc slc dlc"
        #endif
    #else
        #define ST "global_store_b8"
        #define ST2 "global_store_b16"
        #define ST3 "global_store_b32"
        #define ST4 "global_store_b64"
        #define BITS "glc slc dlc"
    #endif
    #define STORE ST " %0 %1 %2 " BITS
    #define STORE2 ST2 " %0 %1 %2 " BITS
    #define STORE3 ST3 " %0 %1 %2 " BITS
    #define STORE4 ST4 " %0 %1 %2 " BITS

    switch (sizeof(T)) {
    case 1:
        asm volatile(STORE :: "v"(0), "v"(uint32_t(val)) , "s"(p));
        break;
    case 2:
        asm volatile(STORE2 :: "v"(0), "v"(val) , "s"(p));
        break;
    case 4:
        asm volatile(STORE3 :: "v"(0), "v"(val) , "s"(p));
        break;
    case 8:
        asm volatile(STORE4 :: "v"(0), "v"(val) , "s"(p));
        break;
    default: __builtin_trap();
    }

    #undef STORE4
    #undef STORE3
    #undef STORE2
    #undef STORE
    #undef BITS
    #undef ST4
    #undef ST3
    #undef ST2
    #undef ST
}

#endif

