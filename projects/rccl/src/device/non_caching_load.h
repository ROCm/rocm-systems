#ifndef NON_CACHING_LOAD_H_
#define NON_CACHING_LOAD_H_

template<typename T>
inline
__attribute__((always_inline))
__host__ __device__ T __non_caching_load(const T* p)
{
    #if !defined(__GFX11__) && !defined(GFX12)
        #define LD "global_load_ubyte"
        #define LD2 "global_load_ushort"
        #define LD3 "global_load_dword"
        #define LD4 "global_load_dwordx2"
        #define LD5 "global_load_dwordx4"
        #if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
            #define BITS "sc0 sc1 nt"
        #elif defined(__GFX9__) || defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__) || defined(__gfx1013__)
            #define BITS "glc slc"
        #else
            #define BITS "glc slc dlc"
        #endif
        #define WAIT ((0 << 14) | (0x3f << 8) | (0x7) << 4)
    #else
        #define LD "global_load_u8"
        #define LD2 "global_load_u16"
        #define LD3 "global_load_b32"
        #define LD4 "global_load_b64"
        #define LD5 "global_load_b128"
        #define BITS "glc slc dlc"
        #define WAIT ((0 << 10) | (0x3f << 4) | 0x7)
    #endif
    #define LOAD LD " %0 %1 off " BITS
    #define LOAD2 LD2 " %0 %1 off " BITS
    #define LOAD3 LD3 " %0 %1 off " BITS
    #define LOAD4 LD4 " %0 %1 off " BITS
    #define LOAD5 LD5 " %0 %1 off " BITS
    
    T r{};

    switch (sizeof(T)) {
    case 1:
        asm volatile(LOAD : "={v0}"(r) : "v"(p));
        break;
    case 2:
        asm volatile(LOAD2 : "={v0}"(r) : "v"(p));
        break;
    case 4:
        asm volatile(LOAD3 : "={v0}"(r) : "v"(p));
        break;
    case 8:
        asm volatile(LOAD4 : "={v[0:1]}"(r) : "v"(p));
        break;
    case 16:
        asm volatile(LOAD5 : "=v"(r) : "v"(p));
        break;
    default: __builtin_trap();
    }
    __builtin_amdgcn_s_waitcnt(WAIT);

    return r;

    #undef LOAD5
    #undef LOAD4
    #undef LOAD3
    #undef LOAD2
    #undef LOAD
    #undef WAIT
    #undef BITS
    #undef LD5
    #undef LD4
    #undef LD3
    #undef LD2
    #undef LD
}
 
#endif

