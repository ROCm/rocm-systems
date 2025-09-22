
/**
 * @brief Opencl kernel to read from a buffer and sum its values
 * into a destination integer
 *
 * @param src Pointer to an array of 16 unsigned integers (32-bit) i.e. one instance
 * has 16 * 32-bit = 64 bytes
 * 
 * @param size Specifies number of uint16 elements in the array
 *
 * @param threads Number of threads running this kernel
 *
 * @param dst Output parameter updated with sum of the input buffer
 *
 * @note: It is critical that the size of 'src' be a integral multiple
 * of (threads * sizeof(uint16)). If it is fractional and less than ONE
 * it will lead to accessing memory that is out-of-bounds. If it is fractional
 * more but more than ONE then it will lead to some threads not doing work
 * at all leading to incorrect benchmark computation
 *
 */

__kernel void
  read_kernel(__global uint16 *src,
              ulong size, uint threads, __global uint* dst) {

  uint16 pval;
  int idx = get_global_id(0);
  __global uint16 *srcEnd = src + size;
  
  uint tmp = 0;
  src = &src[idx];
  while (src < srcEnd) {
    pval = *src;
    src += threads;
    tmp += pval.s0 + pval.s1 + pval.s2 + pval.s3 +  \
           pval.s4 + pval.s5 + pval.s6 +  pval.s7 + \
           pval.s8 + pval.s9 + pval.sa + pval.sb +  \
           pval.sc + pval.sd + pval.se + pval.sf;
  }
  atomic_add(dst, tmp);
}

