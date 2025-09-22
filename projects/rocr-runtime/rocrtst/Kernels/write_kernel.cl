
/**
 * @brief Opencl kernel to write into a buffer the values of const integer list
 *
 * @param dst Pointer to an array of 16 unsigned integers (32-bit) i.e. one instance
 * has 16 * 32-bit = 64 bytes
 * 
 * @param size Specifies number of uint16 elements in the array
 *
 * @param threads Number of threads running this kernel
 *
 * @note: It is critical that the size of 'dst' be a integral multiple
 * of (threads * sizeof(uint16)). If it is fractional and less than ONE
 * it will lead to accessing memory that is out-of-bounds. If it is fractional
 * more but more than ONE then it will lead to some threads not doing work
 * at all leading to incorrect benchmark computation
 *
 */

__kernel void
  write_kernel(__global uint16 *dst,
               ulong size, uint threads) {

  uint16 pval = (uint16)(0xabababab, 0xabababab, 0xabababab, 0xabababab,
                         0xabababab, 0xabababab, 0xabababab, 0xabababab,
                         0xabababab, 0xabababab, 0xabababab, 0xabababab,
                         0xabababab, 0xabababab, 0xabababab, 0xabababab);

  int idx = get_global_id(0);
  __global uint16 *dstEnd = dst + size;
  
  dst = &dst[idx];
  do {
    *dst = pval;
    dst += threads;
  } while (dst < dstEnd);

}


