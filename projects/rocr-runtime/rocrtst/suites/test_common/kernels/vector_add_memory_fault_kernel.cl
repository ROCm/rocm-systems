static __global int ga[] = { 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35 };

__kernel void
vector_add_memory_fault(
  __global const int *a,
  __global const int *b,
  __global const int *c,
  __global int *d,
  __global int *e)
{
    int gid = get_global_id(0);
    d[gid*10] = ga[gid & 31];
}
