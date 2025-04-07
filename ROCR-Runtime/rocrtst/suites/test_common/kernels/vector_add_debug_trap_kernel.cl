__kernel void
vector_add_debug_trap(__global int *a,
                      __global int *b,
                      __global int *c)
{
  int gid = get_global_id(0);
  c[gid] = a[gid] + b[gid];
  __builtin_trap();
}
