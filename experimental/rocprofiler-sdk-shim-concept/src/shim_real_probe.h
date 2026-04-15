#ifndef SHIM_REAL_PROBE_H
#define SHIM_REAL_PROBE_H

#ifdef __cplusplus
extern "C" {
#endif

int shim_real_probe_invoke_hip_get_device_count(int* count);
int shim_real_probe_invoke_hsa_iterate_agents(int* agent_count);

#ifdef __cplusplus
}
#endif

#endif
