/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef SRC_CORE_INCLUDE_AQLPROFILE_SDK_AQL_PROFILE_STATIC_H_
#define SRC_CORE_INCLUDE_AQLPROFILE_SDK_AQL_PROFILE_STATIC_H_

/* Lifecycle entry points for the STATIC aqlprofile build (AQLPROFILE_STATIC_BUILD).
 *
 * When aqlprofile is built as a DLL/so its per-library init and teardown run automatically:
 * DllMain(DLL_PROCESS_ATTACH/DETACH) on Windows, __attribute__((constructor/destructor))
 * elsewhere.  Linked statically into another binary neither applies -- a static library has
 * no DllMain of its own (and one would collide with the host's) -- so the host runtime must
 * drive the lifecycle explicitly.  ROCr calls these from Runtime::Load()/Runtime::Unload().
 *
 * Both are idempotent-safe to call once per process; they are cheap (an env-var read, and
 * tearing down the logger / Pm4Factory singletons).
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Runs aqlprofile's process-attach initialization. */
void hsa_ven_amd_aqlprofile_static_init(void);

/* Runs aqlprofile's process-detach teardown. */
void hsa_ven_amd_aqlprofile_static_fini(void);

#ifdef __cplusplus
}
#endif

#endif /* SRC_CORE_INCLUDE_AQLPROFILE_SDK_AQL_PROFILE_STATIC_H_ */
