#ifndef ROCRTST_SUITES_IMAGE_MIPMAP_CREATE_H_
#define ROCRTST_SUITES_IMAGE_MIPMAP_CREATE_H_

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_image.h"
#include "hsa_amd_mipmap.h"
#include <vector>
#include <string>

// Lightweight test harness just for mipmap array creation validation.
// Step 1 goal: verify that info query and create succeed (or fail) on target GPU (e.g. Navi4x).
// No sampling or population yet.
class MipmapCreateTest : public rocrtst::BaseRocR {
 public:
  MipmapCreateTest() {}
  ~MipmapCreateTest() {}

  // Runs a single scenario: 2D RGBA8 full chain creation and destruction.
  hsa_status_t RunBasic();

  // (Optional future) negative test: request too many levels.
  hsa_status_t RunTooManyLevels();
};

#endif // ROCRTST_SUITES_IMAGE_MIPMAP_CREATE_H_
