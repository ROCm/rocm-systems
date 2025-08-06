
#ifndef INCLUDE_RDC_RDC_PRIVATE_H_
#define INCLUDE_RDC_RDC_PRIVATE_H_

#include "rdc/rdc.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#ifdef __cplusplus

// cstddef include causes issues on older GCC
// use stddef.h instead
#if __GNUC__ < 9
#include <stddef.h>
#else
#include <cstddef>
#endif  // __GNUC__

#include <cstdint>
#else
#include <stddef.h>
#include <stdint.h>
#endif  // __cplusplus

/**
 * @brief The maximum string length occupied by version information.
 */
#define USR_MAX_VERSION_STR_LENGTH 60

/**
 * @brief Version information of mixed components
 */
typedef struct {
  char version[USR_MAX_VERSION_STR_LENGTH];
} mixed_component_version_t;

/**
 * @brief Type of Components
 */
typedef enum {
  RDCD_COMPONENT
  //If needed later, add them one by one
} mixed_component_t;

/**
 *  @brief Get ersion information of mixed components.
 *
 *  @details Given a component type, return its version information.
 *
 *  @param[in] p_rdc_handle The RDC handler.
 *
 *  @param[in] component Component type.
 *
 *  @param[out] p_mixed_compv Version information of the corresponding component.
 *
 *  @retval ::RDC_ST_OK is returned upon successful call.
 */
rdc_status_t get_mixed_component_version(rdc_handle_t p_rdc_handle, mixed_component_t component, mixed_component_version_t* p_mixed_compv);

#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // INCLUDE_RDC_RDC_PRIVATE_H_